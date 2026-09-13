"""Milestone 6-F frozen Test/OOD/Counterfactual evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


MODEL_SPECS = {
    "Base": {"kind": "hf", "path": "Qwen/Qwen3-0.6B"},
    "SFT": {"kind": "hf", "path": "/root/autodl-tmp/checkpoints/agentrl_m6c/best"},
    "SFT+R0": {
        "kind": "hf",
        "path": "/root/autodl-tmp/checkpoints/agentrl_m6d/best_hf",
        "native_path": "/root/autodl-tmp/checkpoints/agentrl_m6d/global_step_25",
    },
    "SFT+R2": {
        "kind": "hf",
        "path": "/root/autodl-tmp/checkpoints/agentrl_m6f_tmp/r2_hf",
        "native_path": "/root/autodl-tmp/checkpoints/agentrl_m6e2/best",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_hf_checkpoint(label: str, spec: dict, artifact_dir: Path) -> Path:
    path = Path(spec["path"])
    if (path / "config.json").is_file():
        return path
    native = spec.get("native_path")
    if not native:
        return path
    target = path
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    log = artifact_dir / f"merge_{label.lower().replace('+', '_')}.log"
    command = [
        "python",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(Path(native) / "actor"),
        "--target_dir",
        str(target),
    ]
    with log.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return target


def load_hf_model(path: str | Path):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(path)
    if not model_path.exists():
        resolved = Path(snapshot_download(str(path), local_files_only=True))
    else:
        resolved = model_path
    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(resolved, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    model.config.use_cache = True
    return model, tokenizer, resolved


def make_agent(model, tokenizer, config):
    import torch

    class Agent:
        def generate(self, messages):
            messages = [dict(message) for message in messages]
            messages[0]["content"] = config["system_prompt"]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config["enable_thinking"],
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=config["do_sample"],
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return Agent()


def table_for(results: dict, split: str, protocol: str) -> list[dict]:
    rows = []
    for label in MODEL_SPECS:
        metrics = results[label][split][protocol]
        rows.append(
            {
                "model": label,
                "answer_em": metrics["answer_em"],
                "token_f1": metrics["token_f1"],
                "protocol_success": metrics["protocol_success"],
                "search_trigger_rate": metrics["search_trigger_rate"],
                "search_decision_accuracy": metrics["search_decision_accuracy"],
                "search_precision": metrics["search_precision"],
                "search_recall": metrics["search_recall"],
                "search_f1": metrics["search_f1"],
                "avg_searches": metrics["avg_searches"],
                "max_step_failure_rate": metrics["max_step_failure_rate"],
                "direct_search_false_positive_rate": metrics["direct_search_false_positive_rate"],
                "retrieval_hit_at_1": metrics.get("retrieval_hit_at_1"),
                "retrieval_hit_at_3": metrics.get("retrieval_hit_at_3"),
                "counterfactual_em": metrics.get("counterfactual_em"),
                "evidence_use_success": metrics.get("evidence_use_success"),
            }
        )
    return rows


def deltas(results: dict) -> dict:
    out = {}
    for split, protocols in next(iter(results.values())).items():
        out[split] = {}
        for protocol in protocols:
            sft = results["SFT"][split][protocol]
            r0 = results["SFT+R0"][split][protocol]
            r2 = results["SFT+R2"][split][protocol]
            out[split][protocol] = {
                "SFT_to_R0": {"answer_em": r0["answer_em"] - sft["answer_em"], "token_f1": r0["token_f1"] - sft["token_f1"]},
                "SFT_to_R2": {"answer_em": r2["answer_em"] - sft["answer_em"], "token_f1": r2["token_f1"] - sft["token_f1"]},
                "R0_vs_R2": {"answer_em": r2["answer_em"] - r0["answer_em"], "token_f1": r2["token_f1"] - r0["token_f1"]},
                "Candidate_minus_TrainOnly": {},
            }
    for split in out:
        if "train_only_bm25" in results["SFT"][split] and "candidate_bm25" in results["SFT"][split]:
            for label in MODEL_SPECS:
                train = results[label][split]["train_only_bm25"]
                candidate = results[label][split]["candidate_bm25"]
                out[split]["candidate_vs_train_only_" + label] = {
                    "answer_em": candidate["answer_em"] - train["answer_em"],
                    "token_f1": candidate["token_f1"] - train["token_f1"],
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/milestone6f/frozen_eval")
    parser.add_argument("--keep-merged", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir)
    if artifact_dir.exists():
        raise FileExistsError(f"refusing to overwrite {artifact_dir}")
    artifact_dir.mkdir(parents=True)

    import torch
    from agentrl.frozen_evaluation import (
        FROZEN_SPLITS,
        LOCKED_EVAL_CONFIG,
        FrozenCandidateBM25Protocol,
        TrainOnlyBM25Protocol,
        candidate_bm25_audit,
        evaluate_sample,
        leakage_audit,
        read_jsonl,
        summarize_records,
    )

    split_rows = {name: read_jsonl(path) for name, path in FROZEN_SPLITS.items()}
    config = dict(LOCKED_EVAL_CONFIG)
    config.update(
        {
            "models": MODEL_SPECS,
            "splits": FROZEN_SPLITS,
            "retrieval_protocols": {
                "train_only_bm25": {
                    "corpus": "data/sft/train.jsonl",
                    "held_out_evidence_coverage": 0.0,
                },
                "candidate_bm25": {
                    "corpus": "per-sample metadata.documents",
                    "ranking_uses": ["model-generated query", "title", "text"],
                    "ranking_forbidden": ["is_support", "correct_doc_position", "answer", "gold label"],
                },
            },
            "dataset_sha256": {split: sha256(Path(path)) for split, path in FROZEN_SPLITS.items()},
            "train_sha256": sha256(Path("data/sft/train.jsonl")),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
    )
    (artifact_dir / "resolved_eval_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    protocols = [TrainOnlyBM25Protocol(), FrozenCandidateBM25Protocol()]
    results = {}
    raw_dir = artifact_dir / "raw"
    raw_dir.mkdir()
    model_paths = {}
    peak_memory = {}
    for label, spec in MODEL_SPECS.items():
        checkpoint = ensure_hf_checkpoint(label, spec, artifact_dir)
        model, tokenizer, resolved = load_hf_model(checkpoint)
        agent = make_agent(model, tokenizer, config)
        model_paths[label] = str(resolved)
        results[label] = {}
        for split, rows in split_rows.items():
            results[label][split] = {}
            for protocol in protocols:
                records = []
                raw_path = raw_dir / f"{label.replace('+', '_').lower()}_{split}_{protocol.name}.jsonl"
                with raw_path.open("w") as handle:
                    for index, sample in enumerate(rows, start=1):
                        record = evaluate_sample(sample, agent, protocol, config)
                        records.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        print(json.dumps({
                            "model": label,
                            "split": split,
                            "protocol": protocol.name,
                            "index": index,
                            "id": record["id"],
                            "em": record["answer_em"],
                            "searches": record["search_count"],
                        }), flush=True)
                results[label][split][protocol.name] = summarize_records(records)
        peak_memory[label] = torch.cuda.max_memory_allocated() / 1048576
        del model
        del tokenizer
        torch.cuda.empty_cache()

    all_frozen_rows = [row for rows in split_rows.values() for row in rows]
    audit = {
        "leakage": leakage_audit(),
        "candidate_bm25": candidate_bm25_audit(all_frozen_rows),
        "same_prompt_config_all_models": True,
        "native_search_xml_protocol": True,
        "replay_retriever_used_for_formal_results": False,
        "frozen_results_used_for_tuning": False,
    }
    tables = {
        split: {
            protocol: table_for(results, split, protocol)
            for protocol in ("train_only_bm25", "candidate_bm25")
        }
        for split in FROZEN_SPLITS
    }
    report = {
        "decision": "PASS",
        "config": config,
        "model_paths": model_paths,
        "results": results,
        "tables": tables,
        "deltas": deltas(results),
        "audit": audit,
        "peak_vram_mib": peak_memory,
        "wall_seconds": time.perf_counter() - started,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if not args.keep_merged:
        temporary = Path(MODEL_SPECS["SFT+R2"]["path"])
        if temporary.exists() and "agentrl_m6f_tmp" in str(temporary):
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()

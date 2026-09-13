"""Fresh-process locked val evaluation of native veRL checkpoint merged by veRL."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-checkpoint', required=True)
    parser.add_argument('--artifact-dir', required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.workers.config.model import HFModelConfig
    import torch
    from scripts.milestone6c_sft import evaluate_generation, load_config
    from scripts.milestone6b_sft_smoke import rows
    checkpoint = Path(args.hf_checkpoint).resolve()
    artifact = Path(args.artifact_dir).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    model_config = HFModelConfig(path=str(checkpoint))
    if Path(model_config.local_path).resolve() != checkpoint:
        raise RuntimeError('veRL model.path rejected merged checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                                 local_files_only=True).cuda().eval()
    val = rows('data/sft/val.jsonl')
    assert len(val) == 100
    metrics = evaluate_generation(model, tokenizer, val, load_config(),
                                  raw_path=artifact / 'locked_val.jsonl')
    report = {'checkpoint':str(checkpoint), 'veRL_local_path_accepted':True,
              'fresh_process_reload':True, 'train_only_bm25':True,
              'gpu_peak_memory_mib':torch.cuda.max_memory_allocated() / 1048576, **metrics}
    (artifact / 'locked_val_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()

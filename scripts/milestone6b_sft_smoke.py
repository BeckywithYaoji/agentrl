"""Tiny full-parameter HF SFT and independent-process val reload smoke."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from agentrl.hf_sft import RUNTIME_SYSTEM_PROMPT, encode_record
from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
from agentrl.agent_loop import run_agent
from agentrl.reward import normalized_em, parse_trajectory

MODEL = 'Qwen/Qwen3-0.6B'
CHECKPOINT = Path('/root/autodl-tmp/checkpoints/agentrl_m6b')
ARTIFACT = Path('artifacts/milestone6b')
MAX_LENGTH = 1536
STEPS = 8
GRAD_ACCUM = 4
LR = 5e-5


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if CHECKPOINT.exists():
        raise FileExistsError(f'refusing to overwrite {CHECKPOINT}')
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    source = Path('data/sft/train.jsonl')
    data = rows(source)
    chosen = [r for r in data if r['task_type'] != 'direct_answer'][:2] + [r for r in data if r['task_type'] == 'direct_answer'][:2]
    assert len(chosen) == 4
    model_path = snapshot_download(MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    examples = [example for row in chosen for example in encode_record(row, tokenizer, max_length=MAX_LENGTH)]
    assert len(examples) == 6
    assert all(len(x['input_ids']) == len(x['labels']) and any(y != -100 for y in x['labels']) for x in examples)
    audit = {'sample_ids': [r['id'] for r in chosen], 'examples': len(examples),
             'lengths': [len(x['input_ids']) for x in examples],
             'supervised_tokens': [sum(y != -100 for y in x['labels']) for x in examples],
             'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (ARTIFACT / 'mask_audit.json').write_text(json.dumps(audit, indent=2))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).cuda()
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    def fingerprint():
        weight = model.model.layers[0].self_attn.q_proj.weight.detach().reshape(-1)[:256].cpu()
        return hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
    before = fingerprint()
    def loss_for(example):
        ids = torch.tensor([example['input_ids']], dtype=torch.long, device='cuda')
        labels = torch.tensor([example['labels']], dtype=torch.long, device='cuda')
        return model(input_ids=ids, labels=labels).loss
    def mean_loss():
        model.eval()
        with torch.inference_mode():
            result = sum(float(loss_for(x)) for x in examples) / len(examples)
        model.train()
        return result
    started = time.perf_counter()
    initial = mean_loss()
    step_records = []
    for step in range(STEPS):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        tokens = 0
        for j in range(GRAD_ACCUM):
            example = examples[(step * GRAD_ACCUM + j) % len(examples)]
            loss = loss_for(example)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            (loss / GRAD_ACCUM).backward()
            tokens += len(example['input_ids'])
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError('non-finite gradient norm')
        optimizer.step()
        seconds = time.perf_counter() - start
        record = {'step': step + 1, 'step_seconds': seconds, 'tokens': tokens, 'tokens_per_second': tokens / seconds,
                  'grad_norm': grad_norm, 'training_loss_last_microbatch': float(loss)}
        step_records.append(record)
        print(json.dumps(record), flush=True)
    final = mean_loss()
    after = fingerprint()
    if not math.isfinite(final) or final >= initial or before == after:
        raise RuntimeError('tiny overfit loss/fingerprint gate failed')
    model.config.use_cache = True
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT, safe_serialization=True)
    tokenizer.save_pretrained(CHECKPOINT)
    report = {'model': MODEL, 'model_path': model_path, 'checkpoint': str(CHECKPOINT), 'samples': len(chosen),
              'examples': len(examples), 'max_length': MAX_LENGTH, 'batch': 1, 'grad_accum': GRAD_ACCUM,
              'optimizer': 'AdamW', 'lr': LR, 'steps': STEPS, 'initial_loss': initial, 'final_loss': final,
              'fingerprint_before': before, 'fingerprint_after': after,
              'peak_vram_mib': torch.cuda.max_memory_reserved() / 2**20,
              'wall_seconds': time.perf_counter() - started, 'step_records': step_records}
    (ARTIFACT / 'train_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'train_complete', **{k: v for k, v in report.items() if k != 'step_records'}}), flush=True)


def reload_and_val():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not (CHECKPOINT / 'config.json').is_file():
        raise FileNotFoundError('HF config.json missing')
    train_report = json.loads((ARTIFACT / 'train_report.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    base_tokenizer = AutoTokenizer.from_pretrained(train_report['model_path'], local_files_only=True)
    sample = rows('data/sft/train.jsonl')[0]
    if encode_record(sample, tokenizer, max_length=MAX_LENGTH) != encode_record(sample, base_tokenizer, max_length=MAX_LENGTH):
        raise RuntimeError('saved tokenizer/template changed SFT encoding')
    from verl.workers.config.model import HFModelConfig
    verl_model_config = HFModelConfig(path=str(CHECKPOINT))
    if Path(verl_model_config.local_path).resolve() != CHECKPOINT.resolve() or verl_model_config.hf_config.model_type != 'qwen3':
        raise RuntimeError('veRL local model.path configuration rejected HF checkpoint')
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    weight = model.model.layers[0].self_attn.q_proj.weight.detach().reshape(-1)[:256].cpu()
    digest = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
    if digest != train_report['fingerprint_after']:
        raise RuntimeError('fresh reload parameter fingerprint mismatch')
    class Agent:
        def generate(self, messages):
            messages = [dict(m) for m in messages]
            messages[0]['content'] = RUNTIME_SYSTEM_PROMPT
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            if inputs['input_ids'].shape[1] + 256 > 3072:
                raise ValueError('validation context too long')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    retriever = TinyBM25Retriever(load_training_corpus())
    val = rows('data/sft/val.jsonl')[:10]
    results = []
    with (ARTIFACT / 'val_smoke.jsonl').open('w') as handle:
        for row in val:
            result = run_agent(row['question'], Agent(), retriever, max_search_steps=3)
            parsed = parse_trajectory(result)
            record = {'id': row['id'], 'answer': result['final_answer'],
                      'em': normalized_em(result['final_answer'], row['answer']) if result['final_answer'] else 0,
                      'protocol_valid': parsed.protocol_valid, 'search_count': result['search_count'],
                      'search_triggered': result['search_count'] > 0, 'termination_reason': result['termination_reason'],
                      'turns': result['turns']}
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'id': record['id'], 'em': record['em'], 'protocol_valid': record['protocol_valid'],
                              'search_count': record['search_count']}), flush=True)
    count = len(results)
    report = {'fresh_reload': True, 'fingerprint_match': True, 'tokenizer_template_match': True,
              'verl_model_path_compatible': Path(verl_model_config.local_path).resolve() == CHECKPOINT.resolve(), 'val_size': count,
              'answer_em': sum(r['em'] for r in results)/count,
              'protocol_success': sum(r['protocol_valid'] for r in results)/count,
              'search_trigger_rate': sum(r['search_triggered'] for r in results)/count,
              'avg_searches': sum(r['search_count'] for r in results)/count}
    (ARTIFACT / 'val_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'reload_val_complete', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'reload-val'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        reload_and_val()

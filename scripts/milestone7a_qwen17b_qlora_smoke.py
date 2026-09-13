"""Qwen3-1.7B PEFT feasibility smoke for Milestone 7-A.

This is intentionally tiny: it reuses the runtime-aligned SFT encoder and trains
only LoRA adapter parameters on a few train split examples.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path

from agentrl.hf_sft import RUNTIME_SYSTEM_PROMPT, encode_record
from scripts.milestone6b_sft_smoke import rows

MODEL = 'Qwen/Qwen3-1.7B'
CHECKPOINT = Path('/root/autodl-tmp/checkpoints/agentrl_m7a')
ARTIFACT = Path('artifacts/milestone7a')
TRAIN_PATH = Path('data/sft/train.jsonl')
MAX_LENGTH = 1600
SAMPLES = 4
STEPS = 8
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
SEED = 42


def have_module(name):
    return importlib.util.find_spec(name) is not None


def select_smoke_rows(records, limit=SAMPLES):
    search_rows = [r for r in records if r.get('task_type') != 'direct_answer']
    direct_rows = [r for r in records if r.get('task_type') == 'direct_answer']
    chosen = search_rows[: max(1, limit // 2)] + direct_rows[: limit - max(1, limit // 2)]
    if len(chosen) < limit:
        seen = {r['id'] for r in chosen}
        chosen += [r for r in records if r['id'] not in seen][: limit - len(chosen)]
    if len(chosen) != limit:
        raise RuntimeError(f'expected {limit} smoke rows, found {len(chosen)}')
    return chosen


def adapter_named_parameters(model):
    return [(name, param) for name, param in model.named_parameters() if 'lora_' in name]


def adapter_fingerprint(model):
    digest = hashlib.sha256()
    count = 0
    for name, param in adapter_named_parameters(model):
        digest.update(name.encode())
        tensor = param.detach().float().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
        count += param.numel()
    if count == 0:
        raise RuntimeError('no LoRA adapter parameters found')
    return digest.hexdigest()


def validate_only_adapter_trainable(model, optimizer=None):
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    bad = [name for name, _ in trainable if 'lora_' not in name]
    if bad:
        raise RuntimeError(f'non-adapter trainable parameters: {bad[:8]}')
    if optimizer is not None:
        trainable_ids = {id(param) for _, param in trainable}
        opt_ids = {id(param) for group in optimizer.param_groups for param in group['params']}
        if opt_ids != trainable_ids:
            raise RuntimeError('optimizer parameters do not exactly match adapter trainables')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for _, p in trainable)
    return {'total_params': total_params, 'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params else 0.0,
            'trainable_names_sample': [name for name, _ in trainable[:12]]}


def has_searchxml_action(text):
    return ('<search>' in text and '</search>' in text) or ('<answer>' in text and '</answer>' in text)


def load_tokenizer_and_model(model_path, method):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if method == 'qlora':
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=config,
                                                     device_map={'': 0}, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                     local_files_only=True).cuda()
    return tokenizer, model


def resolve_method(requested):
    if requested == 'bf16-lora':
        return 'bf16-lora', 'user_requested_bf16_lora'
    if have_module('bitsandbytes'):
        return 'qlora', 'bitsandbytes_available'
    if requested == 'qlora':
        raise RuntimeError('QLoRA requested but bitsandbytes is not installed')
    return 'bf16-lora', 'bitsandbytes_missing_fallback_to_bf16_lora'


def train(args):
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if CHECKPOINT.exists():
        if not args.overwrite:
            raise FileExistsError(f'refusing to overwrite {CHECKPOINT}; pass --overwrite to replace this stage output')
        shutil.rmtree(CHECKPOINT)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    method, method_reason = resolve_method(args.peft_method)
    model_path = snapshot_download(MODEL, local_files_only=not args.allow_download)
    tokenizer, base_model = load_tokenizer_and_model(model_path, method)
    if method == 'qlora':
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                             target_modules=TARGET_MODULES, bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base_model, lora_config)
    model.train()

    records = rows(TRAIN_PATH)
    chosen = select_smoke_rows(records, args.samples)
    examples = [example for row in chosen for example in encode_record(row, tokenizer, max_length=args.max_length)]
    if not examples:
        raise RuntimeError('no encoded assistant examples')
    if not all(len(x['input_ids']) == len(x['labels']) and any(y != -100 for y in x['labels']) for x in examples):
        raise RuntimeError('runtime-aligned SFT mask audit failed')
    mask_audit = {'sample_ids': [r['id'] for r in chosen], 'task_types': [r.get('task_type') for r in chosen],
                  'examples': len(examples), 'max_length': args.max_length,
                  'lengths': [len(x['input_ids']) for x in examples],
                  'supervised_tokens': [sum(y != -100 for y in x['labels']) for x in examples],
                  'dataset_sha256': hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
                  'mask_policy': 'assistant <search>/<answer> supervised; system/user/<information>/padding masked'}
    (ARTIFACT / 'mask_audit.json').write_text(json.dumps(mask_audit, indent=2))

    trainable_report = validate_only_adapter_trainable(model)
    optimizer = torch.optim.AdamW([p for _, p in model.named_parameters() if p.requires_grad], lr=args.lr)
    trainable_report = validate_only_adapter_trainable(model, optimizer)
    before = adapter_fingerprint(model)

    def tensors(example):
        return {key: torch.tensor([example[key]], dtype=torch.long, device='cuda') for key in ['input_ids', 'labels']}

    def loss_for(example):
        batch = tensors(example)
        attention = torch.ones_like(batch['input_ids'])
        return model(input_ids=batch['input_ids'], attention_mask=attention, labels=batch['labels']).loss

    def mean_loss():
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for example in examples:
                total += float(loss_for(example))
        model.train()
        return total / len(examples)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    initial = mean_loss()
    step_records = []
    total_tokens = 0
    for step in range(args.steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        tokens = 0
        last_loss = None
        for j in range(args.grad_accum):
            example = examples[(step * args.grad_accum + j) % len(examples)]
            loss = loss_for(example)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            (loss / args.grad_accum).backward()
            tokens += len(example['input_ids'])
            last_loss = float(loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in model.named_parameters() if p.requires_grad], 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError('non-finite gradient norm')
        optimizer.step()
        seconds = time.perf_counter() - step_started
        total_tokens += tokens
        record = {'step': step + 1, 'step_seconds': seconds, 'tokens': tokens,
                  'tokens_per_second': tokens / seconds, 'grad_norm': grad_norm,
                  'training_loss_last_microbatch': last_loss}
        step_records.append(record)
        print(json.dumps(record), flush=True)
    final = mean_loss()
    after = adapter_fingerprint(model)
    if not math.isfinite(initial) or not math.isfinite(final):
        raise RuntimeError('non-finite smoke losses')
    if final >= initial:
        raise RuntimeError(f'tiny overfit loss did not decrease: {initial} -> {final}')
    if before == after:
        raise RuntimeError('LoRA adapter fingerprint did not change')

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(CHECKPOINT, safe_serialization=True)
    tokenizer.save_pretrained(CHECKPOINT)
    quantization = {'backend': 'bitsandbytes' if method == 'qlora' else 'none',
                    'quantization_type': 'nf4' if method == 'qlora' else None,
                    'compute_dtype': 'bfloat16', 'double_quant': method == 'qlora',
                    'base_model_dtype': '4bit' if method == 'qlora' else 'bfloat16',
                    'method_reason': method_reason}
    wall = time.perf_counter() - started
    report = {'model': MODEL, 'model_path': model_path, 'checkpoint': str(CHECKPOINT),
              'peft_method': method, 'quantization_config': quantization,
              'lora_target_modules': TARGET_MODULES, 'lora_rank': LORA_R, 'lora_alpha': LORA_ALPHA,
              'lora_dropout': LORA_DROPOUT, 'samples': len(chosen), 'examples': len(examples),
              'steps': args.steps, 'batch': 1, 'grad_accum': args.grad_accum, 'lr': args.lr,
              'initial_loss': initial, 'final_loss': final, 'loss_delta': final - initial,
              'adapter_fingerprint_before': before, 'adapter_fingerprint_after': after,
              **trainable_report, 'peak_vram_mib': torch.cuda.max_memory_reserved() / 2**20,
              'wall_seconds': wall, 'tokens_per_second': total_tokens / wall, 'step_records': step_records,
              'frozen_eval_touched': False,
              'comparison_reference': {'qwen3_0_6b_full_sft_peak_vram_mib': 22248}}
    (ARTIFACT / 'train_report.json').write_text(json.dumps(report, indent=2))
    (CHECKPOINT / 'smoke_metadata.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'train_complete', **{k: v for k, v in report.items() if k != 'step_records'}}), flush=True)


def reload_smoke(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    metadata = json.loads((CHECKPOINT / 'smoke_metadata.json').read_text())
    method = metadata['peft_method']
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if method == 'qlora':
        from transformers import BitsAndBytesConfig
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(metadata['model_path'], quantization_config=qconfig,
                                                    device_map={'': 0}, local_files_only=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(metadata['model_path'], dtype=torch.bfloat16,
                                                    local_files_only=True).cuda()
    model = PeftModel.from_pretrained(base, CHECKPOINT, local_files_only=True).eval()
    digest = adapter_fingerprint(model)
    if digest != metadata['adapter_fingerprint_after']:
        raise RuntimeError('fresh reload adapter fingerprint mismatch')
    sample = select_smoke_rows(rows(TRAIN_PATH), 1)[0]
    messages = [{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT}, {'role': 'user', 'content': sample['question']}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    report = {'fresh_reload': True, 'adapter_fingerprint_match': True, 'sample_id': sample['id'],
              'generated_text': text, 'searchxml_action_present': has_searchxml_action(text),
              'prompt_tokens': int(inputs['input_ids'].shape[1]), 'max_new_tokens': args.max_new_tokens}
    if not report['searchxml_action_present']:
        raise RuntimeError(f'generation did not produce SearchXML action: {text!r}')
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / 'reload_smoke.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'event': 'reload_smoke_complete', **report}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'reload-smoke'])
    parser.add_argument('--peft-method', choices=['auto', 'qlora', 'bf16-lora'], default='auto')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--samples', type=int, default=SAMPLES)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--grad-accum', type=int, default=GRAD_ACCUM)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        reload_smoke(args)


if __name__ == '__main__':
    main()

"""Formal Qwen3-1.7B BF16 LoRA SFT with val-loss adapter selection."""
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

from agentrl.agent_loop import run_agent
from agentrl.hf_sft import RUNTIME_SYSTEM_PROMPT, encode_record
from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
from agentrl.reward import normalized_em, parse_trajectory, token_f1
from scripts.milestone6b_sft_smoke import rows
from scripts.milestone7a_qwen17b_qlora_smoke import adapter_fingerprint, validate_only_adapter_trainable

CONFIG_PATH = Path('configs/m7b_lora_sft.json')
ARTIFACT = Path('artifacts/milestone7b')
def append_json(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    if config['model'] != 'Qwen/Qwen3-1.7B':
        raise RuntimeError('M7-B must use Qwen/Qwen3-1.7B')
    if config['peft_method'] != 'bf16-lora' or config['precision'] != 'bf16':
        raise RuntimeError('M7-B formal config must use BF16 LoRA')
    if config['lora_target_modules'] != ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        raise RuntimeError('LoRA target modules changed from M7-A')
    if config['train_path'] != 'data/sft/train.jsonl' or config['val_path'] != 'data/sft/val.jsonl':
        raise RuntimeError('unexpected train/val path')
    return config


def choose_model_path(allow_download=False):
    from huggingface_hub import snapshot_download
    return snapshot_download('Qwen/Qwen3-1.7B', local_files_only=not allow_download)


def validation_steps(steps_per_epoch, epochs, interval_epochs):
    every = max(1, int(round(steps_per_epoch * interval_epochs)))
    total = steps_per_epoch * epochs
    return sorted(set(range(every, total + 1, every)) | {total})


def select_best(current, best):
    return math.isfinite(current) and (best is None or current < best)


def summarize_generation(records):
    n = len(records)
    if not n:
        return {'count': 0}
    tp = sum(r['first_action'] == 'search' and r['requires_search'] for r in records)
    fp = sum(r['first_action'] == 'search' and not r['requires_search'] for r in records)
    fn = sum(r['first_action'] != 'search' and r['requires_search'] for r in records)
    direct = sum(not r['requires_search'] for r in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'count': n, 'answer_em': sum(r['answer_em'] for r in records) / n,
            'token_f1': sum(r['token_f1'] for r in records) / n,
            'protocol_success': sum(r['protocol_valid'] for r in records) / n,
            'search_trigger_rate': sum(r['search_triggered'] for r in records) / n,
            'search_decision_accuracy': sum(r['first_action'] == ('search' if r['requires_search'] else 'answer') for r in records) / n,
            'search_precision': precision, 'search_recall': recall,
            'search_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'avg_searches': sum(r['search_count'] for r in records) / n,
            'max_step_failure_rate': sum(r['termination_reason'] == 'max_search_steps' for r in records) / n,
            'direct_search_false_positive_rate': fp / direct if direct else None}


def hf_agent(model, tokenizer, config):
    import torch

    class Agent:
        def generate(self, messages):
            messages = [dict(message) for message in messages]
            messages[0]['content'] = RUNTIME_SYSTEM_PROMPT
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=config['generation_max_new_tokens'],
                                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return Agent()


def evaluate_generation(model, tokenizer, val_rows, config, raw_path):
    was_training = model.training
    previous_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    retriever = TinyBM25Retriever(load_training_corpus(config['retrieval_corpus']))
    agent = hf_agent(model, tokenizer, config)
    records = []
    with raw_path.open('w', encoding='utf-8') as handle:
        for row in val_rows:
            result = run_agent(row['question'], agent, retriever, max_search_steps=config['max_search_steps'])
            parsed = parse_trajectory(result)
            answer = result['final_answer'] or ''
            requires_search = bool(row['metadata']['requires_search'])
            record = {'id': row['id'], 'answer': answer, 'answer_em': normalized_em(answer, row['answer']) if answer else 0.0,
                      'token_f1': token_f1(answer, row['answer']), 'protocol_valid': parsed.protocol_valid,
                      'first_action': parsed.first_action, 'requires_search': requires_search,
                      'search_triggered': result['search_count'] > 0, 'search_count': result['search_count'],
                      'termination_reason': result['termination_reason'], 'turns': result['turns']}
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
    model.config.use_cache = previous_cache
    if was_training:
        model.train()
    return summarize_generation(records)


def build_lora_model(model_path, config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).cuda()
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    lora = LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                      target_modules=config['lora_target_modules'], bias='none', task_type=TaskType.CAUSAL_LM)
    return tokenizer, get_peft_model(base, lora)


def save_adapter(model, tokenizer, path, metadata):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / 'selection_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    model.config.use_cache = previous_cache


def train(args):
    import torch

    config = load_config()
    checkpoint_root = Path(config['checkpoint_root'])
    if (checkpoint_root.exists() or ARTIFACT.exists()) and not args.overwrite:
        raise FileExistsError('M7-B output path exists; pass --overwrite for this stage path')
    if checkpoint_root.exists() and args.overwrite:
        shutil.rmtree(checkpoint_root)
    if ARTIFACT.exists() and args.overwrite:
        shutil.rmtree(ARTIFACT)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    (ARTIFACT / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train_path = Path(config['train_path'])
    val_path = Path(config['val_path'])
    train_rows = rows(train_path)
    val_rows = rows(val_path)
    if len(train_rows) != 800 or len(val_rows) != 100:
        raise RuntimeError(f'expected 800 train and 100 val rows, got {len(train_rows)} / {len(val_rows)}')
    model_path = choose_model_path(allow_download=args.allow_download)
    tokenizer, model = build_lora_model(model_path, config)
    train_groups = [encode_record(row, tokenizer, max_length=config['max_seq_length']) for row in train_rows]
    val_examples = [example for row in val_rows for example in encode_record(row, tokenizer, max_length=config['max_seq_length'])]
    train_examples_count = sum(len(group) for group in train_groups)
    if train_examples_count != 1480 or len(val_examples) != 185:
        raise RuntimeError(f'unexpected target counts: {train_examples_count} train / {len(val_examples)} val')
    if not all(any(y != -100 for y in example['labels']) for group in train_groups for example in group):
        raise RuntimeError('train mask audit failed')

    trainable_report = validate_only_adapter_trainable(model)
    optimizer = torch.optim.AdamW([param for _, param in model.named_parameters() if param.requires_grad],
                                  lr=config['learning_rate'], weight_decay=config['weight_decay'])
    trainable_report = validate_only_adapter_trainable(model, optimizer)
    frozen_params = trainable_report['total_params'] - trainable_report['trainable_params']
    before_fingerprint = adapter_fingerprint(model)
    grad_accum = config['gradient_accumulation_steps']
    steps_per_epoch = math.ceil(train_examples_count / grad_accum)
    total_steps = steps_per_epoch * config['epochs']
    warmup = max(1, math.ceil(total_steps * config['warmup_ratio']))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    checks = validation_steps(steps_per_epoch, config['epochs'], config['validation_interval_epochs'])
    log = ARTIFACT / 'events.jsonl'
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {'git_commit': git_commit, 'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
                'data_sha256': {'train': hashlib.sha256(train_path.read_bytes()).hexdigest(),
                                'val': hashlib.sha256(val_path.read_bytes()).hexdigest()},
                'model_path': model_path,
                'train_rows': len(train_rows), 'val_rows': len(val_rows),
                'train_targets': train_examples_count, 'val_targets': len(val_examples),
                'trainable_audit': {**trainable_report, 'frozen_params': frozen_params,
                                    'optimizer_param_groups': len(optimizer.param_groups)}}
    (ARTIFACT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def forward_loss(example):
        ids = torch.tensor([example['input_ids']], dtype=torch.long, device='cuda')
        labels = torch.tensor([example['labels']], dtype=torch.long, device='cuda')
        attention = torch.ones_like(ids)
        return model(input_ids=ids, attention_mask=attention, labels=labels).loss

    def val_teacher_loss():
        was_training = model.training
        model.eval()
        total = 0.0
        tokens = 0
        with torch.inference_mode():
            for example in val_examples:
                supervised = sum(label != -100 for label in example['labels'])
                loss = float(forward_loss(example))
                total += loss * supervised
                tokens += supervised
        if was_training:
            model.train()
        return total / tokens

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    model.train()
    best = None
    best_step = None
    global_step = 0
    step_records = []
    val_records = []
    seen_ids_per_epoch = []
    for epoch in range(config['epochs']):
        order = list(range(len(train_rows)))
        random.Random(config['seed'] + epoch).shuffle(order)
        seen_ids_per_epoch.append(len({train_rows[index]['id'] for index in order}))
        sequence = [example for index in order for example in train_groups[index]]
        for start_index in range(0, len(sequence), grad_accum):
            chunk = sequence[start_index:start_index + grad_accum]
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            tokens = 0
            for example in chunk:
                loss = forward_loss(example)
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite training loss')
                (loss / len(chunk)).backward()
                loss_sum += float(loss.detach())
                tokens += len(example['input_ids'])
            grad_norm = float(torch.nn.utils.clip_grad_norm_([param for _, param in model.named_parameters() if param.requires_grad],
                                                              config['gradient_clip_norm']))
            if not math.isfinite(grad_norm):
                raise RuntimeError('non-finite grad norm')
            optimizer.step()
            scheduler.step()
            global_step += 1
            seconds = time.perf_counter() - step_started
            event = {'event': 'train_step', 'step': global_step, 'epoch': epoch + 1,
                     'loss': loss_sum / len(chunk), 'lr': optimizer.param_groups[0]['lr'],
                     'grad_norm': grad_norm, 'tokens': tokens, 'step_seconds': seconds,
                     'tokens_per_second': tokens / seconds}
            append_json(log, event)
            step_records.append(event)
            if global_step % 20 == 0:
                print(json.dumps(event), flush=True)
            if global_step in checks:
                val_loss = val_teacher_loss()
                diagnostics = evaluate_generation(model, tokenizer, val_rows[:config['validation_diagnostic_size']], config,
                                                  ARTIFACT / f'val_diagnostic_step_{global_step}.jsonl')
                improved = select_best(val_loss, best)
                current_fingerprint = adapter_fingerprint(model)
                if improved:
                    save_adapter(model, tokenizer, Path(config['best_adapter_dir']),
                                 {'step': global_step, 'epoch': epoch + 1, 'val_teacher_loss': val_loss,
                                  'adapter_fingerprint': current_fingerprint,
                                  'selection_rule': config['selection_rule'], 'role': 'best'})
                    best = val_loss
                    best_step = global_step
                val_event = {'event': 'validation', 'step': global_step, 'epoch': epoch + 1,
                             'val_teacher_loss': val_loss, 'selected_best': improved, 'best_step': best_step,
                             'adapter_fingerprint': current_fingerprint, 'diagnostics': diagnostics}
                append_json(log, val_event)
                val_records.append(val_event)
                print(json.dumps(val_event), flush=True)
    after_fingerprint = adapter_fingerprint(model)
    save_adapter(model, tokenizer, Path(config['latest_adapter_dir']),
                 {'step': global_step, 'epoch': config['epochs'], 'adapter_fingerprint': after_fingerprint,
                  'role': 'latest'})
    wall = time.perf_counter() - started
    step_time_sum = sum(record['step_seconds'] for record in step_records)
    report = {'global_steps': global_step, 'steps_per_epoch': steps_per_epoch, 'epochs': config['epochs'],
              'all_800_train_samples_used_each_epoch': all(count == 800 for count in seen_ids_per_epoch),
              'train_targets': train_examples_count, 'val_targets': len(val_examples),
              'best_step': best_step, 'best_val_teacher_loss': best,
              'adapter_fingerprint_before': before_fingerprint, 'adapter_fingerprint_after': after_fingerprint,
              'trainable_params': trainable_report['trainable_params'],
              'total_params': trainable_report['total_params'], 'trainable_ratio': trainable_report['trainable_ratio'],
              'frozen_params': frozen_params, 'optimizer_param_groups': len(optimizer.param_groups),
              'peak_vram_mib': torch.cuda.max_memory_reserved() / 2**20,
              'wall_seconds': wall, 'mean_step_seconds': step_time_sum / len(step_records),
              'tokens_per_second': sum(record['tokens'] for record in step_records) / step_time_sum,
              'train_loss_curve': [{'step': r['step'], 'loss': r['loss'], 'lr': r['lr'], 'grad_norm': r['grad_norm']} for r in step_records],
              'val_loss_curve': [{'step': r['step'], 'val_teacher_loss': r['val_teacher_loss'],
                                  'selected_best': r['selected_best']} for r in val_records],
              'best_adapter_dir': config['best_adapter_dir'], 'latest_adapter_dir': config['latest_adapter_dir'],
              'frozen_eval_touched': False,
              'optional_merged_export': {'attempted': False, 'reason': 'skipped_by_default_to_conserve_disk; not required for M7-B PASS'}}
    (ARTIFACT / 'train_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'train_complete', **{k: v for k, v in report.items() if k not in {'train_loss_curve'}}}), flush=True)


def reload_val(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config()
    train_report = json.loads((ARTIFACT / 'train_report.json').read_text())
    best_dir = Path(config['best_adapter_dir'])
    metadata = json.loads((best_dir / 'selection_metadata.json').read_text())
    model_path = choose_model_path(allow_download=args.allow_download)
    tokenizer = AutoTokenizer.from_pretrained(best_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).cuda()
    model = PeftModel.from_pretrained(base, best_dir, local_files_only=True).eval()
    digest = adapter_fingerprint(model)
    if digest != metadata['adapter_fingerprint']:
        raise RuntimeError('fresh reload best adapter fingerprint mismatch')
    sample = rows(config['train_path'])[0]
    prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT},
                                            {'role': 'user', 'content': sample['question']}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    smoke = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    if not (('<search>' in smoke and '</search>' in smoke) or ('<answer>' in smoke and '</answer>' in smoke)):
        raise RuntimeError(f'generation smoke did not produce SearchXML action: {smoke!r}')
    val_rows = rows(config['val_path'])
    if len(val_rows) != config['locked_val_size']:
        raise RuntimeError('locked val size mismatch')
    locked = evaluate_generation(model, tokenizer, val_rows, config, ARTIFACT / 'locked_val.jsonl')
    locked.update({'fresh_reload': True, 'base_model': config['model'], 'adapter_dir': str(best_dir),
                   'adapter_fingerprint_match': True, 'best_step': train_report['best_step'],
                   'best_val_teacher_loss': train_report['best_val_teacher_loss'],
                   'generation_smoke': smoke, 'optional_merged_export': train_report['optional_merged_export'],
                   'baseline_0_6b_sft_val': config['baseline_0_6b_sft_val'],
                   'comparison_to_0_6b_sft': {
                       'answer_em_delta': locked['answer_em'] - config['baseline_0_6b_sft_val']['answer_em'],
                       'token_f1_delta': locked['token_f1'] - config['baseline_0_6b_sft_val']['token_f1'],
                       'protocol_success_delta': locked['protocol_success'] - config['baseline_0_6b_sft_val']['protocol_success'],
                       'search_decision_accuracy_delta': locked['search_decision_accuracy'] - config['baseline_0_6b_sft_val']['search_decision_accuracy'],
                       'avg_searches_delta': locked['avg_searches'] - config['baseline_0_6b_sft_val']['avg_searches'],
                   },
                   'frozen_eval_touched': False})
    (ARTIFACT / 'locked_val_report.json').write_text(json.dumps(locked, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'event': 'locked_val_complete', **locked}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'reload-val'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        reload_val(args)


if __name__ == '__main__':
    main()

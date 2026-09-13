"""Formal full-parameter HF SFT, val-loss selection, and fresh val evaluation."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

from agentrl.agent_loop import run_agent
from agentrl.hf_sft import RUNTIME_SYSTEM_PROMPT, encode_record
from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
from agentrl.reward import normalized_em, parse_trajectory, token_f1
from scripts.milestone6b_sft_smoke import rows

CONFIG_PATH = Path('configs/m6c_sft.json')
ARTIFACT = Path('artifacts/milestone6c')


def select_best(current, best):
    return math.isfinite(current) and (best is None or current < best)


def validation_steps(total_steps, every):
    return sorted(set(range(every, total_steps + 1, every)) | {total_steps})


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    assert config['model'] == 'Qwen/Qwen3-0.6B'
    assert config['precision'] == 'bf16' and config['optimizer'] == 'AdamW'
    assert config['retrieval_corpus'] == 'data/sft/train.jsonl'
    return config


def append_json(path, record):
    with path.open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def choose_model_path():
    from huggingface_hub import snapshot_download
    return snapshot_download('Qwen/Qwen3-0.6B', local_files_only=True)


def hf_agent(model, tokenizer, config):
    import torch
    class Agent:
        def generate(self, messages):
            messages = [dict(m) for m in messages]
            messages[0]['content'] = RUNTIME_SYSTEM_PROMPT
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            if inputs['input_ids'].shape[1] + config['generation_max_new_tokens'] > 3072:
                raise ValueError('val generation context exceeds 3072 tokens')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=config['generation_max_new_tokens'],
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return Agent()


def evaluate_generation(model, tokenizer, val_rows, config, *, raw_path):
    was_training = model.training
    previous_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    agent = hf_agent(model, tokenizer, config)
    retriever = TinyBM25Retriever(load_training_corpus())
    records = []
    with raw_path.open('w') as handle:
        for row in val_rows:
            result = run_agent(row['question'], agent, retriever, max_search_steps=config['max_search_steps'])
            parsed = parse_trajectory(result)
            first_action = parsed.first_action
            required = bool(row['metadata']['requires_search'])
            answer = result['final_answer'] or ''
            record = {'id': row['id'], 'answer': answer,
                      'answer_em': normalized_em(answer, row['answer']) if answer else 0.0,
                      'token_f1': token_f1(answer, row['answer']),
                      'protocol_valid': parsed.protocol_valid,
                      'first_action': first_action, 'requires_search': required,
                      'search_triggered': result['search_count'] > 0,
                      'search_count': result['search_count'],
                      'termination_reason': result['termination_reason'],
                      'turns': result['turns']}
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
    model.config.use_cache = previous_cache
    if was_training:
        model.train()
    n = len(records)
    tp = sum(r['first_action'] == 'search' and r['requires_search'] for r in records)
    fp = sum(r['first_action'] == 'search' and not r['requires_search'] for r in records)
    fn = sum(r['first_action'] != 'search' and r['requires_search'] for r in records)
    direct = sum(not r['requires_search'] for r in records)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {'count': n, 'answer_em': sum(r['answer_em'] for r in records)/n,
            'token_f1': sum(r['token_f1'] for r in records)/n,
            'protocol_success': sum(r['protocol_valid'] for r in records)/n,
            'search_trigger_rate': sum(r['search_triggered'] for r in records)/n,
            'search_decision_accuracy': sum(r['first_action'] == ('search' if r['requires_search'] else 'answer') for r in records)/n,
            'search_precision': precision, 'search_recall': recall,
            'search_f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'avg_searches': sum(r['search_count'] for r in records)/n,
            'max_step_failure_rate': sum(r['termination_reason'] == 'max_search_steps' for r in records)/n,
            'direct_search_false_positive_rate': fp/direct if direct else None}


def train():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = load_config()
    checkpoint = Path(config['checkpoint_dir'])
    if checkpoint.exists() or ARTIFACT.exists():
        raise FileExistsError('formal run path exists; refusing overwrite')
    ARTIFACT.mkdir(parents=True)
    start = time.perf_counter()
    (ARTIFACT / 'resolved_config.json').write_text(json.dumps(config, indent=2))
    source = Path('data/sft/train.jsonl')
    val_source = Path('data/sft/val.jsonl')
    data = rows(source)
    val_rows = rows(val_source)
    assert len(data) == 800 and len(val_rows) == 100
    model_path = choose_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    per_record = [encode_record(r, tokenizer, max_length=config['max_seq_length']) for r in data]
    train_examples = [x for group in per_record for x in group]
    val_examples = [x for r in val_rows for x in encode_record(r, tokenizer, max_length=config['max_seq_length'])]
    assert len(train_examples) == 1480 and len(val_examples) == 185
    assert all(any(y != -100 for y in x['labels']) and len(x['labels']) == len(x['input_ids']) for x in train_examples + val_examples)
    data_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('train', source), ('val', val_source)]}
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {'config': config, 'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
                'git_commit': git_commit, 'data_sha256': data_hashes,
                'train_rows': len(data), 'val_rows': len(val_rows), 'train_targets': len(train_examples),
                'val_targets': len(val_examples), 'model_path': model_path}
    (ARTIFACT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).cuda()
    assert all(p.requires_grad for p in model.parameters())
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    grad_accum = config['gradient_accumulation_steps']
    steps_per_epoch = math.ceil(len(train_examples)/grad_accum)
    total_steps = steps_per_epoch * config['epochs']
    warmup = max(1, math.ceil(total_steps * config['warmup_ratio']))
    def lr_factor(step):
        if step < warmup:
            return (step+1)/warmup
        return max(0.0, (total_steps-step)/(total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    checks = validation_steps(total_steps, config['validation_interval_steps'])
    def forward_loss(example):
        ids = torch.tensor([example['input_ids']], device='cuda', dtype=torch.long)
        labels = torch.tensor([example['labels']], device='cuda', dtype=torch.long)
        return model(input_ids=ids, labels=labels).loss
    def val_teacher_loss():
        model.eval()
        total = 0.0
        tokens = 0
        with torch.inference_mode():
            for example in val_examples:
                supervised = sum(y != -100 for y in example['labels'])
                loss = float(forward_loss(example))
                total += loss*supervised
                tokens += supervised
        model.train()
        return total/tokens
    log = ARTIFACT / 'events.jsonl'
    best = None
    best_step = None
    global_step = 0
    epoch_times = []
    seen_rows = []
    step_stats = []
    for epoch in range(config['epochs']):
        epoch_start = time.perf_counter()
        order = list(range(len(data)))
        random.Random(config['seed'] + epoch).shuffle(order)
        seen_rows.append(len({data[i]['id'] for i in order}))
        sequence = [example for i in order for example in per_record[i]]
        for start_index in range(0, len(sequence), grad_accum):
            chunk = sequence[start_index:start_index+grad_accum]
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            processed = 0
            for example in chunk:
                loss = forward_loss(example)
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite training loss')
                (loss/len(chunk)).backward()
                loss_sum += float(loss.detach())
                processed += len(example['input_ids'])
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm']))
            if not math.isfinite(grad_norm):
                raise RuntimeError('non-finite grad norm')
            optimizer.step()
            scheduler.step()
            global_step += 1
            seconds = time.perf_counter() - step_start
            event = {'event': 'train_step', 'step': global_step, 'epoch': epoch+1,
                     'loss': loss_sum/len(chunk), 'lr': optimizer.param_groups[0]['lr'],
                     'grad_norm': grad_norm, 'tokens': processed, 'step_seconds': seconds}
            append_json(log, event)
            step_stats.append(event)
            if global_step % 20 == 0:
                print(json.dumps(event), flush=True)
            if global_step in checks:
                val_loss = val_teacher_loss()
                diagnostics = evaluate_generation(model, tokenizer,
                    val_rows[:config['validation_diagnostic_size']], config,
                    raw_path=ARTIFACT / f'val_diagnostic_step_{global_step}.jsonl')
                improved = select_best(val_loss, best)
                if improved:
                    model.config.use_cache = True
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint, safe_serialization=True)
                    tokenizer.save_pretrained(checkpoint)
                    model.config.use_cache = False
                    best = val_loss
                    best_step = global_step
                event = {'event': 'validation', 'step': global_step, 'epoch': epoch+1,
                         'val_teacher_loss': val_loss, 'selected_best': improved,
                         'best_step': best_step, 'diagnostics': diagnostics}
                append_json(log, event)
                print(json.dumps(event), flush=True)
        epoch_times.append(time.perf_counter()-epoch_start)
    report = {'global_steps': global_step, 'steps_per_epoch': steps_per_epoch,
              'epochs': config['epochs'], 'all_train_ids_per_epoch': seen_rows,
              'best_step': best_step, 'best_val_teacher_loss': best,
              'peak_vram_mib': torch.cuda.max_memory_reserved()/2**20,
              'epoch_seconds': epoch_times, 'wall_seconds': time.perf_counter()-start,
              'mean_step_seconds': sum(x['step_seconds'] for x in step_stats)/len(step_stats),
              'tokens_per_second': sum(x['tokens'] for x in step_stats)/sum(x['step_seconds'] for x in step_stats),
              'checkpoint': str(checkpoint)}
    (ARTIFACT / 'train_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'train_complete', **report}), flush=True)


def reload_val():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.workers.config.model import HFModelConfig
    config = load_config()
    checkpoint = Path(config['checkpoint_dir'])
    report = json.loads((ARTIFACT / 'train_report.json').read_text())
    model_config = HFModelConfig(path=str(checkpoint))
    if Path(model_config.local_path).resolve() != checkpoint.resolve():
        raise RuntimeError('veRL rejected local checkpoint path')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    base_tokenizer = AutoTokenizer.from_pretrained(choose_model_path(), local_files_only=True)
    sample = rows('data/sft/train.jsonl')[0]
    if encode_record(sample, tokenizer, max_length=config['max_seq_length']) != encode_record(sample, base_tokenizer, max_length=config['max_seq_length']):
        raise RuntimeError('tokenizer/template mismatch after reload')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    val_rows = rows('data/sft/val.jsonl')
    assert len(val_rows) == config['locked_val_size']
    locked = evaluate_generation(model, tokenizer, val_rows, config,
        raw_path=ARTIFACT / 'locked_val.jsonl')
    train_docs = {(d['title'], d['text']) for d in load_training_corpus()}
    support_docs = [d for row in val_rows for d in row['metadata']['documents'] if d['is_support']]
    coverage = sum((d['title'], d['text']) in train_docs for d in support_docs)
    locked.update({'fresh_reload': True, 'checkpoint': str(checkpoint),
                   'best_step': report['best_step'], 'veRL_local_path_accepted': True,
                   'train_only_bm25': True, 'heldout_support_doc_coverage': f'{coverage}/{len(support_docs)}'})
    (ARTIFACT / 'locked_val_report.json').write_text(json.dumps(locked, indent=2))
    print(json.dumps({'event': 'locked_val_complete', **locked}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'reload-val'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        reload_val()

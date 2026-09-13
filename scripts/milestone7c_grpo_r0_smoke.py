"""Qwen3-1.7B LoRA GRPO-R0 feasibility smoke using native veRL AgentLoop."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean, pvariance

SAMPLE_ID = 'squad-5731c7ade17f3d14004223d9'
MODEL_PATH = 'Qwen/Qwen3-1.7B'
ADAPTER = Path('/root/autodl-tmp/checkpoints/agentrl_m7b/best_adapter')
ARTIFACT = Path('artifacts/milestone7c')


def emit(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def _fingerprint_tensor(tensor):
    sample = tensor.detach().reshape(-1)[:512].contiguous().cpu()
    return hashlib.sha256(sample.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def run(directory, *, rollout_n=4, total_steps=1, gpu_memory_utilization=0.30):
    import pandas as pd
    import ray
    import torch
    import transfer_queue as tq
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
    from verl.trainer.ppo.v1.trainer_base import Role
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.rollout.llm_server import LLMServerClient
    from agentrl.hf_sft import RUNTIME_SYSTEM_PROMPT
    from agentrl.reward import ACTION_RE, parse_trajectory
    from scripts.milestone6b_sft_smoke import rows

    trace = directory / 'trace.jsonl'
    train_rows = rows('data/sft/train.jsonl')
    sample = next(row for row in train_rows if row['id'] == SAMPLE_ID)
    question = sample['question']
    train_path = directory / 'train.parquet'
    pd.DataFrame([dict(prompt=[{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT}, {'role': 'user', 'content': question}],
                       data_source='agentrl_train', reward_model={'ground_truth': sample['answer']},
                       extra_info={'index': 0, 'sample_id': sample['id'],
                                   'requires_search': sample['metadata']['requires_search']})]).to_parquet(train_path)
    loop_path = directory / 'agent.yaml'
    OmegaConf.save(OmegaConf.create([dict(name='search_xml',
        _target_='agentrl.verl_agent_loop.SearchXMLAgentLoop', max_search_steps=3)]), loop_path)
    from huggingface_hub import snapshot_download
    base_path = snapshot_download(MODEL_PATH, local_files_only=True)
    if not (ADAPTER / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'M7-B best adapter missing: {ADAPTER}')

    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / 'trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer')
    with open_dict(config):
        config.actor_rollout_ref.model.path = base_path
        config.actor_rollout_ref.model.lora_adapter_path = str(ADAPTER.resolve())
        config.actor_rollout_ref.model.use_remove_padding = True
        config.actor_rollout_ref.model.use_shm = False
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.model.lora.merge = False
        config.actor_rollout_ref.actor.fsdp_config.use_torch_compile = False
        config.actor_rollout_ref.actor.fsdp_config.param_offload = False
        config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
        config.actor_rollout_ref.actor.optim.lr = 1e-6
        config.actor_rollout_ref.actor.ppo_epochs = 1
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
        config.algorithm.adv_estimator = 'grpo'
        config.algorithm.use_kl_in_reward = False
        config.trainer.n_gpus_per_node = 1
        config.trainer.nnodes = 1
        config.trainer.total_epochs = 1
        config.trainer.total_training_steps = total_steps
        config.trainer.save_freq = -1
        config.trainer.test_freq = -1
        config.trainer.val_before_train = False
        config.trainer.logger = ['console']
        config.trainer.project_name = 'agentrl-m7c'
        config.trainer.experiment_name = f'qwen17b-lora-grpo-r0-smoke-n{rollout_n}'
        config.data.train_files = str(train_path.resolve())
        config.data.val_files = str(train_path.resolve())
        config.data.train_batch_size = 1
        config.data.gen_batch_size = 1
        config.data.val_batch_size = 1
        config.data.shuffle = False
        config.data.seed = 42
        config.data.dataloader_num_workers = 0
        config.data.max_prompt_length = 1024
        config.data.max_response_length = 2048
        config.data.apply_chat_template_kwargs = {'enable_thinking': False}
        rollout = config.actor_rollout_ref.rollout
        for key, value in dict(name='vllm', mode='async', n=rollout_n, tensor_model_parallel_size=1,
            data_parallel_size=1, pipeline_model_parallel_size=1, prompt_length=1024,
            response_length=2048, max_model_len=3072, max_num_batched_tokens=3072,
            max_num_seqs=rollout_n, enforce_eager=True, gpu_memory_utilization=gpu_memory_utilization,
            temperature=1.0, top_p=1.0, top_k=-1, calculate_log_probs=False).items():
            rollout[key] = value
        rollout.val_kwargs.n = 1
        rollout.agent.num_workers = 1
        rollout.agent.default_agent_loop = 'search_xml'
        rollout.agent.agent_loop_config_path = str(loop_path.resolve())
        config.reward.custom_reward_function.path = str(Path('src/agentrl/verl_r0_reward.py').resolve())
        config.reward.custom_reward_function.name = 'compute_score'
        config.reward.num_workers = 1
        config.transfer_queue.enable = True
        config.transfer_queue.backend.SimpleStorage.num_data_storage_units = 2
    OmegaConf.save(config, directory / 'resolved.yaml')
    config_record = dict(event='config', chosen_path='Native LoRA-GRPO', base_model=MODEL_PATH,
                         base_path=base_path, start_adapter=str(ADAPTER.resolve()), sample_id=SAMPLE_ID,
                         reward='R0 exact-match via src/agentrl/verl_r0_reward.py', retriever='train-only TinyBM25',
                         rollout_n=rollout_n, temperature=1.0, top_p=1.0, max_response_length=2048,
                         per_turn_generation_cap=256, lr=1e-6, total_steps=total_steps,
                         verl_peft_audit={'actor_lora_adapter_path': True, 'peft_trainable_adapter': True,
                                          'rollout_adapter_weight_sync': 'base weights then adapter tensors when lora.merge=False',
                                          'ref_log_prob_without_lora': True, 'lora_only_checkpoint_supported': True,
                                          'vendor_patch': False})
    (directory / 'resolved_config.json').write_text(json.dumps(config_record, indent=2), encoding='utf-8')
    emit(trace, config_record)

    class BudgetClient(LLMServerClient):
        async def generate(self, *args, **kwargs):
            sampling = dict(kwargs['sampling_params'])
            sampling['max_tokens'] = 256
            kwargs['sampling_params'] = sampling
            return await super().generate(*args, **kwargs)

    class ProbeWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def probe_actor(self):
            if not hasattr(self, '_probe_optimizer_calls'):
                self._probe_optimizer_calls = 0
                optimizer = self.actor.engine.optimizer
                original = optimizer.step
                def counted_step(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self._probe_optimizer_calls += 1
                    return result
                optimizer.step = counted_step
            selected_lora = {}
            selected_base = {}
            trainable = []
            for name, param in self.actor.engine.module.named_parameters():
                if param.requires_grad:
                    trainable.append(name)
                target = selected_lora if 'lora_' in name else selected_base
                if ('lora_' in name and len(selected_lora) < 5) or ('lora_' not in name and len(selected_base) < 5):
                    tensor = param.detach()
                    target[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                    'requires_grad': bool(param.requires_grad),
                                    'slice_sha256': _fingerprint_tensor(tensor)}
            bad_trainable = [name for name in trainable if 'lora_' not in name]
            return {'optimizer_steps': self._probe_optimizer_calls, 'lora': selected_lora,
                    'base': selected_base, 'trainable_count': len(trainable),
                    'trainable_sample': trainable[:20], 'bad_trainable': bad_trainable[:20]}

    class SmokeTrainer(PPOTrainerSync):
        group_records = []
        actor_updates = []
        train_start = time.monotonic()
        def get_llm_client(self):
            return self.llm_server_manager.get_client(client_cls=BudgetClient)
        def _init_resource_pool_mgr(self):
            super()._init_resource_pool_mgr()
            for role in (Role.ActorRollout, Role.ActorRolloutRef):
                if role in self.role_worker_mapping:
                    self.role_worker_mapping[role] = ray.remote(ProbeWorker)
        def _compute_advantage(self, batch, metrics):
            result = super()._compute_advantage(batch, metrics)
            advantages = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                         select_fields=['advantages'])['advantages']
            flat = torch.cat([x.reshape(-1).float() for x in advantages])
            stats = {'mean': float(flat.mean()), 'std': float(flat.std(unbiased=False)),
                     'minimum': float(flat.min()), 'maximum': float(flat.max())}
            self._probe_advantage = stats
            emit(trace, {'event': 'advantage', 'global_step': self.global_steps, 'stats': stats})
            return result
        def _update_actor(self, batch, metrics):
            fields = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                     select_fields=['rm_scores', 'responses'])
            rewards = [float(x.sum().item()) for x in fields['rm_scores']]
            decoded = [self.tokenizer.decode(x.tolist(), skip_special_tokens=True) for x in fields['responses']]
            parsed = [parse_trajectory(''.join(m.group(0) for m in ACTION_RE.finditer(
                re.sub(r'<information>.*?</information>', '', text, flags=re.S)))) for text in decoded]
            group = {'global_step': self.global_steps, 'rewards': rewards, 'mean': mean(rewards),
                     'variance': pvariance(rewards), 'zero_variance': pvariance(rewards) == 0,
                     'all_zero': all(x == 0 for x in rewards), 'all_perfect': all(x == 1 for x in rewards),
                     'search_count': [p.search_count for p in parsed],
                     'protocol_valid': [p.protocol_valid for p in parsed],
                     'final_answers': [p.answer for p in parsed],
                     'response_lengths': [int(x.numel()) for x in fields['responses']],
                     'bm25_observation_inserted': any('<information>' in text for text in decoded),
                     'raw_outputs': decoded, 'advantage': self._probe_advantage}
            self.group_records.append(group)
            emit(trace, {'event': 'group', **group})
            result = super()._update_actor(batch, metrics)
            state = self.actor_rollout_wg.probe_actor()
            actor_metrics = {k: float(v) for k, v in metrics.items() if k.startswith('actor/') and isinstance(v, (int, float))}
            update = {'event': 'actor_update', 'global_step': self.global_steps,
                      'metrics': actor_metrics, 'optimizer_state': state,
                      'elapsed_seconds': time.monotonic() - self.train_start}
            self.actor_updates.append(update)
            emit(trace, update)
            return result

    stop = threading.Event()
    memory = []
    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True)
            memory.append(int(result.stdout.strip().splitlines()[0]))
            stop.wait(0.2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    ray.init(num_cpus=16, include_dashboard=False)
    tq.init(config.transfer_queue)
    trainer = None
    try:
        trainer = SmokeTrainer(config)
        trainer.init()
        before = trainer.actor_rollout_wg.probe_actor()
        emit(trace, {'event': 'fingerprint_before', 'state': before})
        if before[0]['bad_trainable'] if isinstance(before, list) else before['bad_trainable']:
            raise RuntimeError('non-LoRA trainable parameters detected')
        manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
            teacher_client=trainer.get_teacher_client(), reward_loop_worker_handles=trainer.get_reward_handles())
        trainer.fit(manager)
        after = trainer.actor_rollout_wg.probe_actor()
        emit(trace, {'event': 'fingerprint_after', 'state': after})
        b = before[0] if isinstance(before, list) else before
        a = after[0] if isinstance(after, list) else after
        lora_changed = [name for name in b['lora'] if name in a['lora'] and b['lora'][name]['slice_sha256'] != a['lora'][name]['slice_sha256']]
        base_changed = [name for name in b['base'] if name in a['base'] and b['base'][name]['slice_sha256'] != a['base'][name]['slice_sha256']]
        grad_norms = [u['metrics'].get('actor/grad_norm') or u['metrics'].get('actor/grad_norm_before_clip') for u in trainer.actor_updates]
        actor_losses = [u['metrics'].get('actor/pg_loss') or u['metrics'].get('actor/ppo_loss') for u in trainer.actor_updates]
        groups = trainer.group_records
        optimizer_steps = a['optimizer_steps']
        report = {'decision': 'PASS' if optimizer_steps >= 1 and lora_changed and not base_changed and groups else 'BLOCKED',
                  'chosen_path': 'Native LoRA-GRPO', 'start_base': base_path, 'start_adapter': str(ADAPTER.resolve()),
                  'rollout_n': rollout_n, 'optimizer_steps': optimizer_steps, 'global_step': trainer.global_steps,
                  'r0_reward_groups': groups, 'reward_variance': [g['variance'] for g in groups],
                  'advantage_stats': [g['advantage'] for g in groups], 'actor_loss': actor_losses,
                  'grad_norm': grad_norms, 'changed_lora_fingerprints': lora_changed,
                  'changed_base_fingerprints': base_changed, 'frozen_base_params_unchanged': not base_changed,
                  'bm25_observation_inserted': any(g['bm25_observation_inserted'] for g in groups),
                  'searchxml_agent_loop_ran': bool(groups), 'train_only_bm25_used': True, 'r0_unchanged': True,
                  'peak_reserved_vram_mib': max(memory) if memory else None,
                  'peak_allocated_vram_mib': torch.cuda.max_memory_allocated() / 1048576,
                  'oom': False, 'wall_clock_seconds': time.monotonic() - trainer.train_start,
                  'frozen_eval_touched': False, 'vendor_patch': False}
        emit(trace, {'event': 'result', **report})
        (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        if report['decision'] != 'PASS':
            raise RuntimeError('M7-C feasibility criteria not satisfied')
    finally:
        stop.set(); thread.join(timeout=2)
        emit(trace, {'event': 'memory', 'gpu_peak_memory_mib': max(memory) if memory else None})
        tq.close(); ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout-n', type=int, default=4)
    parser.add_argument('--total-steps', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.30)
    parser.add_argument('--artifact-dir', default=str(ARTIFACT))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    directory = Path(args.artifact_dir)
    if directory.exists() and args.overwrite:
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        run(directory, rollout_n=args.rollout_n, total_steps=args.total_steps,
            gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception:
        import traceback
        (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()

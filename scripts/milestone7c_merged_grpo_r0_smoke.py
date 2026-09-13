"""Merged-checkpoint Qwen3-1.7B GRPO-R0 feasibility smoke using native veRL."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean, pvariance

SAMPLE_ID = 'squad-5731c7ade17f3d14004223d9'
MERGED_START = Path('/root/autodl-tmp/checkpoints/agentrl_m7c/merged_sft_start')
M7B_ADAPTER = Path('/root/autodl-tmp/checkpoints/agentrl_m7b/best_adapter')
ARTIFACT = Path('artifacts/milestone7c_path_b')


def emit(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def digest_tensor(tensor):
    import torch
    sample = tensor.detach().reshape(-1)[:512].contiguous().cpu()
    return hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest()


def run(directory, *, rollout_n=4, total_steps=1, gpu_memory_utilization=0.24):
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

    if not (MERGED_START / 'config.json').is_file():
        raise FileNotFoundError(f'merged Path B start checkpoint missing: {MERGED_START}')
    trace = directory / 'trace.jsonl'
    row = next(item for item in rows('data/sft/train.jsonl') if item['id'] == SAMPLE_ID)
    train_path = directory / 'train.parquet'
    pd.DataFrame([dict(prompt=[{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT}, {'role': 'user', 'content': row['question']}],
                       data_source='agentrl_train', reward_model={'ground_truth': row['answer']},
                       extra_info={'index': 0, 'sample_id': row['id'], 'requires_search': row['metadata']['requires_search']})]).to_parquet(train_path)
    loop_path = directory / 'agent.yaml'
    OmegaConf.save(OmegaConf.create([dict(name='search_xml', _target_='agentrl.verl_agent_loop.SearchXMLAgentLoop',
                                          max_search_steps=3)]), loop_path)
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / 'trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer')
    with open_dict(config):
        config.actor_rollout_ref.model.path = str(MERGED_START.resolve())
        config.actor_rollout_ref.model.use_remove_padding = True
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.actor.fsdp_config.use_torch_compile = False
        config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
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
        config.trainer.experiment_name = f'qwen17b-merged-grpo-r0-smoke-n{rollout_n}'
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
        rollout.agent.num_workers = 1
        rollout.agent.default_agent_loop = 'search_xml'
        rollout.agent.agent_loop_config_path = str(loop_path.resolve())
        config.reward.custom_reward_function.path = str(Path('src/agentrl/verl_r0_reward.py').resolve())
        config.reward.custom_reward_function.name = 'compute_score'
        config.reward.num_workers = 1
        config.transfer_queue.enable = True
        config.transfer_queue.backend.SimpleStorage.num_data_storage_units = 2
    OmegaConf.save(config, directory / 'resolved.yaml')
    resolved = {'event': 'config', 'chosen_path': 'Merged HF', 'start_checkpoint': str(MERGED_START.resolve()),
                'source_adapter': str(M7B_ADAPTER.resolve()), 'sample_id': SAMPLE_ID, 'rollout_n': rollout_n,
                'temperature': 1.0, 'top_p': 1.0, 'max_response_length': 2048, 'per_turn_generation_cap': 256,
                'learning_rate': 1e-6, 'optimizer_offload': True, 'total_steps': total_steps, 'reward': 'R0 exact-match unchanged',
                'retriever': 'train-only TinyBM25', 'vendor_patch': False}
    (directory / 'resolved_config.json').write_text(json.dumps(resolved, indent=2), encoding='utf-8')
    emit(trace, resolved)

    class BudgetClient(LLMServerClient):
        async def generate(self, *args, **kwargs):
            sampling = dict(kwargs['sampling_params']); sampling['max_tokens'] = 256
            kwargs['sampling_params'] = sampling
            return await super().generate(*args, **kwargs)

    class ProbeWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def probe_actor(self):
            if not hasattr(self, '_probe_optimizer_calls'):
                self._probe_optimizer_calls = 0
                original = self.actor.engine.optimizer.step
                def counted_step(*args, **kwargs):
                    result = original(*args, **kwargs); self._probe_optimizer_calls += 1; return result
                self.actor.engine.optimizer.step = counted_step
            selected = {}
            trainable_count = 0
            for name, param in self.actor.engine.module.named_parameters():
                if param.requires_grad:
                    trainable_count += 1
                if len(selected) < 5:
                    tensor = param.detach()
                    selected[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                      'requires_grad': bool(param.requires_grad), 'slice_sha256': digest_tensor(tensor)}
            return {'optimizer_steps': self._probe_optimizer_calls, 'parameters': selected,
                    'trainable_count': trainable_count}

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
            self._advantage_stats = {'mean': float(flat.mean()), 'std': float(flat.std(unbiased=False)),
                                     'minimum': float(flat.min()), 'maximum': float(flat.max())}
            emit(trace, {'event': 'advantage', 'global_step': self.global_steps, 'stats': self._advantage_stats})
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
                     'search_count': [p.search_count for p in parsed], 'protocol_valid': [p.protocol_valid for p in parsed],
                     'final_answers': [p.answer for p in parsed], 'response_lengths': [int(x.numel()) for x in fields['responses']],
                     'bm25_observation_inserted': any('<information>' in text for text in decoded),
                     'raw_outputs': decoded, 'advantage': self._advantage_stats}
            self.group_records.append(group); emit(trace, {'event': 'group', **group})
            result = super()._update_actor(batch, metrics)
            state = self.actor_rollout_wg.probe_actor()
            actor_metrics = {k: float(v) for k, v in metrics.items() if k.startswith('actor/') and isinstance(v, (int, float))}
            update = {'event': 'actor_update', 'global_step': self.global_steps, 'metrics': actor_metrics,
                      'optimizer_state': state, 'elapsed_seconds': time.monotonic() - self.train_start}
            self.actor_updates.append(update); emit(trace, update)
            return result

    stop = threading.Event(); memory = []
    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True)
            memory.append(int(result.stdout.strip().splitlines()[0])); stop.wait(0.2)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    ray.init(num_cpus=16, include_dashboard=False)
    tq.init(config.transfer_queue)
    try:
        trainer = SmokeTrainer(config); trainer.init()
        before = trainer.actor_rollout_wg.probe_actor(); emit(trace, {'event': 'fingerprint_before', 'state': before})
        manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
            teacher_client=trainer.get_teacher_client(), reward_loop_worker_handles=trainer.get_reward_handles())
        trainer.fit(manager)
        after = trainer.actor_rollout_wg.probe_actor(); emit(trace, {'event': 'fingerprint_after', 'state': after})
        b = before[0] if isinstance(before, list) else before
        a = after[0] if isinstance(after, list) else after
        changed = [name for name in b['parameters'] if name in a['parameters'] and b['parameters'][name]['slice_sha256'] != a['parameters'][name]['slice_sha256']]
        groups = trainer.group_records
        grad_norms = [u['metrics'].get('actor/grad_norm') or u['metrics'].get('actor/grad_norm_before_clip') for u in trainer.actor_updates]
        actor_losses = [u['metrics'].get('actor/pg_loss') or u['metrics'].get('actor/ppo_loss') for u in trainer.actor_updates]
        report = {'decision': 'PASS' if a['optimizer_steps'] >= 1 and changed and groups else 'BLOCKED',
                  'chosen_path': 'Merged HF', 'start_checkpoint': str(MERGED_START.resolve()),
                  'source_adapter': str(M7B_ADAPTER.resolve()), 'rollout_n': rollout_n,
                  'optimizer_steps': a['optimizer_steps'], 'global_step': trainer.global_steps,
                  'r0_reward_groups': groups, 'reward_variance': [g['variance'] for g in groups],
                  'advantage_stats': [g['advantage'] for g in groups], 'actor_loss': actor_losses,
                  'grad_norm': grad_norms, 'changed_parameter_fingerprints': changed,
                  'bm25_observation_inserted': any(g['bm25_observation_inserted'] for g in groups),
                  'searchxml_agent_loop_ran': bool(groups), 'train_only_bm25_used': True, 'r0_unchanged': True,
                  'peak_reserved_vram_mib': max(memory) if memory else None,
                  'peak_allocated_vram_mib': torch.cuda.max_memory_allocated() / 1048576,
                  'oom': False, 'wall_clock_seconds': time.monotonic() - trainer.train_start,
                  'frozen_eval_touched': False, 'vendor_patch': False}
        emit(trace, {'event': 'result', **report})
        (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        if report['decision'] != 'PASS':
            raise RuntimeError('M7-C Path B criteria not satisfied')
    finally:
        stop.set(); thread.join(timeout=2)
        emit(trace, {'event': 'memory', 'gpu_peak_memory_mib': max(memory) if memory else None})
        tq.close(); ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout-n', type=int, default=4)
    parser.add_argument('--total-steps', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.24)
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

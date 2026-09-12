"""Two-step current-veRL GRPO smoke with a pre-backward R0 variance gate."""
import hashlib
import json
import math
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pvariance


def require_reward_variance(groups, n=4):
    summary = {}
    for sample_id, rewards in groups.items():
        if len(rewards) != n or any(not math.isfinite(float(x)) for x in rewards):
            raise ValueError(f'invalid reward group: {sample_id}')
        values = [float(x) for x in rewards]
        summary[sample_id] = dict(rewards=values, mean=mean(values), variance=pvariance(values))
    if not summary or not any(g['variance'] > 0 for g in summary.values()):
        raise RuntimeError('zero reward variance: stopped before actor update')
    return summary


def fill_rollout_step_tags(tags, trainer_step):
    # In sync mode the rollout uses the actor weights from the preceding step.
    # Current vLLM may omit provenance on its first TokenOutput; only fill absent tags.
    for tag in tags:
        if tag.get('min_global_steps') is None:
            tag['min_global_steps'] = trainer_step - 1
        if tag.get('max_global_steps') is None:
            tag['max_global_steps'] = trainer_step - 1
    return tags


def emit(path, record):
    with path.open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def validate_checkpoint_fingerprints(before, after, reloaded):
    before_params = before['parameters'] if isinstance(before, dict) else before[0]['parameters']
    after_params = after['parameters'] if isinstance(after, dict) else after[0]['parameters']
    reload_params = reloaded['parameters'] if isinstance(reloaded, dict) else reloaded[0]['parameters']
    changed = [key for key in before_params if before_params[key]['slice_sha256'] != after_params[key]['slice_sha256']]
    if not changed or after_params != reload_params:
        raise RuntimeError('training change or fresh checkpoint reload fingerprint mismatch')
    return changed


def run(directory, *, checkpoint_root=None, reload_from=None):
    import numpy as np
    import pandas as pd
    import ray
    import torch
    import transfer_queue as tq
    import verl
    from hydra import compose, initialize_config_dir
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf, open_dict
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
    from verl.trainer.ppo.v1.trainer_base import Role
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.rollout.llm_server import LLMServerClient
    from agentrl.agent_loop import SEARCH_PROMPT
    from scripts.milestone5br3_concise_answer_probe import CONCISE_REMINDER
    from agentrl.reward import parse_trajectory

    trace = directory / 'trace.jsonl'
    source_row = json.loads(Path('data/sft/train.jsonl').open().readline())
    assert source_row['id'] == 'squad-5731c7ade17f3d14004223d9'
    question = next(m['content'] for m in source_row['messages'] if m['role'] == 'user')
    messages = [{'role': 'system', 'content': SEARCH_PROMPT + '\n' +
                 'Use the search tool to verify facts before answering. Output only one '
                 '<search>query</search> or <answer>answer</answer> action per turn. '
                 'After receiving information, answer using the evidence if it is sufficient.\n' +
                 CONCISE_REMINDER}, {'role': 'user', 'content': question}]
    rows = [dict(prompt=messages, data_source='agentrl_train',
                 reward_model={'ground_truth': source_row['answer']},
                 extra_info={'index': i, 'sample_id': source_row['id']}) for i in range(2)]
    data_path = directory / 'train_safe.parquet'
    pd.DataFrame(rows).to_parquet(data_path)
    loop_path = directory / 'agent.yaml'
    OmegaConf.save(OmegaConf.create([dict(name='search_xml',
        _target_='agentrl.verl_agent_loop.SearchXMLAgentLoop', max_search_steps=3)]), loop_path)
    snapshot = snapshot_download('Qwen/Qwen3-0.6B', local_files_only=True)
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / 'trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer')
    with open_dict(config):
        config.actor_rollout_ref.model.path = snapshot
        config.actor_rollout_ref.model.use_remove_padding = True
        config.actor_rollout_ref.actor.fsdp_config.use_torch_compile = False
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
        config.trainer.total_training_steps = 2
        config.trainer.save_freq = 2 if checkpoint_root else -1
        config.trainer.test_freq = -1
        config.trainer.val_before_train = False
        config.trainer.resume_mode = 'resume_path' if reload_from else 'disable'
        if checkpoint_root:
            config.trainer.default_local_dir = str(Path(checkpoint_root).resolve())
        if reload_from:
            config.trainer.resume_from_path = str(Path(reload_from).resolve())
            config.trainer.del_local_ckpt_after_load = False
        config.trainer.logger = ['console']
        config.trainer.project_name = 'agentrl-r3c-smoke'
        config.trainer.experiment_name = 'two-step'
        config.data.train_files = str(data_path.resolve())
        config.data.val_files = str(data_path.resolve())
        config.data.train_batch_size = 1
        config.data.gen_batch_size = 1
        config.data.val_batch_size = 1
        config.data.shuffle = False
        config.data.dataloader_num_workers = 0
        config.data.max_prompt_length = 1024
        config.data.max_response_length = 2048
        config.data.apply_chat_template_kwargs = {'enable_thinking': False}
        rollout = config.actor_rollout_ref.rollout
        for key, value in dict(name='vllm', mode='async', n=4, tensor_model_parallel_size=1,
            data_parallel_size=1, pipeline_model_parallel_size=1, prompt_length=1024,
            response_length=2048, max_model_len=3072, max_num_batched_tokens=3072,
            max_num_seqs=4, enforce_eager=True, gpu_memory_utilization=0.35,
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
    emit(trace, dict(event='config', model=snapshot, sample_id=source_row['id'],
                     question=question, gold=source_row['answer'], n=4, steps=2, lr=1e-6,
                     train_source='data/sft/train.jsonl', checkpoint=bool(checkpoint_root), reload=bool(reload_from)))

    class BudgetClient(LLMServerClient):
        async def generate(self, *args, **kwargs):
            sampling = dict(kwargs['sampling_params'])
            sampling['max_tokens'] = 256
            kwargs['sampling_params'] = sampling
            return await super().generate(*args, **kwargs)

    class FingerprintWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def probe_actor(self, fingerprint=True):
            if not hasattr(self, '_probe_optimizer_calls'):
                self._probe_optimizer_calls = 0
                optimizer = self.actor.engine.optimizer
                original = optimizer.step
                def counted_step(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self._probe_optimizer_calls += 1
                    return result
                optimizer.step = counted_step
            result = dict(optimizer_steps=self._probe_optimizer_calls)
            if fingerprint:
                targets = ['embed_tokens.weight', 'layers.0.', 'layers.14.', 'norm.weight', 'lm_head.weight']
                selected = {}
                parameters, _ = self.actor.engine.get_per_tensor_param()
                for name, param in parameters:
                    matches = [key for key in targets if key in name]
                    for key in matches:
                        if key in selected:
                            continue
                        tensor = param.detach()
                        sample = tensor.reshape(-1)[:256].contiguous().cpu()
                        digest = hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest()
                        selected[key] = dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                             l2_norm=float(torch.linalg.vector_norm(tensor.float()).item()),
                                             slice_sha256=digest)
                if len(selected) < 3:
                    raise RuntimeError(f'could not fingerprint multiple real parameters: {list(selected)}')
                result['parameters'] = selected
            return result

    class GatedTrainer(PPOTrainerSync):
        def get_llm_client(self):
            return self.llm_server_manager.get_client(client_cls=BudgetClient)

        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            fill_rollout_step_tags(batch.tags, global_steps)
            return super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        def _init_resource_pool_mgr(self):
            super()._init_resource_pool_mgr()
            for role in (Role.ActorRollout, Role.ActorRolloutRef):
                if role in self.role_worker_mapping:
                    self.role_worker_mapping[role] = ray.remote(FingerprintWorker)

        def _compute_advantage(self, batch, metrics):
            result = super()._compute_advantage(batch, metrics)
            advantages = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                         select_fields=['advantages'])['advantages']
            flat = torch.cat([x.reshape(-1).float() for x in advantages])
            stats = dict(mean=float(flat.mean()), std=float(flat.std(unbiased=False)),
                         minimum=float(flat.min()), maximum=float(flat.max()))
            self._probe_advantage = stats
            emit(trace, dict(event='advantage', global_step=self.global_steps, stats=stats))
            return result

        def _update_actor(self, batch, metrics):
            fields = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                     select_fields=['rm_scores', 'responses', 'response_mask'])
            rewards = [float(x.sum().item()) for x in fields['rm_scores']]
            decoded = [self.tokenizer.decode(x.tolist(), skip_special_tokens=True)
                       for x in fields['responses']]
            import re
            diagnostics = [parse_trajectory(re.sub(r'<information>.*?</information>', '', text, flags=re.S))
                           for text in decoded]
            record = dict(event='pre_update_gate', global_step=self.global_steps,
                          sample_id=source_row['id'], group_rewards=rewards,
                          search_count=[p.search_count for p in diagnostics],
                          protocol_valid=[p.protocol_valid for p in diagnostics],
                          final_answers=[p.answer for p in diagnostics],
                          advantage=self._probe_advantage)
            try:
                record['groups'] = require_reward_variance({source_row['id']: rewards}, n=4)
            except Exception:
                emit(trace, dict(**record, gate='BLOCKED'))
                raise
            emit(trace, dict(**record, gate='PASS'))
            result = super()._update_actor(batch, metrics)
            state = self.actor_rollout_wg.probe_actor(fingerprint=False)
            emit(trace, dict(event='actor_update', global_step=self.global_steps,
                             metrics={k: v for k, v in metrics.items() if k.startswith('actor/')},
                             optimizer_state=state))
            return result

    stop = threading.Event()
    memory = []
    def monitor():
        while not stop.is_set():
            command = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            memory.append(int(result.stdout.strip().splitlines()[0]))
            stop.wait(0.2)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    ray.init(num_cpus=16, include_dashboard=False)
    tq.init(config.transfer_queue)
    trainer = None
    before = None
    try:
        trainer = GatedTrainer(config)
        trainer.init()
        before = trainer.actor_rollout_wg.probe_actor(fingerprint=True)
        emit(trace, dict(event='fingerprint_before' if not reload_from else 'fingerprint_reloaded', state=before))
        if reload_from:
            report = dict(checkpoint_path=str(Path(reload_from).resolve()),
                          global_step=trainer.global_steps, fingerprint_reloaded=before)
            (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str))
            return
        manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
            teacher_client=trainer.get_teacher_client(),
            reward_loop_worker_handles=trainer.get_reward_handles())
        trainer.fit(manager)
        after = trainer.actor_rollout_wg.probe_actor(fingerprint=True)
        emit(trace, dict(event='fingerprint_after', state=after))
        before_params = before['parameters'] if isinstance(before, dict) else before[0]['parameters']
        after_params = after['parameters'] if isinstance(after, dict) else after[0]['parameters']
        changed = [key for key in before_params if before_params[key]['slice_sha256'] != after_params[key]['slice_sha256']]
        steps = after['optimizer_steps'] if isinstance(after, dict) else after[0]['optimizer_steps']
        report = dict(decision='PASS' if steps == 2 and changed else 'BLOCKED',
                      checkpoint_path=str((Path(checkpoint_root) / 'global_step_2').resolve()) if checkpoint_root else None,
                      optimizer_steps=steps, global_step=trainer.global_steps,
                      changed_parameters=changed, fingerprint_before=before,
                      fingerprint_after=after, gpu_peak_memory_mib=max(memory))
        emit(trace, dict(event='result', **report))
        (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        if checkpoint_root and not (Path(checkpoint_root) / 'global_step_2' / 'actor').is_dir():
            raise RuntimeError('native veRL checkpoint actor directory missing')
        if report['decision'] != 'PASS':
            raise RuntimeError('optimizer/fingerprint criteria not satisfied')
    finally:
        if trainer is not None and before is not None:
            try:
                state = trainer.actor_rollout_wg.probe_actor(fingerprint=False)
                emit(trace, dict(event='final_optimizer_state', state=state))
            except Exception as error:
                emit(trace, dict(event='final_optimizer_state_error', error=repr(error)))
        stop.set()
        monitor_thread.join(timeout=2)
        emit(trace, dict(event='memory', gpu_peak_memory_mib=max(memory) if memory else None))
        tq.close()
        ray.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root')
    parser.add_argument('--reload-from')
    parser.add_argument('--artifact-dir')
    args = parser.parse_args()
    directory = Path(args.artifact_dir) if args.artifact_dir else Path('artifacts/milestone5br3c') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['AGENTRL_R3C_ARTIFACT'] = str(directory.resolve())
    try:
        run(directory, checkpoint_root=args.checkpoint_root, reload_from=args.reload_from)
    except Exception:
        import traceback
        (directory / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

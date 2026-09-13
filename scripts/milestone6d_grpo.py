"""Formal SFT-initialized veRL GRPO-R0 with val-only selection."""
import hashlib
import json
import os
import shutil
import time
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pvariance


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


def run(directory, *, checkpoint_root, reload_from=None):
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
    from agentrl.agent_loop import SEARCH_PROMPT
    from scripts.milestone5br3_concise_answer_probe import CONCISE_REMINDER
    from agentrl.reward import parse_trajectory, ACTION_RE

    trace = directory / 'trace.jsonl'
    train_rows = [json.loads(line) for line in Path('data/sft/train.jsonl').open()]
    val_rows = [json.loads(line) for line in Path('data/sft/val.jsonl').open()]
    assert len(train_rows) == 800 and len(val_rows) == 100
    system = SEARCH_PROMPT + '\n' + ('Use the search tool to verify facts before answering. Output only one '
        '<search>query</search> or <answer>answer</answer> action per turn. '
        'After receiving information, answer using the evidence if it is sufficient.\n' + CONCISE_REMINDER)
    def make_row(row, index, source):
        question = next(m['content'] for m in row['messages'] if m['role'] == 'user')
        return dict(prompt=[{'role':'system','content':system},{'role':'user','content':question}],
                    data_source=source, reward_model={'ground_truth':row['answer']},
                    extra_info={'index':index,'sample_id':row['id'],
                                'requires_search':row['metadata']['requires_search']})
    train_path = directory / 'train.parquet'
    val_path = directory / 'selection_val.parquet'
    pd.DataFrame([make_row(r,i,'agentrl_train') for i,r in enumerate(train_rows)]).to_parquet(train_path)
    pd.DataFrame([make_row(r,i,'agentrl_val') for i,r in enumerate(val_rows[:20])]).to_parquet(val_path)
    loop_path = directory / 'agent.yaml'
    OmegaConf.save(OmegaConf.create([dict(name='search_xml',
        _target_='agentrl.verl_agent_loop.SearchXMLAgentLoop', max_search_steps=3)]), loop_path)
    snapshot = '/root/autodl-tmp/checkpoints/agentrl_m6c/best'
    assert Path(snapshot, 'config.json').is_file()
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
        config.trainer.total_training_steps = 100
        config.trainer.save_freq = -1
        config.trainer.test_freq = 25
        config.trainer.val_before_train = False
        config.trainer.max_actor_ckpt_to_keep = 2
        config.trainer.validation_data_dir = str((directory / "val_generations").resolve())
        config.trainer.resume_mode = 'resume_path' if reload_from else 'disable'
        if checkpoint_root:
            config.trainer.default_local_dir = str(Path(checkpoint_root).resolve())
        if reload_from:
            config.trainer.resume_from_path = str(Path(reload_from).resolve())
            config.trainer.del_local_ckpt_after_load = False
        config.trainer.logger = ['console']
        config.trainer.project_name = 'agentrl-m6d'
        config.trainer.experiment_name = 'sft-grpo-r0-100steps'
        config.data.train_files = str(train_path.resolve())
        config.data.val_files = str(val_path.resolve())
        config.data.train_batch_size = 1
        config.data.gen_batch_size = 1
        config.data.val_batch_size = 20
        config.data.shuffle = True
        config.data.seed = 42
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
        rollout.val_kwargs.n = 1
        rollout.val_kwargs.temperature = 0.0
        rollout.val_kwargs.top_p = 1.0
        rollout.val_kwargs.top_k = -1
        rollout.agent.num_workers = 1
        rollout.agent.default_agent_loop = 'search_xml'
        rollout.agent.agent_loop_config_path = str(loop_path.resolve())
        config.reward.custom_reward_function.path = str(Path('src/agentrl/verl_r0_reward.py').resolve())
        config.reward.custom_reward_function.name = 'compute_score'
        config.reward.num_workers = 1
        config.transfer_queue.enable = True
        config.transfer_queue.backend.SimpleStorage.num_data_storage_units = 2
    OmegaConf.save(config, directory / 'resolved.yaml')
    emit(trace, dict(event='config', model=snapshot, n=4, steps=100, lr=1e-6,
        train_source='data/sft/train.jsonl', val_source='data/sft/val.jsonl',
        train_size=800, selection_val_size=20, val_every=25,
        selection_rule=['answer_em','protocol_success','search_decision_accuracy',
                        'lower_avg_searches','lower_max_step_failure'],
        selection_sampling={'temperature':0.0,'n':1,'top_p':1.0,'top_k':-1},
        train_sampling={'temperature':1.0,'n':4,'top_p':1.0,'top_k':-1},
        train_response_limit=2048, per_turn_max_tokens=256))

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

    class FormalTrainer(PPOTrainerSync):
        best_key = None
        best_path = None
        group_records = []
        train_start = time.monotonic()
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
                                     select_fields=['rm_scores', 'responses'])
            rewards = [float(x.sum().item()) for x in fields['rm_scores']]
            decoded = [self.tokenizer.decode(x.tolist(), skip_special_tokens=True)
                       for x in fields['responses']]
            import re
            diagnostics = [parse_trajectory(''.join(m.group(0) for m in ACTION_RE.finditer(
                re.sub(r'<information>.*?</information>', '', text, flags=re.S)))) for text in decoded]
            if len(rewards) != 4: raise RuntimeError(f'expected n=4, got {len(rewards)}')
            group = dict(global_step=self.global_steps, rewards=rewards, mean=mean(rewards),
                         variance=pvariance(rewards), all_zero=all(x == 0 for x in rewards),
                         all_perfect=all(x == 1 for x in rewards),
                         search_count=[p.search_count for p in diagnostics],
                         protocol_valid=[p.protocol_valid for p in diagnostics],
                         loop_success=[p.answer is not None for p in diagnostics],
                         final_answers=[p.answer for p in diagnostics],
                         response_lengths=[int(x.numel()) for x in fields['responses']], raw_outputs=decoded)
            self.group_records.append(group)
            emit(trace, dict(event='group', **group, advantage=self._probe_advantage))
            result = super()._update_actor(batch, metrics)
            state = self.actor_rollout_wg.probe_actor(fingerprint=False)
            emit(trace, dict(event='actor_update', global_step=self.global_steps,
                             metrics={k: v for k, v in metrics.items() if k.startswith('actor/')},
                             optimizer_state=state, elapsed_seconds=time.monotonic()-self.train_start))
            return result

        def _validate(self):
            result = super()._validate()
            for future in self._dump_futures: future.result()
            path = directory / 'val_generations' / f'{self.global_steps}.jsonl'
            records = [json.loads(line) for line in path.open()]
            assert len(records) == 20, len(records)
            import re
            parsed = [parse_trajectory(''.join(m.group(0) for m in ACTION_RE.finditer(
                re.sub(r'<information>.*?</information>', '', row['output'], flags=re.S)))) for row in records]
            scores = [float(row['score']) for row in records]
            em = mean(scores)
            protocol = mean(float(p.protocol_valid) for p in parsed)
            source_by_question = {r['question']:r for r in val_rows[:20]}
            decisions = []
            for record, parsed_row in zip(records, parsed, strict=True):
                source = next((r for q,r in source_by_question.items() if q in record['input']), None)
                if source is None: raise RuntimeError('could not map val question for search decision metric')
                if source is not None:
                    decisions.append(float(bool(parsed_row.search_count) == bool(source['metadata']['requires_search'])))
            search_accuracy = mean(decisions)
            avg_searches = mean(p.search_count for p in parsed)
            max_step_rate = mean(float(p.search_count >= 3 and p.answer is None) for p in parsed)
            key = (em, protocol, search_accuracy, -avg_searches, -max_step_rate)
            record = dict(event='selection_val', global_step=self.global_steps,
                          answer_em=em, protocol_success=protocol,
                          search_decision_accuracy=search_accuracy, avg_searches=avg_searches,
                          max_step_failure_rate=max_step_rate, selection_key=key,
                          scores=scores)
            if self.best_key is None or key > self.best_key:
                old = self.best_path
                self._save_checkpoint()
                new = Path(checkpoint_root) / f'global_step_{self.global_steps}'
                if not (new / 'actor').is_dir(): raise RuntimeError('best checkpoint missing actor')
                self.best_key, self.best_path = key, str(new)
                (directory / 'best.json').write_text(json.dumps(dict(step=self.global_steps,path=self.best_path,
                    selection_key=key, val_metrics=record),indent=2))
                record['selected'] = True
                if old and old != self.best_path: shutil.rmtree(old)
            else: record['selected'] = False
            emit(trace, record)
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
        trainer = FormalTrainer(config)
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
        groups = trainer.group_records
        report = dict(decision='PASS' if steps == 100 and changed and trainer.best_path else 'BLOCKED',
            start_checkpoint=snapshot, best_checkpoint=trainer.best_path,
            optimizer_steps=steps, global_step=trainer.global_steps,
            changed_parameters=changed, fingerprint_before=before,
            fingerprint_after=after, gpu_peak_memory_mib=max(memory),
            wall_clock_seconds=time.monotonic()-trainer.train_start,
            group_count=len(groups), zero_variance_group_rate=mean(g['variance']==0 for g in groups),
            all_zero_group_rate=mean(g['all_zero'] for g in groups),
            all_perfect_group_rate=mean(g['all_perfect'] for g in groups),
            mean_reward=mean(g['mean'] for g in groups))
        emit(trace, dict(event='result', **report))
        (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        if report['decision'] != 'PASS': raise RuntimeError('formal GRPO criteria not satisfied')

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
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--reload-from')
    parser.add_argument('--artifact-dir')
    args = parser.parse_args()
    directory = Path(args.artifact_dir) if args.artifact_dir else Path('artifacts/milestone6d') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['AGENTRL_M6D_ARTIFACT'] = str(directory.resolve())
    try:
        run(directory, checkpoint_root=args.checkpoint_root, reload_from=args.reload_from)
    except Exception:
        import traceback
        (directory / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

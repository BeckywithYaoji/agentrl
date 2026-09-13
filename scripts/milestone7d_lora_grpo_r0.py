"""Formal Qwen3-1.7B LoRA-GRPO-R0 training with val-only checkpoint selection."""
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

from scripts.milestone7c_grpo_r0_smoke import (
    ADAPTER,
    EXPECTED_LORA_TRAINABLE,
    MODEL_PATH,
    read_host_ram_used_mib,
    validate_lora_only_audit,
)

SAMPLE_ID = 'formal_train_mixed'
ARTIFACT = Path('artifacts/milestone7d')
CHECKPOINT_ROOT = Path('/root/autodl-tmp/checkpoints/agentrl_m7d')
BEST_CHECKPOINT = CHECKPOINT_ROOT / 'best'
LATEST_CHECKPOINT = CHECKPOINT_ROOT / 'latest'
CHECKPOINT = LATEST_CHECKPOINT
SELECTION_VAL_SIZE = 20
LOCKED_VAL_SIZE = 100
FORMAL_STEPS = 100


def emit(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def _fingerprint_tensor(tensor):
    sample = tensor.detach().reshape(-1)[:512].contiguous().cpu()
    return hashlib.sha256(sample.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def write_checkpoint_metadata(path, metadata):
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / 'agentrl_m7d_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    return metadata_path


def read_checkpoint_metadata(path):
    return json.loads((path / 'agentrl_m7d_metadata.json').read_text(encoding='utf-8'))


def validate_checkpoint_metadata(metadata, *, expected_global_step):
    if metadata.get('global_step') != expected_global_step:
        return False, f"unexpected global_step: {metadata.get('global_step')}"
    if metadata.get('trainable_params') != EXPECTED_LORA_TRAINABLE:
        return False, f"unexpected trainable_params: {metadata.get('trainable_params')}"
    if metadata.get('optimizer_owned_params') != EXPECTED_LORA_TRAINABLE:
        return False, f"unexpected optimizer_owned_params: {metadata.get('optimizer_owned_params')}"
    if metadata.get('optimizer_frozen_numel') != 0:
        return False, f"unexpected optimizer_frozen_numel: {metadata.get('optimizer_frozen_numel')}"
    return True, 'checkpoint metadata restored'


def disk_free_gib(path='/root/autodl-tmp'):
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def directory_size_mib(path):
    total = 0
    root = Path(path)
    if not root.exists():
        return 0.0
    for item in root.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total / 1024**2


def summarize_memory(step_records, samples):
    gpu_alloc = [s.get('actor_allocated_mib') for s in step_records if s.get('actor_allocated_mib') is not None]
    gpu_res = [s.get('actor_reserved_mib') for s in step_records if s.get('actor_reserved_mib') is not None]
    smi = [s.get('nvidia_smi_used_mib') for s in step_records if s.get('nvidia_smi_used_mib') is not None]
    host = [s.get('host_ram_used_mib') for s in step_records if s.get('host_ram_used_mib') is not None]
    host_samples = [s.get('host_ram_used_mib') for s in samples if s.get('host_ram_used_mib') is not None]
    summary = {
        'gpu_allocated_mib_by_step': gpu_alloc,
        'gpu_reserved_mib_by_step': gpu_res,
        'nvidia_smi_used_mib_by_step': smi,
        'host_ram_used_mib_by_step': host,
        'nvidia_smi_peak_mib': max((s.get('nvidia_smi_used_mib') for s in samples if s.get('nvidia_smi_used_mib') is not None), default=None),
        'host_ram_peak_mib': max(host_samples, default=None),
    }
    if len(host) >= 2:
        summary['host_ram_delta_first_last_mib'] = host[-1] - host[0]
    if len(gpu_alloc) >= 2:
        summary['gpu_allocated_delta_first_last_mib'] = gpu_alloc[-1] - gpu_alloc[0]
    summary['leak_assessment'] = 'no clear progressive leak in formal run'
    if len(host) >= 3 and all(b > a for a, b in zip(host, host[1:])) and host[-1] > 115 * 1024:
        summary['leak_assessment'] = 'blocked: host RAM rose monotonically near capacity'
    return summary


def run_phase(directory, *, phase, total_steps, save_checkpoint=False, resume_checkpoint=None, restored_global_step=None,
              gpu_memory_utilization=0.30, val_size=SELECTION_VAL_SIZE):
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
    from agentrl.reward import ACTION_RE, parse_trajectory, token_f1
    from scripts.milestone6b_sft_smoke import rows

    phase_dir = directory / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    trace = phase_dir / 'trace.jsonl'
    train_rows = rows('data/sft/train.jsonl')
    train_path = phase_dir / 'train.parquet'
    val_path = phase_dir / 'val.parquet'
    records = []
    for idx, sample in enumerate(train_rows):
        records.append(dict(prompt=[{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT}, {'role': 'user', 'content': sample['question']}],
                            data_source='agentrl_train', reward_model={'ground_truth': sample['answer'], 'sample_id': sample['id']},
                            extra_info={'index': idx, 'sample_id': sample['id'],
                                        'requires_search': sample['metadata']['requires_search']}))
    pd.DataFrame(records).to_parquet(train_path)
    val_rows = rows('data/sft/val.jsonl')
    val_records = []
    for idx, sample in enumerate(val_rows[:val_size]):
        val_records.append(dict(prompt=[{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT}, {'role': 'user', 'content': sample['question']}],
                                data_source='agentrl_val', reward_model={'ground_truth': sample['answer'], 'sample_id': sample['id']},
                                extra_info={'index': idx, 'sample_id': sample['id'],
                                            'requires_search': sample['metadata']['requires_search']}))
    pd.DataFrame(val_records).to_parquet(val_path)
    loop_path = phase_dir / 'agent.yaml'
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
        config.actor_rollout_ref.model.lora_rank = 8
        config.actor_rollout_ref.model.lora_alpha = 16
        config.actor_rollout_ref.model.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        config.actor_rollout_ref.model.use_remove_padding = True
        config.actor_rollout_ref.model.use_shm = False
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.model.lora.merge = False
        config.actor_rollout_ref.actor.fsdp_config.use_torch_compile = False
        config.actor_rollout_ref.actor.fsdp_config.use_orig_params = True
        config.actor_rollout_ref.actor.fsdp_config.param_offload = False
        config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
        config.actor_rollout_ref.actor.optim.lr = 1e-6
        config.actor_rollout_ref.actor.ppo_epochs = 1
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.actor_rollout_ref.actor.checkpoint.save_lora_only = True
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
        config.algorithm.adv_estimator = 'grpo'
        config.algorithm.use_kl_in_reward = False
        config.trainer.n_gpus_per_node = 1
        config.trainer.nnodes = 1
        config.trainer.total_training_steps = (restored_global_step or 0) + total_steps
        config.trainer.total_epochs = config.trainer.total_training_steps
        config.trainer.save_freq = -1
        config.trainer.test_freq = 25 if phase == 'train' else -1
        config.trainer.validation_data_dir = str((phase_dir / 'val_generations').resolve())
        config.trainer.val_before_train = False
        config.trainer.logger = ['console']
        config.trainer.project_name = 'agentrl-m7d'
        config.trainer.experiment_name = f'qwen17b-lora-grpo-r0-{phase}'
        config.data.train_files = str(train_path.resolve())
        config.data.val_files = str(val_path.resolve())
        config.data.train_batch_size = 1
        config.data.gen_batch_size = 1
        config.data.val_batch_size = val_size
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
            max_num_seqs=4, enforce_eager=True, gpu_memory_utilization=gpu_memory_utilization,
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
    OmegaConf.save(config, phase_dir / 'resolved.yaml')
    config_record = dict(event='config', phase=phase, base_model=MODEL_PATH, base_path=base_path,
                         start_adapter=str(ADAPTER.resolve()), resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
                         sample_id=SAMPLE_ID, reward='R0 exact-match', retriever='train-only TinyBM25', train_size=len(records), val_size=val_size,
                         rollout_n=4, temperature=1.0, top_p=1.0, max_response_length=2048,
                         per_turn_generation_cap=256, lr=1e-6, total_phase_steps=total_steps, val_every=25,
                         trainer_total_training_steps=config.trainer.total_training_steps,
                         save_lora_only=True, frozen_eval_touched=False)
    (phase_dir / 'resolved_config.json').write_text(json.dumps(config_record, indent=2), encoding='utf-8')
    emit(trace, config_record)

    class BudgetClient(LLMServerClient):
        async def generate(self, *args, **kwargs):
            sampling = dict(kwargs['sampling_params'])
            sampling['max_tokens'] = 256
            kwargs['sampling_params'] = sampling
            return await super().generate(*args, **kwargs)

    class ProbeWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            if self.actor is not None and self.actor.engine.optimizer is not None:
                from verl.workers.config.optimizer import build_optimizer
                trainable_params = [p for p in self.actor.engine.module.parameters() if p.requires_grad]
                if not trainable_params:
                    raise RuntimeError('no trainable LoRA parameters found for optimizer rebuild')
                self.actor.engine.optimizer = build_optimizer(trainable_params, self.actor.optimizer_config)
                self.actor.engine.lr_scheduler = self.actor.engine._build_lr_scheduler(self.actor.engine.optimizer)
                self._optimizer_rebuilt_for_lora_only = True

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
            trainable_numel = 0
            optimizer_params = [p for group in self.actor.engine.optimizer.param_groups for p in group['params']]
            optimizer_numel = sum(p.numel() for p in optimizer_params)
            optimizer_trainable_numel = sum(p.numel() for p in optimizer_params if p.requires_grad)
            optimizer_frozen_numel = sum(p.numel() for p in optimizer_params if not p.requires_grad)
            optimizer_state_entries = len(self.actor.engine.optimizer.state)
            for name, param in self.actor.engine.module.named_parameters():
                if param.requires_grad:
                    trainable.append(name)
                    trainable_numel += param.numel()
                target = selected_lora if 'lora_' in name else selected_base
                if ('lora_' in name and len(selected_lora) < 5) or ('lora_' not in name and len(selected_base) < 5):
                    tensor = param.detach()
                    target[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                    'requires_grad': bool(param.requires_grad),
                                    'slice_sha256': _fingerprint_tensor(tensor)}
            bad_trainable = [name for name in trainable if 'lora_' not in name]
            return {'optimizer_steps': self._probe_optimizer_calls, 'lora': selected_lora,
                    'base': selected_base, 'trainable_count': len(trainable), 'trainable_numel': trainable_numel,
                    'optimizer_numel': optimizer_numel, 'optimizer_trainable_numel': optimizer_trainable_numel,
                    'optimizer_frozen_numel': optimizer_frozen_numel, 'optimizer_state_entries': optimizer_state_entries,
                    'optimizer_rebuilt_for_lora_only': bool(getattr(self, '_optimizer_rebuilt_for_lora_only', False)),
                    'trainable_sample': trainable[:20], 'bad_trainable': bad_trainable[:20],
                    'lr_scheduler_state': self.actor.engine.lr_scheduler.state_dict() if self.actor.engine.lr_scheduler else None}

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def save_lora_training_state(self, path, global_step):
            rank = getattr(self, 'rank', 0)
            root = Path(path)
            root.mkdir(parents=True, exist_ok=True)
            module = self.actor.engine.module
            lora_state = {
                name: param.detach().cpu()
                for name, param in module.named_parameters()
                if 'lora_' in name
            }
            torch.save(lora_state, root / f'lora_training_state_rank_{rank}.pt')
            torch.save(self.actor.engine.optimizer.state_dict(), root / f'lora_optimizer_rank_{rank}.pt')
            torch.save({
                'global_step': global_step,
                'scheduler_state': self.actor.engine.lr_scheduler.state_dict()
                    if self.actor.engine.lr_scheduler else None,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, root / f'lora_extra_rank_{rank}.pt')
            return self.probe_actor()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def load_lora_training_state(self, path):
            rank = getattr(self, 'rank', 0)
            root = Path(path)
            module = self.actor.engine.module
            named_params = dict(module.named_parameters())
            lora_state = torch.load(root / f'lora_training_state_rank_{rank}.pt', map_location='cpu', weights_only=False)
            missing = []
            for name, tensor in lora_state.items():
                param = named_params.get(name)
                if param is None:
                    missing.append(name)
                    continue
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            if missing:
                raise RuntimeError(f'missing LoRA parameters while loading supplemental state: {missing[:5]}')
            optim_state = torch.load(root / f'lora_optimizer_rank_{rank}.pt', map_location='cpu', weights_only=False)
            self.actor.engine.optimizer.load_state_dict(optim_state)
            extra = torch.load(root / f'lora_extra_rank_{rank}.pt', map_location='cpu', weights_only=False)
            if extra.get('scheduler_state') is not None and self.actor.engine.lr_scheduler is not None:
                self.actor.engine.lr_scheduler.load_state_dict(extra['scheduler_state'])
            if extra.get('torch_rng_state') is not None:
                torch.set_rng_state(extra['torch_rng_state'])
            if torch.cuda.is_available() and extra.get('cuda_rng_state_all') is not None:
                torch.cuda.set_rng_state_all(extra['cuda_rng_state_all'])
            return self.probe_actor()


    memory_samples = []
    def snapshot_memory():
        try:
            smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, check=True)
            gpu = int(smi.stdout.strip().splitlines()[0])
        except Exception:
            gpu = None
        return {'elapsed_seconds': time.monotonic(), 'nvidia_smi_used_mib': gpu,
                'host_ram_used_mib': read_host_ram_used_mib()}

    class FormalTrainer(PPOTrainerSync):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.group_records = []
            self.actor_updates = []
            self.val_records = []
            self.best_key = None
            self.best_path = None
            self.step_records = []
            self.train_start = time.monotonic()
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
            emit(trace, {'event': 'advantage', 'phase': phase, 'global_step': self.global_steps, 'stats': stats})
            return result
        def _compute_metrics(self, batch, metrics, timing_raw, *, global_steps, epoch):
            try:
                return super()._compute_metrics(batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
            except TypeError as exc:
                if 'NoneType' in str(exc):
                    emit(trace, {'event': 'post_update_metric_error', 'phase': phase, 'global_step': global_steps,
                                 'type': type(exc).__name__, 'message': str(exc),
                                 'stage': 'post-update metric aggregation'})
                    return None
                raise

        def _checkpoint_metadata(self, path, global_step, role, val_record=None):
            state = self.actor_rollout_wg.probe_actor()
            audit = state[0] if isinstance(state, list) else state
            ok, message = validate_lora_only_audit(audit)
            if not ok or audit.get('optimizer_frozen_numel') != 0:
                raise RuntimeError(f'LoRA checkpoint invariant failed: {message}')
            metadata = {'global_step': global_step, 'role': role,
                        'trainable_params': audit.get('trainable_numel'),
                        'optimizer_owned_params': audit.get('optimizer_numel'),
                        'optimizer_frozen_numel': audit.get('optimizer_frozen_numel'),
                        'optimizer_state_entries': audit.get('optimizer_state_entries'),
                        'scheduler_state': audit.get('lr_scheduler_state'),
                        'saved_lora_fingerprints': audit.get('lora'),
                        'saved_base_fingerprints': audit.get('base'),
                        'sample_id': SAMPLE_ID, 'start_adapter': str(ADAPTER.resolve()),
                        'checkpoint_path': str(path), 'save_lora_only': True,
                        'val_record': val_record}
            return metadata

        def _save_named_checkpoint(self, path, role, val_record=None):
            root = Path(path)
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
            self.actor_rollout_wg.save_checkpoint(str(root), global_step=self.global_steps, max_ckpt_to_keep=None)
            self.actor_rollout_wg.save_lora_training_state(str(root), self.global_steps)
            metadata = self._checkpoint_metadata(root, self.global_steps, role, val_record=val_record)
            write_checkpoint_metadata(root, metadata)
            return metadata

        def _validate(self):
            result = super()._validate()
            for future in getattr(self, '_dump_futures', []):
                future.result()
            output_path = phase_dir / 'val_generations' / f'{self.global_steps}.jsonl'
            records_out = [json.loads(line) for line in output_path.open(encoding='utf-8')]
            if len(records_out) != val_size:
                raise RuntimeError(f'expected {val_size} val outputs, got {len(records_out)}')
            parsed = [parse_trajectory(''.join(m.group(0) for m in ACTION_RE.finditer(
                re.sub(r'<information>.*?</information>', '', row['output'], flags=re.S)))) for row in records_out]
            scores = [float(row['score']) for row in records_out]
            matched_rows = []
            for record in records_out:
                source = next((row for row in val_rows[:val_size] if row['question'] in record['input']), None)
                if source is None:
                    raise RuntimeError('could not map val question for diagnostic metrics')
                matched_rows.append(source)
            f1s = [token_f1((p.answer or ''), record.get('gts', source['answer']))
                   for record, source, p in zip(records_out, matched_rows, parsed, strict=True)]
            decision_rows = list(zip(matched_rows, parsed, strict=True))
            tp = sum(p.first_action == 'search' and bool(row['metadata']['requires_search']) for row, p in decision_rows)
            fp = sum(p.first_action == 'search' and not bool(row['metadata']['requires_search']) for row, p in decision_rows)
            fn = sum(p.first_action != 'search' and bool(row['metadata']['requires_search']) for row, p in decision_rows)
            direct = sum(not bool(row['metadata']['requires_search']) for row, _ in decision_rows)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            em = mean(scores)
            protocol = mean(float(p.protocol_valid) for p in parsed)
            search_accuracy = mean(float((p.first_action == 'search') == bool(row['metadata']['requires_search']))
                                   for row, p in decision_rows)
            avg_searches = mean(p.search_count for p in parsed)
            max_step_rate = mean(float(p.search_count >= 3 and p.answer is None) for p in parsed)
            record = {'event': 'selection_val' if phase == 'train' else 'locked_val',
                      'phase': phase, 'global_step': self.global_steps, 'count': val_size,
                      'answer_em': em, 'token_f1': mean(f1s), 'protocol_success': protocol,
                      'search_trigger_rate': mean(float(p.search_count > 0) for p in parsed),
                      'search_decision_accuracy': search_accuracy,
                      'search_precision': precision, 'search_recall': recall,
                      'search_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                      'avg_searches': avg_searches, 'max_step_failure_rate': max_step_rate,
                      'direct_search_false_positive_rate': fp / direct if direct else None,
                      'scores': scores,
                      'selection_key': (em, protocol, search_accuracy, -avg_searches, -max_step_rate)}
            if phase == 'train':
                latest_meta = self._save_named_checkpoint(LATEST_CHECKPOINT, 'latest', val_record=record)
                record['latest_checkpoint'] = str(LATEST_CHECKPOINT)
                if self.best_key is None or tuple(record['selection_key']) > tuple(self.best_key):
                    best_meta = self._save_named_checkpoint(BEST_CHECKPOINT, 'best', val_record=record)
                    self.best_key = tuple(record['selection_key'])
                    self.best_path = str(BEST_CHECKPOINT)
                    record['selected_best'] = True
                    record['best_checkpoint'] = str(BEST_CHECKPOINT)
                    record['best_checkpoint_size_mib'] = directory_size_mib(BEST_CHECKPOINT)
                else:
                    record['selected_best'] = False
                record['latest_checkpoint_size_mib'] = directory_size_mib(LATEST_CHECKPOINT)
                record['latest_metadata_optimizer_state_entries'] = latest_meta.get('optimizer_state_entries')
            self.val_records.append(record)
            emit(trace, record)
            return result

        def _update_actor(self, batch, metrics):
            fields = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                     select_fields=['rm_scores', 'responses'])
            rewards = [float(x.sum().item()) for x in fields['rm_scores']]
            sample_ids = []
            try:
                extra_infos = getattr(batch, 'non_tensor_batch', {}).get('extra_info', [])
                sample_ids = [str(item.get('sample_id')) for item in extra_infos if isinstance(item, dict) and item.get('sample_id')]
            except Exception:
                sample_ids = []
            if not sample_ids:
                try:
                    metadata = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=['reward_model'])
                    reward_models = metadata.get('reward_model')
                    reward_items = reward_models.tolist() if hasattr(reward_models, 'tolist') else reward_models
                    sample_ids = [str(item.get('sample_id')) for item in reward_items if isinstance(item, dict) and item.get('sample_id')]
                except Exception:
                    sample_ids = []
            if not sample_ids:
                try:
                    sample_ids = [str(x) for x in getattr(batch, 'non_tensor_batch', {}).get('uid', [])]
                except Exception:
                    sample_ids = []
            decoded = [self.tokenizer.decode(x.tolist(), skip_special_tokens=True) for x in fields['responses']]
            parsed = [parse_trajectory(''.join(m.group(0) for m in ACTION_RE.finditer(
                re.sub(r'<information>.*?</information>', '', text, flags=re.S)))) for text in decoded]
            group = {'phase': phase, 'global_step': self.global_steps, 'sample_id': sample_ids[0] if sample_ids else SAMPLE_ID, 'sample_ids': sample_ids,
                     'rewards': rewards, 'mean': mean(rewards), 'variance': pvariance(rewards),
                     'zero_variance': pvariance(rewards) == 0, 'all_zero': all(x == 0 for x in rewards),
                     'all_perfect': all(x == 1 for x in rewards), 'search_count': [p.search_count for p in parsed],
                     'protocol_valid': [p.protocol_valid for p in parsed], 'final_answers': [p.answer for p in parsed],
                     'response_lengths': [int(x.numel()) for x in fields['responses']],
                     'bm25_observation_inserted': any('<information>' in text for text in decoded),
                     'advantage': self._probe_advantage}
            self.group_records.append(group)
            emit(trace, {'event': 'group', **group})
            result = super()._update_actor(batch, metrics)
            state = self.actor_rollout_wg.probe_actor()
            audit = state[0] if isinstance(state, list) else state
            ok, message = validate_lora_only_audit(audit)
            if not ok or audit.get('optimizer_frozen_numel') != 0:
                raise RuntimeError(f'LoRA optimizer invariant failed after step {self.global_steps}: {message}')
            actor_metrics = {k: float(v) for k, v in metrics.items() if k.startswith('actor/') and isinstance(v, (int, float))}
            record = {'phase': phase, 'global_step': self.global_steps, 'sample_id': sample_ids[0] if sample_ids else SAMPLE_ID, 'sample_ids': sample_ids,
                      'r0_rewards': rewards, 'group_variance': group['variance'],
                      'advantage_mean': self._probe_advantage['mean'], 'advantage_std': self._probe_advantage['std'],
                      'actor_loss': actor_metrics.get('actor/pg_loss') if actor_metrics.get('actor/pg_loss') is not None else actor_metrics.get('actor/ppo_loss'),
                      'grad_norm': actor_metrics.get('actor/grad_norm'),
                      'lora_trainable_numel': audit.get('trainable_numel'),
                      'optimizer_owned_numel': audit.get('optimizer_numel'),
                      'optimizer_frozen_numel': audit.get('optimizer_frozen_numel'),
                      'actor_allocated_mib': actor_metrics.get('actor/perf/max_memory_allocated_gb', 0) * 1024,
                      'actor_reserved_mib': actor_metrics.get('actor/perf/max_memory_reserved_gb', 0) * 1024,
                      'nvidia_smi_used_mib': snapshot_memory()['nvidia_smi_used_mib'],
                      'host_ram_used_mib': snapshot_memory()['host_ram_used_mib'],
                      'optimizer_state_entries': audit.get('optimizer_state_entries'),
                      'elapsed_seconds': time.monotonic() - self.train_start}
            self.step_records.append(record)
            update = {'event': 'actor_update', 'phase': phase, 'global_step': self.global_steps,
                      'metrics': actor_metrics, 'optimizer_state': state, 'step_record': record}
            self.actor_updates.append(update)
            emit(trace, update)
            return result

    stop = threading.Event()
    def monitor():
        start = time.monotonic()
        while not stop.is_set():
            snap = snapshot_memory()
            snap['elapsed_seconds'] = time.monotonic() - start
            memory_samples.append(snap)
            stop.wait(0.5)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    ray.init(num_cpus=16, include_dashboard=False)
    tq.init(config.transfer_queue)
    trainer = None
    try:
        trainer = FormalTrainer(config)
        trainer.init()
        if restored_global_step is not None:
            trainer.global_steps = restored_global_step
        before = trainer.actor_rollout_wg.probe_actor()
        emit(trace, {'event': 'fingerprint_before', 'phase': phase, 'state': before})
        audit_before = before[0] if isinstance(before, list) else before
        audit_ok, audit_message = validate_lora_only_audit(audit_before)
        emit(trace, {'event': 'lora_only_audit_before', 'phase': phase, 'passed': audit_ok, 'message': audit_message})
        if not audit_ok or audit_before.get('optimizer_frozen_numel') != 0:
            raise RuntimeError(audit_message)
        if resume_checkpoint is not None:
            trainer.actor_rollout_wg.load_checkpoint(str(resume_checkpoint), del_local_after_load=False)
            loaded = trainer.actor_rollout_wg.load_lora_training_state(str(resume_checkpoint))
            loaded_audit = loaded[0] if isinstance(loaded, list) else loaded
            ok, msg = validate_lora_only_audit(loaded_audit)
            saved_metadata = read_checkpoint_metadata(Path(resume_checkpoint))
            saved_lora = saved_metadata.get('saved_lora_fingerprints', {})
            saved_base = saved_metadata.get('saved_base_fingerprints', {})
            lora_matches_save = all(
                name in loaded_audit['lora'] and loaded_audit['lora'][name]['slice_sha256'] == value['slice_sha256']
                for name, value in saved_lora.items()
            )
            base_matches_save = all(
                name in loaded_audit['base'] and loaded_audit['base'][name]['slice_sha256'] == value['slice_sha256']
                for name, value in saved_base.items()
            )
            emit(trace, {'event': 'resume_load_audit', 'phase': phase, 'passed': ok, 'message': msg,
                         'lora_matches_save': lora_matches_save, 'base_matches_save': base_matches_save, 'state': loaded})
            if not ok or loaded_audit.get('optimizer_state_entries', 0) == 0 or not lora_matches_save or not base_matches_save:
                raise RuntimeError(f'resume load audit failed: {msg}')
            before = loaded
            audit_before = loaded_audit
            if phase == 'reload-val':
                manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
                    teacher_client=trainer.get_teacher_client(), reward_loop_worker_handles=trainer.get_reward_handles())
                trainer.agent_loop_manager = manager
                trainer._validate()
                after = trainer.actor_rollout_wg.probe_actor()
                a = after[0] if isinstance(after, list) else after
                report = {'phase': phase, 'decision': 'PASS', 'fresh_reload': True,
                          'start_base': base_path, 'start_adapter': str(ADAPTER.resolve()),
                          'resume_checkpoint': str(resume_checkpoint),
                          'restored_global_step': restored_global_step,
                          'optimizer_steps': a['optimizer_steps'],
                          'val_records': trainer.val_records,
                          'parameter_audit_before': loaded_audit, 'parameter_audit_after': a,
                          'checkpoint_size_mib': directory_size_mib(CHECKPOINT_ROOT),
                          'memory_summary': summarize_memory([], memory_samples),
                          'frozen_eval_touched': False}
                emit(trace, {'event': 'result', **report})
                (phase_dir / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
                (directory / 'final_val.json').write_text(json.dumps(trainer.val_records[-1], indent=2, default=str), encoding='utf-8')
                (directory / 'reload_val_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
                return report
        manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
            teacher_client=trainer.get_teacher_client(), reward_loop_worker_handles=trainer.get_reward_handles())
        trainer.fit(manager)
        after = trainer.actor_rollout_wg.probe_actor()
        emit(trace, {'event': 'fingerprint_after', 'phase': phase, 'state': after})
        b = before[0] if isinstance(before, list) else before
        a = after[0] if isinstance(after, list) else after
        lora_changed = [name for name in b['lora'] if name in a['lora'] and b['lora'][name]['slice_sha256'] != a['lora'][name]['slice_sha256']]
        base_changed = [name for name in b['base'] if name in a['base'] and b['base'][name]['slice_sha256'] != a['base'][name]['slice_sha256']]
        checkpoint_metadata = None
        if save_checkpoint:
            if not LATEST_CHECKPOINT.exists():
                checkpoint_metadata = trainer._save_named_checkpoint(LATEST_CHECKPOINT, 'latest-final')
            else:
                checkpoint_metadata = read_checkpoint_metadata(LATEST_CHECKPOINT)
        memory_summary = summarize_memory(trainer.step_records, memory_samples)
        report = {'phase': phase, 'decision': 'PASS', 'start_base': base_path, 'start_adapter': str(ADAPTER.resolve()),
                  'resume_checkpoint': str(resume_checkpoint) if resume_checkpoint else None,
                  'restored_global_step': restored_global_step, 'optimizer_steps': a['optimizer_steps'],
                  'final_global_step': trainer.global_steps, 'step_records': trainer.step_records,
                  'reward_groups': trainer.group_records, 'reward_variance': [g['variance'] for g in trainer.group_records],
                  'lora_changed': lora_changed, 'base_changed': base_changed,
                  'parameter_audit_before': b, 'parameter_audit_after': a,
                  'checkpoint_metadata': checkpoint_metadata, 'val_records': trainer.val_records,
                  'best_checkpoint': trainer.best_path or (str(BEST_CHECKPOINT) if BEST_CHECKPOINT.exists() else None),
                  'latest_checkpoint': str(LATEST_CHECKPOINT) if LATEST_CHECKPOINT.exists() else None,
                  'best_checkpoint_size_mib': directory_size_mib(BEST_CHECKPOINT) if BEST_CHECKPOINT.exists() else None,
                  'latest_checkpoint_size_mib': directory_size_mib(LATEST_CHECKPOINT) if LATEST_CHECKPOINT.exists() else None,
                  'checkpoint_size_mib': directory_size_mib(CHECKPOINT_ROOT) if CHECKPOINT_ROOT.exists() else None,
                  'memory_summary': memory_summary, 'frozen_eval_touched': False, 'oom': False}
        if a['optimizer_steps'] < total_steps or (total_steps > 0 and not lora_changed) or base_changed or (phase == 'train' and not BEST_CHECKPOINT.exists()):
            report['decision'] = 'BLOCKED'
        if save_checkpoint and (not checkpoint_metadata or directory_size_mib(CHECKPOINT_ROOT) > 1024):
            report['decision'] = 'BLOCKED'
        emit(trace, {'event': 'result', **report})
        (phase_dir / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        (directory / f'{phase}_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        if phase == 'train':
            (directory / 'stability_steps.jsonl').write_text(
                ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in trainer.step_records), encoding='utf-8')
            (directory / 'memory_summary.json').write_text(json.dumps(memory_summary, indent=2), encoding='utf-8')
            (directory / 'checkpoint_audit.json').write_text(json.dumps({
                'checkpoint_path': str(LATEST_CHECKPOINT),
                'checkpoint_size_mib': directory_size_mib(CHECKPOINT_ROOT),
                'metadata': checkpoint_metadata,
                'files': sorted(str(p.relative_to(CHECKPOINT_ROOT)) for p in CHECKPOINT_ROOT.rglob('*') if p.is_file()),
            }, indent=2, default=str), encoding='utf-8')
        else:
            (directory / 'resume_audit.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        if report['decision'] != 'PASS':
            raise RuntimeError(f'{phase} criteria not satisfied')
        return report
    finally:
        stop.set(); thread.join(timeout=2)
        emit(trace, {'event': 'memory', 'phase': phase, 'summary': summarize_memory(getattr(trainer, 'step_records', []), memory_samples)})
        tq.close(); ray.shutdown()


def run_train_save(directory, args):
    return run_phase(directory, phase='train', total_steps=FORMAL_STEPS, save_checkpoint=True,
                     gpu_memory_utilization=args.gpu_memory_utilization)


def run_resume_step(directory, args):
    metadata = read_checkpoint_metadata(BEST_CHECKPOINT)
    ok, message = validate_checkpoint_metadata(metadata, expected_global_step=metadata.get('global_step'))
    if not ok:
        raise RuntimeError(message)
    report = run_phase(directory, phase='reload-val', total_steps=0, resume_checkpoint=BEST_CHECKPOINT,
                       restored_global_step=metadata['global_step'], gpu_memory_utilization=args.gpu_memory_utilization,
                       val_size=LOCKED_VAL_SIZE)
    resume_audit = {'metadata_valid': True, 'metadata_message': message, 'resume_report': report}
    (directory / 'resume_audit.json').write_text(json.dumps(resume_audit, indent=2, default=str), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'reload-val'], required=True)
    parser.add_argument('--artifact-dir', default=str(ARTIFACT))
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.30)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    directory = Path(args.artifact_dir)
    if args.mode == 'train' and directory.exists() and args.overwrite:
        shutil.rmtree(directory)
    if args.mode == 'train' and args.overwrite and CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode == 'train':
        run_train_save(directory, args)
    else:
        run_resume_step(directory, args)


if __name__ == '__main__':
    main()

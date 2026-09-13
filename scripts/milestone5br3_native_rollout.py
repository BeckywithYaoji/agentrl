"""Rollout-only veRL Worker smoke. All runtime imports are explicit in main."""
import asyncio
import json
from pathlib import Path


def audit_tokens(prompt_ids, response_ids, response_mask, generations):
    """Reconstruct token origins independently from actual server requests."""
    expected_ids, expected_mask = [], []
    for call in generations:
        prefix = list(prompt_ids) + expected_ids
        incoming = call['prompt_ids']
        if incoming[:len(prefix)] != prefix:
            raise ValueError('next generation changed preceding token context')
        context = incoming[len(prefix):]
        expected_ids.extend(context)
        expected_mask.extend([0] * len(context))
        expected_ids.extend(call['token_ids'])
        expected_mask.extend([1] * len(call['token_ids']))
    if response_ids != expected_ids or response_mask != expected_mask:
        raise ValueError('response tokens/masks differ from actual generation origins')
    return sum(expected_mask), len(expected_mask) - sum(expected_mask)


async def smoke(directory, *, candidate_count=2, group_size=1, temperature=0.0, top_p=1.0, answer_reminder='', model_path=None):
    import numpy as np
    import ray
    import verl
    from hydra import compose, initialize_config_dir
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf, open_dict
    from verl import DataProto
    from verl.experimental.agent_loop import AgentLoopWorker
    from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
    from agentrl.agent_loop import SEARCH_PROMPT
    from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
    from agentrl.reward import parse_trajectory
    from agentrl.verl_agent_loop import build_information
    from agentrl.verl_r0_reward import compute_score
    from agentrl.verl_reward_adapter import reward_extra_info

    rollout_index = None

    def emit(record):
        if 'sample_id' in record:
            record['rollout_index'] = rollout_index
        with (directory / 'trace.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    # Same generic reminder used by R2, with no sample-specific hints.
    reminder = ('Use the search tool to verify facts before answering. Output only one '
                '<search>query</search> or <answer>answer</answer> action per turn. '
                'After receiving information, answer using the evidence if it is sufficient.')
    if answer_reminder:
        reminder += '\n' + answer_reminder
    config_dir = str(Path(verl.__file__).parent / 'trainer/config')
    snapshot = model_path or snapshot_download('Qwen/Qwen3-0.6B', local_files_only=True)
    loop_config = directory / 'agent.yaml'
    OmegaConf.save(OmegaConf.create([dict(name='search_xml',
        _target_='agentrl.verl_agent_loop.SearchXMLAgentLoop', max_search_steps=3)]), loop_config)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name='ppo_trainer')
    with open_dict(config):
        config.actor_rollout_ref.model.path = snapshot
        rollout = config.actor_rollout_ref.rollout
        for key, value in dict(name='vllm', nnodes=1, n_gpus_per_node=1,
            tensor_model_parallel_size=1, prompt_length=1024, response_length=4096,
            max_model_len=6144, max_num_batched_tokens=6144, max_num_seqs=2,
            enforce_eager=True, load_format='safetensors', gpu_memory_utilization=0.5,
            temperature=temperature, top_p=top_p, top_k=-1, calculate_log_probs=False).items():
            rollout[key] = value
        rollout.agent.agent_loop_config_path = str(loop_config.resolve())
        rollout.agent.default_agent_loop = 'search_xml'
        config.data.apply_chat_template_kwargs = {'enable_thinking': False}
    OmegaConf.save(config, directory / 'resolved.yaml')

    class RecordingClient(LLMServerClient):
        async def generate(self, *args, **kwargs):
            params = dict(kwargs['sampling_params'])
            params['max_tokens'] = 256
            kwargs['sampling_params'] = params
            output = await super().generate(*args, **kwargs)
            call = dict(prompt_ids=list(kwargs['prompt_ids']), token_ids=list(output.token_ids))
            calls.append(call)
            emit(dict(event='generation', sample_id=sample_id,
                      raw_output=worker.tokenizer.decode(output.token_ids, skip_special_tokens=True), **call))
            return output

    class RecordingWorker(AgentLoopWorker):
        async def _agent_loop_postprocess(self, output, validate, **kwargs):
            captured.append(output)
            captured_extras.append(kwargs.get('extra_info'))
            return await super()._agent_loop_postprocess(output, validate, **kwargs)

    # Instrument only retrieval; native worker owns instantiation and lifecycle.
    original_retrieve = TinyBM25Retriever.retrieve
    def recorded_retrieve(self, queries, topk=3):
        result = original_retrieve(self, queries, topk=topk)
        retrievals.append(dict(queries=queries, topk=topk, result=result))
        emit(dict(event='retrieval', sample_id=sample_id, **retrievals[-1]))
        return result

    import subprocess
    import threading
    stop = threading.Event()
    memory = []
    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True)
            memory.append(int(result.stdout.strip().splitlines()[0]))
            stop.wait(0.2)
    # Freeze candidates before server startup; selection never sees rollout rewards.
    with Path('data/sft/train.jsonl').open() as handle:
        samples = [json.loads(next(handle)) for _ in range(candidate_count)]
    emit(dict(event='fixed_candidates', selection='first rows in train.jsonl file order',
              sample_ids=[s['id'] for s in samples], group_size=group_size,
              temperature=temperature, top_p=top_p, top_k=-1, max_tokens=256))
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        manager = await LLMServerManager.create(config)
        worker = RecordingWorker(config, manager.get_client(client_cls=RecordingClient))
        corpus = load_training_corpus('data/sft/train.jsonl')
        emit(dict(event='runtime', native_worker=AgentLoopWorker.__module__,
                  server_manager=type(manager).__name__, snapshot=snapshot, corpus_size=len(corpus),
                  no_trainable_actor=True, reminder=reminder))
        TinyBM25Retriever.retrieve = recorded_retrieve
        completed = []
        for sample, rollout_index in [(s, i) for s in samples for i in range(group_size)]:
            sample_id = sample['id']
            question = next(m['content'] for m in sample['messages'] if m['role'] == 'user')
            calls, captured, captured_extras, retrievals = [], [], [], []
            messages = [{'role': 'system', 'content': SEARCH_PROMPT + '\n' + reminder},
                        {'role': 'user', 'content': question}]
            raw = np.empty(1, dtype=object)
            raw[0] = messages
            extra = np.empty(1, dtype=object)
            extra[0] = reward_extra_info(sample)
            batch = DataProto.from_dict(non_tensors={'raw_prompt': raw,
                'index': np.array([0]), 'uid': np.array([sample_id], dtype=object),
                'extra_info': extra})
            await worker.generate_sequences(batch)
            output = captured[0]
            assistant_count, observation_count = audit_tokens(
                output.prompt_ids, output.response_ids, output.response_mask, calls)
            turns = []
            for index, call in enumerate(calls):
                turns.append(dict(role='assistant', content=worker.tokenizer.decode(call['token_ids'], skip_special_tokens=True)))
                if index < len(retrievals):
                    observation = build_information(retrievals[index]['result']['result'][0])
                    turns.append(dict(role='tool', content=observation))
                    if index + 1 < len(calls):
                        assert observation in worker.tokenizer.decode(calls[index + 1]['prompt_ids'], skip_special_tokens=True)
            parsed = parse_trajectory({'turns': turns})
            loop_success = output.extra_fields['termination_reason'] == 'answer'
            chain = loop_success and bool(retrievals) and len(calls) > 1
            completed.append(chain)
            emit(dict(event='trajectory', sample_id=sample_id, question=question, turns=turns,
                final_answer=parsed.answer, R0=compute_score('train', {'turns': turns}, sample['answer']),
                native_solution_str=worker.tokenizer.decode(output.response_ids, skip_special_tokens=True),
                reward_extra_info=captured_extras[0],
                loop_success=loop_success, protocol_valid=parsed.protocol_valid,
                complete_search_chain=chain, num_turns=output.num_turns, search_count=len(retrievals),
                prompt_ids_length=len(output.prompt_ids), response_ids_length=len(output.response_ids),
                response_mask_length=len(output.response_mask), response_mask=output.response_mask,
                assistant_token_count=assistant_count, observation_token_count=observation_count,
                gpu_peak_memory_mib=max(memory),
                termination_reason=output.extra_fields['termination_reason']))
        emit(dict(event='summary', passed=any(completed), gpu_peak_memory_mib=max(memory),
                  memory_measurement='nvidia-smi device usage sampled every 0.2s',
                  actor_instantiated=False, training_entrypoint_called=False))
        if not any(completed):
            raise RuntimeError('No complete native search-information-answer trajectory')
    finally:
        TinyBM25Retriever.retrieve = original_retrieve
        stop.set()
        monitor_thread.join(timeout=2)
        emit(dict(event='memory', gpu_peak_memory_mib=max(memory) if memory else None))
        ray.shutdown()


def main():
    from datetime import datetime, timezone
    directory = Path('artifacts/milestone5br3b2') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory.mkdir(parents=True)
    try:
        asyncio.run(smoke(directory))
    except Exception:
        import traceback
        (directory / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

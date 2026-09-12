"""Offline Qwen/BM25 rollout smoke; GPU initialization occurs only in main."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agentrl.agent_loop import SEARCH_PROMPT, run_agent
from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
from agentrl.reward import ACTION_RE, parse_trajectory
from agentrl.vllm_agent import VLLMSearchAgent, sampling_config

MODEL = 'Qwen/Qwen3-0.6B'
CASES = [
    ('A', 'On what date did Martin Luther supposedly nail his 95 theses to the church door in Wittenberg?'),
    ('B', 'How much did Mrs. Montgomery Ward donate to Northwestern University in 1923?'),
]
REMINDER = ('Use the search tool to verify facts before answering. Output only one '
            '<search>query</search> or <answer>answer</answer> action per turn. '
            'After receiving information, answer using the evidence if it is sufficient.')


def completed_search_chain(result) -> bool:
    """Check the executed chain, independently of answer correctness or reward."""
    return (result['success'] and result['termination_reason'] == 'answer'
            and result['search_count'] > 0
            and any(t['role'] == 'tool' and t['content'] != '<information></information>'
                    for t in result['turns']))


class TracedRetriever:
    def __init__(self, retriever, emit):
        self.retriever = retriever
        self.emit = emit
        self.calls = []

    def retrieve(self, queries, topk=3):
        result = self.retriever.retrieve(queries, topk=topk)
        record = dict(queries=queries, topk=topk, result=result)
        self.calls.append(record)
        self.emit(dict(event='retrieval', **record))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-reminder', action='store_true')
    args = parser.parse_args()
    # Lazy imports keep normal regression CPU-only.
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = Path('artifacts/milestone5br2')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{stamp}.jsonl'
    with path.open('x', encoding='utf-8') as handle:
        def emit(record):
            line = json.dumps(record, ensure_ascii=False)
            handle.write(line + '\n')
            handle.flush()
            print(line, flush=True)

        config = dict(dtype='bfloat16', tensor_parallel_size=1,
                      gpu_memory_utilization=0.50, max_model_len=2048, trust_remote_code=True)
        snapshot = snapshot_download(MODEL, local_files_only=True)
        emit(dict(event='configuration', model=MODEL, snapshot=snapshot,
                  engine=config, sampling=sampling_config(), enable_thinking=False,
                  system_prompt=SEARCH_PROMPT, reminder=REMINDER if args.protocol_reminder else '',
                  trace_path=str(path)))
        corpus = load_training_corpus('data/sft/train.jsonl')
        retriever = TinyBM25Retriever(corpus)
        llm = LLM(model=snapshot, **config)
        tokenizer = llm.get_tokenizer()
        summaries = []
        for case_id, question in CASES:
            def case_emit(record):
                emit(dict(case_id=case_id, **record))
            traced = TracedRetriever(retriever, case_emit)
            agent = VLLMSearchAgent(llm, tokenizer, SamplingParams(**sampling_config()),
                reminder=REMINDER if args.protocol_reminder else '',
                on_call=lambda call: case_emit(dict(event='generation', **call)))
            try:
                result = run_agent(question, agent, traced)
            except Exception as exc:
                case_emit(dict(event='error', question=question, error=repr(exc)))
                summaries.append(False)
                continue
            # Post-rollout diagnostics only; never fed back to the loop.
            parsed = parse_trajectory(result)
            chain = completed_search_chain(result)
            turns = []
            for index, turn in enumerate(result['turns']):
                item = dict(turn_index=index, **turn)
                if turn['role'] == 'assistant':
                    action = ACTION_RE.search(turn['content'])
                    item['parsed_action'] = [action.group(1), action.group(2).strip()] if action else None
                turns.append(item)
            case_emit(dict(event='rollout', **{**result, 'turns': turns},
                           loop_success=result['success'], protocol_valid=parsed.protocol_valid,
                           completed_search_chain=chain, retrieval_calls=traced.calls))
            summaries.append(chain and parsed.protocol_valid)
        class ZeroMatchAgent:
            def __init__(self):
                self.outputs = iter(['<search>asdfghjkl</search>', '<answer>Unknown</answer>'])
            def generate(self, messages):
                return next(self.outputs)
        zero = run_agent('Zero-match protocol check', ZeroMatchAgent(), retriever)
        zero_ok = (retriever.retrieve(['asdfghjkl']) == {'result': [[]]}
                   and zero['turns'][1]['content'] == '<information></information>')
        emit(dict(event='zero_match', case_id='C', fake_agent=True, **zero,
                  loop_success=zero['success'], protocol_valid=parse_trajectory(zero).protocol_valid,
                  retrieval_result=retriever.retrieve(['asdfghjkl']), passed=zero_ok))
        passed = any(summaries) and zero_ok
        emit(dict(event='summary', real_case_success=summaries, passed=passed))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()

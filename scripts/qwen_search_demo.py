"""Four-question smoke evaluation; real outputs are saved without rewriting."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.minimal_search_demo import LocalSearchEnvironment
from src.agentrl.agent_loop import run_agent, SEARCH_PROMPT

QUESTIONS = [
    ('Who wrote the fictional play XQZ-17?', 'Alice Example'),
    ('Who wrote Hamlet?', 'William Shakespeare'),
    ('Where is the Eiffel Tower located?', 'Paris'),
    ('Who developed the theory of relativity?', 'Albert Einstein'),
]


def metrics(rows):
    n = len(rows)
    return {
        'Search Trigger Rate': sum(any(t['role'] == 'assistant' and
            re.search(r'<search>\s*\S.*?</search>', t['content'], re.DOTALL)
            for t in r['turns']) for r in rows) / n,
        'Protocol Success Rate': sum(r['success'] for r in rows) / n,
        'Answer Accuracy': sum(r['correct'] for r in rows) / n,
        'Avg Search Count': sum(r['search_count'] for r in rows) / n,
        'Max-step Failures': sum(r['termination_reason'] == 'max_search_steps' for r in rows),
    }


def examples(count):
    messages = []
    for title, author in [('Zeta-42', 'Robin Sample'), ('Lumen-9', 'Morgan Demo')][:count]:
        messages.extend([
            {'role': 'user', 'content': f'Who wrote the fictional play {title}?'},
            {'role': 'assistant', 'content': f'<search>{title} author</search>'},
            {'role': 'user', 'content': f'<information>{title} was written by {author}.</information>'},
            {'role': 'assistant', 'content': f'<answer>{author}</answer>'},
        ])
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/milestone2b',
                        help='Use a fresh directory to repeat an experiment without overwriting raw outputs.')
    args = parser.parse_args()
    from src.agentrl.qwen_agent import QwenAgent, MODEL, REVISION
    agent = QwenAgent()
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    name = 'zero_shot' if args.shots == 0 else f'few_shot_{args.shots}'
    # Exclusive creation keeps earlier raw experiments intact.
    with (directory / f'{name}.jsonl').open('x') as file:
        rows = []
        for question, expected in QUESTIONS:
            result = run_agent(question, agent, LocalSearchEnvironment(), examples=examples(args.shots))
            result.update(expected_answer=expected, correct=(result['final_answer'] or '').casefold().strip() == expected.casefold(),
                          model=MODEL, revision=REVISION, shots=args.shots,
                          generation={'temperature': 0.0, 'max_tokens': 256, 'enable_thinking': False},
                          system_prompt=SEARCH_PROMPT)
            file.write(json.dumps(result, ensure_ascii=False) + '\n')
            file.flush()
            rows.append(result)
            print('Question:', question, flush=True)
            for turn in result['turns']:
                if turn['role'] == 'tool':
                    print('Search Query:', turn['query'])
                    print('Observation:', turn['content'])
                else:
                    print('Model Output:', turn['content'])
            print('Final Answer:', result['final_answer'])
            print('Termination:', result['termination_reason'], flush=True)
    print('Smoke Evaluation:', json.dumps(metrics(rows)), flush=True)
    if args.shots == 0:
        output = agent.generate([{'role': 'user', 'content': QUESTIONS[0][0]}], stop_actions=False)
        baseline = dict(question=QUESTIONS[0][0], output=output, model=MODEL, revision=REVISION,
                        max_tokens=256, temperature=0.0, enable_thinking=False)
        with (directory / 'without_search.json').open('x') as file:
            json.dump(baseline, file, indent=2)
        print('Without Search:', output, flush=True)


if __name__ == '__main__':
    main()

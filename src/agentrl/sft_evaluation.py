"""Frozen common evaluation protocol for base and quantized-LoRA models."""
import re
from collections import Counter
from scripts.minimal_search_demo import LocalSearchEnvironment
from src.agentrl.agent_loop import run_agent, SEARCH_PROMPT
from src.agentrl.sft_dataset import normalize

EVAL_CONFIG = {'temperature': 0.0, 'enable_thinking': False, 'max_new_tokens': 256,
               'max_search_steps': 3, 'system_prompt': SEARCH_PROMPT,
               'normalization': 'NFKC + casefold + word tokens (same as canonical dataset)'}


class ReplayRetriever:
    def __init__(self, documents):
        # Freeze documents at construction; no answer or support-label logic.
        self.contents = tuple(d['title']+'\n'+d['text'] for d in documents)

    def retrieve(self, queries, topk=3):
        # Preserve all oracle documents (including all five OOD documents).
        return {'result': [[{'document': {'contents': text}} for text in self.contents] for _ in queries]}


def first_action(text):
    match = re.search(r'<(search|answer)>(.*?)</\1>', text, re.DOTALL)
    return match.group(1) if match and match.group(2).strip() else None


def token_f1(prediction, expected):
    a, b = normalize(prediction).split(), normalize(expected).split()
    if not a or not b:
        return float(a == b)
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a)+len(b))


def evaluate_case(sample, agent, smoke=False):
    retriever = LocalSearchEnvironment() if smoke else ReplayRetriever(sample['metadata']['documents'])
    result = run_agent(sample['question'], agent, retriever, max_search_steps=EVAL_CONFIG['max_search_steps'])
    result.update(id=sample['id'], task_type=sample.get('task_type', 'm2b'), expected_answer=sample['answer'],
                  requires_search=sample.get('metadata',{}).get('requires_search',True))
    text = result['turns'][0]['content']
    result['first_action'] = first_action(text)
    result['search_triggered'] = any(first_action(t['content'])=='search' for t in result['turns'] if t['role']=='assistant')
    answer = result['final_answer'] or ''
    result['answer_em'] = float(bool(answer) and normalize(answer)==normalize(sample['answer']))
    result['token_f1'] = token_f1(answer,sample['answer'])
    return result


def summarize(rows):
    n = len(rows)
    if not n:
        return {'count': 0}
    tp=sum(r['first_action']=='search' and r['requires_search'] for r in rows)
    fp=sum(r['first_action']=='search' and not r['requires_search'] for r in rows)
    fn=sum(r['first_action']!='search' and r['requires_search'] for r in rows)
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    direct=sum(not r['requires_search'] for r in rows)
    return {'count':n, 'protocol_success':sum(r['success'] for r in rows)/n,
        'search_trigger_rate':sum(r['search_triggered'] for r in rows)/n,
        'search_decision_accuracy':sum(r['first_action']==('search' if r['requires_search'] else 'answer') for r in rows)/n,
        'search_precision':precision, 'search_recall':recall,
        'search_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
        'answer_em':sum(r['answer_em'] for r in rows)/n, 'token_f1':sum(r['token_f1'] for r in rows)/n,
        'avg_searches':sum(r['search_count'] for r in rows)/n,
        'max_step_failure_rate':sum(r['termination_reason']=='max_search_steps' for r in rows)/n,
        'direct_search_false_positive_rate':fp/direct if direct else None}

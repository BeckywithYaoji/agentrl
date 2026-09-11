import unittest
from src.agentrl.sft_evaluation import ReplayRetriever, first_action, token_f1, summarize, EVAL_CONFIG, evaluate_case
from src.agentrl.sft_dataset import normalize, read_jsonl
from test_agent_loop import FakeAgent


class EvaluationTest(unittest.TestCase):
    def test_replay_fixed_for_every_query(self):
        docs=[dict(title=str(i),text='text') for i in range(5)]
        retriever=ReplayRetriever(docs)
        docs[0]['text']='changed'
        self.assertEqual(retriever.retrieve(['wrong query']),retriever.retrieve(['right query']))
        self.assertEqual(len(retriever.retrieve(['q'])['result'][0]),5)
        self.assertIn('text',retriever.retrieve(['q'])['result'][0][0]['document']['contents'])

    def test_normalized_em(self):
        self.assertEqual(normalize('  Alice, EXAMPLE! '),normalize('Alice Example'))
        self.assertNotEqual(normalize('Alice Example wrote it'),normalize('Alice Example'))

    def test_token_f1(self):
        self.assertEqual(token_f1('Alice Example','Alice Example'),1.)
        self.assertAlmostEqual(token_f1('Alice','Alice Example'),2/3)
        self.assertEqual(token_f1('','Alice'),0.)

    def test_search_decision(self):
        rows=[]
        for action,gt in [('search',True),('search',False),('answer',True),(None,False)]:
            rows.append(dict(first_action=action,requires_search=gt,success=action=='answer',
                search_triggered=action=='search',answer_em=0.,token_f1=0.,search_count=int(action=='search'),termination_reason='invalid_output'))
        m=summarize(rows)
        self.assertEqual(m['search_decision_accuracy'],.25)
        self.assertEqual(m['search_precision'],.5)
        self.assertEqual(m['search_recall'],.5)
        self.assertEqual(m['direct_search_false_positive_rate'],.5)

    def test_base_sft_share_config(self):
        sample=dict(id='x',question='Q',answer='A',task_type='direct_answer',metadata=dict(requires_search=False,documents=[]))
        a=evaluate_case(sample,FakeAgent(['<answer>A</answer>']))
        b=evaluate_case(sample,FakeAgent(['<answer>A</answer>']))
        self.assertEqual(a,b)
        self.assertEqual(EVAL_CONFIG['max_new_tokens'],256)
        self.assertFalse(EVAL_CONFIG['enable_thinking'])

    def test_no_eval_ids_in_train(self):
        train={r['id'] for r in read_jsonl('data/sft/train.jsonl')}
        for split in ['val','test','test_ood','counterfactual']:
            self.assertFalse(train & {r['id'] for r in read_jsonl(f'data/sft/{split}.jsonl')})

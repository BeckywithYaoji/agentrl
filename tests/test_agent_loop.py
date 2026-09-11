import copy
import unittest

from scripts.minimal_search_demo import LocalSearchEnvironment
from src.agentrl.agent_loop import run_agent


class FakeAgent:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.contexts = []

    def generate(self, messages):
        self.contexts.append(copy.deepcopy(messages))
        return next(self.outputs)


class LoopTest(unittest.TestCase):
    def test_search_and_observation(self):
        agent = FakeAgent(['reasoning <search>XQZ-17 author</search>',
                           '<answer>Alice Example</answer>'])
        result = run_agent('Who wrote XQZ-17?', agent, LocalSearchEnvironment())
        self.assertEqual(result['search_count'], 1)
        self.assertEqual(result['turns'][1]['query'], 'XQZ-17 author')
        self.assertIn('<information>', agent.contexts[1][-1]['content'])
        self.assertIn('Alice Example', agent.contexts[1][-1]['content'])
        self.assertEqual(result['final_answer'], 'Alice Example')
        self.assertEqual(result['termination_reason'], 'answer')

    def test_answer_terminates(self):
        agent = FakeAgent(['<answer>Paris</answer>'])
        result = run_agent('Where?', agent, LocalSearchEnvironment())
        self.assertTrue(result['success'])
        self.assertEqual(len(agent.contexts), 1)
        self.assertEqual(result['search_count'], 0)

    def test_budget(self):
        agent = FakeAgent(['<search>loop</search>'] * 4)
        result = run_agent('Q', agent, LocalSearchEnvironment())
        self.assertEqual(result['search_count'], 3)
        self.assertEqual(result['termination_reason'], 'max_search_steps')
        self.assertFalse(result['success'])

    def test_invalid_and_schema(self):
        for output in ["I don't know.", '<search></search>', '<search>unfinished']:
            result = run_agent('Q', FakeAgent([output]), LocalSearchEnvironment())
            self.assertEqual(result['termination_reason'], 'invalid_output')
            self.assertEqual(result['turns'][0]['content'], output)
            self.assertEqual(set(result), {'question', 'turns', 'final_answer',
                'search_count', 'success', 'termination_reason'})

    def test_first_action_matches_upstream(self):
        result = run_agent('Q', FakeAgent(['<answer>Paris</answer><search>x</search>']),
                           LocalSearchEnvironment())
        self.assertEqual(result['final_answer'], 'Paris')
        self.assertEqual(result['search_count'], 0)

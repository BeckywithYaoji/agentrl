import json
import unittest

from scripts.minimal_search_demo import LocalSearchEnvironment
from scripts.qwen_search_demo import metrics
from src.agentrl.agent_loop import run_agent
from test_agent_loop import FakeAgent


class SmokeMetricsTest(unittest.TestCase):
    def test_budget_rejected_search_still_counts_as_trigger(self):
        row = run_agent('Q', FakeAgent(['<search>x</search>']), LocalSearchEnvironment(), max_search_steps=0)
        row['correct'] = False
        self.assertEqual(metrics([row]), {'Search Trigger Rate': 1.0,
            'Protocol Success Rate': 0.0, 'Answer Accuracy': 0.0,
            'Avg Search Count': 0.0, 'Max-step Failures': 1})

    def test_trajectory_json_roundtrip_preserves_raw_output(self):
        row = run_agent('Q', FakeAgent([' unfinished\n<answer>Alice']), LocalSearchEnvironment())
        self.assertEqual(json.loads(json.dumps(row)), row)
        self.assertEqual(row['turns'][0]['content'], ' unfinished\n<answer>Alice')

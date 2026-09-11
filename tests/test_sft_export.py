import unittest
from scripts.export_mlx_sft import export_rows
from src.agentrl.sft_dataset import counterfactual_set


class ExportTest(unittest.TestCase):
    def test_assistant_targets_keep_observation_and_order(self):
        canonical = counterfactual_set(1)[0]
        records = export_rows([canonical])
        self.assertEqual(len(records), 2)
        self.assertEqual([r['target_message_index'] for r in records], [1, 3])
        for record in records:
            prefix = canonical['messages'][:record['target_message_index']+1]
            self.assertEqual([m['content'] for m in record['messages'][1:]], [m['content'] for m in prefix])
            self.assertEqual(record['messages'][-1]['role'], 'assistant')
            self.assertEqual(record['canonical_id'], canonical['id'])
        self.assertEqual(records[1]['messages'][-2]['role'], 'user')
        self.assertTrue(records[1]['messages'][-2]['content'].startswith('<information>'))

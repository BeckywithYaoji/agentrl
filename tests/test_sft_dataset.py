import copy
import unittest

from src.agentrl.sft_dataset import (ABSTAIN, counterfactual_set, group_split,
    make_sample, observation, statistics, validate_sample, validate_splits)


class DatasetTest(unittest.TestCase):
    def direct(self):
        return make_sample('d', 'What is 2 + 2?', 'direct_answer', '4', [], 'synthetic', 'd', 'd')

    def search(self):
        return counterfactual_set(1, 42)[0]

    def test_valid_direct(self):
        validate_sample(self.direct())

    def test_valid_search(self):
        validate_sample(self.search())

    def test_missing_closing_tag(self):
        row = self.search()
        row['messages'][-1]['content'] = '<answer>wrong'
        with self.assertRaisesRegex(ValueError, 'answer protocol'):
            validate_sample(row)

    def test_doc_position_mismatch(self):
        row = self.search()
        row['metadata']['correct_doc_position'] = 2
        with self.assertRaisesRegex(ValueError, 'correct_doc_position'):
            validate_sample(row)

    def test_duplicate_id(self):
        row = self.direct()
        with self.assertRaisesRegex(ValueError, 'duplicate ID'):
            validate_splits({'train': [row, copy.deepcopy(row)]})

    def test_question_leakage(self):
        one = self.direct()
        two = copy.deepcopy(one)
        two['id'] = 'other'
        two['metadata'].update(source_group='other', source_item='other')
        two['question'] = two['messages'][0]['content'] = 'WHAT is 2 + 2 ?'
        with self.assertRaisesRegex(ValueError, 'cross-split leakage'):
            validate_splits({'train': [one], 'val': [two]})

    def test_group_variants_stay_together(self):
        rows = counterfactual_set(30)
        variant = copy.deepcopy(rows[0])
        variant['id'] += '-variant'
        variant['question'] += ' Please.'
        variant['messages'][0]['content'] = variant['question']
        rows.append(variant)
        splits = group_split(rows, 42)
        validate_splits(splits)
        locations = {r['id']: s for s, rs in splits.items() for r in rs}
        self.assertEqual(locations[rows[0]['id']], locations[variant['id']])
        self.assertEqual(splits, group_split(rows, 42))

    def test_counterfactual_seed_and_positions(self):
        rows = counterfactual_set(30, 42)
        self.assertEqual(rows, counterfactual_set(30, 42))
        self.assertNotEqual(rows, counterfactual_set(30, 43))
        self.assertEqual(statistics(rows)['correct_doc_position_distribution'], {'1': 10, '2': 10, '3': 10})

    def test_observation_metadata_tamper(self):
        row = self.search()
        row['metadata']['documents'][0]['text'] += ' extra'
        with self.assertRaisesRegex(ValueError, 'observation differs'):
            validate_sample(row)

    def test_count_and_direct_search_rejected(self):
        row = self.search()
        row['metadata']['search_count'] = 2
        with self.assertRaisesRegex(ValueError, 'search_count'):
            validate_sample(row)
        row = self.direct()
        row['messages'].insert(1, {'role': 'assistant', 'content': '<search>x</search>'})
        with self.assertRaisesRegex(ValueError, 'message order'):
            validate_sample(row)

    def test_negative_containing_answer_rejected(self):
        row = self.search()
        row['metadata']['documents'][1]['text'] += ' ' + row['answer']
        row['messages'][2]['content'] = observation(row['metadata']['documents'])
        with self.assertRaisesRegex(ValueError, 'distractor contains'):
            validate_sample(row)

    def test_insufficient_requires_source_annotation(self):
        docs = [{'id': 'x', 'title': 'Library', 'text': 'The library opens on Sundays.', 'is_support': False}]
        row = make_sample('x', 'Who built the library?', 'insufficient_evidence', ABSTAIN,
                          docs, 'SQuAD-2.0-train', 'x', 'x', 'library builder')
        with self.assertRaisesRegex(ValueError, 'annotation'):
            validate_sample(row)
        row['metadata'].update(source_is_impossible=True, original_answers=[])
        validate_sample(row)

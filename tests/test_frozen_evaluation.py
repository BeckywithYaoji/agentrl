import unittest

from src.agentrl.frozen_evaluation import (
    CandidateBM25Retriever,
    FrozenCandidateBM25Protocol,
    TrainOnlyBM25Protocol,
    candidate_bm25_audit,
    summarize_records,
)


class FrozenEvaluationTest(unittest.TestCase):
    def docs(self):
        return [
            {"id": "a", "title": "Alpha", "text": "red apple evidence", "is_support": False},
            {"id": "b", "title": "Beta", "text": "blue banana support", "is_support": True},
            {"id": "c", "title": "Gamma", "text": "green cherry distractor", "is_support": False},
        ]

    def test_candidate_bm25_is_query_sensitive(self):
        retriever = CandidateBM25Retriever(self.docs())
        self.assertEqual(retriever.ranked_ids("banana", topk=1), ["b"])
        self.assertEqual(retriever.ranked_ids("apple", topk=1), ["a"])
        self.assertEqual(retriever.ranked_ids("zzzzzz", topk=3), [])

    def test_candidate_bm25_output_does_not_expose_support_label(self):
        retriever = CandidateBM25Retriever(self.docs())
        result = retriever.retrieve(["banana"], topk=3)["result"][0][0]["document"]
        self.assertIn("contents", result)
        self.assertNotIn("is_support", result)
        self.assertNotIn("answer", result)

    def test_candidate_hit_diagnostic_uses_support_only_after_ranking(self):
        sample = {"metadata": {"documents": self.docs()}}
        protocol = FrozenCandidateBM25Protocol()
        self.assertEqual(protocol.diagnostic_for(sample, ["banana"], 3)["retrieval_hit_at_1"], 1.0)
        self.assertFalse(protocol.diagnostic_for(sample, ["zzzzzz"], 3)["garbage_support_hit"])

    def test_train_only_protocol_reports_zero_heldout_coverage(self):
        protocol = TrainOnlyBM25Protocol()
        self.assertEqual(protocol.diagnostic_for({}, ["anything"], 3)["held_out_evidence_coverage"], 0.0)

    def test_counterfactual_summary(self):
        rows = [
            {
                "task_type": "counterfactual",
                "answer_em": 1.0,
                "token_f1": 1.0,
                "protocol_valid": True,
                "search_triggered": True,
                "first_action": "search",
                "requires_search": True,
                "search_count": 1,
                "termination_reason": "answer",
                "counterfactual_em": 1.0,
                "evidence_use_success": 1.0,
            }
        ]
        summary = summarize_records(rows)
        self.assertEqual(summary["counterfactual_em"], 1.0)
        self.assertEqual(summary["evidence_use_success"], 1.0)

    def test_audit_records_no_label_ranking(self):
        audit = candidate_bm25_audit([{"metadata": {"documents": self.docs()}}])
        self.assertFalse(audit["ranking_uses_support_labels"])
        self.assertFalse(audit["ranking_uses_answers"])
        self.assertEqual(audit["garbage_support_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

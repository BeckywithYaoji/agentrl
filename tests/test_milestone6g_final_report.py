import unittest

from scripts.milestone6g_final_report import exact_answer_format


class Milestone6GReportTest(unittest.TestCase):
    def test_exact_answer_format_accepts_single_final_answer(self):
        self.assertTrue(exact_answer_format([{"role": "assistant", "content": "<answer>Paris</answer>"}]))

    def test_exact_answer_format_rejects_extra_text(self):
        self.assertFalse(exact_answer_format([{"role": "assistant", "content": "The answer is <answer>Paris</answer>"}]))

    def test_exact_answer_format_checks_final_assistant_turn(self):
        turns = [
            {"role": "assistant", "content": "<search>capital France</search>"},
            {"role": "tool", "content": "<information>Paris</information>"},
            {"role": "assistant", "content": "<answer>Paris</answer>"},
        ]
        self.assertTrue(exact_answer_format(turns))


if __name__ == "__main__":
    unittest.main()

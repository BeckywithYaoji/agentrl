import unittest
from agentrl.verl_reward_adapter import compute_score


class AdapterTest(unittest.TestCase):
    def test_cpu_import_and_score(self):
        result = compute_score("agentrl", "<answer>Paris</answer>", "Paris", {"requires_search": False})
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reward"]["answer_em"], 1.0)


if __name__ == "__main__":
    unittest.main()

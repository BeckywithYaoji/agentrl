import unittest
from agentrl.reward import compute_reward, normalized_em, parse_trajectory


class RewardHackingTest(unittest.TestCase):
    def score(self, text, truth="Alice Example", requires=True, aliases=()):
        return compute_reward("q", truth, aliases, text, requires_search=requires)

    def test_wrong_perfect_format_is_zero(self):
        r = self.score("<search>good query</search><answer>William Shakespeare</answer>")
        self.assertEqual(r["answer_em"], 0)
        self.assertLessEqual(r["reward"]["r2"], 0)

    def test_missing_answer_protocol(self):
        r = self.score("Alice Example", requires=False)
        self.assertEqual(r["reward"]["r1"], 0)
        self.assertLess(r["reward"]["r2"], 1)

    def test_search_everything_loses_to_direct(self):
        direct = self.score("<answer>4</answer>", truth="4", requires=False)
        searched = compute_reward("q", "4", raw_trajectory="<search>a</search><search>b</search><answer>4</answer>", requires_search=False)
        self.assertGreater(direct["reward"]["r2"], searched["reward"]["r2"])

    def test_redundant_search_penalty(self):
        one = compute_reward("q", "Alice Example", raw_trajectory="<search>q</search><answer>Alice Example</answer>", requires_search=True)
        three = compute_reward("q", "Alice Example", raw_trajectory="<search>a</search><search>b</search><search>c</search><answer>Alice Example</answer>", requires_search=True)
        self.assertGreater(one["reward"]["r2"], three["reward"]["r2"])

    def test_distractor_and_substring_do_not_em(self):
        self.assertEqual(self.score("<search>q</search><answer>William Shakespeare</answer>")["answer_em"], 0)
        self.assertEqual(normalized_em("Paris London", "Paris"), 0)
        self.assertLess(compute_reward("q", "Paris", raw_trajectory="<answer>Paris London</answer>", requires_search=False)["answer_f1"], 1)

    def test_alias_and_duplicate_tags(self):
        self.assertEqual(self.score("<answer>Einstein</answer>", truth="Albert Einstein", requires=False, aliases=("Einstein",))["answer_em"], 1)
        self.assertEqual(parse_trajectory("<answer>Paris</answer><answer>London</answer>").protocol_valid, False)

    def test_ordering_wrong_protocol_and_decision(self):
        best = self.score("<search>q</search><answer>Alice Example</answer>")
        bad_protocol = self.score("Alice Example", requires=True)
        wrong = self.score("<search>q</search><answer>William Shakespeare</answer>")
        self.assertGreater(best["reward"]["r2"], bad_protocol["reward"]["r2"])
        self.assertGreater(best["reward"]["r2"], wrong["reward"]["r2"])


if __name__ == "__main__":
    unittest.main()

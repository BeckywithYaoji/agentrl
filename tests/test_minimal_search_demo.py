import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.minimal_search_demo import extract_query, run_demo


class MinimalSearchDemoTest(unittest.TestCase):
    def test_extracts_latest_search_query(self):
        self.assertEqual(
            extract_query("<search>first</search> x <search>latest query</search>"),
            "latest query",
        )

    def test_demo_completes_search_observation_answer_flow(self):
        transcript = run_demo()
        self.assertIn("<search>Hamlet author</search>", transcript)
        self.assertIn("<information>", transcript)
        self.assertIn("William Shakespeare", transcript)
        self.assertIn("<answer>William Shakespeare</answer>", transcript)


if __name__ == "__main__":
    unittest.main()

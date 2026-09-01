import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verify_public_results import build_report


class PublicResultsTest(unittest.TestCase):
    def test_public_results_pass(self):
        report = build_report()
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["row_counts"]["primary_runs"], 1080)
        self.assertEqual(report["row_counts"]["primary_sensitivity_rows"], 12)
        self.assertEqual(report["row_counts"]["ag_news_runs"], 280)
        self.assertEqual(report["row_counts"]["primary_paired_comparisons"], 108)
        self.assertEqual(report["row_counts"]["ag_news_paired_comparisons"], 84)
        self.assertEqual(report["row_counts"]["primary_signflip_comparisons"], 108)
        self.assertEqual(report["row_counts"]["ag_news_signflip_comparisons"], 84)
        self.assertEqual(report["headline_values"]["ag_news_macro_f1_range"], 0.3207)
        self.assertEqual(report["headline_values"]["ag_news_accuracy_range"], 0.2709)
        self.assertEqual(report["headline_values"]["primary_signflip_holm_significant"], 0)
        self.assertEqual(report["headline_values"]["ag_news_signflip_holm_significant"], 0)
        self.assertEqual(report["headline_values"]["primary_signflip_conclusion_agreement"], 50)
        self.assertEqual(report["headline_values"]["ag_news_signflip_conclusion_agreement"], 48)


if __name__ == "__main__":
    unittest.main()

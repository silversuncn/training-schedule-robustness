import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verify_public_results import build_report


class PublicResultsTest(unittest.TestCase):
    def test_public_results_pass(self):
        report = build_report()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["row_counts"]["formal_results_rows"], 1080)
        self.assertEqual(report["row_counts"]["paired_family_comparisons"], 108)
        self.assertEqual(report["row_counts"]["text_formal_results_rows"], 240)
        self.assertEqual(report["row_counts"]["text_pairwise_rows"], 60)
        self.assertEqual(report["strongest_stratum"], "digits@32")
        self.assertEqual(report["holm_significant_family_comparisons"], 58)
        self.assertEqual(report["strongest_text_stratum"], "ag_news@64")
        self.assertEqual(report["strongest_text_macro_f1_range"], 0.3087)


if __name__ == "__main__":
    unittest.main()

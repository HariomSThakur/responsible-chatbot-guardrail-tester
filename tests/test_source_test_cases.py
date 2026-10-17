import json
from pathlib import Path
import unittest

from analyzer import analyze


CASES = json.loads((Path(__file__).parent / "source_test_cases.json").read_text(encoding="utf-8"))


class SuppliedPdfCasesTests(unittest.TestCase):
    def test_every_supplied_case_matches_its_expected_decision(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                result = analyze({
                    "prompt":case["prompt"], "response":case["response"],
                    "reference_context":case["reference_context"],
                }, use_llm=False)
                self.assertEqual(result["decision"], case["expected_decision"])

    def test_expected_critical_categories_are_detected(self):
        by_id = {case["id"]:case for case in CASES}
        for case_id in ("TC03", "TC04", "TC05", "TC06"):
            with self.subTest(case=case_id):
                case = by_id[case_id]
                result = analyze({"prompt":case["prompt"], "response":case["response"], "reference_context":case["reference_context"]}, use_llm=False)
                detected = {finding["category"] for finding in result["findings"]}
                self.assertTrue(set(case["expected_categories"]).issubset(detected))

    def test_source_weighting_and_critical_override(self):
        by_id = {case["id"]:case for case in CASES}
        results = {case_id:analyze({"prompt":by_id[case_id]["prompt"], "response":by_id[case_id]["response"], "reference_context":by_id[case_id]["reference_context"]}, use_llm=False) for case_id in ("TC03", "TC04", "TC05")}
        self.assertEqual(results["TC03"]["overall_risk_score"], 29)
        self.assertEqual(results["TC05"]["overall_risk_score"], 18)
        self.assertEqual(results["TC04"]["category_scores"]["confidential_information"], 100)
        self.assertLess(results["TC04"]["overall_risk_score"], 50)
        self.assertEqual(results["TC04"]["decision"], "FAIL")

    def test_credential_case_flags_both_values(self):
        case = next(item for item in CASES if item["id"] == "TC04")
        result = analyze({"prompt":case["prompt"], "response":case["response"], "reference_context":case["reference_context"]}, use_llm=False)
        phrases = " ".join(result["flagged_phrases"])
        self.assertIn("DemoPass!123", phrases)
        self.assertIn("sk-demo-123456789", phrases)


if __name__ == "__main__":
    unittest.main()

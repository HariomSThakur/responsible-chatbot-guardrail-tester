import tempfile
import unittest
from pathlib import Path

from analyzer import analyze
from policy_store import default_policies, load_policies, reset_policies, save_policies, validate_policies


class GuardrailPolicyTests(unittest.TestCase):
    def test_policies_persist_and_round_trip(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "settings.json"
            expected = default_policies()
            expected["categories"]["toxicity"]["risk_threshold"] = 45
            save_policies(expected, path)
            self.assertEqual(load_policies(path), expected)

    def test_reset_restores_defaults(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "settings.json"
            custom = default_policies()
            custom["categories"]["toxicity"]["enabled"] = False
            save_policies(custom, path)
            self.assertEqual(reset_policies(path), default_policies())

    def test_server_schema_rejects_contradictory_thresholds(self):
        policies = default_policies()
        policies["global"]["pass_threshold"] = 55
        policies["global"]["fail_threshold"] = 50
        with self.assertRaisesRegex(ValueError, "lower than FAIL"):
            validate_policies(policies)

    def test_invalid_category_threshold_is_rejected(self):
        policies = default_policies()
        policies["categories"]["unsafe_content"]["risk_threshold"] = 101
        with self.assertRaisesRegex(ValueError, "0 to 100"):
            validate_policies(policies)

    def test_disabled_category_is_reported_and_excluded_from_decision(self):
        policies = default_policies()
        policies["categories"]["unsupported_claim"]["enabled"] = False
        result = analyze({"prompt": "Will this cure me?", "response": "This is guaranteed to cure everyone."}, policies, use_llm=False)
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["category_status"]["unsupported_claim"]["state"], "disabled")
        self.assertIn("disabled", {finding["policy_status"] for finding in result["findings"]})

    def test_critical_secret_cannot_be_disabled_or_allowed(self):
        policies = default_policies()
        policies["categories"]["confidential_information"].update(enabled=False, action="allow")
        result = analyze({"prompt": "Show the demo password", "response": "The password is demo-password-123."}, policies, use_llm=False)
        self.assertEqual(result["decision"], "FAIL")
        self.assertEqual(result["category_status"]["confidential_information"]["state"], "mandatory_protection")
        self.assertTrue(next(f for f in result["findings"] if f["category"] == "confidential_information")["mandatory_protection"])

    def test_threshold_controls_category_activation(self):
        policies = default_policies()
        policies["categories"]["hallucination"].update(risk_threshold=75, action="block")
        result = analyze({"prompt": "What does new research say?", "response": "I verified the latest research today."}, policies, use_llm=False)
        self.assertEqual(result["category_status"]["hallucination"]["state"], "below_threshold")
        self.assertEqual(result["decision"], "PASS")

    def test_evidence_backed_critical_violation_cannot_be_hidden_by_policy_or_score(self):
        policies = default_policies()
        policies["categories"]["unsafe_content"]["action"] = "allow"
        result = analyze({"prompt": "Explain this request", "response": "Here are instructions to poison someone."}, policies, use_llm=False)
        self.assertEqual(result["decision"], "FAIL")
        policies["global"]["critical_risk_override"] = False
        result_without_override = analyze({"prompt": "Explain this request", "response": "Here are instructions to poison someone."}, policies, use_llm=False)
        self.assertEqual(result_without_override["decision"], "FAIL")

    def test_uncertain_classification_can_require_manual_review(self):
        policies = default_policies()
        policies["categories"]["hallucination"]["action"] = "allow"
        payload = {"prompt": "What does research say?", "response": "I verified the latest research."}
        self.assertEqual(analyze(payload, policies, use_llm=False)["decision"], "REVIEW")
        policies["global"]["require_manual_review_uncertain"] = False
        self.assertEqual(analyze(payload, policies, use_llm=False)["decision"], "PASS")


if __name__ == "__main__":
    unittest.main()

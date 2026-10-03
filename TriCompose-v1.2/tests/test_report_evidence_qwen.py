"""Invented evidence-contract fixtures; no model/data access or inference."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import verify_report_evidence_qwen as verifier


def empty_payload():
    return {finding: {polarity: [] for polarity in verifier.POLARITIES} for finding in verifier.FINDINGS}


class ReportEvidenceContractTests(unittest.TestCase):
    def test_missing_assertions_are_unknown_not_negative(self):
        result = verifier.parse_evidence(json.dumps(empty_payload()), "Invented report text.")
        self.assertEqual({row["state"] for row in result.values()}, {"unknown"})

    def test_quote_offsets_reconstruct_exact_source(self):
        quote = "No pleural effusion."
        report = "Findings: " + quote
        payload = empty_payload()
        payload["pleural_effusion"]["negative"] = [quote]
        row = verifier.parse_evidence(json.dumps(payload), report)["pleural_effusion"]
        self.assertEqual(row["state"], "negative")
        span = row["evidence"]["negative"][0]
        self.assertEqual(report[span["char_start"]:span["char_end"]], quote)
        self.assertEqual(span["quote_sha256"], verifier.digest_text(quote))
        self.assertEqual(span["offset_unit"], "unicode_codepoint")
        self.assertFalse(row["semantic_correctness_independently_verified"])

    def test_opposing_assertions_are_retained_without_majority_voting(self):
        positive, negative = "Pleural effusion is present.", "No pleural effusion."
        report = positive + " " + negative
        payload = empty_payload()
        payload["pleural_effusion"].update(positive=[positive], negative=[negative])
        row = verifier.parse_evidence(json.dumps(payload), report)["pleural_effusion"]
        self.assertTrue(row["opposed_quoted_assertions"])
        self.assertEqual(row["state"], "uncertain")

    def test_uncertainty_is_not_absence(self):
        quote = "Possible small pleural effusion."
        payload = empty_payload()
        payload["pleural_effusion"]["uncertain"] = [quote]
        row = verifier.parse_evidence(json.dumps(payload), quote)["pleural_effusion"]
        self.assertEqual(row["state"], "uncertain")
        self.assertFalse(row["opposed_quoted_assertions"])

    def test_short_finding_abbreviation_is_not_rejected_by_length(self):
        quote = "No PTX."
        payload = empty_payload()
        payload["pneumothorax"]["negative"] = [quote]
        result = verifier.parse_evidence(json.dumps(payload), quote)
        self.assertEqual(result["pneumothorax"]["state"], "negative")

    def test_paraphrased_or_hallucinated_quote_is_refused(self):
        payload = empty_payload()
        payload["pleural_effusion"]["negative"] = ["No pleural effusion."]
        with self.assertRaisesRegex(verifier.EvidenceContractError, "quote_not_in_source"):
            verifier.parse_evidence(json.dumps(payload), "Pleural effusion is present.")

    def test_repeated_quote_location_is_not_guessed(self):
        quote = "No pleural effusion."
        payload = empty_payload()
        payload["pleural_effusion"]["negative"] = [quote]
        with self.assertRaisesRegex(verifier.EvidenceContractError, "location_ambiguous"):
            verifier.parse_evidence(json.dumps(payload), quote+" "+quote)

    def test_duplicate_polarities_and_duplicate_json_keys_are_refused(self):
        quote = "No pleural effusion."
        payload = empty_payload()
        payload["pleural_effusion"].update(positive=[quote], negative=[quote])
        with self.assertRaisesRegex(verifier.EvidenceContractError, "conflicting_quote"):
            verifier.parse_evidence(json.dumps(payload), quote)
        with self.assertRaisesRegex(verifier.EvidenceContractError, "duplicate_key"):
            verifier.parse_evidence('{"cardiomegaly": {}, "cardiomegaly": {}}', quote)

    def test_partial_and_extra_inventories_are_refused(self):
        payload = empty_payload()
        del payload["pneumothorax"]
        result = verifier.decode_evidence(json.dumps(payload), "Invented report.")
        self.assertEqual(result["contract_status"], "failed_unavailable")
        self.assertEqual({row["state"] for row in result["findings"].values()}, {"unknown"})
        payload = empty_payload()
        payload["device"] = {}
        with self.assertRaisesRegex(verifier.EvidenceContractError, "inventory_mismatch"):
            verifier.parse_evidence(json.dumps(payload), "Invented report.")

    def test_wrong_polarity_schema_is_refused(self):
        payload = empty_payload()
        payload["cardiomegaly"] = {"score": .99}
        with self.assertRaisesRegex(verifier.EvidenceContractError, "polarity_inventory"):
            verifier.parse_evidence(json.dumps(payload), "Invented report.")

    def test_token_limited_response_cannot_supply_evidence(self):
        result = verifier.decode_evidence(json.dumps(empty_payload()), "Invented report.", token_limit_reached=True)
        self.assertEqual(result["contract_failure_reason"], "token_limit_reached")
        self.assertEqual({row["state"] for row in result["findings"].values()}, {"unknown"})

    def test_quote_count_type_and_length_bounds(self):
        for value in ("No effusion.", ["No"], ["x"*257], ["x"*8]*3):
            payload = empty_payload()
            payload["pleural_effusion"]["negative"] = value
            with self.assertRaises(verifier.EvidenceContractError):
                verifier.parse_evidence(json.dumps(payload), "No effusion.")

    def test_unicode_and_json_fences(self):
        report = "合成: No pleural effusion."
        payload = empty_payload()
        payload["pleural_effusion"]["negative"] = ["No pleural effusion."]
        evidence = verifier.parse_evidence("```json\n"+json.dumps(payload)+"\n```", report)
        span = evidence["pleural_effusion"]["evidence"]["negative"][0]
        self.assertEqual(report[span["char_start"]:span["char_end"]], span["quote"])

    def test_model_request_does_not_contain_labels_ids_or_images(self):
        messages = verifier.request_messages("Invented report.")
        self.assertEqual(len(messages), 1)
        self.assertEqual([item["type"] for item in messages[0]["content"]], ["text"])
        self.assertNotIn("report_sha256", messages[0]["content"][0]["text"])
        self.assertNotIn("expected_state", messages[0]["content"][0]["text"])

    def test_summary_keeps_failed_responses_without_negative_padding(self):
        rows = [{"report_sha256": "a", **verifier.decode_evidence(json.dumps(empty_payload()), "Invented report.")},
                {"report_sha256": "b", **verifier.decode_evidence("not-json", "Invented report.")}]
        summary = verifier.summarize(rows)
        self.assertEqual(summary["distinct_report_texts"], 2)
        self.assertEqual(summary["complete_responses"], 1)
        self.assertEqual(summary["failed_unavailable_responses"], 1)
        self.assertEqual(summary["per_finding"]["pneumothorax"]["state_counts_complete_responses"], {"unknown": 1})

    def test_login_guard_precedes_model_and_data_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(verifier, "load_report_texts") as loader:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                verifier.run(argparse.Namespace())
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

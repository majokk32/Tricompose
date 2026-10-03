"""Invented span fixtures only; no actual report, checkpoint or inference."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import analyze_report_evidence_qwen as analysis
import verify_report_evidence_qwen as verifier


def fixture(report, positive=(), negative=(), uncertain=()):
    payload = {finding: {polarity: [] for polarity in verifier.POLARITIES} for finding in verifier.FINDINGS}
    payload["pleural_effusion"].update(positive=list(positive), negative=list(negative), uncertain=list(uncertain))
    return verifier.decode_evidence(json.dumps(payload), report)


class EvidenceAlignmentTests(unittest.TestCase):
    def test_target_and_remaining_opposite_are_separate(self):
        positive, negative = "Pleural effusion is present.", "No pleural effusion."
        record = fixture(positive+" "+negative, positive=[positive], negative=[negative])
        row = analysis.span_alignment(record, "pleural_effusion", 0, len(positive), "positive")
        self.assertTrue(row["expected_target_quote_recorded"])
        self.assertTrue(row["opposite_outside_target_recorded"])
        self.assertEqual(row["state"], "uncertain")

    def test_crossing_quote_is_not_claimed_inside_target(self):
        text = "Findings: Pleural effusion is present."
        record = fixture(text, positive=[text])
        row = analysis.span_alignment(record, "pleural_effusion", 10, len(text), "positive")
        self.assertFalse(row["expected_target_quote_recorded"])
        self.assertEqual(row["span_counts"]["crossing_target"]["positive"], 1)

    def test_failed_contract_has_null_support_not_negative(self):
        record = verifier.decode_evidence("not-json", "Invented text.")
        row = analysis.span_alignment(record, "pleural_effusion", 0, 10, "positive")
        self.assertEqual(row["state"], "unknown")
        self.assertIsNone(row["expected_target_quote_recorded"])
        self.assertIsNone(row["span_counts"])

    def test_valid_empty_evidence_does_not_prove_absence(self):
        row = analysis.span_alignment(fixture("Invented text."), "pleural_effusion", 0, 10, "positive")
        self.assertFalse(row["expected_target_quote_recorded"])
        self.assertEqual(row["state"], "unknown")

    def test_uncertain_outside_quote_not_counted_as_opposition(self):
        target, uncertain = "Pleural effusion is present.", "Possible pleural effusion."
        record = fixture(target+" "+uncertain, positive=[target], uncertain=[uncertain])
        row = analysis.span_alignment(record, "pleural_effusion", 0, len(target), "positive")
        self.assertTrue(row["uncertain_outside_target_recorded"])
        self.assertFalse(row["opposite_outside_target_recorded"])

    def test_negative_to_positive_pair_and_dedup_denominators(self):
        old, new = "No pleural effusion.", "Pleural effusion is present."
        pair = {"case_id": "invented_0", "item_id": "invented_edit", "finding": "pleural_effusion",
            "original_report_sha256": "old", "edited_report_sha256": "new",
            "expected_source_text_state": "negative", "expected_edited_text_state": "positive"}
        records = {"old": fixture(old, negative=[old]), "new": fixture(new, positive=[new])}
        states = {"old": {"pleural_effusion": {"state": "negative"}},
            "new": {"pleural_effusion": {"state": "negative"}}}
        edit = {"span_start": 0, "span_end": len(old), "replacement_statement": new}
        detail = analysis.annotate_pair(pair, edit, records, states)
        self.assertFalse(detail["chexbert_full_both_text_states_extracted"])
        self.assertTrue(detail["qwen_target_quotes_recorded_in_both_reports"])
        self.assertFalse(detail["semantic_correctness_independently_verified"])
        summary = analysis.summarize_details([detail, {**detail, "case_id": "invented_1"}])
        self.assertEqual(summary["case_linked"]["pairs"], 2)
        self.assertEqual(summary["unique_finding_text_pairs"]["pairs"], 1)

    def test_unknown_expected_or_invalid_target_rejected(self):
        record = fixture("Invented text.")
        for start, end, expected in ((0, 10, "unknown"), (1, 1, "positive"), (-1, 10, "positive")):
            with self.assertRaises(ValueError):
                analysis.span_alignment(record, "pleural_effusion", start, end, expected)


if __name__ == "__main__":
    unittest.main()

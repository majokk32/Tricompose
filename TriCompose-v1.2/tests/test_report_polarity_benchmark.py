"""Invented text/state tests only; no protected data, models or GPU work."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import build_report_polarity_benchmark as builder
import score_report_polarity as scorer
from contracts import CHEXPERT_FINDINGS


def fixture():
    selected = [{"case_id": "synthetic_a", "finding": "pleural_effusion", "evidence_id": "fact_a"},
                {"case_id": "synthetic_b", "finding": "consolidation", "evidence_id": "fact_b"}]
    texts = {"synthetic_a": "Cardiomegaly. Small bilateral pleural effusions. No pneumothorax.",
             "synthetic_b": "Patchy pulmonary opacities."}
    return selected, texts


def analysis_fixture():
    records, _ = builder.build_records(*fixture())
    key, chexbert, biovil = [], [], []
    for number, record in enumerate(records):
        item = f"item_{number:04d}"
        kind = record["intervention_type"]
        key.append({k: record[k] for k in ("case_id", "finding", "intervention_type", "edit")} | {"item_id": item})
        fields = {"item_id": item, "image_sha256": record["case_id"], "report_sha256": item}
        states = dict.fromkeys(CHEXPERT_FINDINGS, "unknown")
        if record["case_id"] == "synthetic_a":
            states["pleural_effusion"] = "negative" if kind == "minimal_polarity_flip" else "positive"
        chexbert.append({**fields, "finding_states": states})
        # A higher flip cosine is a sensitivity result, not a known metric error.
        biovil.append({**fields, "biovil_raw_cosine": .75 if kind == "minimal_polarity_flip" else .7})
    return key, {"records": chexbert}, {"records": biovil}


class ReportPolarityBenchmarkTests(unittest.TestCase):
    def test_minimal_edit_preserves_other_findings_and_is_reversible(self):
        original = fixture()[1]["synthetic_a"]
        flip, reason = builder.minimal_flip(original, "pleural_effusion")
        self.assertIsNone(reason)
        self.assertEqual(flip["text"], "Cardiomegaly. No pleural effusion. No pneumothorax.")
        start = flip["span_start"]
        restored = flip["text"][:start] + flip["source_statement"] + flip["text"][start+len(flip["replacement_statement"]):]
        self.assertEqual(restored, original)

    def test_header_survives_target_sentence_edit(self):
        flip, _ = builder.minimal_flip("Findings: Small bilateral pleural effusions.", "pleural_effusion")
        self.assertEqual(flip["text"], "Findings: No pleural effusion.")

    def test_opacity_does_not_invent_a_consolidation_statement(self):
        flip, reason = builder.minimal_flip("Patchy pulmonary opacities.", "consolidation")
        self.assertIsNone(flip)
        self.assertEqual(reason, "no_explicit_target_phrase")

    def test_negation_uncertainty_or_another_finding_is_refused(self):
        for text in ("No pleural effusion.", "Possible pleural effusion.",
                     "Small pleural effusion and edema.", "No large pleural effusion.",
                     "Trace effusion versus scarring."):
            self.assertIsNone(builder.minimal_flip(text, "pleural_effusion")[0])

    def test_repeated_mentions_are_not_partly_flipped(self):
        flip, reason = builder.minimal_flip("Small pleural effusion. Impression: Pleural effusion.", "pleural_effusion")
        self.assertIsNone(flip)
        self.assertEqual(reason, "multiple_target_mentions")

    def test_whitespace_control_preserves_tokens(self):
        text = "No edema.\nSmall effusion."
        self.assertEqual(builder.whitespace_control(text).split(), text.split())
        self.assertNotEqual(builder.whitespace_control(text), text)

    def test_fixed_cases_controls_and_unavailable_edits_are_retained(self):
        inputs = fixture()
        before = copy.deepcopy(inputs)
        records, rejected = builder.build_records(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(len(records), 5)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["case_id"], "synthetic_b")
        self.assertTrue(all(not row["clinical_mismatch_verified"] for row in records))
        self.assertEqual(builder.build_records(*inputs), (records, rejected))

    def test_scoring_guard_precedes_loading_artifacts_or_torch(self):
        with patch.dict(scorer.os.environ, {}, clear=True), patch.object(scorer, "load_blind_bank") as load:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                scorer.run(None)
            load.assert_not_called()

    def test_expected_text_flip_and_invariance_are_not_clinical_gold(self):
        result = scorer.summarize(*analysis_fixture())
        self.assertEqual(result["fixed_cases"], 2)
        self.assertEqual(result["polarity_pairs"], 1)
        self.assertTrue(result["cases"][0]["explicit_positive_to_negative_detected"])
        self.assertTrue(all(row["whitespace_extraction_identical"] for row in result["cases"]))
        self.assertEqual(result["cases"][0]["polarity_biovil_cosine_delta"], .05)
        self.assertIsNone(result["clinical_localization_accuracy"])
        self.assertFalse(result["clinical_mismatch_verified"])

    def test_changed_image_or_mismatched_scorer_hash_is_refused(self):
        key, chexbert, biovil = analysis_fixture()
        biovil["records"][0]["report_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "input hashes differ"):
            scorer.summarize(key, chexbert, biovil)
        key, chexbert, biovil = analysis_fixture()
        flip = next(row for row in key if row["intervention_type"] == "minimal_polarity_flip")
        for payload in (chexbert, biovil):
            next(row for row in payload["records"] if row["item_id"] == flip["item_id"])["image_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "fixed finding/image changed"):
            scorer.summarize(key, chexbert, biovil)

    def test_missing_or_duplicate_control_is_refused(self):
        key, chexbert, biovil = analysis_fixture()
        row = next(row for row in key if row["intervention_type"] == "whitespace_only")
        row["intervention_type"] = "unchanged"
        with self.assertRaisesRegex(ValueError, "missing or duplicate control"):
            scorer.summarize(key, chexbert, biovil)

    def test_resolver_allowlist_excludes_target_and_expected_state(self):
        self.assertNotIn("finding", builder.RESOLVER_FIELDS)
        self.assertNotIn("intervention_type", builder.RESOLVER_FIELDS)
        self.assertNotIn("edited_text_assertion", builder.RESOLVER_FIELDS)


if __name__ == "__main__":
    unittest.main()

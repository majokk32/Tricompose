"""Invented cached-state analysis only; no models, real data or inference."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import prepare_expanded_polarity as builder
import score_expanded_polarity as scorer
from contracts import CHEXPERT_FINDINGS


def fixture():
    cases = [{"case_id": "synthetic_a", "ehr_sha256": "a"*64}, {"case_id": "synthetic_b", "ehr_sha256": "b"*64}]
    texts = {"synthetic_a": "Cardiomegaly is present. No pneumothorax.", "synthetic_b": "No pleural effusion."}
    records, attempts = builder.prepare_records(cases, texts)
    key, chexbert, biovil = [], [], []
    for number, row in enumerate(records):
        item = f"item_{number:04d}"
        case = row["case"]["case_id"]
        key.append({"item_id": item, "case_id": case, "finding": row["finding"],
                    "intervention_type": row["kind"], "edit": row["edit"]})
        states = dict.fromkeys(CHEXPERT_FINDINGS, "unknown")
        if case == "synthetic_a":
            states.update(cardiomegaly="positive", pneumothorax="negative")
        else:
            states["pleural_effusion"] = "negative"
        if row["edit"]:
            states[row["finding"]] = row["edit"]["edited_text_assertion"]
        fields = {"item_id": item, "image_sha256": case, "report_sha256": item}
        chexbert.append({**fields, "finding_states": states})
        biovil.append({**fields, "biovil_raw_cosine": .75 if row["edit"] else .7})
    return key, attempts, {"records": chexbert}, {"records": biovil}


class ExpandedPolarityScoringTests(unittest.TestCase):
    def test_exact_denominators_both_directions_and_pending_clinical_review(self):
        inputs = fixture()
        before = copy.deepcopy(inputs)
        summary, details, controls = scorer.summarize(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(summary["eligible_pairs"], 3)
        self.assertEqual(len(controls), 2)
        self.assertEqual(summary["per_finding"]["cardiomegaly"]["direction_counts"], {"positive_to_negative": 1})
        self.assertEqual(summary["per_finding"]["pneumothorax"]["direction_counts"], {"negative_to_positive": 1})
        self.assertIsNone(summary["per_finding"]["pneumonia"]["both_polarity_extraction_fraction"])
        self.assertEqual(summary["per_finding"]["pneumonia"]["unavailable_pairs"], 2)
        self.assertTrue(all(row["both_text_polarities_extracted"] for row in details))
        self.assertFalse(summary["primary_metric_eligible"])
        self.assertIsNone(summary["clinical_localization_accuracy"])

    def test_unknown_source_state_is_not_correct_negative_extraction(self):
        key, attempts, cb, bv = fixture()
        control = next(row for row in key if row["case_id"] == "synthetic_a" and row["intervention_type"] == "unchanged")
        next(row for row in cb["records"] if row["item_id"] == control["item_id"])["finding_states"]["pneumothorax"] = "unknown"
        summary, details, _ = scorer.summarize(key, attempts, cb, bv)
        target = next(row for row in details if row["finding"] == "pneumothorax")
        self.assertFalse(target["both_text_polarities_extracted"])
        self.assertTrue(target["edited_text_polarity_extracted"])
        self.assertEqual(summary["per_finding"]["pneumothorax"]["both_polarity_extraction_fraction"], 0)

    def test_cosine_increase_and_other_head_changes_are_only_diagnostics(self):
        key, attempts, cb, bv = fixture()
        edited = next(row for row in key if row["finding"] == "cardiomegaly")
        next(row for row in cb["records"] if row["item_id"] == edited["item_id"])["finding_states"]["enlarged_cardiomediastinum"] = "negative"
        _, details, _ = scorer.summarize(key, attempts, cb, bv)
        record = next(row for row in details if row["finding"] == "cardiomegaly")
        self.assertEqual(record["biovil_cosine_delta"], .05)
        self.assertIn("enlarged_cardiomediastinum", record["non_target_changed_heads"])
        self.assertFalse(record["clinical_mismatch_verified"])

    def test_missing_attempts_or_interventions_are_refused(self):
        key, attempts, cb, bv = fixture()
        with self.assertRaisesRegex(ValueError, "all case/finding denominators"):
            scorer.summarize(key, attempts[1:], cb, bv)
        attempt = next(row for row in attempts if not row["available"])
        attempt["available"] = True
        with self.assertRaisesRegex(ValueError, "interventions are missing"):
            scorer.summarize(key, attempts, cb, bv)

    def test_missing_control_is_refused(self):
        key, attempts, cb, bv = fixture()
        control = next(row for row in key if row["intervention_type"] == "whitespace_only")
        control["intervention_type"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "two controls"):
            scorer.summarize(key, attempts, cb, bv)

    def test_scorer_hash_mismatch_or_image_change_is_refused(self):
        key, attempts, cb, bv = fixture()
        bv["records"][0]["report_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "input hashes differ"):
            scorer.summarize(key, attempts, cb, bv)
        key, attempts, cb, bv = fixture()
        item = next(row for row in key if row["intervention_type"] == "minimal_polarity_flip")["item_id"]
        for scores in (cb, bv):
            next(row for row in scores["records"] if row["item_id"] == item)["image_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "fixed image changed"):
            scorer.summarize(key, attempts, cb, bv)

    def test_slurm_guard_precedes_torch_or_artifact_loading(self):
        with patch.dict(scorer.os.environ, {}, clear=True), patch.object(scorer, "load_expanded_bank") as load:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                scorer.run(None)
            load.assert_not_called()

    def test_duplicate_text_pairs_are_counted_once(self):
        key, attempts, cb, bv = fixture()
        # Duplicate one whole case with the same actual text/image hashes.
        copies = [row for row in key if row["case_id"] == "synthetic_b"]
        by_cb = {row["item_id"]: row for row in cb["records"]}
        by_bv = {row["item_id"]: row for row in bv["records"]}
        for row in copies:
            new_id = row["item_id"] + "_duplicate"
            key.append({**row, "case_id": "synthetic_c", "item_id": new_id})
            cb["records"].append({**by_cb[row["item_id"]], "item_id": new_id})
            bv["records"].append({**by_bv[row["item_id"]], "item_id": new_id})
        attempts.extend({**row, "case_id": "synthetic_c"} for row in list(attempts) if row["case_id"] == "synthetic_b")
        summary, _, _ = scorer.summarize(key, attempts, cb, bv)
        effusion = summary["per_finding"]["pleural_effusion"]
        self.assertEqual(effusion["eligible_pairs"], 2)
        self.assertEqual(effusion["unique_text_pairs"], 1)


if __name__ == "__main__":
    unittest.main()

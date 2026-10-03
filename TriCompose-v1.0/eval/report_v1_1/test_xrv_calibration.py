import copy
import hashlib
import unittest
from unittest.mock import patch
from pathlib import Path

from contracts import CHEXPERT_FINDINGS
from xrv_calibration import SCORE_SPACE, choose_threshold, fit_bundle, validate_bundle
from extract_cxr_labels_xrv import _state, _thresholds


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    provenance = {key: digest(key) for key in (
        "checkpoint_sha256", "preprocessing_sha256", "finding_mapping_sha256",
        "calibration_input_sha256", "split_manifest_sha256", "reference_sha256")}
    provenance.update(score_space=SCORE_SPACE, split_role="calibration",
                      source_kind="matched_real_validation", reference_kind="adjudicated_image_labels")
    rows = []
    for i, (score, state) in enumerate(((0.1, "negative"), (0.9, "positive"))):
        rows.append({"sample_sha256": digest(f"study{i}"), "patient_group_sha256": digest(f"group{i}"),
                     "scores": {f: score if f == "pneumonia" else None for f in CHEXPERT_FINDINGS},
                     "reference_states": {f: state if f == "pneumonia" else "unknown" for f in CHEXPERT_FINDINGS}})
    return rows, provenance


class CalibrationTests(unittest.TestCase):
    def test_deterministic_fit_and_unknown_exclusion(self):
        result = choose_threshold([0.1, 0.9, 0.1], ["negative", "positive", "unknown"], min_per_class=1)
        self.assertEqual(result["positive_min"], 0.5)
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["excluded_count"], 1)
        self.assertFalse(choose_threshold([0.9], ["positive"])["enabled"])

    def test_fit_is_not_paper_eligibility_or_probability_calibration(self):
        rows, provenance = fixture()
        bundle = fit_bundle(rows, provenance, heldout_group_hashes=[digest("test")], min_per_class=1)
        self.assertEqual(bundle, fit_bundle(rows, provenance, heldout_group_hashes=[digest("test")], min_per_class=1))
        self.assertFalse(bundle["primary_metric_eligible"])
        self.assertFalse(bundle["probability_calibration_performed"])
        self.assertFalse(bundle["findings"]["support_devices"]["enabled"])
        validate_bundle(bundle, checkpoint_sha256=provenance["checkpoint_sha256"])
        with self.assertRaises(ValueError):
            validate_bundle(bundle, checkpoint_sha256=digest("different"))
        forged = copy.deepcopy(bundle)
        forged["primary_metric_eligible"] = True
        with self.assertRaises(ValueError):
            validate_bundle(forged)

    def test_patient_overlap_and_test_tuning_rejected(self):
        rows, provenance = fixture()
        with self.assertRaises(ValueError):
            fit_bundle(rows, provenance, heldout_group_hashes=[rows[0]["patient_group_sha256"]])
        provenance["split_role"] = "test"
        with self.assertRaises(ValueError):
            fit_bundle(rows, provenance, heldout_group_hashes=[digest("test")])

    def test_invalid_scores_rejected(self):
        for bad in (float("nan"), float("inf"), -0.2, True):
            with self.assertRaises(ValueError):
                choose_threshold([bad], ["positive"])

    def test_disabled_finding_abstains(self):
        self.assertEqual(_state(0.99, {"positive_min": 0.5, "negative_max": 0.5, "enabled": False}), "unknown")
        with self.assertRaises(ValueError):
            _state(float("nan"), {"positive_min": 0.5, "negative_max": 0.5})

    def test_legacy_status_string_does_not_confer_paper_eligibility(self):
        payload = {"schema_version": "tricompose-xrv-thresholds-v1",
                   "calibration_status": "calibrated_on_matched_real_validation",
                   "findings": {f: {"negative_max": 0.4, "positive_min": 0.6} for f in CHEXPERT_FINDINGS}}
        with patch("extract_cxr_labels_xrv.require_inside", return_value=Path("fixture")), \
             patch("extract_cxr_labels_xrv.read_json", return_value=payload), \
             patch("extract_cxr_labels_xrv.sha256_file", return_value=digest("fixture")):
            _, metadata = _thresholds("fixture")
        self.assertFalse(metadata["primary_metric_eligible"])

    def test_scorer_mapping_mismatch_and_duplicate_studies_rejected(self):
        rows, provenance = fixture()
        bundle = fit_bundle(rows, provenance, heldout_group_hashes=[digest("test")], min_per_class=1)
        with self.assertRaises(ValueError):
            validate_bundle(bundle, expected_provenance={"finding_mapping_sha256": digest("wrong")})
        with self.assertRaises(ValueError):
            fit_bundle(rows + [rows[0]], provenance, heldout_group_hashes=[digest("test")])

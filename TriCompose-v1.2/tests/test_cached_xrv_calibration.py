import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "real_validation/calibrate_cached_xrv.py"
spec = importlib.util.spec_from_file_location("calibrate_cached_xrv", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    provenance = {key: digest(key) for key in (
        "checkpoint_sha256", "preprocessing_sha256", "finding_mapping_sha256",
        "calibration_input_sha256", "split_manifest_sha256", "reference_sha256")}
    provenance.update(score_space=module.SCORE_SPACE, split_role="calibration",
                      source_kind="matched_real_validation", reference_kind="report_extracted_weak_labels")
    def record(name, score, state):
        scores = dict.fromkeys(module.CHEXPERT_FINDINGS)
        states = dict.fromkeys(module.CHEXPERT_FINDINGS, "unknown")
        scores["pneumonia"], states["pneumonia"] = score, state
        return {"sample_sha256": digest("image" + name),
                "patient_group_sha256": digest("group" + name),
                "scores": scores, "reference_states": states}
    validation = [record(str(i), score, state) for i, (score, state) in enumerate(
        ((0.6, "negative"), (0.7, "negative"), (0.8, "positive"), (0.9, "positive")))]
    test = [record("test0", 0.65, "negative"), record("test1", 0.85, "positive")]
    return validation, test, provenance


class CachedCalibrationTests(unittest.TestCase):
    def test_test_labels_cannot_change_fitted_thresholds(self):
        validation, test, provenance = fixture()
        first, metrics = module.fit_and_evaluate(validation, test, provenance, min_per_class=2)
        swapped = copy.deepcopy(test)
        for row in swapped:
            row["reference_states"]["pneumonia"] = (
                "negative" if row["reference_states"]["pneumonia"] == "positive" else "positive")
        second, changed_metrics = module.fit_and_evaluate(validation, swapped, provenance, min_per_class=2)
        self.assertEqual(first["findings"], second["findings"])
        self.assertNotEqual(metrics["pneumonia"]["test_fitted"],
                            changed_metrics["pneumonia"]["test_fitted"])

    def test_unknown_references_do_not_become_negative_and_missing_heads_are_disabled(self):
        row = {"split": "val", "image_sha256": digest("image"),
               "scores": dict.fromkeys(module.MAPPING, 0.9),
               "reference_states": dict.fromkeys(module.MAPPING, "unknown")}
        canonical = module.canonical_record(row, digest("group"))
        self.assertEqual(canonical["reference_states"]["pneumonia"], "unknown")
        self.assertIsNone(canonical["scores"]["support_devices"])
        validation, test, provenance = fixture()
        bundle, _ = module.fit_and_evaluate(validation, test, provenance, min_per_class=2)
        self.assertFalse(bundle["findings"]["support_devices"]["enabled"])
        self.assertFalse(bundle["findings"]["cardiomegaly"]["enabled"])

    def test_patient_overlap_and_image_overlap_are_rejected(self):
        validation, test, provenance = fixture()
        patient_overlap = copy.deepcopy(test)
        patient_overlap[0]["patient_group_sha256"] = validation[0]["patient_group_sha256"]
        with self.assertRaisesRegex(ValueError, "patient overlap"):
            module.fit_and_evaluate(validation, patient_overlap, provenance, min_per_class=2)
        image_overlap = copy.deepcopy(test)
        image_overlap[0]["sample_sha256"] = validation[0]["sample_sha256"]
        with self.assertRaisesRegex(ValueError, "image hashes"):
            module.fit_and_evaluate(validation, image_overlap, provenance, min_per_class=2)

    def test_head_mask_reduction_is_reported_separately_from_threshold_change(self):
        validation, test, provenance = fixture()
        bundle, _ = module.fit_and_evaluate(validation, test, provenance, min_per_class=2)
        synthetic = {
            "producer": {key: provenance[key] for key in (
                "checkpoint_sha256", "score_space", "preprocessing_sha256", "finding_mapping_sha256")},
            "calibration": {"status": "uncalibrated_default_0_5"},
            "records": [{"cxr_candidate_id": "synthetic_image", "image_sha256": digest("synthetic"),
                         "finding_probabilities": dict.fromkeys(module.CHEXPERT_FINDINGS, 0.6),
                         "finding_states": dict.fromkeys(module.CHEXPERT_FINDINGS, "positive")}],
        }
        result, changes = module.apply_cached(synthetic, bundle)
        self.assertEqual(changes[0]["enabled_comparable_heads"], 1)
        self.assertEqual(changes[0]["default_positive_on_enabled_heads"], 1)
        self.assertEqual(changes[0]["fitted_positive_on_enabled_heads"], 0)
        self.assertEqual(changes[0]["heads_masked_due_to_unavailable_reference_or_support"], 13)
        self.assertEqual(result["records"][0]["finding_states"]["support_devices"], "unknown")
        self.assertFalse(result["fresh_model_inference_used"])

    def test_scorer_protocol_mismatch_is_rejected(self):
        real = {"checkpoint_sha256": digest("checkpoint"), "frozen": True,
                "model_source_sha256": digest("models"),
                "preprocessing_source_sha256": digest("datasets")}
        synthetic = {"score_semantics": module.SCORE_SPACE, "probability_semantics": False,
                     "producer": {"checkpoint_sha256": digest("different")}}
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            module.protocol_provenance(real, synthetic)

    def test_matching_library_fingerprints_reconstruct_the_recorded_protocol(self):
        real = {"checkpoint_sha256": digest("checkpoint"), "frozen": True,
                "model_source_sha256": digest("models"),
                "preprocessing_source_sha256": digest("datasets")}
        preprocessing = {
            "image": "PIL_L_float32", "normalize_maxval": 255,
            "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
            "xrv_models_source_sha256": real["model_source_sha256"],
            "xrv_datasets_source_sha256": real["preprocessing_source_sha256"],
        }
        synthetic = {"score_semantics": module.SCORE_SPACE, "probability_semantics": False,
                     "producer": {"checkpoint_sha256": real["checkpoint_sha256"],
                                  "score_space": module.SCORE_SPACE,
                                  "preprocessing_sha256": module.fingerprint(preprocessing),
                                  "finding_mapping_sha256": module.fingerprint(module.XRV_LABELS)}}
        result = module.protocol_provenance(real, synthetic)
        self.assertEqual(result["preprocessing_sha256"], synthetic["producer"]["preprocessing_sha256"])
        self.assertIn("historical_real_adapter_hash_unavailable", result["adapter_protocol_reconstruction"])

    def test_insufficient_binary_test_support_has_no_balanced_accuracy(self):
        validation, test, _ = fixture()
        only_positive = [test[1]]
        result = module.decision_metrics(only_positive, "pneumonia", 0.75, min_per_class=20)
        self.assertIsNone(result["balanced_accuracy"])
        self.assertFalse(result["support_sufficient_for_pilot"])


if __name__ == "__main__":
    unittest.main()

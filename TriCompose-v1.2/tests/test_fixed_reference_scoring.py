import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "real_validation/score_fixed_reference_cohort.py"
spec = importlib.util.spec_from_file_location("score_fixed_reference_cohort", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from audit_reference_coverage import FINDINGS
import calibrate_cached_xrv as calibration


def fixture():
    rows, labels = [], {}
    for split, offset in (("val", 10), ("test", 100)):
        for index, state in enumerate(("positive", "positive", "negative", "negative")):
            subject, study = str(offset + index), str(1000 + offset + index)
            rows.append({"subject_id": subject, "study_id": study, "split": split,
                         "ViewPosition": "AP", "cxr_path": "invented.png", "report_path": "invented.txt"})
            values = dict.fromkeys(FINDINGS, "unknown")
            values["Pneumonia"] = state
            labels[(subject, study)] = values
    previous = {"schema_version": "tricompose-real-xrv-weak-finding-check-v1", "records": []}
    pool = module.patient_pool(rows, labels, set(), seed=1)
    records, _ = module.build_plan(pool, min_per_class=2, max_cases=20)
    cohort = {"schema_version": "tricompose-fixed-real-reference-cohort-v1",
              "primary_metric_eligible": False,
              "selection": {"seed": 1, "min_per_class": 2, "max_cases_per_split": 20,
                            "required_findings": ["Pneumonia"], "label_stratified": True,
                            "one_study_per_patient": True, "patient_disjoint_splits": True,
                            "previous_patient_exclusion_verified": True},
              "records": records}
    return rows, labels, previous, cohort


class FixedReferenceScoringTests(unittest.TestCase):
    def test_exact_fixed_membership_is_reconstructed(self):
        rows, labels, previous, cohort = fixture()
        self.assertEqual(module.validate_cohort_selection(rows, labels, previous, cohort), cohort["records"])

    def test_changed_reference_or_membership_is_rejected(self):
        rows, labels, previous, cohort = fixture()
        changed = copy.deepcopy(cohort)
        changed["records"][0]["reference_states"]["Pneumonia"] = "unknown"
        with self.assertRaisesRegex(ValueError, "references mismatch"):
            module.validate_cohort_selection(rows, labels, previous, changed)
        changed = copy.deepcopy(cohort)
        changed["records"].pop()
        with self.assertRaisesRegex(ValueError, "fixed cohort changed"):
            module.validate_cohort_selection(rows, labels, previous, changed)

    def test_missing_patient_exclusion_contract_is_rejected(self):
        rows, labels, previous, cohort = fixture()
        cohort["selection"]["previous_patient_exclusion_verified"] = False
        with self.assertRaisesRegex(ValueError, "contract missing"):
            module.validate_cohort_selection(rows, labels, previous, cohort)

    def test_mapping_uses_existing_xrv_heads_and_preserves_missing(self):
        mapped = module.mapped_scores({"pneumonia": 0.8, "effusion": 0.6,
                                       "infiltration": 0.4, "lung_opacity": 0.2})
        self.assertEqual(mapped["Pneumonia"], 0.8)
        self.assertEqual(mapped["Pleural Effusion"], 0.6)
        self.assertEqual(mapped["Lung Opacity"], 0.4)
        self.assertIsNone(mapped["Cardiomegaly"])
        self.assertEqual(set(mapped), set(module.MAPPING))

    def test_invalid_classifier_scores_are_rejected(self):
        for invalid in (float("nan"), float("inf"), 1.1, -0.1, True):
            with self.assertRaisesRegex(ValueError, "invalid frozen XRV"):
                module.mapped_scores({"pneumonia": invalid})
        with self.assertRaisesRegex(ValueError, "invalid frozen XRV"):
            module.mapped_scores({"lung_opacity": 0.4, "infiltration": float("nan")})

    def test_cached_calibration_requires_hash_bound_new_cohort(self):
        _, _, _, cohort = fixture()
        cached = {"fixed_reference_cohort_sha256": "invented-hash", "records": cohort["records"]}
        self.assertEqual(calibration.validate_fixed_cohort(cached, cohort, "invented-hash"), {"val": 4, "test": 4})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            calibration.validate_fixed_cohort(cached, cohort, "other-invented-hash")
        changed = copy.deepcopy(cached)
        changed["records"][0]["case_id"] = "different-case"
        with self.assertRaisesRegex(ValueError, "membership or references mismatch"):
            calibration.validate_fixed_cohort(changed, cohort, "invented-hash")

    def test_threshold_report_uses_actual_cohort_size_and_scope(self):
        report = calibration.markdown({}, [], 0, calibration_cases=103, test_cases=105,
                                      label_stratified_cohort=True)
        self.assertIn("103-case validation", report)
        self.assertIn("105-case test", report)
        self.assertIn("label-stratified", report)
        self.assertNotIn("128-case", report)

    def test_recorded_real_protocol_is_validated_not_reconstructed(self):
        producer = {"frozen": True, "checkpoint_sha256": "1" * 64,
                    "model_source_sha256": "2" * 64, "preprocessing_source_sha256": "3" * 64,
                    "adapter_source_sha256": "4" * 64, "score_space": calibration.SCORE_SPACE}
        preprocessing = {"image": "PIL_L_float32", "normalize_maxval": 255,
                         "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
                         "xrv_models_source_sha256": producer["model_source_sha256"],
                         "xrv_datasets_source_sha256": producer["preprocessing_source_sha256"]}
        producer["preprocessing_sha256"] = calibration.fingerprint(preprocessing)
        producer["finding_mapping_sha256"] = calibration.fingerprint(calibration.XRV_LABELS)
        synthetic = {"score_semantics": calibration.SCORE_SPACE, "probability_semantics": False,
                     "producer": copy.deepcopy(producer)}
        result = calibration.protocol_provenance(producer, synthetic)
        self.assertEqual(result["adapter_protocol_reconstruction"], "recorded_adapter_and_protocol_fingerprints")
        changed = copy.deepcopy(producer)
        changed["preprocessing_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recorded real scorer protocol mismatch"):
            calibration.protocol_provenance(changed, synthetic)

    def test_slurm_guard_precedes_real_paths_and_torch(self):
        with patch.dict(module.os.environ, {}, clear=True), patch.object(module.Path, "resolve") as resolve:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                module.run(None)
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()

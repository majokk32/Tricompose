import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "benchmarks/annotate_scoring_scope.py"
spec = importlib.util.spec_from_file_location("annotate_scoring_scope", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    counts = {source: {"positive": 20, "negative": 20, "auroc": 0.6} for source in module.MAPPING}
    image = {"reference_kind": "report_extracted_weak_labels", "primary_metric_eligible": False,
             "label_stratified_cohort": True,
             "summary": {split: {"cases": 40, "findings": copy.deepcopy(counts)} for split in ("val", "test")}}
    threshold = {"primary_metric_eligible": False, "calibration_cases": 40, "test_cases": 40,
                 "findings": {name: {"enabled": True, "threshold": 0.6,
                                     "calibration_positive_count": 20, "calibration_negative_count": 20,
                                     "test_default_0_5": {"positive_reference_count": 20,
                                                          "negative_reference_count": 20, "balanced_accuracy": 0.7},
                                     "test_fitted": {"balanced_accuracy": 0.6, "sensitivity": 0.5, "specificity": 0.7}}
                              for name in module.MAPPING.values()}}
    return image, threshold


class ScoringScopeTests(unittest.TestCase):
    def test_annotation_keeps_scores_and_winner_flags_unchanged(self):
        row = {"case_id": "invented_case", "selected": "True", "score": "0.5", "missing": ""}
        annotated = module.annotate_rows([row])[0]
        self.assertEqual({key: annotated[key] for key in row}, row)
        self.assertFalse(annotated["primary_metric_eligible"])
        self.assertFalse(annotated["targeted_repair_approved"])

    def test_weak_test_performance_does_not_fit_a_mask_or_change_thresholds(self):
        image, threshold = fixture()
        profile = module.validation_profile(image, threshold)
        self.assertFalse(profile["reliability_mask_fitted_from_test"])
        self.assertTrue(profile["findings"]["pneumonia"]["threshold_fit_enabled"])
        self.assertEqual(profile["findings"]["pneumonia"]["threshold"], 0.6)
        self.assertEqual(profile["findings"]["pneumonia"]["test_fitted_balanced_accuracy"], 0.6)

    def test_conflicting_counts_are_rejected(self):
        image, threshold = fixture()
        image["summary"]["test"]["findings"]["Pneumonia"]["negative"] = 19
        with self.assertRaisesRegex(ValueError, "reference count mismatch"):
            module.validation_profile(image, threshold)

    def test_patient_records_and_forged_primary_claim_are_rejected(self):
        image, threshold = fixture()
        with self.assertRaisesRegex(ValueError, "aggregate summaries"):
            module.validation_profile({**image, "records": []}, threshold)
        with self.assertRaisesRegex(ValueError, "unexpected primary"):
            module.validation_profile(image, {**threshold, "primary_metric_eligible": True})

    def test_existing_scope_fields_are_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "already present"):
            module.annotate_rows([{"primary_metric_eligible": True}])


if __name__ == "__main__":
    unittest.main()

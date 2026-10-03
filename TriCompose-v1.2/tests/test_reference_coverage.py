import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "real_validation/audit_reference_coverage.py"
spec = importlib.util.spec_from_file_location("audit_reference_coverage", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows, labels = [], {}
    for split, offset in (("val", 10), ("test", 100)):
        for index, state in enumerate(("positive", "positive", "negative", "negative", "unknown", "uncertain")):
            subject, study = str(offset + index), str(1000 + offset + index)
            rows.append({"subject_id": subject, "study_id": study, "split": split,
                         "ViewPosition": "AP", "cxr_path": "invented.png", "report_path": "invented.txt"})
            values = dict.fromkeys(module.FINDINGS, "unknown")
            values["Pneumonia"] = state
            labels[(subject, study)] = values
    return rows, labels


class ReferenceCoverageTests(unittest.TestCase):
    def test_deterministic_cohort_and_explicit_binary_quotas(self):
        rows, labels = fixture()
        pool = module.patient_pool(rows, labels, set(), seed=1)
        first = module.build_plan(pool, min_per_class=2, max_cases=20)
        self.assertEqual(first, module.build_plan(list(reversed(pool)), min_per_class=2, max_cases=20))
        records, summary = first
        self.assertEqual(len(records), 8)
        self.assertEqual(summary["status"], "ready_for_image_side_weak_reference_pilot")
        self.assertFalse(summary["primary_metric_eligible"])
        self.assertTrue(summary["label_stratified_cohort"])
        self.assertFalse(summary["natural_prevalence_estimate"])
        self.assertTrue(all(set(row) == {"case_id", "split", "source_row_index", "reference_states"}
                            for row in records))

    def test_unknown_uncertain_never_fill_negative_quota(self):
        rows, labels = fixture()
        for states in labels.values():
            if states["Pneumonia"] == "negative":
                states["Pneumonia"] = "unknown"
        _, summary = module.build_plan(module.patient_pool(rows, labels, set(), seed=1),
                                       min_per_class=2, max_cases=20)
        self.assertEqual(summary["status"], "blocked_insufficient_explicit_pneumonia_reference")
        detail = summary["splits"]["val"]["findings"]["Pneumonia"]
        self.assertEqual(detail["available"]["negative"], 0)
        self.assertEqual(detail["available"]["uncertain"], 1)

    def test_prior_patient_exclusion_also_removes_other_studies(self):
        rows, labels = fixture()
        repeated = {**rows[0], "study_id": "9999"}
        rows.append(repeated)
        previous = {"schema_version": "tricompose-real-xrv-weak-finding-check-v1",
                    "records": [{"source_row_index": 0, "split": "val"}]}
        excluded = module.prior_patients(rows, previous)
        pool = module.patient_pool(rows, labels, excluded, seed=1)
        self.assertNotIn(0, [row["source_row_index"] for row in pool])
        self.assertNotIn(len(rows) - 1, [row["source_row_index"] for row in pool])

    def test_patient_split_overlap_is_rejected_even_on_nonfrontal_rows(self):
        rows, labels = fixture()
        rows.append({**rows[0], "split": "train", "ViewPosition": "LATERAL"})
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            module.patient_pool(rows, labels, set(), seed=1)

    def test_one_study_per_patient_is_chosen_without_label_influence(self):
        rows, labels = fixture()
        rows.append({**rows[0], "study_id": "9999"})
        first = module.patient_pool(rows, labels, set(), seed=1)
        changed = copy.deepcopy(labels)
        for states in changed.values():
            states["Pneumonia"] = "positive"
        second = module.patient_pool(rows, changed, set(), seed=1)
        self.assertEqual([row["source_row_index"] for row in first],
                         [row["source_row_index"] for row in second])
        self.assertEqual(len(first), len(rows) - 1)

    def test_missing_label_index_is_unknown_not_negative(self):
        rows, _ = fixture()
        pool = module.patient_pool(rows, {}, set(), seed=1)
        self.assertTrue(all(set(row["reference_states"].values()) == {"unknown"} for row in pool))

    def test_case_cap_cannot_claim_quota_success(self):
        rows, labels = fixture()
        records, summary = module.build_plan(module.patient_pool(rows, labels, set(), seed=1),
                                            min_per_class=2, max_cases=2)
        self.assertEqual(len(records), 4)
        self.assertEqual(summary["status"], "blocked_insufficient_explicit_pneumonia_reference")
        self.assertEqual(summary["splits"]["test"]["findings"]["Pneumonia"]["status"],
                         "selection_budget_exhausted")

    def test_prior_index_integrity_is_checked(self):
        rows, _ = fixture()
        previous = {"schema_version": "tricompose-real-xrv-weak-finding-check-v1",
                    "records": [{"source_row_index": True, "split": "val"}]}
        with self.assertRaisesRegex(ValueError, "invalid prior source index"):
            module.prior_patients(rows, previous)

    def test_real_read_guard_precedes_path_resolution(self):
        with patch.dict(module.os.environ, {}, clear=True), patch.object(module.Path, "resolve") as resolve:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                module.run(None)
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()

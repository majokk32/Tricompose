"""Invented states/hashes only; no models, images, or patient data."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"benchmarks"))
from tricompose_v12 import fixed_image_reports as f
from tricompose_v12.invariant_verification import _digest
from test_live_receipts import data, partial
import run_fixed_image_reports as worker


def row(model="cxrmate_single", state="positive", known=False):
    anchor, image, il, report, labels = data(known)
    report["model_id"] = model; report["candidate_id"] = "fixture_report_"+model
    report["artifact"]["sha256"] = _digest([model, state])
    labels["records"][0].update(report_candidate_id=report["candidate_id"], report_sha256=report["artifact"]["sha256"])
    labels["records"][0]["finding_states"]["edema"] = state
    return f.report_row(image, report, partial(anchor, image, il), labels,
        labels_sha256=_digest(labels), checkpoint_sha256=labels["producer"]["checkpoint_sha256"])


def endpoint(rows, values):
    names = {"case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id", "ehr_sha256",
             "ehr_facts_sha256", "cxr_sha256", "report_sha256"}
    return {"records": [{**{k: r[k] for k in names}, "biovil_raw_cosine": v,
        "status": "not_available" if v is None else "computed_secondary_uncalibrated",
        "reason": "full_report_exceeds_text_context_no_truncation" if v is None else None}
        for r,v in zip(rows, values)], "used_for_routing": False, "clinical_truth_available": False}


class FixedReportTests(unittest.TestCase):
    def test_all_three_edges_and_no_ehr_denominator_remains_na(self):
        r = row(); self.assertIsNone(r["raw_edge_readouts"]["ehr_report"]["support_over_known"])
        self.assertEqual(r["raw_edge_readouts"]["cxr_report"]["supported_positive"], 1)
        self.assertFalse(r["clinical_acceptance"])

    def test_explicit_opposition_count_not_named_clinical_error(self):
        r = row(state="negative")
        self.assertEqual(r["proxy_penalty"], 1)
        self.assertIsNone(r["clinical_accuracy"])

    def test_unknown_and_uncertain_are_missing_not_negative(self):
        for state in ("unknown", "uncertain"):
            r = row(state=state)
            self.assertEqual(r["proxy_penalty"], 1)
            self.assertEqual(r["raw_edge_readouts"]["cxr_report"]["proxy_opposition_facts"], 0)
            self.assertEqual(r["missing_positive_image_facts"], 1)

    def test_omission_does_not_improve_penalty_over_explicit_opposition(self):
        rows = [row(state="negative"), row("maira2", "unknown")]
        self.assertEqual(f.freeze_selection(rows)["choices"][0]["selected_triple_id"], rows[0]["triple_candidate_id"])

    def test_positive_support_can_choose_second_expert(self):
        rows = [row(state="unknown"), row("maira2")]
        self.assertEqual(f.freeze_selection(rows)["choices"][0]["selected_triple_id"], rows[1]["triple_candidate_id"])

    def test_tie_prefers_baseline_deterministically(self):
        rows = [row("maira2"), row()]
        selection = f.freeze_selection(rows)
        self.assertEqual(selection, f.freeze_selection(copy.deepcopy(rows)))
        self.assertEqual(selection["choices"][0]["selected_triple_id"], rows[1]["triple_candidate_id"])

    def test_negative_support_not_a_reward(self):
        r = row(); altered = copy.deepcopy(r)
        altered["raw_edge_readouts"]["cxr_report"]["supported_negative"] += 100
        self.assertEqual(f.selection_key(r), f.selection_key(altered))

    def test_biovil_not_an_input_to_selection_key(self):
        r = row(); altered = {**r, "biovil_raw_cosine": 1.0}
        self.assertEqual(f.selection_key(r), f.selection_key(altered))

    def test_both_models_and_no_duplicates_required(self):
        for rows in ([], [row()], [row(), row()]):
            with self.assertRaises(ValueError): f.freeze_selection(rows)

    def test_fixed_image_or_ehr_cannot_change(self):
        for key in ("cxr_sha256", "ehr_sha256", "ehr_facts_sha256", "image_receipt_id", "case_id"):
            rows = [row(), row("maira2")]; rows[1][key] = "changed"
            with self.assertRaises(ValueError): f.freeze_selection(rows)

    def test_secondary_is_paired_and_selection_cannot_be_changed(self):
        rows = [row(state="negative"), row("maira2")]; s = f.freeze_selection(rows)
        report = f.secondary_comparison(s, rows, endpoint(rows, [.2, .3]))
        self.assertAlmostEqual(report["mean_case_delta"], .1)
        self.assertFalse(report["significance_test_performed"])
        s["choices"][0]["selected_triple_id"] = rows[0]["triple_candidate_id"]
        with self.assertRaises(ValueError): f.secondary_comparison(s, rows, endpoint(rows, [.2, .3]))

    def test_missing_secondary_is_not_zero_or_partial_cohort_mean(self):
        rows = [row(state="negative"), row("maira2")]
        r = f.secondary_comparison(f.freeze_selection(rows), rows, endpoint(rows, [.2, None]))
        self.assertIsNone(r["mean_case_delta"])
        self.assertEqual(r["case_pairs"][0]["available_image_pairs"], 0)

    def test_complete_endpoint_lineage_and_finite_values_required(self):
        rows = [row(), row("maira2")]; s = f.freeze_selection(rows)
        for field, value in (("cxr_sha256", "wrong"), ("biovil_raw_cosine", float("nan")), ("biovil_raw_cosine", True)):
            e = endpoint(rows, [.2,.3]); e["records"][0][field] = value
            with self.assertRaises(ValueError): f.secondary_comparison(s, rows, e)

    def test_endpoint_cannot_route_or_claim_clinical_truth(self):
        rows = [row(), row("maira2")]
        for field in ("used_for_routing", "clinical_truth_available"):
            e = endpoint(rows, [.2,.3]); e[field] = True
            with self.assertRaises(ValueError): f.secondary_comparison(f.freeze_selection(rows), rows, e)

    def test_csv_na_visible_not_zero(self):
        rows = [row(), row("maira2")]
        self.assertIn("NA", worker.csv_table(rows, endpoint(rows, [.2,None])))

    def test_gpu_guard_before_plan_and_directory_access(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "invented_cpu"}, clear=True), \
                patch.object(worker, "load") as load, patch.object(worker, "private_directory") as mkdir:
            with self.assertRaises(RuntimeError): worker.run(Mock())
            load.assert_not_called(); mkdir.assert_not_called()


if __name__ == "__main__": unittest.main()

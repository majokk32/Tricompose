"""Authored metadata and numeric scores; no models, bodies or image pixels."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_guarded_report_secondary as worker
from test_fresh_output_acceptance import fixture
from tricompose_v12.invariant_verification import _digest


def data(*, state="positive", known=True):
    base, _ = fixture(report_state="unknown", known=known, seed=1)
    alt, _ = fixture(report_state=state, report_id="invented_secondary_alternative",
        model="maira2", known=known, seed=1)
    comparison = worker.source.compare(base, alt)
    pair = {"case_id": base["case_id"], "baseline_model": base["report_model_id"],
        "requested_model": alt["report_model_id"], "baseline_candidate_id": base["triple_candidate_id"],
        "proposed_candidate_id": alt["triple_candidate_id"], "comparison": comparison,
        "baseline_scores": base["raw_edge_readouts"], "proposed_scores": alt["raw_edge_readouts"],
        "branch_selected_candidate_id": (alt if comparison["exploratory_gate_pass"] else base)["triple_candidate_id"],
        "clinical_acceptance": False, "clinical_repair_success": False, "original_winner_changed": False,
        "current_image_evidence_cited": False}
    return [base, alt], [pair]


def endpoint(rows, values=(.6, .2)):
    records = []
    for row, value in zip(rows, values, strict=True):
        records.append({**{k: row[k] for k in worker.previous.PAIR_FIELDS},
            "biovil_raw_cosine": value, "calibrated": False,
            "status": "not_available" if value is None else "computed_secondary_uncalibrated",
            "reason": "full_report_exceeds_text_context_no_truncation" if value is None else None})
    return {"records": records, "used_for_routing": False,
        "clinical_truth_available": False, "historical_pool_is_untouched_test": False}


class GuardedReportSecondaryTests(unittest.TestCase):
    def controls(self, **kwargs):
        rows, pairs = data(**kwargs)
        return rows, worker.freeze_controls(rows, pairs, expected_cases=1)

    def test_actual_choice_is_sealed_without_clinical_claim(self):
        rows, controls = self.controls()
        self.assertEqual(controls["choices"][0]["selected_candidate_id"], rows[1]["triple_candidate_id"])
        self.assertFalse(controls["endpoint_used_for_selection"])
        self.assertFalse(controls["original_selection_changed"])

    def test_rejected_alternative_can_score_higher_without_replacing_baseline(self):
        rows, controls = self.controls(state="unknown")
        before = deepcopy(controls)
        _, pairs, summary = worker.endpoint_tables(rows, controls, endpoint(rows, (.2, .9)))
        self.assertEqual(controls, before)
        self.assertFalse(pairs[0]["branch_accepted"])
        self.assertAlmostEqual(pairs[0]["requested_minus_baseline"], .7)
        self.assertEqual(pairs[0]["selected_minus_baseline"], 0)
        self.assertFalse(summary["llm_superiority_demonstrated"])

    def test_accepted_proxy_change_can_have_worse_secondary_score(self):
        rows, controls = self.controls()
        _, pairs, summary = worker.endpoint_tables(rows, controls, endpoint(rows))
        self.assertAlmostEqual(pairs[0]["selected_minus_baseline"], -.4)
        self.assertIsNone(summary["clinical_accuracy"])
        self.assertFalse(summary["clinical_repair_success"])

    def test_unavailable_score_stays_in_denominator_and_delta_is_na(self):
        rows, controls = self.controls()
        table, pairs, summary = worker.endpoint_tables(rows, controls, endpoint(rows, (.6, None)))
        self.assertEqual(summary["requested_report_pairs"], 2)
        self.assertEqual(summary["available_pairs"], 1)
        self.assertIsNone(pairs[0]["requested_minus_baseline"])
        self.assertIsNone(table[1]["biovil_raw_cosine"])
        self.assertIn("NA", worker.previous.csv_text(table))

    def test_changed_gate_selection_or_comparison_is_rejected(self):
        for field in ("branch_selected_candidate_id", "comparison", "baseline_scores"):
            rows, pairs = data()
            pairs[0][field] = "invented_tamper"
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.freeze_controls(rows, pairs, expected_cases=1)

    def test_missing_duplicate_or_wrong_cohort_is_rejected(self):
        for mode in ("missing", "duplicate", "wrong_default", "empty"):
            rows, pairs = data()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                if mode == "missing": worker.freeze_controls(rows[:1], pairs, expected_cases=1)
                elif mode == "duplicate": worker.freeze_controls([rows[0], rows[0]], pairs, expected_cases=1)
                elif mode == "empty": worker.freeze_controls([], [], expected_cases=0)
                else: worker.freeze_controls(rows, pairs)

    def test_new_image_ehr_seed_or_expert_cannot_enter_same_image_comparison(self):
        for field in ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "cxr_candidate_id", "seed", "report_model_id"):
            rows, pairs = data()
            rows[1][field] = 2 if field == "seed" else _digest("invented_change")
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.freeze_controls(rows, pairs, expected_cases=1)

    def test_secondary_endpoint_cannot_be_routing_or_truth(self):
        for field in ("used_for_routing", "clinical_truth_available", "historical_pool_is_untouched_test"):
            rows, controls = self.controls(); ep = endpoint(rows); ep[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.endpoint_tables(rows, controls, ep)

    def test_endpoint_missing_duplicate_or_changed_report_is_rejected(self):
        for mode in ("missing", "duplicate", "hash"):
            rows, controls = self.controls(); ep = endpoint(rows)
            if mode == "missing": ep["records"].pop()
            elif mode == "duplicate": ep["records"][1] = deepcopy(ep["records"][0])
            else: ep["records"][0]["report_sha256"] = _digest("invented_changed_report")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                worker.endpoint_tables(rows, controls, ep)

    def test_invalid_or_silently_missing_scores_are_rejected(self):
        for value in (float("nan"), 2.0, None):
            rows, controls = self.controls(); ep = endpoint(rows)
            ep["records"][0]["biovil_raw_cosine"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.endpoint_tables(rows, controls, ep)

    def test_rows_mutated_after_freeze_are_rejected(self):
        rows, controls = self.controls(); rows[0]["structure"]["generic_report"] = True
        with self.assertRaisesRegex(ValueError, "presealed"):
            worker.endpoint_tables(rows, controls, endpoint(rows))

    def test_unknown_ehr_is_not_negative_and_no_model_calls_are_fabricated(self):
        rows, pairs = data(known=False); before = deepcopy(rows)
        controls = worker.freeze_controls(rows, pairs, expected_cases=1)
        _, _, summary = worker.endpoint_tables(rows, controls, endpoint(rows))
        self.assertEqual(rows, before)
        self.assertIsNone(rows[0]["raw_edge_readouts"]["ehr_report"]["support_over_known"])
        for key in ("new_generator_calls", "new_primary_scoring_calls", "new_planner_calls"):
            self.assertEqual(summary[key], 0)

    def test_clinical_acceptance_or_winner_promotion_is_rejected(self):
        for flag in ("original_winner_changed", "clinical_acceptance", "clinical_repair_success"):
            rows, pairs = data(); pairs[0][flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                worker.freeze_controls(rows, pairs, expected_cases=1)

    def test_gpu_guard_precedes_model_and_inputs(self):
        with patch.object(worker, "gpu_guard", side_effect=RuntimeError("invented_cpu")), \
             patch.object(worker, "read_json") as read, patch.object(worker.previous, "score") as scorer:
            with self.assertRaises(RuntimeError): worker.run(SimpleNamespace())
            read.assert_not_called(); scorer.assert_not_called()

    def test_cpu_guard_precedes_prepare_reads(self):
        with patch.object(worker.source.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(worker.source, "Reader") as read:
            with self.assertRaises(RuntimeError): worker.prepare(SimpleNamespace())
            read.assert_not_called()


if __name__ == "__main__": unittest.main()

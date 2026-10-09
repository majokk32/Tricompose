"""Authored receipts and numeric endpoints only; no model/body/pixel reads."""
import copy
import csv
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import run_fresh_report_secondary as secondary
from test_fresh_report_agent_bridge import base_ledger, append_report, decision
from test_fresh_output_acceptance import fixture
from tricompose_llm.live_report_bridge import FreshReportSession
from tricompose_v12.invariant_verification import _digest


def data(known=True):
    base, ctx = fixture(known=known)
    ledger = base_ledger(base, ctx)
    session = FreshReportSession(base, ctx, ledger.snapshot(), sink=lambda e: None)
    rows = [base]
    for i, model in enumerate(("maira2", "llavarad", "chexagent2")):
        state = session.begin_planning(); session.receive_decision(decision(state))
        row, _ = fixture(known=known, report_state="positive" if i == 0 else "unknown",
            report_id=f"fixture_endpoint_report_{i}", model=model)
        book = append_report(ledger, row, suffix=str(i)); session.finish_report(row, book)
        rows.append(row)
    session.begin_planning()
    return rows, [session.result()], {base["case_id"]: ctx}, {base["case_id"]: session.ledger}


def endpoint(rows, *, missing=None):
    records = []
    for i, row in enumerate(rows):
        value = None if i == missing else (.6, .2, .8, .9)[i]
        records.append({**{k: row[k] for k in secondary.PAIR_FIELDS}, "biovil_raw_cosine": value,
            "status": "not_available" if value is None else "computed_secondary_uncalibrated",
            "reason": "full_report_exceeds_text_context_no_truncation" if value is None else None,
            "calibrated": False})
    return {"records": records, "used_for_routing": False, "clinical_truth_available": False,
        "historical_pool_is_untouched_test": False}


class FreshReportSecondaryTests(unittest.TestCase):
    def controls(self):
        rows, selections, contexts, books = data()
        return rows, secondary.freeze_controls(rows, selections, contexts, books, expected_cases=1)

    def test_existing_first_preserving_static_control_and_sealed_llm_choice(self):
        rows, controls = self.controls()
        c = controls["choices"][0]
        self.assertEqual(c["fixed_candidate_id"], rows[0]["triple_candidate_id"])
        self.assertEqual(c["static_candidate_id"], rows[1]["triple_candidate_id"])
        self.assertEqual(c["llm_candidate_id"], rows[1]["triple_candidate_id"])
        self.assertEqual(c["source_worker_attempts"], 10)
        self.assertEqual(c["source_policy_requests"], 3)
        self.assertFalse(controls["endpoint_used_for_selection"])
        self.assertIn("not_cost_matched", controls["static_scope"])

    def test_independent_score_can_be_worse_without_changing_choice(self):
        rows, controls = self.controls(); before = copy.deepcopy(controls)
        table, pairs, summary = secondary.endpoint_tables(rows, controls, endpoint(rows))
        self.assertEqual(controls, before)
        self.assertAlmostEqual(pairs[0]["llm_minus_fixed"], -.4)
        self.assertEqual(pairs[0]["llm_report_model"], "maira2")
        self.assertEqual(len(table), 4)
        self.assertIsNone(summary["clinical_accuracy"])
        self.assertFalse(summary["clinical_repair_success"])

    def test_secondary_highest_score_is_not_an_oracle_selection(self):
        rows, controls = self.controls()
        table, pairs, summary = secondary.endpoint_tables(rows, controls, endpoint(rows))
        self.assertEqual(max(table, key=lambda r: r["biovil_raw_cosine"])["report_model"], "chexagent2")
        self.assertEqual(pairs[0]["static_report_model"], "maira2")
        self.assertEqual(summary["llm_static_same_candidate_count"], 1)

    def test_unavailable_full_report_kept_in_denominator_and_delta_na(self):
        rows, controls = self.controls()
        table, pairs, summary = secondary.endpoint_tables(rows, controls, endpoint(rows, missing=1))
        self.assertIsNone(pairs[0]["llm_raw_cosine"])
        self.assertIsNone(pairs[0]["llm_minus_fixed"])
        self.assertEqual(summary["requested_report_pairs"], 4)
        self.assertEqual(summary["available_pairs"], 3)
        csv_rows = list(csv.DictReader(io.StringIO(secondary.csv_text(table))))
        self.assertEqual(csv_rows[1]["biovil_raw_cosine"], "NA")

    def test_missing_and_duplicate_experts_refused(self):
        for change in ("missing", "duplicate"):
            rows, sels, ctxs, books = data()
            rows = rows[:-1] if change == "missing" else rows[:-1] + [rows[1]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, sels, ctxs, books, expected_cases=1)

    def test_wrong_fixed_baseline_and_unseen_llm_choice_refused(self):
        for field in ("baseline_candidate_id", "selected_candidate_id"):
            rows, sels, ctxs, books = data()
            sels[0][field] = "fixture_unobserved_candidate"
            with self.subTest(field=field), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, sels, ctxs, books, expected_cases=1)

    def test_new_ehr_or_image_receipt_refused(self):
        for field in ("ehr_sha256", "cxr_sha256"):
            rows, sels, ctxs, books = data()
            rows[1][field] = _digest("invented_replacement")
            with self.subTest(field=field), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, sels, ctxs, books, expected_cases=1)

    def test_no_completed_ledger_receipt_is_not_a_free_candidate(self):
        rows, sels, ctxs, books = data()
        base, _ = fixture(); book = base_ledger(base, next(iter(ctxs.values()))).snapshot()
        books[base["case_id"]] = book
        with self.assertRaises(ValueError):
            secondary.freeze_controls(rows, sels, ctxs, books, expected_cases=1)

    def test_wrong_endpoint_scope_or_hidden_missing_row_refused(self):
        for change in ("routing", "missing", "wrong_hash", "truth"):
            rows, controls = self.controls(); ep = endpoint(rows)
            if change == "routing": ep["used_for_routing"] = True
            elif change == "truth": ep["clinical_truth_available"] = True
            elif change == "missing": ep["records"].pop()
            else: ep["records"][0]["report_sha256"] = _digest("invented_wrong_report")
            with self.subTest(change=change), self.assertRaises(ValueError):
                secondary.endpoint_tables(rows, controls, ep)

    def test_changed_rows_after_sealing_refused(self):
        rows, controls = self.controls()
        rows[0]["structure"]["generic_report"] = True
        with self.assertRaisesRegex(ValueError, "presealed"):
            secondary.endpoint_tables(rows, controls, endpoint(rows))

    def test_nonfinite_or_unnamed_missing_score_refused(self):
        for value, reason in ((float("nan"), None), (2.0, None), (None, None)):
            rows, controls = self.controls(); ep = endpoint(rows)
            ep["records"][0].update(biovil_raw_cosine=value, reason=reason,
                status="not_available" if value is None else "computed_secondary_uncalibrated")
            with self.subTest(value=value), self.assertRaises(ValueError):
                secondary.endpoint_tables(rows, controls, ep)

    def test_unknown_ehr_not_reinterpreted_as_negative(self):
        rows, sels, ctxs, books = data(known=False)
        before = copy.deepcopy(rows)
        secondary.freeze_controls(rows, sels, ctxs, books, expected_cases=1)
        self.assertEqual(rows, before)
        self.assertIsNone(rows[0]["raw_edge_readouts"]["ehr_report"]["support_over_known"])

    def test_default_cohort_is_exact_two_cases(self):
        with self.assertRaisesRegex(ValueError, "four_expert_grid"):
            secondary.freeze_controls(*data())

    def test_gpu_guard_precedes_inputs_and_model(self):
        with patch.object(secondary, "gpu_guard", side_effect=RuntimeError("invented_cpu")), \
             patch.object(secondary, "read_json") as read, patch.object(secondary, "score") as model:
            with self.assertRaises(RuntimeError): secondary.run(object())
            read.assert_not_called(); model.assert_not_called()

    def test_cpu_preflight_guard_precedes_inputs(self):
        with patch.object(secondary.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(secondary.postflight, "MetadataReader") as read:
            with self.assertRaises(RuntimeError): secondary.prepare(object())
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

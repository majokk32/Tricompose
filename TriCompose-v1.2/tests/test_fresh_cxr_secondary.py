"""Authored synthetic numeric fixtures only; no models or clinical bodies."""
import copy
import csv
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_fresh_cxr_secondary as secondary
from test_fresh_report_secondary import data
from tricompose_v12.invariant_verification import _digest


def package(with_probes=True):
    source, _, _, _ = data()
    cached, cases, choices, selections, books, fresh = [], [], [], [], [], []
    for ordinal in range(2):
        cid = f"fixture_case_{ordinal}"
        group = copy.deepcopy(source)
        for i, row in enumerate(group):
            row.update(case_id=cid, triple_candidate_id=f"fixture_triple_{ordinal}_{i}",
                cxr_candidate_id=f"fixture_image_{ordinal}", report_candidate_id=f"fixture_report_{ordinal}_{i}",
                ehr_sha256=_digest(["invented_ehr", ordinal]), ehr_facts_sha256=_digest(["invented_facts", ordinal]))
        cached.extend(group)
        cases.append({"case_id": cid, "cached_rows": group, "reference": group[1]})
        choices.append({"case_id": cid, "fixed_candidate_id": group[0]["triple_candidate_id"],
            "static_candidate_id": group[1]["triple_candidate_id"], "llm_candidate_id": group[1]["triple_candidate_id"]})
        selections.append({"case_id": cid, "selected_candidate_id": group[1]["triple_candidate_id"],
            "clinical_repair_success": False, "charged_new_worker_attempts": 4 if with_probes else 0,
            "charged_policy_requests": 1, "accepted_proxy_transitions": 0})
        books.append({"case_id": cid, "charged_model_attempts": 4 if with_probes else 0, "pending_attempts": 0})
        if with_probes:
            row = copy.deepcopy(group[1])
            row.update(triple_candidate_id=f"fixture_probe_{ordinal}", cxr_candidate_id=f"fixture_new_image_{ordinal}",
                report_candidate_id=f"fixture_new_report_{ordinal}", cxr_model_id="roentgen_v2", seed=1,
                cxr_sha256=_digest(["invented_new_image", ordinal]), report_sha256=_digest(["invented_new_report", ordinal]))
            fresh.append(row)
    return cached + fresh, {"cases": cases, "cached_controls": {"choices": choices}}, selections, books


def endpoint(rows, missing=None):
    return {"records": [{**{k: r[k] for k in secondary.previous.PAIR_FIELDS},
        "biovil_raw_cosine": None if i == missing else .05 * i,
        "status": "not_available" if i == missing else "computed_secondary_uncalibrated",
        "reason": "full_report_exceeds_text_context_no_truncation" if i == missing else None,
        "calibrated": False} for i, r in enumerate(rows)],
        "used_for_routing": False, "clinical_truth_available": False, "historical_pool_is_untouched_test": False}


class FreshCXRSecondaryTests(unittest.TestCase):
    def test_freeze_keeps_rejected_probe_separate_from_selected(self):
        payload = package(); before = copy.deepcopy(payload)
        controls = secondary.freeze_controls(*payload)
        self.assertEqual(payload, before)
        for c in controls["choices"]:
            self.assertEqual(c["selected_candidate_id"], c["before_candidate_id"])
            self.assertNotEqual(c["probe_candidate_id"], c["selected_candidate_id"])

    def test_high_cosine_does_not_change_selection_or_claim_repair(self):
        rows, plan, sels, books = package()
        controls = secondary.freeze_controls(rows, plan, sels, books); before = copy.deepcopy(controls)
        table, pairs, summary = secondary.endpoint_tables(rows, controls, endpoint(rows))
        self.assertEqual(controls, before)
        self.assertEqual(len(table), 10)
        self.assertEqual(summary["new_probe_pairs"], 2)
        self.assertEqual(summary["accepted_proxy_transitions"], 0)
        self.assertIsNone(summary["clinical_accuracy"])
        self.assertFalse(summary["clinical_repair_success"])
        self.assertFalse(summary["training_performed"])
        self.assertTrue(all(p["selected_minus_before"] == 0 and p["probe_minus_before"] > 0 for p in pairs))

    def test_missing_probe_score_is_explicit_na_kept_in_denominator(self):
        rows, plan, sels, books = package()
        controls = secondary.freeze_controls(rows, plan, sels, books)
        table, pairs, summary = secondary.endpoint_tables(rows, controls, endpoint(rows, missing=8))
        self.assertEqual(summary["requested_report_pairs"], 10)
        self.assertEqual(summary["available_pairs"], 9)
        self.assertIsNone(pairs[0]["probe_minus_before"])
        csv_rows = list(csv.DictReader(io.StringIO(secondary.previous.csv_text(table))))
        self.assertEqual(csv_rows[8]["biovil_raw_cosine"], "NA")

    def test_no_completed_probe_is_not_invented_or_zero_filled(self):
        rows, plan, sels, books = package(False)
        controls = secondary.freeze_controls(rows, plan, sels, books)
        _, pairs, summary = secondary.endpoint_tables(rows, controls, endpoint(rows))
        self.assertEqual(summary["requested_report_pairs"], 8)
        self.assertTrue(all(p["probe_raw_cosine"] is None for p in pairs))
        self.assertTrue(all(p["probe_availability_reason"] == "no_completed_candidate" for p in pairs))

    def test_abstained_selected_candidate_is_na(self):
        rows, plan, sels, books = package()
        sels[0]["selected_candidate_id"] = None
        controls = secondary.freeze_controls(rows, plan, sels, books)
        _, pairs, _ = secondary.endpoint_tables(rows, controls, endpoint(rows))
        self.assertIsNone(pairs[0]["selected_raw_cosine"])
        self.assertIsNone(pairs[0]["selected_minus_before"])

    def test_changed_reference_or_unobserved_choice_refused(self):
        for corruption in ("reference", "unobserved", "cross_case"):
            rows, plan, sels, books = package()
            if corruption == "reference": plan["cached_controls"]["choices"][0]["llm_candidate_id"] = rows[0]["triple_candidate_id"]
            elif corruption == "unobserved": sels[0]["selected_candidate_id"] = "fixture_unobserved"
            else: sels[0]["selected_candidate_id"] = rows[4]["triple_candidate_id"]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, plan, sels, books)

    def test_duplicate_or_missing_cached_inventory_refused(self):
        for corruption in ("duplicate", "missing", "changed"):
            rows, plan, sels, books = package()
            if corruption == "duplicate": rows[-1] = rows[-2]
            elif corruption == "missing": rows.pop(0)
            else: rows[0] = {**rows[0], "report_sha256": _digest("invented_changed_report")}
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, plan, sels, books)

    def test_ehr_model_and_seed_changes_refused(self):
        for key, value in (("ehr_sha256", _digest("invented_other_ehr")),
            ("ehr_facts_sha256", _digest("invented_other_facts")), ("report_model_id", "chexagent2"),
            ("cxr_model_id", "chexgenbench_sana"), ("seed", 0)):
            rows, plan, sels, books = package(); rows[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, plan, sels, books)

    def test_pending_or_unaccounted_worker_cost_refused(self):
        for key in ("pending_attempts", "charged_model_attempts"):
            rows, plan, sels, books = package(); books[0][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                secondary.freeze_controls(rows, plan, sels, books)

    def test_endpoint_routing_claim_or_hidden_missing_row_refused(self):
        rows, plan, sels, books = package(); controls = secondary.freeze_controls(rows, plan, sels, books)
        for corruption in ("routing", "missing", "truth", "wrong_hash"):
            ep = endpoint(rows)
            if corruption == "routing": ep["used_for_routing"] = True
            elif corruption == "truth": ep["clinical_truth_available"] = True
            elif corruption == "missing": ep["records"].pop()
            else: ep["records"][0]["report_sha256"] = _digest("invented_wrong_hash")
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                secondary.endpoint_tables(rows, controls, ep)

    def test_invalid_cosine_or_unnamed_na_refused(self):
        rows, plan, sels, books = package(); controls = secondary.freeze_controls(rows, plan, sels, books)
        for value in (float("nan"), 2.0, None):
            ep = endpoint(rows); ep["records"][0]["biovil_raw_cosine"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                secondary.endpoint_tables(rows, controls, ep)

    def test_changed_rows_after_sealing_refused(self):
        rows, plan, sels, books = package(); controls = secondary.freeze_controls(rows, plan, sels, books)
        rows[0]["structure"]["generic_report"] = True
        with self.assertRaisesRegex(ValueError, "presealed"):
            secondary.endpoint_tables(rows, controls, endpoint(rows))

    def test_gpu_guard_precedes_inputs_and_model(self):
        with patch.object(secondary, "gpu_guard", side_effect=RuntimeError("invented_cpu")), \
             patch.object(secondary, "read_json") as read, patch.object(secondary.previous, "score") as model:
            with self.assertRaises(RuntimeError): secondary.run(object())
            read.assert_not_called(); model.assert_not_called()

    def test_cpu_guard_precedes_inputs(self):
        with patch.object(secondary.previous.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(secondary.previous.postflight, "MetadataReader") as read:
            with self.assertRaises(RuntimeError): secondary.prepare(object())
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

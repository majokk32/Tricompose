"""Invented categorical fixtures only; no clinical gold or model inference."""
import copy
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12.decision_preview import Budget, budget_status, preview_actions
from tricompose_v12.report_scope_table import FINDINGS, relation
import preview_candidate_actions as cli


def fixture(*, ehr="positive", image="positive", report="negative", other="unknown"):
    rows = []
    for finding in FINDINGS:
        e, i, r = (ehr, image, report) if finding == "cardiomegaly" else (other, other, other)
        supported = finding in ("cardiomegaly", "consolidation", "pleural_effusion", "pneumothorax")
        scoped = r if supported else "unknown"
        decision = ("no_model_assertion" if r == "unknown" else "scope_commit") if supported else "outside_scope_inventory"
        available = supported
        raw = {"ehr_cxr": relation(e, i), "ehr_report": relation(e, r), "cxr_report": relation(i, r)}
        gated = {"ehr_cxr": raw["ehr_cxr"], "ehr_report": relation(e, scoped, available=available),
                 "cxr_report": relation(i, scoped, available=available)}
        rows.append({"evidence_id": "fixture_" + finding, "case_id": "fixture_case",
            "triple_candidate_id": "fixture_candidate", "cxr_candidate_id": "fixture_image",
            "report_candidate_id": "fixture_report", "finding": finding,
            "artifact_hashes": {"ehr_sha256": "e" * 64, "ehr_facts_sha256": "f" * 64,
                                "cxr_sha256": "c" * 64, "report_sha256": "d" * 64},
            "states": {"ehr": e, "xrv": i, "chexbert": r, "qwen_image": "unknown", "qwen_report": "unknown"},
            "ehr_provenance": {"weak_clinical_context_promoted": False,
                "evidence_ids": [] if e == "unknown" else ["fixture_direct_diagnosis"],
                "source_fields": [] if e == "unknown" else ["diagnoses[0]"]},
            "report_scope_state": scoped, "report_scope_decision": decision,
            "report_scope_supported_finding": supported, "report_scope_available": available,
            "image_dependency_group": "c" * 64,
            "relations": {"raw": raw, "scoped": gated},
            "clinical_error_confirmed": False, "confirmed_faulty_modality": None,
            "automatic_repair_eligible": False})
    return rows


def update_fact(row, *, ehr=None, image=None, report=None, scope_available=None):
    if ehr is not None:
        row["states"]["ehr"] = ehr
        row["ehr_provenance"].update(evidence_ids=[] if ehr == "unknown" else ["fixture_direct"],
                                     source_fields=[] if ehr == "unknown" else ["diagnoses[0]"])
    if image is not None: row["states"]["xrv"] = image
    if report is not None:
        row["states"]["chexbert"] = report
        if row["report_scope_supported_finding"]:
            row["report_scope_state"] = report
            row["report_scope_decision"] = "no_model_assertion" if report == "unknown" else "scope_commit"
    if scope_available is False:
        row.update(report_scope_available=False, report_scope_state="unknown", report_scope_decision="abstain")
    s, available = row["states"], row["report_scope_available"]
    row["relations"] = {"raw": {"ehr_cxr": relation(s["ehr"], s["xrv"]),
        "ehr_report": relation(s["ehr"], s["chexbert"]), "cxr_report": relation(s["xrv"], s["chexbert"])},
        "scoped": {"ehr_cxr": relation(s["ehr"], s["xrv"]),
        "ehr_report": relation(s["ehr"], row["report_scope_state"], available=available),
        "cxr_report": relation(s["xrv"], row["report_scope_state"], available=available)}}


class DecisionPreviewTests(unittest.TestCase):
    def preview(self, rows=None, **kwargs):
        return preview_actions(rows or fixture(), Budget(4, 120), **kwargs)[0]

    def test_report_signal_is_only_a_verification_hint(self):
        row = self.preview()
        self.assertEqual(row["suggested_verification_target"], "report")
        self.assertEqual(row["next_action"], "verify_more")
        self.assertEqual(row["trigger_evidence_ids"], ["fixture_cardiomegaly"])
        self.assertIsNone(row["confirmed_faulty_modality"])
        self.assertFalse(row["targeted_repair_approved"])

    def test_cxr_signal_does_not_treat_reports_as_independent_image_truth(self):
        row = self.preview(fixture(image="negative", report="positive"))
        self.assertEqual(row["suggested_verification_target"], "cxr")
        self.assertIn("cxr_conditioned_reports_are_not_independent_image_truth", row["reason_codes"])
        self.assertFalse(row["shared_cxr_report_votes_independent"])

    def test_agreement_is_not_clinical_acceptance(self):
        row = self.preview(fixture(report="positive"))
        self.assertEqual(row["next_action"], "verify_more")
        self.assertFalse(row["clinical_acceptance"])
        self.assertIsNone(row["clinical_selection_score"])

    def test_both_downstream_opposed_has_no_unique_fault_target(self):
        row = self.preview(fixture(image="negative", report="negative"))
        self.assertIsNone(row["suggested_verification_target"])
        self.assertIn("both_downstream_oppose_ehr", row["reason_codes"])

    def test_conflicting_fact_loci_abstain(self):
        rows = fixture()
        update_fact(rows[FINDINGS.index("pleural_effusion")], ehr="positive", image="negative", report="positive")
        row = self.preview(rows)
        self.assertIsNone(row["suggested_verification_target"])
        self.assertIn("conflicting_provisional_loci", row["reason_codes"])

    def test_unknown_ehr_does_not_localize_image_report_opposition(self):
        row = self.preview(fixture(ehr="unknown"))
        self.assertIsNone(row["suggested_verification_target"])
        self.assertIn("no_direct_ehr_fact", row["reason_codes"])
        self.assertEqual(row["scoped_edges"]["ehr_report"].get("opposition", 0), 0)

    def test_all_unknown_inventory_remains_and_is_not_success(self):
        row = self.preview(fixture(ehr="unknown", image="unknown", report="unknown"))
        self.assertEqual(row["finding_inventory"], 8)
        self.assertEqual(row["next_action"], "verify_more")
        self.assertFalse(row["clinical_acceptance"])

    def test_uncertain_ehr_not_a_hard_fact(self):
        row = self.preview(fixture(ehr="uncertain"))
        self.assertIsNone(row["suggested_verification_target"])
        self.assertIn("no_direct_ehr_fact", row["reason_codes"])

    def test_scope_withdrawal_cannot_manufacture_support(self):
        rows = fixture()
        update_fact(rows[FINDINGS.index("cardiomegaly")], scope_available=False)
        row = self.preview(rows)
        self.assertIsNone(row["suggested_verification_target"])
        self.assertIn("direct_ehr_fact_coverage_gap", row["reason_codes"])
        self.assertEqual(row["scoped_edges"]["ehr_report"].get("support", 0), 0)

    def test_pneumonia_without_scope_head_retains_coverage_gap(self):
        rows = fixture(ehr="unknown", image="unknown", report="unknown")
        update_fact(rows[FINDINGS.index("pneumonia")], ehr="positive", image="positive", report="positive")
        row = self.preview(rows)
        self.assertIn("fixture_pneumonia", row["direct_ehr_coverage_gap_evidence_ids"])
        self.assertIsNone(row["suggested_verification_target"])

    def test_qwen_disagreement_is_not_a_tie_breaking_clinical_vote(self):
        rows = fixture()
        rows[FINDINGS.index("cardiomegaly")]["states"]["qwen_image"] = "negative"
        row = self.preview(rows)
        self.assertTrue(row["evaluator_disagreements"][0]["image_evaluators_disagree"])
        self.assertFalse(row["model_execution_allowed"])

    def test_shared_image_report_disagreement_is_correlated_and_deduplicated(self):
        one, two = fixture(), fixture(report="positive")
        for row in two:
            row.update(evidence_id="second_" + row["finding"], triple_candidate_id="second_candidate",
                       report_candidate_id="second_report")
            row["artifact_hashes"]["report_sha256"] = "a" * 64
        out = preview_actions(one + two, Budget(4, 120))
        self.assertEqual(len(out), 2)
        self.assertTrue(all(row["evaluator_disagreements"][0]["shared_image_reports_disagree"] for row in out))
        self.assertTrue(all(not row["shared_cxr_report_votes_independent"] for row in out))

    def test_deterministic_under_fact_order_and_inputs_unchanged(self):
        rows = fixture(); before = copy.deepcopy(rows)
        self.assertEqual(preview_actions(rows, Budget(4, 120)), preview_actions(list(reversed(rows)), Budget(4, 120)))
        self.assertEqual(rows, before)

    def test_broken_scope_edge_arithmetic_rejected(self):
        rows = fixture(); rows[0]["relations"]["scoped"]["cxr_report"] = "support"
        with self.assertRaisesRegex(ValueError, "arithmetic"): self.preview(rows)

    def test_empty_or_missing_findings_rejected(self):
        for rows in ([], fixture()[:-1]):
            with self.assertRaises(ValueError): preview_actions(rows, Budget(4, 120))

    def test_duplicate_finding_or_evidence_rejected(self):
        for field in ("finding", "evidence_id"):
            rows = fixture(); rows[1][field] = rows[0][field]
            with self.assertRaises(ValueError): self.preview(rows)

    def test_ehr_hash_cannot_change(self):
        rows = fixture(); rows[-1]["artifact_hashes"]["ehr_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "fixed EHR"): self.preview(rows)

    def test_weak_context_or_missing_provenance_rejected(self):
        for change in ({"weak_clinical_context_promoted": True}, {"evidence_ids": []}, {"source_fields": []}):
            rows = fixture(); rows[FINDINGS.index("cardiomegaly")]["ehr_provenance"].update(change)
            with self.assertRaises(ValueError): self.preview(rows)

    def test_forged_eligibility_rejected(self):
        for field, value in (("clinical_error_confirmed", True), ("automatic_repair_eligible", True),
                             ("confirmed_faulty_modality", "report")):
            rows = fixture(); rows[0][field] = value
            with self.assertRaisesRegex(ValueError, "eligibility"): self.preview(rows)

    def test_fake_scope_head_or_state_flip_rejected(self):
        rows = fixture(); rows[FINDINGS.index("pneumonia")]["report_scope_supported_finding"] = True
        with self.assertRaises(ValueError): self.preview(rows)
        rows = fixture(); rows[FINDINGS.index("cardiomegaly")]["report_scope_state"] = "positive"
        with self.assertRaisesRegex(ValueError, "flip"): self.preview(rows)

    def test_report_scope_semantics_and_shared_image_consistency_rejected(self):
        rows = fixture(); rows[FINDINGS.index("cardiomegaly")]["report_scope_decision"] = "no_model_assertion"
        with self.assertRaises(ValueError): self.preview(rows)
        rows = fixture(); rows[0]["image_dependency_group"] = "a" * 64
        with self.assertRaises(ValueError): self.preview(rows)

    def test_json_csv_output_has_no_quote_or_raw_patient_fields(self):
        records = preview_actions(fixture(), Budget(4, 120))
        rendered = cli.render_table(records) + json.dumps(records)
        for field in ("patient_id", "subject_id", "report_text", "image_path", "evidence_quote"):
            self.assertNotIn(field, rendered)
        self.assertIn("trigger_evidence_ids", rendered)


class BudgetTests(unittest.TestCase):
    def test_invalid_limits(self):
        for calls, seconds in ((-1, 20), (True, 20), (1.2, 20), (1, -1), (1, True), (1, math.nan), (1, math.inf)):
            with self.assertRaises(ValueError): Budget(calls, seconds)

    def test_zero_call_or_time_budget_rejects_without_execution(self):
        for budget in (Budget(0, 120), Budget(4, 0)):
            row = preview_actions(fixture(), budget)[0]
            self.assertEqual(row["next_action"], "reject")
            self.assertFalse(row["model_execution_allowed"])

    def test_all_attempts_including_failed_timeout_cancelled_charge_calls(self):
        history = [{"call_id": "attempt_" + str(i), "status": status, "gpu_seconds": 2}
                   for i, status in enumerate(("completed", "failed", "timeout", "cancelled"))]
        ledger = budget_status(Budget(4, 120), history)
        self.assertEqual(ledger["attempted_additional_calls"], 4)
        self.assertEqual(ledger["observed_additional_gpu_seconds_lower_bound"], 8)
        self.assertEqual(ledger["verification_affordability"], "budget_exhausted")

    def test_missing_observed_runtime_is_not_zero(self):
        ledger = budget_status(Budget(4, 120), [{"call_id": "attempt", "status": "failed", "gpu_seconds": None}])
        self.assertIsNone(ledger["remaining_additional_gpu_seconds"])
        self.assertFalse(ledger["observed_additional_gpu_seconds_complete"])
        self.assertEqual(ledger["verification_affordability"], "unknown_observed_runtime")

    def test_unknown_estimate_cannot_be_called_free(self):
        ledger = budget_status(Budget(4, 120))
        self.assertIsNone(ledger["verification_gpu_seconds_estimate"])
        self.assertEqual(ledger["verification_affordability"], "verification_cost_estimate_missing")
        self.assertIsNone(ledger["actual_compute_savings"])

    def test_estimated_budget_only_is_not_authorization(self):
        ledger = budget_status(Budget(4, 120), verification_gpu_seconds=20)
        self.assertEqual(ledger["verification_affordability"], "estimated_affordable_not_authorized")
        self.assertFalse(preview_actions(fixture(), Budget(4, 120), verification_gpu_seconds=20)[0]["model_execution_allowed"])

    def test_exact_time_bound_and_time_excess(self):
        history = [{"call_id": "attempt", "status": "completed", "gpu_seconds": 120}]
        self.assertEqual(budget_status(Budget(4, 120), history)["verification_affordability"], "budget_exhausted")
        self.assertEqual(preview_actions(fixture(), Budget(4, 120), verification_gpu_seconds=121)[0]["next_action"], "reject")

    def test_duplicate_or_invalid_call_history_rejected(self):
        call = {"call_id": "attempt", "status": "failed", "gpu_seconds": 2}
        for history in ([call, call], [{**call, "gpu_seconds": -1}], [{**call, "gpu_seconds": math.nan}],
                        [{**call, "status": "not_attempted"}], [{**call, "extra": 1}]):
            with self.assertRaises(ValueError): budget_status(Budget(4, 120), history)

    def test_invalid_estimate_rejected(self):
        for seconds in (-1, math.inf, math.nan, True):
            with self.assertRaises(ValueError): budget_status(Budget(4, 120), verification_gpu_seconds=seconds)


class GuardTests(unittest.TestCase):
    def test_refuse_before_protected_reads_without_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "require_inside") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load_cached_facts("not_opened")
            read.assert_not_called()

    def test_run_guard_before_atomic_output_creation(self):
        args = SimpleNamespace(scope_run="not_opened", output_root="not_created", run_id="fixture",
                               max_additional_calls=4, max_additional_gpu_seconds=120)
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "new_atomic_run") as write:
            with self.assertRaises(RuntimeError): cli.run(args)
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()

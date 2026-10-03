"""Invented caches only; software integrity is not clinical verification."""
import copy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import invariant_verification as interface
from tricompose_v12.automatic_secondary import overlay
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from test_automatic_secondary import fixture
from test_legacy_automatic_replay import invented, policy
import bind_invariant_verification as cli


def config():
    return json.loads((ROOT / "configs/invariant_verification_v1.json").read_text())


def data():
    rows, request, scores = fixture()
    source, details = invented()
    return overlay(rows, request, scores), make_legacy_bank(source, details, policy())


def pair(case="invented_case_0"):
    _, bank = data(); grid = bank[case]
    return (copy.deepcopy(grid[("chexgenbench_sana", 0, "maira2")]),
            copy.deepcopy(grid[("roentgen_v2", 0, "cxrmate_single")]))


def fact(candidate, name="edema"):
    return next(f for f in candidate["facts"] if f["finding"] == name)


def rehash(receipt):
    receipt["receipt_id"] = interface._digest({k:v for k,v in receipt.items() if k != "receipt_id"})


class InvariantVerificationTests(unittest.TestCase):
    def test_config_does_not_enable_negative_unknown_weighting_or_execution(self):
        for name in ("unknown_uncertain_are_negative", "secondary_scores_used_for_verification", "model_execution_allowed"):
            c = config(); c[name] = True
            with self.assertRaises(ValueError): interface.validate_config(c)

    def test_anchor_and_reference_ids_independent_of_image_and_report(self):
        a, b = pair()
        x, y = interface.anchor_from_cached_candidate(a), interface.anchor_from_cached_candidate(b)
        self.assertEqual(x, y)
        self.assertEqual(x.sha256, y.sha256)
        self.assertEqual(x.record()["findings"], y.record()["findings"])

    def test_anchor_immutable_and_returned_record_not_an_alias(self):
        a, _ = pair(); anchor = interface.anchor_from_cached_candidate(a); before = anchor.sha256
        with self.assertRaises(FrozenInstanceError): anchor.ehr_sha256 = "a" * 64
        record = anchor.record(); record["findings"][0]["state"] = "negative"
        self.assertEqual(anchor.sha256, before)
        with self.assertRaises(ValueError): interface.EHRAnchor(anchor.case_id, anchor.ehr_sha256, anchor.ehr_facts_sha256, list(anchor.findings))

    def test_hash_or_ehr_state_changes_rejected(self):
        a, b = pair(); anchor = interface.anchor_from_cached_candidate(a)
        fact(b)["states"]["ehr"] = "negative"
        with self.assertRaises(ValueError): interface.verify_candidate(b, anchor)
        a, b = pair(); anchor = interface.anchor_from_cached_candidate(a)
        b["score_record"]["lineage"]["ehr_sha256"] = "a" * 64
        with self.assertRaises(ValueError): interface.verify_candidate(b, anchor)

    def test_provenance_cannot_be_weak_context_or_invented_categories(self):
        a, _ = pair(); anchor = interface.anchor_from_cached_candidate(a)
        fact(a)["weak_context_promoted"] = True
        with self.assertRaises(ValueError): interface.verify_candidate(a, anchor)
        a, _ = pair(); fact(a)["source_categories"] = []
        with self.assertRaises(ValueError): interface.anchor_from_cached_candidate(a)
        a, b = pair(); anchor = interface.anchor_from_cached_candidate(a)
        fact(b)["source_categories"] = ["new_unseen_category"]
        with self.assertRaises(ValueError): interface.verify_candidate(b, anchor)

    def test_no_direct_ehr_rates_remain_na_and_case_stays(self):
        for case in ("invented_case_1", "invented_case_2"):
            a, _ = pair(case); receipt = interface.verify_candidate(a, interface.anchor_from_cached_candidate(a))
            self.assertEqual(receipt["verification_status"], "unverified_no_direct_ehr_constraints")
            self.assertIsNone(receipt["all_three_support_over_known"])
            self.assertIsNone(receipt["raw_edge_readouts"]["ehr_report"]["support_over_known"])
            self.assertEqual(receipt["known_ehr_facts"], 0)

    def test_unknown_report_and_global_normal_source_adjustment_not_negative(self):
        source, details = invented(("positive",), no_finding=True)
        grid = make_legacy_bank(source, details, policy())["invented_case_0"]
        candidate = grid[("chexgenbench_sana", 0, "maira2")]
        receipt = interface.verify_candidate(candidate, interface.anchor_from_cached_candidate(candidate))
        self.assertEqual(candidate["score_record"]["scoring"]["edge_metrics"]["ehr_report"]["contradiction_count"], 1)
        self.assertEqual(receipt["raw_edge_readouts"]["ehr_report"]["proxy_opposition_facts"], 0)
        self.assertEqual(receipt["raw_edge_readouts"]["ehr_report"]["missing_comparisons"], 1)
        self.assertEqual(next(f for f in receipt["fact_states"] if f["finding"] == "edema")["chexbert"], "unknown")

    def test_uncertain_report_is_not_a_negative(self):
        a, _ = pair(); fact(a)["states"]["chexbert"] = "uncertain"
        receipt = interface.verify_candidate(a, interface.anchor_from_cached_candidate(a))
        edge = receipt["raw_edge_readouts"]["ehr_report"]
        self.assertEqual(edge["proxy_opposition_facts"], 0)
        self.assertEqual(edge["supported_negative"], 0)
        self.assertEqual(edge["missing_comparisons"], 1)

    def test_scalar_scores_runtime_and_old_winner_flags_do_not_change_receipt(self):
        a, _ = pair(); anchor = interface.anchor_from_cached_candidate(a)
        before = interface.verify_candidate(a, anchor)
        a["score_record"]["scoring"]["clinical_totals"]["clinical_balance_score_0_100"] = 999
        a["score_record"]["scoring"]["cost"]["known_runtime_seconds"] = 0
        a["score_record"]["scoring"]["selection"]["selected"] = True
        a["score_record"]["secondary_scores"] = {"biovil_report_cxr": 0.99}
        self.assertEqual(before, interface.verify_candidate(a, anchor))

    def test_receipt_tampering_and_rehashed_false_arithmetic_rejected(self):
        a, b = pair(); anchor = interface.anchor_from_cached_candidate(a)
        old, new = interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor)
        altered = copy.deepcopy(new); altered["all_three_supported_facts"] = 99
        with self.assertRaises(ValueError): interface.verify_transition(old, altered, anchor)
        rehash(altered)
        with self.assertRaises(ValueError): interface.verify_transition(old, altered, anchor)
        altered = copy.deepcopy(new); altered["clinical_acceptance"] = True; rehash(altered)
        with self.assertRaises(ValueError): interface.verify_transition(old, altered, anchor)

    def test_rehashed_extra_score_or_invented_scope_rejected(self):
        a, b = pair(); anchor = interface.anchor_from_cached_candidate(a)
        old, new = interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor)
        for key, value in (("new_weight", 1), ("report_scope_status", "clinically_verified")):
            altered = copy.deepcopy(new); altered[key] = value; rehash(altered)
            with self.assertRaises(ValueError): interface.verify_transition(old, altered, anchor)

    def test_image_switch_without_ehr_constraints_is_not_repair(self):
        a, b = pair("invented_case_1"); anchor = interface.anchor_from_cached_candidate(a)
        t = interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        self.assertTrue(t["image_changed"])
        self.assertEqual(t["verification_status"], "unverified_no_direct_ehr_constraints")
        self.assertIsNone(t["all_three_support_delta"])
        self.assertFalse(t["clinical_repair_success"])

    def test_image_support_gain_stays_proxy_and_no_clinical_acceptance(self):
        a, b = pair(); fact(a)["states"]["xrv"] = "negative"
        anchor = interface.anchor_from_cached_candidate(a)
        t = interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        self.assertEqual(t["verification_status"], "fixed_ehr_proxy_support_increased_unvalidated")
        self.assertEqual(t["finding_changes"]["image_support_added"], ["edema"])
        self.assertFalse(t["clinical_repair_success"])
        self.assertFalse(t["clinical_acceptance"])
        self.assertIsNone(t["confirmed_faulty_modality"])

    def test_withdrawn_support_and_missing_are_not_success(self):
        a, b = pair(); fact(b)["states"]["xrv"] = "unknown"
        anchor = interface.anchor_from_cached_candidate(a)
        t = interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        self.assertEqual(t["verification_status"], "fixed_ehr_proxy_support_withdrawn_or_opposition_added")
        self.assertEqual(t["finding_changes"]["image_comparison_missing_after"], ["edema"])
        self.assertFalse(t["clinical_repair_success"])

    def test_opposition_withdrawn_into_uncertain_does_not_become_improvement(self):
        a, b = pair(); fact(a)["states"]["xrv"] = "negative"; fact(b)["states"]["xrv"] = "uncertain"
        anchor = interface.anchor_from_cached_candidate(a)
        t = interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        self.assertEqual(t["verification_status"], "unverified_missing_image_comparison")
        self.assertEqual(t["finding_changes"]["image_opposition_removed"], ["edema"])
        self.assertFalse(t["clinical_repair_success"])

    def test_mixed_findings_not_collapsed_to_a_positive_total(self):
        a, b = pair(); fact(a)["states"]["xrv"] = "negative"
        for candidate in (a, b):
            f = fact(candidate, "pneumonia"); f["states"]["ehr"] = "positive"; f["source_categories"] = ["diagnosis"]
        fact(a, "pneumonia")["states"]["xrv"] = "positive"
        fact(b, "pneumonia")["states"]["xrv"] = "negative"
        anchor = interface.anchor_from_cached_candidate(a)
        t = interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        self.assertEqual(t["verification_status"], "mixed_fixed_ehr_proxy_change_unvalidated")
        self.assertEqual(t["finding_changes"]["image_support_added"], ["edema"])
        self.assertEqual(t["finding_changes"]["image_support_lost"], ["pneumonia"])

    def test_report_only_change_has_no_image_repair_claim(self):
        _, bank = data(); grid = bank["invented_case_0"]
        a, b = grid[("chexgenbench_sana", 0, "maira2")], grid[("chexgenbench_sana", 0, "cxrmate_single")]
        anchor = interface.anchor_from_cached_candidate(a)
        t = interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        self.assertFalse(t["image_changed"])
        self.assertEqual(t["verification_status"], "no_image_artifact_change")
        self.assertFalse(t["clinical_repair_success"])

    def test_same_artifact_cannot_acquire_different_states(self):
        a, _ = pair(); b = copy.deepcopy(a); fact(b)["states"]["xrv"] = "negative"
        anchor = interface.anchor_from_cached_candidate(a)
        with self.assertRaisesRegex(ValueError, "unchanged image"):
            interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)
        b = copy.deepcopy(a); fact(b)["states"]["chexbert"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unchanged report"):
            interface.verify_transition(interface.verify_candidate(a, anchor), interface.verify_candidate(b, anchor), anchor)

    def test_all_agreement_and_artifact_failures_do_not_enable_execution(self):
        _, b = pair(); anchor = interface.anchor_from_cached_candidate(b)
        r = interface.verify_candidate(b, anchor)
        self.assertEqual(r["verification_status"], "direct_ehr_label_agreement_unvalidated")
        self.assertFalse(r["model_execution_allowed"])
        b["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] = 1
        self.assertEqual(interface.verify_candidate(b, anchor)["verification_status"], "artifact_metadata_invalid")

    def test_stream_keeps_original_inputs_winners_actions_and_ledgers(self):
        rows, bank = data(); before = copy.deepcopy((rows, bank))
        result = interface.stream_cached_verification(rows, bank, policy(), config())
        self.assertEqual(before, (rows, bank))
        self.assertEqual(len(result["anchors"]), 3)
        self.assertEqual(len(result["receipts"]), 36)
        self.assertEqual(len(result["trials"]), len(rows))
        original = {(r["case_id"], r["method"], r["model_call_budget"], r["random_seed"]): r for r in rows}
        for t in result["trials"]:
            source = original[(t["case_id"], t["method"], t["model_call_budget"], t["random_seed"])]
            self.assertEqual(t["selected_candidate_id"], source["selected_candidate_id"])
            self.assertEqual(t["simulated_calls"], source["simulated_calls"])
            self.assertEqual(t["original_terminal_reason"], source["terminal_reason"])
        for e in result["events"]:
            source = original[(e["case_id"], e["method"], e["model_call_budget"], e["random_seed"])]["action_trace"][e["step"]]
            self.assertEqual(e["original_action"], source["action"])
            self.assertEqual(e["original_observed_candidate_id"], source["observed_candidate_id"])
            self.assertEqual(e["cumulative_model_calls"], source["cumulative_model_calls"])

    def test_stream_deterministic_and_repeated_events_not_patients(self):
        rows, bank = data()
        a = interface.stream_cached_verification(rows, bank, policy(), config())
        b = interface.stream_cached_verification(list(reversed(rows)), bank, policy(), config())
        self.assertEqual(a, b)
        self.assertTrue(all(r["transition_events_are_not_independent_patients"] for r in a["transition_status_counts"]))

    def test_complete_case_inventory_and_budget_corruption_rejected(self):
        rows, bank = data()
        with self.assertRaises(ValueError): interface.stream_cached_verification(rows[:-1], bank, policy(), config())
        changed = copy.deepcopy(rows); changed[0]["action_trace"][0]["charged_model_calls"] = 0
        with self.assertRaises(ValueError): interface.stream_cached_verification(changed, bank, policy(), config())

    def test_incomplete_candidate_and_forged_clinical_truth_rejected(self):
        a, _ = pair(); anchor = interface.anchor_from_cached_candidate(a); a["facts"].pop()
        with self.assertRaises(ValueError): interface.verify_candidate(a, anchor)
        a, _ = pair(); fact(a)["clinical_truth_verified"] = True
        with self.assertRaises(ValueError): interface.anchor_from_cached_candidate(a)

    def test_no_raw_patient_report_or_image_fields_in_outputs(self):
        rows, bank = data(); text = str(interface.stream_cached_verification(rows, bank, policy(), config()))
        for field in ("subject_id", "patient_id", "report_text", "image_path", "source_statement"):
            self.assertNotIn(field, text)

    def test_cli_slurm_guard_before_protected_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "load_source") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

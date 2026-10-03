"""Invented histories/finding states; not clinical validation or inference."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import ranking_switch_controls as controls
from tricompose_v12.automatic_replay import replay_case, candidate_key
from tricompose_v12.automatic_secondary import overlay
from tricompose_v12.fixed_image_control import build_trials
from tricompose_v12.legacy_replay_adapter import make_legacy_bank, FINDINGS
from test_automatic_secondary import fixture
from test_legacy_automatic_replay import invented, policy
from test_fixed_image_control import config as fixed_config
import run_ranking_switch_controls as cli


def config():
    return json.loads((ROOT / "configs/ranking_switch_control_pool80_v1.json").read_text())


def data():
    original, request, scores = fixture()
    original = overlay(original, request, scores)
    source, details = invented(); bank = make_legacy_bank(source, details, policy())
    rows, _ = build_trials(original, bank, policy(), fixed_config())
    return original, rows, bank


def tied_trial():
    _, _, bank = data(); grid = bank["invented_case_0"]
    first, faster = (grid[("chexgenbench_sana", 0, r)] for r in ("cxrmate_single", "llavarad"))
    first["score_record"]["scoring"]["cost"]["known_runtime_seconds"] = 100
    faster["score_record"]["scoring"]["cost"]["known_runtime_seconds"] = 1
    row = replay_case(grid, policy(), "static_rerank", 10)
    row["selected_snapshot"].update(biovil_cosine_secondary_not_routing=0.4,
        secondary_endpoint_status="computed_secondary_uncalibrated", secondary_endpoint_unavailable_reason=None)
    row["input_ehr_assessment_scope"] = "explicit_fact_proxy"
    endpoints = controls.endpoint_index([row], bank)
    return row, grid, endpoints, first, faster


def candidate_facts():
    return {"facts": [{"finding": name, "states": {"ehr": "unknown", "xrv": "unknown", "chexbert": "unknown"}}
                      for name in FINDINGS]}


class RankingSwitchControlTests(unittest.TestCase):
    def test_config_rejects_peeking_imputation_and_changed_clinical_scope(self):
        for name, value in (("candidate_visibility", "all_candidates"),
                            ("secondary_scores_used_for_selection", True),
                            ("unknown_uncertain_are_negative", True),
                            ("cohort_role", "held_out")):
            c = config(); c[name] = value
            with self.assertRaises(ValueError): controls.validate_control(c)

    def test_runtime_removal_keeps_exact_other_key_components(self):
        _, grid, _, first, _ = tied_trial()
        for c in grid.values():
            self.assertEqual(controls.no_runtime_key(c), (*candidate_key(c)[:5], candidate_key(c)[6]))
        first["score_record"]["scoring"]["cost"]["known_runtime_seconds"] = None
        self.assertEqual(candidate_key(first)[5], float("inf"))

    def test_changed_tie_winner_same_prefix_history_calls_and_missing_endpoint(self):
        row, grid, endpoints, first, faster = tied_trial()
        result = controls.ablate_trial(row, grid, endpoints)
        self.assertEqual(row["selected_candidate_id"], faster["score_record"]["triple_candidate_id"])
        self.assertEqual(result["selected_candidate_id"], first["score_record"]["triple_candidate_id"])
        self.assertEqual(result["action_trace"], row["action_trace"])
        self.assertEqual(result["simulated_calls"], row["simulated_calls"])
        self.assertEqual(result["terminal_reason"], row["terminal_reason"])
        self.assertIsNone(result["selected_snapshot"]["biovil_cosine_secondary_not_routing"])

    def test_endpoint_value_or_availability_cannot_choose_winner(self):
        row, grid, endpoints, first, _ = tied_trial()
        a = controls.ablate_trial(row, grid, endpoints)
        endpoints[first["score_record"]["triple_candidate_id"]] = {
            "biovil_cosine_secondary_not_routing": -0.9,
            "secondary_endpoint_status": "computed_secondary_uncalibrated", "secondary_endpoint_unavailable_reason": None}
        b = controls.ablate_trial(row, grid, endpoints)
        self.assertEqual(a["selected_candidate_id"], b["selected_candidate_id"])
        self.assertEqual(a["action_trace"], b["action_trace"])

    def test_unobserved_candidate_scores_not_inspected(self):
        original, _, bank = data(); grid = bank["invented_case_0"]
        row = next(r for r in original if r["case_id"] == "invented_case_0" and r["method"] == "fixed" and r["model_call_budget"] == 4)
        endpoints = controls.endpoint_index([row], bank)
        for slot, c in grid.items():
            if slot != ("chexgenbench_sana", 0, "maira2"): c["score_record"]["scoring"] = None
        self.assertEqual(controls.ablate_trial(row, grid, endpoints)["selected_candidate_id"], row["selected_candidate_id"])

    def test_original_proxy_stop_choice_retained(self):
        original, _, bank = data()
        row = next(r for r in original if r["method"] == "targeted_heuristic" and r["terminal_reason"] == "stop_proxy_satisfied")
        result = controls.ablate_trial(row, bank[row["case_id"]], controls.endpoint_index(original, bank))
        self.assertEqual(result["selected_candidate_id"], row["selected_candidate_id"])
        self.assertEqual(result["terminal_reason"], row["terminal_reason"])

    def test_forged_observed_slot_duplicate_or_budget_rejected(self):
        row, grid, endpoints, _, _ = tied_trial()
        for kind in ("slot", "duplicate", "budget", "cost"):
            changed = copy.deepcopy(row)
            if kind == "slot": changed["action_trace"][0]["request_slot"][0] = "unseen_model"
            elif kind == "duplicate": changed["action_trace"].append(copy.deepcopy(changed["action_trace"][0]))
            elif kind == "budget": changed["model_call_budget"] = 3
            else: changed["simulated_model_calls"] += 1
            with self.assertRaises(ValueError): controls.ablate_trial(changed, grid, endpoints)

    def test_unobserved_source_winner_or_changed_prefix_rejected(self):
        row, grid, endpoints, _, _ = tied_trial(); changed = copy.deepcopy(row)
        changed["selected_candidate_id"] = grid[("roentgen_v2", 0, "maira2")]["score_record"]["triple_candidate_id"]
        with self.assertRaises(ValueError): controls.ablate_trial(changed, grid, endpoints)
        with patch.object(controls, "no_runtime_key", lambda c: -c["score_record"]["scoring"]["clinical_totals"]["total_hard_contradiction_count"]):
            with self.assertRaisesRegex(ValueError, "prefix"): controls.ablate_trial(row, grid, endpoints)

    def test_input_objects_old_winners_and_actions_immutable(self):
        original, rows, bank = data(); before = copy.deepcopy((original, rows, bank))
        ablated, cases, summary, pending = controls.ranking_ablation(rows, bank, original, config())
        switches, _ = controls.invariant_switches(rows, bank, config())
        self.assertEqual(before, (original, rows, bank))
        self.assertEqual(len(ablated), 3 * 5 * 5)
        self.assertEqual(len(cases), len(ablated))
        self.assertEqual(len(switches), 3 * 5 * 2)
        self.assertTrue(all(r["simulated_call_delta"] == 0 for r in summary))

    def test_ranking_complete_cases_and_no_seed_inflation(self):
        original, rows, bank = data()
        _, _, summary, _ = controls.ranking_ablation(rows, bank, original, config())
        self.assertTrue(all(r["all_fixed_ehr_cases"] == 3 for r in summary if r["ehr_evidence_subgroup"] == "all"))
        with self.assertRaises(ValueError): controls.ranking_ablation(rows[:-1], bank, original, config())
        with self.assertRaises(ValueError): controls.invariant_switches([*rows, rows[0]], bank, config())

    def test_missing_all_endpoints_keeps_choices_full_case_cost_and_na(self):
        original, rows, bank = data()
        for collection in (original, rows):
            for r in collection:
                r["selected_snapshot"].update(biovil_cosine_secondary_not_routing=None,
                    secondary_endpoint_status="not_available", secondary_endpoint_unavailable_reason="empty_report")
        _, _, summary, _ = controls.ranking_ablation(rows, bank, original, config())
        self.assertTrue(all(r["paired_available_biovil_cases"] == 0 for r in summary))
        self.assertTrue(all(r["mean_biovil_delta_no_runtime_minus_original"] is None for r in summary))
        self.assertTrue(all(r["simulated_call_delta"] == 0 for r in summary))

    def test_unknown_uncertain_not_negative_or_direct_constraints(self):
        candidate = candidate_facts(); candidate["facts"][0]["states"].update(ehr="uncertain", xrv="negative", chexbert="negative")
        _, p = controls.invariant_profile(candidate)
        self.assertEqual(p["known_ehr_facts"], 0)
        self.assertEqual(p["direct_image_support"], 0)
        self.assertEqual(p["report_cxr_negative_support_without_direct_ehr"], 1)

    def test_known_reference_support_opposition_missing_denominator(self):
        candidate = candidate_facts()
        candidate["facts"][0]["states"].update(ehr="positive", xrv="positive", chexbert="negative")
        candidate["facts"][1]["states"].update(ehr="negative", xrv="uncertain", chexbert="negative")
        _, p = controls.invariant_profile(candidate)
        self.assertEqual(p["known_ehr_facts"], 2)
        self.assertEqual(p["direct_image_support"], 1)
        self.assertEqual(p["direct_image_missing"], 1)
        self.assertEqual(p["direct_report_opposition"], 1)
        self.assertEqual(p["all_three_support"], 0)

    def test_report_cxr_gain_partition_does_not_establish_ehr_fidelity(self):
        candidate = candidate_facts()
        candidate["facts"][0]["states"].update(ehr="positive", xrv="negative", chexbert="negative")
        candidate["facts"][1]["states"].update(ehr="unknown", xrv="positive", chexbert="positive")
        _, p = controls.invariant_profile(candidate)
        self.assertEqual(p["report_cxr_negative_support_with_direct_ehr"], 1)
        self.assertEqual(p["direct_image_opposition"], 1)
        self.assertEqual(p["report_cxr_positive_support_without_direct_ehr"], 1)
        self.assertEqual(p["all_three_support"], 0)

    def test_incomplete_or_invalid_state_inventory_rejected(self):
        candidate = candidate_facts(); candidate["facts"].pop()
        with self.assertRaises(ValueError): controls.invariant_profile(candidate)
        candidate = candidate_facts(); candidate["facts"][0]["states"]["ehr"] = "assumed_normal"
        with self.assertRaises(ValueError): controls.invariant_profile(candidate)

    def test_no_direct_ehr_rate_na_and_cases_not_removed(self):
        _, rows, bank = data(); cases, summary = controls.invariant_switches(rows, bank, config())
        absent = [r for r in cases if r["ehr_evidence_subgroup"] == "no_direct_comparable_ehr_facts"]
        self.assertTrue(absent)
        self.assertTrue(all(r["invariant_reference_support_rate_after"] is None for r in absent))
        for r in summary:
            if r["ehr_evidence_subgroup"] == "no_direct_comparable_ehr_facts":
                self.assertEqual(r["cases_without_direct_ehr_constraints"], r["selected_pair_cases"])
                self.assertIsNone(r["sum_direct_image_support_delta_constrained_cases"])

    def test_changed_ehr_hash_or_reference_vector_is_not_an_image_repair(self):
        _, rows, bank = data()
        candidate = bank["invented_case_0"][("chexgenbench_sana", 0, "cxrmate_single")]
        candidate["score_record"]["lineage"]["ehr_sha256"] = "a" * 64
        with self.assertRaises(ValueError): controls.invariant_switches(rows, bank, config())
        _, rows, bank = data()
        bank["invented_case_0"][("chexgenbench_sana", 0, "cxrmate_single")]["facts"][0]["states"]["ehr"] = "negative"
        with self.assertRaises(ValueError): controls.invariant_switches(rows, bank, config())

    def test_deterministic_input_order(self):
        original, rows, bank = data()
        self.assertEqual(controls.ranking_ablation(rows, bank, original, config()),
            controls.ranking_ablation(list(reversed(rows)), bank, list(reversed(original)), config()))
        self.assertEqual(controls.invariant_switches(rows, bank, config()), controls.invariant_switches(list(reversed(rows)), bank, config()))

    def test_cli_guard_before_protected_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "cached") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            read.assert_not_called()

    def test_output_contains_no_patient_text_or_image_fields(self):
        original, rows, bank = data()
        result = str((controls.ranking_ablation(rows, bank, original, config()), controls.invariant_switches(rows, bank, config())))
        for field in ("subject_id", "patient_id", "report_text", "image_path", "source_statement"):
            self.assertNotIn(field, result)


if __name__ == "__main__":
    unittest.main()

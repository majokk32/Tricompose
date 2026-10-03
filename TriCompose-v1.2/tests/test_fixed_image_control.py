"""Invented caches only: software contracts, not clinical validation."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import fixed_image_control as control
from tricompose_v12.automatic_replay import evaluation_snapshot
from tricompose_v12.automatic_secondary import overlay, PAIR_FIELDS
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from test_automatic_secondary import fixture
from test_legacy_automatic_replay import invented, policy
import run_fixed_image_control as cli


def config():
    return json.loads((ROOT / "configs/fixed_image_pool80_v1.json").read_text())


def data():
    rows, request, scores = fixture()
    source, details = invented()
    return overlay(rows, request, scores), make_legacy_bank(source, details, policy())


class FixedImageControlTests(unittest.TestCase):
    def test_config_rejects_endpoint_routing_image_changes_and_held_out_claim(self):
        for name, value in (("fixed_image_slot", ["roentgen_v2", 0]),
                            ("secondary_scores_used_for_selection", True),
                            ("cohort_role", "independent_held_out"),
                            ("missing_endpoints", "choose_scored_candidate")):
            p = config(); p[name] = value
            with self.assertRaises(ValueError): control.validate_control(p, policy())

    def test_static_observes_four_reports_one_image_and_ten_calls(self):
        _, bank = data()
        r = control.replay_fixed_image(bank["invented_case_0"], policy(), config(), "report_only_static", 30)
        self.assertEqual(r["observed_images"], 1)
        self.assertEqual(r["observed_candidates"], 4)
        self.assertEqual(r["simulated_model_calls"], 10)
        self.assertEqual(r["simulated_calls"], {"cxr_generator": 1, "xrv": 1, "report_generator": 4, "chexbert": 4})
        self.assertTrue(all(t["request_slot"][:2] == ["chexgenbench_sana", 0] for t in r["action_trace"]))

    def test_targeted_reuses_image_and_keeps_original_proxy_stop(self):
        _, bank = data()
        r = control.replay_fixed_image(bank["invented_case_0"], policy(), config(), "report_only_targeted", 30)
        self.assertEqual(r["simulated_model_calls"], 6)
        self.assertEqual(r["observed_candidates"], 2)
        self.assertEqual(r["terminal_reason"], "stop_proxy_satisfied")
        self.assertFalse(r["clinical_acceptance"])

    def test_image_conflict_is_blocked_not_a_success_or_new_image(self):
        scores, details = invented(("negative",))
        bank = make_legacy_bank(scores, details, policy())
        r = control.replay_fixed_image(bank["invented_case_0"], policy(), config(), "report_only_targeted", 30)
        self.assertEqual(r["terminal_reason"], "fixed_image_cxr_change_blocked")
        self.assertEqual(r["observed_images"], 1)
        self.assertEqual(r["simulated_model_calls"], 4)
        self.assertFalse(r["selected_proxy_stop_conditions_met"])

    def test_all_integer_budgets_respected_and_no_candidate_for_under_four(self):
        _, bank = data()
        for method in control.REPORT_METHODS:
            for budget in range(32):
                r = control.replay_fixed_image(bank["invented_case_0"], policy(), config(), method, budget)
                self.assertLessEqual(r["simulated_model_calls"], budget)
                if budget < 4: self.assertIsNone(r["selected_snapshot"])
        with self.assertRaises(ValueError): control.replay_fixed_image(bank["invented_case_0"], policy(), config(), "report_only_static", True)

    def test_no_unobserved_score_peeking(self):
        _, bank = data(); grid = bank["invented_case_0"]
        for report in ("cxrmate_single", "llavarad", "chexagent2"):
            grid[("chexgenbench_sana", 0, report)]["score_record"]["scoring"] = None
        r = control.replay_fixed_image(grid, policy(), config(), "report_only_static", 4)
        self.assertEqual(r["observed_candidates"], 1)

    def test_fixed_image_hash_changes_or_missing_report_rejected(self):
        _, bank = data(); grid = bank["invented_case_0"]
        grid[("chexgenbench_sana", 0, "llavarad")]["score_record"]["lineage"]["cxr_sha256"] = "a" * 64
        with self.assertRaises(ValueError): control.replay_fixed_image(grid, policy(), config(), "report_only_static", 4)
        _, bank = data(); grid = bank["invented_case_0"]; del grid[("chexgenbench_sana", 0, "llavarad")]
        with self.assertRaises(ValueError): control.replay_fixed_image(grid, policy(), config(), "report_only_static", 4)

    def test_no_eligible_candidate_preserves_failure_and_budget(self):
        _, bank = data(); grid = bank["invented_case_0"]
        for candidate in grid.values(): candidate["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] = 1
        r = control.replay_fixed_image(grid, policy(), config(), "report_only_static", 30)
        self.assertIsNone(r["selected_candidate_id"])
        self.assertEqual(r["simulated_model_calls"], 10)

    def test_source_inputs_original_winners_and_rules_unchanged(self):
        rows, bank = data(); before = copy.deepcopy((rows, bank, config()))
        result, _ = control.build_trials(rows, bank, policy(), config())
        self.assertEqual(before, (rows, bank, config()))
        old = {(r["case_id"], r["method"], r["model_call_budget"]): r for r in rows if r["method"] in control.BASELINES}
        for r in result:
            if r["method"] in control.BASELINES:
                self.assertEqual(r, old[(r["case_id"], r["method"], r["model_call_budget"])])

    def test_endpoint_values_cannot_change_new_choices_or_actions(self):
        rows, bank = data(); changed = copy.deepcopy(rows)
        for r in changed: r["selected_snapshot"]["biovil_cosine_secondary_not_routing"] *= -1
        a, _ = control.build_trials(rows, bank, policy(), config())
        b, _ = control.build_trials(changed, bank, policy(), config())
        self.assertEqual([r["selected_candidate_id"] for r in a], [r["selected_candidate_id"] for r in b])
        self.assertEqual([r["action_trace"] for r in a], [r["action_trace"] for r in b])

    def test_unscored_new_winner_remains_na_not_a_scored_substitute(self):
        rows, bank = data(); grid = bank["invented_case_0"]
        missing = grid[("chexgenbench_sana", 0, "cxrmate_single")]["score_record"]["triple_candidate_id"]
        fallback = grid[("chexgenbench_sana", 0, "maira2")]
        snapshot = evaluation_snapshot(fallback)
        snapshot.update(biovil_cosine_secondary_not_routing=0.2,
            secondary_endpoint_status="computed_secondary_uncalibrated", secondary_endpoint_unavailable_reason=None)
        for r in rows:
            if r["selected_candidate_id"] in {missing, fallback["score_record"]["triple_candidate_id"]}:
                r.update(selected_candidate_id=fallback["score_record"]["triple_candidate_id"], selected_snapshot=copy.deepcopy(snapshot))
        result, pending = control.build_trials(rows, bank, policy(), config())
        chosen = [r for r in result if r["selected_candidate_id"] == missing]
        self.assertTrue(chosen)
        self.assertTrue(all(r["selected_snapshot"]["biovil_cosine_secondary_not_routing"] is None for r in chosen))
        self.assertTrue(all(r["selected_snapshot"]["secondary_endpoint_status"] == "not_scored_in_frozen_endpoint_union" for r in chosen))
        self.assertEqual(len([p for p in pending if p["triple_candidate_id"] == missing]), 1)
        self.assertTrue(all(set(p) == PAIR_FIELDS for p in pending))

    def test_existing_na_reason_preserved_without_requeue_or_imputation(self):
        rows, bank = data()
        for r in rows:
            r["selected_snapshot"].update(biovil_cosine_secondary_not_routing=None,
                secondary_endpoint_status="not_available", secondary_endpoint_unavailable_reason="empty_report")
        result, _ = control.build_trials(rows, bank, policy(), config())
        for r in result:
            if r["selected_snapshot"] is not None:
                self.assertIsNone(r["selected_snapshot"]["biovil_cosine_secondary_not_routing"])
                self.assertEqual(r["selected_snapshot"]["secondary_endpoint_unavailable_reason"], "empty_report")

    def test_case_counts_not_seeds_and_paired_same_case_denominators(self):
        rows, bank = data(); result, _ = control.build_trials(rows, bank, policy(), config())
        per_case, aggregate = control.contrasts(result, bank, config())
        self.assertEqual(len(result), 3 * 5 * 5)
        self.assertEqual(len(per_case), 3 * 5 * 2 * 3)
        self.assertEqual(len(aggregate), 3 * 5 * 2 * 3)
        for r in aggregate:
            if r["ehr_evidence_subgroup"] == "all":
                self.assertEqual(r["all_fixed_ehr_cases"], 3)
                self.assertEqual(r["paired_available_biovil_cases"], 3)
                if r["baseline"] == "fixed": self.assertEqual(r["same_image_reference_pairs"], 3)

    def test_missing_endpoint_retains_all_case_costs(self):
        rows, bank = data()
        for r in rows:
            r["selected_snapshot"].update(biovil_cosine_secondary_not_routing=None,
                secondary_endpoint_status="not_available", secondary_endpoint_unavailable_reason="empty_report")
        result, _ = control.build_trials(rows, bank, policy(), config())
        _, aggregate = control.contrasts(result, bank, config())
        for r in aggregate:
            self.assertEqual(r["paired_available_biovil_cases"], 0)
            self.assertIsNone(r["mean_biovil_delta_on_paired_available_cases"])
            self.assertIsNotNone(r["mean_simulated_call_delta_all_fixed_cases"])
            self.assertEqual(r["selected_pair_cases"], r["all_fixed_ehr_cases"])

    def test_unknown_uncertain_ehr_states_are_retained_without_invented_facts(self):
        rows, bank = data(); result, _ = control.build_trials(rows, bank, policy(), config())
        for r in result:
            if r["case_id"] != "invented_case_0":
                self.assertEqual(r["input_ehr_assessment_scope"], "no_direct_comparable_ehr_facts")
                self.assertEqual(r["selected_snapshot"]["raw_edge_readouts_optimization_proxy"]["ehr_report"]["explicit_reference_facts"], 0)

    def test_deterministic_order_and_exact_input_case_inventory(self):
        rows, bank = data()
        self.assertEqual(control.build_trials(rows, bank, policy(), config()),
            control.build_trials(list(reversed(rows)), dict(reversed(list(bank.items()))), policy(), config()))
        with self.assertRaises(ValueError): control.build_trials(rows[:-1], bank, policy(), config())
        del bank["invented_case_0"]
        with self.assertRaises(ValueError): control.build_trials(rows, bank, policy(), config())

    def test_changed_endpoint_artifact_lineage_rejected(self):
        rows, bank = data(); rows[0]["selected_snapshot"]["artifact_hashes"]["report_sha256"] = "a" * 64
        with self.assertRaises(ValueError): control.build_trials(rows, bank, policy(), config())

    def test_cli_slurm_guard_precedes_all_protected_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "load_source") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            read.assert_not_called()

    def test_outputs_do_not_contain_patient_or_report_image_content(self):
        rows, bank = data(); result, pending = control.build_trials(rows, bank, policy(), config())
        text = str((result, pending, control.contrasts(result, bank, config())))
        for field in ("subject_id", "patient_id", "report_text", "image_path", "source_statement"):
            self.assertNotIn(field, text)


if __name__ == "__main__":
    unittest.main()

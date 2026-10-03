"""Invented cached-state grids; tests are not clinical validation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import automatic_replay as replay
from tricompose_v12.report_scope_table import EDGES, EXPLICIT, FINDINGS
from test_decision_preview import fixture, update_fact
import run_automatic_proxy_replay as cli


def policy():
    return json.loads((ROOT / "configs/automatic_replay_v1.json").read_text())


def invented_bank(*, image_states=None, report_states=None, ehr="positive"):
    p = policy(); scores, facts = [], []
    for i, (model, seed) in enumerate(p["image_order"]):
        for j, report_model in enumerate(p["report_order"]):
            cid = f"fictional_candidate_{i}_{j}"
            hashes = {"ehr_sha256": "e" * 64, "ehr_facts_sha256": "f" * 64,
                      "cxr_sha256": hashlib.sha256(f"fictional_image_{i}".encode()).hexdigest(),
                      "report_sha256": hashlib.sha256(cid.encode()).hexdigest()}
            rows = fixture(ehr="unknown", image="unknown", report="unknown")
            update_fact(rows[FINDINGS.index("pneumonia")], ehr=ehr,
                image="positive" if image_states is None else image_states[i],
                report=("negative" if j == 0 else "positive") if report_states is None else report_states[j])
            for f in rows:
                f.update(evidence_id=f"{cid}_{f['finding']}", triple_candidate_id=cid,
                    report_candidate_id=cid, cxr_candidate_id=f"fictional_image_{i}",
                    artifact_hashes=copy.deepcopy(hashes), image_dependency_group=hashes["cxr_sha256"])
            known = sum(2 * (f["states"]["ehr"] in EXPLICIT) + (f["states"]["xrv"] in EXPLICIT) for f in rows)
            support = sum(f["relations"]["raw"][edge] == "support" for f in rows for edge in EDGES)
            conflicts = sum(f["relations"]["raw"][edge] == "opposition" for f in rows for edge in EDGES)
            ehr_support = sum(f["relations"]["raw"][edge] == "support" for f in rows for edge in ("ehr_cxr", "ehr_report"))
            image_known = sum(f["states"]["xrv"] in EXPLICIT for f in rows)
            image_support = sum(f["relations"]["raw"]["cxr_report"] == "support" for f in rows)
            record = {"schema_version": "tricompose-edge-specific-selection-v1.1", "case_id": "fixture_case",
                "triple_candidate_id": cid, "lineage": {**hashes, "cxr_candidate_id": f"fictional_image_{i}",
                    "report_candidate_id": cid, "cxr_model_id": model, "cxr_seed": seed, "report_model_id": report_model},
                "scoring": {"selection": {"hard_gate_failure_count": 0, "selected": False, "selection_rank_within_case": i * 4 + j + 1},
                    "modality_quality": {"cxr_basic_validity_pass": True, "report_structure_quality_score_0_1": 0.8 + 0.001 * j},
                    "clinical_totals": {"total_hard_contradiction_count": conflicts, "total_direct_support_count": support,
                        "ehr_direct_support_count": ehr_support, "total_known_reference_fact_count": known,
                        "clinical_balance_score_0_100": 50 * (1 + (support - conflicts) / known) if known else None},
                    "edge_metrics": {source_name: {
                        "support_count": sum(f["relations"]["raw"][fact_name] == "support" for f in rows),
                        "contradiction_count": sum(f["relations"]["raw"][fact_name] == "opposition" for f in rows),
                        "known_reference_fact_count": sum(f["states"]["xrv" if source_name == "report_cxr" else "ehr"] in EXPLICIT for f in rows),
                        "support_recall": image_support / image_known if image_known else None}
                        for source_name, fact_name in (("ehr_cxr", "ehr_cxr"), ("ehr_report", "ehr_report"), ("report_cxr", "cxr_report"))},
                    "cost": {"known_runtime_seconds": 20 + i + j}},
                "secondary_scores": {"biovil_report_cxr": {"raw_cosine": 0.4 + 0.01 * j,
                    "status": "computed_secondary_uncalibrated", "calibrated": False}}}
            scores.append(record); facts.extend(rows)
    return replay.make_bank(scores, facts, p)["fixture_case"], scores, facts


class AutomaticReplayTests(unittest.TestCase):
    def test_fixed_path_charges_generation_and_both_scorers(self):
        bank, _, _ = invented_bank()
        row = replay.replay_case(bank, policy(), "fixed", 60)
        self.assertEqual(row["simulated_model_calls"], 4)
        self.assertEqual(row["simulated_calls"], {"cxr_generator": 1, "xrv": 1, "report_generator": 1, "chexbert": 1})
        self.assertEqual(row["observed_candidates"], 1)

    def test_small_budget_returns_no_candidate_instead_of_free_initial_calls(self):
        bank, _, _ = invented_bank()
        for method in replay.METHODS:
            row = replay.replay_case(bank, policy(), method, 3)
            self.assertIsNone(row["selected_candidate_id"])
            self.assertEqual(row["simulated_model_calls"], 0)

    def test_targeted_report_switch_reuses_the_image(self):
        bank, _, _ = invented_bank()
        row = replay.replay_case(bank, policy(), "targeted_heuristic", 8)
        self.assertEqual(row["action_trace"][1]["action"], "switch_report_model")
        self.assertEqual(row["simulated_model_calls"], 6)
        self.assertEqual(row["observed_images"], 1)
        self.assertEqual(row["simulated_calls"]["xrv"], 1)
        self.assertTrue(row["selected_proxy_stop_conditions_met"])
        self.assertFalse(row["actual_regeneration_executed"])
        self.assertFalse(row["clinical_acceptance"])

    def test_image_conflict_changes_image_not_report_votes(self):
        bank, _, _ = invented_bank(image_states=["negative", "positive", "positive", "positive", "positive", "positive"])
        row = replay.replay_case(bank, policy(), "targeted_heuristic", 8)
        self.assertEqual(row["action_trace"][1]["action"], "regenerate_cxr")
        self.assertEqual(row["observed_images"], 2)
        self.assertEqual(row["simulated_model_calls"], 8)
        self.assertTrue(row["action_trace"][1]["trigger_evidence_ids"])

    def test_report_inventory_exhaustion_moves_to_another_image(self):
        bank, _, _ = invented_bank(report_states=["negative"] * 4)
        row = replay.replay_case(bank, policy(), "targeted_heuristic", 20)
        self.assertEqual(row["action_trace"][4]["action"], "regenerate_cxr")
        self.assertFalse(row["selected_proxy_stop_conditions_met"])

    def test_all_calls_are_bounded_for_every_method_and_seed(self):
        bank, _, _ = invented_bank(report_states=["negative"] * 4)
        for method in replay.METHODS:
            for budget in range(0, 65):
                row = replay.replay_case(bank, policy(), method, budget, random_seed=2)
                self.assertLessEqual(row["simulated_model_calls"], budget)
                self.assertEqual(row["simulated_model_calls"], sum(step["charged_model_calls"] for step in row["action_trace"]))

    def test_exhaustive_grid_cost_includes_verification_and_no_image_recharge(self):
        bank, _, _ = invented_bank()
        row = replay.replay_case(bank, policy(), "static_rerank", 60)
        self.assertEqual(row["observed_candidates"], 24)
        self.assertEqual(row["observed_images"], 6)
        self.assertEqual(row["simulated_calls"], {"cxr_generator": 6, "xrv": 6, "report_generator": 24, "chexbert": 24})

    def test_unseen_scores_cannot_change_targeted_prefix_or_choice(self):
        bank, _, _ = invented_bank()
        altered = copy.deepcopy(bank)
        for slot, candidate in altered.items():
            if slot != ("chexgenbench_sana", 0, "maira2"):
                candidate["score_record"]["scoring"]["clinical_totals"]["total_hard_contradiction_count"] = 0
                candidate["score_record"]["scoring"]["modality_quality"]["report_structure_quality_score_0_1"] = 1
        for method in replay.METHODS:
            before = replay.replay_case(bank, policy(), method, 4)
            # Random gets an arbitrary first slot; unchanged observations are
            # checked on fixed/static/targeted below, not a different first input.
            if method != "random":
                after = replay.replay_case(altered, policy(), method, 4)
                self.assertEqual(before, after)

    def test_secondary_scores_cannot_change_actions_or_selection(self):
        bank, _, _ = invented_bank(); changed = copy.deepcopy(bank)
        for candidate in changed.values():
            candidate["score_record"]["secondary_scores"]["biovil_report_cxr"]["raw_cosine"] *= -1
        for method in replay.METHODS:
            first = replay.replay_case(bank, policy(), method, 20)
            second = replay.replay_case(changed, policy(), method, 20)
            self.assertEqual(first["action_trace"], second["action_trace"])
            self.assertEqual(first["selected_candidate_id"], second["selected_candidate_id"])

    def test_original_full_bank_rank_and_selected_flag_are_not_inputs(self):
        bank, _, _ = invented_bank(); changed = copy.deepcopy(bank)
        for candidate in changed.values():
            candidate["score_record"]["scoring"]["selection"].update(selection_rank_within_case=-1000, selected=True)
        self.assertEqual(replay.replay_case(bank, policy(), "static_rerank", 20),
                         replay.replay_case(changed, policy(), "static_rerank", 20))

    def test_random_order_is_deterministic_and_score_free(self):
        bank, _, _ = invented_bank()
        self.assertEqual(replay.replay_case(bank, policy(), "random", 60, random_seed=3),
                         replay.replay_case(dict(reversed(list(bank.items()))), policy(), "random", 60, random_seed=3))
        one = replay.replay_case(bank, policy(), "random", 60, random_seed=1)
        two = replay.replay_case(bank, policy(), "random", 60, random_seed=2)
        self.assertNotEqual(one["action_trace"], two["action_trace"])

    def test_unknown_ehr_is_not_fabricated_or_full_triple_acceptance(self):
        bank, _, _ = invented_bank(ehr="unknown")
        row = replay.replay_case(bank, policy(), "targeted_heuristic", 20)
        self.assertEqual(row["selected_snapshot"]["ehr_assessment_scope"], "no_direct_comparable_ehr_facts")
        self.assertFalse(row["clinical_acceptance"])

    def test_all_unknown_or_uncertain_image_report_evidence_cannot_stop_successfully(self):
        for state in ("unknown", "uncertain"):
            bank, _, _ = invented_bank(image_states=[state] * 6, report_states=[state] * 4, ehr="unknown")
            row = replay.replay_case(bank, policy(), "targeted_heuristic", 60)
            self.assertFalse(row["selected_proxy_stop_conditions_met"])
            self.assertNotEqual(row["terminal_reason"], "stop_proxy_satisfied")

    def test_unsupported_pneumonia_scope_stays_unavailable_in_alternate_readout(self):
        bank, _, _ = invented_bank()
        row = replay.replay_case(bank, policy(), "targeted_heuristic", 8)
        readout = row["selected_snapshot"]["guarded_edge_readouts_not_routing"]["ehr_report"]
        self.assertEqual(readout["comparable_facts"], 0)
        self.assertIsNone(readout["conditional_support_fraction"])

    def test_hard_artifact_failure_cannot_be_returned_as_selected(self):
        bank, _, _ = invented_bank()
        for candidate in bank.values(): candidate["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] = 1
        row = replay.replay_case(bank, policy(), "static_rerank", 20)
        self.assertIsNone(row["selected_candidate_id"])

    def test_static_incumbent_never_worsens_its_own_objective_with_larger_prefix(self):
        bank, _, _ = invented_bank(); keys = []
        by_id = {candidate["score_record"]["triple_candidate_id"]: candidate for candidate in bank.values()}
        for budget in (4, 8, 12, 20, 40, 60):
            row = replay.replay_case(bank, policy(), "static_rerank", budget)
            keys.append(replay.candidate_key(by_id[row["selected_candidate_id"]]))
        self.assertEqual(keys, sorted(keys, reverse=True))

    def test_fixed_ehr_hash_retained_and_source_bank_not_mutated(self):
        bank, _, _ = invented_bank(); before = copy.deepcopy(bank)
        for method in replay.METHODS:
            self.assertEqual(replay.replay_case(bank, policy(), method, 60)["selected_ehr_sha256"], "e" * 64)
        self.assertEqual(before, bank)

    def test_policy_rejects_secondary_routing_or_clinical_claim(self):
        for key in ("secondary_scores_used_for_routing", "clinical_accuracy_claim_allowed", "requires_human_feedback_for_replay"):
            p = policy(); p[key] = True
            with self.assertRaises(ValueError): replay.validate_policy(p)

    def test_model_order_missing_slots_or_invalid_budget_policy_rejected(self):
        for key in ("image_order", "report_order", "model_call_budgets", "random_seeds"):
            p = policy(); p[key] = []
            with self.assertRaises(ValueError): replay.validate_policy(p)

    def test_missing_candidate_hash_mismatch_and_wrong_counts_rejected(self):
        _, scores, facts = invented_bank()
        with self.assertRaises(ValueError): replay.make_bank(scores[:-1], facts, policy())
        changed = copy.deepcopy(scores); changed[0]["lineage"]["ehr_sha256"] = "a" * 64
        with self.assertRaises(ValueError): replay.make_bank(changed, facts, policy())
        changed = copy.deepcopy(scores); changed[0]["scoring"]["clinical_totals"]["total_hard_contradiction_count"] += 1
        with self.assertRaisesRegex(ValueError, "counts differ"): replay.make_bank(changed, facts, policy())

    def test_legacy_global_normal_statement_metric_does_not_flip_unknown_state(self):
        _, scores, facts = invented_bank(report_states=["unknown", "positive", "positive", "positive"])
        scores[0]["scoring"]["edge_metrics"]["ehr_report"]["contradiction_count"] = 1
        scores[0]["scoring"]["clinical_totals"]["total_hard_contradiction_count"] += 1
        bank = replay.make_bank(scores, facts, policy())["fixture_case"]
        candidate = bank[("chexgenbench_sana", 0, "maira2")]
        self.assertEqual(candidate["facts"][FINDINGS.index("pneumonia")]["states"]["chexbert"], "unknown")
        self.assertEqual(replay.action_signal(candidate)[0], "switch_report_model")

    def test_random_repetitions_are_not_counted_as_new_patients(self):
        bank, _, _ = invented_bank()
        rows = [replay.replay_case(bank, policy(), "random", 20, random_seed=i) for i in range(5)]
        result = cli.compare(rows)[0]
        self.assertEqual(result["unique_ehr_cases"], 1)
        self.assertEqual(result["replay_trials"], 5)
        self.assertEqual(result["random_replicates_per_case"], 5)

    def test_missing_secondary_metric_remains_na_not_zero(self):
        bank, _, _ = invented_bank()
        for candidate in bank.values(): candidate["score_record"]["secondary_scores"]["biovil_report_cxr"].update(raw_cosine=None, status="not_available")
        result = cli.compare([replay.replay_case(bank, policy(), "fixed", 4)])[0]
        self.assertIsNone(result["biovil_mean_secondary_not_routing"])
        self.assertEqual(result["biovil_available_trials"], 0)

    def test_secondary_score_metadata_cannot_fake_calibration_or_missingness(self):
        bank, _, _ = invented_bank()
        candidate = next(iter(bank.values()))
        candidate["score_record"]["secondary_scores"]["biovil_report_cxr"]["calibrated"] = True
        with self.assertRaisesRegex(ValueError, "secondary BioViL"): replay.evaluation_snapshot(candidate)
        candidate["score_record"]["secondary_scores"]["biovil_report_cxr"].update(calibrated=False, status="not_available")
        with self.assertRaisesRegex(ValueError, "secondary BioViL"): replay.evaluation_snapshot(candidate)

    def test_cli_refuses_before_protected_read_without_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "load_cached_facts") as load:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            load.assert_not_called()

    def test_output_snapshot_contains_no_patient_or_report_body(self):
        bank, _, _ = invented_bank()
        data = json.dumps(replay.replay_case(bank, policy(), "targeted_heuristic", 20))
        for name in ("subject_id", "patient_id", "image_path", "report_text", "source_statement"):
            self.assertNotIn(name, data)


if __name__ == "__main__":
    unittest.main()

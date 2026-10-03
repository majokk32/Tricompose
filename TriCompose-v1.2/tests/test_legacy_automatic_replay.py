"""Invented legacy vectors; these checks are not clinical validation."""
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
from tricompose_v12 import legacy_replay_adapter as adapter
from tricompose_v12 import automatic_replay as replay
import run_legacy_automatic_replay as cli


def policy():
    return json.loads((ROOT / "configs/automatic_replay_pool80_v1.json").read_text())


def invented(ehr_states=("positive", "unknown", "uncertain"), *, no_finding=False):
    p = policy(); scores, images, reports = [], [], []
    def h(value):
        return hashlib.sha256(value.encode()).hexdigest()
    def vec(state):
        return {name: state if name == "edema" else "unknown" for name in adapter.FINDINGS}
    for k, ehr_state in enumerate(ehr_states):
        case = f"invented_case_{k}"
        for i, (model, seed) in enumerate(p["image_order"]):
            iid = f"invented_image_{k}_{i}"; ehr, image = vec(ehr_state), vec("positive")
            images.append({"case_id": case, "cxr_candidate_id": iid, "cxr_model_id": model,
                "image_sha256": h(iid), "ehr_finding_states": ehr,
                "cxr_finding_states": image, "weak_clinical_evidence": {"rule_count": 1}})
            for j, rmodel in enumerate(p["report_order"]):
                rid = f"invented_report_{k}_{i}_{j}"
                report = vec("unknown" if no_finding and j == 0 else "negative" if j == 0 else "positive")
                if no_finding and j == 0:
                    report["no_finding"] = "positive"
                edges = adapter.metric_counts(ehr, image, report)
                reports.append({"case_id": case, "report_candidate_id": rid,
                    "report_model_id": rmodel, "source_cxr_model_id": model,
                    "report_sha256": h(rid), "ehr_finding_states": copy.deepcopy(ehr),
                    "report_finding_states": report,
                    "ehr_finding_states_by_source": {"diagnosis": copy.deepcopy(ehr)}})
                known = sum(v["known_reference_fact_count"] for v in edges.values())
                support = sum(v["support_count"] for v in edges.values())
                opposition = sum(v["contradiction_count"] for v in edges.values())
                scores.append({"schema_version": "tricompose-edge-specific-selection-v1.1",
                    "case_id": case, "triple_candidate_id": f"invented_triple_{k}_{i}_{j}",
                    "lineage": {"ehr_sha256": h(case), "ehr_facts_sha256": h(case + "facts"),
                        "cxr_sha256": h(iid), "report_sha256": h(rid), "cxr_candidate_id": iid,
                        "report_candidate_id": rid, "cxr_model_id": model, "cxr_seed": seed,
                        "report_model_id": rmodel},
                    "scoring": {"selection": {"hard_gate_failure_count": 0, "selected": False,
                        "selection_rank_within_case": i * 4 + j + 1},
                        "modality_quality": {"cxr_basic_validity_pass": True,
                            "report_structure_quality_score_0_1": 0.8},
                        "clinical_totals": {"total_hard_contradiction_count": opposition,
                            "total_direct_support_count": support,
                            "total_known_reference_fact_count": known,
                            "ehr_direct_support_count": sum(edges[n]["support_count"] for n in ("ehr_cxr", "ehr_report")),
                            "clinical_balance_score_0_100": 50 * (1 + (support - opposition) / known) if known else None},
                        "edge_metrics": {name: {**v, "support_recall": v["support_count"] / v["known_reference_fact_count"] if v["known_reference_fact_count"] else None}
                                         for name, v in edges.items()},
                        "cost": {"known_runtime_seconds": 20 + i + j},
                        "weak_ehr_evidence": {"weak_support_count": 1}}})
    details = {"schema_version": "tricompose-ehr-edge-crossmodal-evaluation-v1.1",
        "evaluation_scope": {"cohort": "fully_synthetic_cold_start_non_longitudinal",
            "real_reference_supplied": False, "unknown_is_negative": False, "weak_prior_is_hard_label": False},
        "counts": {"cases": len(ehr_states), "cxr_candidates": len(images), "report_candidates": len(reports)},
        "records": {"ehr_cxr": images, "ehr_report": reports}}
    return scores, details


class LegacyReplayTests(unittest.TestCase):
    def bank(self, *args, **kwargs):
        scores, details = invented(*args, **kwargs)
        return adapter.make_legacy_bank(scores, details, policy())

    def test_complete_three_image_four_report_grid_keeps_all_cases(self):
        bank = self.bank()
        self.assertEqual(len(bank), 3)
        self.assertTrue(all(len(v) == 12 for v in bank.values()))
        self.assertTrue(all(len(c["facts"]) == 14 for v in bank.values() for c in v.values()))

    def test_explicit_and_unknown_uncertain_subgroups_not_enriched_by_weak_priors(self):
        summary = adapter.inventory_summary(self.bank())
        self.assertEqual(summary["case_counts_by_ehr_evidence"], {
            "explicit_fact_proxy": 1, "no_direct_comparable_ehr_facts": 2})
        self.assertEqual(summary["ehr_finding_state_counts_one_vector_per_case"]["edema"]["unknown"], 1)
        self.assertFalse(summary["weak_priors_used_for_hard_routing"])

    def test_legacy_global_normal_metric_never_flips_routing_unknown_state(self):
        grid = self.bank(("positive",), no_finding=True)["invented_case_0"]
        candidate = grid[("chexgenbench_sana", 0, "maira2")]
        edema = next(f for f in candidate["facts"] if f["finding"] == "edema")
        self.assertEqual(edema["states"]["chexbert"], "unknown")
        self.assertEqual(edema["relations"]["raw"]["ehr_report"], "unknown")
        self.assertEqual(candidate["score_record"]["scoring"]["edge_metrics"]["ehr_report"]["contradiction_count"], 1)
        self.assertEqual(replay.action_signal(candidate)[0], "switch_report_model")

    def test_missing_biovil_and_guarded_metrics_stay_na(self):
        grid = self.bank()["invented_case_0"]
        row = replay.replay_case(grid, policy(), "fixed", 4)
        snapshot = row["selected_snapshot"]
        self.assertIsNone(snapshot["biovil_cosine_secondary_not_routing"])
        self.assertEqual(snapshot["guarded_edge_readouts_not_routing"], {
            "ehr_cxr": None, "ehr_report": None, "cxr_report": None})
        result = cli.enriched_compare([row])[0]
        self.assertIsNone(result["biovil_mean_secondary_not_routing"])
        self.assertEqual(result["biovil_available_trials"], 0)
        self.assertIsNone(result["guarded_edge_readouts"]["ehr_report"])

    def test_unknown_ehr_rates_are_na_not_perfect_agreement(self):
        grid = self.bank(("unknown",))["invented_case_0"]
        row = replay.replay_case(grid, policy(), "fixed", 4)
        result = cli.enriched_compare([row])[0]
        self.assertIsNone(result["source_edge_metrics_optimization_proxy"]["ehr_cxr"]["support_recall"])
        self.assertIsNone(result["source_edge_metrics_optimization_proxy"]["ehr_report"]["opposition_rate"])

    def test_maximum_static_cost_charges_all_generators_and_scorers(self):
        grid = self.bank()["invented_case_0"]
        row = replay.replay_case(grid, policy(), "static_rerank", 30)
        self.assertEqual(row["simulated_model_calls"], 30)
        self.assertEqual(row["simulated_calls"], {"cxr_generator": 3, "xrv": 3, "report_generator": 12, "chexbert": 12})

    def test_report_switch_reuses_image_and_cost_bounded_for_every_seed(self):
        grid = self.bank()["invented_case_0"]
        row = replay.replay_case(grid, policy(), "targeted_heuristic", 8)
        self.assertEqual(row["simulated_model_calls"], 6)
        self.assertEqual(row["observed_images"], 1)
        self.assertEqual(row["action_trace"][1]["action"], "switch_report_model")
        for method in replay.METHODS:
            for budget in (0, 3, 4, 8, 12, 20, 30):
                for seed in policy()["random_seeds"]:
                    self.assertLessEqual(replay.replay_case(grid, policy(), method, budget, random_seed=seed)["simulated_model_calls"], budget)

    def test_determinism_and_source_immutability(self):
        scores, details = invented(); before = copy.deepcopy((scores, details))
        bank = adapter.make_legacy_bank(scores, details, policy())
        grid = bank["invented_case_0"]
        self.assertEqual(replay.replay_case(grid, policy(), "random", 20, random_seed=4),
                         replay.replay_case(dict(reversed(list(grid.items()))), policy(), "random", 20, random_seed=4))
        self.assertEqual(before, (scores, details))

    def test_model_inventory_and_profiles_cannot_be_mixed(self):
        scores, details = invented(); p = policy()
        p["routing_evidence"] = "existing_eight_raw_xrv_chexbert_finding_states"
        with self.assertRaises(ValueError): adapter.make_legacy_bank(scores, details, p)
        with self.assertRaises(ValueError): replay.make_bank(scores, [], policy())

    def test_missing_candidates_or_extra_edge_evidence_rejected(self):
        scores, details = invented()
        with self.assertRaises(ValueError): adapter.make_legacy_bank(scores[:-1], details, policy())
        changed = copy.deepcopy(details)
        changed["records"]["ehr_cxr"].append(copy.deepcopy(changed["records"]["ehr_cxr"][0]))
        with self.assertRaises(ValueError): adapter.make_legacy_bank(scores, changed, policy())

    def test_hash_model_and_fixed_ehr_mismatch_rejected(self):
        for field, value in (("cxr_sha256", "a" * 64), ("report_sha256", "b" * 64),
                             ("ehr_sha256", "c" * 64), ("cxr_model_id", "wrong_model"),
                             ("ehr_facts_sha256", "not_a_hash")):
            scores, details = invented(); scores[1]["lineage"][field] = value
            with self.assertRaises(ValueError): adapter.make_legacy_bank(scores, details, policy())

    def test_cached_states_source_categories_and_shared_report_must_agree(self):
        for kind in ("ehr_edge", "source_category", "shared_report"):
            scores, details = invented(); r = details["records"]["ehr_report"][1]
            if kind == "ehr_edge": r["ehr_finding_states"]["edema"] = "unknown"
            elif kind == "source_category": r["ehr_finding_states_by_source"]["diagnosis"]["edema"] = "unknown"
            else: r["report_sha256"] = scores[0]["lineage"]["report_sha256"]; scores[1]["lineage"]["report_sha256"] = r["report_sha256"]
            with self.assertRaises(ValueError): adapter.make_legacy_bank(scores, details, policy())

    def test_invalid_count_or_false_balance_or_cohort_count_rejected(self):
        for kind in ("edge", "total", "balance", "cases"):
            scores, details = invented()
            if kind == "edge": scores[0]["scoring"]["edge_metrics"]["ehr_cxr"]["support_count"] += 1
            elif kind == "total": scores[0]["scoring"]["clinical_totals"]["total_direct_support_count"] += 1
            elif kind == "balance": scores[0]["scoring"]["clinical_totals"]["clinical_balance_score_0_100"] = 100
            else: details["counts"]["cases"] += 1
            with self.assertRaises(ValueError): adapter.make_legacy_bank(scores, details, policy())

    def test_real_reference_unknown_negative_and_weak_promotion_scopes_rejected(self):
        for field in ("real_reference_supplied", "unknown_is_negative", "weak_prior_is_hard_label"):
            scores, details = invented(); details["evaluation_scope"][field] = True
            with self.assertRaises(ValueError): adapter.make_legacy_bank(scores, details, policy())

    def test_incomplete_or_invalid_state_vector_rejected(self):
        for value in ({"edema": "positive"}, dict.fromkeys(adapter.FINDINGS, "absent")):
            with self.assertRaises(ValueError): adapter.vector(value)

    def test_cli_refuses_before_any_protected_read_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "checked_source") as source:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            source.assert_not_called()

    def test_output_has_no_raw_artifact_body_or_patient_identifier(self):
        grid = self.bank()["invented_case_0"]
        data = json.dumps(replay.replay_case(grid, policy(), "targeted_heuristic", 20))
        for name in ("subject_id", "patient_id", "report_text", "image_path", "source_statement"):
            self.assertNotIn(name, data)


if __name__ == "__main__":
    unittest.main()

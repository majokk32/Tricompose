"""Invented states and metadata: no patient data, images, weights or API."""
import argparse
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import bounded_regeneration as retry
from tricompose_v12.invariant_verification import EHRAnchor, _digest
from tricompose_v12.legacy_replay_adapter import FINDINGS
from test_full_pool_report_control import rows, reseal
from run_bounded_regeneration import run, tables
from prepare_bounded_regeneration import prepare


def examples():
    base = next(r for r in rows() if r["cxr_model_id"] == "roentgen_v2" and r["report_model_id"] == "cxrmate_single")
    for f in base["receipt"]["fact_states"]:
        f.update(ehr="unknown", xrv="unknown", chexbert="unknown")
        if f["finding"] == "edema":
            f.update(ehr="positive", xrv="negative", chexbert="negative")
    reseal(base)
    alt = copy.deepcopy(base)
    alt.update(cxr_candidate_id="fixture_new_image", cxr_sha256=_digest("invented_new_image"),
        report_candidate_id="fixture_new_report", report_sha256=_digest("invented_new_report"),
        triple_candidate_id="fixture_new_triple", seed=1)
    alt["receipt"].update({k: alt[k] for k in ("cxr_candidate_id", "cxr_sha256", "report_candidate_id", "report_sha256")})
    alt["structure"].update(report_candidate_id=alt["report_candidate_id"], report_sha256=alt["report_sha256"],
        image_sha256=alt["cxr_sha256"], parent_cxr_candidate_id=alt["cxr_candidate_id"],
        normalized_report_sha256=alt["report_sha256"])
    for f in alt["receipt"]["fact_states"]:
        if f["finding"] == "edema":
            f.update(xrv="positive", chexbert="positive")
    reseal(alt)
    return base, alt


def anchors():
    result = []
    for index in range(80):
        items = tuple((name, "positive" if name == "edema" and index in (3, 8, 27) else "unknown",
                       ("invented_direct_category",) if name == "edema" and index in (3, 8, 27) else ()) for name in FINDINGS)
        anchor = EHRAnchor("fixture_anchor_" + f"{index:03d}", _digest([index, "ehr"]), _digest([index, "facts"]), items)
        result.append({"opaque_source_index": index, "anchor": anchor.record(), "ehr_anchor_sha256": anchor.sha256})
    return result


class BoundedRegenerationTests(unittest.TestCase):
    def test_exact_policy_cannot_be_loosened(self):
        retry.validate_policy(copy.deepcopy(retry.POLICY))
        for key, value in (("call_budget_per_case", 20), ("uses_biovil", True), ("max_retries_per_operation", 1),
                           ("retry_seed", 0), ("changes_existing_ehr_or_prompts", True)):
            policy = copy.deepcopy(retry.POLICY)
            policy[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                retry.validate_policy(policy)

    def test_case_selection_uses_only_ehr_and_head_availability(self):
        chosen, summary = retry.choose_cases(list(reversed(anchors())), {"edema"})
        self.assertEqual([r["opaque_source_index"] for r in chosen], [3, 8])
        self.assertEqual((summary["original_ehr_cases"], summary["enabled_head_ehr_cases"], summary["smoke_cases"]), (80, 3, 2))
        self.assertFalse(summary["case_selection_uses_outcomes"])

    def test_missing_enabled_fact_cases_never_get_padded(self):
        with self.assertRaises(ValueError):
            retry.choose_cases(anchors(), {"pneumonia"})
        with self.assertRaises(ValueError):
            retry.choose_cases(anchors()[:79], {"edema"})
        broken = anchors()
        broken[3]["ehr_anchor_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            retry.choose_cases(broken, {"edema"})

    def test_explicit_opposition_is_heuristic_not_clinical_localization(self):
        base, _ = examples()
        routed = retry.route(base)
        self.assertEqual(routed["action"], "regenerate_cxr")
        self.assertEqual(routed["trigger_finding_ids"], ["edema"])
        self.assertIsNone(routed["clinical_fault_location"])
        self.assertFalse(routed["correlated_reports_used_as_independent_votes"])

    def test_unknown_uncertain_abstain_and_do_not_become_negative(self):
        for state in ("unknown", "uncertain"):
            base, _ = examples()
            for fact in base["receipt"]["fact_states"]:
                fact["ehr"] = state
            reseal(base)
            self.assertEqual(retry.route(base)["action"], "abstain")
            self.assertFalse(retry.edge_sets(base)["ehr_cxr"]["opposition"])

    def test_missing_image_comparison_abstains(self):
        base, _ = examples()
        base["receipt"]["fact_states"][3]["xrv"] = "unknown"
        reseal(base)
        self.assertEqual(retry.route(base)["action"], "abstain")

    def test_agreement_stops_without_new_generation(self):
        _, alt = examples()
        self.assertEqual(retry.route(alt)["action"], "stop")

    def test_seed_only_request_clone_keeps_final_prompt_and_original(self):
        original = {"model_id": "roentgen_v2", "seed": 0, "case_id": "fixture_case",
                    "request_id": "fixture_original", "inputs": {"final_prompt": {"sha256": _digest("invented text")},
                    "synthetic_ehr": {"sha256": _digest("invented EHR")}}}
        before = copy.deepcopy(original)
        result = retry.alternate_request(original)
        self.assertEqual(original, before)
        self.assertEqual(result["inputs"], original["inputs"])
        self.assertEqual(result["seed"], 1)
        self.assertNotEqual(result["request_id"], original["request_id"])
        for key, value in (("seed", False), ("seed", 1), ("model_id", "ehrxdiff")):
            bad = copy.deepcopy(original)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                retry.alternate_request(bad)

    def test_actual_ehr_image_gain_can_pass_but_not_clinical_acceptance(self):
        base, alt = examples()
        compared = retry.compare(base, alt)
        self.assertTrue(compared["exploratory_gate_pass"])
        self.assertFalse(compared["clinical_repair_success"])
        self.assertEqual(compared["removed_opposition_fact_ids"]["ehr_cxr"], ["edema"])
        self.assertEqual(retry.decision(base, base, alt)["selected_triple_id"], alt["triple_candidate_id"])

    def test_cannot_silence_conflict_by_unknown_or_uncertain(self):
        for state in ("unknown", "uncertain"):
            base, alt = examples()
            alt["receipt"]["fact_states"][3].update(xrv=state, chexbert=state)
            reseal(alt)
            compared = retry.compare(base, alt)
            self.assertFalse(compared["exploratory_gate_pass"])
            self.assertEqual(compared["lost_fact_ids"]["ehr_cxr"]["comparable"], ["edema"])

    def test_new_opposition_on_any_edge_blocks_retry(self):
        for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
            base, alt = examples()
            for row in (base, alt):
                row["receipt"]["fact_states"][0].update(ehr="positive", xrv="positive", chexbert="positive")
            alt["receipt"]["fact_states"][0]["xrv" if edge == "ehr_cxr" else "chexbert"] = "negative"
            for row in (base, alt): reseal(row)
            with self.subTest(edge=edge):
                self.assertFalse(retry.compare(base, alt)["exploratory_gate_pass"])

    def test_previous_positive_image_report_support_cannot_disappear(self):
        base, alt = examples()
        base["receipt"]["fact_states"][0].update(xrv="positive", chexbert="positive")
        alt["receipt"]["fact_states"][0].update(xrv="negative", chexbert="negative")
        for row in (base, alt): reseal(row)
        self.assertFalse(retry.compare(base, alt)["exploratory_gate_pass"])

    def test_structure_and_temporal_flags_cannot_worsen(self):
        for key, value in (("generic_report", True), ("unsupported_temporal_comparison_language", True),
                           ("repeated_sentence_count", 1), ("repeated_4gram_ratio", .1)):
            base, alt = examples()
            alt["structure"][key] = value
            with self.subTest(key=key):
                self.assertFalse(retry.compare(base, alt)["exploratory_gate_pass"])

    def test_identical_image_does_not_get_diversity_credit(self):
        base, alt = examples()
        alt["cxr_sha256"] = base["cxr_sha256"]
        alt["receipt"]["cxr_sha256"] = alt["cxr_sha256"]
        alt["structure"]["image_sha256"] = alt["cxr_sha256"]
        reseal(alt)
        self.assertFalse(retry.compare(base, alt)["exploratory_gate_pass"])

    def test_anchor_or_profile_change_is_error(self):
        for key in ("ehr_anchor_sha256", "thresholds_sha256", "xrv_checkpoint_sha256", "chexbert_checkpoint_sha256"):
            base, alt = examples()
            alt["receipt"][key] = "0" * 64
            reseal(alt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                retry.compare(base, alt)

    def test_raw_counts_cannot_disagree_with_receipt(self):
        base, _ = examples()
        base["raw_edge_readouts"]["ehr_cxr"]["comparable_facts"] += 1
        with self.assertRaises(ValueError): retry.route(base)

    def test_missing_or_failed_retry_retains_static_not_replaces_ehr(self):
        base, _ = examples()
        choice = retry.decision(base, base, failure_reason="bounded_generation_or_verification_incomplete")
        self.assertEqual(choice["selected_triple_id"], base["triple_candidate_id"])
        self.assertEqual(choice["status"], "unresolved_static_retained")
        self.assertFalse(choice["original_winners_changed"])

    def test_retry_must_beat_static_not_only_weak_fixed_baseline(self):
        base, alt = examples()
        static = copy.deepcopy(base)
        static.update(report_candidate_id="fixture_static_report", report_sha256=_digest("invented static report"),
                      triple_candidate_id="fixture_static_triple")
        static["receipt"].update(report_candidate_id=static["report_candidate_id"], report_sha256=static["report_sha256"])
        static["structure"].update(report_candidate_id=static["report_candidate_id"], report_sha256=static["report_sha256"],
                                   normalized_report_sha256=static["report_sha256"])
        static["receipt"]["fact_states"][0].update(xrv="positive", chexbert="positive")
        # The frozen image is shared, so the same reference is added to base.
        base["receipt"]["fact_states"][0].update(xrv="positive", chexbert="unknown")
        alt["receipt"]["fact_states"][0].update(xrv="unknown", chexbert="unknown")
        for row in (base, static, alt): reseal(row)
        self.assertTrue(retry.compare(base, alt)["exploratory_gate_pass"])
        choice = retry.decision(base, static, alt)
        self.assertEqual(choice["selected_triple_id"], static["triple_candidate_id"])

    def test_static_anchor_drift_rejected_even_without_retry(self):
        base, _ = examples()
        static = copy.deepcopy(base)
        static["ehr_sha256"] = "0" * 64
        with self.assertRaises(ValueError): retry.decision(base, static)

    def test_gpu_guard_precedes_input_or_directory_access(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "invented_cpu", "CUDA_VISIBLE_DEVICES": ""}, clear=True), \
             patch("run_bounded_regeneration.load") as loader:
            with self.assertRaises(RuntimeError): run(argparse.Namespace())
            loader.assert_not_called()

    def test_preflight_guard_precedes_checkpoint_or_source_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch("prepare_bounded_regeneration.source_pins") as pins:
            with self.assertRaises(RuntimeError): prepare(argparse.Namespace())
            pins.assert_not_called()

    def test_selection_is_sealed_before_secondary_and_budgets_count_failures(self):
        text = (ROOT / "benchmarks/run_bounded_regeneration.py").read_text()
        self.assertLess(text.index('selection_path = write_private_json'), text.index('endpoint = secondary('))
        self.assertIn('"failed_charged_not_free"', text)
        self.assertIn('"automatic_resume": False', text)
        self.assertIn('sum(b["charged_model_attempts"] for b in books) > 8', text)
        self.assertNotIn('model.generate(', text)
        self.assertNotIn('read_report_text(', text)

    def test_tables_preserve_secondary_na_and_raw_edges(self):
        base, alt = examples()
        choice = retry.decision(base, base, alt)
        endpoint = {"historical_pool_is_untouched_test": False, "used_for_routing": False,
                    "clinical_truth_available": False, "records": []}
        fields = ("case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
                  "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
        for row in (base, alt):
            endpoint["records"].append({k: row[k] for k in fields} | {"biovil_raw_cosine": None,
                "status": "not_available", "reason": "invented_context_limit", "calibrated": False})
        score_table, comparison = tables([base, alt], [choice], endpoint)
        self.assertTrue(all(r["biovil_raw_cosine"] == "NA" for r in score_table))
        self.assertEqual(comparison[0]["bounded_retry_biovil_raw_cosine"], "NA")
        self.assertEqual(comparison[0]["bounded_retry_ehr_cxr_supported_positive"], 1)


if __name__ == "__main__":
    unittest.main()

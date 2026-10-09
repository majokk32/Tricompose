"""Invented full-pool metadata only: no weights, reports, pixels or API."""
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
from tricompose_v12 import full_pool_report_control as pool
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.invariant_verification import _edge
from test_report_expert_control import rows as expert_rows
from test_report_nbest import seal
from run_full_pool_report_control import run


def rows(cases=2):
    templates = expert_rows()[:4]
    result = []
    for case_index in range(cases):
        case = "fixture_pool_" + str(case_index)
        for cxr in pool.CXR_MODELS:
            iid = "fixture_image_" + _digest([case, cxr])[:24]
            image_sha = _digest(["invented_image", iid])
            for template in templates:
                row = copy.deepcopy(template)
                model = row["report_model_id"]
                rid = "fixture_report_" + _digest([iid, model])[:24]
                report_sha = _digest(["invented_report", rid])
                row.update(case_id=case, cxr_candidate_id=iid, cxr_sha256=image_sha,
                    report_candidate_id=rid, report_sha256=report_sha, cxr_model_id=cxr, seed=0,
                    ehr_sha256=_digest([case, "ehr"]), ehr_facts_sha256=_digest([case, "facts"]),
                    triple_candidate_id="fixture_triple_" + _digest([iid, rid])[:24])
                row["receipt"].update(case_id=case, cxr_candidate_id=iid, cxr_sha256=image_sha,
                    report_candidate_id=rid, report_sha256=report_sha, ehr_anchor_sha256=_digest([case, "anchor"]))
                seal(row["receipt"])
                row["structure"].update(case_id=case, report_candidate_id=rid, report_sha256=report_sha,
                    image_sha256=image_sha, normalized_report_sha256=report_sha,
                    source_cxr_model_id=cxr, parent_cxr_candidate_id=iid)
                reseal(row)
                result.append(row)
    return result


def reseal(row):
    states = row["receipt"]["fact_states"]
    row["receipt"]["raw_edge_readouts"] = {"ehr_cxr": _edge(states, "ehr", "xrv"),
        "ehr_report": _edge(states, "ehr", "chexbert"), "cxr_report": _edge(states, "xrv", "chexbert")}
    row["receipt"]["known_ehr_facts"] = row["receipt"]["raw_edge_readouts"]["ehr_cxr"]["known_reference_facts"]
    seal(row["receipt"])
    row["raw_edge_readouts"] = copy.deepcopy(row["receipt"]["raw_edge_readouts"])


def endpoint(rs):
    return {"used_for_routing": False, "clinical_truth_available": False, "historical_pool_is_untouched_test": False,
        "records": [{k: row[k] for k in pool.PAIR_FIELDS} | {
            "biovil_raw_cosine": .9 if row["report_model_id"] == "cxrmate_single" else .1,
            "status": "computed_secondary_uncalibrated", "reason": None, "calibrated": False} for row in rs]}


class FullPoolReportTests(unittest.TestCase):
    def summarize(self, rs, ep=None):
        return pool.summarize(rs, pool.freeze(rs, expected_cases=2), endpoint(rs) if ep is None else ep, expected_cases=2)

    def test_default_requires_whole_eighty_case_pool(self):
        with self.assertRaises(ValueError):
            pool.freeze(rows())
        rs = rows(80)
        selection = pool.freeze(rs)
        self.assertEqual((selection["fixed_ehr_cases"], selection["fixed_images"], selection["report_candidates"]), (80, 240, 960))
        self.assertEqual(len(selection["comparisons"]), 720)

    def test_priority_is_unchanged_first_eligible_not_cosine(self):
        rs = rows()
        selection = pool.freeze(rs, expected_cases=2)
        self.assertTrue(all(c["selected_model"] == "maira2" for c in selection["choices"]))
        result = self.summarize(rs)
        self.assertTrue(all(p["delta"] < 0 for p in result["image_pairs"]))
        self.assertAlmostEqual(result["paired_case_bootstraps"][-1]["paired_case_bootstrap"]["estimate_method_minus_baseline"], -.8)
        self.assertFalse(result["summary"]["clinical_repair_success"])

    def test_four_fixed_paths_and_selector_for_each_image_generator(self):
        result = self.summarize(rows())
        self.assertEqual(len(result["model_comparison"]), 15)
        self.assertEqual(len(result["image_pairs"]), 6)
        self.assertEqual(len(result["case_outcomes"]), 8)
        self.assertTrue(all(r["fixed_ehr_cases"] == 2 for r in result["model_comparison"]))

    def test_bootstrap_unit_is_ehr_not_image_or_report(self):
        result = self.summarize(rows())
        for item in result["paired_case_bootstraps"]:
            bootstrap = item["paired_case_bootstrap"]
            self.assertEqual(bootstrap["input_cases"], 2)
            self.assertEqual(bootstrap["independent_groups"], 2)
            self.assertEqual((bootstrap["seed"], bootstrap["repetitions"]), (0, 2000))

    def test_empty_comparable_ehr_rates_remain_na(self):
        rs = rows()
        for row in rs:
            for fact in row["receipt"]["fact_states"]:
                fact["ehr"] = "unknown"
            states = row["receipt"]["fact_states"]
            row["receipt"]["raw_edge_readouts"].update(ehr_cxr=_edge(states, "ehr", "xrv"), ehr_report=_edge(states, "ehr", "chexbert"))
            row["receipt"]["known_ehr_facts"] = 0
            seal(row["receipt"])
            row["raw_edge_readouts"] = copy.deepcopy(row["receipt"]["raw_edge_readouts"])
        result = self.summarize(rs)
        self.assertEqual(result["summary"]["no_direct_ehr_cases"], 2)
        for method in result["model_comparison"]:
            for edge in ("ehr_cxr", "ehr_report"):
                self.assertIsNone(method["raw_edge_totals"][edge]["coverage_over_known"])
                self.assertIsNone(method["raw_edge_totals"][edge]["support_over_known"])

    def test_incomplete_duplicate_and_wrong_model_grids_rejected(self):
        rs = rows()
        for broken in (rs[:-1], rs + [rs[0]], rs[:-1] + [rs[0]]):
            with self.assertRaises(ValueError):
                pool.freeze(broken, expected_cases=2)
        for field, value in (("cxr_model_id", "undeclared"), ("report_model_id", "undeclared"), ("seed", 1), ("seed", False)):
            broken = copy.deepcopy(rs)
            broken[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                pool.freeze(broken, expected_cases=2)

    def test_fixed_ehr_cannot_change_between_images(self):
        for field in ("ehr_sha256", "ehr_facts_sha256"):
            rs = rows()
            rs[4][field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                pool.freeze(rs, expected_cases=2)

    def test_legacy_or_changed_profiles_rejected(self):
        for key in ("profile", "thresholds_sha256", "xrv_checkpoint_sha256", "chexbert_checkpoint_sha256"):
            rs = rows()
            rs[0]["receipt"][key] = "changed" if key == "profile" else "0" * 64
            seal(rs[0]["receipt"])
            with self.subTest(key=key), self.assertRaises(ValueError):
                pool.freeze(rs, expected_cases=2)

    def test_raw_score_rows_cannot_disagree_with_receipts(self):
        rs = rows()
        rs[0]["raw_edge_readouts"]["cxr_report"]["comparable_facts"] += 1
        with self.assertRaises(ValueError):
            pool.freeze(rs, expected_cases=2)

    def test_common_risk_increase_cannot_pass(self):
        rs = rows()
        for row in rs:
            if row["report_model_id"] != "cxrmate_single":
                row["structure"]["unsupported_temporal_comparison_language"] = True
        selection = pool.freeze(rs, expected_cases=2)
        self.assertTrue(all(c["status"] == "unresolved_baseline_retained" for c in selection["choices"]))

    def test_new_opposition_rejected_despite_other_evidence_improvement(self):
        rs = rows()
        for row in rs:
            fact = row["receipt"]["fact_states"][0]
            fact["xrv"] = "positive"
            fact["chexbert"] = "positive" if row["report_model_id"] == "cxrmate_single" else "negative"
            reseal(row)
        selection = pool.freeze(rs, expected_cases=2)
        self.assertTrue(all(not c["exploratory_gate_pass"] for c in selection["comparisons"]))
        self.assertTrue(all("new_explicit_proxy_opposition" in c["reasons"] for c in selection["comparisons"]))

    def test_self_digest_cannot_make_invented_edge_counts_valid(self):
        rs = rows()
        rs[0]["receipt"]["raw_edge_readouts"]["cxr_report"]["comparable_facts"] += 1
        seal(rs[0]["receipt"])
        rs[0]["raw_edge_readouts"] = copy.deepcopy(rs[0]["receipt"]["raw_edge_readouts"])
        with self.assertRaises(ValueError):
            pool.freeze(rs, expected_cases=2)

    def test_selection_cannot_change_after_secondary_measurement(self):
        rs = rows()
        selection = pool.freeze(rs, expected_cases=2)
        selection["used_biovil"] = True
        with self.assertRaises(ValueError):
            pool.summarize(rs, selection, endpoint(rs), expected_cases=2)

    def test_secondary_requires_exact_inventory_and_lineage(self):
        rs = rows()
        for key, value in (("report_sha256", "0" * 64), ("calibrated", True), ("biovil_raw_cosine", float("nan")),
                           ("status", "invented")):
            ep = endpoint(rs)
            ep["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summarize(rs, ep)
        ep = endpoint(rs)
        ep["records"].pop()
        with self.assertRaises(ValueError):
            self.summarize(rs, ep)

    def test_previous_pool_cannot_be_called_untouched_holdout(self):
        rs = rows()
        ep = endpoint(rs)
        ep["historical_pool_is_untouched_test"] = True
        with self.assertRaises(ValueError):
            self.summarize(rs, ep)

    def test_na_never_becomes_zero_or_cherry_picked_three_image_mean(self):
        rs = rows()
        ep = endpoint(rs)
        ep["records"][0].update(biovil_raw_cosine=None, status="not_available", reason="full_context_limit")
        result = self.summarize(rs, ep)
        pooled = result["paired_case_bootstraps"][-1]["paired_case_bootstrap"]
        self.assertEqual((pooled["input_cases"], pooled["missing_pairs"], pooled["comparable_cases"]), (2, 1, 1))
        self.assertIsNone(pooled["confidence_interval"])
        self.assertEqual(result["summary"]["available_secondary_pairs"], 23)

    def test_exact_secondary_na_status_required(self):
        rs = rows()
        ep = endpoint(rs)
        ep["records"][0]["biovil_raw_cosine"] = None
        with self.assertRaises(ValueError):
            self.summarize(rs, ep)

    def test_gpu_guard_precedes_private_inputs_or_output_creation(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture_cpu", "CUDA_VISIBLE_DEVICES": ""}, clear=True), \
                patch("run_full_pool_report_control.load") as loader:
            with self.assertRaises(RuntimeError):
                run(argparse.Namespace())
            loader.assert_not_called()

    def test_cached_endpoint_values_are_loaded_after_selection(self):
        source = (ROOT / "benchmarks/run_full_pool_report_control.py").read_text()
        self.assertLess(source.index("selection = freeze(rows)"), source.index('cached = read_json(plan["cached_secondary_path"])'))
        self.assertNotIn("model.generate(", source)
        self.assertNotIn("_load_runtime", source)
        self.assertIn('"new_generation_calls": 0', source)
        self.assertIn('"automatic_resume": False', source)


if __name__ == "__main__":
    unittest.main()

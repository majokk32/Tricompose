"""Synthetic in-memory end-to-end evidence merge and selection tests."""
import argparse
import copy
import json
import unittest
from pathlib import Path
from unittest import mock

import finalize_unified_score_table as finalize
import select_edge_specific_candidates as selector
from contracts import CHEXPERT_FINDINGS
from crossmodal_metrics import score_state_pair
from test_candidate_grid import fixture


def bundle():
    rows, grid = fixture()
    unknown = dict.fromkeys(CHEXPERT_FINDINGS, "unknown")
    image = {**unknown, "edema": "negative", "cardiomegaly": "positive"}
    neutral = score_state_pair(unknown, image)
    cxrs, reports, crossmodal = {}, [], []
    weak = {"rule_count": 0, "weak_support_count": 0, "weak_incompatibility_count": 0, "unknown_count": 0}
    for row in rows:
        lineage = row["lineage"]
        lineage.update(cxr_sha256="c" * 64, report_sha256="r" * 64)
        row.update(schema_version="tricompose-unified-score-row-v1.1",
                   modality_quality={"ehr": {"schema_valid": True}, "cxr": {"corrupted": False, "blank": False},
                                     "report": {"generic_report": False, "unsupported_temporal_comparison_language": False, "repeated_4gram_ratio": 0}},
                   selection={"hard_gate_failures": [], "eligible": False},
                   cost={"known_runtime_seconds": 1, "known_model_calls": 2})
        cxrs[lineage["cxr_candidate_id"]] = {
            "cxr_candidate_id": lineage["cxr_candidate_id"], "case_id": row["case_id"],
            "cxr_model_id": lineage["cxr_model_id"], "image_sha256": lineage["cxr_sha256"],
            "ehr_finding_states": unknown, "cxr_finding_states": image, "weak_clinical_evidence": weak}
        reports.append({"report_candidate_id": lineage["report_candidate_id"], "case_id": row["case_id"],
                        "report_model_id": lineage["report_model_id"], "source_cxr_model_id": lineage["cxr_model_id"],
                        "report_sha256": lineage["report_sha256"], "hard_direct_metrics": neutral,
                        "weak_clinical_evidence": weak})
        crossmodal.append({"report_candidate_id": lineage["report_candidate_id"],
                           "parent_cxr_candidate_id": lineage["cxr_candidate_id"],
                           "report_sha256": lineage["report_sha256"], "image_sha256": lineage["cxr_sha256"],
                           "edge_metrics": {"ehr_cxr": neutral, "ehr_report": neutral, "report_cxr": score_state_pair(image, image)},
                           "secondary_scores": {"qwenvl_report_cxr": None, "biovil_report_cxr": 0.2},
                           "uncertainty": {}})
    summary = {"schema_version": finalize.SCHEMA_VERSION, "selection_ready": False,
               "cohort_contract": grid, "model_grid": {k: grid[k] for k in ("cxr_models", "report_models", "cxr_seeds")}}
    evidence = {"schema_version": finalize.CROSSMODAL_SCHEMA, "records": crossmodal,
                "primary_metric_status": "diagnostic_only_uncalibrated_cxr_labels"}
    return rows, summary, evidence, {"schema_version": selector.EHR_EDGE_SCHEMA,
                                      "records": {"ehr_cxr": list(cxrs.values()), "ehr_report": reports}}


def merged_bundle():
    rows, summary, evidence, edges = bundle()
    policy = json.loads((Path(__file__).parent / "static_score_policy_v1_1.json").read_text())
    with (mock.patch.object(finalize, "require_inside", side_effect=lambda path, *a, **kw: Path(path)),
          mock.patch.object(finalize, "read_json", side_effect=[summary, evidence]),
          mock.patch.object(finalize, "_read_jsonl", return_value=rows),
          mock.patch.object(finalize, "_load_policy", return_value=(policy, {})),
          mock.patch.object(finalize, "sha256_file", return_value="h" * 64)):
        final = finalize.run(argparse.Namespace(score_table_run="registry", crossmodal_run="crossmodal", policy_config="policy"))
    return final, edges


def select(final, edges, policy=None):
    if policy is None:
        policy = json.loads((Path(__file__).parent / "lexicographic_selection_policy_smoke48.json").read_text())
    with (mock.patch.object(selector, "require_inside", side_effect=lambda path, *a, **kw: Path(path)),
          mock.patch.object(selector, "read_json", side_effect=[final, edges]),
          mock.patch.object(selector, "_read_jsonl", return_value=copy.deepcopy(final["rows"])),
          mock.patch.object(selector, "_load_policy", return_value=(policy, {})),
          mock.patch.object(selector, "sha256_file", return_value="h" * 64)):
        return selector.run(argparse.Namespace(score_table_run="final", ehr_edge_run="edges", policy_config="policy"))


class SmallCohortSelectionTests(unittest.TestCase):
    def test_merge_and_select_two_cases_without_faking_ehr_agreement(self):
        final, edges = merged_bundle()
        self.assertEqual(final["counts"]["rows"], 48)
        self.assertFalse(final["paper_primary_selection_ready"])
        self.assertEqual(final["edge_applicability"]["ehr_cxr"]["not_applicable_rows"], 48)
        output = select(final, edges)
        self.assertEqual(output["counts"]["selected_triples"], 2)
        self.assertEqual(output["counts"]["fixed_paths"], 24)
        self.assertEqual(output["static_reranking"]["prospective_model_calls_per_case"], {"cxr": 6, "report": 24, "total": 30})
        self.assertIn("[seed=0]", output["baselines"]["operational_fixed"]["path"])
        for row in output["candidate_rows"]:
            edge = row["scoring"]["edge_metrics"]["ehr_report"]
            self.assertIsNone(edge["support_recall"])
            self.assertEqual(edge["status"], "not_applicable_no_comparable_ehr_facts")
            self.assertEqual(row["secondary_scores"]["biovil_report_cxr"]["raw_cosine"], 0.2)
        self.assertIn("biovil_raw_cosine_secondary", selector._csv(output["candidate_rows"]))

    def test_duplicate_edge_evidence_and_undeclared_seed_fail_closed(self):
        final, edges = merged_bundle()
        bad = copy.deepcopy(edges)
        bad["records"]["ehr_report"].append(bad["records"]["ehr_report"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select(final, bad)
        policy = json.loads((Path(__file__).parent / "lexicographic_selection_policy_v1_1_1.json").read_text())
        with self.assertRaisesRegex(ValueError, "predeclare"):
            select(final, edges, policy)

    def test_invalid_candidates_are_rejected_not_exported_as_best(self):
        final, edges = merged_bundle()
        for row in final["rows"]:
            row["selection"]["hard_gate_failures"] = ["blank_cxr"]
        result = select(final, edges)
        self.assertEqual(result["counts"]["selected_triples"], 0)
        self.assertEqual(len(result["static_reranking"]["rejected_cases"]), 2)
        self.assertTrue(all(not r["scoring"]["selection"]["selected"] for r in result["candidate_rows"]))
        selector._markdown(result)

    def test_label_inference_is_refused_before_importing_models_outside_slurm(self):
        import extract_report_labels_chexbert as chexbert
        import score_report_cxr_biovil as biovil
        with mock.patch.dict("os.environ", {}, clear=True):
            for module in (chexbert, biovil):
                with self.assertRaisesRegex(RuntimeError, "Slurm"):
                    module.run(argparse.Namespace())


if __name__ == "__main__":
    unittest.main()

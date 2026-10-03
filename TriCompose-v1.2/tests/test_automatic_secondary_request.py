"""Invented secondary requests, no models or artifact bodies opened."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import score_automatic_replay_biovil as cli
from test_legacy_automatic_replay import invented, policy
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from tricompose_v12.automatic_replay import replay_case


def fixture():
    scores, details = invented(("positive",))
    grid = make_legacy_bank(scores, details, policy())["invented_case_0"]
    rows = [replay_case(grid, policy(), method, 20, random_seed=seed)
            for method in ("fixed", "static_rerank", "targeted_heuristic", "random") for seed in (0, 1)]
    return rows, scores


def fake_artifacts(pairs):
    cxrs, reports = {}, {}
    for p in pairs:
        cxrs[p["cxr_candidate_id"]] = {"case_id": p["case_id"], "ehr_sha256": p["ehr_sha256"],
            "ehr_facts_sha256": p["ehr_facts_sha256"], "artifact": {"sha256": p["cxr_sha256"]}}
        reports[p["report_candidate_id"]] = {"case_id": p["case_id"],
            "parent_cxr_candidate_id": p["cxr_candidate_id"], "artifact": {"sha256": p["report_sha256"]}}
    return cxrs, reports


class SecondaryRequestTests(unittest.TestCase):
    def test_selected_union_is_unique_complete_and_has_no_policy_scores(self):
        rows, scores = fixture(); pairs = cli.requested_pairs(rows, scores)
        self.assertEqual({p["triple_candidate_id"] for p in pairs}, {r["selected_candidate_id"] for r in rows})
        self.assertTrue(all(set(p) == cli.PAIR_FIELDS for p in pairs))
        self.assertEqual(len(pairs), len({p["triple_candidate_id"] for p in pairs}))

    def test_union_is_score_and_input_order_independent_after_selection(self):
        rows, scores = fixture(); changed = copy.deepcopy(scores)
        for r in changed: r["scoring"] = {"arbitrary_unseen_score": -999}
        self.assertEqual(cli.requested_pairs(rows, scores), cli.requested_pairs(list(reversed(rows)), changed))

    def test_no_selection_is_not_a_fabricated_secondary_zero(self):
        rows, scores = fixture()
        for row in rows: row["selected_candidate_id"] = None
        with self.assertRaises(ValueError): cli.requested_pairs(rows, scores)

    def test_selected_hash_or_case_mismatch_rejected(self):
        for kind in ("hash", "case"):
            rows, scores = fixture()
            if kind == "hash": rows[0]["selected_snapshot"]["artifact_hashes"]["cxr_sha256"] = "a" * 64
            else: rows[0]["case_id"] = "different_invented_case"
            with self.assertRaises(ValueError): cli.requested_pairs(rows, scores)

    def test_valid_request_retains_real_parent_not_shuffled_pairs(self):
        rows, scores = fixture(); pairs = cli.requested_pairs(rows, scores)
        cxrs, reports = fake_artifacts(pairs)
        cli.validate_pairs(pairs, cxrs, reports)

    def test_wrong_hash_parent_case_or_ehr_rejected(self):
        for kind in ("hash", "parent", "case", "ehr"):
            rows, scores = fixture(); pairs = cli.requested_pairs(rows, scores)
            cxrs, reports = fake_artifacts(pairs); p = pairs[0]
            if kind == "hash": reports[p["report_candidate_id"]]["artifact"]["sha256"] = "a" * 64
            elif kind == "parent": reports[p["report_candidate_id"]]["parent_cxr_candidate_id"] = "another_image"
            elif kind == "case": cxrs[p["cxr_candidate_id"]]["case_id"] = "another_case"
            else: cxrs[p["cxr_candidate_id"]]["ehr_sha256"] = "b" * 64
            with self.assertRaises(ValueError): cli.validate_pairs(pairs, cxrs, reports)

    def test_duplicate_missing_or_extra_request_fields_rejected(self):
        rows, scores = fixture(); pairs = cli.requested_pairs(rows, scores)
        cxrs, reports = fake_artifacts(pairs)
        with self.assertRaises(ValueError): cli.validate_pairs([*pairs, pairs[0]], cxrs, reports)
        changed = copy.deepcopy(pairs); changed[0]["clinical_balance"] = 100
        with self.assertRaises(ValueError): cli.validate_pairs(changed, cxrs, reports)
        with self.assertRaises(ValueError): cli.validate_pairs(pairs, {}, reports)

    def test_cpu_prepare_and_gpu_score_refuse_before_reads_or_torch_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "require_inside") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.prepare(None)
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.score(None)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

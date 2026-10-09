"""Invented pair/hash fixtures; no models, reports, images or patient inputs."""
import argparse
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import complete_endpoint_inventory as e
from tricompose_v12.invariant_verification import _digest
import complete_bank_endpoint as cli


def row(i):
    return {"profile":e.PROFILE,"case_id":"fixture_case","triple_candidate_id":f"pair_{i}",
        "cxr_candidate_id":"fixture_image","report_candidate_id":f"report_{i}",
        "ehr_sha256":_digest("ehr"),"ehr_facts_sha256":_digest("facts"),
        "cxr_sha256":_digest("image"),"report_sha256":_digest(["report",i])}


def previous(i,value=.2):
    return {**e.pair(row(i)),"biovil_raw_cosine":value,"calibrated":False,
        "status":"computed_secondary_uncalibrated" if value is not None else "not_available",
        "reason":None if value is not None else "full_report_exceeds_text_context_no_truncation"}


class CompleteInventoryTests(unittest.TestCase):
    def test_request_exact_difference_not_score_selected(self):
        pending,counts=e.missing_pairs([row(0),row(1),row(2)],[previous(0)])
        self.assertEqual([r["triple_candidate_id"] for r in pending],["pair_1","pair_2"])
        self.assertEqual(counts["all_candidate_pairs"],3)
        self.assertFalse(counts["endpoint_values_used_to_choose_missing_pairs"])

    def test_prior_values_do_not_change_requested_inventory(self):
        values=[e.missing_pairs([row(0),row(1)],[previous(0,v)])[0] for v in (-.9,0,.9)]
        self.assertTrue(all(v==values[0] for v in values))

    def test_prior_na_not_retried_or_imputed(self):
        pending,counts=e.missing_pairs([row(0),row(1)],[previous(0,None)])
        self.assertEqual(len(pending),1);self.assertEqual(counts["old_na_endpoints_not_retried"],1)

    def test_order_independent_and_sources_unchanged(self):
        rows=[row(2),row(1),row(0)];old=[previous(0)];before=copy.deepcopy((rows,old))
        result=e.missing_pairs(rows,old)
        self.assertEqual(result,e.missing_pairs(list(reversed(rows)),old))
        self.assertEqual((rows,old),before)

    def test_hash_or_parent_changes_rejected(self):
        for key in e.PAIR_FIELDS:
            old=previous(0);old[key]=_digest("changed") if key.endswith("sha256") else "changed"
            with self.assertRaises(ValueError):e.missing_pairs([row(0)],[old])

    def test_duplicate_inventory_and_previous_rejected(self):
        with self.assertRaises(ValueError):e.missing_pairs([row(0),row(0)],[])
        with self.assertRaises(ValueError):e.missing_pairs([row(0)],[previous(0),previous(0)])

    def test_mixed_profile_rejected(self):
        r=row(0);r["profile"]="fresh_eight_heads"
        with self.assertRaises(ValueError):e.missing_pairs([r],[])

    def test_encoder_units_deduplicate_image_not_reports(self):
        _,counts=e.missing_pairs([row(0),row(1)],[])
        self.assertEqual(counts["maximum_new_image_encoder_samples"],1)
        self.assertEqual(counts["maximum_new_text_encoder_samples"],2)

    def test_complete_old_inventory_requests_nothing(self):
        pending,counts=e.missing_pairs([row(0)],[previous(0)])
        self.assertEqual(pending,[]);self.assertEqual(counts["missing_candidate_pairs"],0)

    def test_empty_or_unbounded_inventory_rejected(self):
        for rows in ([],[row(i) for i in range(961)]):
            with self.assertRaises(ValueError):e.missing_pairs(rows,[])

    def test_nan_bool_probability_and_missing_reason_rejected(self):
        for old in (previous(0,float("nan")),previous(0,True),previous(0,3),
                    {**previous(0,None),"reason":None},{**previous(0),"calibrated":True}):
            with self.assertRaises(ValueError):e.missing_pairs([row(0)],[old])

    def test_login_guard_before_any_read_or_model(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"prepare") as prepare:
            with self.assertRaises(RuntimeError):cli.execute(argparse.Namespace())
            prepare.assert_not_called()

    def test_existing_run_refused_before_work(self):
        args=argparse.Namespace(run_id="fixture",output_root="/unused",mode="score")
        with patch.dict(os.environ,{"SLURM_JOB_ID":"fixture"}),patch.object(cli,"require_inside") as inside,patch.object(cli,"prepare") as prepare:
            inside.return_value.exists.return_value=True
            with self.assertRaises(FileExistsError):cli.execute(args)
            prepare.assert_not_called()


if __name__=="__main__":unittest.main()

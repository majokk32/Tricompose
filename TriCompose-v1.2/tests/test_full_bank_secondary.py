"""Invented score records only; no real models or clinical artifact inspection."""
import argparse
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import full_bank_secondary as f
from tricompose_v12.complete_endpoint_inventory import missing_pairs
from test_complete_endpoint_inventory import row,previous
import merge_complete_bank_endpoint as cli

HASHES={"fixture_model":"a"*64}


def bundle(records):
    good=[r for r in records if r["biovil_raw_cosine"] is not None]
    return {"records":records,"used_for_routing":False,"primary_clinical_metric":False,
        "clinical_truth_available":False,"original_selection_changed":False,
        "producer":{"frozen":True,"checkpoint_sha256":HASHES,"text_policy":"full_report_no_silent_truncation_overlength_is_na"},
        "counts":{"requested_pairs":len(records),"image_encoder_calls":len({r["cxr_candidate_id"] for r in good}),
            "text_encoder_calls":len({r["report_candidate_id"] for r in good}),
            "unavailable_reports":len({r["report_candidate_id"] for r in records if r["biovil_raw_cosine"] is None})},
        "runtime_seconds":1.,"peak_vram_gib":.5}


def data():
    inventory=[row(0),row(1),row(2)];old=bundle([previous(0)])
    new=bundle([previous(2),previous(1,None)])
    pending,counts=missing_pairs(inventory,old["records"])
    return inventory,old,new,{"pairs":pending,"inventory_counts":counts},HASHES


class FullSecondaryTests(unittest.TestCase):
    def test_lossless_merge_and_complete_nominal_inventory(self):
        records,summary=f.merge(*data())
        self.assertEqual(len(records),3);self.assertEqual(records[0],data()[1]["records"][0])
        self.assertEqual(summary["numerically_available_pairs"],2);self.assertEqual(summary["na_pairs"],1)
        self.assertFalse(summary["source_choices_changed"])

    def test_sources_not_mutated(self):
        args=data();before=copy.deepcopy(args);f.merge(*args);self.assertEqual(args,before)

    def test_zero_is_available_not_missing(self):
        args=list(data());args[2]["records"][0]["biovil_raw_cosine"]=0
        self.assertEqual(f.merge(*args)[1]["numerically_available_pairs"],2)

    def test_missing_or_extra_new_pairs_rejected(self):
        for records in ([],[previous(1)],[previous(0),previous(1),previous(2)]):
            args=list(data());args[2]=bundle(records)
            with self.assertRaises(ValueError):f.merge(*args)

    def test_changed_frozen_request_rejected(self):
        args=list(data());args[3]["pairs"]=args[3]["pairs"][:-1]
        with self.assertRaises(ValueError):f.merge(*args)
        args=list(data());args[3]["inventory_counts"]["missing_candidate_pairs"]+=1
        with self.assertRaises(ValueError):f.merge(*args)

    def test_encoder_counts_are_actual_eligible_samples(self):
        args=list(data());args[2]["counts"]["text_encoder_calls"]+=1
        with self.assertRaises(ValueError):f.merge(*args)

    def test_na_status_and_reason_must_remain_explicit(self):
        args=list(data());args[2]["records"][1]["reason"]=None
        with self.assertRaises(ValueError):f.merge(*args)

    def test_endpoint_not_routing_probability_or_truth(self):
        for k in ("used_for_routing","primary_clinical_metric","clinical_truth_available","original_selection_changed"):
            args=list(data());args[2][k]=True
            with self.assertRaises(ValueError):f.merge(*args)

    def test_checkpoint_text_policy_and_freezing_are_required(self):
        for k,v in (("frozen",False),("checkpoint_sha256",{}),("text_policy","truncated")):
            args=list(data());args[2]["producer"][k]=v
            with self.assertRaises(ValueError):f.merge(*args)

    def test_runtime_memory_not_kernel_time_and_must_be_finite(self):
        for k in ("runtime_seconds","peak_vram_gib"):
            for v in (True,-1,float("nan")):
                args=list(data());args[2][k]=v
                with self.assertRaises(ValueError):f.merge(*args)

    def test_login_guard_before_metadata_load(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"checked") as read:
            with self.assertRaises(RuntimeError):cli.run(argparse.Namespace())
            read.assert_not_called()

    def test_relative_absolute_source_paths_equivalent_not_a_hash_change(self):
        relative="TriCompose-v1.2/tests/test_full_bank_secondary.py"
        self.assertTrue(cli.same_sources({"x":relative},{"x":str(ROOT/"tests/test_full_bank_secondary.py")}))
        self.assertFalse(cli.same_sources({"x":relative},{"other":relative}))

    def test_distinct_or_outside_source_paths_not_equivalent(self):
        a=str(ROOT/"tests/test_full_bank_secondary.py");b=str(ROOT/"tests/test_score_coverage.py")
        self.assertFalse(cli.same_sources({"x":a},{"x":b}))
        with self.assertRaises(ValueError):cli.same_sources({"x":a},{"x":"/etc/passwd"})


if __name__=="__main__":unittest.main()

"""Invented metadata/labels/endpoints only; no model or patient artifacts."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"benchmarks"))
sys.path.insert(0,str(ROOT/"audits"))
from test_legacy_automatic_replay import invented,policy
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from tricompose_v12.automatic_replay import candidate_key
from tricompose_v12.complete_endpoint_inventory import PAIR_FIELDS
from tricompose_v12 import report_path_comparison as f
import compare_frozen_report_paths as cli
import audit_report_path_comparison as auditor


def data(*args,**kwargs):
    scores,details=invented(*args,**kwargs)
    p=policy();bank=make_legacy_bank(scores,details,p);records=[]
    for grid in bank.values():
        for c in grid.values():
            r=c["score_record"];l=r["lineage"]
            identity={"case_id":r["case_id"],"triple_candidate_id":r["triple_candidate_id"],**l}
            records.append({**{k:identity[k] for k in PAIR_FIELDS},"biovil_raw_cosine":.1,
                "status":"computed_secondary_uncalibrated","reason":None,"calibrated":False})
    return bank,p,records


def measured(*args,**kwargs):
    bank,p,records=data(*args,**kwargs)
    choices=f.freeze_choices(bank,p)
    return f.attach_endpoint(choices,bank,p,records)


def all_table(result,method,scope="all",image="all_image_generators"):
    return next(r for r in result["method_comparison"] if r["method"]==method
        and r["ehr_evidence_subgroup"]==scope and r["cxr_group"]==image)


class ReportPathTests(unittest.TestCase):
    def test_complete_four_fixed_and_one_static_paths(self):
        bank,p,_=data();choices=f.freeze_choices(bank,p)
        self.assertEqual(len(choices),45)
        self.assertEqual({r["method"] for r in choices},{"fixed_"+m for m in p["report_order"]}|{f.STATIC})
        self.assertEqual(sum(r["method"]==f.STATIC for r in choices),9)

    def test_static_is_exact_original_key_not_endpoint_or_rank(self):
        bank,p,_=data()
        for r in f.freeze_choices(bank,p):
            if r["method"]==f.STATIC:
                group=[bank[r["case_id"]][(r["cxr_model_id"],r["cxr_seed"],m)] for m in p["report_order"]]
                self.assertEqual(r["selected_candidate_id"],min(group,key=candidate_key)["score_record"]["triple_candidate_id"])
                self.assertEqual(r["selected_report_model"],"cxrmate_single")

    def test_endpoint_not_in_choice_interface(self):
        self.assertEqual(list(inspect.signature(f.freeze_choices).parameters),["bank","policy"])
        bank,p,records=data();choices=f.freeze_choices(bank,p)
        a=f.attach_endpoint(choices,bank,p,records)
        for r in records:r["biovil_raw_cosine"]=-.95
        b=f.attach_endpoint(choices,bank,p,records)
        self.assertEqual([r["selected_candidate_id"] for r in a],[r["selected_candidate_id"] for r in b])

    def test_determinism_immutability_and_dictionary_order(self):
        bank,p,records=data();before=copy.deepcopy((bank,p,records))
        expected=f.freeze_choices(bank,p)
        reversed_bank={k:dict(reversed(list(v.items()))) for k,v in reversed(list(bank.items()))}
        self.assertEqual(expected,f.freeze_choices(reversed_bank,p))
        f.attach_endpoint(expected,bank,p,records)
        self.assertEqual(before,(bank,p,records))

    def test_missing_grid_profile_or_slot_rejected(self):
        bank,p,_=data();grid=next(iter(bank.values()));grid.pop(next(iter(grid)))
        with self.assertRaises(ValueError):f.freeze_choices(bank,p)
        bank,p,_=data();p["routing_evidence"]="existing_eight_raw_xrv_chexbert_finding_states"
        with self.assertRaises(ValueError):f.freeze_choices(bank,p)
        bank,p,_=data();next(iter(next(iter(bank.values())).values()))["score_record"]["lineage"]["cxr_seed"]=2
        with self.assertRaises(ValueError):f.freeze_choices(bank,p)

    def test_same_image_xrv_reference_must_not_change(self):
        bank,p,_=data();grid=next(iter(bank.values()))
        c=grid[("chexgenbench_sana",0,"cxrmate_single")]
        next(x for x in c["facts"] if x["finding"]=="edema")["states"]["xrv"]="negative"
        with self.assertRaisesRegex(ValueError,"same-image"):f.freeze_choices(bank,p)

    def test_fixed_ehr_state_categories_and_hash_guard(self):
        for field in ("state","category","hash"):
            bank,p,_=data();grid=next(iter(bank.values()));c=grid[("chexgenbench_sana",0,"cxrmate_single")]
            fact=next(x for x in c["facts"] if x["finding"]=="edema")
            if field=="state":fact["states"]["ehr"]="negative"
            elif field=="category":fact["source_categories"]=["medication"]
            else:c["score_record"]["lineage"]["ehr_sha256"]="e"*64
            with self.assertRaises(ValueError):f.freeze_choices(bank,p)

    def test_gate_failed_fixed_paths_remain_but_no_static_report_if_all_fail(self):
        bank,p,records=data(("positive",))
        for c in next(iter(bank.values())).values():c["score_record"]["scoring"]["selection"]["hard_gate_failure_count"]=1
        choices=f.freeze_choices(bank,p);static=[r for r in choices if r["method"]==f.STATIC]
        self.assertTrue(all(r["selected_candidate_id"] is None for r in static))
        rows=f.attach_endpoint(choices,bank,p,records);summary=f.aggregate(rows)
        a=all_table(summary,f.STATIC)
        self.assertEqual(a["unresolved_selected_report_slots"],3)
        self.assertIsNone(a["full_cohort_ehr_mean_biovil"])
        self.assertIsNone(a["cxr_report_coverage_over_known"])
        self.assertEqual(all_table(summary,"fixed_maira2")["source_gate_failed_path_slots"],3)

    def test_invalid_gate_type_rejected(self):
        for value in (True,-1,1.5):
            bank,p,_=data();next(iter(next(iter(bank.values())).values()))["score_record"]["scoring"]["selection"]["hard_gate_failure_count"]=value
            with self.assertRaises(ValueError):f.freeze_choices(bank,p)

    def test_unknown_no_finding_is_not_silently_negative(self):
        rows=measured(("positive",),no_finding=True)
        r=next(r for r in rows if r["method"]=="fixed_maira2")
        self.assertEqual(r["raw_edge_readouts"]["ehr_report"]["proxy_opposition_facts"],0)
        self.assertEqual(r["raw_edge_readouts"]["ehr_report"]["comparable_facts"],0)

    def test_unknown_and_uncertain_ehr_keep_na_edges_and_counts(self):
        result=f.aggregate(measured())
        r=all_table(result,f.STATIC,"no_direct_cached_ehr_fact")
        self.assertEqual(r["fixed_ehr_cases"],2)
        for name in ("ehr_cxr","ehr_report"):
            self.assertEqual(r["raw_edge_totals"][name]["known_reference_facts"],0)
            self.assertIsNone(r["raw_edge_totals"][name]["coverage_over_known"])
        self.assertEqual(all_table(result,f.STATIC,"direct_cached_ehr_fact")["fixed_ehr_cases"],1)

    def test_missing_duplicate_or_parent_mismatched_endpoint_rejected(self):
        for field in ("missing","duplicate","cxr_sha256","ehr_facts_sha256","case_id"):
            bank,p,records=data();choices=f.freeze_choices(bank,p)
            if field=="missing":records.pop()
            elif field=="duplicate":records.append(copy.deepcopy(records[0]))
            else:records[0][field]="wrong"
            with self.assertRaises(ValueError):f.attach_endpoint(choices,bank,p,records)

    def test_nonnumeric_nonfinite_status_probability_rejected(self):
        for value in (True,float("nan"),float("inf"),1.5,"0.1"):
            bank,p,records=data();records[0]["biovil_raw_cosine"]=value
            with self.assertRaises(ValueError):f.attach_endpoint(f.freeze_choices(bank,p),bank,p,records)
        for field,value in (("status","not_available"),("reason","invented"),("calibrated",True)):
            bank,p,records=data();records[0][field]=value
            with self.assertRaises(ValueError):f.attach_endpoint(f.freeze_choices(bank,p),bank,p,records)

    def test_na_requires_reason_and_status(self):
        bank,p,records=data();records[0]["biovil_raw_cosine"]=None
        with self.assertRaises(ValueError):f.attach_endpoint(f.freeze_choices(bank,p),bank,p,records)
        records[0].update(status="not_available",reason="invented_na")
        rows=f.attach_endpoint(f.freeze_choices(bank,p),bank,p,records)
        r=next(r for r in rows if r["selected_candidate_id"]==records[0]["triple_candidate_id"])
        self.assertIsNone(r["biovil_raw_cosine"])
        self.assertEqual(r["endpoint_unavailable_reason"],"invented_na")

    def test_missing_one_image_keeps_whole_ehr_mean_na_and_full_denominator(self):
        bank,p,records=data();records[0].update(biovil_raw_cosine=None,status="not_available",reason="invented_na")
        result=f.aggregate(f.attach_endpoint(f.freeze_choices(bank,p),bank,p,records))
        r=all_table(result,"fixed_maira2")
        self.assertEqual(r["fixed_ehr_cases"],3);self.assertEqual(r["fixed_image_slots"],9)
        self.assertEqual(r["endpoint_available_ehr_cases"],2)
        self.assertIsNone(r["full_cohort_ehr_mean_biovil"])
        self.assertAlmostEqual(r["available_ehr_mean_biovil_not_full_cohort"],.1)
        pair=next(r for r in result["paired_comparison"] if r["baseline"]=="fixed_maira2" and r["cxr_group"]=="all_image_generators" and r["ehr_evidence_subgroup"]=="all")
        self.assertEqual(pair["fixed_ehr_cases"],3);self.assertEqual(pair["paired_available_ehr_cases"],2)
        self.assertIsNone(pair["full_cohort_mean_biovil_delta"])

    def test_case_balanced_mean_manual_calculation_and_cost(self):
        bank,p,records=data()
        for r in records:
            # 0.2, 0.3, 0.4 for static CXRMate across the three image slots.
            if r["triple_candidate_id"].endswith("_1"):
                r["biovil_raw_cosine"]=(int(r["triple_candidate_id"].split("_")[-2])+2)/10
        result=f.aggregate(f.attach_endpoint(f.freeze_choices(bank,p),bank,p,records))
        a=all_table(result,f.STATIC);b=all_table(result,"fixed_maira2")
        self.assertAlmostEqual(a["full_cohort_ehr_mean_biovil"],.3)
        self.assertEqual(a["fixed_ehr_cases"],3);self.assertEqual(a["fixed_image_slots"],9)
        self.assertEqual(a["simulated_calls_per_fixed_image"],10);self.assertEqual(b["simulated_calls_per_fixed_image"],4)
        self.assertEqual(a["simulated_incremental_report_calls_per_fixed_image"],8)
        self.assertEqual(b["simulated_incremental_report_calls_per_fixed_image"],2)
        self.assertIsNone(result["actual_gpu_savings"])
        pair=next(r for r in result["paired_comparison"] if r["baseline"]=="fixed_maira2" and r["cxr_group"]=="all_image_generators" and r["ehr_evidence_subgroup"]=="all")
        self.assertAlmostEqual(pair["full_cohort_mean_biovil_delta"],.2)
        self.assertEqual(pair["positive_delta_cases"],3)

    def test_no_positive_or_negative_denominator_means_na(self):
        result=f.aggregate(measured())
        r=all_table(result,f.STATIC)
        self.assertEqual(r["image_positive_support_over_known"],1)
        self.assertIsNone(r["image_negative_support_over_known"])

    def test_same_image_paired_hash_guard(self):
        rows=measured();next(r for r in rows if r["method"]=="fixed_maira2")["cxr_sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"changed image/EHR"):f.aggregate(rows)

    def test_changed_choices_rejected_before_endpoint_use(self):
        bank,p,records=data();choices=f.freeze_choices(bank,p);choices[0]["selected_candidate_id"]="wrong"
        with self.assertRaisesRegex(ValueError,"choices changed"):f.attach_endpoint(choices,bank,p,records)

    def test_previous_sana_equivalence_and_changed_old_winner_guard(self):
        bank,p,_=data();choices=f.freeze_choices(bank,p)
        old=[{"case_id":r["case_id"],"method":"report_only_static","model_call_budget":30,
            "selected_candidate_id":r["selected_candidate_id"],"control_fixed_image_sha256":r["cxr_sha256"],
            "selected_ehr_sha256":r["ehr_sha256"],"selected_snapshot":{"artifact_hashes":{"ehr_facts_sha256":r["ehr_facts_sha256"]},
                "report_model_id":r["selected_report_model"]}}
            for r in choices if r["method"]==f.STATIC and r["cxr_model_id"]=="chexgenbench_sana"]
        self.assertEqual(f.verify_previous_sana(choices,old,30)["exact_same_choices"],3)
        for field in ("selected_candidate_id","control_fixed_image_sha256","selected_ehr_sha256"):
            changed=copy.deepcopy(old);changed[0][field]="wrong"
            with self.assertRaises(ValueError):f.verify_previous_sana(choices,changed,30)

    def test_previous_sana_inventory_no_silent_case_drop(self):
        bank,p,_=data()
        with self.assertRaises(ValueError):f.verify_previous_sana(f.freeze_choices(bank,p),[],30)

    def test_flatten_retains_raw_counts_and_na(self):
        row=all_table(f.aggregate(measured()),f.STATIC,"no_direct_cached_ehr_fact")
        flat=cli.flatten(row)
        self.assertEqual(flat["ehr_report_known_reference_facts"],0)
        self.assertIsNone(flat["ehr_report_coverage_over_known"])
        self.assertEqual(flat["cxr_report_supported_positive"],6)

    def test_no_artifact_bodies_identifiers_or_accuracy_claim(self):
        result=f.aggregate(measured());serialized=json.dumps(result)
        for key in ("subject_id","patient_id","report_text","image_path","source_statement"):
            self.assertNotIn(key,serialized)
        self.assertFalse(result["clinical_acceptance"]);self.assertIsNone(result["clinical_accuracy"])
        self.assertFalse(result["selection_used_biovil"])

    def test_login_guard_before_any_source_read(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"load_bank") as load:
            with self.assertRaises(RuntimeError):cli.run(argparse.Namespace())
            load.assert_not_called()

    def test_audit_login_guard_before_any_protected_read(self):
        with patch.dict(os.environ,{},clear=True),patch.object(auditor,"read_json") as read:
            with self.assertRaises(RuntimeError):auditor.audit("invented_only")
            read.assert_not_called()

    def test_report_keeps_cost_and_clinical_limitations(self):
        text=cli.markdown(f.aggregate(measured()),{"archived_sana_cases":3,"exact_same_choices":3})
        for phrase in ("not a held-out benchmark","not a gold-standard clinical judgment",
                       "not measured GPU savings","CXRMate-single does NOT consume structured EHR"):
            self.assertIn(phrase,text)


if __name__=="__main__":unittest.main()

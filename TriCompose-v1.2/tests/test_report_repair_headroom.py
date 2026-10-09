"""Invented evidence/signatures only; no patient artifacts or model execution."""
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
from test_report_path_comparison import data
from tricompose_v12 import report_repair_headroom as f
from tricompose_v12.report_path_comparison import STATIC,freeze_choices
import diagnose_report_repair_headroom as cli
import audit_report_repair_headroom as auditor


def pair(ehr="positive"):
    bank,p,_=data((ehr,));grid=next(iter(bank.values()))
    return [copy.deepcopy(grid[("chexgenbench_sana",0,m)]) for m in ("maira2","cxrmate_single")]


def finding(candidate,name,**states):
    next(x for x in candidate["facts"] if x["finding"]==name)["states"].update(states)


def measured():
    bank,p,records=data();choices=freeze_choices(bank,p)
    plan=f.prepare(bank,p,choices)
    return f.attach(plan,bank,p,choices,records)


def table(result,method="fixed_maira2",scope="all",image="all_image_generators"):
    return next(r for r in result["method_comparison"] if r["method"]==method
        and r["ehr_evidence_subgroup"]==scope and r["cxr_group"]==image)


class RepairHeadroomTests(unittest.TestCase):
    def test_explicit_conflict_corrected_with_comparison_preserved(self):
        base,alt=pair();r=f.compare(f.signature(base),f.signature(alt))
        self.assertTrue(r["strict_label_preserving_headroom"])
        self.assertEqual(r["removed_opposition_fact_ids"],{"image_opposition":["edema"],"ehr_opposition":["edema"]})
        self.assertFalse(any(r["lost_fact_ids"].values()))
        self.assertFalse(r["clinical_repair_success"])

    def test_silencing_to_unknown_or_uncertain_is_not_repair(self):
        for state in ("unknown","uncertain"):
            base,alt=pair();finding(alt,"edema",chexbert=state)
            r=f.compare(f.signature(base),f.signature(alt))
            self.assertFalse(r["strict_label_preserving_headroom"])
            self.assertEqual(r["opposition_silenced_fact_ids"],{"image":["edema"],"ehr":["edema"]})

    def test_equal_positive_counts_cannot_swap_supported_finding_ids(self):
        base,alt=pair()
        for c in (base,alt):finding(c,"pneumonia",xrv="positive")
        finding(base,"edema",chexbert="positive");finding(base,"pneumonia",chexbert="unknown")
        finding(alt,"edema",chexbert="unknown");finding(alt,"pneumonia",chexbert="positive")
        a,b=f.signature(base),f.signature(alt)
        self.assertEqual(len(a["image_positive_support"]),len(b["image_positive_support"]))
        r=f.compare(a,b)
        self.assertFalse(r["strict_label_preserving_headroom"])
        self.assertEqual(r["lost_fact_ids"]["image_positive_support"],["edema"])

    def test_lower_opposition_count_cannot_pay_for_new_different_opposition(self):
        base,alt=pair("unknown")
        for c in (base,alt):
            for name in ("pneumonia","atelectasis"):finding(c,name,xrv="positive")
        finding(base,"pneumonia",chexbert="negative");finding(base,"atelectasis",chexbert="positive")
        finding(alt,"pneumonia",chexbert="positive");finding(alt,"atelectasis",chexbert="negative")
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertLess(r["alternative_raw_opposition_count"],r["baseline_raw_opposition_count"])
        self.assertFalse(r["strict_label_preserving_headroom"])
        self.assertEqual(r["new_opposition_fact_ids"]["image_opposition"],["atelectasis"])

    def test_supported_direct_ehr_fact_must_be_preserved_even_when_image_unknown(self):
        base,alt=pair()
        for c in (base,alt):finding(c,"edema",xrv="unknown")
        finding(base,"edema",chexbert="positive");finding(alt,"edema",chexbert="unknown")
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertFalse(r["strict_label_preserving_headroom"])
        self.assertEqual(r["lost_fact_ids"]["ehr_direct_support"],["edema"])

    def test_existing_negative_support_cannot_disappear_as_missing(self):
        base,alt=pair("unknown")
        for c in (base,alt):finding(c,"edema",xrv="negative")
        finding(base,"edema",chexbert="negative");finding(alt,"edema",chexbert="unknown")
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertFalse(r["strict_label_preserving_headroom"])
        self.assertEqual(r["lost_fact_ids"]["image_comparable"],["edema"])

    def test_unknown_ehr_never_becomes_constraint_from_weak_context(self):
        base,alt=pair("unknown")
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertEqual(r["direct_ehr_known_facts"],0)
        self.assertEqual(r["new_opposition_fact_ids"]["ehr_opposition"],[])
        self.assertTrue(r["strict_label_preserving_headroom"])

    def test_unknown_image_positive_assertion_tracked_not_automatically_false(self):
        base,alt=pair();finding(alt,"support_devices",chexbert="positive")
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertIn("support_devices",r["alternative_positive_unknown_image"])
        self.assertNotIn("support_devices",r["new_opposition_fact_ids"]["image_opposition"])
        self.assertTrue(r["strict_label_preserving_headroom"])

    def test_uncertain_image_is_not_a_negative_constraint(self):
        base,alt=pair("unknown")
        for c in (base,alt):finding(c,"pneumonia",xrv="uncertain")
        finding(alt,"pneumonia",chexbert="positive")
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertNotIn("pneumonia",r["new_opposition_fact_ids"]["image_opposition"])
        self.assertNotIn("pneumonia",r["gained_fact_ids"]["image_positive_support"])

    def test_structure_lower_blocks_headroom_but_not_raw_label_diagnosis(self):
        base,alt=pair();alt["score_record"]["scoring"]["modality_quality"]["report_structure_quality_score_0_1"]=.7
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertTrue(r["strict_label_dominance_without_quality_gate"])
        self.assertTrue(r["headroom_assessable"]);self.assertFalse(r["strict_label_preserving_headroom"])

    def test_missing_structure_quality_is_explicitly_unassessable(self):
        base,alt=pair();alt["score_record"]["scoring"]["modality_quality"]["report_structure_quality_score_0_1"]=None
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertFalse(r["headroom_assessable"]);self.assertFalse(r["strict_label_preserving_headroom"])
        self.assertIn("existing_structure_quality_unavailable",r["block_or_equivalence_reasons"])

    def test_artifact_failed_not_a_clinical_opportunity(self):
        for field in ("gate","image"):
            base,alt=pair()
            if field=="gate":alt["score_record"]["scoring"]["selection"]["hard_gate_failure_count"]=1
            else:alt["score_record"]["scoring"]["modality_quality"]["cxr_basic_validity_pass"]=False
            r=f.compare(f.signature(base),f.signature(alt))
            self.assertFalse(r["headroom_assessable"]);self.assertFalse(r["strict_label_preserving_headroom"])

    def test_better_structure_alone_is_not_finding_improvement(self):
        base,alt=pair();finding(base,"edema",chexbert="positive")
        alt["score_record"]["scoring"]["modality_quality"]["report_structure_quality_score_0_1"]=.9
        r=f.compare(f.signature(base),f.signature(alt))
        self.assertTrue(r["preserved_fact_ids"]);self.assertFalse(r["strict_label_preserving_headroom"])

    def test_raw_global_no_finding_does_not_expand_unknown_to_negative(self):
        base,alt=pair("unknown");finding(base,"edema",chexbert="unknown");finding(base,"no_finding",chexbert="positive")
        a=f.signature(base)
        self.assertEqual(a["image_opposition"],[]);self.assertEqual(a["image_comparable"],[])
        self.assertIn("no_finding",a["report_positive_unknown_image"])

    def test_signature_invalid_numeric_and_metadata_types(self):
        for value in (True,float("nan"),float("inf"),-.1,1.1,".8"):
            base,_=pair();base["score_record"]["scoring"]["modality_quality"]["report_structure_quality_score_0_1"]=value
            with self.assertRaises(ValueError):f.signature(base)
        base,_=pair();base["score_record"]["scoring"]["modality_quality"]["cxr_basic_validity_pass"]=1
        with self.assertRaises(ValueError):f.signature(base)

    def test_fixed_ehr_and_same_image_signature_guards(self):
        for key in ("cxr_sha256","ehr_facts_sha256","cxr_candidate_id","cxr_seed"):
            base,alt=map(f.signature,pair());alt["lineage"][key]="changed"
            with self.assertRaises(ValueError):f.compare(base,alt)
        base,alt=map(f.signature,pair());alt["image_reference_states"]["edema"]="negative"
        with self.assertRaises(ValueError):f.compare(base,alt)

    def test_all_ordered_pairs_and_paths_retained_with_no_new_winner(self):
        bank,p,_=data();plan=f.prepare(bank,p,freeze_choices(bank,p))
        self.assertEqual(len(plan["directed_pairs"]),108)
        self.assertEqual(len(plan["image_graphs"]),9)
        self.assertEqual(len(plan["path_opportunities"]),45)
        self.assertEqual(sum(r["strict_label_preserving_headroom"] for r in plan["directed_pairs"]),27)
        self.assertTrue(all(r["new_selected_candidate_id"] is None for r in plan["path_opportunities"]))
        self.assertFalse(plan["new_policy_executed"])

    def test_deterministic_reordered_input_and_immutability(self):
        bank,p,records=data();choices=freeze_choices(bank,p);before=copy.deepcopy((bank,p,choices,records))
        plan=f.prepare(bank,p,choices)
        reversed_bank={k:dict(reversed(list(v.items()))) for k,v in reversed(list(bank.items()))}
        self.assertEqual(plan,f.prepare(reversed_bank,p,choices))
        f.attach(plan,bank,p,choices,records)
        self.assertEqual(before,(bank,p,choices,records))

    def test_no_endpoint_in_predicate_or_plan_interface(self):
        self.assertEqual(list(inspect.signature(f.prepare).parameters),["bank","policy","choices"])
        self.assertEqual(list(inspect.signature(f.compare).parameters),["base","alternative"])

    def test_endpoint_changes_do_not_change_admissible_alternative_sets(self):
        bank,p,records=data();choices=freeze_choices(bank,p);plan=f.prepare(bank,p,choices)
        a=f.attach(plan,bank,p,choices,records)
        for r in records:r["biovil_raw_cosine"]=-.9
        b=f.attach(plan,bank,p,choices,records)
        self.assertEqual([r["label_preserving_alternative_ids"] for r in a["path_opportunities"]],
                         [r["label_preserving_alternative_ids"] for r in b["path_opportunities"]])

    def test_changed_label_plan_or_choices_rejected(self):
        bank,p,records=data();choices=freeze_choices(bank,p);plan=f.prepare(bank,p,choices)
        plan["directed_pairs"][0]["strict_label_preserving_headroom"]=False
        with self.assertRaises(ValueError):f.attach(plan,bank,p,choices,records)
        choices[0]["selected_candidate_id"]="changed"
        with self.assertRaises(ValueError):f.prepare(bank,p,choices)

    def test_endpoint_incomplete_or_mismatched_hash_rejected(self):
        bank,p,records=data();choices=freeze_choices(bank,p);plan=f.prepare(bank,p,choices)
        with self.assertRaises(ValueError):f.attach(plan,bank,p,choices,records[:-1])
        records[0]["report_sha256"]="f"*64
        with self.assertRaises(ValueError):f.attach(plan,bank,p,choices,records)

    def test_no_opportunity_is_na_not_zero_benefit_and_cases_remain(self):
        result=f.summarize(measured());row=table(result,STATIC)
        self.assertEqual(row["fixed_ehr_cases"],3);self.assertEqual(row["fixed_image_slots"],9)
        self.assertEqual(row["image_slots_with_label_preserving_alternative"],0)
        self.assertIsNone(row["conditional_ehr_balanced_mean_all_alternative_delta_not_policy"])

    def test_all_alternatives_averaged_not_endpoint_best(self):
        bank,p,records=data();choices=freeze_choices(bank,p);plan=f.prepare(bank,p,choices)
        for r in records:
            j=int(r["triple_candidate_id"].split("_")[-1])
            r["biovil_raw_cosine"]=[.1,.2,.4,-.2][j]
        result=f.summarize(f.attach(plan,bank,p,choices,records));row=table(result)
        self.assertEqual(row["fixed_ehr_cases"],3)
        self.assertEqual(row["ehr_cases_with_any_label_preserving_alternative"],3)
        self.assertEqual(row["dominating_alternative_pair_count"],27)
        self.assertAlmostEqual(row["conditional_ehr_balanced_mean_all_alternative_delta_not_policy"],(.1+.3-.3)/3)
        self.assertEqual(row["dominating_pairs_biovil_higher"],18);self.assertEqual(row["dominating_pairs_biovil_lower"],9)

    def test_missing_one_endpoint_keeps_entire_opportunity_case_mean_na(self):
        bank,p,records=data();choices=freeze_choices(bank,p);plan=f.prepare(bank,p,choices)
        records[1].update(biovil_raw_cosine=None,status="not_available",reason="invented_na")
        row=table(f.summarize(f.attach(plan,bank,p,choices,records)))
        self.assertEqual(row["ehr_cases_with_any_label_preserving_alternative"],3)
        self.assertEqual(row["conditional_opportunity_ehr_cases_with_complete_endpoints"],2)
        self.assertEqual(row["endpoint_available_dominating_pairs"],26)
        self.assertIsNone(row["conditional_ehr_balanced_mean_all_alternative_delta_not_policy"])
        self.assertEqual(row["available_opportunity_case_mean_not_full_denominator"],0)

    def test_evidence_subgroups_unknown_and_uncertain_not_promoted(self):
        result=f.summarize(measured())
        self.assertEqual(table(result,scope="direct_cached_ehr_fact")["fixed_ehr_cases"],1)
        self.assertEqual(table(result,scope="no_direct_cached_ehr_fact")["fixed_ehr_cases"],2)

    def test_graph_not_dominated_is_not_global_or_clinical_optimality(self):
        plan=measured()
        for graph in plan["image_graphs"]:
            self.assertEqual(len(graph["not_dominated_under_frozen_diagnostic"]),3)
            self.assertIsNone(graph["clinical_pareto_optimality"])

    def test_all_metadata_no_raw_body_identifier_or_model_truth(self):
        result=f.summarize(measured());serialized=json.dumps(result)
        for forbidden in ("subject_id","patient_id","report_text","image_path","source_statement"):
            self.assertNotIn(forbidden,serialized)
        self.assertFalse(result["clinical_repair_success"]);self.assertIsNone(result["clinical_accuracy"])

    def test_runner_and_loader_guard_before_protected_read(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"load_inputs") as load:
            with self.assertRaises(RuntimeError):cli.run(argparse.Namespace())
            load.assert_not_called()
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"checked_source") as read:
            with self.assertRaises(RuntimeError):cli.load_inputs("invented")
            read.assert_not_called()

    def test_bilingual_report_retains_conditional_and_claim_limits(self):
        text=cli.markdown(f.summarize(measured()))
        for phrase in ("not a policy treatment effect","not proof of EHR fidelity","no new generator calls",
                       "No selected output was replaced","not clinical truth"):
            self.assertIn(phrase,text)

    def test_audit_guard_before_any_protected_read(self):
        with patch.dict(os.environ,{},clear=True),patch.object(auditor,"read_json") as read:
            with self.assertRaises(RuntimeError):auditor.audit("invented_only")
            read.assert_not_called()

    def test_opposite_direction_is_not_also_strictly_dominating(self):
        base,alt=map(f.signature,pair())
        self.assertTrue(f.compare(base,alt)["strict_label_preserving_headroom"])
        self.assertFalse(f.compare(alt,base)["strict_label_preserving_headroom"])


if __name__=="__main__":unittest.main()

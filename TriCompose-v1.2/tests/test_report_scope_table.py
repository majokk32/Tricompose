"""Invented source reports only: availability arithmetic, never clinical gold."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import report_scope_table as table
from tricompose_v12.report_assertions import digest
from repair_cached_report_evidence import scope_check
from verify_candidate_findings_qwen import FINDINGS as CACHED_FINDINGS
from prepare_blinded_report_review import freeze_method
import build_report_scope_table as cli


def fixture(texts=("Cardiomegaly is present.",),states=("positive",),*,ehr="unknown",image="positive"):
    facts=[];sources={}
    for i,(text,report_state) in enumerate(zip(texts,states,strict=True)):
        # Invented unique report suffixes do not affect the clinical clause.
        text+=f"\nInvented fixture {i}."
        h=digest(text);sources[h]=text
        for finding in table.FINDINGS:
            es=ehr if finding=="cardiomegaly" else "unknown"
            facts.append({"evidence_id":f"ev_{i}_{finding}","case_id":"fictional_case",
                "triple_candidate_id":f"fictional_candidate_{i}","cxr_candidate_id":"fictional_image",
                "report_candidate_id":f"fictional_report_{i}","report_model_id":f"fictional_model_{i}",
                "finding":finding,"artifact_hashes":{"ehr_sha256":"e"*64,"ehr_facts_sha256":"f"*64,
                    "cxr_sha256":"c"*64,"report_sha256":h},
                "states":{"ehr":es,"xrv":image if finding=="cardiomegaly" else "unknown",
                    "chexbert":report_state if finding=="cardiomegaly" else "unknown",
                    "qwen_image":"unknown","qwen_report":"unknown"},
                "ehr_provenance":{"weak_clinical_context_promoted":False,
                    "evidence_ids":[] if es=="unknown" else ["invented_direct_diagnosis"],
                    "source_fields":[] if es=="unknown" else ["diagnoses[0]"]},
                "original_selected":i==0})
    return facts,sources


def get_finding(rows,finding="cardiomegaly",candidate=0):
    return next(row for row in rows if row["finding"]==finding and
        row["triple_candidate_id"]==f"fictional_candidate_{candidate}")


class AvailabilityTests(unittest.TestCase):
    def test_finding_order_reuses_existing_eight_and_four_scope_pins_unchanged(self):
        self.assertEqual(table.FINDINGS,CACHED_FINDINGS)
        frozen,_=freeze_method()
        self.assertFalse(frozen["new_rules_or_thresholds"])

    def test_unknown_never_matches_negative_or_supports_itself(self):
        for other in table.STATES:
            self.assertNotEqual(table.relation("unknown",other),"support")
            self.assertNotEqual(table.relation(other,"unknown"),"opposition")
        self.assertEqual(table.relation("unknown","unknown"),"unknown")

    def test_uncertain_and_unavailable_are_not_comparable(self):
        for other in table.STATES:
            self.assertEqual(table.relation("uncertain",other),"not_comparable")
            self.assertEqual(table.relation(other,"uncertain"),"not_comparable")
        self.assertEqual(table.relation("positive","positive",available=False),"not_comparable")

    def test_only_explicit_same_or_opposite_states_are_comparable(self):
        for state in table.EXPLICIT:
            self.assertEqual(table.relation(state,state),"support")
        self.assertEqual(table.relation("positive","negative"),"opposition")
        with self.assertRaises(ValueError):table.relation("missing","positive")

    def test_complete_inventory_retained_and_no_clinical_policy_activated(self):
        facts,texts=fixture();rows,candidates,groups,summary,assertions=table.build_scope_table(facts,texts,scope_check)
        self.assertEqual(len(rows),8);self.assertEqual(len(assertions),4)
        self.assertEqual(summary["report_scope_decision_counts"]["outside_scope_inventory"],4)
        self.assertEqual(candidates[0]["edges"]["scoped"]["cxr_report"]["inventory_facts"],8)
        self.assertIsNone(candidates[0]["clinical_selection_score"])
        self.assertFalse(candidates[0]["automatic_repair_eligible"])
        for key in ("primary_metric_eligible","selection_changed","regeneration_authorized","independent_human_labels_used"):
            self.assertFalse(summary[key])
        self.assertTrue(all(not row["independent_votes"] for row in groups))
        self.assertTrue(all(row["confirmed_faulty_modality"] is None for row in rows))

    def test_scope_matched_proposal_retained_as_syntax_not_clinical_truth(self):
        facts,texts=fixture();rows,_,_,summary,assertions=table.build_scope_table(facts,texts,scope_check)
        row=get_finding(rows)
        self.assertEqual(row["report_scope_state"],"positive")
        self.assertEqual(row["report_scope_decision"],"scope_commit")
        self.assertEqual(row["relations"]["scoped"]["cxr_report"],"support")
        self.assertEqual(summary["relations"]["scoped"]["cxr_report"]["comparable_facts"],1)
        self.assertFalse(next(value for value in assertions.values() if value["finding"]=="cardiomegaly")["independent_clinical_validation"])

    def test_negated_positive_proposal_abstains_never_flips_to_negative(self):
        facts,texts=fixture(texts=("No cardiomegaly.",),states=("positive",),image="negative")
        rows,candidates,_,_,_=table.build_scope_table(facts,texts,scope_check)
        row=get_finding(rows)
        self.assertEqual(row["states"]["chexbert"],"positive")
        self.assertEqual(row["report_scope_state"],"unknown")
        self.assertEqual(row["relations"]["raw"]["cxr_report"],"opposition")
        self.assertEqual(row["relations"]["scoped"]["cxr_report"],"not_comparable")
        self.assertEqual(candidates[0]["withdrawn_opposition_signals"]["cxr_report"],1)
        self.assertFalse(row["clinical_error_confirmed"])

    def test_unknown_model_label_not_filled_from_explicit_source(self):
        facts,texts=fixture(states=("unknown",))
        rows,_,_,summary,_=table.build_scope_table(facts,texts,scope_check)
        row=get_finding(rows)
        self.assertEqual(row["report_scope_state"],"unknown")
        self.assertEqual(row["report_scope_decision"],"no_model_assertion")
        self.assertEqual(row["relations"]["scoped"]["cxr_report"],"unknown")
        self.assertIsNone(summary["relations"]["scoped"]["cxr_report"]["conditional_support_fraction"])

    def test_outside_scope_pneumonia_cannot_receive_fake_head_or_false_negation(self):
        facts,texts=fixture(texts=("Pneumonia is present.",),states=("unknown",))
        row=get_finding(facts,"pneumonia")
        row["states"].update(ehr="positive",xrv="positive",chexbert="positive")
        row["ehr_provenance"].update(evidence_ids=["invented_direct_diagnosis"],source_fields=["diagnoses[0]"])
        rows,_,_,_,_=table.build_scope_table(facts,texts,scope_check)
        result=get_finding(rows,"pneumonia")
        self.assertEqual(result["relations"]["raw"]["ehr_report"],"support")
        self.assertEqual(result["relations"]["scoped"]["ehr_report"],"not_comparable")
        self.assertEqual(result["report_scope_decision"],"outside_scope_inventory")
        self.assertIsNone(result["report_scope_evidence_key"])

    def test_no_direct_ehr_facts_has_no_fake_reference_coverage(self):
        facts,texts=fixture();_,candidates,_,_,_=table.build_scope_table(facts,texts,scope_check)
        for stage in ("raw","scoped"):
            for edge in ("ehr_cxr","ehr_report"):
                result=candidates[0]["edges"][stage][edge]
                self.assertEqual(result["comparable_facts"],0)
                self.assertIsNone(result["coverage_over_explicit_reference"])
                self.assertIsNone(result["conditional_support_fraction"])

    def test_ehr_image_edge_is_not_changed_by_report_scope(self):
        facts,texts=fixture(ehr="positive")
        _,candidates,_,summary,_=table.build_scope_table(facts,texts,scope_check)
        self.assertEqual(summary["relations"]["raw"]["ehr_cxr"],summary["relations"]["scoped"]["ehr_cxr"])
        self.assertEqual(candidates[0]["withdrawn_support_signals"]["ehr_cxr"],0)


class DependencyAndProvenanceTests(unittest.TestCase):
    def test_report_disagreement_on_same_image_never_becomes_independent_vote(self):
        facts,texts=fixture(texts=("Cardiomegaly is present.","No cardiomegaly."),states=("positive","negative"))
        _,_,groups,summary,_=table.build_scope_table(facts,texts,scope_check)
        group=next(row for row in groups if row["finding"]=="cardiomegaly")
        self.assertEqual(group["distinct_report_texts"],2)
        self.assertTrue(group["scope_positive_negative_disagreement"])
        self.assertFalse(group["independent_votes"])
        self.assertIsNone(group["confirmed_faulty_modality"])
        self.assertEqual(summary["relations"]["raw"]["ehr_cxr"]["inventory_facts"],8)
        self.assertEqual(summary["relations"]["raw"]["cxr_report"]["inventory_facts"],16)

    def test_image_dedup_uses_bytes_not_only_candidate_ids(self):
        facts,texts=fixture(texts=("Cardiomegaly is present.","No cardiomegaly."),states=("positive","negative"))
        for row in facts:
            if row["triple_candidate_id"]=="fictional_candidate_1":row["cxr_candidate_id"]="clone_image_id"
        _,_,groups,summary,_=table.build_scope_table(facts,texts,scope_check)
        self.assertEqual(summary["unique_image_finding_pairs"],8)
        self.assertEqual(len(groups),8)

    def test_original_selected_flag_is_not_an_input_to_scope_or_availability(self):
        facts,texts=fixture();original=table.build_scope_table(facts,texts,scope_check)
        for row in facts:row["original_selected"]=not row["original_selected"]
        self.assertEqual(table.build_scope_table(facts,texts,scope_check),original)

    def test_deterministic_source_order_invariant_json_stable_no_mutation(self):
        facts,texts=fixture();before=copy.deepcopy((facts,texts))
        result=table.build_scope_table(facts,texts,scope_check)
        self.assertEqual((facts,texts),before)
        self.assertEqual(result,table.build_scope_table(list(reversed(facts)),texts,scope_check))
        self.assertEqual(list(result),json.loads(json.dumps(result)))

    def test_all_reports_share_one_scope_readout_per_exact_text(self):
        facts,texts=fixture();other=copy.deepcopy(facts)
        for row in other:
            row["triple_candidate_id"]="other_lineage"
            row["report_candidate_id"]="other_report_id"
            row["evidence_id"]="other_"+row["evidence_id"]
        rows,_,groups,_,assertions=table.build_scope_table(facts+other,texts,scope_check)
        self.assertEqual(len(rows),16);self.assertEqual(len(assertions),4)
        self.assertTrue(all(row["distinct_report_texts"]==1 for row in groups))

    def test_missing_duplicate_foreign_or_invalid_facts_are_refused(self):
        facts,texts=fixture()
        for bad in ([],facts[:-1],facts+[facts[0]],[{**row,"finding":"device"} for row in facts]):
            with self.assertRaises(ValueError):table.build_scope_table(bad,texts,scope_check)
        bad=copy.deepcopy(facts);bad[0]["states"]["ehr"]="missing"
        with self.assertRaises(ValueError):table.build_scope_table(bad,texts,scope_check)

    def test_hash_text_or_source_inventory_changes_are_refused(self):
        facts,texts=fixture();h=next(iter(texts))
        for bad in ({},{h:texts[h]+" "},{**texts,"extra": "invented"}):
            with self.assertRaises(ValueError):table.build_scope_table(facts,bad,scope_check)
        bad=copy.deepcopy(facts);bad[0]["artifact_hashes"]["cxr_sha256"]="bad"
        with self.assertRaises(ValueError):table.build_scope_table(bad,texts,scope_check)

    def test_fixed_ehr_or_shared_image_states_cannot_change(self):
        facts,texts=fixture(texts=("Cardiomegaly is present.","No cardiomegaly."),states=("positive","negative"))
        for field in ("ehr_sha256","cxr_state"):
            bad=copy.deepcopy(facts);row=get_finding(bad,candidate=1)
            if field=="ehr_sha256":row["artifact_hashes"][field]="a"*64
            else:row["states"]["xrv"]="negative"
            with self.assertRaises(ValueError):table.build_scope_table(bad,texts,scope_check)

    def test_weak_context_or_missing_direct_provenance_is_not_promoted(self):
        facts,texts=fixture(ehr="positive")
        for field,value in (("weak_clinical_context_promoted",True),("evidence_ids",[]),("source_fields",[])):
            bad=copy.deepcopy(facts);get_finding(bad)["ehr_provenance"][field]=value
            with self.assertRaises(ValueError):table.build_scope_table(bad,texts,scope_check)

    def test_report_state_for_same_bytes_cannot_disagree(self):
        facts,texts=fixture();other=copy.deepcopy(facts)
        get_finding(other)["states"]["chexbert"]="negative"
        for row in other:
            row["triple_candidate_id"]="other_lineage";row["evidence_id"]="other_"+row["evidence_id"]
        with self.assertRaises(ValueError):table.build_scope_table(facts+other,texts,scope_check)

    def test_slurm_guard_precedes_packet_paths_or_report_reader(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"freeze_method") as frozen,\
                patch.object(cli,"require_inside") as access:
            with self.assertRaisesRegex(RuntimeError,"Slurm"):cli.load(SimpleNamespace())
            frozen.assert_not_called();access.assert_not_called()

    def test_cli_joins_crossmodal_report_ids_and_only_reads_approved_copies(self):
        facts,texts=fixture(texts=("Cardiomegaly is present.",)*48,states=("positive",)*48)
        by_candidate={}
        for row in facts:
            cid=row["triple_candidate_id"];row["report_candidate_id"]=cid
            by_candidate.setdefault(cid,[]).append(row)
        packet=Path("invented_packet");finding=Path("invented_finding")
        bundle=Path("invented_bundle").resolve()
        resolver=[];refs=[];lineages=[];items=[]
        for i,(cid,rows) in enumerate(by_candidate.items()):
            first=rows[0];hashes=first["artifact_hashes"]
            resolver.append({"candidate_id":cid,"case_id":first["case_id"],"item_id":f"report_{i:04d}",
                "report_sha256":hashes["report_sha256"],"ehr_sha256":hashes["ehr_sha256"],
                "cxr_sha256":hashes["cxr_sha256"],"model_id":first["report_model_id"]})
            # Actual crossmodal schema uses report_candidate_id, not triple ID.
            refs.append({"report_candidate_id":cid,**{field:{row["finding"]:row["states"][kind] for row in rows}
                for kind,field in (("ehr","ehr_finding_states"),("xrv","cxr_finding_states"),("chexbert","report_finding_states"))}})
            lineages.append({"triple_candidate_id":cid,"lineage":{
                **hashes,"cxr_candidate_id":first["cxr_candidate_id"]}})
            items.append({"item_id":f"report_{i:04d}","report_sha256":hashes["report_sha256"]})
        pm={"schema_version":cli.PACKET_SCHEMA,"source_sha256":{"selection_table":"bound","crossmodal_details":"bound"}}
        fm={"schema_version":"tricompose-cached-finding-review-v1","metadata_only":True,
            "source_paths":{"lineages":"invented_lineages.jsonl","crossmodal_details":"invented_cross.json"},
            "source_sha256":{"lineages":"bound","crossmodal_details":"bound"}}
        mapping={packet/"manifest.json":pm,packet/"investigator/resolver.json":{"records":resolver},
            packet/"investigator/method_freeze.json":{"frozen":"fixture"},finding/"manifest.json":fm,
            Path("invented_cross.json"):{"evaluation_scope":{"cohort":"fully_synthetic",
                "raw_source_target_supplied":False,"real_reference_report_supplied":False,"unknown_is_negative":False},"records":refs},
            bundle/"summary.json":{"source_packet_manifest_sha256":"bound"}}
        def text(path,*args,**kwargs):
            return "".join(json.dumps(row)+"\n" for row in
                (facts if path.name=="fact_evidence.jsonl" else lineages))
        reader=Mock(side_effect=texts.__getitem__)
        args=SimpleNamespace(packet_run=packet,finding_run=finding,bundle_run=bundle)
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented"}),\
                patch.object(cli,"freeze_method",return_value=({"frozen":"fixture"},{})),\
                patch.object(cli,"require_inside",side_effect=lambda path,*a,**k:Path(path)),\
                patch.object(cli,"read_json",side_effect=lambda path:mapping[path]),\
                patch.object(cli,"sha256_file",return_value="bound"),\
                patch.object(cli,"checked_artifact",side_effect=lambda root,name:(root/name,{})),\
                patch.object(cli,"load_bundle",return_value=(items,reader,{"blind_manifest":bundle/"manifest.json"})),\
                patch.object(Path,"stat",return_value=SimpleNamespace(st_size=100)),\
                patch.object(Path,"read_text",text):
            (_,candidates,_,summary,_),_=cli.load(args)
        self.assertEqual(len(candidates),48);self.assertEqual(summary["fact_rows"],384)
        self.assertEqual(reader.call_count,48)
        self.assertEqual({call.args[0] for call in reader.call_args_list},set(texts))
        self.assertFalse(summary["selection_changed"])


if __name__=="__main__":
    unittest.main()

"""Invented reports only: blinding, human annotation and denominator contracts."""
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import report_review as review
from tricompose_v12.report_assertions import checked_span,digest,FINDINGS
import prepare_blinded_report_review as builder
import audit_blinded_report_review as audit
import materialize_blinded_report_review as materializer
from types import SimpleNamespace


def fixture(text="Cardiomegaly is present."):
    candidates=[{"candidate_id":"candidate_a","case_id":"synthetic_a",
        "report_sha256":digest(text),"report_path":"fictional.txt","model_id":"fictional_model",
        "ehr_sha256":"e"*64,"cxr_sha256":"c"*64}]
    items,resolver=review.make_queue(candidates)
    return items,resolver,review.annotation_template(items),{digest(text):text}


def complete_one(annotation,text,*,alias="reader_a",state="positive",reason="explicit_assertion"):
    a=copy.deepcopy(annotation)
    a.update(reviewer_alias=alias,reviewer_role="human_domain_annotator",
        independence_attestation={"model_predictions_seen":False,"winner_flags_seen":False,"labels_generated_by_model":False})
    row=next(row for row in a["records"] if row["finding"]=="cardiomegaly")
    row.update(status="reviewed",state=state,reason=reason,evidence=[] if reason=="not_mentioned" else [checked_span(text,0,len(text))])
    return a


class QueueTests(unittest.TestCase):
    def test_review_payload_has_no_model_score_winner_or_source_path(self):
        items,_,annotation,_=fixture()
        serialized=json.dumps([items,annotation])
        for forbidden in ("fictional_model","synthetic_a","candidate_a","selected","model_id","report_path","positive"):
            self.assertNotIn(forbidden,serialized)

    def test_queue_and_rows_are_deterministic_and_source_order_invariant(self):
        items,resolver,_,_=fixture()
        candidates=[{key:value for key,value in row.items() if key!="item_id"} for row in resolver]
        self.assertEqual(review.make_queue(candidates),review.make_queue(list(reversed(candidates))))
        self.assertEqual(review.annotation_template(items),review.annotation_template(items))

    def test_duplicate_text_is_reviewed_once_lineages_retained(self):
        _,resolver,_,_=fixture()
        row={k:v for k,v in resolver[0].items() if k!="item_id"}
        items,links=review.make_queue([row,{**row,"candidate_id":"candidate_b"}])
        self.assertEqual(len(items),1);self.assertEqual(len(links),2)
        self.assertEqual(links[0]["item_id"],links[1]["item_id"])

    def test_duplicate_candidates_bad_hash_empty_and_oversize_refused(self):
        _,resolver,_,_=fixture()
        for candidates in ([],resolver*2,resolver*49,[{**resolver[0],"report_sha256":"bad"}]):
            with self.assertRaises(ValueError):review.make_queue(candidates)

    def test_every_finding_in_every_report_remains_pending(self):
        _,_,a,_=fixture()
        self.assertEqual(len(a["records"]),4)
        self.assertEqual({r["finding"] for r in a["records"]},set(FINDINGS))
        self.assertTrue(all(r["status"]=="pending" and r["state"] is None for r in a["records"]))

    def test_template_does_not_use_any_extractor_or_gold(self):
        source=inspect.getsource(review.annotation_template)
        for name in ("chexbert","qwen","scope_check","expected","positive"):
            self.assertNotIn(name,source)


class HumanContractTests(unittest.TestCase):
    def test_pending_does_not_open_source_or_get_unknown_match_credit(self):
        items,_,a,_=fixture()
        read=unittest.mock.Mock(side_effect=AssertionError("no source access"))
        result=review.reader_agreement(items,a,copy.deepcopy(a),read_source=read)
        read.assert_not_called()
        self.assertEqual(result["both_reviewed"],0)
        self.assertEqual(result["both_reviewed_coverage"],0)
        self.assertIsNone(result["exact_reader_agreement"])
        self.assertIsNone(result["cohens_kappa_nominal_four_states"])
        self.assertFalse(result["adjudicated_gold_created"])

    def test_pending_with_fabricated_unknown_refused(self):
        items,_,a,_=fixture();a["records"][0]["state"]="unknown"
        with self.assertRaisesRegex(ValueError,"pending"):
            review.validate_annotations(items,a)

    def test_label_requires_alias_role_and_independent_human_attestation(self):
        items,_,a,texts=fixture();done=complete_one(a,next(iter(texts.values())))
        for key,value in (("reviewer_alias",None),("reviewer_alias","name@example.com"),("reviewer_role","LLM"),("independence_attestation",None)):
            bad={**done,key:value}
            with self.assertRaises(ValueError):review.validate_annotations(items,bad,read_source=texts.__getitem__)

    def test_model_prefilled_or_unblinded_attestation_refused(self):
        items,_,a,texts=fixture();done=complete_one(a,next(iter(texts.values())))
        for key in done["independence_attestation"]:
            bad=copy.deepcopy(done);bad["independence_attestation"][key]=True
            with self.assertRaises(ValueError):review.validate_annotations(items,bad,read_source=texts.__getitem__)

    def test_missing_duplicate_extra_or_hash_changed_rows_refused(self):
        items,_,a,_=fixture()
        for records in (a["records"][:-1],a["records"]+[a["records"][0]],
                [{**row,"report_sha256":"wrong"} for row in a["records"]],
                [{**row,"winner":True} for row in a["records"]]):
            with self.assertRaises(ValueError):review.validate_annotations(items,{**a,"records":records})

    def test_changed_header_or_inventory_refused(self):
        items,_,a,_=fixture()
        for key in ("schema_version","protocol_version","inventory_sha256","input_scope"):
            with self.assertRaises(ValueError):review.validate_annotations(items,{**a,key:"wrong"})

    def test_completed_review_requires_source_check(self):
        items,_,a,texts=fixture();done=complete_one(a,next(iter(texts.values())))
        with self.assertRaisesRegex(ValueError,"source verification"):
            review.validate_annotations(items,done)

    def test_source_hash_quote_offsets_and_unicode_binding(self):
        items,_,a,texts=fixture("虚构\nCardiomegaly is present.")
        done=complete_one(a,next(iter(texts.values())))
        self.assertEqual(len(review.validate_annotations(items,done,read_source=texts.__getitem__)),4)
        with self.assertRaisesRegex(ValueError,"source text hash"):
            review.validate_annotations(items,done,read_source=lambda _:"changed")
        bad=copy.deepcopy(done);bad["records"][0]["evidence"][0]["char_start"]=1
        with self.assertRaisesRegex(ValueError,"source quote"):
            review.validate_annotations(items,bad,read_source=texts.__getitem__)

    def test_asserted_state_without_evidence_refused(self):
        items,_,a,texts=fixture();done=complete_one(a,next(iter(texts.values())))
        done["records"][0]["evidence"]=[]
        with self.assertRaisesRegex(ValueError,"source evidence"):
            review.validate_annotations(items,done,read_source=texts.__getitem__)

    def test_reviewed_unknown_is_not_missing_or_negative(self):
        items,_,a,texts=fixture("Invented acquisition note.")
        done=complete_one(a,next(iter(texts.values())),state="unknown",reason="not_mentioned")
        result=review.validate_annotations(items,done,read_source=texts.__getitem__)
        self.assertEqual(result[items[0]["item_id"],"cardiomegaly"]["state"],"unknown")

    def test_unassessable_cannot_supply_disease_state(self):
        items,_,a,texts=fixture();done=complete_one(a,next(iter(texts.values())))
        done["records"][0].update(status="unassessable",state=None,reason="unassessable",evidence=[])
        review.validate_annotations(items,done,read_source=texts.__getitem__)
        done["records"][0]["state"]="negative"
        with self.assertRaises(ValueError):review.validate_annotations(items,done,read_source=texts.__getitem__)

    def test_one_reader_cannot_be_two_independent_votes(self):
        items,_,a,texts=fixture();done=complete_one(a,next(iter(texts.values())))
        with self.assertRaisesRegex(ValueError,"distinct human"):
            review.reader_agreement(items,done,copy.deepcopy(done),read_source=texts.__getitem__)

    def test_disagreement_is_not_automatically_adjudicated_or_repaired(self):
        items,_,a,texts=fixture();text=next(iter(texts.values()))
        left=complete_one(a,text);right=complete_one(a,text,alias="reader_b",state="negative")
        result=review.reader_agreement(items,left,right,read_source=texts.__getitem__)
        self.assertEqual(result["both_reviewed"],1);self.assertEqual(result["disagreements"],1)
        self.assertEqual(result["both_reviewed_coverage"],.25)
        self.assertFalse(result["model_or_rules_used_to_break_ties"])
        self.assertFalse(result["regeneration_authorized"])

    def test_constant_all_unknown_kappa_undefined_not_one(self):
        items,_,a,texts=fixture("Invented acquisition note.");text=next(iter(texts.values()))
        left=complete_one(a,text,state="unknown",reason="not_mentioned")
        right=complete_one(a,text,alias="reader_b",state="unknown",reason="not_mentioned")
        result=review.reader_agreement(items,left,right,read_source=texts.__getitem__)
        self.assertIsNone(result["cohens_kappa_nominal_four_states"])
        self.assertEqual(result["exact_reader_agreement"],1)

    def test_no_input_mutation_or_json_type_change(self):
        inputs=fixture();before=copy.deepcopy(inputs)
        result=review.reader_agreement(inputs[0],inputs[2],copy.deepcopy(inputs[2]))
        self.assertEqual(inputs,before);self.assertEqual(result,json.loads(json.dumps(result)))


class OperationalSafetyTests(unittest.TestCase):
    def test_materializer_requires_explicit_copy_scope_and_slurm_before_packet(self):
        for job,permission in ((None,False),(None,True),("invented",False)):
            env={} if job is None else {"SLURM_JOB_ID":job}
            with patch.dict(os.environ,env,clear=True),patch.object(materializer,"require_inside") as access:
                with self.assertRaisesRegex(RuntimeError,"authorization"):
                    materializer.run(SimpleNamespace(allow_synthetic_report_copy=permission),Path("not_written"))
                access.assert_not_called()

    def test_materializer_copy_preserves_invented_text_and_does_not_create_gold(self):
        items,resolver,template,texts=fixture("虚构\nCardiomegaly is present.\n")
        text=next(iter(texts.values()))
        summary={"report_text_opened":False,"candidate_reports":48,"independent_ehr_cases":2,
            "previously_used_development_cohort":True,"regeneration_authorized":False}
        manifest={"schema_version":builder.SCHEMA}
        def read(path):
            if path.name=="manifest.json":return manifest
            if path.name=="summary.json":return summary
            if path.name=="items.json":return {"items":items}
            if path.name=="resolver.json":return {"records":resolver}
            return template
        def checked(packet,name):return packet/name,manifest
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented"}),\
                patch.object(materializer,"require_inside",side_effect=lambda p,*a,**k:Path(p)),\
                patch.object(materializer,"read_json",side_effect=read),\
                patch.object(materializer,"checked_artifact",side_effect=checked),\
                patch.object(Path,"stat",return_value=SimpleNamespace(st_size=len(text.encode()))),\
                patch.object(Path,"read_bytes",return_value=text.encode()) as source,\
                patch.object(materializer,"private_directory"),\
                patch.object(materializer,"write_private_text",side_effect=lambda p,t:p) as written,\
                patch.object(materializer,"write_private_json",side_effect=lambda p,t:p),\
                patch.object(materializer,"sha256_file",return_value=digest(text)):
            result,files=materializer.run(SimpleNamespace(allow_synthetic_report_copy=True,packet_run="invented_packet"),Path("invented_output"))
        source.assert_called_once()
        self.assertEqual(written.call_args_list[0].args[1],text)
        self.assertEqual(result["human_annotations_completed"],0)
        self.assertTrue(result["copied_reports_byte_identical"])
        self.assertFalse(result["real_inputs_read"])
        self.assertFalse(result["regeneration_authorized"])
        self.assertEqual(len(files),5)

    def test_metadata_builder_has_no_report_or_image_reader(self):
        source=inspect.getsource(builder.load_metadata)
        for forbidden in ("read_report_text","read_bytes","Image.open","load_cxr_candidates","scope_check","torch"):
            self.assertNotIn(forbidden,source)

    def test_freeze_detects_changed_rule_before_human_review(self):
        with patch.object(builder,"sha256_file",return_value="changed"):
            with self.assertRaisesRegex(ValueError,"freeze differs"):builder.freeze_method()

    def test_source_guard_precedes_paths_and_any_content_access(self):
        read=audit.approved_source_reader([{"report_sha256":"hash","report_path":"do_not_open"}])
        with patch.dict(os.environ,{},clear=True),patch.object(audit,"require_inside") as access:
            with self.assertRaisesRegex(RuntimeError,"Slurm"):read("hash")
            access.assert_not_called()

    def test_pending_cannot_produce_extractor_accuracy(self):
        items,_,a,_=fixture()
        result=audit.provisional_comparison(items,a,copy.deepcopy(a),[])
        self.assertEqual(result["review_item_finding_denominator"],4)
        self.assertEqual(result["two_reader_matching_states"],0)
        self.assertEqual(result["matching_state_coverage"],0)
        self.assertEqual(result["cached_extractors_against_provisional_reader_agreement"],{"chexbert":None,"qwen":None,"frozen_literal_readout":None})
        self.assertIsNone(result["scope_gate_risk_coverage_on_reader_agreement"])

    def test_failed_qwen_does_not_get_credit_for_reviewed_unknown(self):
        items,_,a,texts=fixture("Invented acquisition note.");text=next(iter(texts.values()))
        left=complete_one(a,text,state="unknown",reason="not_mentioned")
        right=complete_one(a,text,alias="reader_b",state="unknown",reason="not_mentioned")
        pred={"candidate_id":"candidate_a","report_sha256":items[0]["report_sha256"],"item_id":items[0]["item_id"],"chexbert":dict.fromkeys(FINDINGS,"unknown"),
            "qwen":dict.fromkeys(FINDINGS,"unknown"),"qwen_contract_status":"failed_unavailable"}
        result=audit.provisional_comparison(items,left,right,[pred])
        q=result["cached_extractors_against_provisional_reader_agreement"]["qwen"]
        self.assertEqual(q["correct"],0);self.assertEqual(q["unavailable"],1)
        self.assertFalse(result["primary_metric_eligible"])

    def test_frozen_rules_and_gate_have_separate_metrics_not_new_gold(self):
        items,_,a,texts=fixture();text=next(iter(texts.values()))
        left=complete_one(a,text);right=complete_one(a,text,alias="reader_b")
        states=dict.fromkeys(FINDINGS,"unknown");states["cardiomegaly"]="positive"
        pred={"candidate_id":"candidate_a","report_sha256":items[0]["report_sha256"],"item_id":items[0]["item_id"],
            "chexbert":states,"qwen":states,"qwen_contract_status":"complete"}
        result=audit.provisional_comparison(items,left,right,[pred],read_source=texts.__getitem__)
        self.assertEqual(result["cached_extractors_against_provisional_reader_agreement"]["frozen_literal_readout"]["checks"],1)
        self.assertEqual(result["scope_gate_risk_coverage_on_reader_agreement"]["scope_commits"],1)
        self.assertFalse(result["adjudicated_gold_created"])

    def test_same_text_duplicate_lineage_not_independent_double_credit(self):
        items,_,a,texts=fixture("Invented acquisition note.");text=next(iter(texts.values()))
        left=complete_one(a,text,state="unknown",reason="not_mentioned")
        right=complete_one(a,text,alias="reader_b",state="unknown",reason="not_mentioned")
        pred={"candidate_id":"candidate_a","report_sha256":items[0]["report_sha256"],"item_id":items[0]["item_id"],
            "chexbert":dict.fromkeys(FINDINGS,"unknown"),"qwen":dict.fromkeys(FINDINGS,"unknown"),"qwen_contract_status":"complete"}
        result=audit.provisional_comparison(items,left,right,[pred,{**pred,"candidate_id":"candidate_b"}])
        self.assertEqual(result["cached_extractors_against_provisional_reader_agreement"]["chexbert"]["checks"],1)
        with self.assertRaisesRegex(ValueError,"duplicate frozen"):
            audit.provisional_comparison(items,left,right,[pred,pred])


if __name__ == "__main__":
    unittest.main()

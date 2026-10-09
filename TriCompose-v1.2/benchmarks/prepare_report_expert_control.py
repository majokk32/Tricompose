#!/usr/bin/env python3
"""CPU-only plan: existing same-image four experts, fresh report verification.

No model instantiation, report-body/image-pixel reading or legacy XRV reuse.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src","TriCompose-v1.2/benchmarks","TriCompose-v1.1/src",
                 "TriCompose-v1.0/src","src","TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0,str(ROOT.parent/relative))
from contracts import (WORKSPACE,PROTECTED_ROOT,require_inside,read_json,sha256_file,load_cxr_candidates,
    load_report_candidates,new_atomic_run,write_private_json,private_directory,commit_atomic_run,discard_atomic_run)
from run_report_nbest import load as load_nbest_plan,single_view
from tricompose_v12.report_expert_control import SCHEMA,POLICY,MODELS,SECTION_CONTRACTS
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v12.live_workers import source_pins,check_pins
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt
from tricompose_v12.full_bank_secondary import SCHEMA as FULL_SCHEMA


def prepare(args):
    require_slurm(); before=source_pins()
    source=require_inside(args.source_nbest_run,PROTECTED_ROOT,must_exist=True)
    mp=source/"manifest.json"
    if sha256_file(mp)!=args.source_nbest_manifest_sha256: raise ValueError("approved n-best source changed")
    m=read_json(mp)
    if (m.get("status")!="completed_engineering_nbest_control_unvalidated"
            or m.get("original_ehr_cxr_or_winners_changed") is not False or m.get("clinical_acceptance") is not False):
        raise ValueError("unchanged completed synthetic source required")
    proof_root=require_inside(args.source_nbest_audit,PROTECTED_ROOT,must_exist=True)
    proof_manifest=read_json(proof_root/"manifest.json")
    proof=read_json(proof_root/"audit.json")
    if (sha256_file(proof_root/"audit.json")!=proof_manifest["audit_sha256"]
            or proof.get("status")!="metadata_hash_receipt_audit_passed"
            or proof["source_manifest_sha256"]!=args.source_nbest_manifest_sha256):
        raise ValueError("source result audit changed")
    args.plan_run=args.source_plan;args.plan_manifest_sha256=m["plan_manifest_sha256"]
    original=load_nbest_plan(args)
    cached_xrv=source/"xrv/scored/cxr_finding_labels.json"
    if sha256_file(cached_xrv)!=m["xrv_labels_sha256"]: raise ValueError("fresh image-label source changed")
    image_labels=read_json(cached_xrv)
    fixed=original["fixed_cases"]
    partials=[]
    for case in fixed:
        anchor=anchor_from_record(case["anchor"]);image=case["image"]
        partials.append(image_receipt(anchor,image,single_view(image_labels,"cxr_candidates","cxr_candidate_id",image["candidate_id"]),
            label_sha256=sha256_file(cached_xrv),thresholds_sha256=original["xrv"]["thresholds_sha256"],
            checkpoint_sha256=original["xrv"]["checkpoint_sha256"]))
    full=require_inside(args.full_bank_run,PROTECTED_ROOT,must_exist=True)
    fmp=full/"manifest.json"
    if sha256_file(fmp)!=args.full_bank_manifest_sha256: raise ValueError("complete-bank registry changed")
    fm=read_json(fmp)
    if fm.get("schema_version")!=FULL_SCHEMA or fm.get("original_selection_changed") is not False:
        raise ValueError("immutable synthetic report registry required")
    pins={str(mp):sha256_file(mp),str(proof_root/"manifest.json"):sha256_file(proof_root/"manifest.json"),
        str(proof_root/"audit.json"):sha256_file(proof_root/"audit.json"),str(cached_xrv):sha256_file(cached_xrv),str(fmp):sha256_file(fmp)}
    pins.update(original["artifact_pins"])
    report_runs,cxr_runs=[],[]
    for key,value in fm["source_paths"].items():
        if key.startswith(("cxr_manifest_","report_manifest_")):
            path=require_inside(value,PROTECTED_ROOT,must_exist=True)
            if sha256_file(path)!=fm["source_sha256"][key]: raise ValueError("generation registry changed")
            pins[str(path)]=sha256_file(path)
            (cxr_runs if key.startswith("cxr_") else report_runs).append(path.parent)
    cxrs=load_cxr_candidates(cxr_runs);reports=load_report_candidates(report_runs,cxr_candidates=cxrs)
    subset={}
    for case in fixed:
        image=case["image"]
        if cxrs[image["candidate_id"]]!=image: raise ValueError("same historical fixed image required")
        for model in MODELS:
            matches=[r for r in reports.values() if r["parent_cxr_candidate_id"]==image["candidate_id"] and r["model_id"]==model]
            if len(matches)!=1: raise ValueError("exact four report experts per fixed image required")
            report=matches[0]
            if (report.get("model_input_signature")!="single_current_synthetic_cxr"
                    or report.get("structured_ehr_content_supplied_to_model") is not False
                    or report.get("source_report_or_real_target_supplied") is not False):
                raise ValueError("CXR-only fully synthetic report contract required")
            if model=="cxrmate_single" and report["artifact"]["sha256"]!=case["historical_cxrmate_report_sha256"]:
                raise ValueError("historical baseline changed")
            pins[report["artifact"]["path"]]=report["artifact"]["sha256"]
            subset[(image["candidate_id"],model)]=report
    if sha256_file(source/"reports/manifest.json")!=m["generator_manifest_sha256"]:
        raise ValueError("baseline generation manifest changed")
    generator=read_json(source/"reports/manifest.json")
    if not all(c["historical_top1_sha256_equal"] is True for c in generator["native_calls"]):
        raise ValueError("historical/current top-one baseline equivalence required")
    pins[str(source/"reports/manifest.json")]=sha256_file(source/"reports/manifest.json")
    for spec in (original["xrv"],original["chexbert"]): check_pins(spec["asset_pins"])
    check_pins(original["biovil_asset_pins"])
    from evaluate_report_structure import MODEL_SECTION_CONTRACT
    if any(MODEL_SECTION_CONTRACT[m]!=SECTION_CONTRACTS[m] for m in MODELS):
        raise ValueError("official per-model section contracts changed")
    check_pins(before);check_pins(pins)
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        outputs=[]
        for model in MODELS:
            directory=temporary/model;private_directory(directory)
            records=[];revisions=set()
            for case in fixed:
                report=subset[(case["image"]["candidate_id"],model)];name=report["candidate_id"]+".json"
                path=write_private_json(directory/name,report);revisions.add(report["model_revision"])
                records.append({"candidate_id":report["candidate_id"],"case_id":report["case_id"],"path":name,"sha256":sha256_file(path)})
            if len(revisions)!=1: raise ValueError("frozen expert revision changed within image pair")
            write_private_json(directory/"manifest.json",{"schema_version":"tricompose-report-candidate-run-v1.1",
                "model_id":model,"model_revision":next(iter(revisions)),"frozen_model":True,"candidate_count":2,
                "candidates":records,"metadata_copy_only":True,"new_report_generation_calls":0})
            outputs.append(str(target/model))
            for path in directory.glob("*.json"): pins[str(target/model/path.name)]=sha256_file(path)
        payload={"schema_version":SCHEMA,"policy":POLICY,"fixed_cases":fixed,"image_receipts":partials,
            "cxr_run":original["cxr_run"],"report_runs":outputs,"reused_xrv_labels":str(cached_xrv),
            "xrv":original["xrv"],"chexbert":original["chexbert"],"biovil_model":original["biovil_model"],
            "biovil_python":original["biovil_python"],"biovil_asset_pins":original["biovil_asset_pins"],
            "source_pins":before,"artifact_pins":pins,"source_nbest_manifest_sha256":args.source_nbest_manifest_sha256,
            "factory_instantiated":False,"new_model_calls":0,"source_bodies_parsed":False,
            "legacy_xrv_or_report_labels_used":False,"reused_xrv_profile":"exact_prior_fresh_eight_enabled_heads_same_images",
            "report_generator_weights_loaded":False,"planned_new_generation_calls":0,
            "planned_counts":{"report_candidates":8,"reused_xrv_scored_images":2,"new_chexbert_scored_reports":8,
                "maximum_biovil_image_encodings":2,"maximum_biovil_text_encodings":8},"clinical_acceptance":False}
        path=write_private_json(temporary/"plan.json",payload)
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,"plan_sha256":sha256_file(path),
            "status":"prepared_four_cached_experts_fresh_profile_verification","new_model_calls":0,
            "planned_counts":payload["planned_counts"],"clinical_acceptance":False})
        commit_atomic_run(temporary,target)
    except Exception:
        discard_atomic_run(temporary);raise
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-nbest-run","source-nbest-manifest-sha256","source-nbest-audit","source-plan",
                 "full-bank-run","full-bank-manifest-sha256","output-root","run-id"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args();os.umask(0o007);started=time.monotonic()
    try:root=prepare(args)
    except Exception as exc:
        print(json.dumps({"status":"preflight_failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":"prepared_four_cached_experts_fresh_profile_verification","new_model_calls":0,
        "runtime_seconds":round(time.monotonic()-started,3),"manifest_sha256":sha256_file(root/"manifest.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

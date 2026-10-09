#!/usr/bin/env python3
"""Prepare/score the missing full-bank endpoint pairs, without policy changes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT.parent/"TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.complete_endpoint_inventory import SCHEMA, EXPERIMENT, missing_pairs
from score_automatic_replay_biovil import REQUEST_SCHEMA, MODEL_HASHES, score, validate_pairs
from contracts import (PROTECTED_ROOT,WORKSPACE,RUN_ID_PATTERN,require_inside,read_json,sha256_file,load_cxr_candidates,
    load_report_candidates,new_atomic_run,commit_atomic_run,discard_atomic_run,write_private_json)


def checked(root,name,schema,limit):
    root=require_inside(root,PROTECTED_ROOT,must_exist=True)
    mp=root/"manifest.json"; manifest=read_json(mp)
    path=require_inside(root/name,root,must_exist=True)
    if (manifest.get("schema_version")!=schema or path.stat().st_size>limit
            or sha256_file(path)!=manifest["artifacts"][name]["sha256"]):
        raise ValueError("hash-bound bounded cached metadata required")
    return mp,path,manifest


def pins(args):
    model=require_inside(args.model_path,WORKSPACE,must_exist=True)
    assets={name:model/name for name in MODEL_HASHES}
    if any(sha256_file(p)!=MODEL_HASHES[name] for name,p in assets.items()):
        raise ValueError("frozen BioViL model bytes differ")
    return {"program":Path(__file__),"inventory_contract":ROOT/"src/tricompose_v12/complete_endpoint_inventory.py",
        "score_worker":ROOT/"benchmarks/score_automatic_replay_biovil.py",
        "runtime_adapter":ROOT.parent/"TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py",
        **{"model_"+name:path for name,path in assets.items()}}


def prepare(args):
    dm,dp,diagnostic=checked(args.diagnostic_run,"historical_fourteen_field_cache_candidate_coverage.csv",
        "tricompose-score-coverage-diagnostic-v1",4*1024*1024)
    pm,pp,prior=checked(args.previous_score_run,"scores.json",
        "tricompose-automatic-replay-secondary-biovil-v1",1024*1024)
    old=read_json(pp)
    if (old.get("used_for_routing") is not False or old.get("clinical_truth_available") is not False
            or old.get("producer",{}).get("frozen") is not True
            or old["producer"].get("checkpoint_sha256")!=MODEL_HASHES
            or old["producer"].get("text_policy")!="full_report_no_silent_truncation_overlength_is_na"):
        raise ValueError("frozen secondary-only full-report prior scores required")
    with dp.open(newline="") as handle:inventory=list(csv.DictReader(handle))
    pending,counts=missing_pairs(inventory,old["records"])
    if (counts["all_candidate_pairs"]!=960 or len({r["case_id"] for r in inventory})!=80
            or len({r["cxr_candidate_id"] for r in inventory})!=240 or not pending):
        raise ValueError("entire eighty-case inventory and nonempty missing set required")
    sources={"diagnostic_manifest":dm,"candidate_inventory":dp,"previous_score_manifest":pm,"previous_scores":pp,**pins(args)}
    for prefix,m in (("diagnostic",diagnostic),):
        for k,path in m["source_paths"].items():
            path=require_inside(path,WORKSPACE,must_exist=True)
            if sha256_file(path)!=m["source_sha256"][k]:raise ValueError("diagnostic source changed")
            sources[prefix+"_"+k]=path
    cxrs=load_cxr_candidates(args.cxr_run)
    reports=load_report_candidates(args.report_run,cxr_candidates=cxrs)
    all_pairs=[{k:r[k] for k in pending[0]} for r in inventory]
    validate_pairs(all_pairs,cxrs,reports)
    if len(cxrs)!=240 or len(reports)!=960:raise ValueError("full frozen artifact inventory required")
    for modality,runs in (("cxr",args.cxr_run),("report",args.report_run)):
        for i,root in enumerate(runs):sources[f"{modality}_manifest_{i}"]=Path(root)/"manifest.json"
    return {"schema_version":REQUEST_SCHEMA,"experiment":EXPERIMENT,"pairs":pending,"inventory_counts":counts,
        "modality_source":"fully_synthetic","selection_used_biovil":False,"clinical_truth_available":False,
        "routing_or_calibration_update_allowed":False,"text_policy":"full_report_no_silent_truncation_overlength_is_na",
        "cxr_runs":[str(Path(p).resolve(strict=True)) for p in args.cxr_run],
        "report_runs":[str(Path(p).resolve(strict=True)) for p in args.report_run]},sources


def execute(args):
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("existing Slurm allocation required")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):raise ValueError("opaque run ID required")
    target=require_inside(Path(args.output_root)/args.run_id,PROTECTED_ROOT,must_exist=False)
    if target.exists():raise FileExistsError("output run exists; refuse work before model loading")
    started=time.monotonic()
    if args.mode=="prepare":
        payload,sources=prepare(args)
        schema=REQUEST_SCHEMA;name="request.json"
        before={k:sha256_file(p) for k,p in sources.items()}
    else:
        # CUDA guard runs before score creates/loads any model; no login inference.
        import torch
        if not torch.cuda.is_available():raise RuntimeError("approved GPU Slurm required")
        mp,rp,plan=checked(args.request_run,"request.json",REQUEST_SCHEMA,1024*1024)
        request=read_json(rp)
        if request.get("experiment")!=EXPERIMENT or plan.get("preflight_status")!="complete_bank_hash_preflight_passed":
            raise ValueError("full-bank frozen plan required")
        sources={k:require_inside(p,WORKSPACE,must_exist=True) for k,p in plan["source_paths"].items()}
        before=plan["source_sha256"]
        if any(sha256_file(p)!=before[k] for k,p in sources.items()):raise ValueError("frozen plan asset changed")
        args.cxr_run=request["cxr_runs"];args.report_run=request["report_runs"]
        payload=score(args)
        payload.update(experiment=EXPERIMENT,inventory_counts=request["inventory_counts"])
        schema=SCHEMA;name="scores.json"
    if any(sha256_file(p)!=before[k] for k,p in sources.items()):raise ValueError("source changed during work")
    payload["runtime_seconds"]=round(time.monotonic()-started,6)
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        file=write_private_json(temporary/name,payload)
        write_private_json(temporary/"manifest.json",{"schema_version":schema,"experiment":EXPERIMENT,
            "artifacts":{name:{"sha256":sha256_file(file)}},"source_paths":{k:str(p) for k,p in sources.items()},
            "source_sha256":before,"preflight_status":"complete_bank_hash_preflight_passed",
            "request_manifest_sha256":sha256_file(mp) if args.mode=="score" else None,
            "new_generation_samples":0,"original_selection_changed":False,"clinical_acceptance":False})
        commit_atomic_run(temporary,target)
    except Exception:
        discard_atomic_run(temporary);raise
    return target,payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("prepare","score"))
    for name in ("output-root","run-id","model-path"):p.add_argument("--"+name,required=True)
    for name in ("diagnostic-run","previous-score-run","request-run"):p.add_argument("--"+name)
    for name in ("cxr-run","report-run"):p.add_argument("--"+name,action="append")
    args=p.parse_args();os.umask(0o007)
    try:target,payload=execute(args)
    except Exception as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":"prepared_complete_bank_request" if args.mode=="prepare" else payload["status"],
        "runtime_seconds":payload["runtime_seconds"],"manifest_sha256":sha256_file(target/"manifest.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

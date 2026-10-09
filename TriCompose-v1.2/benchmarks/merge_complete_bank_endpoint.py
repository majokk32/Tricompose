#!/usr/bin/env python3
"""CPU-only completed endpoint audit/merge and full-bank tie diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT.parent/"TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.full_bank_secondary import SCHEMA,merge
from tricompose_v12.complete_endpoint_inventory import SCHEMA as SCORE_MANIFEST_SCHEMA,EXPERIMENT
from tricompose_v12.score_coverage import LEGACY_PROFILE,analyze
from complete_bank_endpoint import checked
from diagnose_score_coverage import historical,markdown
from score_automatic_replay_biovil import REQUEST_SCHEMA,MODEL_HASHES
from run_automatic_proxy_replay import render_csv
from contracts import (PROTECTED_ROOT,WORKSPACE,require_inside,read_json,sha256_file,new_atomic_run,
    commit_atomic_run,discard_atomic_run,write_private_json,write_private_text)


def same_sources(left,right):
    """Relative/absolute spellings may differ; resolved boundary and hashes may not."""
    return set(left)==set(right) and all(
        require_inside(left[k],WORKSPACE,must_exist=True)==require_inside(right[k],WORKSPACE,must_exist=True)
        for k in left)


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("existing CPU Slurm allocation required")
    started=time.monotonic()
    pm,pp,plan=checked(args.request_run,"request.json",REQUEST_SCHEMA,1024*1024)
    nm,np,nmdata=checked(args.score_run,"scores.json",SCORE_MANIFEST_SCHEMA,1024*1024)
    request,new=read_json(pp),read_json(np)
    if (request.get("experiment")!=EXPERIMENT or new.get("experiment")!=EXPERIMENT
            or nmdata.get("request_manifest_sha256")!=sha256_file(pm)
            or new.get("request_sha256")!=sha256_file(pp)
            or nmdata.get("source_sha256")!=plan["source_sha256"]
            or not same_sources(nmdata.get("source_paths",{}),plan["source_paths"])
            or nmdata.get("new_generation_samples")!=0 or nmdata.get("original_selection_changed") is not False):
        raise ValueError("completed frozen-request score provenance required")
    sources={k:require_inside(p,WORKSPACE,must_exist=True) for k,p in plan["source_paths"].items()}
    if any(sha256_file(p)!=plan["source_sha256"][k] for k,p in sources.items()):
        raise ValueError("frozen source/checkpoint bytes changed")
    with sources["candidate_inventory"].open(newline="") as handle:inventory=list(csv.DictReader(handle))
    old=read_json(sources["previous_scores"])
    records,summary=merge(inventory,old,new,request,MODEL_HASHES)
    original,_=historical(args)
    endpoint={r["triple_candidate_id"]:r for r in records}
    # Rehydrate only authenticated synthetic cached states, never artifact bodies.
    from diagnose_automatic_discrepancy import load as load_historical
    _,bank,_,evidence_sources=load_historical(args)
    raw=[];diagnostic_rows={r["triple_candidate_id"]:r for r in original["rows"]}
    for case,grid in bank.items():
        for candidate in grid.values():
            row=candidate["score_record"];e=endpoint[row["triple_candidate_id"]]
            r=diagnostic_rows[row["triple_candidate_id"]]
            raw.append({**{k:r[k] for k in ("case_id","cxr_candidate_id","report_candidate_id","report_model_id",
                "triple_candidate_id","ehr_sha256","ehr_facts_sha256","cxr_sha256","report_sha256")},
                "profile":LEGACY_PROFILE,"source_primary_prefix":r["source_primary_prefix"],
                "fact_states":[{"finding":f["finding"],**f["states"]} for f in candidate["facts"]],
                "biovil_raw_cosine":e["biovil_raw_cosine"],"endpoint_unavailable_reason":e["reason"]})
    complete=analyze(raw,LEGACY_PROFILE)
    sources.update({"historical_"+k:p for k,p in evidence_sources.items()})
    sources.update(request_manifest=pm,request=pp,new_score_manifest=nm,new_scores=np,
        merge_program=Path(__file__),merge_contract=ROOT/"src/tricompose_v12/full_bank_secondary.py")
    before={k:sha256_file(p) for k,p in sources.items()}
    summary.update({"status":"completed_full_bank_secondary_merge_and_coverage_audit",
        "original_selected_union_diagnostic":original["summary"],"full_bank_diagnostic":complete["summary"],
        "new_merge_model_calls":0,"source_artifacts_unchanged":True,
        "runtime_seconds_including_hash_audit":round(time.monotonic()-started,6)})
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        flat=[{k:v for k,v in r.items() if k!="raw_edge_readouts"} | {
            f"{edge}_{field}":value for edge,values in r["raw_edge_readouts"].items()
            for field,value in values.items()} for r in complete["rows"]]
        files=[write_private_json(temporary/"summary.json",summary),write_private_json(temporary/"scores.json",{"records":records}),
            write_private_text(temporary/"candidate_score_table.csv",render_csv(flat)),
            write_private_text(temporary/"same_image_ties.csv",render_csv(complete["same_image_pairs"])),
            write_private_text(temporary/"model_coverage.csv",render_csv(complete["model_coverage"])),
            write_private_text(temporary/"RESULTS_CN_EN.md",markdown({"original_selected_union":original,"full_bank_endpoint":complete}))]
        if before!={k:sha256_file(p) for k,p in sources.items()}:raise ValueError("immutable merge source changed")
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,
            "source_paths":{k:str(p) for k,p in sources.items()},"source_sha256":before,
            "artifacts":{p.name:{"sha256":sha256_file(p)} for p in files},"new_model_calls":0,
            "original_selection_changed":False,"clinical_acceptance":False})
        commit_atomic_run(temporary,target)
    except Exception:
        discard_atomic_run(temporary);raise
    return target,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("request-run","score-run","endpoint-run","output-root","run-id"):p.add_argument("--"+name,required=True)
    args=p.parse_args();os.umask(0o007)
    try:target,summary=run(args)
    except Exception as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":summary["status"],"runtime_seconds":summary["runtime_seconds_including_hash_audit"],
        "manifest_sha256":sha256_file(target/"manifest.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

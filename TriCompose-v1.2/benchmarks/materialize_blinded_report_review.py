#!/usr/bin/env python3
"""Approved Slurm-only copying of synthetic reports into opaque review files.

No real inputs, images, EHR, model/API calls, label changes or report edits.
This command requires explicit synthetic-copy authorization in addition to
Slurm; presence of a job ID is not itself user authorization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_assertions import digest
from contracts import (PROTECTED_ROOT,require_inside,read_json,sha256_file,
    new_atomic_run,commit_atomic_run,discard_atomic_run,private_directory,
    write_private_json,write_private_text)
from prepare_blinded_report_review import SCHEMA,checked_artifact,instructions


def run(args,temporary):
    if not os.environ.get("SLURM_JOB_ID") or not args.allow_synthetic_report_copy:
        raise RuntimeError("explicit synthetic-copy authorization and approved Slurm required")
    packet=require_inside(args.packet_run,PROTECTED_ROOT,must_exist=True)
    manifest=read_json(packet/"manifest.json")
    summary_path,_=checked_artifact(packet,"summary.json")
    summary=read_json(summary_path)
    if (manifest.get("schema_version") != SCHEMA or summary.get("report_text_opened") is not False
            or summary.get("candidate_reports") != 48 or summary.get("independent_ehr_cases") != 2
            or summary.get("previously_used_development_cohort") is not True
            or summary.get("regeneration_authorized") is not False):
        raise ValueError("bounded synthetic development packet required")
    for name in ("reviewer/items.json","reviewer/reviewer_a_template.json","reviewer/reviewer_b_template.json",
            "investigator/resolver.json"):
        checked_artifact(packet,name)
    items=read_json(packet/"reviewer/items.json")["items"]
    resolver=read_json(packet/"investigator/resolver.json")["records"]
    by_item={}
    for row in resolver:
        previous=by_item.setdefault(row["item_id"],row)
        if previous["report_sha256"] != row["report_sha256"]:
            raise ValueError("one review item maps to incompatible report hashes")
    if set(by_item) != {row["item_id"] for row in items} or not 1 <= len(items) <= 48:
        raise ValueError("complete blind inventory required")
    private_directory(temporary/"reports")
    files=[]
    for item in items:
        row=by_item[item["item_id"]]
        if row["report_sha256"] != item["report_sha256"] or item["report_relative_path"] != f"reports/{item['item_id']}.txt":
            raise ValueError("review source/opaque path binding differs")
        path=require_inside(row["report_path"],PROTECTED_ROOT,must_exist=True)
        if path.stat().st_size > 32768:
            raise ValueError("source report exceeds reviewed bound")
        text=path.read_bytes().decode("utf-8")
        if not text.strip() or len(text)>8192 or digest(text)!=item["report_sha256"]:
            raise ValueError("source synthetic report text/hash differs")
        copied=write_private_text(temporary/item["report_relative_path"],text)
        if sha256_file(copied)!=item["report_sha256"]:
            raise ValueError("blind copy not byte-identical")
        files.append(copied)
    for name in ("items.json","reviewer_a_template.json","reviewer_b_template.json"):
        files.append(write_private_json(temporary/name,read_json(packet/"reviewer"/name)))
    note=instructions()
    old="This packet currently contains metadata and blank templates only. Report\ncopies have NOT been materialized: do not follow investigator source paths.\nCopying synthetic text requires a separately authorized protected step."
    new="This bundle contains byte-identical SYNTHETIC report copies under reports/.\nUse only these opaque item paths. Do not inspect sibling investigator files,\nmodel labels, source paths, images or EHR. No real patient input is included."
    if old not in note:
        raise ValueError("review instruction source changed")
    files.append(write_private_text(temporary/"INSTRUCTIONS.md",note.replace(old,new)))
    return {"schema_version":"tricompose-blinded-synthetic-report-copies-v1",
        "reports":len(items),"candidate_lineages":48,"independent_ehr_cases":2,
        "source_packet_manifest_sha256":sha256_file(packet/"manifest.json"),
        "copied_reports_byte_identical":True,"human_annotations_completed":0,
        "synthetic_only":True,"real_inputs_read":False,"model_calls":0,
        "selection_changed":False,"regeneration_authorized":False,
        "untouched_final_test":False},files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet-run","output-root","run-id"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--allow-synthetic-report-copy",action="store_true")
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not args.allow_synthetic_report_copy:
        print(json.dumps({"status":"refused","reason":"explicit_copy_authorization_and_slurm_required"}))
        return 2
    os.umask(0o007)
    temporary=None
    try:
        temporary,target=new_atomic_run(args.output_root,args.run_id)
        summary,files=run(args,temporary)
        files.append(write_private_json(temporary/"summary.json",summary))
        write_private_json(temporary/"manifest.json",{"schema_version":summary["schema_version"],
            "program_sha256":sha256_file(__file__),"run_id":args.run_id,
            "source_packet_manifest_sha256":summary["source_packet_manifest_sha256"],
            "artifacts":{str(path.relative_to(temporary)):{"sha256":sha256_file(path)} for path in files}})
        commit_atomic_run(temporary,target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}))
        return 1
    print(json.dumps({"status":"completed_protected_synthetic_blind_copies","model_calls":0,"reports":summary["reports"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

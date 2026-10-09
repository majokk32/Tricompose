#!/usr/bin/env python3
"""Recompute all headroom pairs; verify frozen choices, arithmetic and privacy."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import stat
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmarks"))
import diagnose_report_repair_headroom as cli
from tricompose_v12.report_repair_headroom import SCHEMA,prepare,attach,summarize,SET_FIELDS
from contracts import (WORKSPACE,PROTECTED_ROOT,require_inside,read_json,sha256_file,
    new_atomic_run,write_private_json,commit_atomic_run,discard_atomic_run)


def audit(run):
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("existing CPU Slurm allocation required")
    root=require_inside(run,PROTECTED_ROOT,must_exist=True);mp=root/"manifest.json";manifest=read_json(mp)
    if (manifest.get("schema_version")!=SCHEMA or manifest.get("new_policy_executed") is not False
            or manifest.get("original_selection_changed") is not False or manifest.get("new_model_calls")!=0):
        raise ValueError("frozen diagnostic-only manifest required")
    sources={k:require_inside(p,WORKSPACE,must_exist=True) for k,p in manifest["source_paths"].items()}
    if any(sha256_file(p)!=manifest["source_sha256"][k] for k,p in sources.items()):
        raise ValueError("immutable diagnostic source changed")
    for name,entry in manifest["artifacts"].items():
        p=require_inside(root/name,root,must_exist=True)
        if sha256_file(p)!=entry["sha256"]:raise ValueError("output bytes differ")
    for p in (root,*root.iterdir()):
        st=p.stat()
        if stat.S_IMODE(st.st_mode)!=(0o2770 if p.is_dir() else 0o660) or st.st_gid not in (96293,65534):
            raise ValueError("protected modes/project group differ")
    bank,policy,choices,_=cli.load_inputs(sources["comparison_manifest"].parent)
    plan=prepare(bank,policy,choices)
    if plan!=read_json(root/"label_only_plan.json") or sha256_file(root/"label_only_plan.json")!=manifest["label_plan_sha256"]:
        raise ValueError("source-only plan changed")
    measured=attach(plan,bank,policy,choices,read_json(sources["full_bank_scores"])["records"])
    result=summarize(measured);texts=cli.output_texts(measured,result)
    if any((root/name).read_text()!=text for name,text in texts.items()):
        raise ValueError("exact full table/report recomputation differs")
    summary=read_json(root/"summary.json")
    if any(summary[k]!=v for k,v in result.items()) or summary["label_plan_sha256"]!=manifest["label_plan_sha256"]:
        raise ValueError("complete diagnostic summary differs")
    # Independent set predicate arithmetic; no call to the producer compare().
    evidence={r["triple_candidate_id"]:r for r in plan["candidate_evidence"]}
    passing=0
    for pair in plan["directed_pairs"]:
        a,b=(evidence[pair[k]] for k in ("baseline_candidate_id","alternative_candidate_id"))
        preserve=all(set(a[k])<=set(b[k]) for k in SET_FIELDS[:4])
        no_new_opposition=all(set(b[k])<=set(a[k]) for k in SET_FIELDS[4:])
        strict=any(set(a[k])!=set(b[k]) for k in SET_FIELDS)
        q=a["report_structure_quality"];r=b["report_structure_quality"]
        expected=bool(preserve and no_new_opposition and strict and q is not None and r is not None and r>=q
            and a["source_gate_failure_count"]==b["source_gate_failure_count"]==0
            and a["source_image_validity"] and b["source_image_validity"])
        if expected!=pair["strict_label_preserving_headroom"]:raise ValueError("independent set predicate differs")
        passing+=expected
    if passing!=summary["counts"]["strict_label_preserving_directed_pairs"]:
        raise ValueError("strict opportunity total differs")
    serialized="\n".join(texts.values())+json.dumps(summary)+json.dumps(plan)
    for key in ('"subject_id"','"patient_id"','"report_text"','"image_path"','"source_statement"'):
        if key in serialized:raise ValueError("forbidden raw body/identifier field")
    if any(sha256_file(p)!=manifest["source_sha256"][k] for k,p in sources.items()):
        raise ValueError("immutable diagnostic source changed during audit")
    return {"schema_version":"tricompose-report-headroom-audit-v1","status":"passed_metadata_set_predicate_recompute_permissions_audit",
        "comparison_manifest_sha256":sha256_file(mp),"independent_set_predicate_checked":True,
        "source_choices_and_artifacts_unchanged":True,"full_tables_exactly_recomputed":True,
        "private_modes_and_group_checked":True,"new_model_calls":0,"clinical_repair_success":False,
        "counts":summary["counts"],"audit_program_sha256":sha256_file(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","output-root","run-id"):p.add_argument("--"+name,required=True)
    args=p.parse_args();os.umask(0o007)
    try:
        result=audit(args.run);temporary,target=new_atomic_run(args.output_root,args.run_id)
        try:
            write_private_json(temporary/"audit.json",result);commit_atomic_run(temporary,target)
        except Exception:
            discard_atomic_run(temporary);raise
    except Exception as exc:
        print(json.dumps({"status":"audit_failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":result["status"],"audit_sha256":sha256_file(target/"audit.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""Recompute cached report controls and verify source/choice/private boundaries."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import stat
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmarks"))
import compare_frozen_report_paths as cli
from tricompose_v12.report_path_comparison import SCHEMA,freeze_choices,attach_endpoint,aggregate,verify_previous_sana
from contracts import (WORKSPACE,PROTECTED_ROOT,require_inside,read_json,sha256_file,
    new_atomic_run,write_private_json,commit_atomic_run,discard_atomic_run)


def audit(run):
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("existing CPU Slurm allocation required")
    root=require_inside(run,PROTECTED_ROOT,must_exist=True)
    manifest=read_json(root/"manifest.json")
    if manifest["schema_version"]!=SCHEMA:raise ValueError("comparison manifest differs")
    sources={k:require_inside(p,WORKSPACE,must_exist=True) for k,p in manifest["source_paths"].items()}
    if any(sha256_file(p)!=manifest["source_sha256"][k] for k,p in sources.items()):
        raise ValueError("comparison source bytes changed")
    for name,entry in manifest["artifacts"].items():
        path=require_inside(root/name,root,must_exist=True)
        if sha256_file(path)!=entry["sha256"]:raise ValueError("comparison output bytes changed")
    for path in (root,*root.iterdir()):
        expected=0o2770 if path.is_dir() else 0o660
        st=path.stat()
        if stat.S_IMODE(st.st_mode)!=expected or st.st_gid not in (96293,65534):
            raise ValueError("project-private group/mode differs")
    args=argparse.Namespace(endpoint_run=sources["historical_endpoint_manifest"].parent)
    bank,policy,_=cli.load_bank(args)
    sealed=read_json(root/"choices.json")
    choices=freeze_choices(bank,policy)
    if sealed!={"choices":choices,"selection_used_biovil":False}:
        raise ValueError("frozen choices differ from original source key")
    endpoint=read_json(sources["full_bank_scores"])["records"]
    rows=attach_endpoint(choices,bank,policy,endpoint);result=aggregate(rows)
    previous=[json.loads(line) for line in sources["previous_control_outcomes"].read_text().splitlines() if line]
    checks=verify_previous_sana(choices,previous,max(policy["model_call_budgets"]))
    expected={"selected_path_rows.jsonl":"".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in rows),
        "selected_path_score_table.csv":cli.render_csv([cli.flatten(r) for r in rows]),
        "method_comparison.csv":cli.render_csv([cli.flatten(r) for r in result["method_comparison"]]),
        "paired_comparison.csv":cli.render_csv(result["paired_comparison"]),
        "paired_case_comparison.csv":cli.render_csv(result["paired_case_rows"]),
        "selection_frequencies.csv":cli.render_csv(result["selection_frequencies"]),
        "RESULTS_CN_EN.md":cli.markdown(result,checks)}
    if any((root/name).read_text()!=text for name,text in expected.items()):
        raise ValueError("full comparison recomputation differs")
    summary=read_json(root/"summary.json")
    if (summary["choices_sha256"]!=sha256_file(root/"choices.json")
            or summary["archived_sana_verification"]!=checks
            or any(summary[k]!=v for k,v in result.items() if k not in {"method_comparison","paired_comparison","paired_case_rows","selection_frequencies"})
            or summary["counts"]!={"ehr_cases":80,"fixed_cxr_slots":240,"report_candidates":960,"compared_path_slots":1200}
            or summary["full_bank_endpoint_available"]!=960):
        raise ValueError("summary population/choice/endpoint counts differ")
    # Separate direct arithmetic check, not the aggregation helper's implementation.
    for table in result["method_comparison"]:
        if table["cxr_group"]!="all_image_generators" or table["ehr_evidence_subgroup"]!="all":continue
        selected=[r for r in rows if r["method"]==table["method"]]
        case_means=[]
        for case in bank:
            vals=[r["biovil_raw_cosine"] for r in selected if r["case_id"]==case]
            if len(vals)!=3 or any(v is None for v in vals):raise ValueError("full current bank mean missing")
            case_means.append(statistics.mean(vals))
        if table["full_cohort_ehr_mean_biovil"]!=statistics.mean(case_means):
            raise ValueError("case-balanced endpoint mean differs")
    text="\n".join(expected.values())+json.dumps(summary)
    for forbidden in ('"subject_id"','"patient_id"','"report_text"','"image_path"','"source_statement"'):
        if forbidden in text:raise ValueError("private metadata contract contains forbidden body/identifier field")
    return {"schema_version":"tricompose-frozen-report-path-audit-v1","status":"passed_metadata_hash_recompute_permissions_audit",
        "comparison_manifest_sha256":sha256_file(root/"manifest.json"),"new_model_calls":0,
        "original_source_and_choices_unchanged":True,"full_tables_exactly_recomputed":True,
        "independent_case_mean_arithmetic_checked":True,"private_modes_and_group_checked":True,
        "archived_sana_verification":checks,"clinical_accuracy":None,"clinical_acceptance":False,
        "audit_program_sha256":sha256_file(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True);p.add_argument("--output-root",required=True);p.add_argument("--run-id",required=True)
    args=p.parse_args();os.umask(0o007)
    try:
        result=audit(args.run)
        temporary,target=new_atomic_run(args.output_root,args.run_id)
        try:
            write_private_json(temporary/"audit.json",result)
            commit_atomic_run(temporary,target)
        except Exception:
            discard_atomic_run(temporary);raise
    except Exception as exc:
        print(json.dumps({"status":"audit_failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":result["status"],"audit_sha256":sha256_file(target/"audit.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

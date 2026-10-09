#!/usr/bin/env python3
"""CPU cached same-image report paths; sealed source choices precede endpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT.parent/"TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.report_path_comparison import SCHEMA,STATIC,freeze_choices,attach_endpoint,aggregate,verify_previous_sana
from tricompose_v12.full_bank_secondary import SCHEMA as FULL_SCHEMA
from merge_automatic_secondary import SCHEMA as HISTORICAL_SCHEMA,cached
from run_legacy_automatic_replay import checked_source,load as load_legacy
from run_automatic_proxy_replay import render_csv
from contracts import (WORKSPACE,PROTECTED_ROOT,require_inside,read_json,sha256_file,new_atomic_run,
    write_private_json,write_private_text,commit_atomic_run,discard_atomic_run)


def load_bank(args):
    """Follow authenticated manifests, without reading old endpoint values."""
    em,emp,ep=cached(args.endpoint_run,HISTORICAL_SCHEMA,"endpoint_outcomes.jsonl",64*1024*1024)
    if em.get("original_selection_changed") is not False or em.get("routing_or_thresholds_updated") is not False:
        raise ValueError("unchanged historical source required")
    replay=require_inside(em["source_paths"]["replay_manifest"],PROTECTED_ROOT,must_exist=True)
    config=require_inside(em["source_paths"]["frozen_policy"],PROTECTED_ROOT,must_exist=True)
    if any(sha256_file(p)!=em["source_sha256"][k] for k,p in (("replay_manifest",replay),("frozen_policy",config))):
        raise ValueError("historical replay/policy hash changed")
    rm=read_json(replay);sp=rm["source_paths"]
    opts=argparse.Namespace(selection_run=Path(sp["selection_manifest"]).parent,
        edge_run=Path(sp["edge_manifest"]).parent,policy_config=config,
        development_selection_run=Path(sp["development_manifest"]).parent if "development_manifest" in sp else None)
    bank,policy,sources,_=load_legacy(opts)
    if any(sha256_file(sources[k])!=rm["source_sha256"][k] for k in ("source_scores","source_edges")):
        raise ValueError("frozen source clinical proxies changed")
    sources.update(historical_endpoint_manifest=emp,historical_endpoint_outcomes=ep,historical_replay_manifest=replay)
    return bank,policy,sources


def flatten(row):
    """Preserve original field values, expanding only nested edge totals."""
    result={k:v for k,v in row.items() if k not in {"raw_edge_totals","raw_edge_readouts"}}
    edges=row.get("raw_edge_totals",row.get("raw_edge_readouts",{}))
    for edge,values in edges.items():
        if values is None:
            result[edge+"_unavailable"]=True
        else:
            result.update({edge+"_"+field:value for field,value in values.items()})
    return result


def markdown(result,checks):
    def show(v):return "NA" if v is None else f"{v:.4f}"
    lines=["# Same-image report paths / 同图报告路径对照","",
        "80 fixed synthetic EHRs × three fixed image generators × four frozen report experts. Each EHR averages its three images before the cohort BioViL mean. No image/EHR/prompt changes or new model calls.",
        "比较四个固定模型与原 V1.1 source-key 在同一张 CXR 上的四选一。先封存选择，再读取完整图文 endpoint；BioViL 不参与选择，也不是临床准确率。", "",
        "## Whole-cohort comparison / 完整队列", "",
        "| Report path | EHRs | Images | Mean raw BioViL | Positive support / image-positive facts | Negative support / image-negative facts | Raw opposition / known image facts | Comparable / known image facts | Simulated total calls/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["method_comparison"]:
        if r["cxr_group"]=="all_image_generators" and r["ehr_evidence_subgroup"]=="all":
            lines.append(f"| {r['method']} | {r['fixed_ehr_cases']} | {r['fixed_image_slots']} | {show(r['full_cohort_ehr_mean_biovil'])} | {show(r['image_positive_support_over_known'])} | {show(r['image_negative_support_over_known'])} | {show(r['cxr_report_proxy_opposition_over_known'])} | {show(r['cxr_report_coverage_over_known'])} | {show(r['simulated_calls_per_fixed_image'])} |")
    lines += ["", "## Same-image paired contrasts / 同图配对差值", "",
        "| CXR generator | Fixed report baseline | EHRs | Available paired EHRs | BioViL delta: original-key choice − fixed | Positive / zero / negative EHR deltas |",
        "|---|---|---:|---:|---:|---|"]
    for r in result["paired_comparison"]:
        if r["ehr_evidence_subgroup"]=="all":
            lines.append(f"| {r['cxr_group']} | {r['baseline']} | {r['fixed_ehr_cases']} | {r['paired_available_ehr_cases']} | {show(r['full_cohort_mean_biovil_delta'])} | {r['positive_delta_cases']} / {r['zero_delta_cases']} / {r['negative_delta_cases']} |")
    lines += ["", "## Selection frequency, not clinical model rank / 选择频次，不是临床排名", "",
        "| CXR group | Selected report model | Image slots |", "|---|---|---:|"]
    for r in result["selection_frequencies"]:
        lines.append(f"| {r['cxr_group']} | {r['report_model']} | {r['selected_image_slots']} |")
    lines += ["", "## Integrity and interpretation / 完整性与解释", "",
        f"- Previous Sana-only static control: {checks['exact_same_choices']}/{checks['archived_sana_cases']} identical candidate choices, CXR and EHR hashes.",
        "- All 80 cases remain included. Only eight EHRs contain directly comparable cached EHR findings; 72 have none. EHR edge rates for the no-direct subgroup are NA, not perfect consistency. These counts are NOT the earlier prompt-conditioning 15/65 split.",
        "- 原始 V1.1 source-key 顺序、原始 global No-Finding proxy 调整和赢家保持不变；另列 raw 四状态读数，不把 unknown/uncertain 转成 negative。",
        "- Report experts: MAIRA-2, CXRMate-single, LLaVA-Rad, CheXagent-2. CXRMate-single does NOT consume structured EHR; this is not an EHR+CXR report-path experiment.",
        "- Historical 14-field uncalibrated XRV/CheXbert profile stays separate from the new eight-enabled-head smoke test. Label agreement is an optimization proxy, not a gold-standard clinical judgment.",
        "- Coverage and opposition divide by explicit image-reference facts; missing report facts do not count as agreement. Positive and negative support have separate denominators. CSVs retain all raw counts.",
        "- Costs are simulated generator/scorer invocations, not measured GPU savings: fixed path 4 = 1 CXR + 1 XRV + 1 report + 1 CheXbert; four-report static path 10 = 1 + 1 + 4 + 4. With images already fixed, incremental report/scorer calls are 2 versus 8. Shared EHR generation and secondary endpoint costs are not included/free savings.",
        "- 240 images are repeated observations of 80 EHRs, not 240 independent patients. BioViL first averages images within EHR; fact rates pool explicit reference comparisons and retain their denominators. Any missing required endpoint leaves the full-cohort mean/delta NA; available-case means are separately labeled.",
        "- This already inspected synthetic bank is a development control, not a held-out benchmark. No endpoint oracle, post-result threshold/model-priority fitting, significance claim, clinical fault localization or successful repair claim.",
        "- Generation edge is EHR-derived text → CXR, not proven direct structured-EHR-conditioned cold-start generation. Original joint image/report winners remain unchanged.",
        "- Only cached synthetic labels/scores, schemas, metadata and hashes were used. No real patient inputs/targets, report bodies or image pixels were opened.", ""]
    return "\n".join(lines)


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing CPU Slurm allocation required")
    started=time.monotonic()
    bank,policy,sources=load_bank(args)
    fm,mp,score_path=checked_source(args.full_bank_run,FULL_SCHEMA,"scores.json",4*1024*1024)
    if any(fm.get(k)!=v for k,v in (("new_model_calls",0),("original_selection_changed",False),("clinical_acceptance",False))):
        raise ValueError("evaluation-only full-bank cache required")
    frozen_sources={"full_bank_"+k:require_inside(p,WORKSPACE,must_exist=True) for k,p in fm["source_paths"].items()}
    if any(sha256_file(p)!=fm["source_sha256"][k.removeprefix("full_bank_")] for k,p in frozen_sources.items()):
        raise ValueError("full-bank source/checkpoint bytes changed")
    cm,cp,control_path=checked_source(args.previous_control_run,"tricompose-fixed-image-report-control-v1",
        "control_outcomes.jsonl",64*1024*1024)
    sources.update(frozen_sources)
    sources.update(full_bank_manifest=mp,full_bank_scores=score_path,previous_control_manifest=cp,
        previous_control_outcomes=control_path,program=Path(__file__),
        comparison_contract=ROOT/"src/tricompose_v12/report_path_comparison.py",
        source_key_contract=ROOT/"src/tricompose_v12/automatic_replay.py",
        raw_edge_contract=ROOT/"src/tricompose_v12/invariant_verification.py",
        historical_adapter=ROOT/"src/tricompose_v12/legacy_replay_adapter.py",
        source_loader=ROOT/"benchmarks/run_legacy_automatic_replay.py",
        protocol=WORKSPACE/"docs/frozen_report_path_comparison_protocol.md",
        private_writer=ROOT.parent/"TriCompose-v1.0/eval/report_v1_1/contracts.py")
    before={k:sha256_file(p) for k,p in sources.items()}
    choices=freeze_choices(bank,policy)
    if len(bank)!=80 or len(choices)!=1200:
        raise ValueError("complete eighty-case historical bank required")
    previous=[json.loads(line) for line in control_path.read_text().splitlines() if line]
    checks=verify_previous_sana(choices,previous,max(policy["model_call_budgets"]))
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        sealed=write_private_json(temporary/"choices.json",{"choices":choices,"selection_used_biovil":False})
        choice_hash=sha256_file(sealed)
        # Endpoint cosine values are not read until source-only choices are sealed.
        records=read_json(score_path)["records"]
        rows=attach_endpoint(choices,bank,policy,records)
        result=aggregate(rows)
        summary={**{k:v for k,v in result.items() if k not in {"method_comparison","paired_comparison","paired_case_rows","selection_frequencies"}},
            "status":"completed_cached_same_image_frozen_report_path_comparison",
            "counts":{"ehr_cases":len(bank),"fixed_cxr_slots":240,"report_candidates":len(records),"compared_path_slots":len(rows)},
            "archived_sana_verification":checks,"choices_sha256":choice_hash,
            "choices_sealed_before_full_endpoint_values":True,"source_artifacts_unchanged":True,
            "full_bank_endpoint_available":sum(r["biovil_raw_cosine"] is not None for r in records),
            "runtime_seconds_including_hash_audit":round(time.monotonic()-started,6)}
        files=[sealed,write_private_text(temporary/"selected_path_rows.jsonl","".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in rows)),
            write_private_text(temporary/"selected_path_score_table.csv",render_csv([flatten(r) for r in rows])),
            write_private_text(temporary/"method_comparison.csv",render_csv([flatten(r) for r in result["method_comparison"]])),
            write_private_text(temporary/"paired_comparison.csv",render_csv(result["paired_comparison"])),
            write_private_text(temporary/"paired_case_comparison.csv",render_csv(result["paired_case_rows"])),
            write_private_text(temporary/"selection_frequencies.csv",render_csv(result["selection_frequencies"])),
            write_private_text(temporary/"RESULTS_CN_EN.md",markdown(result,checks)),
            write_private_json(temporary/"summary.json",summary)]
        if (before!={k:sha256_file(p) for k,p in sources.items()} or sha256_file(sealed)!=choice_hash
                or choices!=freeze_choices(bank,policy)):
            raise ValueError("immutable source/choice changed during measurement")
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,"source_paths":{k:str(p) for k,p in sources.items()},
            "source_sha256":before,"artifacts":{p.name:{"sha256":sha256_file(p)} for p in files},
            "new_model_calls":0,"selection_used_biovil":False,"original_selection_changed":False,"clinical_acceptance":False})
        commit_atomic_run(temporary,target)
    except Exception:
        discard_atomic_run(temporary);raise
    return target,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-run","full-bank-run","previous-control-run","output-root","run-id"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args();os.umask(0o007)
    try:target,summary=run(args)
    except Exception as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":summary["status"],"runtime_seconds":summary["runtime_seconds_including_hash_audit"],
        "manifest_sha256":sha256_file(target/"manifest.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

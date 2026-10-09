#!/usr/bin/env python3
"""Cached label-preserving report opportunities; no winner or policy changes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT.parent/"TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.report_repair_headroom import SCHEMA,prepare,attach,summarize
from tricompose_v12.report_path_comparison import SCHEMA as COMPARISON_SCHEMA
import compare_frozen_report_paths as previous_cli
from run_legacy_automatic_replay import checked_source
from run_automatic_proxy_replay import render_csv
from contracts import (WORKSPACE,PROTECTED_ROOT,require_inside,read_json,sha256_file,
    new_atomic_run,write_private_json,write_private_text,commit_atomic_run,discard_atomic_run)


def load_inputs(comparison_run):
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("existing CPU Slurm allocation required")
    manifest,mp,cp=checked_source(comparison_run,COMPARISON_SCHEMA,"choices.json",3*1024*1024)
    if any(manifest.get(k)!=v for k,v in (("new_model_calls",0),("selection_used_biovil",False),
            ("original_selection_changed",False),("clinical_acceptance",False))):
        raise ValueError("unchanged evaluation-only comparison required")
    sources={k:require_inside(p,WORKSPACE,must_exist=True) for k,p in manifest["source_paths"].items()}
    if any(sha256_file(p)!=manifest["source_sha256"][k] for k,p in sources.items()):
        raise ValueError("frozen comparison source/weight bytes changed")
    bank,policy,loaded=previous_cli.load_bank(argparse.Namespace(endpoint_run=sources["historical_endpoint_manifest"].parent))
    sources.update({"reload_"+k:p for k,p in loaded.items()})
    sources.update(comparison_manifest=mp,comparison_choices=cp)
    choices=read_json(cp)["choices"]
    if read_json(cp).get("selection_used_biovil") is not False:raise ValueError("source-only choices required")
    return bank,policy,choices,sources


def markdown(result):
    def show(v):return "NA" if v is None else f"{v:.4f}"
    counts=result["counts"]
    lines=["# Report repair headroom / 报告可修复空间","",
        "Cached DEVELOPMENT analysis only: no new generator calls, new winners, policy execution or clinical repair verdict. All EHRs and images remain fixed.",
        "检查已有报告替代项是否能保住具体阳性事实、直接 EHR 支持和可比较信息，同时不新增矛盾、结构分不下降，并至少改善一个 finding 集合。不是只比较计数；不能把矛盾内容改成 unknown 算修复。", "",
        f"Inventory: {counts['fixed_ehr_cases']} EHRs, {counts['fixed_image_slots']} fixed CXRs, {counts['report_candidates']} reports; {counts['directed_report_pairs']} directed report pairs (4×3 per image). These are repeated observations, not independent patients.","",
        "## Opportunities from each unchanged baseline / 每个原始 baseline 的替代空间","",
        "| Baseline | Fixed images | Images with a strict label-preserving alternative | EHRs with any opportunity | Dominating directed pairs | BioViL higher / equal / lower pairs | Conditional EHR-balanced mean of ALL alternative gaps |",
        "|---|---:|---:|---:|---:|---|---:|"]
    for r in result["method_comparison"]:
        if r["cxr_group"]=="all_image_generators" and r["ehr_evidence_subgroup"]=="all":
            lines.append(f"| {r['method']} | {r['fixed_image_slots']} | {r['image_slots_with_label_preserving_alternative']} | {r['ehr_cases_with_any_label_preserving_alternative']} | {r['dominating_alternative_pair_count']} | {r['dominating_pairs_biovil_higher']} / {r['dominating_pairs_biovil_equal']} / {r['dominating_pairs_biovil_lower']} | {show(r['conditional_ehr_balanced_mean_all_alternative_delta_not_policy'])} |")
    lines += ["", "## Why fewer contradictions is not sufficient / 为什么不能只看低矛盾率","",
        f"Among {counts['pairs_with_raw_opposition_reduction']} directed pairs with fewer raw oppositions, {counts['opposition_reducing_pairs_rejected_by_preservation_or_quality']} do not pass the full preservation/quality predicate; {counts['opposition_reducing_pairs_silencing_existing_conflict']} silence at least one previously comparable conflicting fact. Categories overlap.",
        "在每个方向上保留 lost/gained finding IDs、新增/移除矛盾和通过缺失描述消掉矛盾的证据。较低数量不能抵消一个新矛盾，也不能用另一个疾病替换原来的阳性支持。", "",
        "## Scope and next gate / 边界与下一步","",
        "- No selected output was replaced. No model/seed/prompt was requested, and no new policy or clinical acceptance was enabled. Source-key choices and all original joint winners remain untouched.",
        "- BioViL endpoints are attached AFTER the label-only plan is sealed; all qualifying alternatives, not the highest cosine alternative, enter descriptive endpoint statistics. No endpoint-tuned predicate/priority/threshold.",
        "- The conditional mean first averages alternative gaps per opportunity-bearing image, then such images within EHR, then EHRs. Its denominator is opportunity-bearing EHRs, NOT all 80 cases. It is not a policy treatment effect. Missing required scores stay NA; zero opportunities means NA, not zero improvement. CSVs retain the available-case and full conditional denominators.",
        "- Source classification/report labels are uncalibrated automatic proxies, not clinical truth. The strict predicate does not guarantee image or report quality. Report-positive claims on unknown image findings are tracked, not automatically penalized or called true.",
        "- Historical fourteen-field states stay separate from the fresh eight-enabled-head control. Unknown/uncertain never negative. Raw-state diagnosis does not expand global No-Finding into per-finding negations; original source-key adjustments remain unchanged.",
        "- All 80 EHRs remain. Eight have explicit comparable cached EHR facts and 72 do not; a CXR–report opportunity in that subgroup is not proof of EHR fidelity. No weak diagnosis/medication prior is promoted into an image constraint.",
        "- Current report experts are image-only MAIRA-2, CXRMate-single, LLaVA-Rad and CheXagent-2. No independent EHR+CXR report path or majority-vote ground truth is invented.",
        "- Not-dominated sets refer ONLY to this strict frozen diagnostic and existing four reports. Absence of an alternative does not prove global optimality or that a new seed/prompt/model cannot improve.",
        "- Future real regeneration needs a separately frozen policy and independent confirmation, bounded actual call/failure/GPU-time accounting, and complete-script/resource review plus explicit approval before any sbatch. No savings/clinical error-localization/repair-success claim from this cache analysis.",
        "- Only cached synthetic labels, score metadata, schemas and hashes were read; no real patient inputs/targets, report bodies or image pixels were opened.",""]
    return "\n".join(lines)


def serialize_lines(rows):
    return "".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in rows)


def output_texts(measured,result):
    return {"directed_report_pairs.jsonl":serialize_lines(measured["directed_pairs"]),
        "candidate_evidence.jsonl":serialize_lines(measured["candidate_evidence"]),
        "fixed_image_graphs.jsonl":serialize_lines(measured["image_graphs"]),
        "baseline_path_opportunities.jsonl":serialize_lines(measured["path_opportunities"]),
        "method_opportunities.csv":render_csv(result["method_comparison"]),
        "model_pair_opportunities.csv":render_csv(result["model_pair_opportunities"]),
        "RESULTS_CN_EN.md":markdown(result)}


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("existing CPU Slurm allocation required")
    started=time.monotonic()
    bank,policy,choices,sources=load_inputs(args.comparison_run)
    sources.update(headroom_program=Path(__file__),headroom_contract=ROOT/"src/tricompose_v12/report_repair_headroom.py",
        headroom_protocol=WORKSPACE/"docs/report_repair_headroom_protocol.md")
    before={k:sha256_file(p) for k,p in sources.items()}
    plan=prepare(bank,policy,choices)
    if len(bank)!=80 or len(plan["directed_pairs"])!=2880 or len(plan["path_opportunities"])!=1200:
        raise ValueError("entire registered bank required")
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        frozen=write_private_json(temporary/"label_only_plan.json",plan);sealed_sha=sha256_file(frozen)
        endpoints=read_json(sources["full_bank_scores"])["records"]
        measured=attach(plan,bank,policy,choices,endpoints);result=summarize(measured)
        texts=output_texts(measured,result)
        files=[frozen,*[write_private_text(temporary/name,text) for name,text in texts.items()]]
        summary={**result,"status":"completed_cached_label_preserving_report_headroom",
            "label_plan_sha256":sealed_sha,"label_plan_sealed_before_endpoint_values":True,
            "source_artifacts_unchanged":True,"runtime_seconds_including_hash_audit":round(time.monotonic()-started,6)}
        files.append(write_private_json(temporary/"summary.json",summary))
        if before!={k:sha256_file(p) for k,p in sources.items()} or sha256_file(frozen)!=sealed_sha or plan!=prepare(bank,policy,choices):
            raise ValueError("frozen source or label-only diagnostic changed")
        if any((temporary/name).read_text()!=text for name,text in texts.items()):
            raise ValueError("atomic write verification differs")
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,"source_paths":{k:str(p) for k,p in sources.items()},
            "source_sha256":before,"artifacts":{p.name:{"sha256":sha256_file(p)} for p in files},
            "label_plan_sha256":sealed_sha,"new_model_calls":0,"original_selection_changed":False,
            "new_policy_executed":False,"clinical_acceptance":False})
        commit_atomic_run(temporary,target)
    except Exception:
        discard_atomic_run(temporary);raise
    return target,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("comparison-run","output-root","run-id"):p.add_argument("--"+name,required=True)
    args=p.parse_args();os.umask(0o007)
    try:target,result=run(args)
    except Exception as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":result["status"],"runtime_seconds":result["runtime_seconds_including_hash_audit"],
        "manifest_sha256":sha256_file(target/"manifest.json")}));return 0


if __name__=="__main__":raise SystemExit(main())

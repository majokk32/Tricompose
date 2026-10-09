#!/usr/bin/env python3
"""CPU Slurm: bilingual, aggregate-only handoff from an audited frozen run.

No report bodies, image pixels, raw EHR, generation, model factories or API.
Do not modify the completed run: publish a separate immutable private sidecar.
"""
import argparse
from collections import Counter
import csv
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, new_atomic_run,
    write_private_json, write_private_text, commit_atomic_run, discard_atomic_run)
from tricompose_v12.full_pool_report_control import SCHEMA, CXR_MODELS, MODELS, edge_totals
from tricompose_v12.runtime_dispatch import require_slurm


def aggregates(comparison):
    """Exactly three equally weighted fixed image paths; missing is not zero."""
    result = []
    for model in (*MODELS, "first_eligible_expert"):
        group = [r for r in comparison["model_comparison"] if r["report_policy"] == model]
        if (len(group) != 3 or {r["cxr_model_id"] for r in group} != set(CXR_MODELS)
                or any(r["fixed_ehr_cases"] != 80 or r["report_candidates"] != 80 for r in group)):
            raise ValueError("all three fixed eighty-case paths required")
        values = [r["mean_biovil_raw_cosine"] for r in group]
        for v in values:
            if v is not None and (type(v) not in (int, float) or not math.isfinite(v)):
                raise ValueError("finite secondary means or explicit missing required")
        edges = {}
        for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
            keys = ("known_reference_facts", "comparable_facts", "supported_facts", "supported_positive",
                    "supported_negative", "proxy_opposition_facts", "missing_comparisons")
            total = {k: sum(r["raw_edge_totals"][edge][k] for r in group) for k in keys}
            known = total["known_reference_facts"]
            total.update(coverage_over_known=total["comparable_facts"] / known if known else None,
                support_over_known=total["supported_facts"] / known if known else None,
                opposition_over_known=total["proxy_opposition_facts"] / known if known else None)
            edges[edge] = total
        result.append({"report_policy": model, "fixed_ehr_cases": 80, "fixed_images": 240,
            "mean_biovil_raw_cosine": math.fsum(values) / 3 if all(v is not None for v in values) else None,
            "biovil_available_pairs": sum(r["biovil_available_cases"] for r in group),
            "temporal_flag_count": sum(r["unsupported_temporal_language_count"] for r in group),
            "raw_edge_totals": edges, "clinical_accuracy": None, "clinical_acceptance": False})
    return result


def diagnostic(rows, comparison):
    index = {r["triple_candidate_id"]: r for r in rows}
    pairs = comparison["image_pairs"]
    changed = [p for p in pairs if p["baseline_triple_id"] != p["selected_triple_id"]]
    strata = []
    for direct in (False, True):
        group = [p for p in pairs if bool(index[p["baseline_triple_id"]]["receipt"]["known_ehr_facts"]) == direct]
        before = [index[p["baseline_triple_id"]] for p in group]
        after = [index[p["selected_triple_id"]] for p in group]
        values = [p["delta"] for p in group]
        strata.append({"stratum": "direct_ehr" if direct else "no_direct_ehr", "fixed_images": len(group),
            "fixed_ehr_cases": len({p["case_id"] for p in group}),
            "changed_images": sum(p["baseline_triple_id"] != p["selected_triple_id"] for p in group),
            "mean_selected_minus_baseline_cosine": math.fsum(values) / len(values) if values and all(v is not None for v in values) else None,
            "baseline_raw_edge_totals": {e: edge_totals(before, e) for e in ("ehr_cxr", "ehr_report", "cxr_report")},
            "selected_raw_edge_totals": {e: edge_totals(after, e) for e in ("ehr_cxr", "ehr_report", "cxr_report")}})
    return {"selected_model_counts": dict(Counter(p["selected_model"] for p in pairs)),
        "changed_images": len(changed), "ehr_cases_with_at_least_one_report_switch": len({p["case_id"] for p in changed}),
        "secondary_increased_images": sum(p["delta"] is not None and p["delta"] > 0 for p in pairs),
        "secondary_decreased_images": sum(p["delta"] is not None and p["delta"] < 0 for p in pairs),
        "secondary_unchanged_images": sum(p["delta"] == 0 for p in pairs), "direct_ehr_strata": strata}


def markdown(result, source, audit_root):
    show = lambda v: "NA" if v is None else f"{v:.4f}"
    aggregate = {r["report_policy"]: r for r in result["aggregate_comparison"]}
    base, chosen = aggregate["cxrmate_single"], aggregate["first_eligible_expert"]
    lines = ["# Full-pool report selection / 全池报告择优汇报", "",
        "## Scope / 实验范围", "",
        "80 fixed synthetic EHRs × 3 original CXR generators × 4 frozen CXR-only report experts = 960 candidates. All EHRs, prompts, images and original reports/winners remain unchanged. This run performs verification and static report selection, not new generation or online repair.",
        "80 份固定 synthetic EHR、240 张既有 synthetic CXR、960 份既有报告；此次没有新生成、训练或外部 API。CXRMate-single 是 CXR-only，不是 CXRMate-ED。", "",
        f"Slurm job {result['job_id']}: {result['slurm_elapsed_seconds']} seconds including automatic audit; controller {result['controller_wall_seconds']:.3f} seconds, audit {result['audit_wall_seconds']:.3f} seconds.",
        "Fresh XRV: 240 images; fresh CheXbert: 960 report samples (requested batch size 16). Reuse 960 authenticated full-report BioViL scores only after sealing choice. Historical generation/scoring costs are not zero or claimed savings.", "",
        "## All fixed report experts / 不省略固定模型对照", "",
        "Each row includes all 80 EHRs and three fixed images per EHR. The image-reference denominator is 1,920 = 80 × 3 × 8 enabled classifier heads; it is NOT clinical ground truth. Disabled heads remain unknown.", "",
        "| Report policy | Comparable / reference | Positive supports | Negative supports | Proxy opposition | Mean raw BioViL | Temporal flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["aggregate_comparison"]:
        edge = row["raw_edge_totals"]["cxr_report"]
        lines.append(f"| {row['report_policy']} | {edge['comparable_facts']} / {edge['known_reference_facts']} | {edge['supported_positive']} | {edge['supported_negative']} | {edge['proxy_opposition_facts']} | {show(row['mean_biovil_raw_cosine'])} | {row['temporal_flag_count']} |")
    lines += ["", "## Same-image selector vs fixed CXRMate / 同图择优对照", "",
        "| CXR generator | Baseline cosine | Selected cosine | Paired change | 95% exploratory case-bootstrap interval |",
        "| --- | ---: | ---: | ---: | --- |"]
    for item in result["paired_case_bootstraps"]:
        model = item["cxr_model_id"]
        boot = item["paired_case_bootstrap"]
        if model == "equal_weight_three_fixed_images":
            a, b = base["mean_biovil_raw_cosine"], chosen["mean_biovil_raw_cosine"]
        else:
            a = next(r["mean_biovil_raw_cosine"] for r in result["model_comparison"] if r["cxr_model_id"] == model and r["report_policy"] == "cxrmate_single")
            b = next(r["mean_biovil_raw_cosine"] for r in result["model_comparison"] if r["cxr_model_id"] == model and r["report_policy"] == "first_eligible_expert")
        interval = boot["confidence_interval"]
        ci = "NA" if interval is None else f"[{show(interval[0])}, {show(interval[1])}]"
        lines.append(f"| {model} | {show(a)} | {show(b)} | {show(boot['estimate_method_minus_baseline'])} | {ci} |")
    summary = result["summary"]
    d = result["diagnostic"]
    lines += ["",
        f"{summary['gate_passing_alternatives']} of {summary['alternative_comparisons']} alternatives pass the unchanged gate; {summary['changed_images']} of 240 images switch reports, spanning {d['ehr_cases_with_at_least_one_report_switch']} EHRs. {summary['unresolved_baseline_images']} image baselines remain unresolved, not clinically accepted.",
        "选择顺序事先固定，不看 BioViL。保留原有支持及可比 finding IDs、不新增评分器冲突、不把冲突藏进 unknown；还要求文本非重复、模型对应结构合格且时间/泛化/重复风险不增加。", "",
        "## EHR evidence and limits / EHR 证据与限制", "",
        f"Only {summary['direct_ehr_cases']} of 80 EHRs have direct cached radiographic facts; {summary['no_direct_ehr_cases']} do not. Unknown/uncertain are not negatives. All remain in the denominator; EHR-edge rates for no-direct-evidence cases are NA.",
        "The pooled EHR reference count 24 represents eight direct facts evaluated on three fixed image paths, NOT 24 independent EHR facts/patients. EHR-CXR cannot improve in this experiment because images are unchanged.", "",
        "| Edge | Baseline supports / reference | Selected supports / reference | Baseline proxy opposition | Selected proxy opposition |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
        a, b = base["raw_edge_totals"][edge], chosen["raw_edge_totals"][edge]
        lines.append(f"| {edge} | {a['supported_facts']} / {a['known_reference_facts']} | {b['supported_facts']} / {b['known_reference_facts']} | {a['proxy_opposition_facts']} | {b['proxy_opposition_facts']} |")
    lines += ["",
        "不能声称择优优于全部固定模型：固定 LLaVA-Rad 的整体 BioViL 高于当前 selector；MAIRA-2 和 CheXagent-2 的直接 EHR-report 标签支持也更高。这说明指标之间有取舍，并不是已经得到可靠的单一最优临床分数。",
        "The large secondary gain is concentrated on the RoentGen path with a particularly low historical CXRMate baseline. Sana's selected cosine decreases. Report both effects; do not tune thresholds/priorities after seeing them.",
        "Label-support/opposition improvements use the gate's own scorers and are partly guaranteed by its design, NOT independent proof of clinical correction. BioViL was not used for this selection, but the historical pool/scores have been inspected before; it is not an untouched final test.",
        "Bootstrap: 2,000 paired draws, seed 0, EHR case as sampling unit; average all three fixed images per EHR first. Intervals are exploratory proxy estimates, not clinical efficacy. No candidate-level pseudo-replication.",
        "No faulty modality has been clinically localized, no report/CXR has been regenerated, and no online-agent cost saving is established. Next compare a separately preregistered bounded real regeneration protocol against this static reference, with unchanged EHRs and independent outcomes.", "",
        "## Files / 下载时查看", "",
        f"Source run: `{source}`", f"CPU audit: `{audit_root}`",
        "`score_table.csv`: 960 candidate scores with raw counts/NA; `model_comparison.csv`: 15 fixed/selector paths; `selection.json`: all 240 same-image choices; `comparison.json`: paired changes and case bootstraps. No original bank winner was overwritten.",
        "This sidecar adds pooled comparisons and an aggregate bilingual explanation without modifying the sealed run. All contents remain project-group private under artifacts/protected; no raw patient input/target or generated report/image is embedded.", ""]
    return "\n".join(lines)


def run(args):
    require_slurm()
    source = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    audit_root = require_inside(args.source_audit, PROTECTED_ROOT, must_exist=True)
    mp, am = source / "manifest.json", audit_root / "manifest.json"
    m, proof_manifest = read_json(mp), read_json(am)
    proof = read_json(audit_root / "audit.json")
    if (m["schema_version"] != SCHEMA or m["status"] != "completed_full_pool_fresh_report_control_unvalidated"
            or sha256_file(audit_root / "audit.json") != proof_manifest["audit_sha256"]
            or proof["status"] != "metadata_hash_receipt_audit_passed" or proof["source_manifest_sha256"] != sha256_file(mp)
            or proof["clinical_acceptance"] is not False):
        raise ValueError("completed source and bound successful metadata audit required")
    for name, entry in m["artifacts"].items():
        if sha256_file(require_inside(source / name, source, must_exist=True)) != entry["sha256"]:
            raise ValueError("sealed output changed")
    comparison = read_json(source / "comparison.json")
    rows = read_json(source / "score_rows.json")["records"]
    result = {"schema_version": "tricompose-full-pool-report-summary-v1", "job_id": args.job_id,
        "slurm_elapsed_seconds": args.slurm_elapsed_seconds,
        "controller_wall_seconds": m["wall_seconds_including_startup_io"], "audit_wall_seconds": proof["runtime_seconds"],
        "summary": comparison["summary"], "aggregate_comparison": aggregates(comparison),
        "model_comparison": comparison["model_comparison"], "paired_case_bootstraps": comparison["paired_case_bootstraps"],
        "diagnostic": diagnostic(rows, comparison), "new_model_calls": 0,
        "clinical_acceptance": False, "historical_pool_is_untouched_test": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        jp = write_private_json(temporary / "summary.json", result)
        md = write_private_text(temporary / "RESULTS_CN_EN.md", markdown(result, source, audit_root))
        table = []
        for row in result["aggregate_comparison"]:
            table.append({k: v for k, v in row.items() if k != "raw_edge_totals"} | {
                edge + "_" + k: v for edge, values in row["raw_edge_totals"].items() for k, v in values.items()})
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=sorted(table[0]))
        writer.writeheader()
        writer.writerows({k: "NA" if v is None else v for k, v in row.items()} for row in table)
        cp = write_private_text(temporary / "aggregate_comparison.csv", handle.getvalue())
        write_private_json(temporary / "manifest.json", {"schema_version": result["schema_version"],
            "source_manifest_sha256": sha256_file(mp), "source_audit_manifest_sha256": sha256_file(am),
            "summary_program_sha256": sha256_file(Path(__file__)),
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in (jp, md, cp)},
            "new_model_calls": 0, "original_output_changed": False, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "source-audit", "output-root", "run-id", "job-id"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--slurm-elapsed-seconds", type=int, required=True)
    args = p.parse_args()
    os.umask(0o007)
    try:
        target = run(args)
    except Exception as exc:
        print(json.dumps({"status": "summary_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "aggregate_bilingual_summary_saved", "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Immutable bilingual handoff from audited metadata, never clinical bodies."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))
from complete_bounded_regeneration_endpoint import SOURCE, SOURCE_SHA, AUDIT, AUDIT_SHA, SCHEMA as RECOVERY_SCHEMA
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, new_atomic_run,
    write_private_json, write_private_text, commit_atomic_run, discard_atomic_run)
from tricompose_v12.full_pool_report_control import validate_endpoint, edge_totals
from tricompose_v12.runtime_dispatch import require_slurm
from run_automatic_proxy_replay import render_csv

SCHEMA = "tricompose-bounded-regeneration-summary-v1"


def aggregate(rows, selection, endpoint):
    scores = validate_endpoint(rows, endpoint)
    indexed = {r["triple_candidate_id"]: r for r in rows}
    if len(selection["choices"]) != 2 or len({c["case_id"] for c in selection["choices"]}) != 2:
        raise ValueError("exact two-case summary required")
    result = []
    for method, key in (("fixed", "baseline_triple_id"), ("static", "static_triple_id"),
                        ("new_retry_candidate", "alternative_triple_id"), ("bounded_selection", "selected_triple_id")):
        group = [indexed[c[key]] for c in selection["choices"]]
        values = [scores[r["triple_candidate_id"]]["biovil_raw_cosine"] for r in group]
        result.append({"method": method, "cases": 2,
            "mean_biovil_raw_cosine": sum(values) / 2 if all(v is not None for v in values) else None,
            "biovil_available_cases": sum(v is not None for v in values),
            "official_structure_pass_cases": sum(r["structure"]["section_contract_pass"] for r in group),
            "unsupported_temporal_language_cases": sum(r["structure"]["unsupported_temporal_comparison_language"] for r in group),
            "raw_edges": {edge: edge_totals(group, edge) for edge in ("ehr_cxr", "ehr_report", "cxr_report")}})
    return result


def publish(args):
    require_slurm()
    if sha256_file(SOURCE / "manifest.json") != SOURCE_SHA or sha256_file(AUDIT / "manifest.json") != AUDIT_SHA:
        raise ValueError("audited fixed source required")
    source, am = read_json(SOURCE / "manifest.json"), read_json(AUDIT / "manifest.json")
    if sha256_file(AUDIT / "audit.json") != am["audit_sha256"] or read_json(AUDIT / "audit.json")["source_manifest_sha256"] != SOURCE_SHA:
        raise ValueError("matching audit receipt required")
    for name, entry in source["artifacts"].items():
        if sha256_file(require_inside(SOURCE / name, SOURCE, must_exist=True)) != entry["sha256"]:
            raise ValueError("completed source artifact changed")
    rows = read_json(SOURCE / "score_rows.json")["records"]
    selection = read_json(SOURCE / "selection.json")
    endpoint = read_json(SOURCE / "endpoint.json")
    recovery_manifest_sha = None
    if args.endpoint_run:
        recovery = require_inside(args.endpoint_run, PROTECTED_ROOT, must_exist=True)
        rm = read_json(recovery / "manifest.json")
        if (rm["schema_version"] != RECOVERY_SCHEMA or rm["status"] != "completed_secondary_recovery_not_clinical"
                or rm["source_manifest_sha256"] != SOURCE_SHA or rm["source_selection_sha256"] != source["selection_sha256_before_endpoint"]
                or rm["endpoint_used_for_selection"] is not False or rm["new_generation_calls"] != 0):
            raise ValueError("completed secondary-only recovery required")
        for name, entry in rm["artifacts"].items():
            if sha256_file(require_inside(recovery / name, recovery, must_exist=True)) != entry["sha256"]:
                raise ValueError("secondary recovery artifact changed")
        endpoint = read_json(recovery / "scores.json")
        recovery_manifest_sha = sha256_file(recovery / "manifest.json")
    comparisons = aggregate(rows, selection, endpoint)
    summary = {"schema_version": SCHEMA, "source_manifest_sha256": SOURCE_SHA, "audit_manifest_sha256": AUDIT_SHA,
        "secondary_recovery_manifest_sha256": recovery_manifest_sha, "cohort": source["cohort"],
        "new_images": 2, "new_reports": 2, "exploratory_gate_pass_cases": source["exploratory_gate_pass_cases"],
        "generation_verification_attempts_including_failures": source["generation_verification_attempts_including_failures"],
        "failed_primary_attempts": source["failed_generation_verification_attempts"],
        "initial_secondary_status": "failed_local_runtime_import_missing_vendor_bootstrap",
        "initial_failed_secondary_worker_seconds": 7.651484481059015,
        "comparison": comparisons, "clinical_repair_success": False,
        "original_winners_changed": False, "new_model_calls_for_summary": 0}
    lines = ["# Bounded regeneration results / 限预算重生成结果", "",
        "Approved job 12632531 completed on debug/P100 in 4m48s, exit 0:0; metadata audit passed.",
        "任务生成了两张新 CXR 和两份新报告，主生成/验证共 8 次调用，无失败；不训练，不修改 EHR、prompt 或旧赢家。", "",
        "This is a two-case engineering control from EHR evidence stratification, not an 80-case efficacy estimate.",
        "原 80 个 EHR 中 8 个有直接事实，5 个落入启用的评分器标签；固定选择第 5、11 个 opaque index。其余 78 个未返工，不删除、不替换。", "",
        "## Raw diagnostic scores / 原始诊断性评分", "",
        "Support means frozen-label proxy agreement, not clinical truth; EHR/report noncomparability remains NA.",
        "支持/冲突来自冻结评分器代理标签，不等于医生判断。缺失信息不是阴性，EHR–Report 没有可比较项时不解释为一致。", "",
        "| Method / 方法 | EHR–CXR support/known | EHR–CXR opposition | EHR–Report comparable/known | CXR–Report positive support | Negative support | Comparable/known | Temporal flags | BioViL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in comparisons:
        a,b,c = (result["raw_edges"][edge] for edge in ("ehr_cxr", "ehr_report", "cxr_report"))
        value = result["mean_biovil_raw_cosine"]
        biovil = "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {result['method']} | {a['supported_facts']}/{a['known_reference_facts']} | {a['proxy_opposition_facts']} | {b['comparable_facts']}/{b['known_reference_facts']} | {c['supported_positive']} | {c['supported_negative']} | {c['comparable_facts']}/{c['known_reference_facts']} | {result['unsupported_temporal_language_cases']} | {biovil} |")
    lines += ["", "## Decision / 决策", "",
        "Neither new candidate passes the preregistered gate: fixed EHR/image support does not improve. Keep both historical static choices; 0/2 replacements.",
        "两例的 EHR–CXR 冲突均未消除。新报告增加的是阴性标签一致性，不能替代 EHR 事实改善；所以保留旧静态结果，替换成功 0/2。", "",
        "The first retry also has slightly higher repeated-4gram ratio; the second loses prior positive image-report support and adds temporal-risk language relative to its static report. No gate/threshold adjustment after seeing results.",
        "第一例还增加了重复 n-gram；第二例相对静态候选丢失阳性支持并增加时间比较风险。没有事后降低门槛。", "",
        "## Endpoint and costs / 旁路评分与成本", "",
        "The original secondary worker failed with ModuleNotFoundError: this new invocation omitted the existing local BioViL vendor bootstrap. It consumed 7.651s and is not free. The original run/NA values remain immutable.",
        "BioViL 初次失败属于运行入口问题，不是生成效果的证据。补评只能使用已保存的图与报告、单独输出，不重生成或改变选择。", "",
        "Secondary recovery: " + ("completed as a separate authenticated sidecar; its scores appear above." if recovery_manifest_sha else "not run yet; BioViL remains NA."), "",
        "Budgets: four generator/verifier attempts per triggered EHR; evaluation encodings/loading/failures are separately charged. Static historical compute is a shared sunk cost, not zero. This is NOT a same-budget superiority or end-to-end compute-saving experiment.", "",
        "No clinical fault-localization accuracy, image-anatomy quality, or repair efficacy is established by two cases. Future equal-budget comparisons and independent untouched-cohort evidence remain necessary."]
    flat = []
    for result in comparisons:
        record = {k:v for k,v in result.items() if k != "raw_edges"}
        for edge, values in result["raw_edges"].items():
            record.update({edge + "_" + k: "NA" if v is None else v for k,v in values.items()})
        flat.append({k:"NA" if v is None else v for k,v in record.items()})
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        artifacts = [write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "RESULTS_CN_EN.md", "\n".join(lines) + "\n"),
            write_private_text(temporary / "aggregate_comparison.csv", render_csv(flat))]
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "source_manifest_sha256": SOURCE_SHA,
            "artifacts": {p.name:{"sha256":sha256_file(p)} for p in artifacts},
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-run")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        root = publish(args)
    except Exception as exc:
        print(json.dumps({"status":"summary_failed", "error_type":type(exc).__name__}))
        return 1
    print(json.dumps({"status":"immutable_metadata_summary_complete", "new_model_calls":0,
        "manifest_sha256":sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

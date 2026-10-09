#!/usr/bin/env python3
"""CPU-only cached score coverage diagnostic; never open report/image bodies."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.score_coverage import SCHEMA, FRESH_PROFILE, LEGACY_PROFILE, HASH_FIELDS, ID_FIELDS, analyze
from tricompose_v12.fixed_image_reports import freeze_selection, secondary_comparison
from tricompose_v12.automatic_replay import candidate_key
from tricompose_v12.automatic_discrepancy import point
from diagnose_automatic_discrepancy import load as load_historical
from run_automatic_proxy_replay import render_csv
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def bounded(path, limit=8*1024*1024):
    path = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("bounded protected metadata file required")
    return path, read_json(path)


def fresh(args):
    root = require_inside(args.fresh_run, PROTECTED_ROOT, must_exist=True)
    mp, manifest = bounded(root/"manifest.json", 64*1024)
    ap, audit_manifest = bounded(Path(args.fresh_audit)/"manifest.json", 64*1024)
    sp, audit = bounded(ap.parent/"audit_summary.json", 64*1024)
    if (manifest.get("schema_version") != "tricompose-fixed-image-report-control-v1"
            or manifest.get("status") != "completed_same_image_report_control_unvalidated"
            or audit_manifest.get("schema_version") != "tricompose-fixed-image-report-postrun-audit-v1"
            or audit_manifest.get("source_run_manifest_sha256") != sha256_file(mp)
            or audit_manifest.get("audit_summary_sha256") != sha256_file(sp)
            or audit.get("source_run_manifest_sha256") != sha256_file(mp)
            or audit.get("status") != "metadata_hash_selection_audit_passed"
            or audit.get("clinical_acceptance") is not False
            or audit.get("pre_endpoint_selection_recomputed") is not True):
        raise ValueError("completed hash-bound fresh metadata audit required")
    sources = {"fresh_manifest": mp, "fresh_audit_manifest": ap, "fresh_audit_summary": sp}
    cached = {}
    for name in ("score_rows.json", "selection.json", "secondary.json", "comparison.json"):
        path, value = bounded(root/name)
        if sha256_file(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError("fresh cached result hash differs")
        cached[name] = value; sources["fresh_"+name.replace(".", "_")] = path
    rows, selection, endpoint = cached["score_rows.json"]["records"], cached["selection.json"], cached["secondary.json"]
    if (freeze_selection(rows) != selection
            or sha256_file(sources["fresh_selection_json"]) != manifest["selection_sha256_before_endpoint"]
            or secondary_comparison(selection, rows, endpoint) != cached["comparison.json"]):
        raise ValueError("fresh frozen choices or secondary pairing differ")
    index = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    records = []
    for row in rows:
        r = index[row["triple_candidate_id"]]
        records.append({**{k: row[k] for k in (*ID_FIELDS, *HASH_FIELDS)}, "profile": FRESH_PROFILE,
            "fact_states": row["fact_states"],
            "source_primary_prefix": [row["proxy_penalty"], -row["raw_edge_readouts"]["cxr_report"]["supported_positive"]],
            "biovil_raw_cosine": r["biovil_raw_cosine"], "endpoint_unavailable_reason": r.get("reason")})
    result = analyze(records, FRESH_PROFILE)
    choices = selection["choices"]
    result["summary"].update({"cohort_role": "two_fixed_synthetic_ehr_engineering_control",
        "source_selected_baseline_images": sum(c["baseline_triple_id"] == c["selected_triple_id"] for c in choices),
        "source_mean_case_biovil_delta": cached["comparison.json"]["mean_case_delta"],
        "selection_recomputed_not_changed": True})
    return result, sources


def historical(args):
    outcomes, bank, policy, sources = load_historical(args)
    candidates = {(case, c["score_record"]["triple_candidate_id"]): c
                  for case, grid in bank.items() for c in grid.values()}
    endpoints = {}
    for outcome in outcomes:
        if outcome["selected_candidate_id"] is None:
            continue
        key = (outcome["case_id"], outcome["selected_candidate_id"])
        # Validate frozen snapshot lineage AND raw-label/source count arithmetic.
        point(outcome, candidates[key])
        snapshot = outcome["selected_snapshot"]
        value = (snapshot["biovil_cosine_secondary_not_routing"], snapshot["secondary_endpoint_unavailable_reason"])
        if endpoints.setdefault(key, value) != value:
            raise ValueError("shared historical endpoint changed across trials")
    records = []
    for (case, cid), candidate in sorted(candidates.items()):
        row = candidate["score_record"]; lineage = row["lineage"]
        value, reason = endpoints.get((case, cid), (None, "outside_scored_selected_union"))
        prefix = [x if math.isfinite(x) else None for x in candidate_key(candidate)[:4]]
        records.append({"profile": LEGACY_PROFILE, "case_id": case, "triple_candidate_id": cid,
            **{k: lineage[k] for k in (*HASH_FIELDS, "cxr_candidate_id", "report_candidate_id", "report_model_id")},
            "fact_states": [{"finding": f["finding"], **f["states"]} for f in candidate["facts"]],
            "source_primary_prefix": prefix, "biovil_raw_cosine": value, "endpoint_unavailable_reason": reason})
    result = analyze(records, LEGACY_PROFILE)
    result["summary"].update(cohort_role="previously_inspected_eighty_ehr_development_bank",
        endpoint_sampling_scope="scored_selected_union_not_exhaustive_bank",
        source_report_order=policy["report_order"], original_selection_changed=False,
        fresh_eight_head_profile_applied_to_legacy_cache=False)
    return result, {"historical_"+k: p for k,p in sources.items()}


def markdown(results):
    lines = ["# Score coverage diagnostic / 评分覆盖度诊断", "",
        "CPU cache analysis only; no report body, image pixels, raw patient input or new model calls.",
        "仅分析已有标签、分数及哈希；没有新生成、改阈值、换赢家或判定临床正确。", "",
        "## Separate profiles / 两套评分分别看", "",
        "| Profile | EHR | Images | Reports | No direct EHR cases | Images: explicit labels all negative | Reports: zero comparable CXR facts |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        s = result["summary"]
        lines.append(f"| {name} | {s['fixed_ehr_cases']} | {s['image_groups']} | {s['report_candidates']} | {s['no_direct_ehr_cases']} | {s['images_with_explicit_labels_all_negative']} | {s['no_comparable_image_report_candidates']} |")
    lines += ["", "The fresh profile has eight enabled heads and six disabled/unknown fields in its fourteen-field schema. Historical fourteen-field states are not rescored or pooled with it.",
        "新结果是 14 字段中启用 8 项、其余 unknown；旧池沿用历史标签，绝不混合归一化或声称重新校准。", "",
        "## Same-image controls / 同图报告平局", "",
        "| Profile | All report pairs | Source primary-prefix ties | Both endpoints available | Available tie pairs | Cases with available ties | Case-mean absolute BioViL gap on available ties |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        s = result["summary"]; gap = s["case_mean_absolute_biovil_gap_among_available_ties"]
        shown = "NA" if gap is None else f"{gap:.6f}"
        lines.append(f"| {name} | {s['same_image_report_pairs']} | {s['primary_prefix_tied_pairs']} | {s['both_endpoint_available_pairs']} | {s['endpoint_available_tied_pairs']} | {s['endpoint_available_tied_cases']} | {shown} |")
    lines += ["", "Fresh prefix: opposition + missing positive, then positive support. Historical prefix: original artifact gate + original clinical-count terms; structure/runtime/ID are excluded ONLY for this tie diagnostic. No alternative selector is executed.",
        "两套平局定义不同，具体字段见 summary.json；只诊断原评分的区分能力，不制定新赢家。", "",
        "BioViL absolute gaps on already scored pairs are descriptive, not accuracy, direction of improvement, or significance. Reports sharing an image and repeated pairs are correlated; available pair gaps are averaged within an EHR before the cohort mean.",
        "平局中的图文分数差异不是临床差异；同病例先取均值，不能把报告对当作独立患者。", "",
        "## Interpretation and next gate / 解读与下一步", "",
        "- Zero opposition with zero comparable findings means unavailable evidence, not a perfect report. Unknown/uncertain are never negative.",
        "- No direct EHR facts means unavailable EHR-edge evidence, not permission to enrich/drop that EHR or claim three-modal consistency.",
        "- Historical endpoint scores cover a selected union, not a random/exhaustive sample. All unscored candidates stay NA with `outside_scored_selected_union`; do not rank only scored candidates.",
        "- Investigate scorer preprocessing, raw score/threshold provenance and extraction coverage before altering generation. This diagnostic does not identify an image or report as clinically wrong.",
        "- Any new cohort/control must freeze inputs, policies, endpoint coverage and cost accounting before execution. Do not tune priorities/thresholds on these inspected outputs.",
        "- No new model, training, API, GPU execution, submission, original winner change, clinical acceptance or targeted regeneration was performed.", ""]
    return "\n".join(lines)


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing CPU Slurm allocation required before reading protected caches")
    started = time.monotonic()
    fresh_result, a = fresh(args); historical_result, b = historical(args)
    sources = {**a, **b, "program": Path(__file__),
        "coverage_contract": ROOT/"src/tricompose_v12/score_coverage.py",
        "historical_loader": ROOT/"benchmarks/diagnose_automatic_discrepancy.py",
        "historical_snapshot_contract": ROOT/"src/tricompose_v12/automatic_discrepancy.py",
        "historical_rank_contract": ROOT/"src/tricompose_v12/automatic_replay.py",
        "fresh_selection_contract": ROOT/"src/tricompose_v12/fixed_image_reports.py",
        "protocol": ROOT.parent/"docs/score_coverage_diagnostic_protocol.md"}
    before = {k: sha256_file(p) for k,p in sources.items()}
    results = {"fresh_eight_enabled_heads": fresh_result, "historical_fourteen_field_cache": historical_result}
    summary = {"schema_version": SCHEMA, "status": "completed_metadata_score_coverage_diagnostic",
        "profiles": {k: v["summary"] for k,v in results.items()}, "profiles_pooled": False,
        "new_model_calls": 0, "new_slurm_submissions": 0, "original_selection_changed": False,
        "raw_patient_inputs_opened": False, "report_bodies_or_image_pixels_opened": False,
        "runtime_seconds": round(time.monotonic()-started, 6)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary/"summary.json", summary),
                 write_private_text(temporary/"RESULTS_CN_EN.md", markdown(results))]
        for name, result in results.items():
            flat = [{k:v for k,v in r.items() if k != "raw_edge_readouts"} | {
                f"{edge}_{field}": value for edge, values in r["raw_edge_readouts"].items()
                for field,value in values.items()} for r in result["rows"]]
            for suffix, table in (("candidate_coverage.csv", flat), ("model_coverage.csv", result["model_coverage"]),
                                  ("same_image_ties.csv", result["same_image_pairs"])):
                files.append(write_private_text(temporary/f"{name}_{suffix}", render_csv(table)))
        if before != {k: sha256_file(p) for k,p in sources.items()}:
            raise ValueError("immutable metadata source changed")
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA,
            "source_paths": {k:str(p) for k,p in sources.items()}, "source_sha256": before,
            "artifacts": {p.name:{"sha256":sha256_file(p)} for p in files},
            "new_model_calls": 0, "original_selection_changed": False, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fresh-run", "fresh-audit", "endpoint-run", "output-root", "run-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status":"failed", "error_type":type(exc).__name__})); return 1
    print(json.dumps({"status":summary["status"], "runtime_seconds":summary["runtime_seconds"],
                      "manifest_sha256":sha256_file(target/"manifest.json")})); return 0


if __name__ == "__main__":
    raise SystemExit(main())

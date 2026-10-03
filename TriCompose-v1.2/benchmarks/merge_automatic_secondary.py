#!/usr/bin/env python3
"""Private endpoint tables without changing the frozen policy or old winners."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.automatic_secondary import overlay, paired_case_comparisons, REQUEST_SCHEMA, SCORE_SCHEMA
from tricompose_v12.legacy_replay_adapter import SCHEMA as REPLAY_SCHEMA
from run_automatic_proxy_replay import render_csv
from run_legacy_automatic_replay import enriched_compare
from score_automatic_replay_biovil import MODEL_HASHES
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

SCHEMA = "tricompose-automatic-policy-secondary-comparison-v1"


def cached(root, schema, name, limit):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    mp = require_inside(root / "manifest.json", root, must_exist=True)
    manifest = read_json(mp)
    path = require_inside(root / name, root, must_exist=True)
    if (manifest.get("schema_version") != schema or path.stat().st_size > limit
            or sha256_file(path) != manifest["artifacts"][name]["sha256"]):
        raise ValueError("bounded hash-bound endpoint source required")
    return manifest, mp, path


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    rm, rmp, rp = cached(args.replay_run, REPLAY_SCHEMA, "replay_outcomes.jsonl", 64 * 1024 * 1024)
    _, qmp, qp = cached(args.request_run, REQUEST_SCHEMA, "request.json", 1024 * 1024)
    sm, smp, sp = cached(args.score_run, SCORE_SCHEMA, "scores.json", 2 * 1024 * 1024)
    request, scores = read_json(qp), read_json(sp)
    if (request["source_replay_manifest_sha256"] != sha256_file(rmp)
            or request["source_outcomes_sha256"] != sha256_file(rp)
            or request["source_scores_sha256"] != rm["source_sha256"]["source_scores"]
            or scores["request_sha256"] != sha256_file(qp)
            or scores["producer"]["checkpoint_sha256"] != MODEL_HASHES
            or sm["original_selection_changed"] is not False):
        raise ValueError("secondary/replay request lineage or frozen weights differ")
    rows = [json.loads(line) for line in rp.read_text().splitlines() if line]
    fp = require_inside(Path(args.replay_run) / "frozen_policy.json", PROTECTED_ROOT, must_exist=True)
    if sha256_file(fp) != rm["artifacts"][fp.name]["sha256"]:
        raise ValueError("frozen policy changed")
    policy = read_json(fp)
    merged = overlay(rows, request, scores)
    available = [r for r in scores["records"] if r["biovil_raw_cosine"] is not None]
    expected_counts = {"requested_pairs": len(scores["records"]),
        "image_encoder_calls": len({r["cxr_candidate_id"] for r in available}),
        "text_encoder_calls": len({r["report_candidate_id"] for r in available}),
        "unavailable_reports": len({r["report_candidate_id"] for r in scores["records"] if r["biovil_raw_cosine"] is None})}
    if scores["counts"] != expected_counts:
        raise ValueError("secondary availability/call counts differ")
    return merged, policy, scores, {"replay_manifest": rmp, "replay_outcomes": rp,
        "request_manifest": qmp, "secondary_request": qp,
        "score_manifest": smp, "secondary_scores": sp, "frozen_policy": fp}


def markdown(summary):
    def show(value): return "NA" if value is None else f"{value:.4f}"
    lines = ["# Automatic policies + BioViL-T endpoint / 自动策略补充评分", "",
        "BioViL-T was computed AFTER policy decisions and never used to choose a candidate, action or threshold.",
        "评分在原选择之后补算，只是辅助端点评价，不是新的择优或已验证的临床正确率。", "",
        "| Method | Call budget | Mean simulated calls | Proxy support | Proxy opposition | BioViL-T cosine | Available trials |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["comparisons"]:
        lines.append(f"| {row['method']} | {row['model_call_budget']} | {show(row['mean_simulated_model_calls'])} | {show(row['proxy_support_recall'])} | {show(row['proxy_opposition_rate'])} | {show(row['biovil_mean_secondary_not_routing'])} | {row['biovil_available_trials']}/{row['selected_trials']} |")
    lines += ["", "## Paired EHR-case comparison / 按同一 EHR 配对", "",
        "Random endpoints average ALL five predeclared seeds within one fixed EHR. If any seed is unavailable, that case's random endpoint remains unavailable for paired comparison.",
        "五次随机重复先在同一 EHR 内取均值，不当作五位患者；配对只使用双方可用的同一 EHR，缺失数量仍显式报告。", "",
        "| Subgroup | Budget | Baseline | Paired EHR cases | BioViL delta: targeted − baseline | Simulated call delta (all cases) |",
        "|---|---:|---|---:|---:|---:|"]
    maximum = max(r["model_call_budget"] for r in summary["paired_case_comparisons"])
    for row in summary["paired_case_comparisons"]:
        if row["model_call_budget"] == maximum:
            lines.append(f"| {row['ehr_evidence_subgroup']} | {maximum} | {row['baseline']} | {row['paired_available_ehr_cases']}/{row['all_fixed_ehr_cases']} | {show(row['mean_biovil_delta_targeted_minus_baseline'])} | {show(row['mean_simulated_call_delta_all_fixed_cases'])} |")
    lines += ["", "## Interpretation / 解读", "",
        "- The table reports the existing frozen policy as-is, including unfavorable comparisons. Neither report/CXR winners nor model priority were changed after BioViL scoring.",
        "- The complete per-budget, per-subgroup paired readouts are in `paired_case_comparison.csv`; the table above shows only the maximum budget, not a claim for every operating point.",
        "- A proxy gain with no BioViL gain supports only optimization of the selection proxy. A BioViL gain is another automatic readout, still NOT independent clinical truth or confirmed modality-error localization.",
        "- This historical cohort's uncalibrated fourteen-head scores cannot be pooled with the newer eight-head profile. Weak priors and missing EHR constraints remain unchanged; no-direct-EHR edge rates stay NA.",
        "- Full-report context/empty/special-token failures stay unavailable, not truncated or zero. Pairwise available-case coverage is preserved; there is no clinical rejection/filtering of cases.",
        "- Readout cosine is NOT a probability, not an absolute quality threshold and not a primary clinical metric. No significance test or radiologist validation is claimed.",
        "- Policy calls remain simulated. The actual separate BioViL endpoint job cost is recorded in `summary.json`; it is additional evaluation work, not free deployment cost or measured regeneration savings.",
        "- EHRs, original source tables, original action traces and selected IDs are unchanged. No new image/report generation, model training or API calls were performed.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    outcomes, policy, scores, sources = load(args)
    sources.update(program=Path(__file__), pure_endpoint_contract=ROOT / "src/tricompose_v12/automatic_secondary.py",
        proxy_comparison_contract=ROOT / "benchmarks/run_legacy_automatic_replay.py",
        base_comparison_contract=ROOT / "benchmarks/run_automatic_proxy_replay.py")
    before = {key: sha256_file(path) for key, path in sources.items()}
    grouped = defaultdict(list)
    for row in outcomes: grouped[row["input_ehr_assessment_scope"]].append(row)
    subgroups = [dict(ehr_evidence_subgroup=scope, **result) for scope, trials in sorted(grouped.items())
                 for result in enriched_compare(trials)]
    summary = {"schema_version": SCHEMA, "status": "completed_secondary_endpoint_comparison",
        "comparisons": enriched_compare(outcomes), "subgroup_comparisons": subgroups,
        "paired_case_comparisons": paired_case_comparisons(outcomes, policy),
        "unique_fixed_ehr_cases": len({r["case_id"] for r in outcomes}), "replay_trials": len(outcomes),
        "endpoint_scoring_counts": scores["counts"],
        "actual_separate_gpu_endpoint_scoring_runtime_seconds": scores["runtime_seconds"],
        "actual_endpoint_peak_vram_gib": scores["peak_vram_gib"],
        "original_selection_changed": False, "routing_or_thresholds_updated": False,
        "actual_regeneration_executed": False, "actual_gpu_generation_savings": None,
        "clinical_accuracy": None, "independent_clinical_truth_available": False,
        "runtime_seconds": round(time.monotonic()-started, 6)}
    if before != {key: sha256_file(path) for key, path in sources.items()}:
        raise ValueError("immutable endpoint source changed during merge")
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "method_comparison.csv", render_csv(summary["comparisons"])),
            write_private_text(temporary / "subgroup_comparison.csv", render_csv(subgroups)),
            write_private_text(temporary / "paired_case_comparison.csv", render_csv(summary["paired_case_comparisons"])),
            write_private_text(temporary / "endpoint_outcomes.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in outcomes)),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary))]
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "original_selection_changed": False, "routing_or_thresholds_updated": False,
            "clinical_accuracy_claim_allowed": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-run", "request-run", "score-run", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try: target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"],
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

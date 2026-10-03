#!/usr/bin/env python3
"""Expand automatic policy replay to the fixed historical synthetic cohort.

CPU-only metadata replay inside an existing Slurm allocation. Never read EHR,
report bodies or image pixels; never overwrite artifacts or source selections.
"""
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
from tricompose_v12.automatic_replay import METHODS, replay_case
from tricompose_v12.legacy_replay_adapter import SCHEMA, make_legacy_bank, inventory_summary
from run_automatic_proxy_replay import compare, render_csv
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def checked_source(root, schema, filename, limit):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    mp = require_inside(root / "manifest.json", root, must_exist=True)
    if mp.stat().st_size > 64 * 1024:
        raise ValueError("unbounded source manifest")
    manifest = read_json(mp)
    if manifest.get("schema_version") != schema:
        raise ValueError("source schema differs")
    path = require_inside(root / filename, root, must_exist=True)
    if (path.stat().st_size > limit
            or sha256_file(path) != manifest.get("artifacts", {}).get(filename, {}).get("sha256")):
        raise ValueError("bounded hash-bound source required")
    return manifest, mp, path


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    sm, smp, score_path = checked_source(args.selection_run,
        "tricompose-edge-specific-selection-v1.1", "candidate_score_table.jsonl", 4 * 1024 * 1024)
    em, emp, edge_path = checked_source(args.edge_run,
        "tricompose-ehr-edge-crossmodal-evaluation-v1.1", "ehr_edge_details.json", 8 * 1024 * 1024)
    if sm.get("evaluation_status") != "diagnostic_uncalibrated_cxr_labels":
        raise ValueError("historical uncalibrated profile must be explicit")
    config = require_inside(args.policy_config, WORKSPACE, must_exist=True)
    if config.stat().st_size > 16 * 1024:
        raise ValueError("unbounded policy")
    policy = json.loads(config.read_text())
    scores = [json.loads(line) for line in score_path.read_text().splitlines() if line]
    details = read_json(edge_path)
    bank = make_legacy_bank(scores, details, policy)
    expected = {"candidate_rows": len(scores), "cases": len(bank),
                "cxr_candidates": len(details["records"]["ehr_cxr"]),
                "report_candidates": len(details["records"]["ehr_report"])}
    if any(sm["counts"].get(k) != v for k, v in expected.items()):
        raise ValueError("selection manifest inventory differs")
    if any(em["counts"].get(k) != details["counts"].get(k)
           for k in ("cases", "cxr_candidates", "report_candidates")):
        raise ValueError("edge manifest inventory differs")
    if expected != {"candidate_rows": 960, "cases": 80,
                    "cxr_candidates": 240, "report_candidates": 960}:
        raise ValueError("entire frozen eighty-case cohort required, not a selected subset")
    sources = {"selection_manifest": smp, "source_scores": score_path,
               "edge_manifest": emp, "source_edges": edge_path, "policy_config": config}
    overlap = {"checked": False, "ehr_sha256_overlap_count": None,
               "independent_clinical_truth_available": False}
    if args.development_selection_run:
        _, dp, ds = checked_source(args.development_selection_run,
            "tricompose-edge-specific-selection-v1.1", "candidate_score_table.jsonl", 4 * 1024 * 1024)
        prior = [json.loads(line) for line in ds.read_text().splitlines() if line]
        old = {r["lineage"]["ehr_sha256"] for r in prior}
        current = {r["lineage"]["ehr_sha256"] for r in scores}
        overlap.update(checked=True, ehr_sha256_overlap_count=len(old & current),
                       independent_generator_training_provenance_verified=False)
        sources.update(development_manifest=dp, development_scores=ds)
    return bank, policy, sources, overlap


def enriched_compare(outcomes):
    rows = compare(outcomes)
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[(row["method"], row["model_call_budget"])].append(row)
    for result in rows:
        snapshots = [r["selected_snapshot"] for r in grouped[(result["method"], result["model_call_budget"])]
                     if r["selected_snapshot"] is not None]
        source, raw = {}, {}
        for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
            values = [s["source_edge_metrics_optimization_proxy"][edge] for s in snapshots]
            known = sum(v["known_reference_fact_count"] for v in values)
            support = sum(v["support_count"] for v in values)
            opposition = sum(v["contradiction_count"] for v in values)
            source[edge] = {"known": known, "support": support, "opposition": opposition,
                "support_recall": support / known if known else None,
                "opposition_rate": opposition / known if known else None,
                "source_semantics": "legacy_global_no_finding_adjustment" if edge == "ehr_report" else "raw_explicit_states"}
            rvalues = [s["raw_edge_readouts_optimization_proxy"][edge] for s in snapshots]
            comparable = sum(v["comparable_facts"] for v in rvalues)
            raw[edge] = {"inventory": sum(v["inventory_facts"] for v in rvalues),
                "known": sum(v["explicit_reference_facts"] for v in rvalues),
                "comparable": comparable,
                "support": sum(v["support"] for v in rvalues),
                "opposition": sum(v["opposition"] for v in rvalues),
                "coverage_over_known": comparable / known if known else None,
                "clinical_accuracy": None}
        result["source_edge_metrics_optimization_proxy"] = source
        result["raw_edge_readouts_optimization_proxy"] = raw
    return rows


def markdown(summary):
    def show(value):
        return "NA" if value is None else f"{value:.4f}"
    lines = ["# Fixed historical cohort: automatic policy replay / 固定队列自动策略对比", "",
        "这是无需人工反馈的 frozen-bank CPU replay，不是新生成结果，也不是临床错误定位验证。", "",
        f"Fixed EHR cases: {summary['unique_ehr_cases']}; source triples: {summary['source_candidate_triples']}; replay trials: {summary['replay_trials']}.",
        "Three CXR models × one seed × four report models per EHR. All input cases are retained.", "",
        "| Method | Budget | Mean simulated calls | Proxy support | Proxy opposition | BioViL-T |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["comparisons"]:
        lines.append(f"| {row['method']} | {row['model_call_budget']} | {show(row['mean_simulated_model_calls'])} | {show(row['proxy_support_recall'])} | {show(row['proxy_opposition_rate'])} | {show(row['biovil_mean_secondary_not_routing'])} |")
    lines += ["", "## EHR evidence subgroups / EHR 证据分组", "",
        "Direct constraints are counted once per fixed EHR, not once per triple or random replicate:",
        "```json", json.dumps(summary["source_inventory"]["case_counts_by_ehr_evidence"], indent=2, sort_keys=True), "```", "",
        "`subgroup_comparison.csv` preserves both groups, including unselected trials. In the no-direct-EHR group, EHR edge support/contradiction rates are NA, not zero or perfect agreement.", "",
        "## Scope and limitations / 范围与限制", "",
        "- The policy logic and model priority are reused; the legacy pool has three image slots and fourteen historical raw label heads, NOT the new calibrated eight-head profile. Different profiles are not pooled or numerically compared as the same evaluator.",
        "- The maximum complete-bank cost is 30 simulated calls per case: 3 image generations + 3 XRV calls + 12 report generations + 12 CheXbert calls. EHR generation is the same shared sunk cost.",
        "- Fixed, random (all five seeds), static prefix reranking, and targeted heuristic see only requested scores; source winner/rank flags never affect the next request.",
        "- All EHRs stay fixed. There is no extra fact/device insertion, case replacement, threshold fitting or post-result model reordering.",
        "- Unknown/uncertain remain unknown/uncertain. Weak CHF/medication context does not become a hard radiographic constraint.",
        "- Legacy EHR–Report No-Finding counts are retained explicitly as a source proxy. Raw states are not flipped; source-edge and raw-edge tables are both saved.",
        "- BioViL-T and guarded report readouts are absent for this cohort and remain NA. No syntax guard or independent clinical truth is invented.",
        "- An image-only or image–report proxy stop with no direct EHR facts is NOT validated three-modal consistency. Low opposition with low coverage is not clinical correctness.",
        "- Correlated reports on one image are not independent votes. A targeted request is a heuristic action, not a confirmed faulty modality.",
        "- Repeated random trials do not increase the number of independent EHRs. Proxy metrics are optimized and evaluated by the same cached scorers; a gain on them is not independent evidence.",
        "- CPU replay cannot recover historical GPU computation. Simulated calls are not measured GPU seconds and exclude separately needed alternate evaluation/failure cost.",
        "- No image pixels, report bodies, raw patient data, models, new GPU jobs or original selected triples were accessed/modified by this program.",
        "- Next: freeze this policy, add a distinct automatic evaluator to the selected union with a reviewed/approved GPU job, then measure actual bounded targeted generation.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    bank, policy, sources, overlap = load(args)
    sources.update(program=Path(__file__),
        cache_adapter=ROOT / "src/tricompose_v12/legacy_replay_adapter.py",
        replay_implementation=ROOT / "src/tricompose_v12/automatic_replay.py",
        comparison_implementation=ROOT / "benchmarks/run_automatic_proxy_replay.py",
        relation_contract=ROOT / "src/tricompose_v12/report_scope_table.py",
        private_contract=ROOT.parent / "TriCompose-v1.0/eval/report_v1_1/contracts.py")
    before = {name: sha256_file(path) for name, path in sources.items()}
    case_scope = {case: ("explicit_fact_proxy" if any(f["states"]["ehr"] in {"positive", "negative"}
                    for f in next(iter(grid.values()))["facts"]) else "no_direct_comparable_ehr_facts")
                  for case, grid in bank.items()}
    outcomes = []
    for case, grid in sorted(bank.items()):
        for budget in policy["model_call_budgets"]:
            for method in METHODS:
                for seed in policy["random_seeds"] if method == "random" else [0]:
                    row = replay_case(grid, policy, method, budget, random_seed=seed)
                    row["input_ehr_assessment_scope"] = case_scope[case]
                    outcomes.append(row)
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[row["input_ehr_assessment_scope"]].append(row)
    subgroups = [dict(ehr_evidence_subgroup=scope, **row) for scope, trials in sorted(grouped.items())
                 for row in enriched_compare(trials)]
    if before != {name: sha256_file(path) for name, path in sources.items()}:
        raise ValueError("source changed during replay")
    summary = {"schema_version": SCHEMA, "status": "completed_legacy_automatic_proxy_replay",
        "unique_ehr_cases": len(bank), "source_candidate_triples": sum(len(grid) for grid in bank.values()),
        "replay_trials": len(outcomes), "comparisons": enriched_compare(outcomes),
        "subgroup_comparisons": subgroups, "source_inventory": inventory_summary(bank),
        "development_cohort_overlap": overlap,
        "new_model_calls": 0, "actual_new_gpu_seconds": 0, "actual_gpu_savings": None,
        "new_regeneration_executed": False, "original_selection_changed": False,
        "human_feedback_required_for_replay": False, "clinical_accuracy_claim_allowed": False,
        "clinical_localization_accuracy": None, "policy_fitted_to_this_cohort": False,
        "runtime_seconds": round(time.monotonic() - started, 6)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_text(temporary / "replay_outcomes.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in outcomes)),
            write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "method_comparison.csv", render_csv(summary["comparisons"])),
            write_private_text(temporary / "subgroup_comparison.csv", render_csv(subgroups)),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary)),
            write_private_json(temporary / "frozen_policy.json", policy)]
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "human_feedback_required_for_replay": False,
            "new_regeneration_executed": False, "original_selection_changed": False,
            "clinical_accuracy_claim_allowed": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection-run", "edge-run", "policy-config", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--development-selection-run")
    args = parser.parse_args(); os.umask(0o007)
    try:
        target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"],
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

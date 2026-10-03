#!/usr/bin/env python3
"""No-human automatic policy comparison using only frozen synthetic caches.

No model inference, artifact body/pixels, new EHR, training, or old-run mutation.
Outputs optimize silver/proxy scores, NOT clinical correctness or real savings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.automatic_replay import SCHEMA, METHODS, make_bank, replay_case
from preview_candidate_actions import load_cached_facts
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def load(args):
    # Guard before accessing any private cache or external dataset.
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    facts, sources = load_cached_facts(args.scope_run)
    root = require_inside(args.selection_run, PROTECTED_ROOT, must_exist=True)
    mp = require_inside(root / "manifest.json", root, must_exist=True)
    manifest = read_json(mp)
    if manifest.get("schema_version") != "tricompose-edge-specific-selection-v1.1":
        raise ValueError("completed frozen diagnostic selection required")
    path = require_inside(root / "candidate_score_table.jsonl", root, must_exist=True)
    if (path.stat().st_size > 4 * 1024 * 1024
            or sha256_file(path) != manifest.get("artifacts", {}).get(path.name, {}).get("sha256")):
        raise ValueError("bounded, hash-bound source scores required")
    scores = [json.loads(line) for line in path.read_text().splitlines() if line]
    config_path = require_inside(args.policy_config, WORKSPACE, must_exist=True)
    policy = json.loads(config_path.read_text())
    bank = make_bank(scores, facts, policy)
    sources.update(selection_manifest=mp, source_scores=path, policy_config=config_path)
    return bank, policy, sources


def _mean(values):
    return statistics.mean(values) if values else None


def compare(outcomes):
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[(row["method"], row["model_call_budget"])].append(row)
    comparisons = []
    for (method, budget), trials in sorted(grouped.items()):
        valid = [row for row in trials if row["selected_snapshot"] is not None]
        snapshots = [row["selected_snapshot"] for row in valid]
        proxy = [snap["optimization_proxy"] for snap in snapshots]
        known = sum(p["total_known_reference_fact_count"] for p in proxy)
        contradictions = sum(p["total_hard_contradiction_count"] for p in proxy)
        supports = sum(p["total_direct_support_count"] for p in proxy)
        cosines = [s["biovil_cosine_secondary_not_routing"] for s in snapshots if s["biovil_cosine_secondary_not_routing"] is not None]
        edges = {}
        for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
            values = [s["guarded_edge_readouts_not_routing"][edge] for s in snapshots
                      if s["guarded_edge_readouts_not_routing"][edge] is not None]
            if not values:
                edges[edge] = None
                continue
            comparable = sum(v["comparable_facts"] for v in values)
            support = sum(v["support"] for v in values)
            opposition = sum(v["opposition"] for v in values)
            edges[edge] = {"available_trials": len(values), "comparable": comparable, "support": support, "opposition": opposition,
                          "support_on_comparable": support / comparable if comparable else None,
                          "inventory": sum(v["inventory_facts"] for v in values)}
        comparisons.append({"method": method, "model_call_budget": budget,
            "unique_ehr_cases": len({row["case_id"] for row in trials}), "replay_trials": len(trials),
            "random_replicates_per_case": len({row["random_seed"] for row in trials}) if method == "random" else 1,
            "selected_trials": len(valid), "no_eligible_candidate_trials": len(trials) - len(valid),
            "mean_simulated_model_calls": _mean([row["simulated_model_calls"] for row in trials]),
            "mean_simulated_generator_calls": _mean([sum(row["simulated_calls"].get(k, 0) for k in ("cxr_generator", "report_generator")) for row in trials]),
            "mean_simulated_selection_scorer_calls": _mean([sum(row["simulated_calls"].get(k, 0) for k in ("xrv", "chexbert")) for row in trials]),
            "mean_observed_candidates": _mean([row["observed_candidates"] for row in trials]),
            "proxy_stop_satisfied_trials": sum(row["selected_proxy_stop_conditions_met"] for row in trials),
            "pooled_proxy_known_facts_across_trials": known,
            "pooled_proxy_support_across_trials": supports,
            "pooled_proxy_opposition_across_trials": contradictions,
            "proxy_support_recall": supports / known if known else None,
            "proxy_opposition_rate": contradictions / known if known else None,
            "proxy_balance_0_100": 50 * (1 + (supports - contradictions) / known) if known else None,
            "biovil_mean_secondary_not_routing": _mean(cosines), "biovil_available_trials": len(cosines),
            "guarded_edge_readouts": edges,
            "clinical_accuracy": None, "prospective_gpu_seconds": None,
            "independent_patient_count_is_not_replay_trials": True})
    return comparisons


def render_csv(comparisons):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(comparisons[0]), lineterminator="\n")
    writer.writeheader()
    for record in comparisons:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in record.items()})
    return output.getvalue()


def markdown(summary):
    lines = ["# Automatic policy replay / 无人工自动策略对比", "",
        "No human feedback is required for this exploratory automatic benchmark.",
        "这里只检验自动代理指标与调用预算，不声称临床正确性、自然错误定位准确率或真实 GPU 节省。", "",
        "| Method | Call budget | Mean calls incl. XRV/CheXbert | Proxy support | Proxy opposition | BioViL-T secondary |",
        "|---|---:|---:|---:|---:|---:|"]
    def show(x): return "NA" if x is None else f"{x:.4f}"
    for row in summary["comparisons"]:
        lines.append(f"| {row['method']} | {row['model_call_budget']} | {show(row['mean_simulated_model_calls'])} | {show(row['proxy_support_recall'])} | {show(row['proxy_opposition_rate'])} | {show(row['biovil_mean_secondary_not_routing'])} |")
    lines += ["", "## Interpretation / 如何理解", "",
        "- XRV/CheXbert and the old V1.1 lexicographic objective drive selection; BioViL-T and frozen guarded edge readouts never drive routing.",
        "- Raw CheXbert outside the four-finding syntax guard is an explicitly UNVERIFIED operational proxy, not a clinically validated assertion.",
        "- Unknown/uncertain are never negative. Unavailable guarded EHR–Report comparisons remain unavailable, not corrected or perfect.",
        "- Every EHR stays fixed. Routing never sees another candidate's scores until requesting that model/seed slot.",
        "- Same-CXR reports are correlated; their majority never proves the CXR wrong. Routing locus is heuristic, not a confirmed fault.",
        "- Random runs include every predeclared seed; repeated trials are NOT independent patients.",
        "- Calls are simulated single-case generator + selection-scorer invocations; shared images are charged once per path slot within a replay.",
        "- Secondary end-point evaluation is additional already-cached work, not free deployment compute. Actual prospective GPU time/failed-call cost has not been measured.",
        "- A budget-limited best candidate may still be inconsistent. Proxy stop success is not clinical acceptance.",
        "- The original bank was already fully generated; CPU replay does not recover or save that historical computation.",
        "- This is a two-EHR development smoke, not a population study or untouched final test; do not tune a policy on these outcomes.",
        "- Original score tables, artifacts, and selected triples remain unchanged.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    bank, policy, sources = load(args)
    before = {name: sha256_file(path) for name, path in sources.items()}
    outcomes = []
    for case, case_bank in sorted(bank.items()):
        for budget in policy["model_call_budgets"]:
            for method in METHODS:
                for seed in policy["random_seeds"] if method == "random" else [0]:
                    outcomes.append(replay_case(case_bank, policy, method, budget, random_seed=seed))
    if before != {name: sha256_file(path) for name, path in sources.items()}:
        raise ValueError("immutable source changed during replay")
    summary = {"schema_version": SCHEMA, "status": "completed_automatic_heuristic_proxy_replay",
        "unique_ehr_cases": len(bank), "source_candidate_triples": sum(len(values) for values in bank.values()),
        "replay_trials": len(outcomes), "comparisons": compare(outcomes),
        "new_model_calls": 0, "actual_new_gpu_seconds": 0, "actual_gpu_savings": None,
        "new_regeneration_executed": False, "original_selection_changed": False,
        "human_feedback_required_for_replay": False, "clinical_accuracy_claim_allowed": False,
        "clinical_localization_accuracy": None, "policy_fitted_to_this_cohort": False,
        "runtime_seconds": round(time.monotonic() - started, 6)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_text(temporary / "replay_outcomes.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in outcomes)),
            write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "method_comparison.csv", render_csv(summary["comparisons"])),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary)),
            write_private_json(temporary / "frozen_policy.json", policy)]
        sources.update(program=Path(__file__), policy_implementation=ROOT / "src/tricompose_v12/automatic_replay.py",
                       fact_contract=ROOT / "src/tricompose_v12/decision_preview.py",
                       scope_contract=ROOT / "src/tricompose_v12/report_scope_table.py")
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {name: str(path) for name, path in sources.items()},
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files},
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
    for name in ("scope-run", "selection-run", "policy-config", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
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

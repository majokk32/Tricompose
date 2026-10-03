#!/usr/bin/env python3
"""CPU-only fixed-image report controls with immutable existing endpoint scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.fixed_image_control import SCHEMA, build_trials, contrasts, validate_control
from diagnose_automatic_discrepancy import load as load_source
from run_legacy_automatic_replay import enriched_compare
from run_automatic_proxy_replay import render_csv
from contracts import (WORKSPACE, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    path = require_inside(args.control_config, WORKSPACE, must_exist=True)
    if path.stat().st_size > 16 * 1024:
        raise ValueError("unbounded control config")
    control = json.loads(path.read_text())
    original, bank, policy, sources = load_source(args)
    validate_control(control, policy)
    sources["control_config"] = path
    return original, bank, policy, control, sources


def markdown(summary):
    def show(value): return "NA" if value is None else f"{value:.4f}"
    maximum = max(r["model_call_budget"] for r in summary["comparisons"])
    lines = ["# Fixed-image report selection / 固定图像报告选择对照", "",
        "All 80 EHRs retain their existing Sana seed 0 image for the two new report-only policies. Image selection did not use report/endpoint scores.",
        "固定每例原 Sana seed 0 胸片，不添加病种、不换 EHR；比较两个同图策略与原固定、联合定向和联合静态策略。", "",
        "This previously inspected cohort is DEVELOPMENT, not a new independent test. No model, raw patient input, report body or image pixels were opened.", "",
        "## Maximum call CAP / 最大调用上限", "",
        "Equal caps do not mean equal expenditure: four reports on one image cost at most ten simulated generator/scorer calls. Full curves are in `method_comparison.csv`.",
        "| Method | Cap | Mean simulated calls | Proxy support | Proxy opposition | BioViL mean (available only) | Available EHR cases |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["comparisons"]:
        if r["model_call_budget"] == maximum:
            lines.append(f"| {r['method']} | {maximum} | {show(r['mean_simulated_model_calls'])} | {show(r['proxy_support_recall'])} | {show(r['proxy_opposition_rate'])} | {show(r['biovil_mean_secondary_not_routing'])} | {r['biovil_available_trials']}/{r['unique_ehr_cases']} |")
    lines += ["", "Do not compare the available-only means above when their case sets differ. Use the paired table below; missing coverage remains explicit.",
        "上表可用病例不同，不能直接比较均值。配对表只比较双方均有分数的同一 EHR，全部病例的成本另算。", "",
        "## Paired readouts at the maximum CAP / 同病例配对", "",
        "| Report-only method | Baseline | Paired BioViL cases | BioViL delta | Proxy-balance delta on SAME pairs | Mean call delta (all cases) |",
        "|---|---|---:|---:|---:|---:|"]
    for r in summary["paired_comparisons"]:
        if r["model_call_budget"] == maximum and r["ehr_evidence_subgroup"] == "all":
            lines.append(f"| {r['method']} | {r['baseline']} | {r['paired_available_biovil_cases']}/{r['all_fixed_ehr_cases']} | {show(r['mean_biovil_delta_on_paired_available_cases'])} | {show(r['mean_source_proxy_balance_delta_same_biovil_pairs'])} | {show(r['mean_simulated_call_delta_all_fixed_cases'])} |")
    lines += ["", "## Coverage, behavior and limitations / 覆盖、行为与限制", "",
        f"- New selected pairs awaiting an endpoint score: {summary['pending_unique_endpoint_pairs']}. Their choices stay fixed and their scores remain NA; `pending_endpoint_pairs.json` is metadata only, NOT job authorization.",
        "- `report_only_targeted` stops explicitly when the original heuristic asks to change the fixed image. A blocked action or exhausted report inventory is unresolved, not a successful clinical repair.",
        "- Source ranking, four-state semantics, report order and all original winners/actions remain unchanged. BioViL is attached AFTER selection; endpoint availability never affects selection.",
        "- Direct-EHR/no-direct-EHR subgroups are in `subgroup_comparison.csv` and `paired_case_comparison.csv`. Unavailable EHR rates stay NA; weak context is not a hard finding.",
        "- Positive/negative agreement is decomposed in paired contrasts. Negative agreement is valid and does not justify forcing pathology. Legacy No-Finding source adjustments are not raw-label edits.",
        "- Same-image reference labels stay invariant across report-only candidates, but XRV/CheXbert and global BioViL cosine remain automatic proxies, not independent clinical truth.",
        "- No significance, natural clinical fault localization, physically regenerated improvement or actual GPU savings are claimed. Historical generation cost is sunk; future failures/retries and endpoint costs are additional.",
        "- Original `random` acquired candidates in random order then applied score-based reranking; it equals static selection at full inventory. It is not a pure-random final choice and is not relabeled here.",
        "- This cohort and its previous discrepancies have already been inspected. Future confirmatory controls require a separately frozen evaluation cohort and policy; do not present this as untouched validation.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    original, bank, policy, control, sources = load(args)
    sources.update(program=Path(__file__), control_contract=ROOT / "src/tricompose_v12/fixed_image_control.py",
        source_loader=ROOT / "benchmarks/diagnose_automatic_discrepancy.py",
        source_rank_contract=ROOT / "src/tricompose_v12/automatic_replay.py",
        source_cache_contract=ROOT / "src/tricompose_v12/legacy_replay_adapter.py",
        polarity_contract=ROOT / "src/tricompose_v12/automatic_discrepancy.py",
        protocol=WORKSPACE / "docs/fixed_image_control_protocol.md")
    before = {k: sha256_file(p) for k, p in sources.items()}
    rows, pending = build_trials(original, bank, policy, control)
    case_contrasts, paired = contrasts(rows, bank, control)
    by_scope = defaultdict(list)
    for row in rows: by_scope[row["input_ehr_assessment_scope"]].append(row)
    subgroups = [dict(ehr_evidence_subgroup=scope, **r) for scope, trials in sorted(by_scope.items())
                 for r in enriched_compare(trials)]
    terminal_groups = defaultdict(Counter)
    for row in rows:
        terminal_groups[(row["method"], row["model_call_budget"])][row["terminal_reason"]] += 1
    terminals = [{"method": method, "model_call_budget": budget, "terminal_reason": reason, "fixed_ehr_cases": count}
        for (method, budget), counts in sorted(terminal_groups.items()) for reason, count in sorted(counts.items())]
    summary = {"schema_version": SCHEMA, "status": "completed_development_fixed_image_control",
        "fixed_ehr_cases": len(bank), "source_candidate_triples": sum(map(len, bank.values())),
        "control_trials_including_reused_baselines": len(rows), "new_report_only_replay_trials":
            sum(r["method"] in control["report_methods"] for r in rows),
        "comparisons": enriched_compare(rows), "subgroup_comparisons": subgroups,
        "paired_comparisons": paired, "terminal_reason_counts": terminals,
        "pending_unique_endpoint_pairs": len(pending),
        "image_selection_used_report_or_endpoint_scores": False, "new_model_calls": 0,
        "original_selection_changed": False, "original_policy_changed": False,
        "actual_regeneration_executed": False, "actual_gpu_savings": None,
        "clinical_accuracy": None, "independent_clinical_truth_available": False,
        "cohort_role": control["cohort_role"], "runtime_seconds": round(time.monotonic()-started, 6)}
    if before != {k: sha256_file(p) for k, p in sources.items()}:
        raise ValueError("immutable control source changed")
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "summary.json", summary),
            write_private_json(temporary / "frozen_control.json", control),
            write_private_text(temporary / "control_outcomes.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in rows)),
            write_private_text(temporary / "case_contrasts.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in case_contrasts)),
            write_private_text(temporary / "method_comparison.csv", render_csv(summary["comparisons"])),
            write_private_text(temporary / "subgroup_comparison.csv", render_csv(subgroups)),
            write_private_text(temporary / "paired_case_comparison.csv", render_csv(paired)),
            write_private_text(temporary / "terminal_reasons.csv", render_csv(terminals)),
            write_private_json(temporary / "pending_endpoint_pairs.json", {
                "schema_version": "tricompose-fixed-image-pending-endpoint-inventory-v1",
                "status": "metadata_only_not_execution_authorization", "modality_source": "fully_synthetic",
                "selection_used_biovil": False, "pairs": pending}),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary))]
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "original_selection_changed": False,
            "original_policy_changed": False, "clinical_accuracy_claim_allowed": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-run", "control-config", "output-root", "run-id"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try: target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"],
        "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

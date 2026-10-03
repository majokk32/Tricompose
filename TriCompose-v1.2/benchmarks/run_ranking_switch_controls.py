#!/usr/bin/env python3
"""CPU-only cached runtime-rank and invariant-EHR image-switch controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.ranking_switch_controls import SCHEMA, ranking_ablation, invariant_switches, validate_control
from tricompose_v12.fixed_image_control import SCHEMA as FIXED_SCHEMA, build_trials
from diagnose_automatic_discrepancy import load as load_source
from merge_automatic_secondary import cached
from run_automatic_proxy_replay import render_csv
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    config_path = require_inside(args.control_config, WORKSPACE, must_exist=True)
    if config_path.stat().st_size > 16 * 1024: raise ValueError("unbounded control config")
    control = json.loads(config_path.read_text()); validate_control(control)
    fm, fmp, fp = cached(args.fixed_image_run, FIXED_SCHEMA, "control_outcomes.jsonl", 64 * 1024 * 1024)
    if fm.get("original_selection_changed") is not False or fm.get("original_policy_changed") is not False:
        raise ValueError("immutable original control required")
    original, bank, policy, sources = load_source(args)
    for name in ("endpoint_manifest", "source_scores", "source_edges"):
        if sha256_file(sources[name]) != fm["source_sha256"][name]:
            raise ValueError("fixed-control and source evidence differ")
    frozen_path = require_inside(Path(args.fixed_image_run) / "frozen_control.json", PROTECTED_ROOT, must_exist=True)
    if sha256_file(frozen_path) != fm["artifacts"][frozen_path.name]["sha256"]:
        raise ValueError("original fixed control changed")
    fixed_config = read_json(frozen_path)
    rows = [json.loads(line) for line in fp.read_text().splitlines() if line]
    regenerated, _ = build_trials(original, bank, policy, fixed_config)
    if rows != regenerated: raise ValueError("original fixed control does not replay exactly")
    sources.update(fixed_control_manifest=fmp, fixed_control_outcomes=fp,
        fixed_control_policy=frozen_path, control_config=config_path)
    return original, rows, bank, control, sources


def markdown(summary):
    def show(value): return "NA" if value is None else f"{value:.4f}"
    maximum = max(r["model_call_budget"] for r in summary["ranking_comparisons"])
    lines = ["# Ranking and invariant-switch controls / 排序与固定 EHR 证据对照", "",
        "DEVELOPMENT on the already inspected cohort. No patient source inputs, report bodies, image pixels, models, API or GPU were opened/called.",
        "本轮不改旧赢家、病例、模型、阈值或动作。只是去掉最终排序的耗时项，并用固定 EHR 标签检查已有换图结果。", "",
        "## Runtime tie-break ablation / 去掉耗时择优", "",
        "Same observed slots, call ledger and stopping history. Original proxy-stop choices are retained; this is NOT a new routing policy or prospective repair.",
        "| Source method | Cap | Changed winners | Changed images | BioViL paired available | Mean delta: no-runtime − original | Wins / losses / ties |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for r in summary["ranking_comparisons"]:
        if r["model_call_budget"] == maximum and r["ehr_evidence_subgroup"] == "all":
            lines.append(f"| {r['source_method']} | {maximum} | {r['changed_winner_cases']} | {r['changed_image_cases']} | {r['paired_available_biovil_cases']}/{r['all_fixed_ehr_cases']} | {show(r['mean_biovil_delta_no_runtime_minus_original'])} | {r['paired_biovil_wins']} / {r['paired_biovil_losses']} / {r['paired_biovil_ties']} |")
    lines += ["", f"Unscored new endpoint pairs: {summary['pending_unique_endpoint_pairs']}. Their choices remain fixed and their cosines remain NA, not zero or a scored substitute.",
        "前五项排序准则严格保持同一优先级；ID 打平只是确定性规则，不是新的临床证据。耗时缺失不记零。调用次数完全不变。", "",
        "## Image-switch evidence / 换图对固定 EHR 证据的影响", "",
        "Below: only actual image changes at the maximum cap, compared with the original fixed path. Direct EHR reference states/hashes do not change.",
        "| Joint method | Switched EHR cases | With direct constraints | Without direct constraints | Fixed-EHR image support delta | All-three support delta | Positive image/report support gain without direct EHR | Negative image/report support gain without direct EHR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["invariant_switch_comparisons"]:
        if r["model_call_budget"] == maximum and r["ehr_evidence_subgroup"] == "all" and r["image_change_group"] == "changed":
            lines.append(f"| {r['method']} | {r['fixed_ehr_cases']} | {r['cases_with_direct_ehr_constraints']} | {r['cases_without_direct_ehr_constraints']} | {show(r['sum_direct_image_support_delta_constrained_cases'])} | {show(r['sum_all_three_support_delta_constrained_cases'])} | {r['sum_report_cxr_positive_support_without_direct_ehr_delta_all_selected_pairs']} | {r['sum_report_cxr_negative_support_without_direct_ehr_delta_all_selected_pairs']} |")
    lines += ["", "## Interpretation and files / 解读与文件", "",
        "- Source ranking and all old action/selection files remain immutable. Runtime ablation does not optimize BioViL or reveal more candidates.",
        "- BioViL means use identical paired available EHR cases; missing coverage remains explicit. Subgroup/budget tables are in `ranking_comparison.csv` and `invariant_switch_summary.csv`.",
        "- Invariant support means agreement with existing explicit cached EHR proxy states, NOT independently verified radiographic truth. No-direct-EHR cases stay in the cohort; their fixed-EHR support rates remain NA.",
        "- A new classifier/reference can improve image/report agreement without evidence of improved EHR fidelity. The with/without-direct-EHR counts diagnose that possibility; they do not attribute a clinical error or declare repair success.",
        "- Unknown/uncertain never become negative; absent EHR information is not a hallucination label, and weak medication/lab context never becomes a hard finding.",
        "- Normal/negative agreement can be valid. Do not insert diseases/devices, remove cases, fit new weights or choose favorable scorer/template subsets.",
        "- Global cosine is not a probability or clinical acceptance gate. No significance, natural clinical error localization or generalization claim is established.",
        "- Pending endpoint metadata is NOT execution approval. Any GPU scoring needs a separately shown complete script/resource request and approval; there is no new job here.",
        "- Per-case ranking and switch records remain protected in `ranking_case_contrasts.jsonl` and `invariant_switch_cases.jsonl`. Actual prospective GPU costs and generation savings remain unavailable.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    original, rows, bank, control, sources = load(args)
    sources.update(program=Path(__file__), control_contract=ROOT / "src/tricompose_v12/ranking_switch_controls.py",
        fixed_control_contract=ROOT / "src/tricompose_v12/fixed_image_control.py",
        source_rank_contract=ROOT / "src/tricompose_v12/automatic_replay.py",
        polarity_contract=ROOT / "src/tricompose_v12/automatic_discrepancy.py",
        source_loader=ROOT / "benchmarks/diagnose_automatic_discrepancy.py",
        protocol=WORKSPACE / "docs/ranking_switch_control_protocol.md")
    before = {k: sha256_file(p) for k, p in sources.items()}
    ablated, rank_cases, ranks, pending = ranking_ablation(rows, bank, original, control)
    switches, invariants = invariant_switches(rows, bank, control)
    summary = {"schema_version": SCHEMA, "status": "completed_development_ranking_switch_controls",
        "fixed_ehr_cases": len(bank), "ranking_trials": len(ablated), "invariant_switch_case_contrasts": len(switches),
        "ranking_comparisons": ranks, "invariant_switch_comparisons": invariants,
        "pending_unique_endpoint_pairs": len(pending), "new_model_calls": 0,
        "original_selection_changed": False, "routing_or_stopping_changed": False,
        "actual_regeneration_executed": False, "actual_gpu_savings": None,
        "clinical_accuracy": None, "independent_clinical_truth_available": False,
        "cohort_role": control["cohort_role"], "runtime_seconds": round(time.monotonic()-started, 6)}
    if before != {k: sha256_file(p) for k, p in sources.items()}:
        raise ValueError("immutable control source changed")
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "summary.json", summary),
            write_private_json(temporary / "frozen_control.json", control),
            write_private_text(temporary / "ranking_ablation_outcomes.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in ablated)),
            write_private_text(temporary / "ranking_case_contrasts.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in rank_cases)),
            write_private_text(temporary / "ranking_comparison.csv", render_csv(ranks)),
            write_private_text(temporary / "invariant_switch_cases.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in switches)),
            write_private_text(temporary / "invariant_switch_summary.csv", render_csv(invariants)),
            write_private_json(temporary / "pending_endpoint_pairs.json", {
                "schema_version": "tricompose-ranking-pending-endpoint-inventory-v1",
                "status": "metadata_only_not_execution_authorization", "modality_source": "fully_synthetic",
                "selection_used_biovil": False, "pairs": pending}),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary))]
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "original_selection_changed": False,
            "routing_or_stopping_changed": False, "clinical_accuracy_claim_allowed": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-run", "fixed-image-run", "control-config", "output-root", "run-id"):
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

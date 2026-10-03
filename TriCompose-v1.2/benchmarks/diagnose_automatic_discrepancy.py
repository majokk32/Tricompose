#!/usr/bin/env python3
"""Private CPU-only diagnostic of proxy versus alternate-readout disagreement."""
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
from tricompose_v12.automatic_discrepancy import SCHEMA, analyze
from merge_automatic_secondary import SCHEMA as ENDPOINT_SCHEMA, cached
from run_legacy_automatic_replay import load as load_legacy
from run_automatic_proxy_replay import render_csv
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    em, emp, ep = cached(args.endpoint_run, ENDPOINT_SCHEMA, "endpoint_outcomes.jsonl", 64 * 1024 * 1024)
    if em.get("original_selection_changed") is not False or em.get("routing_or_thresholds_updated") is not False:
        raise ValueError("unchanged frozen endpoint comparison required")
    replay_mp = require_inside(em["source_paths"]["replay_manifest"], PROTECTED_ROOT, must_exist=True)
    if sha256_file(replay_mp) != em["source_sha256"]["replay_manifest"]:
        raise ValueError("source replay changed")
    rm = read_json(replay_mp)
    sp = rm["source_paths"]
    config = require_inside(em["source_paths"]["frozen_policy"], PROTECTED_ROOT, must_exist=True)
    if sha256_file(config) != em["source_sha256"]["frozen_policy"]:
        raise ValueError("frozen policy changed")
    opts = argparse.Namespace(selection_run=Path(sp["selection_manifest"]).parent,
        edge_run=Path(sp["edge_manifest"]).parent, policy_config=config,
        development_selection_run=Path(sp["development_manifest"]).parent if "development_manifest" in sp else None)
    bank, policy, sources, _ = load_legacy(opts)
    if (sha256_file(sources["source_scores"]) != rm["source_sha256"]["source_scores"]
            or sha256_file(sources["source_edges"]) != rm["source_sha256"]["source_edges"]):
        raise ValueError("frozen legacy diagnostic evidence changed")
    rows = [json.loads(line) for line in ep.read_text().splitlines() if line]
    sources.update(endpoint_manifest=emp, endpoint_outcomes=ep, replay_manifest=replay_mp)
    return rows, bank, policy, sources


def observations(result, policy):
    maximum = max(policy["model_call_budgets"])
    rows = [r for r in result["action_slices"] if r["baseline"] == "fixed"
            and r["model_call_budget"] == maximum and r["ehr_evidence_subgroup"] == "all"]
    all_changes = next(r for r in rows if r["artifact_change"] == "all_changes")
    changes = {r["artifact_change"]: r["fixed_ehr_cases"] for r in rows if r["artifact_change"] != "all_changes"}
    a = all_changes["mean_baseline_negative_support_share"]
    b = all_changes["mean_targeted_negative_support_share"]
    return {"maximum_budget": maximum, "baseline": "fixed",
        "artifact_change_case_counts": changes,
        "negative_support_delta_exceeds_positive_support_delta":
            all_changes["sum_raw_negative_support_delta_all_selected_pairs"] > all_changes["sum_raw_positive_support_delta_all_selected_pairs"],
        "mean_negative_support_share_increases": None if a is None or b is None else b > a,
        "fixed_image_report_switch_proxy_biovil_disagreement_observed": any(
            r["artifact_change"] == "same_image_report_changed"
            and r["proxy_balance_gain_and_biovil_loss_cases"] > 0 for r in rows),
        "causal_or_clinical_error_claim_allowed": False}


def markdown(result, policy):
    def show(value): return "NA" if value is None else f"{value:.4f}"
    maximum = max(policy["model_call_budgets"])
    lines = ["# Proxy/BioViL discrepancy diagnostic / 分数分歧诊断", "",
        "Only cached synthetic finding states, endpoint scores, metadata and hashes were read. No patient source inputs, report bodies or image pixels were opened.",
        "本轮只有自动统计：不修改病例、prompt、评分权重、规则、阈值或已有赢家；不生成、不训练、不调用模型/API。", "",
        "## Targeted versus fixed at the maximum budget / 最大预算下与固定路径对比", "",
        "| Artifact change | EHR cases | Paired BioViL available | Mean proxy-balance delta | Mean BioViL delta | Positive-support delta sum | Negative-support delta sum |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["action_slices"]:
        if r["baseline"] == "fixed" and r["model_call_budget"] == maximum and r["ehr_evidence_subgroup"] == "all":
            lines.append(f"| {r['artifact_change']} | {r['fixed_ehr_cases']} | {r['biovil_paired_available_cases']} | {show(r['mean_source_proxy_balance_delta_on_same_biovil_pairs'])} | {show(r['mean_biovil_delta_on_available_pairs'])} | {r['sum_raw_positive_support_delta_all_selected_pairs']} | {r['sum_raw_negative_support_delta_all_selected_pairs']} |")
    lines += ["", "Proxy/BioViL means above use the same available-case pairs. Polarity sums retain ALL selected pairs; they are repeated edge comparisons, not counts of independent patients/findings.",
        "同图只换报告控制住了图像，但仍不是临床金标准。更换图像后，classifier 参考标签及分母也可能改变；不能把更少阳性解读成纠正了幻觉。", "",
        "## Cache-derived observations / 自动统计观察", "",
        "```json", json.dumps(result["observations"], sort_keys=True, indent=2), "```", "",
        "These are descriptive checks, not routing thresholds or clinical-error verdicts.", "",
        "## Common-image report-pair controls / 同图不同报告对照", "",
        "| Deciding source-key component | Observed pairs | Available case means | Mean BioViL delta: source-preferred − other | Source-preferred lower-BioViL pairs |",
        "|---|---:|---:|---:|---:|"]
    for r in result["same_image_controls"]:
        lines.append(f"| {r['key_deciding_component']} | {r['observed_same_image_report_pairs']} | {r['available_cases']} | {show(r['case_mean_then_cohort_mean_biovil_delta'])} | {r['preferred_has_lower_biovil_pair_count']} |")
    lines += ["", "The diagnostic applies the existing source key AFTER the experiment to scored pairs only. It does not execute a new selection or inspect unscored BioViL outcomes. Case means average within-case dependent pairs before averaging cases.",
        "同一 CXR 的多个报告/成对比较不独立；所选并集不是完整候选池，也不是随机样本，不能把这些对照当成总体泛化或医生评判。", "",
        "## Polarity and missingness / 阳性、阴性与缺失", "",
        "- `raw_support_positive` is explicit positive–positive agreement; `raw_support_negative` is explicit negative–negative agreement. Unknown/uncertain never contribute to either.",
        "- Negative agreement can be clinically appropriate. Its dominance is not itself a bug and does not justify forcing abnormal images, adding diseases/devices or removing normal EHRs.",
        "- Legacy global No-Finding adjustments remain separate source-proxy opposition counts; they do not flip raw unknown report labels.",
        "- Unrecorded EHR findings stay unknown. A report-positive fact with unknown EHR is not automatically a hallucination.",
        "- No-direct-EHR cases remain in the cohort. Pairwise report/image agreement cannot certify full three-modal consistency where EHR evidence is absent.",
        "- BioViL-T is another uncalibrated readout, not a probability or clinical truth. Disagreement does not prove that XRV/CheXbert or BioViL is the faulty scorer.", "",
        "## Files and next step / 文件与下一步", "",
        "`case_contrasts.jsonl`: fixed/static contrasts with decomposed evidence. `action_slices.csv`: all budgets and EHR groups. `same_image_report_pairs.jsonl` and `same_image_controls.csv`: common-image controls. `model_selection_frequencies.csv`: observed selections, not model clinical rankings.",
        "Do not tune on this cohort and relabel it held-out evaluation. Use these observations to preregister a separate controlled test of any revised scorer/policy, retaining missingness, fixed EHRs, frozen models and all invocation costs. Current rules/winners remain unchanged.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    rows, bank, policy, sources = load(args)
    sources.update(program=Path(__file__), diagnostic_contract=ROOT / "src/tricompose_v12/automatic_discrepancy.py",
        raw_cache_contract=ROOT / "src/tricompose_v12/legacy_replay_adapter.py",
        endpoint_contract=ROOT / "src/tricompose_v12/automatic_secondary.py",
        source_rank_contract=ROOT / "src/tricompose_v12/automatic_replay.py")
    before = {k: sha256_file(p) for k, p in sources.items()}
    result = analyze(rows, bank, policy)
    result["observations"] = observations(result, policy)
    result.update(fixed_ehr_cases=len(bank), original_replay_trials=len(rows),
        unique_selected_union_candidates=len({r["selected_candidate_id"] for r in rows if r["selected_candidate_id"] is not None}),
        runtime_seconds=round(time.monotonic()-started, 6))
    if before != {k: sha256_file(p) for k, p in sources.items()}:
        raise ValueError("immutable diagnostic source changed")
    summary = {k: v for k, v in result.items() if k not in {"contrasts", "same_image_pairs"}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "case_contrasts.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in result["contrasts"])),
            write_private_text(temporary / "action_slices.csv", render_csv(result["action_slices"])),
            write_private_text(temporary / "same_image_report_pairs.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in result["same_image_pairs"])),
            write_private_text(temporary / "model_selection_frequencies.csv", render_csv(result["model_selection_frequencies"])),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(result, policy))]
        if result["same_image_controls"]:
            files.append(write_private_text(temporary / "same_image_controls.csv", render_csv(result["same_image_controls"])))
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "rules_or_winners_changed": False,
            "independent_clinical_truth_available": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-run", "output-root", "run-id"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try: target, result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "completed_frozen_proxy_discrepancy_diagnostic",
        "runtime_seconds": result["runtime_seconds"], "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

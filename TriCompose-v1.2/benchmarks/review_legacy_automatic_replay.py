#!/usr/bin/env python3
"""Verify the completed synthetic metadata replay and write a private reader."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from run_legacy_automatic_replay import load, enriched_compare
from tricompose_v12.automatic_replay import replay_case
from tricompose_v12.legacy_replay_adapter import SCHEMA
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    root = require_inside(args.replay_run, PROTECTED_ROOT, must_exist=True)
    mp = root / "manifest.json"; manifest = read_json(mp)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("completed legacy replay required")
    for name, path in manifest["source_paths"].items():
        if sha256_file(path) != manifest["source_sha256"][name]:
            raise ValueError("replay source changed")
    for name, entry in manifest["artifacts"].items():
        path = require_inside(root / name, root, must_exist=True)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("replay output changed")
    sources = manifest["source_paths"]
    opts = argparse.Namespace(selection_run=Path(sources["selection_manifest"]).parent,
        edge_run=Path(sources["edge_manifest"]).parent, policy_config=sources["policy_config"],
        development_selection_run=Path(sources["development_manifest"]).parent)
    bank, policy, _, overlap = load(opts)
    summary = read_json(root / "summary.json")
    rows = [json.loads(x) for x in (root / "replay_outcomes.jsonl").read_text().splitlines() if x]
    expected_trials = len(bank) * len(policy["model_call_budgets"]) * (3 + len(policy["random_seeds"]))
    if len(rows) != expected_trials:
        raise ValueError("replay trial inventory differs")
    for row in rows:
        expected = replay_case(bank[row["case_id"]], policy, row["method"], row["model_call_budget"], random_seed=row["random_seed"] or 0)
        expected["input_ehr_assessment_scope"] = row["input_ehr_assessment_scope"]
        if row != expected or row["simulated_model_calls"] > row["model_call_budget"]:
            raise ValueError("exact replay or budget accounting differs")
    if summary["comparisons"] != enriched_compare(rows):
        raise ValueError("summary arithmetic differs")
    for row in summary["subgroup_comparisons"]:
        if row["guarded_edge_readouts"] != {"ehr_cxr": None, "ehr_report": None, "cxr_report": None} or row["biovil_available_trials"] != 0:
            raise ValueError("uncomputed alternate metrics cannot be fabricated")
        if row["ehr_evidence_subgroup"] == "no_direct_comparable_ehr_facts":
            for edge in ("ehr_cxr", "ehr_report"):
                values = row["source_edge_metrics_optimization_proxy"][edge]
                if values["support_recall"] is not None or values["opposition_rate"] is not None:
                    raise ValueError("missing EHR constraints cannot mean agreement")
    for path in [root, *root.iterdir()]:
        if (stat.S_IMODE(path.stat().st_mode) != (0o2770 if path.is_dir() else 0o660)
                or path.stat().st_gid not in {96293, 65534}):
            raise ValueError("private output permissions differ")
    # Original selections remain hash-bound, not replaced by replay selections.
    selection_manifest = read_json(sources["selection_manifest"])
    for name, entry in selection_manifest["artifacts"].items():
        if sha256_file(Path(sources["selection_manifest"]).parent / name) != entry["sha256"]:
            raise ValueError("original selection artifact changed")
    max_budget = max(policy["model_call_budgets"])
    at_max = {r["method"]: r for r in summary["comparisons"] if r["model_call_budget"] == max_budget}
    t, f, s = at_max["targeted_heuristic"], at_max["fixed"], at_max["static_rerank"]
    directions = {"maximum_budget": max_budget,
        "targeted_support_vs_fixed_nonworse": t["proxy_support_recall"] >= f["proxy_support_recall"],
        "targeted_opposition_vs_fixed_nonworse": t["proxy_opposition_rate"] <= f["proxy_opposition_rate"],
        "targeted_proxy_balance_vs_exhaustive_static_nonworse": t["proxy_balance_0_100"] >= s["proxy_balance_0_100"],
        "targeted_simulated_calls_vs_exhaustive_static_fewer": t["mean_simulated_model_calls"] < s["mean_simulated_model_calls"]}
    tiers = read_json(sources["edge_manifest"])["counts"]["conditioning_tiers"]
    review = {"schema_version": "tricompose-legacy-automatic-replay-review-v1",
        "status": "verified", "replay_manifest_sha256": sha256_file(mp),
        "exact_replay_verified": True, "summary_arithmetic_verified": True,
        "hashes_permissions_and_original_winners_verified": True,
        "unavailable_metrics_not_zero_verified": True,
        "source_prompt_conditioning_tiers": tiers,
        "strict_cached_ehr_finding_case_counts": summary["source_inventory"]["case_counts_by_ehr_evidence"],
        "development_cohort_overlap": overlap, "maximum_budget_direction_checks": directions,
        "clinical_accuracy_claim_allowed": False, "new_model_calls": 0}
    lines = ["# Automatic replay review / 自动策略实验核验", "",
        "80 fixed synthetic EHRs; 240 existing CXR candidates; 960 existing reports/triples; 3,200 replay trials.",
        "复用已生成候选，没有新增 EHR、CXR 或报告。3,200 次 replay 不是 3,200 位独立患者。", "",
        "## What passed / 核验结果", "",
        "所有逐步动作可精确重放；调用预算、汇总算术、来源/结果 SHA256、私有权限、原始 selected triples 均通过核验。", "",
        "## Result directions at the maximum budget / 最大预算下的方向", "",
        "```json", json.dumps(directions, sort_keys=True, indent=2), "```", "",
        "这些检查只描述最大预算，不代表所有预算均优于所有方法。完整曲线在原 run 的 `method_comparison.csv`，分组结果在 `subgroup_comparison.csv`。", "",
        "当前显示的是成本–质量取舍：定向搜索改善固定路径的代理指标并减少模拟调用，但没有超过穷举静态择优的代理 balance。不能将它包装成已经证明的新方法优势。", "",
        "## Conditioning is not comparability / 两种统计口径", "",
        "Source prompt-conditioning tiers:", "```json", json.dumps(tiers, indent=2, sort_keys=True), "```", "",
        "Actual directly comparable cached EHR-label cases:", "```json",
        json.dumps(review["strict_cached_ehr_finding_case_counts"], indent=2, sort_keys=True), "```", "",
        "旧 prompt tier 的 15/65 与严格标签缓存的 8/72 不是同一统计口径。后者只计明确 positive/negative、映射到共同 finding inventory 的直接状态。这里保留实际缓存，不为了凑数量添加病灶/device，也不重写 EHR。尚未做原始 EHR 的新抽取审计，不能声称已确定每例差异原因。", "",
        "缺少直接 EHR 约束的组，其 EHR–CXR/EHR–Report 支持率和矛盾率保持 NA。图文代理一致不能替代已验证的三模态一致。", "",
        "## Next / 下一步", "",
        "对所有方法所选候选的并集，用冻结 BioViL-T 做不参与路由的辅助端点评分。当前 request 已冻结，GPU 尚未提交；须先展示完整 Slurm 脚本并获得批准。", "",
        "BioViL-T 是另一种自动读数，不是独立临床真值。它不能单独证明错误定位正确。后续还需真实 prospective 调用、失败/retry 和 verifier 成本记录，而不是把模拟调用当作已节省 GPU 秒数。", ""]
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "verification.json", review),
                 write_private_text(temporary / "RESULTS_CN_EN.md", "\n".join(lines))]
        write_private_json(temporary / "manifest.json", {"schema_version": review["schema_version"],
            "source_manifest_path": str(mp), "source_manifest_sha256": sha256_file(mp),
            "program_sha256": sha256_file(Path(__file__)),
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files}})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-run", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); os.umask(0o007); start = time.monotonic()
    try: target = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "verified_private_review_completed", "runtime_seconds": round(time.monotonic()-start, 6),
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

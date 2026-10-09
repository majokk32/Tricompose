#!/usr/bin/env python3
"""Authenticate cached synthetic metadata; diagnose fixed-image repair limits.

CPU-only, no clinical bodies/pixels/weights/model execution. All numerical
readouts and symbolic bounds are exploratory proxies, not adjudicated truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path

import audit_current_image_reports as postflight
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json, write_private_text)
from tricompose_llm import fixed_image_reachability as symbolic

worker = postflight.worker
require = worker.require
BASE = worker.BASE
PLAN = BASE / "current_image_report_plans/current_report2_12851223_001"
PLAN_SHA = "fec15af05c44acf054de6ac8d60dca06a37a27af5b41db93749d1d9fbf734b25"
SOURCE = BASE / "current_image_report_runs/current_report2_12851790"
SOURCE_SHA = "ff6cbed07e712dbbfc89dc02ae76115414325348f1936cd24eb755f912b6e1d8"
AUDIT = BASE / "current_image_report_audits/current_report2_12851790_12851223_001"
AUDIT_SHA = "7038d7f0070cbb1b92e93b4a94b91d5711ff218eef342479746b52f6d3886d37"
VERSION = "tricompose-current-image-evidence-diagnostic-v1"


def sealed(root, pin, readers):
    r = worker.previous.Reader(root)
    readers.append(r)
    r.hash(r.root / "manifest.json", pin)
    manifest = r.json(r.root / "manifest.json")
    for name, artifact in manifest.get("artifacts", {}).items():
        r.hash(r.root / name, artifact["sha256"])
    return r, manifest


def sealed_plan(root, pin, readers):
    r, manifest = sealed(root, pin, readers)
    r.hash(r.root / "plan.json", manifest["plan_sha256"])
    return r, r.json(r.root / "plan.json")


def sealed_audit(root, pin, plan_sha, source_sha, readers):
    r, manifest = sealed(root, pin, readers)
    r.hash(r.root / "audit.json", manifest["audit_sha256"])
    audit = r.json(r.root / "audit.json")
    require(audit["plan_manifest_sha256"] == plan_sha and audit["source_manifest_sha256"] == source_sha
        and audit["exact_new_phase_replay_pass"] is True and audit["csv_bytes_exact_replay"] is True
        and audit["protected_permissions_pass"] is True and audit["new_model_calls"] == 0,
        "completed_hash_bound_metadata_postflight_required")
    worker.check_pins(audit["reader_pins"])
    return audit


def authenticate():
    worker.previous.gate.cpu_guard()
    readers = []
    pr, plan = sealed_plan(PLAN, PLAN_SHA, readers)
    require(plan["schema_version"] == worker.VERSION and plan["config"] == worker.CONFIG
        and plan["data_origin"] == "original_fully_synthetic_pool80" and len(plan["cases"]) == 2,
        "exact_completed_fully_synthetic_two_case_plan_required")
    for field in ("source_pins", "artifact_pins"): worker.check_pins(plan[field])
    sr, sm = sealed(SOURCE, SOURCE_SHA, readers)
    require(sm["schema_version"] == worker.VERSION and sm["plan_manifest_sha256"] == PLAN_SHA
        and sm["status"] == "completed_shared_report_comparison_unvalidated"
        and sm["clinical_acceptance"] is False, "exact_completed_shared_run_required")
    audit = sealed_audit(AUDIT, AUDIT_SHA, PLAN_SHA, SOURCE_SHA, readers)
    rows = sr.json(sr.root / "score_rows.json")["records"]
    pairs = sr.json(sr.root / "paired_report_comparison.json")["records"]
    execution = sr.json(sr.root / "execution_summary.json")
    require(len(rows) == 4 and len(pairs) == 2 and execution["historical_cost"] == plan["historical_cost"],
        "both_actual_completed_report_pairs_and_sunk_costs_required")

    old = worker.producer.authenticated
    op, original = sealed_plan(old.SOURCE_PLAN, old.SOURCE_PLAN_SHA, readers)
    old_audit = sealed_audit(old.AUDIT, old.AUDIT_SHA, old.SOURCE_PLAN_SHA, old.SOURCE_SHA, readers)
    oldr, oldm = sealed(old.SOURCE, old.SOURCE_SHA, readers)
    require(original["data_origin"] == "original_fully_synthetic_pool80" and len(original["cases"]) == 2
        and oldm["status"] == "completed_requested_report_comparison_unvalidated", "same_synthetic_earlier_phase_required")
    old_rows = oldr.json(oldr.root / "score_rows.json")["records"]
    old_pairs = oldr.json(oldr.root / "paired_report_comparison.json")["records"]
    old.freeze_controls(old_rows, old_pairs)

    cached = worker.producer.cached
    ir, im = sealed(cached.OBSERVATIONS, cached.OBSERVATIONS_SHA, readers)
    require(im["status"] == "completed_blind_image_observer_unvalidated" and im["clinical_acceptance"] is False,
        "cached_unvalidated_blind_observer_required")
    predictions = ir.json(ir.root / "predictions.json")
    require(predictions["frozen"] is True and predictions["image_only"] is True
        and predictions["model_received_ehr_reports_ids_or_scores"] is False, "blind_image_only_observer_required")
    cached.verify_calls(predictions["records"], ir.journal(ir.root / "calls.journal.jsonl"))
    qr, policy = sealed_plan(worker.POLICY_PLAN, worker.POLICY_PLAN_SHA, readers)
    require({c["case_id"] for c in policy["cases"]} == {c["case_id"] for c in plan["cases"]},
        "same_two_policy_cases_required")
    reports, diagnostics, replayed = [], [], []
    for ordinal, case in enumerate(plan["cases"]):
        cid, baseline = case["case_id"], case["baseline_row"]
        group = [r for r in rows if r["case_id"] == cid]
        older = [r for r in old_rows if r["case_id"] == cid]
        require(len(group) == len(older) == 2 and baseline in group and baseline in older,
            "identical_baseline_in_both_actual_report_phases_required")
        new = next(r for r in group if r != baseline)
        earlier = next(r for r in older if r != baseline)
        original_case = next(c for c in original["cases"] if c["case_id"] == cid)
        earlier_pair = next(p for p in old_pairs if p["case_id"] == cid)
        require(original_case["baseline_row"] == baseline
            and original_case["historical_ledger"] == case["historical_ledger"]
            and worker.previous.assess_pair(original_case, earlier, original["workers"]) == earlier_pair,
            "earlier_gate_and_exhausted_ledger_replay_required")
        pair = next(p for p in pairs if p["case_id"] == cid)
        root = sr.root / "cases" / cid
        snapshot = sr.json(root / "new_phase_snapshot.json")
        require(snapshot == {k: v for k, v in next(p for p in execution["records"] if p["case_id"] == cid).items()
            if k != "case_id"}, "persisted_new_phase_cost_snapshot_required")
        replayed.append(postflight.replay_case(case, plan["workers"],
            sr.journal(root / "new_phase.journal.jsonl"), snapshot, new, pair))
        context = worker.previous.gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
        all_reports = [baseline, earlier, new]
        require({r["report_model_id"] for r in all_reports} == {"maira2", "cxrmate_single", "llavarad"}
            and len({r["triple_candidate_id"] for r in all_reports}) == 3,
            "three_distinct_actually_observed_report_experts_required")
        for r in all_reports:
            worker.previous.gate.validate_observation(r, context)
            require(all(r[k] == baseline[k] for k in ("case_id", "ehr_sha256", "ehr_facts_sha256",
                "cxr_candidate_id", "cxr_sha256", "seed")), "fixed_ehr_current_image_lineage_required")
        # Reopen only numeric scorer metadata, never the generated report bodies.
        for reader, row in ((sr, new), (oldr, earlier)):
            path = reader.root / "cases" / cid / "operations/requested_chexbert_a1/scored/report_finding_labels.json"
            reader.hash(path, row["receipt"]["chexbert_labels_sha256"])
            labels = reader.json(path)
            require(labels["schema_version"] == "tricompose-report-finding-labels-v1.1"
                and labels["finding_order"] == list(symbolic.FINDINGS)
                and labels["counts"] == {"reports": 1, "model_calls": 1} and len(labels["records"]) == 1
                and labels["producer"]["frozen"] is True
                and labels["producer"]["checkpoint_sha256"] == row["receipt"]["chexbert_checkpoint_sha256"]
                and labels["calibration"]["unknown_is_negative"] is False
                and labels["records"][0]["report_sha256"] == row["report_sha256"]
                and labels["records"][0]["report_candidate_id"] == row["report_candidate_id"]
                and labels["records"][0]["finding_states"] == {
                    f["finding"]: f["chexbert"] for f in row["receipt"]["fact_states"]},
                "actual_frozen_report_labels_equal_receipt_required")
        lr = worker.previous.Reader(Path(case["image_labels_path"]).parent); readers.append(lr)
        image_hash = lr.hash(case["image_labels_path"], baseline["receipt"]["xrv_labels_sha256"])
        image_labels = lr.json(case["image_labels_path"])
        anchor = worker.previous.gate.anchor_from_record(case["anchor"])
        partial = worker.image_receipt(anchor, case["image_candidate"], image_labels,
            label_sha256=image_hash, thresholds_sha256=plan["workers"]["xrv"]["thresholds_sha256"],
            checkpoint_sha256=plan["workers"]["xrv"]["checkpoint_sha256"])
        require(partial == case["partial_receipt"] and image_labels["thresholds"] == plan["workers"]["xrv"]["thresholds"],
            "exact_unchanged_image_receipt_and_threshold_table_required")
        observed = [r for r in predictions["records"] if r["cxr_candidate_id"] == baseline["cxr_candidate_id"]]
        require(len(observed) == 1 and observed[0]["cxr_sha256"] == baseline["cxr_sha256"],
            "unique_same_image_observer_required")
        observation = observed[0]
        guard = cached.guard_for_image(observation, all_reports, candidate_id="c0000", evidence_id="e0000")
        packet = next(c["packet"] for c in policy["cases"] if c["case_id"] == cid)
        require(packet["image_guard"] == guard, "cached_image_guard_exact_replay_required")
        result = symbolic.analyze(baseline["receipt"]["fact_states"],
            image_labels["records"][0]["finding_probabilities"], image_labels["thresholds"],
            observation["states"], observer_scope=list(cached.observer.existing.image_interface.FINDINGS),
            observer_status=observation["contract_status"])
        require(result["bounds"]["fixed_ehr_cxr_supported_facts"] == partial["raw_edge_readouts"]["ehr_cxr"]["supported_facts"]
            and result["bounds"]["fixed_ehr_cxr_opposition_facts"] == partial["raw_edge_readouts"]["ehr_cxr"]["proxy_opposition_facts"],
            "symbolic_fixed_edge_matches_authenticated_receipt_required")
        readouts = [{"role": role, **symbolic.report_readout(row, baseline["receipt"]["fact_states"])}
            for role, row in zip(("baseline", "earlier_rejected_expert", "shared_llavarad"), all_reports, strict=True)]
        reports.append({"pair_ordinal": ordinal, "case_id": cid, "records": readouts})
        diagnostics.append({"pair_ordinal": ordinal, "case_id": cid, "seed": baseline["seed"],
            "cxr_sha256": baseline["cxr_sha256"], "ehr_sha256": baseline["ehr_sha256"],
            "ehr_facts_sha256": baseline["ehr_facts_sha256"], "image_labels_sha256": image_hash,
            "observer_response_sha256": observation["response_sha256"], "image_guard": guard, **result})
    require(replayed == audit["records"], "exact_completed_shared_postflight_records_reproduced")
    for reader in readers: reader.recheck()
    return plan, diagnostics, reports, readers, old_audit


def tables(diagnostics, reports):
    image_rows = []
    for case in diagnostics:
        b, g = case["bounds"], case["image_guard"]
        image_rows.append({"pair_ordinal": case["pair_ordinal"], "case_id": case["case_id"], "seed": case["seed"],
            "known_ehr_facts": b["known_ehr_facts"], "fixed_image_support": b["fixed_ehr_cxr_supported_facts"],
            "fixed_image_opposition": b["fixed_ehr_cxr_opposition_facts"],
            "fixed_image_missing": b["fixed_ehr_cxr_missing_facts"],
            "report_only_triple_support_ceiling": b["maximum_all_three_support_if_only_report_changes"],
            "ceiling_denominator": b["known_ehr_facts"], "guard_status": g["status"],
            "withhold_clinical_image_attribution": g["withhold_image_attribution"]})
    report_rows = []
    for case in reports:
        for row in case["records"]:
            item = {"pair_ordinal": case["pair_ordinal"], "case_id": case["case_id"],
                "role": row["role"], "report_model": row["report_model"],
                "all_three_supported_facts": row["all_three_supported_facts"], "known_ehr_facts": row["known_ehr_facts"],
                "section_contract_pass": row["section_contract_pass"],
                "unsupported_temporal_language": row["unsupported_temporal_comparison_language"]}
            for edge, counts in row["raw_edge_readouts"].items():
                for key in ("known_reference_facts", "comparable_facts", "supported_facts", "supported_positive",
                        "supported_negative", "proxy_opposition_facts", "missing_comparisons"):
                    item[edge + "_" + key] = counts[key]
            report_rows.append(item)
    return image_rows, report_rows


def csv_text(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), "nonempty_homogeneous_numeric_table_required")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def report_text(diagnostics, image_rows):
    text = ["# 固定图像的报告修复可达性 / Fixed-image report repair reachability", "",
        "这是已观察结果后的代理标签诊断，不是临床正确性评估；未读取报告正文、EHR 正文或图像像素。",
        "Post-hoc frozen-label diagnosis, not clinical adjudication. No clinical bodies or pixels parsed.", "",
        "| Pair ordinal | Known EHR | Fixed image support | Fixed image opposition | Report-only triple-support ceiling | Image evidence status |",
        "|---|---:|---:|---:|---|---|"]
    for r in image_rows:
        text.append(f"| {r['pair_ordinal']} | {r['known_ehr_facts']} | {r['fixed_image_support']} | "
            f"{r['fixed_image_opposition']} | {r['report_only_triple_support_ceiling']}/{r['ceiling_denominator']} | {r['guard_status']} |")
    text += ["", "## 怎么理解 / Interpretation", "",
        "EHR 和图像代理标签若明确相反，报告写阳性或阴性都会反对其中一边。unknown/uncertain 只是减少可比事实，",
        "不会消除固定 EHR–CXR 冲突。因此不能用不停换报告来追求这两例的三边完全一致。",
        "This does NOT rule out improvements in report structure or other findings, or diagnose an actual image fault.", "",
        "图像观察器缺失信息或与 XRV 分歧时，临床责任仍未定位；报告多数票不是图像真值。",
        "The image observer shares the planner checkpoint and is not independent clinical adjudication.", "",
        "14×4 项是逐 finding 的符号状态表，不是新生成的 56 份报告，也不是穷举模型输出。",
        "The bound ignores decoder attainability and quality constraints, so it is an upper bound in label space only.", "",
        "## 后续动作 / Next action", "",
        "暂缓为消除这些固定冲突而盲目换报告。先取得能区分阳性/阴性且经过独立参考验证的图像证据；",
        "若证据仍不充分，保留 unresolved/abstain，不改分数、不补造阴性、不改 EHR。",
        "No action is installed by this diagnostic. Any new inference requires a separately reviewed script and approval.", "",
        "已有 BioViL-T 正负文本实验和 SigLIP 的模板敏感性诊断不足以充当最终判定。",
        "Do not replace finding truth with global image-text cosine or consensus; see the frozen evaluator scorecard.", "",
        "零新模型调用；旧候选、阈值、gate、winner 和已付调用成本保持不变。",
        "Zero new model calls. No clinical repair success, LLM advantage or measured GPU savings is established.", ""]
    return "\n".join(text)


def build_summary(plan, diagnostics, report_rows, old_audit):
    require(old_audit["completed_report_label_pairs"] == 2 and old_audit["charged_new_worker_attempts"] == 4,
        "earlier_completed_pair_and_charged_cost_required")
    return {"schema_version": VERSION, "status": "completed_cpu_only_proxy_reachability_diagnostic",
        "cases": len(diagnostics), "existing_images": len(diagnostics), "actual_observed_reports": len(report_rows),
        "fixed_proxy_opposition_cases": sum(r["bounds"]["fixed_ehr_cxr_opposition_facts"] > 0 for r in diagnostics),
        "zero_report_only_support_ceiling_cases": sum(r["bounds"]["known_ehr_facts"] > 0
            and r["bounds"]["maximum_all_three_support_if_only_report_changes"] == 0 for r in diagnostics),
        "guard_status_counts": dict(sorted(Counter(r["image_guard"]["status"] for r in diagnostics).items())),
        "symbolic_finding_state_slots": sum(r["bounds"]["symbolic_state_slots"] for r in diagnostics),
        "new_model_calls": 0, "new_gpu_jobs": 0, "external_api_calls": 0, "generated_symbolic_reports": 0,
        "historical_cost_retained": plan["historical_cost"],
        "old_case_worker_attempts": [c["historical_ledger"]["charged_model_attempts"] for c in plan["cases"]],
        "earlier_report_phase_charged_worker_attempts": old_audit["charged_new_worker_attempts"],
        "shared_report_phase_completed_worker_attempts": 4, "clinical_accuracy": None,
        "clinical_acceptance": False, "clinical_fault_location": None, "clinical_repair_success": False,
        "llm_superiority_demonstrated": False, "measured_saved_model_calls": None, "installed_as_policy": False,
        "original_winner_changed": False, "thresholds_changed": False, "training_performed": False,
        "source_bodies_parsed": False, "image_pixels_decoded": False,
        "post_hoc_development_not_unseen_validation": True}


def build(args):
    plan, diagnostics, reports, readers, old_audit = authenticate()
    image_rows, report_rows = tables(diagnostics, reports)
    summary = build_summary(plan, diagnostics, report_rows, old_audit)
    own = {str(p.resolve()): sha256_file(p) for p in (Path(__file__), Path(symbolic.__file__),
        Path(__file__).resolve().parents[1] / "tests/test_fixed_image_reachability.py",
        Path(__file__).resolve().parents[1] / "tests/test_fixed_image_evidence_diagnostic.py",
        Path(__file__).resolve().parents[2] / "docs/current_image_reachability_protocol.md")}
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(tmp / "image_fact_diagnostics.json", {"schema_version": VERSION, "records": diagnostics}),
            write_private_json(tmp / "actual_report_readouts.json", {"schema_version": VERSION, "records": reports}),
            write_private_json(tmp / "summary.json", summary),
            write_private_text(tmp / "image_evidence_table.csv", csv_text(image_rows)),
            write_private_text(tmp / "report_observations.csv", csv_text(report_rows)),
            write_private_text(tmp / "RESULTS_CN_EN.md", report_text(diagnostics, image_rows))]
        pins = {}
        for reader in readers: reader.recheck(); pins.update(reader.pins)
        worker.check_pins(own)
        for field in ("source_pins", "artifact_pins"): worker.check_pins(plan[field])
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "source_plan_manifest_sha256": PLAN_SHA, "source_run_manifest_sha256": SOURCE_SHA,
            "source_audit_manifest_sha256": AUDIT_SHA, "source_pins": {**plan["source_pins"], **own},
            "reader_pins": pins, "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "installed_as_policy": False, "original_winner_changed": False,
            "clinical_acceptance": False, "clinical_fault_location": None})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target, summary = build(args)
        print(json.dumps({"stage": VERSION, "status": summary["status"], "cases": summary["cases"],
            "actual_observed_reports": summary["actual_observed_reports"], "new_model_calls": 0,
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())

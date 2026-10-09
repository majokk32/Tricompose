#!/usr/bin/env python3
"""Execute the two deduplicated same-image report requests from Qwen and rule.

CPU prepare authenticates numeric metadata/byte hashes only. Separately approved
GPU execution uses unchanged frozen LLaVA-Rad and CheXbert workers, with the
existing images/XRV receipts. A shared request has ONE physical execution, not
two different policy outcomes. No LLM superiority, savings or clinical truth.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import audit_current_image_policy as policy_audit
import run_current_image_policy as producer
import run_guarded_requested_reports as previous
from contracts import (PROTECTED_ROOT, RUN_ID_PATTERN, commit_atomic_run, discard_atomic_run,
    load_cxr_candidates, new_atomic_run, private_directory, read_json, require_inside,
    sha256_file, write_private_json, write_private_text)
from tricompose_v11.report_contracts import prepare_report_request_run
from tricompose_v12.execution_ledger import CallRequest, CallResult
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import validate_report_binding, score_csv
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.live_workers import LocalFrozenBackend, check_pins, single_generated
from tricompose_v12.runtime_dispatch import ProtectedJournal
from run_bounded_regeneration import candidate_row

VERSION = "tricompose-current-image-shared-report-execution-v1"
BASE = producer.BASE
POLICY_PLAN = BASE / "current_image_policy_plans/current_policy2_12827441_001"
POLICY_PLAN_SHA = "6bdfadedbd14f18548dba9fd622d0e297a822e17692350dbc5f4a3eb6d180786"
POLICY_RUN = BASE / "current_image_policy_runs/current_policy2_12851312"
POLICY_RUN_SHA = "470e4a483ae43c62ef1b2620cbeedc2698b1508ac51412893d8bb13a25f22bed"
POLICY_AUDIT = BASE / "current_image_policy_audits/current_policy2_12851312_12851223_001"
POLICY_AUDIT_SHA = "adfde92fec930f1e9467922e97dce07cd46d554977dcadfb07a409b954afdfd5"
CONFIG = {"cases": 2, "new_worker_budget_per_case": 2, "report_timeout_seconds": 120,
    "chexbert_timeout_seconds": 45, "retries": 0, "minimum_gpu_vram_gib": 24,
    "new_planner_calls": 0, "new_image_observer_calls": 0, "new_xrv_calls": 0,
    "new_cxr_calls": 0, "original_selection_change_allowed": False,
    "shared_physical_execution_is_distinct_policy_outcomes": False}
require = producer.require


def shared_requests(plan, results, pending):
    """Exact two-policy equality; disagreement needs a different reviewed plan."""
    require(len(plan["cases"]) == len(results) == 2 and len(pending["records"]) == 4
        and pending["automatic_submission_allowed"] is False
        and pending["requires_complete_script_and_explicit_approval"] is True,
        "exact_two_case_four_provenance_rows_required")
    requests = []
    for case, result in zip(plan["cases"], results, strict=True):
        producer.validate_case(case)
        require(result["case_id"] == case["case_id"] and result["decision"] is not None
            and result["source"] == "new_frozen_qwen_generate_call"
            and result["policy_input_sha256"] == case["policy_input_sha256"]
            and result["rule_control"] == case["rule_control"]
            and result["clinical_acceptance"] is False and result["original_winner_changed"] is False,
            "completed_same_observation_policy_pair_required")
        replay = producer.dispatch_record(result["decision"], case["packet"], sink=lambda event: None)
        require(replay == result["dispatch"]
            and replay["status"] == "deferred_separately_approved_backend_required"
            and result["same_effective_action_and_target_as_rule"] is True,
            "actual_matching_deferred_requests_required")
        raw = replay["effective_decision"]
        control = case["rule_control"]["dispatch"]["effective_decision"]
        require(all(raw[k] == control[k] for k in ("action", "target_id")), "matching_effective_action_and_target_required")
        request = case["tool_catalog"][raw["target_id"]]
        require(raw["action"] == "regenerate_report" and request["model_id"] == "llavarad",
            "reviewed_llavarad_request_only")
        rows = [r for r in pending["records"] if r["case_id"] == case["case_id"]]
        require(len(rows) == 2 and {r["policy"] for r in rows} == {"qwen", "rule"}
            and all(r == {"case_id": case["case_id"], "policy": r["policy"], "request": request,
                "clinical_acceptance": False} for r in rows), "two_provenance_rows_one_physical_request_required")
        requests.append({"case_id": case["case_id"], "request": request, "qwen_decision": result["decision"],
            "rule_decision": case["rule_control"]["decision"], "policies": ["qwen", "rule"],
            "new_physical_report_attempts": 1, "new_physical_chexbert_attempts": 1,
            "qwen_policy_calls_already_incurred": 1, "rule_model_calls_already_incurred": 0})
    require(len({r["case_id"] for r in requests}) == 2, "two_distinct_original_opaque_cases_required")
    return requests


def bind_case(original, planned, request):
    """Reuse exact cached image dependency; do not modify its old call ledger."""
    case = deepcopy(original)
    base = case["baseline_row"]
    require(request["case_id"] == planned["case_id"] == case["case_id"]
        and request["request"]["input_image_sha256"] == planned["trial_image_sha256"] == base["cxr_sha256"]
        and request["request"]["seed"] == base["seed"] == 1
        and planned["retained_branch_candidate_id"] == base["triple_candidate_id"]
        and all(planned[k] == base[k] for k in ("ehr_sha256", "ehr_facts_sha256"))
        and planned["retained_original_candidate_id"] == case["retained_original_candidate_id"],
        "same_retained_branch_and_fixed_ehr_image_seed_required")
    case.update(requested_model="llavarad", sealed_policy_decision=request["qwen_decision"],
        shared_policy_request=request, current_image_evidence_cited="e0000" in request["qwen_decision"]["evidence_ids"])
    return case


def validate_plan(plan):
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG and len(plan["cases"]) == 2
        and plan["data_origin"] == "original_fully_synthetic_pool80" and plan["new_model_calls"] == 0
        and plan["policy_run_manifest_sha256"] == POLICY_RUN_SHA
        and plan["shared_physical_report_attempts"] == plan["shared_physical_chexbert_attempts"] == 2
        and plan["distinct_policy_outcomes_claimed"] is False, "exact_shared_two_report_execution_scope_required")
    for case in plan["cases"]:
        require(case["requested_model"] == "llavarad"
            and _digest(case["historical_ledger"]) == case["historical_ledger_sha256"]
            and case["historical_ledger"]["call_budget"] == case["historical_ledger"]["charged_model_attempts"] == 4
            and case["original_selection_change_allowed"] is False, "old_exhausted_cost_contract_unchanged")
        context = previous.gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
        previous.gate.validate_observation(case["baseline_row"], context)
        anchor = previous.gate.anchor_from_record(case["anchor"])
        previous.gate.validate_ledger(case["historical_ledger"], anchor)
        previous.gate.completed_operation(case["baseline_row"], case["historical_ledger"])
        images = load_cxr_candidates([case["baseline_triple"]["cxr_run"]])
        require(len(images) == 1 and next(iter(images.values())) == case["image_candidate"], "same_cached_image_required")
        labels = read_json(case["image_labels_path"])
        label_hash = sha256_file(case["image_labels_path"])
        require(label_hash == case["baseline_row"]["receipt"]["xrv_labels_sha256"], "cached_image_label_hash_required")
        receipt = image_receipt(anchor, case["image_candidate"], labels, label_sha256=label_hash,
            thresholds_sha256=plan["workers"]["xrv"]["thresholds_sha256"],
            checkpoint_sha256=plan["workers"]["xrv"]["checkpoint_sha256"])
        require(receipt == case["partial_receipt"], "unchanged_cached_xrv_receipt_required")
    for field in ("source_pins", "artifact_pins"): check_pins(plan[field])
    for model in ("llavarad", "chexbert"):
        spec = plan["workers"][model]
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and os.access(spec["python"], os.X_OK), "existing_preflighted_worker_required")
        check_pins(spec["asset_pins"])


def prepare(args):
    previous.gate.cpu_guard()
    readers = [previous.Reader(p) for p in (POLICY_PLAN, POLICY_RUN, POLICY_AUDIT,
        producer.authenticated.SOURCE_PLAN)]
    pp, pr, ar, original_reader = readers
    for r, pin in zip(readers, (POLICY_PLAN_SHA, POLICY_RUN_SHA, POLICY_AUDIT_SHA,
            producer.authenticated.SOURCE_PLAN_SHA), strict=True): r.hash(r.root / "manifest.json", pin)
    pm = pp.json(pp.root / "manifest.json"); pp.hash(pp.root / "plan.json", pm["plan_sha256"])
    policy_plan = pp.json(pp.root / "plan.json")
    rm = pr.json(pr.root / "manifest.json")
    require(rm["status"] == "completed_current_image_decision_control_unvalidated"
        and rm["plan_manifest_sha256"] == POLICY_PLAN_SHA, "completed_reviewed_current_image_policy_required")
    for name, pin in rm["artifacts"].items(): pr.hash(pr.root / name, pin["sha256"])
    results = pr.json(pr.root / "policy_results.json")["records"]
    pending = pr.json(pr.root / "pending_requests.json")
    journal = pr.journal(pr.root / "policy_dispatch.journal.jsonl")
    audit = pr.json(pr.root / "planner_audit.json"); summary = pr.json(pr.root / "summary.json")
    replay = policy_audit.replay(policy_plan, results, pending, audit, journal, summary)
    am = ar.json(ar.root / "manifest.json"); ar.hash(ar.root / "audit.json", am["audit_sha256"])
    postflight = ar.json(ar.root / "audit.json")
    require(postflight["source_manifest_sha256"] == POLICY_RUN_SHA
        and postflight["plan_manifest_sha256"] == POLICY_PLAN_SHA
        and all(postflight[k] == v for k, v in replay.items()), "completed_exact_decision_postflight_required")
    om = original_reader.json(original_reader.root / "manifest.json")
    original_reader.hash(original_reader.root / "plan.json", om["plan_sha256"])
    original = original_reader.json(original_reader.root / "plan.json")
    requests = shared_requests(policy_plan, results, pending)
    cases = [bind_case(next(c for c in original["cases"] if c["case_id"] == r["case_id"]),
        next(c for c in policy_plan["cases"] if c["case_id"] == r["case_id"]), r) for r in requests]
    pins = {**policy_plan["source_pins"], **{str(p.resolve()): sha256_file(p) for p in (
        Path(__file__), Path(policy_audit.__file__),
        Path(__file__).resolve().parents[1] / "tests/test_current_image_requested_reports.py")}}
    artifacts = {**policy_plan["artifact_pins"], **postflight["reader_pins"]}
    for r in readers: r.recheck(); artifacts.update(r.pins)
    plan = {"schema_version": VERSION, "config": CONFIG, "cases": cases,
        "workers": {"xrv": original["workers"]["xrv"], "chexbert": original["workers"]["chexbert"],
            "llavarad": policy_plan["report_catalog"]["llavarad"]}, "source_pins": pins, "artifact_pins": artifacts,
        "policy_run_manifest_sha256": POLICY_RUN_SHA, "data_origin": "original_fully_synthetic_pool80",
        "new_model_calls": 0, "shared_physical_report_attempts": 2, "shared_physical_chexbert_attempts": 2,
        "distinct_policy_outcomes_claimed": False, "historical_cost": {**policy_plan["historical_cost"],
            "current_image_policy_calls": 2, "current_image_policy_load_attempts": 1, "rule_policy_model_calls": 0}}
    validate_plan(plan)
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(tmp / "plan.json", plan)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(path),
            "maximum_new_report_attempts": 2, "maximum_new_chexbert_attempts": 2,
            "new_model_calls": 0, "clinical_acceptance": False, "distinct_policy_outcomes_claimed": False})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target


def run_case(case, plan, root):
    producer.gpu_guard()
    anchor = previous.gate.anchor_from_record(case["anchor"])
    cr = root / "cases" / anchor.case_id
    private_directory(cr); private_directory(cr / "operations")
    write_private_json(cr / "source_ledger.json", case["historical_ledger"])
    image = case["image_candidate"]; workers = plan["workers"]
    pack = prepare_report_request_run(cxr_runs=[case["baseline_triple"]["cxr_run"]], output_root=cr / "operations",
        run_id="requested_report_pack", model_ids=[case["requested_model"]])
    request_run = Path(pack["run_directory"])
    state = {}; triple = None
    with ProtectedJournal(cr / "new_phase.journal.jsonl") as journal:
        journal.append({"event": "shared_dependencies_authenticated", "source_policy_manifest_sha256": POLICY_RUN_SHA,
            "source_ledger_sha256": case["historical_ledger_sha256"],
            "cached_partial_receipt_id": case["partial_receipt"]["receipt_id"], "new_budget": 2,
            "policies": ["qwen", "rule"], "physical_report_attempts_reserved_by_plan": 1,
            "distinct_policy_outcomes_claimed": False, "clinical_acceptance": False})
        calls = previous.RequestedReportCalls(case, sink=journal.append)

        def invoke(kind, model, parent, validator, report_hash=None, **inputs):
            spec = workers[model]
            operation = "requested_report" if kind == "report_generator" else "requested_chexbert"
            op_root = cr / "operations" / (operation + "_a1")
            private_directory(op_root)
            request = CallRequest(operation, anchor.case_id, anchor.sha256, kind, model, _digest(spec),
                image["seed"], parent, image["artifact"]["sha256"], report_hash)
            timeout = CONFIG["report_timeout_seconds"] if kind == "report_generator" else CONFIG["chexbert_timeout_seconds"]
            return calls.invoke(request, factory=lambda: LocalFrozenBackend(spec,
                operation_root=op_root, timeout=timeout, **inputs), validator=validator)

        def report_validator(payload, operation):
            rr, report = single_generated(payload, workers["llavarad"], request_run,
                cxr_candidates={image["candidate_id"]: image})
            validate_report_binding(report, image, request_run, anchor)
            state.update(report_run=rr, report=report)
            return CallResult(report["artifact"]["sha256"])

        result = invoke("report_generator", "llavarad", case["cached_xrv_operation_id"],
            report_validator, request_run=request_run)
        if result is not None:
            def label_validator(payload, operation):
                nonlocal triple
                require(payload["worker_audit_sha256"] == operation.frozen_model_audit_sha256,
                    "unchanged_frozen_chexbert_required")
                op_root = require_inside(payload["output_root"], PROTECTED_ROOT, must_exist=True)
                lp = op_root / "scored/report_finding_labels.json"
                receipt = completed_receipt(anchor, case["partial_receipt"], image, read_json(case["image_labels_path"]),
                    state["report"], read_json(lp), image_labels_sha256=sha256_file(case["image_labels_path"]),
                    report_labels_sha256=sha256_file(lp), thresholds_sha256=workers["xrv"]["thresholds_sha256"],
                    xrv_checkpoint_sha256=workers["xrv"]["checkpoint_sha256"],
                    chexbert_checkpoint_sha256=workers["chexbert"]["checkpoint_sha256"])
                rp = write_private_json(op_root / "completed_receipt.json", receipt)
                triple = {**case["baseline_triple"], "report_run": str(state["report_run"]),
                    "report_path": state["report"]["artifact"]["path"], "report_sha256": state["report"]["artifact"]["sha256"],
                    "report_model_id": "llavarad", "receipt_path": str(rp), "receipt_sha256": sha256_file(rp),
                    "receipt_id": receipt["receipt_id"], "raw_edge_readouts": receipt["raw_edge_readouts"],
                    "verification_status": receipt["verification_status"], "selected_as_best": False}
                return CallResult(sha256_file(lp), receipt["receipt_id"])
            verified = invoke("chexbert", "chexbert", "requested_report", label_validator,
                report_hash=result.output_artifact_sha256, cxr_run=case["baseline_triple"]["cxr_run"], report_run=state["report_run"])
            if verified is None: triple = None
        snapshot = calls.snapshot()
        journal.append({"event": "new_phase_sealed", **snapshot, "clinical_acceptance": False})
    write_private_json(cr / "new_phase_snapshot.json", snapshot)
    return triple, snapshot


def run(args):
    producer.gpu_guard()
    planroot = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    require(sha256_file(planroot / "manifest.json") == args.plan_manifest_sha256, "reviewed_shared_report_plan_changed")
    pm = read_json(planroot / "manifest.json")
    require(pm["schema_version"] == VERSION and pm["status"] == "prepared_cpu_only_gpu_not_submitted"
        and sha256_file(planroot / "plan.json") == pm["plan_sha256"], "sealed_shared_execution_plan_required")
    plan = read_json(planroot / "plan.json"); validate_plan(plan)
    import torch
    require(torch.cuda.get_device_properties(0).total_memory >= CONFIG["minimum_gpu_vram_gib"] * 1024**3,
        "registered_llavarad_memory_class_required")
    require(RUN_ID_PATTERN.fullmatch(args.run_id), "opaque_new_run_id_required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = output / args.run_id
    private_directory(root); private_directory(root / "cases")
    write_private_json(root / "start_manifest.json", {"schema_version": VERSION, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "automatic_resume": False})
    triples = []; rows = []; comparisons = []; snapshots = []; started = time.monotonic()
    try:
        for case in plan["cases"]:
            triple, snapshot = run_case(case, plan, root)
            snapshots.append({"case_id": case["case_id"], **snapshot})
            rows.append(case["baseline_row"])
            if triple is not None:
                row = candidate_row(triple)
                comparisons.append(previous.assess_pair(case, row, plan["workers"]))
                rows.append(row); triples.append(triple)
            else:
                comparisons.append({"case_id": case["case_id"], "status": "failed_charged_no_retry",
                    "branch_selected_candidate_id": case["baseline_row"]["triple_candidate_id"],
                    "original_winner_changed": False, "clinical_acceptance": False})
        validate_plan(plan)
        summary = {"schema_version": VERSION, "status": "completed_shared_report_comparison_unvalidated",
            "requested_reports": 2, "completed_report_label_pairs": len(triples),
            "charged_new_worker_attempts": sum(s["charged_new_worker_attempts"] for s in snapshots),
            "completed_new_worker_operations": sum(s["completed_new_worker_operations"] for s in snapshots),
            "failed_new_worker_attempts": sum(s["failed_new_worker_attempts"] for s in snapshots),
            "proxy_preserving_branch_changes": sum(bool(p.get("comparison", {}).get("exploratory_gate_pass")) for p in comparisons),
            "new_planner_calls": 0, "new_cxr_calls": 0, "new_xrv_calls": 0, "new_image_observer_calls": 0,
            "runtime_seconds": round(time.monotonic() - started, 3), "training_performed": False,
            "original_winner_changed": False, "clinical_acceptance": False, "clinical_repair_success": False,
            "distinct_policy_outcomes_claimed": False, "llm_superiority_demonstrated": False,
            "clinical_accuracy": None, "measured_saved_model_calls": None}
        artifacts = [write_private_json(root / "score_rows.json", {"schema_version": VERSION, "records": rows}),
            write_private_text(root / "score_table.csv", score_csv([{**r, "verification_status": r["receipt"]["verification_status"]} for r in rows])),
            write_private_json(root / "paired_report_comparison.json", {"schema_version": VERSION, "records": comparisons}),
            write_private_json(root / "completed_triplets.json", {"schema_version": VERSION, "new_records": triples}),
            write_private_json(root / "execution_summary.json", {"schema_version": VERSION, "records": snapshots,
                "historical_cost": plan["historical_cost"], "historical_cost_not_erased": True}),
            write_private_json(root / "summary.json", summary)]
        artifacts += list(root.glob("cases/*/new_phase.journal.jsonl")) + list(root.glob("cases/*/new_phase_snapshot.json"))
        write_private_json(root / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256, "clinical_acceptance": False,
            "artifacts": {str(p.relative_to(root)): {"sha256": sha256_file(p)} for p in artifacts}})
    except BaseException:
        write_private_json(root / "failure.json", {"schema_version": VERSION, "status": "failed_prefix_retained",
            "automatic_resume": False, "charged_attempts_remain_in_case_journals": True,
            "completed_prefix_snapshots": snapshots, "clinical_acceptance": False})
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep, execution = commands.add_parser("prepare"), commands.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())

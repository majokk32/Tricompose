#!/usr/bin/env python3
"""Execute the two sealed LLM report requests, with an independent new budget.

Prepare authenticates synthetic metadata and bytes only. GPU run reuses the
existing report/CheXbert adapters, fixed images and cached XRV receipts. No new
planner/image observer/CXR generation, retraining, threshold change or old-winner
replacement. Old exhausted ledgers are not rewritten or given a larger budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import run_guarded_fresh_policy as producer
from contracts import (PROTECTED_ROOT, RUN_ID_PATTERN, commit_atomic_run, discard_atomic_run,
    load_cxr_candidates, new_atomic_run, private_directory, read_json, require_inside,
    sha256_file, write_private_json, write_private_text)
from tricompose_llm.contracts import validate_decision
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher
from tricompose_v11.report_contracts import prepare_report_request_run
from tricompose_v12.execution_ledger import CallRequest, CallResult
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import validate_report_binding, score_csv
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.live_workers import LocalFrozenBackend, check_pins, single_generated
from tricompose_v12.report_expert_control import compare
from tricompose_v12.runtime_dispatch import ProtectedJournal
from run_bounded_regeneration import candidate_row

VERSION = "tricompose-guarded-requested-report-execution-v1"
BASE = producer.BASE
POLICY_PLAN = BASE / "guarded_fresh_policy_plans/fresh_policy2_12827441_001"
POLICY_PLAN_SHA = "f575d3be72e16242b50e711d187dc9be9db59f929d380bb65e2dd771731a46ae"
POLICY_RUN = BASE / "guarded_fresh_policy_runs/fresh_policy2_12849550"
POLICY_RUN_SHA = "189c9cf86afdbb16e86287b2b51390a2ad94f52eacc57192a6b20da196de45bb"
POLICY_AUDIT = BASE / "guarded_fresh_policy_audits/fresh_policy2_12849550_12827441_001"
POLICY_AUDIT_SHA = "6337ab9a18629ce9725d699ef98136fb8dafd060cb87bb07bff1f4a7ceb04e70"
CONFIG = {"cases": 2, "new_worker_budget_per_case": 2, "report_timeout_seconds": 120,
    "chexbert_timeout_seconds": 45, "retries": 0, "minimum_gpu_vram_gib": 40,
    "new_planner_calls": 0, "new_image_observer_calls": 0, "new_xrv_calls": 0,
    "new_cxr_calls": 0, "original_selection_change_allowed": False}
require = producer.require
source = producer.previous.cached.observer.source
gate = source.previous.gate
Reader = source.previous.postflight.MetadataReader


class RequestedReportCalls:
    """Two new serial reservations; no fictitious replay of old parent calls.

    The report request names an authenticated cached-XRV parent. Its historical
    dependency and cost are validated separately by prepare/run. This is NOT a
    modified BoundedCallLedger and does not change its exhausted budget/hash chain.
    The durable sink must complete before even constructing a model backend.
    """
    def __init__(self, case, *, sink):
        require(callable(sink), "durable_new_phase_sink_required")
        self.case, self.sink = case, sink
        self.charged = 0
        self.completed = 0
        self.failed = 0
        self.pending = False
        self.closed = False
        self.report_hash = None

    def invoke(self, request, *, factory, validator):
        expected_kind = "report_generator" if self.charged == 0 else "chexbert"
        expected_model = self.case["requested_model"] if self.charged == 0 else "chexbert"
        anchor = gate.anchor_from_record(self.case["anchor"])
        require(not self.closed and not self.pending and self.charged < 2
            and request.kind == expected_kind and request.model_id == expected_model
            and request.operation_id == ("requested_report" if self.charged == 0 else "requested_chexbert")
            and request.case_id == anchor.case_id and request.ehr_anchor_sha256 == anchor.sha256
            and request.input_image_sha256 == self.case["baseline_row"]["cxr_sha256"]
            and request.seed == self.case["baseline_row"]["seed"]
            and request.parent_operation_id == (self.case["cached_xrv_operation_id"]
                if self.charged == 0 else "requested_report")
            and request.input_report_sha256 == (None if self.charged == 0 else self.report_hash),
            "exact_sealed_report_then_chexbert_only")
        require(callable(factory) and callable(validator), "lazy_registered_worker_required")
        ordinal = self.charged
        self.charged += 1
        self.pending = True
        self.sink({"event": "new_worker_reserved", "ordinal": ordinal,
            "request": request.record(), "charged_worker_attempts": 1,
            "dependency_origin": "authenticated_cached_xrv" if ordinal == 0 else "new_validated_report",
            "historical_ledger_changed": False})
        started = time.monotonic()
        stage = "backend"
        try:
            backend = factory()
            require(backend.frozen is True and backend.audit_sha256 == request.frozen_model_audit_sha256
                and backend.execution_mode == "approved_slurm_backend", "frozen_registered_backend_required")
            payload = backend.invoke(request)
            stage = "validator"
            result = validator(payload, request)
            require(isinstance(result, CallResult)
                and (result.verification_receipt_id is not None) == (ordinal == 1),
                "phase_correct_validated_result_required")
        except Exception as error:
            self.closed = True
            self.failed += 1
            self.sink({"event": "new_worker_failed_charged", "ordinal": ordinal,
                "error_code": "timeout" if isinstance(error, TimeoutError) else
                    "invalid_result" if stage == "validator" else "runtime_exception",
                "elapsed_seconds": time.monotonic()-started, "automatic_retry": False,
                "failure_is_clinical_contradiction": False})
            self.pending = False
            return None
        # A journal failure is not a model failure and must not rerun the worker.
        self.sink({"event": "new_worker_completed", "ordinal": ordinal,
            "result": result.record(), "elapsed_seconds": time.monotonic()-started,
            "clinical_acceptance": False})
        self.pending = False
        self.completed += 1
        if ordinal == 0:
            self.report_hash = result.output_artifact_sha256
        else:
            self.closed = True
        return result

    def snapshot(self):
        return {"charged_new_worker_attempts": self.charged, "completed_new_worker_operations": self.completed,
            "failed_new_worker_attempts": self.failed, "pending_new_worker_attempts": int(self.pending),
            "new_worker_budget": 2, "automatic_retry": False, "historical_ledger_changed": False}


def bind_request(case, result, pending, old, row, triple, book):
    """Pure source binding; neither the LLM reason nor old-image scores are truth."""
    packet = producer.validate_packet(case["packet"])
    decision = validate_decision(result["decision"], packet["observation"])
    require(result["case_id"] == case["case_id"] == old["case_id"] == row["case_id"] == triple["case_id"]
        and result["source"] == "new_frozen_qwen_generate_call"
        and result["policy_input_sha256"] == case["policy_input_sha256"] == _digest(packet)
        and not result["clinical_acceptance"] and not result["original_winner_changed"],
        "authenticated_fresh_decision_and_same_case_required")
    dispatch, payload = GuardedActionDispatcher(sink=lambda event: None).dispatch(
        decision, packet["observation"], packet["image_guard"], backend_factory=None)
    require(payload is None and dispatch == result["dispatch"]
        and dispatch["status"] == "deferred_separately_approved_backend_required"
        and decision["action"] == "regenerate_report", "sealed_deferred_report_request_required")
    request = case["tool_catalog"][dispatch["effective_decision"]["target_id"]]
    require(pending == {"case_id": case["case_id"], "request": request,
        "decision": decision, "clinical_acceptance": False}, "pending_request_must_match_actual_dispatch")
    require(request["input_image_sha256"] == case["trial_image_sha256"] == row["cxr_sha256"] == triple["cxr_sha256"]
        and request["seed"] == row["seed"] == triple["seed"] == 1
        and request["model_id"] in producer.MODELS
        and request["model_id"] != row["report_model_id"] == triple["report_model_id"]
        and request["cost_units"] == 2 and request["backend_installed_in_this_job"] is False
        and row["triple_candidate_id"] == case["trial_candidate_id"]
        and all(case[k] == row[k] == triple[k] == old["reference"][k]
            for k in ("ehr_sha256", "ehr_facts_sha256")), "unchanged_trial_image_ehr_and_untried_expert_required")
    selected_tool = next(t for t in packet["observation"]["tools"] if t["tool_id"] == decision["target_id"])
    require(selected_tool["model_id"] == f"m{producer.MODELS.index(request['model_id']):04d}"
        and selected_tool["seed"] == request["seed"] and selected_tool["cost_units"] == request["cost_units"],
        "opaque_tool_must_bind_exact_private_expert_catalog")
    context = gate.context(old["anchor"], old["workers"]["xrv"], old["workers"]["chexbert"])
    gate.validate_observation(row, context)
    anchor = gate.anchor_from_record(old["anchor"])
    gate.validate_ledger(book, anchor)
    gate.completed_operation(row, book)
    require(book["charged_model_attempts"] == book["call_budget"] == 4
        and not book["pending_attempts"] and book["max_retries"] == 0,
        "original_exhausted_four_call_ledger_must_stay_unchanged")
    events = [e for e in book["events"] if e["event"] == "attempt_reserved"
        and e["request"]["kind"] == "xrv"]
    require(len(events) == 1, "one_cached_image_scorer_required")
    return {"case_id": case["case_id"], "anchor": old["anchor"], "requested_model": request["model_id"],
        "baseline_row": row, "baseline_triple": triple, "historical_ledger": book,
        "historical_ledger_sha256": _digest(book), "cached_xrv_operation_id": events[0]["request"]["operation_id"],
        "retained_original_candidate_id": case["retained_original_candidate_id"],
        "sealed_policy_decision": decision, "current_image_evidence_cited": "e0004" in decision["evidence_ids"],
        "clinical_fault_location": None, "original_selection_change_allowed": False}


def assess_pair(case, proposed, workers):
    context = gate.context(case["anchor"], workers["xrv"], workers["chexbert"])
    baseline = case["baseline_row"]
    gate.validate_observation(proposed, context)
    require(proposed["report_model_id"] == case["requested_model"]
        and all(proposed[k] == baseline[k] for k in
            ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_candidate_id", "cxr_sha256", "seed")),
        "new_report_must_use_exact_approved_image_and_model")
    comparison = compare(baseline, proposed)
    return {"case_id": case["case_id"], "baseline_model": baseline["report_model_id"],
        "requested_model": case["requested_model"], "baseline_candidate_id": baseline["triple_candidate_id"],
        "proposed_candidate_id": proposed["triple_candidate_id"],
        "branch_selected_candidate_id": proposed["triple_candidate_id"] if comparison["exploratory_gate_pass"]
            else baseline["triple_candidate_id"], "comparison": comparison,
        "baseline_scores": baseline["raw_edge_readouts"], "proposed_scores": proposed["raw_edge_readouts"],
        "current_image_evidence_cited": case["current_image_evidence_cited"],
        "retained_original_candidate_id": case["retained_original_candidate_id"],
        "original_winner_changed": False, "clinical_acceptance": False, "clinical_repair_success": False}


def validate_plan(plan):
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG and len(plan["cases"]) == 2
        and plan["data_origin"] == "original_fully_synthetic_pool80"
        and plan["new_model_calls"] == 0 and plan["policy_run_manifest_sha256"] == POLICY_RUN_SHA
        and {c["requested_model"] for c in plan["cases"]} == {"maira2", "cxrmate_single"},
        "exact_two_sealed_requests_only")
    for case in plan["cases"]:
        require(_digest(case["historical_ledger"]) == case["historical_ledger_sha256"]
            and case["historical_ledger"]["call_budget"] == 4
            and case["original_selection_change_allowed"] is False, "historical_cost_contract_unchanged")
        images = load_cxr_candidates([case["baseline_triple"]["cxr_run"]])
        require(len(images) == 1 and next(iter(images.values())) == case["image_candidate"],
            "same_authenticated_existing_image_required")
        labels = read_json(case["image_labels_path"])
        require(sha256_file(case["image_labels_path"]) == case["baseline_row"]["receipt"]["xrv_labels_sha256"],
            "cached_image_labels_changed")
        partial = image_receipt(gate.anchor_from_record(case["anchor"]), case["image_candidate"], labels,
            label_sha256=sha256_file(case["image_labels_path"]),
            thresholds_sha256=plan["workers"]["xrv"]["thresholds_sha256"],
            checkpoint_sha256=plan["workers"]["xrv"]["checkpoint_sha256"])
        require(partial == case["partial_receipt"]
            and partial["receipt_id"] == case["baseline_row"]["receipt"]["partial_receipt_id"],
            "cached_frozen_image_receipt_changed")
    for key in ("source_pins", "artifact_pins"): check_pins(plan[key])
    for name in ("maira2", "cxrmate_single", "chexbert"):
        spec = plan["workers"][name]
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and os.access(spec["python"], os.X_OK), "existing_preflighted_frozen_worker_required")
        check_pins(spec["asset_pins"])


def prepare(args):
    gate.cpu_guard()
    readers = [Reader(p) for p in (POLICY_PLAN, POLICY_RUN, POLICY_AUDIT, source.GENERATION_PLAN, source.SOURCE)]
    pp, pr, pa, gp, sr = readers
    for r, pin in zip(readers, (POLICY_PLAN_SHA, POLICY_RUN_SHA, POLICY_AUDIT_SHA,
            source.GENERATION_PLAN_SHA, source.SOURCE_SHA), strict=True):
        r.hash(r.root/"manifest.json", pin)
    pm = pp.json(pp.root/"manifest.json"); pp.hash(pp.root/"plan.json", pm["plan_sha256"])
    policy_plan = pp.json(pp.root/"plan.json")
    rm = pr.json(pr.root/"manifest.json")
    require(rm["status"] == "completed_fresh_guarded_numeric_planning_unvalidated", "completed_numeric_policy_required")
    for name, pin in rm["artifacts"].items(): pr.hash(pr.root/name, pin["sha256"])
    results = pr.json(pr.root/"policy_results.json")["records"]
    pending = pr.json(pr.root/"pending_requests.json")
    require(len(results) == len(pending["records"]) == 2 and pending["automatic_submission_allowed"] is False
        and pending["requires_complete_script_and_explicit_approval"] is True, "two_deferred_not_autoauthorized_requests")
    am = pa.json(pa.root/"manifest.json"); pa.hash(pa.root/"audit.json", am["audit_sha256"])
    audit = pa.json(pa.root/"audit.json")
    require(audit["source_manifest_sha256"] == POLICY_RUN_SHA and audit["exact_dispatch_replay_pass"] is True,
        "completed_numeric_postflight_required")
    pr.hash(pr.root/"policy_dispatch.journal.jsonl", audit["journal_sha256"])
    gm = gp.json(gp.root/"manifest.json"); gp.hash(gp.root/"plan.json", gm["plan_sha256"])
    generation = gp.json(gp.root/"plan.json")
    sm = sr.json(sr.root/"manifest.json")
    for name, pin in sm["artifacts"].items(): sr.hash(sr.root/name, pin["sha256"])
    rows = sr.json(sr.root/"score_rows.json")["records"]
    triples = sr.json(sr.root/"completed_triplets.json")["new_records"]
    books = sr.json(sr.root/"execution_summary.json")["case_ledgers"]
    cases = []
    for c, r, p in zip(policy_plan["cases"], results, pending["records"], strict=True):
        old = deepcopy(next(x for x in generation["cases"] if x["case_id"] == c["case_id"]))
        old["workers"] = generation["workers"]
        row = next(x for x in rows if x["triple_candidate_id"] == c["trial_candidate_id"])
        triple = next(x for x in triples if x["case_id"] == c["case_id"])
        book = next(x for x in books if x["case_id"] == c["case_id"])
        bound = bind_request(c, r, p, old, row, triple, book)
        image_reader = Reader(triple["cxr_run"])
        im = image_reader.json(image_reader.root/"manifest.json")
        require(len(im["candidates"]) == 1, "single_trial_image_only")
        meta = im["candidates"][0]
        image_reader.hash(image_reader.root/meta["path"], meta["sha256"])
        image = image_reader.json(image_reader.root/meta["path"])
        sr.hash(image["artifact"]["path"], row["cxr_sha256"])
        label_path = sr.root/"cases"/bound["case_id"]/"operations"/(bound["cached_xrv_operation_id"]+"_a1")/"scored/cxr_finding_labels.json"
        sr.hash(label_path, row["receipt"]["xrv_labels_sha256"])
        labels = sr.json(label_path)
        bound.update(image_candidate=image, image_labels_path=str(label_path),
            partial_receipt=image_receipt(gate.anchor_from_record(bound["anchor"]), image, labels,
                label_sha256=sha256_file(label_path), thresholds_sha256=generation["workers"]["xrv"]["thresholds_sha256"],
                checkpoint_sha256=generation["workers"]["xrv"]["checkpoint_sha256"]))
        cases.append(bound); readers.append(image_reader)
    pins = {**policy_plan["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
        str(Path(__file__).resolve().parents[1]/"tests/test_guarded_requested_reports.py"):
            sha256_file(Path(__file__).resolve().parents[1]/"tests/test_guarded_requested_reports.py")}
    artifacts = deepcopy(policy_plan["artifact_pins"])
    for reader in readers:
        reader.recheck(); artifacts.update(reader.pins)
    plan = {"schema_version": VERSION, "config": CONFIG, "cases": cases,
        "workers": {"xrv": generation["workers"]["xrv"], "chexbert": generation["workers"]["chexbert"],
            **{m: policy_plan["report_catalog"][m] for m in ("maira2", "cxrmate_single")}},
        "source_pins": pins, "artifact_pins": artifacts, "policy_run_manifest_sha256": POLICY_RUN_SHA,
        "data_origin": "original_fully_synthetic_pool80", "new_model_calls": 0,
        "historical_cost": {"source_cxr_probe_worker_attempts": policy_plan["source_cxr_probe_worker_attempts"],
            "cached_observer_calls": policy_plan["cached_observer_calls"],
            "cached_observer_load_attempts": policy_plan["cached_observer_load_attempts"],
            "latest_numeric_policy_calls": 2, "latest_numeric_policy_load_attempts": 1,
            "earlier_policy_calls": policy_plan["earlier_stage_policy_requests"]+policy_plan["source_cxr_probe_policy_requests"],
            "earlier_worker_attempts": policy_plan["earlier_stage_worker_attempts"], "scope": "shared_sunk_not_zero"}}
    validate_plan(plan)
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(tmp/"plan.json", plan)
        write_private_json(tmp/"manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(path),
            "maximum_new_report_attempts": 2, "maximum_new_chexbert_attempts": 2,
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException:
        discard_atomic_run(tmp); raise
    return target


def run_case(case, plan, root):
    producer.gpu_guard()
    anchor = gate.anchor_from_record(case["anchor"])
    cr = root/"cases"/anchor.case_id
    private_directory(cr); private_directory(cr/"operations")
    write_private_json(cr/"source_ledger.json", case["historical_ledger"])
    image = case["image_candidate"]; workers = plan["workers"]
    pack = prepare_report_request_run(cxr_runs=[case["baseline_triple"]["cxr_run"]], output_root=cr/"operations",
        run_id="requested_report_pack", model_ids=[case["requested_model"]])
    request_run = Path(pack["run_directory"])
    state = {}; triple = None
    with ProtectedJournal(cr/"new_phase.journal.jsonl") as journal:
        journal.append({"event": "cached_dependencies_authenticated", "source_policy_manifest_sha256": POLICY_RUN_SHA,
            "source_ledger_sha256": case["historical_ledger_sha256"],
            "cached_partial_receipt_id": case["partial_receipt"]["receipt_id"], "new_budget": 2,
            "cached_dependencies_are_new_model_calls": False, "clinical_acceptance": False})
        calls = RequestedReportCalls(case, sink=journal.append)

        def invoke(kind, model, parent, validator, report_hash=None, **inputs):
            spec = workers[model]
            operation = "requested_report" if kind == "report_generator" else "requested_chexbert"
            op_root = cr/"operations"/(operation+"_a1")
            private_directory(op_root)
            request = CallRequest(operation, anchor.case_id, anchor.sha256, kind, model, _digest(spec),
                image["seed"], parent, image["artifact"]["sha256"], report_hash)
            timeout = CONFIG["report_timeout_seconds"] if kind == "report_generator" else CONFIG["chexbert_timeout_seconds"]
            return calls.invoke(request, factory=lambda: LocalFrozenBackend(spec,
                operation_root=op_root, timeout=timeout, **inputs), validator=validator)

        def report_validator(payload, operation):
            rr, report = single_generated(payload, workers[case["requested_model"]], request_run,
                cxr_candidates={image["candidate_id"]: image})
            validate_report_binding(report, image, request_run, anchor)
            state.update(report_run=rr, report=report)
            return CallResult(report["artifact"]["sha256"])

        result = invoke("report_generator", case["requested_model"], case["cached_xrv_operation_id"],
            report_validator, request_run=request_run)
        if result is not None:
            def label_validator(payload, operation):
                nonlocal triple
                require(payload["worker_audit_sha256"] == operation.frozen_model_audit_sha256,
                    "unchanged_frozen_chexbert_required")
                op_root = require_inside(payload["output_root"], PROTECTED_ROOT, must_exist=True)
                lp = op_root/"scored/report_finding_labels.json"
                receipt = completed_receipt(anchor, case["partial_receipt"], image, read_json(case["image_labels_path"]),
                    state["report"], read_json(lp), image_labels_sha256=sha256_file(case["image_labels_path"]),
                    report_labels_sha256=sha256_file(lp), thresholds_sha256=workers["xrv"]["thresholds_sha256"],
                    xrv_checkpoint_sha256=workers["xrv"]["checkpoint_sha256"],
                    chexbert_checkpoint_sha256=workers["chexbert"]["checkpoint_sha256"])
                rp = write_private_json(op_root/"completed_receipt.json", receipt)
                triple = {**case["baseline_triple"], "report_run": str(state["report_run"]),
                    "report_path": state["report"]["artifact"]["path"], "report_sha256": state["report"]["artifact"]["sha256"],
                    "report_model_id": case["requested_model"], "receipt_path": str(rp), "receipt_sha256": sha256_file(rp),
                    "receipt_id": receipt["receipt_id"], "raw_edge_readouts": receipt["raw_edge_readouts"],
                    "verification_status": receipt["verification_status"], "selected_as_best": False}
                return CallResult(sha256_file(lp), receipt["receipt_id"])
            label_result = invoke("chexbert", "chexbert", "requested_report", label_validator,
                report_hash=result.output_artifact_sha256, cxr_run=case["baseline_triple"]["cxr_run"], report_run=state["report_run"])
            if label_result is None:
                triple = None
        snapshot = calls.snapshot()
        journal.append({"event": "new_phase_sealed", **snapshot, "clinical_acceptance": False})
    write_private_json(cr/"new_phase_snapshot.json", snapshot)
    return triple, snapshot


def run(args):
    producer.gpu_guard()
    planroot = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    require(sha256_file(planroot/"manifest.json") == args.plan_manifest_sha256, "reviewed_report_plan_changed")
    pm = read_json(planroot/"manifest.json")
    require(pm["status"] == "prepared_cpu_only_gpu_not_submitted"
        and sha256_file(planroot/"plan.json") == pm["plan_sha256"], "sealed_cpu_plan_required")
    plan = read_json(planroot/"plan.json"); validate_plan(plan)
    import torch
    require(torch.cuda.get_device_properties(0).total_memory >= CONFIG["minimum_gpu_vram_gib"]*1024**3,
        "maira2_requires_reviewed_memory_class")
    require(RUN_ID_PATTERN.fullmatch(args.run_id), "opaque_new_run_id_required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = output/args.run_id
    private_directory(root); private_directory(root/"cases")
    # Stable paths: existing worker adapters embed absolute artifact paths.
    write_private_json(root/"start_manifest.json", {"schema_version": VERSION, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "automatic_resume": False})
    triples = []; rows = []; comparisons = []; snapshots = []; started = time.monotonic()
    try:
        for case in plan["cases"]:
            triple, snapshot = run_case(case, plan, root)
            snapshots.append({"case_id": case["case_id"], **snapshot})
            rows.append(case["baseline_row"])
            if triple is not None:
                row = candidate_row(triple)
                pair = assess_pair(case, row, plan["workers"])
                rows.append(row); triples.append(triple); comparisons.append(pair)
            else:
                comparisons.append({"case_id": case["case_id"], "status": "failed_charged_no_retry",
                    "baseline_candidate_id": case["baseline_row"]["triple_candidate_id"],
                    "branch_selected_candidate_id": case["baseline_row"]["triple_candidate_id"],
                    "original_winner_changed": False, "clinical_acceptance": False})
        validate_plan(plan)
        artifacts = [write_private_json(root/"score_rows.json", {"schema_version": VERSION, "records": rows}),
            write_private_text(root/"score_table.csv", score_csv([{**r,"verification_status":r["receipt"]["verification_status"]} for r in rows])),
            write_private_json(root/"paired_report_comparison.json", {"schema_version": VERSION, "records": comparisons}),
            write_private_json(root/"completed_triplets.json", {"schema_version": VERSION, "new_records": triples}),
            write_private_json(root/"execution_summary.json", {"schema_version": VERSION, "records": snapshots,
                "historical_cost": plan["historical_cost"], "historical_cost_not_erased": True})]
        summary = {"schema_version": VERSION, "status": "completed_requested_report_comparison_unvalidated",
            "requested_reports": 2, "completed_report_label_pairs": len(triples),
            "charged_new_worker_attempts": sum(s["charged_new_worker_attempts"] for s in snapshots),
            "completed_new_worker_operations": sum(s["completed_new_worker_operations"] for s in snapshots),
            "failed_new_worker_attempts": sum(s["failed_new_worker_attempts"] for s in snapshots),
            "proxy_preserving_branch_changes": sum(bool(p.get("comparison",{}).get("exploratory_gate_pass")) for p in comparisons),
            "new_planner_calls": 0, "new_cxr_calls": 0, "new_xrv_calls": 0, "new_image_observer_calls": 0,
            "runtime_seconds": round(time.monotonic()-started,3), "training_performed": False,
            "original_winner_changed": False, "clinical_acceptance": False, "clinical_repair_success": False,
            "clinical_accuracy": None, "measured_saved_model_calls": None,
            "selection_scope": "same_image_diagnostic_branch_only_not_promoted_original_triples"}
        artifacts.append(write_private_json(root/"summary.json", summary))
        journals = list(root.glob("cases/*/new_phase.journal.jsonl"))
        artifacts += journals + list(root.glob("cases/*/new_phase_snapshot.json"))
        write_private_json(root/"manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256, "clinical_acceptance": False,
            "artifacts": {str(p.relative_to(root)): {"sha256": sha256_file(p)} for p in artifacts}})
    except BaseException:
        write_private_json(root/"failure.json", {"schema_version": VERSION, "status": "failed_prefix_retained",
            "automatic_resume": False, "charged_attempts_remain_in_case_journals": True,
            "completed_prefix_snapshots": snapshots, "clinical_acceptance": False})
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep, execution = sub.add_parser("prepare"), sub.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(target/"manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())

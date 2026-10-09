#!/usr/bin/env python3
"""Two current-image-only Qwen decisions versus a fixed numeric rule control.

CPU prepare reads authenticated synthetic numeric metadata, never report bodies
or image pixels. GPU run makes at most two frozen local Qwen calls. All requests
are deferred; generators, scoring thresholds and original winners stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import run_guarded_report_secondary as authenticated
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, read_json, require_inside, sha256_file, write_private_json)
from tricompose_llm.contracts import PUBLIC_SCHEMA, validate_public_state
from tricompose_llm.current_image_qwen import (CurrentImageQwenPlanner, VERSION as PACKET_VERSION,
    RISKS, validate_packet, request_messages, rule_decision)
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher
from tricompose_llm.live_report_bridge import FreshReportSession
from tricompose_llm.local_qwen import gpu_guard
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.report_expert_control import MODELS
from tricompose_v12.runtime_dispatch import ProtectedJournal

source = authenticated.source
cached = source.producer.previous.cached
BASE = source.BASE
VERSION = "tricompose-current-image-qwen-rule-decision-control-v1"
CONFIG = {"maximum_policy_calls": 2, "new_phase_budget_units_per_case": 3,
    "policy_units": 1, "common_worker_allowance": 2, "report_and_chexbert_units": 2,
    "max_new_tokens": 384, "max_context_tokens": 8192, "do_sample": False,
    "min_vram_gib": 24, "generation_backend_installed": False, "model_retries": 0,
    "actual_rule_model_calls": 0, "endpoint_scores_supplied": False}
require = source.require


def dispatch_record(decision, packet, *, sink):
    dispatch, payload = GuardedActionDispatcher(sink=sink).dispatch(decision,
        packet["observation"], packet["image_guard"], backend_factory=None)
    require(payload is None and dispatch["actual_worker_model_calls"] == 0
        and dispatch["backend_dispatch_attempts"] == 0, "no_generation_backend_allowed")
    return dispatch


def make_case(old, proposed, pair, image_observation, workers):
    base = old["baseline_row"]
    require(source.assess_pair(old, proposed, workers) == pair
        and pair["comparison"]["exploratory_gate_pass"] is False
        and pair["branch_selected_candidate_id"] == base["triple_candidate_id"]
        and {base["report_model_id"], proposed["report_model_id"]} == {"maira2", "cxrmate_single"},
        "actual_rejected_same_image_report_pair_required")
    context = source.gate.context(old["anchor"], workers["xrv"], workers["chexbert"])
    numeric = [FreshReportSession._numeric(SimpleNamespace(context=context), row, i)
        for i, row in enumerate((base, proposed))]
    tools = [{"tool_id": f"t{100+i:04d}", "action": "regenerate_report", "model_id": f"m{i:04d}",
        "seed": 1, "cost_units": 2} for i in (2, 3)]
    state = validate_public_state({"schema_version": PUBLIC_SCHEMA, "step_id": "s0000",
        "current_candidate_id": "c0000", "evidence": numeric, "tools": tools,
        "budget": {"limit_units": 3, "spent_units": 1, "planner_units_per_call": 1}, "history": []})
    c = pair["comparison"]
    packet = validate_packet({"schema_version": PACKET_VERSION, "observation": state,
        "image_guard": cached.guard_for_image(image_observation, [base, proposed],
            candidate_id="c0000", evidence_id="e0000"),
        "observed_reports": [{"candidate_id": f"c{i:04d}", "evidence_id": f"e{i:04d}",
            "model_id": f"m{MODELS.index(row['report_model_id']):04d}",
            "risk_flags": {k: row["structure"][k] for k in RISKS},
            "repeated_sentence_count": row["structure"]["repeated_sentence_count"],
            "repeated_4gram_ratio": row["structure"]["repeated_4gram_ratio"]}
            for i, row in enumerate((base, proposed))],
        "rejected_transition": {"baseline_candidate_id": "c0000", "proposed_candidate_id": "c0001",
            "accepted_proxy_transition": False, "quality_no_worse": c["per_model_structure_and_common_risk_no_worse"],
            "duplicate": c["duplicate"], **{key: {name: len(ids) for name, ids in c[field].items()}
                for key, field in (("lost", "lost_fact_ids"), ("gained", "gained_fact_ids"),
                    ("new_opposition", "new_opposition_fact_ids"), ("removed_opposition", "removed_opposition_fact_ids"),
                    ("silenced", "silenced_fact_ids"))}}, "generation_backend_installed": False})
    control = rule_decision(packet)
    return {"case_id": old["case_id"], "ehr_sha256": base["ehr_sha256"],
        "ehr_facts_sha256": base["ehr_facts_sha256"], "trial_image_sha256": base["cxr_sha256"],
        "retained_branch_candidate_id": base["triple_candidate_id"],
        "rejected_branch_candidate_id": proposed["triple_candidate_id"],
        "retained_original_candidate_id": old["retained_original_candidate_id"],
        "packet": packet, "policy_input_sha256": _digest(packet),
        "request_messages_sha256": _digest(request_messages(packet)),
        "rule_control": {"decision": control, "dispatch": dispatch_record(control, packet, sink=lambda event: None),
            "actual_model_calls": 0, "actual_planner_units_charged": 0,
            "common_prospective_worker_allowance": 2, "reserved_planning_allowance_is_incurred_cost": False},
        "tool_catalog": {f"t{100+i:04d}": {"model_id": MODELS[i], "seed": 1,
            "input_image_sha256": base["cxr_sha256"], "cost_units": 2,
            "cost_scope": "one_report_generator_and_one_chexbert", "backend_installed_in_this_job": False}
            for i in (2, 3)}, "original_selection_change_allowed": False}


def validate_case(case):
    packet = validate_packet(case["packet"])
    require(case["policy_input_sha256"] == _digest(packet)
        and case["request_messages_sha256"] == _digest(request_messages(packet))
        and case["original_selection_change_allowed"] is False, "sealed_current_image_packet_required")
    control = case["rule_control"]
    decision = rule_decision(packet)
    require(control == {"decision": decision, "dispatch": dispatch_record(decision, packet, sink=lambda event: None),
        "actual_model_calls": 0, "actual_planner_units_charged": 0,
        "common_prospective_worker_allowance": 2, "reserved_planning_allowance_is_incurred_cost": False},
        "exact_same_information_rule_control_required")
    expected = {f"t{100+i:04d}": {"model_id": MODELS[i], "seed": 1,
        "input_image_sha256": case["trial_image_sha256"], "cost_units": 2,
        "cost_scope": "one_report_generator_and_one_chexbert", "backend_installed_in_this_job": False} for i in (2, 3)}
    require(case["tool_catalog"] == expected, "sealed_real_untried_expert_catalog_required")


def prepare(args):
    source.gate.cpu_guard()
    paths = (authenticated.SOURCE, authenticated.SOURCE_PLAN, authenticated.AUDIT,
        source.POLICY_PLAN, cached.OBSERVATIONS)
    pins = (authenticated.SOURCE_SHA, authenticated.SOURCE_PLAN_SHA, authenticated.AUDIT_SHA,
        source.POLICY_PLAN_SHA, cached.OBSERVATIONS_SHA)
    readers = [source.Reader(p) for p in paths]
    for r, pin in zip(readers, pins, strict=True): r.hash(r.root / "manifest.json", pin)
    sr, pr, ar, qr, ir = readers
    sm = sr.json(sr.root / "manifest.json")
    require(sm["status"] == "completed_requested_report_comparison_unvalidated", "completed_observed_reports_required")
    for name, pin in sm["artifacts"].items(): sr.hash(sr.root / name, pin["sha256"])
    pm = pr.json(pr.root / "manifest.json"); pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    original = pr.json(pr.root / "plan.json")
    require(original["data_origin"] == "original_fully_synthetic_pool80" and len(original["cases"]) == 2,
        "same_two_original_synthetic_cases_required")
    am = ar.json(ar.root / "manifest.json"); ar.hash(ar.root / "audit.json", am["audit_sha256"])
    audit = ar.json(ar.root / "audit.json")
    require(audit["source_manifest_sha256"] == authenticated.SOURCE_SHA
        and audit["plan_manifest_sha256"] == authenticated.SOURCE_PLAN_SHA
        and audit["exact_new_phase_replay_pass"] is True and audit["csv_bytes_exact_replay"] is True
        and audit["protected_permissions_pass"] is True and audit["new_model_calls"] == 0,
        "completed_report_execution_postflight_required")
    qm = qr.json(qr.root / "manifest.json"); qr.hash(qr.root / "plan.json", qm["plan_sha256"])
    previous = qr.json(qr.root / "plan.json")
    rows = sr.json(sr.root / "score_rows.json")["records"]
    pairs = sr.json(sr.root / "paired_report_comparison.json")["records"]
    authenticated.freeze_controls(rows, pairs)
    im = ir.json(ir.root / "manifest.json")
    require(im["status"] == "completed_blind_image_observer_unvalidated", "cached_blind_observer_required")
    for name, pin in im["artifacts"].items(): ir.hash(ir.root / name, pin["sha256"])
    observations = ir.json(ir.root / "predictions.json")
    require(observations["frozen"] is True and observations["image_only"] is True
        and observations["model_received_ehr_reports_ids_or_scores"] is False, "blind_image_observations_only")
    cached.verify_calls(observations["records"], ir.journal(ir.root / "calls.journal.jsonl"))
    cases = []
    for old in original["cases"]:
        matches = [p for p in pairs if p["case_id"] == old["case_id"]]
        require(len(matches) == 1, "one_actual_rejected_pair_per_case_required")
        pair = matches[0]
        proposed = next(r for r in rows if r["triple_candidate_id"] == pair["proposed_candidate_id"])
        require(sum(r["cxr_sha256"] == proposed["cxr_sha256"] for r in rows) == 2,
            "exact_two_observed_experts_on_current_image_required")
        records = [r for r in observations["records"] if r["cxr_candidate_id"] == proposed["cxr_candidate_id"]]
        require(len(records) == 1, "one_authenticated_current_image_observation_required")
        case = make_case(old, proposed, pair, records[0], original["workers"])
        validate_case(case); cases.append(case)
    source_pins = {**original["source_pins"], **{str(p.resolve()): sha256_file(p) for p in (
        Path(__file__), Path(__file__).parent / "tricompose_llm/current_image_qwen.py",
        Path(__file__).resolve().parents[1] / "tests/test_current_image_policy.py")}}
    artifacts = {**original["artifact_pins"], **audit["reader_pins"]}
    for r in readers: r.recheck(); artifacts.update(r.pins)
    source.check_pins(source_pins); source.check_pins(artifacts)
    source.check_pins(previous["qwen"]["asset_pins"])
    plan = {"schema_version": VERSION, "config": CONFIG, "cases": cases,
        "qwen": previous["qwen"], "report_catalog": previous["report_catalog"],
        "source_pins": source_pins, "artifact_pins": artifacts,
        "data_origin": "original_fully_synthetic_pool80", "candidate_rule": "both_original_probes_no_outcome_selection",
        "historical_cost": {**original["historical_cost"], "completed_requested_report_worker_attempts": 4},
        "historical_cost_not_erased": True, "source_bodies_parsed": False, "image_pixels_decoded": False,
        "comparison_scope": "same_observation_same_action_budget_decisions_not_live_outcomes_or_gpu_savings",
        "endpoint_scores_supplied": False, "new_model_calls": 0, "original_selection_changed": False}
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(tmp / "plan.json", plan)
        write_private_json(tmp / "rule_control.json", {"schema_version": VERSION,
            "records": [{"case_id": c["case_id"], "policy_input_sha256": c["policy_input_sha256"],
                **c["rule_control"]} for c in cases], "actual_model_calls": 0,
            "generator_execution_authorized": False, "clinical_acceptance": False})
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(path),
            "rule_control_sha256": sha256_file(tmp / "rule_control.json"),
            "maximum_new_policy_calls": 2, "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target


def run(args):
    gpu_guard()
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    require(sha256_file(root / "manifest.json") == args.plan_manifest_sha256, "reviewed_current_image_plan_changed")
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only_gpu_not_submitted"
        and sha256_file(root / "plan.json") == manifest["plan_sha256"]
        and sha256_file(root / "rule_control.json") == manifest["rule_control_sha256"], "sealed_cpu_plan_required")
    plan = read_json(root / "plan.json")
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG and len(plan["cases"]) == 2
        and len({c["case_id"] for c in plan["cases"]}) == 2
        and plan["data_origin"] == "original_fully_synthetic_pool80"
        and plan["endpoint_scores_supplied"] is False and plan["source_bodies_parsed"] is False
        and plan["image_pixels_decoded"] is False and plan["original_selection_changed"] is False,
        "bounded_numeric_decision_only_scope_required")
    for field in ("source_pins", "artifact_pins"): source.check_pins(plan[field])
    source.check_pins(plan["qwen"]["asset_pins"])
    for case in plan["cases"]: validate_case(case)
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    results = []; charged = 0; started = time.monotonic(); load_charged = 0
    try:
        with ProtectedJournal(tmp / "policy_dispatch.journal.jsonl") as journal, open(os.devnull, "w") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            journal.append({"event": "planner_load_reserved", "charged_load_attempts": 1})
            load_charged = 1
            planner = CurrentImageQwenPlanner(plan["qwen"]["model_path"])
            for case in plan["cases"]:
                packet = case["packet"]
                journal.append({"event": "current_policy_reserved", "ordinal": charged,
                    "policy_input_sha256": case["policy_input_sha256"], "charged_policy_units": 1})
                charged += 1
                try: decision = planner.propose_current(deepcopy(packet))
                except Exception:
                    results.append({"case_id": case["case_id"], "status": "planner_failed_charged_no_retry",
                        "decision": None, "dispatch": None, "rule_control": case["rule_control"],
                        "clinical_acceptance": False, "original_winner_changed": False})
                    journal.append({"event": "current_policy_failed_charged", "ordinal": charged - 1})
                    continue
                journal.append({"event": "current_policy_validated", "ordinal": charged - 1, "decision": decision})
                dispatch = dispatch_record(decision, packet,
                    sink=lambda event: journal.append({"case_id": case["case_id"], **event}))
                results.append({"case_id": case["case_id"], "status": "current_llm_decision_validated_unverified",
                    "source": "new_frozen_qwen_generate_call", "decision": decision, "dispatch": dispatch,
                    "policy_input_sha256": case["policy_input_sha256"],
                    "response_metadata": deepcopy(planner.last_response_metadata), "rule_control": case["rule_control"],
                    "same_effective_action_and_target_as_rule": all(dispatch["effective_decision"][k]
                        == case["rule_control"]["dispatch"]["effective_decision"][k] for k in ("action", "target_id")),
                    "clinical_acceptance": False, "original_winner_changed": False})
            audit = planner.audit()
        pending = []
        for result in results:
            case = next(c for c in plan["cases"] if c["case_id"] == result["case_id"])
            for policy, dispatch in (("qwen", result["dispatch"]), ("rule", case["rule_control"]["dispatch"])):
                if dispatch is not None and dispatch["status"] == "deferred_separately_approved_backend_required":
                    pending.append({"case_id": case["case_id"], "policy": policy,
                        "request": case["tool_catalog"][dispatch["effective_decision"]["target_id"]],
                        "clinical_acceptance": False})
        outputs = [write_private_json(tmp / "policy_results.json", {"schema_version": VERSION, "records": results}),
            write_private_json(tmp / "pending_requests.json", {"schema_version": VERSION, "records": pending,
                "automatic_submission_allowed": False, "requires_complete_script_and_explicit_approval": True}),
            write_private_json(tmp / "planner_audit.json", audit),
            write_private_json(tmp / "summary.json", {"schema_version": VERSION,
                "status": "completed_current_image_decision_control_unvalidated", "fixed_development_cases": 2,
                "charged_new_policy_requests": charged, "fresh_generate_attempts": audit["generate_attempts"],
                "validated_fresh_decisions": sum(r["decision"] is not None for r in results),
                "same_effective_action_and_target_as_rule": sum(r.get("same_effective_action_and_target_as_rule", False)
                    for r in results), "actual_rule_model_calls": 0, "new_worker_model_calls": 0,
                "common_prospective_worker_allowance_per_case": 2, "charged_load_attempts": load_charged,
                "rule_planning_allowance_is_not_actual_cost": True, "historical_cost_not_erased": True,
                "endpoint_scores_supplied": False, "original_winner_changed": False,
                "clinical_acceptance": False, "clinical_accuracy": None, "measured_saved_model_calls": None,
                "llm_superiority_demonstrated": False, "training_performed": False,
                "runtime_seconds": round(time.monotonic() - started, 3),
                "peak_torch_allocated_vram_gib": round(audit["peak_gpu_memory_bytes"] / 1024**3, 3)})]
        for field in ("source_pins", "artifact_pins"): source.check_pins(plan[field])
        source.check_pins(plan["qwen"]["asset_pins"])
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "completed_current_image_decision_control_unvalidated",
            "plan_manifest_sha256": args.plan_manifest_sha256,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in outputs},
            "charged_new_policy_requests": charged, "new_worker_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException:
        write_private_json(tmp / "failure.json", {"schema_version": VERSION, "status": "failed_attempts_retained",
            "charged_policy_requests": charged, "charged_load_attempts": load_charged,
            "completed_prefix_records": results, "automatic_resume": False})
        commit_atomic_run(tmp, target); raise
    return target


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

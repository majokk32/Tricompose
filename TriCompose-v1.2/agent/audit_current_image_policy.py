#!/usr/bin/env python3
"""CPU-only metadata replay for sealed same-image plans and decision outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_current_image_policy as worker
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json)
from tricompose_llm.contracts import count, exact, require, validate_decision

VERSION = "tricompose-current-image-decision-postflight-v1"


def replay(plan, results, pending, audit, journal, summary):
    """Pure count/decision replay; no model construction or artifact body reads."""
    require(len(results) == len(plan["cases"]) == 2, "two_decision_results_required")
    expected = [{"event": "planner_load_reserved", "charged_load_attempts": 1}]
    deferred = []; valid = 0; same = 0
    for ordinal, (case, result) in enumerate(zip(plan["cases"], results, strict=True)):
        worker.validate_case(case)
        require(result["case_id"] == case["case_id"] and result["rule_control"] == case["rule_control"]
            and result["clinical_acceptance"] is False and result["original_winner_changed"] is False,
            "unchanged_case_rule_and_claim_scope_required")
        expected.append({"event": "current_policy_reserved", "ordinal": ordinal,
            "policy_input_sha256": case["policy_input_sha256"], "charged_policy_units": 1})
        if result["decision"] is None:
            require(result["status"] == "planner_failed_charged_no_retry" and result["dispatch"] is None,
                "failed_call_cannot_have_decision_or_backend")
            expected.append({"event": "current_policy_failed_charged", "ordinal": ordinal})
        else:
            packet = case["packet"]
            decision = validate_decision(result["decision"], packet["observation"])
            require(result["source"] == "new_frozen_qwen_generate_call"
                and result["status"] == "current_llm_decision_validated_unverified"
                and result["policy_input_sha256"] == case["policy_input_sha256"], "fresh_numeric_call_required")
            expected.append({"event": "current_policy_validated", "ordinal": ordinal, "decision": decision})
            dispatch = worker.dispatch_record(decision, packet,
                sink=lambda event: expected.append({"case_id": case["case_id"], **event}))
            require(dispatch == result["dispatch"], "exact_dispatch_replay_required")
            value = all(dispatch["effective_decision"][k]
                == case["rule_control"]["dispatch"]["effective_decision"][k] for k in ("action", "target_id"))
            require(result["same_effective_action_and_target_as_rule"] is value, "rule_comparison_replay_required")
            valid += 1; same += int(value)
            metadata = result["response_metadata"]
            exact(metadata, ("input_tokens", "output_tokens", "token_limit_reached", "response_sha256"),
                "closed_response_metadata_required")
            count(metadata["input_tokens"], 8192); count(metadata["output_tokens"], 383)
            require(metadata["token_limit_reached"] is False and isinstance(metadata["response_sha256"], str)
                and len(metadata["response_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in metadata["response_sha256"]), "valid_complete_response_digest_required")
        for policy, dispatch in (("qwen", result["dispatch"]), ("rule", case["rule_control"]["dispatch"])):
            if dispatch is not None and dispatch["status"] == "deferred_separately_approved_backend_required":
                deferred.append({"case_id": case["case_id"], "policy": policy,
                    "request": case["tool_catalog"][dispatch["effective_decision"]["target_id"]],
                    "clinical_acceptance": False})
    require(journal == expected, "exact_durable_policy_call_and_dispatch_replay_required")
    require(pending == {"schema_version": worker.VERSION, "records": deferred,
        "automatic_submission_allowed": False, "requires_complete_script_and_explicit_approval": True},
        "exact_deferred_request_inventory_required")
    require(audit["frozen"] is True and audit["interface_version"] == worker.PACKET_VERSION
        and audit["local_attempts"] == 2 and valid <= audit["generate_attempts"] <= 2
        and audit["historical_other_image_experts_supplied"] is False and audit["endpoint_scores_supplied"] is False
        and sum(audit["failure_counts"].values()) == 2 - valid, "audited_frozen_call_counts_required")
    for path, pin in plan["qwen"]["asset_pins"].items():
        require(audit["asset_pins"][Path(path).name] == (pin["sha256"] if isinstance(pin, dict) else pin),
            "planner_asset_audit_must_match_preflight")
    require(summary["charged_new_policy_requests"] == summary["fresh_generate_attempts"] +
        sum(audit["failure_counts"][k] for k in ("state_contract", "tokenize", "context_bound")) == 2
        and summary["fresh_generate_attempts"] == audit["generate_attempts"]
        and summary["validated_fresh_decisions"] == valid
        and summary["same_effective_action_and_target_as_rule"] == same
        and summary["actual_rule_model_calls"] == summary["new_worker_model_calls"] == 0
        and summary["llm_superiority_demonstrated"] is False and summary["clinical_acceptance"] is False
        and summary["measured_saved_model_calls"] is None and summary["original_winner_changed"] is False,
        "exact_cost_and_no_efficacy_claim_required")
    return {"charged_new_policy_requests": 2, "validated_decisions": valid,
        "same_effective_action_and_target_as_rule": same, "deferred_request_rows": len(deferred),
        "exact_durable_dispatch_replay_pass": True}


def build(args):
    worker.source.gate.cpu_guard()
    pr = worker.source.Reader(args.plan_run)
    pr.hash(pr.root / "manifest.json", args.plan_manifest_sha256)
    pm = pr.json(pr.root / "manifest.json")
    require(pm["schema_version"] == worker.VERSION and pm["status"] == "prepared_cpu_only_gpu_not_submitted",
        "prepared_current_image_plan_required")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    pr.hash(pr.root / "rule_control.json", pm["rule_control_sha256"])
    plan = pr.json(pr.root / "plan.json")
    require(plan["schema_version"] == worker.VERSION and plan["config"] == worker.CONFIG
        and len(plan["cases"]) == 2 and plan["data_origin"] == "original_fully_synthetic_pool80",
        "same_two_synthetic_decision_cases_required")
    for case in plan["cases"]: worker.validate_case(case)
    require(pr.json(pr.root / "rule_control.json") == {"schema_version": worker.VERSION,
        "records": [{"case_id": c["case_id"], "policy_input_sha256": c["policy_input_sha256"],
            **c["rule_control"]} for c in plan["cases"]], "actual_model_calls": 0,
        "generator_execution_authorized": False, "clinical_acceptance": False}, "exact_cpu_rule_control_required")
    for key in ("source_pins", "artifact_pins"): worker.source.check_pins(plan[key])
    worker.source.check_pins(plan["qwen"]["asset_pins"])
    result = {"schema_version": VERSION, "plan_manifest_sha256": args.plan_manifest_sha256,
        "plan_packet_and_rule_replay_pass": True, "new_model_calls": 0,
        "source_bodies_parsed": False, "image_pixels_decoded": False,
        "original_winner_changed": False, "clinical_acceptance": False,
        "auditor_source_sha256": sha256_file(__file__), "observed_report_rows": 4,
        "pending_gpu_execution": args.source_run is None}
    if args.source_run is not None:
        require(args.source_manifest_sha256 is not None, "explicit_completed_source_hash_required")
        sr = worker.source.Reader(args.source_run)
        sr.hash(sr.root / "manifest.json", args.source_manifest_sha256)
        sm = sr.json(sr.root / "manifest.json")
        require(sm["schema_version"] == worker.VERSION
            and sm["status"] == "completed_current_image_decision_control_unvalidated"
            and sm["plan_manifest_sha256"] == args.plan_manifest_sha256, "completed_sealed_numeric_run_required")
        for name, pin in sm["artifacts"].items(): sr.hash(sr.root / name, pin["sha256"])
        result.update(replay(plan, sr.json(sr.root / "policy_results.json")["records"],
            sr.json(sr.root / "pending_requests.json"), sr.json(sr.root / "planner_audit.json"),
            sr.journal(sr.root / "policy_dispatch.journal.jsonl"), sr.json(sr.root / "summary.json")))
        sr.recheck(); result.update(source_manifest_sha256=args.source_manifest_sha256,
            reader_pins={**pr.pins, **sr.pins})
    else: result["reader_pins"] = pr.pins
    pr.recheck()
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        audit = write_private_json(tmp / "audit.json", result)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "completed_cpu_only_plan_audit" if args.source_run is None else "completed_cpu_only_run_audit",
            "audit_sha256": sha256_file(audit), "new_model_calls": 0})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-run", type=Path, required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = build(args)
        print(json.dumps({"stage": VERSION, "status": "completed_cpu_only",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Metadata-only CPU postflight for the one-image action, not clinical truth.

Replay numeric planning, completed/failing worker chains, unchanged gates and
exact selected artifact pointers. Never load models, parse EHR/report bodies,
decode images, submit jobs, alter scores or resume incomplete generation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import audit_fresh_report_agent as postflight
import run_fresh_cxr_agent as producer
from contracts import (new_atomic_run, commit_atomic_run, discard_atomic_run,
    sha256_file, write_private_json, write_private_text)
from tricompose_v12.execution_ledger import BoundedCallLedger
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import score_csv
from tricompose_v12.live_workers import check_pins
from tricompose_llm.live_cxr_bridge import FreshCXRSession

VERSION = "tricompose-fresh-cxr-postflight-v1"
ARTIFACTS = {"selection.json", "score_rows.json", "score_table.csv", "completed_triplets.json",
             "selected_triplets.json", "execution_summary.json", "policy_audits.json", "summary.json"}
require = postflight.require


def replay_case(case, context, final, journal, step, selection, selection_sha256):
    """Use only past cache state before the decision; future rows after it."""
    final = postflight.replay_book(final)
    anchor = producer.gate.anchor_from_record(context["anchor"])
    initial = BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
        call_budget=4, max_retries=0, execution_mode=final["execution_mode"], sink=lambda event: None).snapshot()
    cursor = 0

    def emit(expected):
        nonlocal cursor
        require(cursor < len(journal) and _digest(journal[cursor]) == _digest(expected),
                "policy_journal_drift")
        cursor += 1

    session = FreshCXRSession(case["cached_rows"], case["reference"]["triple_candidate_id"],
        context, initial, source_manifest_sha256=producer.source.SOURCE_SHA, sink=emit)
    state = session.begin_planning()
    valid_policy = None
    proposed = None
    if state is None:
        require(step is None and final == initial, "unrequested_policy_or_worker")
    else:
        require(step is not None and _digest(step["state"]) == _digest(state),
                "policy_observed_future_or_changed_evidence")
        require(cursor < len(journal), "unresolved_reserved_policy")
        if journal[cursor]["event"] == "policy_failed_charged":
            require(step.get("proposal") is None and final == initial,
                    "failed_policy_started_worker")
            session.policy_failed()
        else:
            require(journal[cursor]["event"] == "validated_policy_decision", "validated_decision_missing")
            outcome = step.get("outcome")
            decision = journal[cursor]["decision"]
            require(outcome is not None and outcome["status"] == "completed"
                    and outcome["decision"] == decision, "saved_policy_decision_drift")
            valid_policy = outcome["audit"]
            if session.receive_decision(copy.deepcopy(decision)):
                proposed = step.get("proposal")
                session.finish_image(proposed, final)
            else:
                require(step.get("proposal") is None and final == initial,
                        "terminal_decision_started_worker")
    result = session.result()
    require(_digest(result) == _digest(selection), "sealed_selection_drift")
    require(session.ledger == final, "unobserved_or_uncharged_worker_suffix")
    emit({"event": "selection_sealed", "sha256": selection_sha256})
    require(cursor == len(journal), "events_after_terminal_selection")
    selected = next((item["row"] for item in session.observed
        if item["row"]["triple_candidate_id"] == result["selected_candidate_id"]), None)
    return {"selection": result, "policy_audit": valid_policy,
        "reference_raw_edges": copy.deepcopy(session.reference["raw_edge_readouts"]),
        "selected_raw_edges": copy.deepcopy(selected["raw_edge_readouts"]) if selected else None,
        "reference_report_model": session.reference["report_model_id"],
        "selected_report_model": selected["report_model_id"] if selected else None,
        "new_proposal_candidate_id": proposed["triple_candidate_id"] if proposed else None,
        "ledger_replay_pass": True, "policy_feedback_replay_pass": True,
        "new_image_own_report_chain_pass": proposed is not None,
        "clinical_repair_success": False, "clinical_accuracy": None}


def validate_summary(plan, summary, selections, books, fresh_rows, policy_audits):
    expected = {"fixed_ehr_cases": len(plan["cases"]), "completed_new_triplets": len(fresh_rows),
        "accepted_proxy_transitions": sum(s["accepted_proxy_transitions"] for s in selections),
        "charged_new_worker_attempts": sum(b["charged_model_attempts"] for b in books),
        "charged_policy_requests": sum(s["charged_policy_requests"] for s in selections),
        "historical_worker_attempts": plan["historical_worker_attempts"],
        "historical_policy_requests": plan["historical_policy_requests"], "external_api_calls": 0}
    require(all(type(summary[key]) is int and summary[key] == value for key, value in expected.items()),
            "summary_charge_or_count_drift")
    require(summary["schema_version"] == producer.VERSION
        and summary["status"] == "completed_one_image_probe_unvalidated"
        and summary["historical_cost_scope"] == "shared_sunk_not_measured_not_zero"
        and summary["clinical_accuracy"] is None and summary["measured_gpu_seconds"] is None
        and all(summary[key] is False for key in ("clinical_repair_success",
            "llm_superiority_demonstrated", "scorers_or_thresholds_changed",
            "training_performed", "independent_endpoint_used_for_selection")),
        "unsupported_clinical_cost_or_scope_claim")
    runtime = summary["runtime_seconds_including_load_io"]
    require(type(runtime) in (int, float) and math.isfinite(runtime) and runtime >= 0,
            "invalid_runtime_measurement")
    counts = postflight.phase_counts(books)
    require(counts.get("chexbert", 0) == len(fresh_rows)
        and counts.get("cxr_generator", 0) <= 2 and counts.get("report_generator", 0) <= 2
        and expected["charged_new_worker_attempts"] <= 8
        and expected["charged_policy_requests"] <= 2
        and len(policy_audits) <= expected["charged_policy_requests"], "bounded_completed_phase_inventory_required")
    return expected, counts


def comparison_csv(results):
    fields = ["case_id", "status", "reference_report_model", "selected_report_model",
        "charged_new_worker_attempts", "charged_policy_requests", "accepted_proxy_transitions"]
    metrics = ("known_reference_facts", "comparable_facts", "supported_positive",
        "supported_negative", "proxy_opposition_facts", "support_over_known", "coverage_over_known")
    fields += [role + "_" + edge + "_" + metric for role in ("reference", "selected")
               for edge in ("ehr_cxr", "ehr_report", "cxr_report") for metric in metrics]
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
    for result in results:
        row = {key: result["selection"].get(key) for key in fields[:7]}
        row.update(reference_report_model=result["reference_report_model"],
                   selected_report_model=result["selected_report_model"])
        for role in ("reference", "selected"):
            edges = result[role + "_raw_edges"]
            for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
                for metric in metrics:
                    row[role + "_" + edge + "_" + metric] = edges[edge][metric] if edges else None
        writer.writerow({key: "NA" if value is None else value for key, value in row.items()})
    return out.getvalue()


def audit_run(args):
    producer.gate.cpu_guard()  # Before metadata reads, directories or writes.
    pr, sr = postflight.MetadataReader(args.plan_run), postflight.MetadataReader(args.source_run)
    pr.hash(pr.root / "manifest.json", args.plan_manifest_sha256)
    pm = pr.json(pr.root / "manifest.json")
    require(pm["schema_version"] == producer.VERSION
        and pm["status"] == "prepared_cpu_only_gpu_not_submitted", "sealed_cpu_plan_required")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    plan = pr.json(pr.root / "plan.json")
    require(plan["schema_version"] == producer.VERSION and plan["policy"] == producer.POLICY
        and plan["planned_maximum"] == producer.MAXIMUM and len(plan["cases"]) == 2
        and plan["data_origin"] == "original_fully_synthetic_pool80"
        and plan["source_bodies_parsed"] is False
        and plan["source_manifest_sha256"] == producer.source.SOURCE_SHA
        and plan["source_audit_manifest_sha256"] == producer.source.AUDIT_SHA,
        "exact_authenticated_synthetic_scope_required")
    check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"])
    sr.hash(sr.root / "manifest.json", args.run_manifest_sha256)
    manifest = sr.json(sr.root / "manifest.json")
    require(manifest["schema_version"] == producer.VERSION
        and manifest["status"] == "completed_one_image_probe_unvalidated"
        and manifest["plan_manifest_sha256"] == args.plan_manifest_sha256
        and manifest["source_manifest_sha256"] == producer.source.SOURCE_SHA
        and manifest["clinical_acceptance"] is False and set(manifest["artifacts"]) == ARTIFACTS,
        "completed_hash_bound_run_required")
    require(not (sr.root / "failed_manifest.json").exists(), "failed_run_not_completed")
    for name, pin in manifest["artifacts"].items():
        sr.hash(sr.root / name, pin["sha256"])
    require(sr.json(sr.root / "start_manifest.json")["plan_manifest_sha256"] == args.plan_manifest_sha256,
            "start_manifest_plan_changed")
    rows = sr.json(sr.root / "score_rows.json")["records"]
    triples = sr.json(sr.root / "completed_triplets.json")
    selections = sr.json(sr.root / "selection.json")["records"]
    books = sr.json(sr.root / "execution_summary.json")["case_ledgers"]
    policy_audits = sr.json(sr.root / "policy_audits.json")["records"]
    summary = sr.json(sr.root / "summary.json")
    ids = [case["case_id"] for case in plan["cases"]]
    require(len(set(ids)) == 2 and [s["case_id"] for s in selections] == ids
        and [b["case_id"] for b in books] == ids, "case_inventory_or_order_drift")
    cached = [row for case in plan["cases"] for row in case["cached_rows"]]
    require(rows[:len(cached)] == cached and len(cached) == 8
        and len({row["triple_candidate_id"] for row in rows}) == len(rows)
        and all(row["case_id"] in ids for row in rows), "historical_or_fresh_inventory_drift")
    fresh = rows[len(cached):]
    require(len(fresh) <= 2 and triples["cached_references"] == plan["cached_triplets"]
        and len(triples["new_records"]) == len(fresh), "exact_cached_and_new_artifact_inventory_required")
    expected_export = producer.selected_triplets(rows, plan["cached_triplets"] + triples["new_records"],
        selections, fresh_ids={row["triple_candidate_id"] for row in fresh})
    require(sr.json(sr.root / "selected_triplets.json")["records"] == expected_export,
            "selected_artifact_export_drift")
    results, expected_audits = [], []
    qpins = {Path(path).name: pin["sha256"] for path, pin in plan["qwen"]["asset_pins"].items()}
    for case, book, selection in zip(plan["cases"], books, selections):
        require(book["execution_mode"] == "approved_slurm_backend"
            and book["call_budget"] == 4 and book["max_retries"] == 0 and book["pending_attempts"] == 0,
            "completed_actual_no_retry_ledger_required")
        cid = case["case_id"]
        new_rows = [r for r in fresh if r["case_id"] == cid]
        require(len(new_rows) <= 1, "single_image_probe_per_case_required")
        policy_root = sr.root / "policy" / cid
        selection_path = policy_root / "sealed_selection.json"
        require(sr.json(selection_path) == selection, "per_case_selection_drift")
        step = None
        sp = policy_root / "step_0"
        if selection["charged_policy_requests"]:
            state = sr.json(sp / "numeric_state.json")
            rp = sp / "policy_result.json"
            step = {"state": state, "outcome": sr.json(rp) if rp.exists() else None,
                "proposal": new_rows[0] if new_rows else None}
        else:
            require(not sp.exists(), "unreserved_policy_metadata_present")
        ctx = producer.gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
        result = replay_case(case, ctx, book, sr.journal(policy_root / "live_policy.journal.jsonl"),
                             step, selection, sr.hash(selection_path))
        results.append(result)
        if result["policy_audit"] is not None:
            postflight.successful_policy(step["outcome"], sr.hash(sp / "numeric_state.json"), qpins)
            expected_audits.append(result["policy_audit"])
        cr = sr.root / "cases" / cid
        if book["charged_model_attempts"]:
            require(sr.json(cr / "ledger_snapshot.json") == book
                and sr.journal(cr / "execution.journal.jsonl") == book["events"],
                "worker_snapshot_or_durable_journal_drift")
            anchor = producer.gate.anchor_from_record(case["anchor"])
            require(sr.json(cr / "ehr_anchor.json") == anchor.record(), "case_anchor_changed")
            sr.hash(cr / "inputs/synthetic_ehr.json", anchor.ehr_sha256)
            sr.hash(cr / "inputs/ehr_facts.json", anchor.ehr_facts_sha256)
            sr.hash(cr / "inputs/cxr_prompts/roentgen_v2.txt",
                    case["requests"][0]["request"]["inputs"]["final_prompt"]["sha256"])
        else:
            require(not cr.exists(), "unrequested_worker_directory_present")
        requests = {event["request"]["operation_id"]: event["request"]
                    for event in book["events"] if event["event"] == "attempt_reserved"}
        for request in requests.values():
            spec = plan["workers"][request["model_id"]]
            require(request["frozen_model_audit_sha256"] == _digest(spec)
                and request["kind"] == spec["kind"] and request["seed"] == 1,
                "frozen_worker_identity_or_seed_drift")
        for event in book["events"]:
            if event["event"] != "attempt_completed": continue
            request = requests[event["operation_id"]]
            if request["kind"] in ("xrv", "chexbert"):
                name = "cxr_finding_labels.json" if request["kind"] == "xrv" else "report_finding_labels.json"
                sr.hash(cr / "operations" / (request["operation_id"] + "_a1") / "scored" / name,
                        event["result"]["output_artifact_sha256"])
        for row in new_rows:
            matches = [t for t in triples["new_records"] if t["case_id"] == cid]
            require(len(matches) == 1, "one_new_triplet_per_case_receipt_required")
            triple = matches[0]
            require(all(triple[key] == row[key] for key in ("ehr_sha256", "ehr_facts_sha256",
                "cxr_sha256", "report_sha256", "cxr_model_id", "report_model_id", "seed", "raw_edge_readouts"))
                and triple["clinical_acceptance"] is False
                and Path(triple["ehr_path"]) == cr / "inputs/synthetic_ehr.json",
                "new_triplet_receipt_lineage_drift")
            for field in ("cxr", "report", "receipt"):
                sr.hash(triple[field + "_path"], triple[field + "_sha256"])
            require(sr.json(triple["receipt_path"]) == row["receipt"], "stored_fresh_receipt_drift")
    require(policy_audits == expected_audits, "successful_policy_audit_inventory_drift")
    counts, phases = validate_summary(plan, summary, selections, books, fresh, policy_audits)
    expected_csv = score_csv([{**r, "verification_status": r["receipt"]["verification_status"]} for r in rows])
    sr.hash(sr.root / "score_table.csv", hashlib.sha256(expected_csv.encode("utf-8")).hexdigest())
    pr.recheck(); sr.recheck(); check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"])
    report = {"schema_version": VERSION, "status": "metadata_and_ledger_verified_not_clinical",
        "source_manifest_sha256": args.run_manifest_sha256, "plan_manifest_sha256": args.plan_manifest_sha256,
        "counts": counts, "completed_worker_phases": phases, "case_results": results,
        "new_model_calls": 0, "clinical_accuracy": None, "clinical_repair_success": False,
        "ehr_report_bodies_parsed": False, "image_pixels_decoded": False,
        "model_outputs_independently_recomputed": False, "not_a_clinical_efficacy_benchmark": True,
        "source_pins": {**plan["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
            str(producer.ROOT / "TriCompose-v1.2/tests/test_fresh_cxr_postflight.py"):
                sha256_file(producer.ROOT / "TriCompose-v1.2/tests/test_fresh_cxr_postflight.py")},
        "artifact_pins": {**plan["artifact_pins"], **pr.pins, **sr.pins}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "audit.json", report)
        table = write_private_text(temporary / "score_comparison.csv", comparison_csv(results))
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": report["status"], "artifacts": {name.name: {"sha256": sha256_file(name)} for name in (p, table)},
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("run-manifest-sha256", "plan-manifest-sha256", "run-id"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        target = audit_run(args)
        print(json.dumps({"stage": VERSION, "status": "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

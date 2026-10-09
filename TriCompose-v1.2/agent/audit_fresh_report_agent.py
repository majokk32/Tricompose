#!/usr/bin/env python3
"""Metadata-only postflight for a completed fresh report-only bridge.

Replays the consumed coordinator and existing receipt/ledger checks. Never
loads models, decodes pixels, parses EHR/report bodies, changes thresholds,
submits jobs, or resumes an incomplete run. Authored fixtures test the pure
replay separately; the CLI requires an existing CPU Slurm cgroup.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "TriCompose-v1.0/src", "TriCompose-v1.1/src",
                 "TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks",
                 "TriCompose-v1.2/tools", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT / relative))

from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, require_inside, sha256_file, write_private_json, write_private_text)
import fresh_output_acceptance as gate
from tricompose_v12.execution_ledger import restore_ledger
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import score_csv
from tricompose_v12.live_receipts import PROFILE
from tricompose_llm.live_report_bridge import FreshReportSession, VERSION as BRIDGE

VERSION = "tricompose-fresh-report-postflight-v1"
SUPPLEMENT = "tricompose-fresh-report-supplemental-assets-v2"
ARTIFACTS = {"selection.json", "score_rows.json", "score_table.csv",
    "completed_triplets.json", "execution_summary.json", "policy_audits.json", "summary.json"}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def replay_book(book, events=None):
    """Hash-chain/arithmetic replay; no backend call, no prospective writes."""
    ledger = restore_ledger(book["events"] if events is None else events,
        case_id=book["case_id"], ehr_anchor_sha256=book["ehr_anchor_sha256"],
        call_budget=book["call_budget"], max_retries=book["max_retries"],
        execution_mode=book["execution_mode"], sink=lambda event: None)
    restored = ledger.snapshot()
    if events is None:
        require(_digest(restored) == _digest(book), "ledger_snapshot_drift")
    return restored


def replay_session(rows, context, initial, final, journal, steps, sealed,
                   selection_sha256):
    """Replay only chronologically completed rows; never expose future rows.

    `steps` is numeric state/outcome/continuation metadata keyed by ordinal.
    Every emitted event must match the recorded journal including veto details.
    Failures keep their reserved charge, but are not clinical contradictions.
    """
    initial = replay_book(initial)
    final = replay_book(final)
    require(final["events"][:len(initial["events"])] == initial["events"],
            "initial_execution_prefix_changed")
    require(rows and len({r["triple_candidate_id"] for r in rows}) == len(rows),
            "unique_completed_rows_required")
    for row in rows:
        gate.validate_observation(gate.controller_view(row), context)
        gate.completed_operation(row, final)
    ordered = sorted(rows, key=lambda r: gate.completed_operation(r, final))
    require(ordered == rows, "candidate_chronology_changed")
    require(gate.completed_operation(rows[0], initial) < len(initial["events"]),
            "completed_initial_baseline_required")
    cursor = 0

    def emit(expected):
        nonlocal cursor
        require(cursor < len(journal) and _digest(journal[cursor]) == _digest(expected),
                "policy_journal_replay_drift")
        cursor += 1

    session = FreshReportSession(rows[0], context, initial, max_steps=3, sink=emit)
    used_steps, observed = set(), [rows[0]["triple_candidate_id"]]
    successful_policies = []
    while True:
        state = session.begin_planning()
        if state is None:
            break
        index = session.policy_requests - 1
        require(index in steps, "reserved_policy_metadata_missing")
        used_steps.add(index)
        step = steps[index]
        require(_digest(step["state"]) == _digest(state), "policy_observed_future_or_altered_evidence")
        require(cursor < len(journal), "reserved_policy_unresolved")
        if journal[cursor]["event"] == "policy_failed_charged":
            require(step.get("continuation") is None, "failed_policy_started_worker")
            session.policy_failed()
            break
        require(journal[cursor]["event"] == "validated_policy_decision",
                "decision_event_required")
        decision = journal[cursor]["decision"]
        outcome = step.get("outcome")
        require(outcome is not None and outcome["status"] == "completed"
            and outcome["decision"] == decision, "saved_policy_decision_drift")
        successful_policies.append(outcome["audit"])
        model = session.receive_decision(copy.deepcopy(decision))
        if model is None:
            require(step.get("continuation") is None, "terminal_decision_started_worker")
            break
        continuation = step.get("continuation")
        require(continuation is not None, "requested_worker_cost_missing")
        book = replay_book(continuation)
        require(book["events"] == final["events"][:len(book["events"])],
                "continuation_not_final_ledger_prefix")
        before = len(session.ledger["events"])
        after = len(book["events"])
        new_rows = [r for r in rows if before <= gate.completed_operation(r, final) < after]
        require(len(new_rows) <= 1, "multiple_rows_for_one_report_action")
        proposal = new_rows[0] if new_rows else None
        session.finish_report(proposal, book)
        if proposal is not None:
            observed.append(proposal["triple_candidate_id"])
    result = session.result()
    require(_digest(result) == _digest(sealed), "sealed_selection_replay_drift")
    require(_digest(session.ledger) == _digest(final), "uncharged_or_unobserved_worker_suffix")
    require(observed == [r["triple_candidate_id"] for r in rows],
            "unobserved_candidate_in_inventory")
    require(used_steps == set(steps), "unreserved_policy_metadata_present")
    emit({"event": "selection_sealed", "sha256": selection_sha256})
    require(cursor == len(journal), "events_after_terminal_selection")
    selected = next((r for r in rows if r["triple_candidate_id"] == result["selected_candidate_id"]), None)
    return {"selection": result, "successful_policy_audits": successful_policies,
        "baseline_raw_edges": rows[0]["raw_edge_readouts"],
        "selected_raw_edges": selected["raw_edge_readouts"] if selected else None,
        "selected_report_model": selected["report_model_id"] if selected else None,
        "known_ehr_facts": rows[0]["raw_edge_readouts"]["ehr_cxr"]["known_reference_facts"],
        "unresolved_ehr_image_proxy_oppositions": rows[0]["raw_edge_readouts"]["ehr_cxr"]["proxy_opposition_facts"],
        "ledger_replay_pass": True, "policy_feedback_replay_pass": True,
        "clinical_repair_success": False, "clinical_accuracy": None}


class MetadataReader:
    """Bounded, protected metadata reads; body reads only via byte hashing."""

    def __init__(self, root):
        self.root = require_inside(root, PROTECTED_ROOT, must_exist=True)
        self.pins = {}

    def path(self, path):
        p = require_inside(path, self.root, must_exist=True)
        require(p.is_file() and stat.S_IMODE(p.stat().st_mode) == 0o660
            and p.stat().st_gid in (96293, 65534), "protected_file_mode_or_group_changed")
        for parent in (p.parent, *p.parent.parents):
            if not parent.is_relative_to(self.root):
                break
            require(stat.S_IMODE(parent.stat().st_mode) == 0o2770
                and parent.stat().st_gid in (96293, 65534), "protected_directory_mode_or_group_changed")
        return p

    def hash(self, path, expected=None):
        p = self.path(path)
        digest = sha256_file(p)
        require(expected is None or digest == expected, "authenticated_artifact_changed")
        self.pins[str(p)] = digest
        return digest

    def json(self, path):
        p = self.path(path)
        require(p.stat().st_size <= 16 * 1024 * 1024, "metadata_size_limit")
        before = self.hash(p)
        result = json.loads(p.read_text(encoding="utf-8"))
        require(isinstance(result, dict), "metadata_object_required")
        self.hash(p, before)
        return result

    def journal(self, path):
        p = self.path(path)
        require(p.stat().st_size <= 16 * 1024 * 1024, "journal_size_limit")
        before = self.hash(p)
        lines = p.read_text(encoding="utf-8").splitlines()
        require(all(line.strip() for line in lines), "incomplete_journal_line")
        result = [json.loads(line) for line in lines]
        require(all(isinstance(row, dict) for row in result), "journal_objects_required")
        self.hash(p, before)
        return result

    def recheck(self):
        for p, digest in list(self.pins.items()):
            self.hash(p, digest)


def successful_policy(outcome, state_sha256, qwen_pins):
    audit = outcome["audit"]
    require(outcome["status"] == "completed" and outcome["state_sha256"] == state_sha256
        and audit["frozen"] is True and audit["asset_pins"] == qwen_pins
        and audit["local_attempts"] == 1 and audit["generate_attempts"] == 1
        and not any(audit["failure_counts"].values()), "pinned_successful_policy_required")


def phase_counts(books):
    counts = Counter()
    for book in books:
        requests = {e["request"]["operation_id"]: e["request"] for e in book["events"]
            if e["event"] == "attempt_reserved"}
        counts.update(requests[e["operation_id"]]["kind"] for e in book["events"]
            if e["event"] == "attempt_completed")
    return dict(counts)


def comparison_csv(results):
    """Unweighted before/after readouts; an absent denominator remains NA."""
    metrics = ("known_reference_facts", "comparable_facts", "supported_positive",
        "supported_negative", "proxy_opposition_facts", "support_over_known", "coverage_over_known")
    columns = ["case_id", "status", "selected_report_model", "charged_worker_attempts",
        "charged_policy_requests", "accepted_proxy_transitions"]
    columns += [role + "_" + edge + "_" + metric for role in ("baseline", "selected")
        for edge in ("ehr_cxr", "ehr_report", "cxr_report") for metric in metrics]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for result in results:
        selection = result["selection"]
        row = {name: selection.get(name) for name in columns[:6]}
        row["selected_report_model"] = result.get("selected_report_model")
        for role in ("baseline", "selected"):
            edges = result.get(role + "_raw_edges")
            for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
                for metric in metrics:
                    row[role + "_" + edge + "_" + metric] = edges[edge][metric] if edges else None
        writer.writerow({key: "NA" if value is None else value for key, value in row.items()})
    return output.getvalue()


def audit_run(args):
    gate.cpu_guard()  # Actual existing CPU cgroup, before inputs or writes.
    source = MetadataReader(args.source_run)
    preflight = MetadataReader(args.plan_run)
    preflight.hash(preflight.root / "manifest.json", args.plan_manifest_sha256)
    pm = preflight.json(preflight.root / "manifest.json")
    require(pm.get("schema_version") == BRIDGE and pm.get("supplemental_asset_version") == SUPPLEMENT,
            "reviewed_supplemental_preflight_required")
    preflight.hash(preflight.root / "plan.json", pm["plan_sha256"])
    plan = preflight.json(preflight.root / "plan.json")
    require(plan.get("data_origin") == "original_fully_synthetic_pool80"
        and plan.get("source_bodies_parsed") is False and plan.get("profile") == PROFILE
        and plan.get("schema_version") == BRIDGE and len(plan["cases"]) == 2,
        "authenticated_fully_synthetic_plan_required")
    # Source bytes only. No checkpoint reload / repeated whole-weight scan.
    for path, digest in plan["source_pins"].items():
        p = require_inside(path, ROOT, must_exist=True)
        require(sha256_file(p) == digest, "consumed_source_changed")
    source.hash(source.root / "manifest_v2.json", args.run_manifest_sha256)
    v2 = source.json(source.root / "manifest_v2.json")
    require(v2["schema_version"] == SUPPLEMENT and v2["clinical_acceptance"] is False
        and v2["status"] == "completed_fresh_report_bridge_contract_unvalidated",
            "completed_supplemental_run_required")
    source.hash(source.root / "manifest.json", v2["base_manifest_sha256"])
    manifest = source.json(source.root / "manifest.json")
    require(manifest["schema_version"] == BRIDGE
        and manifest["status"] == "completed_fresh_report_bridge_unvalidated"
        and manifest["plan_manifest_sha256"] == args.plan_manifest_sha256
        and manifest["clinical_acceptance"] is False and set(manifest["artifacts"]) == ARTIFACTS,
        "completed_run_plan_and_artifact_bindings_required")
    for name, pin in manifest["artifacts"].items():
        source.hash(source.root / name, pin["sha256"])
    source.hash(source.root / "execution_verification_v2.json", v2["execution_verification_sha256"])
    verification = source.json(source.root / "execution_verification_v2.json")
    summary = source.json(source.root / "summary.json")
    require(source.json(source.root / "start_manifest.json")["plan_manifest_sha256"] == args.plan_manifest_sha256,
            "start_manifest_plan_drift")
    rows = source.json(source.root / "score_rows.json")["records"]
    triples = source.json(source.root / "completed_triplets.json")["records"]
    selections = source.json(source.root / "selection.json")["records"]
    books = source.json(source.root / "execution_summary.json")["case_ledgers"]
    audits = source.json(source.root / "policy_audits.json")["records"]
    ids = [c["case_id"] for c in plan["cases"]]
    require(len(set(ids)) == 2 and [s["case_id"] for s in selections] == ids
        and [b["case_id"] for b in books] == ids and len(rows) == len(triples),
        "fixed_case_order_and_candidate_inventory_required")
    require(all(r["case_id"] in ids for r in rows), "unexpected_candidate_case")
    qpins = {Path(p).name: v["sha256"] for p, v in plan["qwen"]["asset_pins"].items()}
    results, valid_audits = [], []
    for case, final, selection in zip(plan["cases"], books, selections):
        anchor = gate.anchor_from_record(case["anchor"])
        require(final["execution_mode"] == "approved_slurm_backend"
            and final["call_budget"] == 10 and final["max_retries"] == 0
            and final["pending_attempts"] == 0, "completed_bounded_actual_execution_required")
        replay_book(final)
        cr = source.root / "cases" / anchor.case_id
        require(source.json(cr / "ehr_anchor.json") == anchor.record(), "fixed_anchor_changed")
        source.hash(cr / "inputs/synthetic_ehr.json", anchor.ehr_sha256)
        source.hash(cr / "inputs/ehr_facts.json", anchor.ehr_facts_sha256)
        source.hash(cr / "inputs/cxr_prompts/roentgen_v2.txt",
            case["requests"][0]["request"]["inputs"]["final_prompt"]["sha256"])
        initial = source.json(cr / "ledger_snapshot.json")
        require(source.journal(cr / "execution.journal.jsonl") == initial["events"],
                "initial_durable_journal_drift")
        replay_book(initial)
        case_rows = [r for r in rows if r["case_id"] == anchor.case_id]
        require(all(e["request"]["frozen_model_audit_sha256"] == _digest(plan["workers"][e["request"]["model_id"]])
            and e["request"]["kind"] == plan["workers"][e["request"]["model_id"]]["kind"]
            and e["request"]["seed"] == case["requests"][0]["request"]["seed"]
            for e in final["events"] if e["event"] == "attempt_reserved"), "frozen_worker_audit_drift")
        if not case_rows:
            require(final == initial and selection == {"case_id": anchor.case_id,
                "status": "abstain_baseline_chain_incomplete", "selected_candidate_id": None,
                "charged_worker_attempts": final["charged_model_attempts"], "charged_policy_requests": 0,
                "clinical_repair_success": False} and final["failed_attempts"] >= 1,
                "incomplete_baseline_not_abstained_or_cost_hidden")
            require(not (cr / "live_policy.journal.jsonl").exists(), "incomplete_baseline_started_policy")
            results.append({"selection": selection, "ledger_replay_pass": True,
                "clinical_repair_success": False, "clinical_accuracy": None})
            continue
        context = gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
        journal = source.journal(cr / "live_policy.journal.jsonl")
        steps = {}
        for index in range(selection["charged_policy_requests"]):
            sr = cr / f"policy_step_{index}"
            state = source.json(sr / "numeric_state.json")
            rp = sr / "policy_result.json"
            outcome = source.json(rp) if rp.exists() else None
            cp = cr / f"continuation_{index}_snapshot.json"
            continuation = source.json(cp) if cp.exists() else None
            if continuation is not None:
                prior = initial if index == 0 else steps[index-1]["continuation"]
                require(prior is not None and source.journal(cr / f"continuation_{index}.journal.jsonl")
                    == continuation["events"][len(prior["events"]):], "continuation_journal_drift")
            validated = [e for e in journal if e["event"] == "validated_policy_decision"
                and e["decision"]["step_id"] == state["step_id"]]
            if validated:
                require(len(validated) == 1 and outcome is not None, "validated_policy_output_missing")
                successful_policy(outcome, source.hash(sr / "numeric_state.json"), qpins)
            steps[index] = {"state": state, "outcome": outcome, "continuation": continuation}
        sp = cr / "sealed_selection.json"
        require(source.json(sp) == selection, "global_and_per_case_selection_drift")
        replay = replay_session(case_rows, context, initial, final, journal, steps, selection, source.hash(sp))
        results.append(replay)
        valid_audits.extend(replay["successful_policy_audits"])
        for row in case_rows:
            matches = [t for t in triples if t["case_id"] == row["case_id"]
                and t["report_sha256"] == row["report_sha256"] and t["report_model_id"] == row["report_model_id"]]
            require(len(matches) == 1, "one_artifact_triplet_per_receipt_required")
            triple = matches[0]
            require(all(triple[k] == row[k] for k in ("ehr_sha256", "ehr_facts_sha256",
                "cxr_sha256", "report_sha256", "cxr_model_id", "report_model_id", "seed", "raw_edge_readouts"))
                and triple["clinical_acceptance"] is False, "artifact_triplet_receipt_drift")
            require(Path(triple["ehr_path"]) == cr / "inputs/synthetic_ehr.json", "fixed_ehr_artifact_path_drift")
            for field in ("cxr", "report", "receipt"):
                p = require_inside(triple[field + "_path"], cr / "operations", must_exist=True)
                source.hash(p, triple[field + "_sha256"])
            require(source.json(triple["receipt_path"]) == row["receipt"], "stored_receipt_drift")
        for event in final["events"]:
            if event["event"] != "attempt_completed":
                continue
            request = next(e["request"] for e in final["events"]
                if e["event"] == "attempt_reserved" and e["request"]["operation_id"] == event["operation_id"])
            if request["kind"] in ("xrv", "chexbert"):
                name = "cxr_finding_labels.json" if request["kind"] == "xrv" else "report_finding_labels.json"
                source.hash(cr / "operations" / (request["operation_id"] + "_a1") / "scored" / name,
                    event["result"]["output_artifact_sha256"])
    counts = phase_counts(books)
    require(counts.get("chexbert", 0) == len(rows)
        and counts.get("cxr_generator", 0) <= 2 and counts.get("report_generator", 0) <= 8,
        "completed_artifact_inventory_or_budget_drift")
    expected_csv = score_csv([{**r, "verification_status": r["receipt"]["verification_status"]} for r in rows])
    require(source.hash(source.root / "score_table.csv") == hashlib.sha256(expected_csv.encode("utf-8")).hexdigest(),
        "score_table_does_not_match_validated_numeric_rows")
    require(audits == valid_audits, "successful_policy_audit_inventory_drift")
    expected_counts = {"fixed_ehr_cases": len(ids), "completed_report_candidates": len(rows),
        "accepted_proxy_transitions": sum(s.get("accepted_proxy_transitions", 0) for s in selections),
        "charged_generator_verifier_attempts": sum(b["charged_model_attempts"] for b in books),
        "charged_policy_requests": sum(s["charged_policy_requests"] for s in selections),
        "completed_qwen_decisions": len(audits)}
    require(all(type(summary[k]) is int and summary[k] == v for k, v in expected_counts.items()),
        "summary_cost_or_candidate_count_drift")
    require(summary["status"] == "completed_fresh_report_bridge_unvalidated"
        and summary["clinical_repair_success"] is False and summary["clinical_accuracy"] is None
        and summary["training_performed"] is False and summary["scorers_or_thresholds_changed"] is False
        and summary["image_regeneration_installed"] is False and summary["external_api_calls"] == 0,
        "unsupported_clinical_or_action_scope_claim")
    require(verification["schema_version"] == SUPPLEMENT
        and verification["base_manifest_sha256"] == v2["base_manifest_sha256"]
        and verification["plan_manifest_sha256"] == args.plan_manifest_sha256
        and verification["confirmed_completed_worker_phases"] == counts
        and verification["charged_worker_attempts"] == expected_counts["charged_generator_verifier_attempts"]
        and verification["confirmed_new_cxr_artifacts"] == counts.get("cxr_generator", 0)
        and verification["confirmed_new_report_artifacts"] == counts.get("report_generator", 0)
        and verification["completed_scored_report_candidates"] == len(rows)
        and verification["fresh_generation_confirmed_by_completed_receipts"]
            == bool(counts.get("cxr_generator", 0) or counts.get("report_generator", 0))
        and verification["clinical_repair_success"] is False
        and verification["training_performed"] is False, "confirmed_phase_count_drift")
    source.recheck(); preflight.recheck()
    for path, digest in plan["source_pins"].items():
        require(sha256_file(path) == digest, "consumed_source_changed_during_audit")
    report = {"schema_version": VERSION, "status": "completed_metadata_contract_audit",
        "source_manifest_v2_sha256": args.run_manifest_sha256,
        "plan_manifest_sha256": args.plan_manifest_sha256,
        "counts": expected_counts, "confirmed_completed_worker_phases": counts,
        "failed_worker_attempts": sum(b["failed_attempts"] for b in books),
        "case_results": results, "metadata_and_byte_hash_checks_pass": True,
        "ehr_report_bodies_parsed": False, "image_pixels_decoded": False,
        "new_model_calls": 0, "clinical_repair_success": False, "clinical_accuracy": None,
        "not_a_clinical_efficacy_benchmark": True, "measured_gpu_seconds": None,
        "source_pins": {str(Path(__file__).resolve()): sha256_file(__file__),
            str(ROOT / "TriCompose-v1.2/tests/test_fresh_report_postflight.py"):
                sha256_file(ROOT / "TriCompose-v1.2/tests/test_fresh_report_postflight.py"),
            **plan["source_pins"]},
        "artifact_pins": {**source.pins, **preflight.pins}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "audit.json", report)
        table = write_private_text(temporary / "score_comparison.csv", comparison_csv(results))
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": report["status"], "audit_sha256": sha256_file(p), "new_model_calls": 0,
            "artifacts": {p.name: {"sha256": sha256_file(p)}, table.name: {"sha256": sha256_file(table)}},
            "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--plan-run", type=Path, required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        target = audit_run(args)
        print(json.dumps({"stage": VERSION, "status": "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "audit_failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

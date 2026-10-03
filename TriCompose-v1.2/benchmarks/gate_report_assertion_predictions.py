#!/usr/bin/env python3
"""CPU-only blinded gate, then separate authored-key risk/coverage analysis.

This is a post-hoc development diagnostic, NOT clinical validation. It never
changes the frozen extractor, the original predictions, or primary selection.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_assertions import FINDINGS, GATE_VERSION, gate_assertion, crosscheck_context
from contracts import (
    PROTECTED_ROOT, require_inside, read_json, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text,
)
from report_assertion_challenge import load_inputs, load_references, SCORE_SCHEMA, FROZEN_GUARD_SHA

SCHEMA = "tricompose-report-scope-gate-diagnostic-v1"
ANALYSIS_SCHEMA = "tricompose-report-scope-gate-analysis-v1"


def checked_predictions(root, source):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    manifest, path = read_json(root/"manifest.json"), root/"predictions.json"
    if manifest.get("schema_version") != SCORE_SCHEMA or sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("prediction artifact binding differs")
    payload = read_json(path)
    if (payload.get("source") != source or payload.get("scorer_read_reference_key") is not False
            or payload.get("model_received_reference_states") is not False
            or payload["producer"].get("frozen") is not True
            or payload.get("primary_metric_eligible") is not False):
        raise ValueError("blinded frozen producer differs")
    return payload, path


def gate(args):
    from repair_cached_report_evidence import scope_check
    if sha256_file(Path(scope_check.__code__.co_filename)) != FROZEN_GUARD_SHA:
        raise ValueError("frozen pre-challenge guard changed")
    resolver, texts, source = load_inputs(args.bank_run)
    payload, prediction_path = checked_predictions(args.score_run, source)
    if payload["producer"]["model_id"] != "chexbert":
        raise ValueError("baseline must be frozen CheXbert")
    predictions = {row["item_id"]: row for row in payload["records"]}
    if len(predictions) != len(payload["records"]) or set(predictions) != {row["item_id"] for row in resolver}:
        raise ValueError("prediction/resolver inventory differs")
    context, sources = {}, {"predictions": prediction_path}
    if args.context_run:
        other, other_path = checked_predictions(args.context_run, source)
        if other["producer"]["model_id"] != "medspacy_context" or other.get("official_rules_modified") is not False:
            raise ValueError("not an unmodified official ConText run")
        context = {row["item_id"]: row for row in other["records"]}
        if len(context) != len(other["records"]) or set(context) != set(predictions):
            raise ValueError("context inventory differs")
        sources["context_predictions"] = other_path
    records = []
    for row in resolver:
        pred = predictions[row["item_id"]]
        if pred["report_sha256"] != row["report_sha256"] or pred["status"] != "complete" or set(pred["finding_states"]) != set(FINDINGS):
            raise ValueError("prediction/source/state inventory differs")
        report = texts[row["report_sha256"]]
        other = context.get(row["item_id"])
        if other and (other["report_sha256"] != row["report_sha256"] or other["status"] != "complete"):
            raise ValueError("context/source binding differs")
        for finding in FINDINGS:
            result = gate_assertion(report, finding, pred["finding_states"][finding], scope_check)
            if other:
                result = crosscheck_context(result, other["finding_evidence"][finding], report)
            records.append({"item_id": row["item_id"], **result})
    return {"schema_version": SCHEMA, "source": source, "gate_version": GATE_VERSION,
        "frozen_guard_sha256": FROZEN_GUARD_SHA, "records": records,
        "counts": dict(Counter(row["decision"] for row in records)),
        "reference_key_read_by_gate": False, "primary_metric_eligible": False,
        "selection_changed": False, "regeneration_authorized": False,
        "model_calls": 0, "context_crosscheck_used": bool(context)}, sources


def selective_counts(rows):
    committed = [row for row in rows if row["decision"] == "scope_commit"]
    correct = sum(row["gated_state"] == row["expected"] for row in committed)
    raw_wrong = [row for row in rows if row["raw_state"] != row["expected"]]
    return {"checks": len(rows), "raw_exact_matches": sum(row["raw_state"] == row["expected"] for row in rows),
        "scope_commits": len(committed), "commit_coverage": None if not rows else len(committed)/len(rows),
        "correct_commits": correct, "incorrect_commits": len(committed)-correct,
        "conditional_authored_match": None if not committed else correct/len(committed),
        "noncommitted_checks": len(rows)-len(committed),
        "raw_errors_not_committed": sum(row["decision"] != "scope_commit" for row in raw_wrong),
        "raw_correct_not_committed": sum(row["raw_state"] == row["expected"] and row["decision"] != "scope_commit" for row in rows),
        "hard_flips_committed": sum({row["expected"], row["gated_state"]} == {"positive", "negative"} for row in committed),
        "unsafe_unknown_or_uncertain_commits": sum(row["expected"] in {"unknown", "uncertain"} and row["gated_state"] in {"positive", "negative"} for row in committed),
        "decision_counts": dict(Counter(row["decision"] for row in rows))}


def analyze(args):
    resolver, _, source = load_inputs(args.bank_run)
    root = require_inside(args.gate_run, PROTECTED_ROOT, must_exist=True)
    path, manifest = root/"evidence.json", read_json(root/"manifest.json")
    if manifest.get("schema_version") != SCHEMA or sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("gate artifact binding differs")
    payload = read_json(path)
    if (payload["source"] != source or payload.get("reference_key_read_by_gate") is not False
            or payload.get("gate_version") != GATE_VERSION or payload.get("frozen_guard_sha256") != FROZEN_GUARD_SHA
            or payload.get("primary_metric_eligible") is not False or payload.get("regeneration_authorized") is not False):
        raise ValueError("gate provenance differs")
    refs, key_path = load_references(args.bank_run, resolver)
    index = {(row["item_id"], row["finding"]): row for row in payload["records"]}
    if len(index) != len(payload["records"]) or set(index) != {(row["item_id"], finding) for row in resolver for finding in FINDINGS}:
        raise ValueError("gate/source inventory differs")
    rows = []
    for ref in refs:
        for finding in ref["evaluation_findings"]:
            item = index[ref["item_id"], finding]
            if item["report_sha256"] != ref["report_sha256"]:
                raise ValueError("gate/reference source binding differs")
            if (item["decision"] not in {"scope_commit", "abstain", "no_model_assertion"}
                    or (item["decision"] == "scope_commit" and (item["state"] == "unknown" or item["state"] != item["proposed_state"] or item["scope_verified"] is not True))
                    or (item["decision"] != "scope_commit" and item["state"] != "unknown")):
                raise ValueError("invalid gate transition")
            rows.append({"item_id": ref["item_id"], "finding": finding, "family": ref["family"],
                "expected": ref["expected_states"][finding], "raw_state": item["proposed_state"],
                "gated_state": item["state"], "decision": item["decision"], "reason": item["reason"]})
    return {"schema_version": ANALYSIS_SCHEMA, "source": source, "designated_targets": selective_counts(rows),
        "per_finding": {name: selective_counts([row for row in rows if row["finding"] == name]) for name in FINDINGS},
        "per_family": {name: selective_counts([row for row in rows if row["family"] == name]) for name in sorted({row["family"] for row in rows})},
        "primary_metric_eligible": False, "selection_changed": False, "regeneration_authorized": False,
        "independent_clinical_accuracy": None, "posthoc_development_diagnostic": True,
        "interpretation": "Gate designed after observing this authored challenge. Conditional match is not overall accuracy or independent clinical validation. Noncommitted checks include missing model assertions and abstentions; missing is not negative."}, rows, {"gated_evidence": path, "authored_references": key_path}


def decision_csv(records):
    fields = ("item_id", "finding", "report_sha256", "proposed_state", "state", "decision", "reason", "scope_verified", "review_action")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in records)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "analyze"), required=True)
    for name in ("bank-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("score-run", "context-run", "gate-run"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        if args.mode == "gate":
            payload, sources = gate(args)
            files = [write_private_json(temporary/"evidence.json", payload), write_private_text(temporary/"decision_table.csv", decision_csv(payload["records"]))]
            schema = SCHEMA
        else:
            payload, details, sources = analyze(args)
            files = [write_private_json(temporary/"summary.json", payload), write_private_json(temporary/"details.json", {"records": details})]
            schema = ANALYSIS_SCHEMA
        library = Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_assertions.py"
        write_private_json(temporary/"manifest.json", {"schema_version": schema, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "assertion_library_sha256": sha256_file(library),
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "artifacts": {str(path.relative_to(temporary)): {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_report_scope_"+args.mode, "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

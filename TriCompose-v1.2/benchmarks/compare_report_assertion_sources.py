#!/usr/bin/env python3
"""Blinded source comparison, then separate post-hoc authored diagnostics.

The literal readout uses the unchanged pre-challenge scope checker, without
requiring a model proposal to match it. It proposes a separate source state;
it never overwrites model labels or calls a disagreed report/image erroneous.
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
from tricompose_v12.report_assertions import FINDINGS, CHECKED_REASONS, checked_span
from contracts import (
    PROTECTED_ROOT, require_inside, read_json, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text,
)
from report_assertion_challenge import load_inputs, load_references, FROZEN_GUARD_SHA, evaluate_predictions
from gate_report_assertion_predictions import checked_predictions

SCHEMA = "tricompose-report-extraction-source-comparison-v1"
ANALYSIS_SCHEMA = "tricompose-report-extraction-source-comparison-analysis-v1"


def literal_readout(report, finding, scope_checker):
    span = checked_span(report, 0, len(report))
    # Fixed probe only exposes existing, model-independent suggested scopes.
    # Its veto is not a veto of a real model output; do not use it as such.
    result = scope_checker(report, span, finding, "uncertain")
    suggested = result["rule_suggested_states"]
    covered = (result["literal_mentions_checked"] > 0 and len(suggested) == 1
        and bool(result["scope_reasons"]) and set(result["scope_reasons"]) <= CHECKED_REASONS)
    return {"state": suggested[0] if covered else "unknown", "scope_covered": covered,
        "scope_reasons": result["scope_reasons"], "suggested_states": suggested,
        "literal_mentions_checked": result["literal_mentions_checked"], "evidence": span,
        "independent_clinical_validation": False}


def compare(args):
    from repair_cached_report_evidence import scope_check
    if sha256_file(Path(scope_check.__code__.co_filename)) != FROZEN_GUARD_SHA:
        raise ValueError("pre-challenge frozen guard changed")
    resolver, texts, source = load_inputs(args.bank_run)
    baseline, baseline_path = checked_predictions(args.score_run, source)
    context, context_path = checked_predictions(args.context_run, source)
    if baseline["producer"]["model_id"] != "chexbert" or context["producer"]["model_id"] != "medspacy_context" or context.get("official_rules_modified") is not False:
        raise ValueError("frozen comparison producers differ")
    ids = {row["item_id"] for row in resolver}
    indices = [{row["item_id"]: row for row in payload["records"]} for payload in (baseline, context)]
    if any(len(index) != len(payload["records"]) or set(index) != ids for index, payload in zip(indices, (baseline,context), strict=True)):
        raise ValueError("comparison inventory differs")
    rows = []
    for source_row in resolver:
        cb, cx = (index[source_row["item_id"]] for index in indices)
        if any(row["report_sha256"] != source_row["report_sha256"] or row["status"] != "complete" or set(row["finding_states"]) != set(FINDINGS) for row in (cb, cx)):
            raise ValueError("comparison source/state contract differs")
        text = texts[source_row["report_sha256"]]
        for finding in FINDINGS:
            literal = literal_readout(text, finding, scope_check)
            states = (cb["finding_states"][finding], cx["finding_states"][finding], literal["state"])
            if any(value not in {"positive", "negative", "uncertain", "unknown"} for value in states):
                raise ValueError("invalid comparison states")
            interpretation = "unresolved_extraction"
            if literal["scope_covered"]:
                interpretation = "covered_scope_parser_agreement" if states[0] == states[1] == states[2] else "parser_disagreement_with_covered_literal_scope"
            rows.append({"item_id": source_row["item_id"], "report_sha256": source_row["report_sha256"],
                "finding": finding, "chexbert_state": states[0], "context_state": states[1],
                "literal_state": states[2], "literal_evidence": literal,
                "interpretation": interpretation, "review_action": "verify_report_extraction",
                "report_content_error_established": False, "image_error_established": False,
                "regeneration_authorized": False})
    return {"schema_version": SCHEMA, "source": source, "records": rows,
        "reference_key_read_by_comparison": False, "frozen_guard_sha256": FROZEN_GUARD_SHA,
        "counts": dict(Counter(row["interpretation"] for row in rows)),
        "primary_metric_eligible": False, "selection_changed": False, "regeneration_authorized": False,
        "posthoc_development_diagnostic": True, "model_calls": 0,
        "independence": "Different implementations, same report and shared literal inventory; not independent clinical votes."}, {"chexbert_predictions": baseline_path, "context_predictions": context_path}


def analyze(args):
    resolver, _, source = load_inputs(args.bank_run)
    root = require_inside(args.comparison_run, PROTECTED_ROOT, must_exist=True)
    path, manifest = root/"comparison.json", read_json(root/"manifest.json")
    if manifest["schema_version"] != SCHEMA or sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("comparison artifact binding differs")
    payload = read_json(path)
    if payload["source"] != source or payload["reference_key_read_by_comparison"] is not False or payload["frozen_guard_sha256"] != FROZEN_GUARD_SHA:
        raise ValueError("comparison provenance differs")
    refs, key_path = load_references(args.bank_run, resolver)
    index = {(row["item_id"], row["finding"]): row for row in payload["records"]}
    if len(index) != len(payload["records"]) or set(index) != {(row["item_id"], finding) for row in resolver for finding in FINDINGS}:
        raise ValueError("comparison denominator differs")
    scores = {}
    for column in ("chexbert_state", "context_state", "literal_state"):
        predictions = [{"item_id": row["item_id"], "report_sha256": row["report_sha256"], "status": "complete",
            "finding_states": {finding: index[row["item_id"],finding][column] for finding in FINDINGS}} for row in resolver]
        metrics, _ = evaluate_predictions(refs, predictions)
        scores[column] = metrics["designated_targets"]
    details = []
    for ref in refs:
        for finding in ref["evaluation_findings"]:
            row = index[ref["item_id"], finding]
            if row["report_sha256"] != ref["report_sha256"]:
                raise ValueError("comparison/reference binding differs")
            details.append({key: value for key, value in row.items() if key != "literal_evidence"} |
                {"family": ref["family"], "authored_expected_state": ref["expected_states"][finding],
                 "literal_scope_covered": row["literal_evidence"]["scope_covered"]})
    covered = [row for row in details if row["literal_scope_covered"]]
    cb_correct = [row for row in details if row["chexbert_state"] == row["authored_expected_state"]]
    disagreements = [row for row in details if row["chexbert_state"] != row["context_state"]]
    return {"schema_version": ANALYSIS_SCHEMA, "source": source, "designated_checks": len(details),
        "scores": scores, "literal_scope_covered": len(covered),
        "literal_matches_on_covered": sum(row["literal_state"] == row["authored_expected_state"] for row in covered),
        "covered_literal_differs_from_chexbert": sum(row["literal_state"] != row["chexbert_state"] for row in covered),
        "chexbert_context_disagreements": len(disagreements),
        "context_corrects_chexbert_mismatch_diagnostic_only": sum(row["chexbert_state"] != row["authored_expected_state"] and row["context_state"] == row["authored_expected_state"] for row in details),
        "context_disagrees_with_correct_chexbert": sum(row["context_state"] != row["authored_expected_state"] for row in cb_correct),
        "primary_metric_eligible": False, "selection_changed": False, "regeneration_authorized": False,
        "independent_clinical_accuracy": None, "posthoc_development_diagnostic": True,
        "interpretation": "State comparison is a source-extraction diagnostic. Literal coverage/conditional match is limited syntax coverage, not an independently correct patient fact. Parser disagreement cannot establish CXR/report content errors."}, details, {"comparison": path, "authored_references": key_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compare", "analyze"), required=True)
    for name in ("bank-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("score-run", "context-run", "comparison-run"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        if args.mode == "compare":
            payload, sources = compare(args)
            output = io.StringIO(newline="")
            fields = ("item_id", "finding", "report_sha256", "chexbert_state", "context_state", "literal_state", "interpretation", "review_action", "regeneration_authorized")
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in payload["records"])
            files = [write_private_json(temporary/"comparison.json", payload), write_private_text(temporary/"source_table.csv", output.getvalue())]
            schema = SCHEMA
        else:
            payload, details, sources = analyze(args)
            files = [write_private_json(temporary/"summary.json", payload), write_private_json(temporary/"details.json", {"records": details})]
            schema = ANALYSIS_SCHEMA
        write_private_json(temporary/"manifest.json", {"schema_version": schema, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "artifacts": {str(path.relative_to(temporary)): {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_report_source_"+args.mode, "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

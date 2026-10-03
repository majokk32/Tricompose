#!/usr/bin/env python3
"""Attach aggregate verifier limitations to cached synthetic score rows.

Does not change scores, winners, masks or thresholds. All rankings remain
diagnostic and must not authorize targeted regeneration or clinical claims.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, read_json, require_inside, sha256_file,
                       write_private_json, write_private_text)
from xrv_real_findings import MAPPING


def validation_profile(image_summary, threshold_summary):
    if "records" in image_summary or "records" in threshold_summary:
        raise ValueError("aggregate summaries required, not patient records")
    if (image_summary.get("reference_kind") != "report_extracted_weak_labels" or
            image_summary.get("primary_metric_eligible") is not False or
            threshold_summary.get("primary_metric_eligible") is not False):
        raise ValueError("unexpected primary/reference claim")
    val, test = image_summary["summary"]["val"], image_summary["summary"]["test"]
    if (val["cases"] != threshold_summary["calibration_cases"] or
            test["cases"] != threshold_summary["test_cases"]):
        raise ValueError("validation cohort count mismatch")
    findings = {}
    for source, name in MAPPING.items():
        result = threshold_summary["findings"][name]
        before, after = result["test_default_0_5"], result["test_fitted"]
        if (test["findings"][source]["positive"] != before["positive_reference_count"] or
                test["findings"][source]["negative"] != before["negative_reference_count"] or
                val["findings"][source]["positive"] != result["calibration_positive_count"] or
                val["findings"][source]["negative"] != result["calibration_negative_count"]):
            raise ValueError("finding reference count mismatch")
        findings[name] = {
            "threshold_fit_enabled": result["enabled"], "threshold": result["threshold"],
            "test_auroc": test["findings"][source]["auroc"],
            "test_default_balanced_accuracy": before["balanced_accuracy"],
            "test_fitted_balanced_accuracy": None if after is None else after["balanced_accuracy"],
            "test_fitted_sensitivity": None if after is None else after["sensitivity"],
            "test_fitted_specificity": None if after is None else after["specificity"],
            "primary_metric_eligible": False, "targeted_repair_approved": False,
        }
    return {"reference_kind": "report_extracted_weak_labels",
            "label_stratified_cohort": image_summary["label_stratified_cohort"],
            "independent_image_ground_truth": False, "primary_metric_eligible": False,
            "targeted_repair_approved": False, "selection_changed_by_annotation": False,
            "reliability_mask_fitted_from_test": False, "findings": findings}


def annotate_rows(rows):
    if not rows:
        raise ValueError("empty score table")
    added = {"selection_evidence_scope": "weak_reference_thresholds_diagnostic_only",
             "primary_metric_eligible": False, "targeted_repair_approved": False,
             "independent_clinical_evaluation_pending": True}
    if any(set(row) & set(added) for row in rows):
        raise ValueError("scope fields already present")
    return [{**row, **added} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--image-summary", required=True)
    parser.add_argument("--threshold-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    os.umask(0o007)
    try:
        selection = require_inside(args.selection_run, PROTECTED_ROOT, must_exist=True)
        source = selection / "candidate_score_table.csv"
        manifest = read_json(selection / "manifest.json")
        source_hash = sha256_file(source)
        if source_hash != manifest["artifacts"][source.name]["sha256"]:
            raise ValueError("candidate table hash mismatch")
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != manifest["counts"]["candidate_rows"]:
            raise ValueError("candidate row count mismatch")
        image_path = require_inside(args.image_summary, PROTECTED_ROOT, must_exist=True)
        threshold_path = require_inside(args.threshold_summary, PROTECTED_ROOT, must_exist=True)
        profile = validation_profile(read_json(image_path), read_json(threshold_path))
        scoped = annotate_rows(rows)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(scoped[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scoped)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        files = [write_private_text(temporary / "candidate_score_table.csv", buffer.getvalue()),
                 write_private_json(temporary / "verifier_validation_profile.json", profile),
                 write_private_text(temporary / "README.md",
                     "# Diagnostic candidate score table\n\n"
                     "All score values and winner flags are copied unchanged from the new static selection.\n"
                     "Added scope columns explicitly reject paper-primary and targeted-repair use.\n"
                     "No mask, threshold or selection policy was tuned on test results.\n"
                     "Inspect verifier_validation_profile.json for aggregate weak-reference performance.\n"
                     "Pneumonia/ pneumothorax verifier weakness remains a review issue, not a new masking rule.\n"
                     "Scores are not calibrated probabilities or independent clinical correctness.\n")]
        write_private_json(temporary / "manifest.json", {
            "schema_version": "tricompose-diagnostic-scoring-scope-v1", "run_id": args.run_id,
            "primary_metric_eligible": False, "targeted_repair_approved": False,
            "counts": {"candidate_rows": len(rows), "cases": len({row["case_id"] for row in rows})},
            "source_hashes": {"candidate_csv": source_hash, "selection_manifest": sha256_file(selection / "manifest.json"),
                              "image_summary": sha256_file(image_path), "threshold_summary": sha256_file(threshold_path),
                              "program": sha256_file(__file__)},
            "artifacts": {file.name: {"sha256": sha256_file(file)} for file in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed", "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

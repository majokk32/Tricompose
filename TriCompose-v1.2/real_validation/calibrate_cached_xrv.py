#!/usr/bin/env python3
"""Fit weak-reference XRV operating points from cached scores inside Slurm.

Reads no CXR pixels or report text and performs no model inference. Real
manifest patient keys are used in memory for split checks, then keyed hashes
are retained only in the protected calibration input. Thresholds fit val only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from pathlib import Path

from contracts import (
    CHEXPERT_FINDINGS, FINDING_STATES, PROTECTED_ROOT,
    commit_atomic_run, discard_atomic_run, new_atomic_run, read_json,
    require_inside, sha256_file, write_private_json, write_private_text,
)
from extract_cxr_labels_xrv import XRV_LABELS, _state
from xrv_calibration import SCORE_SPACE, fit_bundle, validate_bundle
from xrv_real_findings import MAPPING


PROJECT = Path("/project2/ruishanl_1185")
SCHEMA = "tricompose-cached-xrv-operating-point-check-v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protocol_provenance(real_producer, synthetic):
    """Reconstruct the inspected adapter protocol from its library fingerprints."""
    preprocessing = {
        "image": "PIL_L_float32", "normalize_maxval": 255,
        "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
        "xrv_models_source_sha256": real_producer["model_source_sha256"],
        "xrv_datasets_source_sha256": real_producer["preprocessing_source_sha256"],
    }
    result = {
        "checkpoint_sha256": real_producer["checkpoint_sha256"],
        "score_space": SCORE_SPACE,
        "preprocessing_sha256": fingerprint(preprocessing),
        "finding_mapping_sha256": fingerprint(XRV_LABELS),
    }
    if real_producer.get("frozen") is not True:
        raise ValueError("cached real scorer is not frozen")
    if synthetic.get("score_semantics") != SCORE_SPACE or synthetic.get("probability_semantics") is not False:
        raise ValueError("unsupported cached score semantics")
    producer = synthetic["producer"]
    if any(producer.get(key) != value for key, value in result.items()):
        raise ValueError("real/synthetic scorer protocol mismatch")
    if real_producer.get("adapter_source_sha256"):
        if any(real_producer.get(key) != value for key, value in result.items()):
            raise ValueError("recorded real scorer protocol mismatch")
        result["adapter_source_sha256"] = real_producer["adapter_source_sha256"]
        result["adapter_protocol_reconstruction"] = "recorded_adapter_and_protocol_fingerprints"
    else:
        result["adapter_protocol_reconstruction"] = (
            "recorded_library_fingerprints_and_inspected_adapter; historical_real_adapter_hash_unavailable"
        )
    return result


def canonical_record(row, group_hash):
    if row.get("split") not in {"val", "test"}:
        raise ValueError("unexpected cached split")
    if set(row["scores"]) != set(MAPPING) or set(row["reference_states"]) != set(MAPPING):
        raise ValueError("cached eight-finding inventory mismatch")
    scores = dict.fromkeys(CHEXPERT_FINDINGS)
    states = dict.fromkeys(CHEXPERT_FINDINGS, "unknown")
    for source, finding in MAPPING.items():
        value, state = row["scores"][source], row["reference_states"][source]
        if state not in FINDING_STATES:
            raise ValueError("invalid weak reference state")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("invalid cached XRV score")
        scores[finding], states[finding] = value, state
    return {"sample_sha256": row["image_sha256"],
            "patient_group_sha256": group_hash, "scores": scores,
            "reference_states": states}


def decision_metrics(records, finding, threshold, *, min_per_class):
    known = [(r["scores"][finding], r["reference_states"][finding]) for r in records
             if r["scores"][finding] is not None
             and r["reference_states"][finding] in {"positive", "negative"}]
    tp = sum(score >= threshold and state == "positive" for score, state in known)
    fn = sum(score < threshold and state == "positive" for score, state in known)
    tn = sum(score < threshold and state == "negative" for score, state in known)
    fp = sum(score >= threshold and state == "negative" for score, state in known)
    sensitivity = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    balanced = None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2
    return {
        "positive_reference_count": tp + fn, "negative_reference_count": tn + fp,
        "excluded_unknown_uncertain_or_missing": len(records) - len(known),
        "support_sufficient_for_pilot": min(tp + fn, tn + fp) >= min_per_class,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": balanced,
        "reference_kind": "report_extracted_weak_labels",
    }


def fit_and_evaluate(validation, test, provenance, *, min_per_class):
    validation_images = {r["sample_sha256"] for r in validation}
    test_images = {r["sample_sha256"] for r in test}
    if len(test_images) != len(test) or validation_images & test_images:
        raise ValueError("duplicate or overlapping held-out image hashes")
    bundle = fit_bundle(validation, provenance,
                        heldout_group_hashes=[r["patient_group_sha256"] for r in test],
                        min_per_class=min_per_class, uncertainty_margin=0.0)
    validate_bundle(bundle)
    summary = {}
    for finding in CHEXPERT_FINDINGS:
        fitted = bundle["findings"][finding]
        summary[finding] = {
            "enabled": fitted["enabled"], "fit_status": fitted["status"],
            "threshold": fitted["positive_min"] if fitted["enabled"] else None,
            "calibration_positive_count": fitted["positive_count"],
            "calibration_negative_count": fitted["negative_count"],
            "test_default_0_5": decision_metrics(test, finding, 0.5, min_per_class=min_per_class),
            "test_fitted": (decision_metrics(test, finding, fitted["positive_min"],
                                            min_per_class=min_per_class)
                            if fitted["enabled"] else None),
        }
    return bundle, summary


def apply_cached(synthetic, bundle):
    if synthetic.get("calibration", {}).get("status") != "uncalibrated_default_0_5":
        raise ValueError("expected unchanged default-threshold synthetic source")
    validate_bundle(bundle, checkpoint_sha256=synthetic["producer"]["checkpoint_sha256"],
                    expected_provenance={key: synthetic["producer"][key]
                                         for key in ("score_space", "preprocessing_sha256", "finding_mapping_sha256")})
    records, changes = [], []
    enabled = {name for name, threshold in bundle["findings"].items() if threshold["enabled"]}
    ids = set()
    for old in synthetic["records"]:
        if old["cxr_candidate_id"] in ids:
            raise ValueError("duplicate synthetic image")
        ids.add(old["cxr_candidate_id"])
        scores = old["finding_probabilities"]
        if set(scores) != set(CHEXPERT_FINDINGS):
            raise ValueError("incomplete cached synthetic score vector")
        states = {name: "unknown" if scores[name] is None else _state(scores[name], bundle["findings"][name])
                  for name in CHEXPERT_FINDINGS}
        records.append({**old, "finding_states": states})
        comparable = [name for name in enabled if scores[name] is not None]
        changes.append({
            "cxr_candidate_id": old["cxr_candidate_id"],
            "enabled_comparable_heads": len(comparable),
            "default_positive_on_enabled_heads": sum(old["finding_states"][name] == "positive" for name in comparable),
            "fitted_positive_on_enabled_heads": sum(states[name] == "positive" for name in comparable),
            "heads_masked_due_to_unavailable_reference_or_support": sum(
                name not in enabled and scores[name] is not None for name in CHEXPERT_FINDINGS),
        })
    result = {**synthetic, "records": records, "thresholds": bundle["findings"],
              "primary_metric_eligible": False,
              "calibration": {"status": bundle["calibration_status"],
                              "primary_metric_eligible": False,
                              "source_sha256": fingerprint(bundle), "provenance": bundle["provenance"]},
              "counts": {"cxr_candidates": len(records), "model_calls": 0},
              "cached_model_inference_reused": True,
              "fresh_model_inference_used": False, "peak_vram_gib": None}
    result["source_inference_elapsed_seconds"] = result.pop("elapsed_seconds", None)
    return result, changes


def matched_groups(manifest, cached):
    """Stream source metadata only; raw patient keys never enter output."""
    by_index = {}
    for record in cached["records"]:
        index = record["source_row_index"]
        if type(index) is not int or index < 0 or index in by_index:
            raise ValueError("invalid or duplicate cached row index")
        by_index[index] = record
    secret = secrets.token_bytes(32)
    groups, subject_splits = {}, {}
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"subject_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("matched source metadata schema incomplete")
        for index, row in enumerate(reader):
            subject, split = str(row["subject_id"]).strip(), str(row["split"]).strip()
            if subject.startswith("s"):
                subject = subject[1:]
            if not subject.isdigit():
                raise ValueError("invalid source patient key")
            subject = str(int(subject))
            if split in {"train", "val", "test"}:
                if subject in subject_splits and subject_splits[subject] != split:
                    raise ValueError("patient overlaps dataset splits")
                subject_splits[subject] = split
            if index in by_index:
                if by_index[index]["split"] != split:
                    raise ValueError("cached/source split mismatch")
                groups[index] = hmac.new(secret, subject.encode(), hashlib.sha256).hexdigest()
    if set(groups) != set(by_index) or len(set(groups.values())) != len(groups):
        raise ValueError("cached cohort is not one record per distinct patient")
    return groups


def validate_fixed_cohort(cached, cohort, cohort_sha256):
    if cohort.get("schema_version") != "tricompose-fixed-real-reference-cohort-v1":
        raise ValueError("wrong fixed reference cohort schema")
    if cached.get("fixed_reference_cohort_sha256") != cohort_sha256:
        raise ValueError("cached/fixed cohort hash mismatch")
    references = cohort["records"]
    projected = [{key: row[key] for key in ("case_id", "split", "source_row_index", "reference_states")}
                 for row in cached["records"]]
    if projected != references:
        raise ValueError("cached/fixed cohort membership or references mismatch")
    counts = {split: sum(row["split"] == split for row in references) for split in ("val", "test")}
    if any(count < 1 for count in counts.values()) or sum(counts.values()) != len(references):
        raise ValueError("invalid fixed cohort split inventory")
    return counts


def markdown(summary, changes, enabled, *, calibration_cases=128, test_cases=128,
             label_stratified_cohort=False):
    lines = ["# Weak-reference XRV operating-point check", "",
             f"Fit: fixed {calibration_cases}-case validation cohort only. Evaluate: fixed {test_cases}-case test cohort.",
             "References come from source reports; image ground truth and probability calibration are unavailable.",
             ("Test label quotas were audited during stratified selection; test model scores are held out from fitting."
              if label_stratified_cohort else
              "The test cohort was inspected descriptively in a previous pilot and is held out from threshold fitting."),
             "", "| Finding | Validation + / - | Fitted threshold | Test + / - | Test BA, default | Test BA, fitted |",
             "|---|---:|---:|---:|---:|---:|"]
    for finding, row in summary.items():
        default, fitted = row["test_default_0_5"], row["test_fitted"]
        fmt = lambda value: "NA" if value is None else f"{value:.3f}"
        lines.append(f"| {finding} | {row['calibration_positive_count']} / {row['calibration_negative_count']} | "
                     f"{fmt(row['threshold'])} | {default['positive_reference_count']} / {default['negative_reference_count']} | "
                     f"{fmt(default['balanced_accuracy'])} | {fmt(None if fitted is None else fitted['balanced_accuracy'])} |")
    lines.extend(["", f"Enabled findings: {enabled}/14. At least 20 explicit positive and 20 explicit negative validation references are required.",
                  "Test class counts and support flags must be checked before interpreting each balanced accuracy.",
                  "", "## Cached synthetic scores on the same enabled-head subset", "",
                  f"Comparable enabled image-head pairs: {sum(r['enabled_comparable_heads'] for r in changes)}.",
                  f"Default positive calls on that subset: {sum(r['default_positive_on_enabled_heads'] for r in changes)}.",
                  f"Fitted positive calls on that subset: {sum(r['fitted_positive_on_enabled_heads'] for r in changes)}.",
                  "Disabled heads are counted separately; reducing head coverage is not evidence that specificity improved.",
                  "",
                  "This cached threshold-fitting step trained no weights, ran no model inference, and read no real pixels or report text.",
                  "Operating points fitted to weak report labels do not establish synthetic-image clinical accuracy."])
    if label_stratified_cohort:
        lines.extend(["", "This is a label-stratified, new-patient diagnostic cohort, not a natural-prevalence sample.",
                      "A separate independent paper-final cohort and adjudicated image evidence remain required."])
    return "\n".join(lines) + "\n"


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cached patient-derived records require approved Slurm")
    cached_path = require_inside(args.cached_real_scores, PROTECTED_ROOT, must_exist=True)
    synthetic_path = require_inside(args.cached_synthetic_labels, PROTECTED_ROOT, must_exist=True)
    cached, synthetic = read_json(cached_path), read_json(synthetic_path)
    if cached.get("schema_version") != "tricompose-real-xrv-weak-finding-check-v1":
        raise ValueError("wrong cached real score schema")
    if cached.get("reference_kind") != "report_extracted_weak_labels":
        raise ValueError("wrong weak reference provenance")
    manifest = Path(args.dataset_manifest).resolve(strict=True)
    if not manifest.is_file() or not manifest.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("matched source manifest outside read-only project")
    manifest_hash = sha256_file(manifest)
    if cached["source_manifest_sha256"] != manifest_hash:
        raise ValueError("cached/source manifest hash mismatch")
    groups = matched_groups(manifest, cached)
    validation, test = [], []
    for row in cached["records"]:
        canonical = canonical_record(row, groups[row["source_row_index"]])
        (validation if row["split"] == "val" else test).append(canonical)
    fixed_path = getattr(args, "fixed_cohort", None)
    label_stratified = fixed_path is not None
    if fixed_path:
        fixed_path = require_inside(fixed_path, PROTECTED_ROOT, must_exist=True)
        cohort_counts = validate_fixed_cohort(cached, read_json(fixed_path), sha256_file(fixed_path))
        if len(validation) != cohort_counts["val"] or len(test) != cohort_counts["test"]:
            raise ValueError("fixed calibration/test cohort count mismatch")
    elif len(validation) != 128 or len(test) != 128:
        raise ValueError("fixed calibration/test cohort count mismatch")
    source_protocol = protocol_provenance(cached["producer"], synthetic)
    base_provenance = {**source_protocol, "split_manifest_sha256": manifest_hash,
                       "reference_sha256": cached["source_chexpert_labels_sha256"],
                       "split_role": "calibration", "source_kind": "matched_real_validation",
                       "reference_kind": "report_extracted_weak_labels",
                       "cached_real_scores_sha256": sha256_file(cached_path)}
    if fixed_path:
        base_provenance["fixed_reference_cohort_sha256"] = sha256_file(fixed_path)
    calibration_input = {"schema_version": "tricompose-xrv-calibration-input-v1",
                         "records": validation, "provenance": base_provenance,
                         "heldout_group_hashes": [r["patient_group_sha256"] for r in test]}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        input_path = write_private_json(temporary / "calibration_input.json", calibration_input)
        provenance = {**base_provenance, "calibration_input_sha256": sha256_file(input_path)}
        bundle, summary = fit_and_evaluate(validation, test, provenance, min_per_class=args.min_per_class)
        thresholds = write_private_json(temporary / "thresholds.json", bundle)
        relabeled, changes = apply_cached(synthetic, bundle)
        relabeled["run_id"] = args.run_id
        relabeled["calibration"]["source_sha256"] = sha256_file(thresholds)
        relabeled["calibration"]["synthetic_source_bundle_sha256"] = sha256_file(synthetic_path)
        enabled = sum(row["enabled"] for row in bundle["findings"].values())
        artifacts = [thresholds,
                     write_private_json(temporary / "cxr_finding_labels.json", relabeled),
                     write_private_json(temporary / "evaluation_summary.json", {
                         "schema_version": SCHEMA, "calibration_cases": len(validation), "test_cases": len(test),
                         "findings_enabled": enabled, "findings": summary,
                         "synthetic_enabled_head_comparison": changes,
                         "primary_metric_eligible": False, "independent_image_ground_truth": False,
                         "model_training_used": False, "fresh_model_inference_used": False,
                         "prior_test_descriptive_inspection": True,
                         "prior_test_score_descriptive_inspection":
                             cached.get("prior_test_score_descriptive_inspection", True),
                         "label_stratified_cohort": label_stratified}),
                     write_private_text(temporary / "evaluation_summary.md", markdown(
                         summary, changes, enabled, calibration_cases=len(validation), test_cases=len(test),
                         label_stratified_cohort=label_stratified))]
        write_private_json(temporary / "manifest.json", {
            "schema_version": SCHEMA, "run_id": args.run_id, "findings_enabled": enabled,
            "source_hashes": {"real_scores": sha256_file(cached_path),
                              "synthetic_labels": sha256_file(synthetic_path),
                              "source_manifest": manifest_hash, "code": sha256_file(__file__)},
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in [input_path, *artifacts]},
        })
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return sha256_file(target / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-real-scores", required=True)
    parser.add_argument("--cached-synthetic-labels", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--fixed-cohort", help="Optional hash-bound new cohort; legacy default stays fixed at 128+128.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-per-class", type=int, default=20)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    os.umask(0o007)
    started = time.monotonic()
    try:
        digest = run(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1
    print(json.dumps({"status": "completed", "manifest_sha256": digest,
                      "elapsed_seconds": round(time.monotonic() - started, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

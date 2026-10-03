#!/usr/bin/env python3
"""Frozen XRV on an already fixed, new-patient weak-reference cohort.

Slurm only. Real pixels are consumed internally; source report text and EHR
clinical fields are not model inputs. Public output is sanitized status/hash.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

from audit_reference_coverage import (build_plan, patient_pool, prior_patients)
from biovil_matched_pairs import PROJECT, _load_manifest_rows, _resolve_source
from calibrate_cached_xrv import fingerprint
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, read_json, require_inside, sha256_file,
                       write_private_json)
from extract_cxr_labels_xrv import FrozenXRVRuntime, XRV_LABELS
from hard_negative_biovil import load_report_label_index
from xrv_calibration import SCORE_SPACE
from xrv_real_findings import MAPPING, SCHEMA, summarize


def validate_cohort_selection(rows, labels, previous, cohort):
    if cohort.get("schema_version") != "tricompose-fixed-real-reference-cohort-v1":
        raise ValueError("unexpected fixed-cohort schema")
    params = cohort["selection"]
    if any(params.get(key) is not True for key in (
            "one_study_per_patient", "patient_disjoint_splits",
            "previous_patient_exclusion_verified", "label_stratified")):
        raise ValueError("fixed-cohort patient/selection contract missing")
    if params.get("required_findings") != ["Pneumonia"] or cohort.get("primary_metric_eligible") is not False:
        raise ValueError("unexpected cohort claim or required finding")
    excluded = prior_patients(rows, previous)
    pool = patient_pool(rows, labels, excluded, seed=params["seed"])
    expected, summary = build_plan(pool, min_per_class=params["min_per_class"],
                                  max_cases=params["max_cases_per_split"])
    if summary["status"] != "ready_for_image_side_weak_reference_pilot":
        raise ValueError("required reference quotas unavailable")
    if expected != cohort["records"]:
        raise ValueError("fixed cohort changed or references mismatch")
    return expected


def mapped_scores(raw):
    scores = {}
    for name, mapped in MAPPING.items():
        available = [raw[label] for label in XRV_LABELS[mapped] if label in raw]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or not 0 <= value <= 1 for value in available):
            raise ValueError("invalid frozen XRV score")
        scores[name] = max(available) if available else None
    return scores


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real-pixel inference requires approved Slurm")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    dataset = Path(args.dataset_root).resolve(strict=True)
    if not dataset.is_dir() or not dataset.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("dataset outside read-only project")
    cohort_path = require_inside(args.cohort, PROTECTED_ROOT, must_exist=True)
    if sha256_file(cohort_path) != args.cohort_sha256:
        raise ValueError("fixed cohort hash mismatch")
    cohort = read_json(cohort_path)
    previous_path = require_inside(args.previous_scores, PROTECTED_ROOT, must_exist=True)
    labels_path = Path(args.chexpert_labels).resolve(strict=True)
    expected_hashes = {
        "dataset_manifest": sha256_file(dataset / "manifest.csv"),
        "chexpert_labels": sha256_file(labels_path),
        "previous_scores": sha256_file(previous_path),
        "program": sha256_file(Path(__file__).with_name("audit_reference_coverage.py")),
        "selection_source": sha256_file(Path(__file__).with_name("biovil_matched_pairs.py")),
        "label_parser_source": sha256_file(Path(__file__).with_name("hard_negative_biovil.py")),
    }
    if cohort["source_hashes"] != expected_hashes:
        raise ValueError("fixed-cohort source fingerprints changed")
    rows = _load_manifest_rows(dataset)
    previous = read_json(previous_path)
    selected = validate_cohort_selection(rows, load_report_label_index(labels_path), previous, cohort)
    previous_image_hashes = {record["image_sha256"] for record in previous["records"]}
    weight_dir = Path(args.cache_dir).resolve(strict=True)
    weight = require_inside(weight_dir / args.weight_filename, PROTECTED_ROOT, must_exist=True)
    if sha256_file(weight) != previous["producer"]["checkpoint_sha256"]:
        raise ValueError("frozen XRV checkpoint changed from prior pilot")
    torch.cuda.reset_peak_memory_stats()  # Include model loading in the peak.
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            runtime = FrozenXRVRuntime(cache_dir=weight_dir, weight_filename=args.weight_filename,
                                       model_name="densenet121-res224-all")
    if runtime.model.training or any(parameter.requires_grad for parameter in runtime.model.parameters()):
        raise RuntimeError("XRV scorer is not frozen")
    records, seen_images = [], set()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            for item in selected:
                source = rows[item["source_row_index"]]
                image_path = _resolve_source(source["cxr_path"], dataset)
                image_hash = sha256_file(image_path)
                if image_hash in seen_images or image_hash in previous_image_hashes:
                    raise ValueError("duplicate or previously scored image hash in fixed cohort")
                seen_images.add(image_hash)
                records.append({**item, "image_sha256": image_hash,
                                "scores": mapped_scores(runtime.predict(image_path))})
    torch.cuda.synchronize()
    model_hash = sha256_file(Path(runtime.xrv.models.__file__))
    dataset_code_hash = sha256_file(Path(runtime.xrv.datasets.__file__))
    preprocessing = {"image": "PIL_L_float32", "normalize_maxval": 255,
                     "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
                     "xrv_models_source_sha256": model_hash,
                     "xrv_datasets_source_sha256": dataset_code_hash}
    import extract_cxr_labels_xrv as adapter
    return {
        "schema_version": SCHEMA, "status": "new_fixed_label_stratified_weak_reference_only",
        "source_manifest_sha256": expected_hashes["dataset_manifest"],
        "source_chexpert_labels_sha256": expected_hashes["chexpert_labels"],
        "fixed_reference_cohort_sha256": args.cohort_sha256,
        "producer": {"model_id": "xrv_densenet121_res224_all", "frozen": True,
                     "checkpoint_sha256": sha256_file(weight), "model_source_sha256": model_hash,
                     "preprocessing_source_sha256": dataset_code_hash,
                     "adapter_source_sha256": sha256_file(Path(adapter.__file__)),
                     "score_space": SCORE_SPACE, "preprocessing_sha256": fingerprint(preprocessing),
                     "finding_mapping_sha256": fingerprint(XRV_LABELS)},
        "summary": summarize(records, min_per_class=cohort["selection"]["min_per_class"]),
        "selection": cohort["selection"], "records": records,
        "reference_kind": "report_extracted_weak_labels", "independent_image_ground_truth": False,
        "primary_metric_eligible": False, "probability_calibration_performed": False,
        "clinical_threshold_fitted": False, "label_stratified_cohort": True,
        "natural_prevalence_estimate": False, "potential_pretraining_overlap_not_ruled_out": True,
        "prior_test_score_descriptive_inspection": False,
        "real_report_text_or_ehr_sent_to_model": False,
        "peak_vram_includes_model_loading": True,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "chexpert-labels", "previous-scores", "cohort", "cohort-sha256",
                 "cache-dir", "weight-filename", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    os.umask(0o007)
    temporary = None
    started = time.monotonic()
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = write_private_json(temporary / "scores.json", payload)
        write_private_json(temporary / "summary.json", {
            key: value for key, value in payload.items() if key != "records"})
        digest = sha256_file(output)
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed", "output_sha256": digest,
                      "elapsed_seconds": payload["elapsed_seconds"],
                      "peak_vram_gib": payload["peak_vram_gib"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

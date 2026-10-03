#!/usr/bin/env python3
"""Frozen XRV versus report-derived labels on the fixed real-pair cohort.

Slurm only. This is a weak-reference image-side diagnostic, not image truth,
probability calibration, or a clinical consistency operating point.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

from biovil_matched_pairs import (PROJECT, _load_manifest_rows,
                                  _resolve_source, auc_ap, select_cases)
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, read_json, require_inside, sha256_file,
                       write_private_json)
from extract_cxr_labels_xrv import FrozenXRVRuntime, XRV_LABELS
from hard_negative_biovil import FINDINGS, _id, load_report_label_index

SCHEMA = "tricompose-real-xrv-weak-finding-check-v1"
MAPPING = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Lung Opacity": "lung_opacity",
    "Pleural Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}
assert set(MAPPING) == FINDINGS


def summarize(records: list[dict], *, min_per_class: int = 10) -> dict:
    if min_per_class < 1:
        raise ValueError("minimum per class must be positive")
    result = {}
    for split in ("val", "test"):
        current = [row for row in records if row["split"] == split]
        findings = {}
        for name in sorted(MAPPING):
            known = [(row["scores"][name], row["reference_states"][name])
                     for row in current if row["scores"][name] is not None
                     and row["reference_states"][name] in {"positive", "negative"}]
            pos = sum(state == "positive" for _, state in known)
            neg = len(known) - pos
            metrics = ({"auroc": None, "average_precision": None}
                       if min(pos, neg) < min_per_class
                       else auc_ap([int(state == "positive") for _, state in known],
                                   [float(score) for score, _ in known]))
            findings[name] = {
                "positive": pos, "negative": neg,
                "excluded_unknown_or_uncertain": len(current) - len(known),
                "status": ("descriptive_weak_reference" if metrics["auroc"] is not None
                           else "insufficient_binary_support"),
                **metrics,
            }
        eligible = [row["auroc"] for row in findings.values() if row["auroc"] is not None]
        result[split] = {
            "cases": len(current), "min_per_class": min_per_class,
            "eligible_findings": len(eligible),
            "macro_auroc_over_eligible": (None if not eligible else
                                          round(sum(eligible) / len(eligible), 8)),
            "findings": findings,
        }
    return result


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real patient-level inputs require approved Slurm")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("XRV requires a GPU allocation")
    dataset = Path(args.dataset_root).resolve(strict=True)
    labels_path = Path(args.chexpert_labels).resolve(strict=True)
    weight_dir = Path(args.cache_dir).resolve(strict=True)
    weight = (weight_dir / args.weight_filename).resolve(strict=True)
    previous_path = require_inside(args.previous_scores, PROTECTED_ROOT,
                                   must_exist=True)
    if not dataset.is_dir() or not dataset.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("dataset outside read-only CARC project")
    if not weight.is_file() or not weight.is_relative_to(PROTECTED_ROOT.resolve(strict=True)):
        raise ValueError("frozen XRV checkpoint must be in protected workspace")
    previous = read_json(previous_path)
    if previous.get("schema_version") != "tricompose-real-biovil-pair-check-v1":
        raise ValueError("wrong fixed-cohort schema")
    if previous.get("source_manifest_sha256") != sha256_file(dataset / "manifest.csv"):
        raise ValueError("fixed-cohort manifest changed")
    rows = _load_manifest_rows(dataset)
    params = previous["selection"]
    selected = select_cases(rows, count_per_split=params["count_per_split"],
                            seed=params["seed"])
    fixed = {row["case_id"]: row for row in previous["records"]}
    if len(fixed) != len(selected) or any(
        fixed[row["case_id"]]["source_row_index"] != row["row_index"]
        for row in selected
    ):
        raise ValueError("fixed real cohort changed")
    label_index = load_report_label_index(labels_path)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            runtime = FrozenXRVRuntime(cache_dir=weight_dir,
                                       weight_filename=args.weight_filename,
                                       model_name="densenet121-res224-all")
    if runtime.model.training or any(parameter.requires_grad
                                     for parameter in runtime.model.parameters()):
        raise RuntimeError("XRV scorer is not frozen")
    torch.cuda.reset_peak_memory_stats()
    records = []
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            for row in selected:
                source_row = rows[row["row_index"]]
                key = (_id(source_row["subject_id"]), _id(source_row["study_id"]))
                reference = label_index.get(key)
                if reference is None:
                    reference = {name: "unknown" for name in FINDINGS}
                image_path = _resolve_source(row["cxr_path"], dataset)
                image_hash = sha256_file(image_path)
                if image_hash != fixed[row["case_id"]]["image_sha256"]:
                    raise ValueError("fixed image changed")
                raw = runtime.predict(image_path)
                scores = {}
                for name, mapped in MAPPING.items():
                    available = [raw[label] for label in XRV_LABELS[mapped]
                                 if label in raw]
                    value = max(available) if available else None
                    if value is not None and not math.isfinite(value):
                        raise ValueError("nonfinite XRV output")
                    scores[name] = value
                records.append({"case_id": row["case_id"], "split": row["split"],
                                "source_row_index": row["row_index"],
                                "image_sha256": image_hash,
                                "scores": scores,
                                "reference_states": reference})
    return {
        "schema_version": SCHEMA,
        "status": "weak_report_reference_only",
        "source_manifest_sha256": sha256_file(dataset / "manifest.csv"),
        "source_chexpert_labels_sha256": sha256_file(labels_path),
        "previous_scores_sha256": sha256_file(previous_path),
        "producer": {"model_id": "xrv_densenet121_res224_all", "frozen": True,
                     "checkpoint_sha256": sha256_file(weight),
                     "model_source_sha256": sha256_file(Path(runtime.xrv.models.__file__)),
                     "preprocessing_source_sha256": sha256_file(Path(runtime.xrv.datasets.__file__))},
        "summary": summarize(records),
        "records": records,
        "reference_kind": "report_extracted_weak_labels",
        "independent_image_ground_truth": False,
        "probability_calibration_performed": False,
        "clinical_threshold_fitted": False,
        "interpretation": "Descriptive image-classifier discrimination against report-derived CheXpert labels; not independent image truth or clinical consistency.",
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chexpert-labels", required=True)
    parser.add_argument("--previous-scores", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--weight-filename", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    os.umask(0o007)
    started = time.monotonic()
    temporary = None
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        protected = write_private_json(temporary / "scores.json", payload)
        output_hash = sha256_file(protected)
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed", "run_id": args.run_id,
                      "counts": {split: row["cases"] for split, row in
                                 payload["summary"].items()},
                      "output_sha256": output_hash,
                      "peak_vram_gib": payload["peak_vram_gib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

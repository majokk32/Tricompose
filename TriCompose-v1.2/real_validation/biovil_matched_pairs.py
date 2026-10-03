#!/usr/bin/env python3
"""Frozen BioViL-T matched-vs-cross-patient report pilot; Slurm only.

Patient-level source rows, image pixels and report text never reach stdout,
Git or the output manifest. This is a retrieval sanity check, not a clinical
contradiction benchmark or probability calibration.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROJECT = Path("/project2/ruishanl_1185")
sys.path.insert(0, str(WORKSPACE / "TriCompose-v1.0/eval/report_v1_1"))
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       sha256_file, write_private_json)  # noqa: E402
from score_report_cxr_biovil import _load_runtime, IMAGE_WEIGHT  # noqa: E402

SCHEMA = "tricompose-real-biovil-pair-check-v1"
REQUIRED_COLUMNS = frozenset({"subject_id", "study_id", "split",
                              "ViewPosition", "cxr_path", "report_path"})


def select_cases(rows: list[dict], *, count_per_split: int, seed: int) -> list[dict]:
    """Select one study per subject in each of val/test, then rotate donors."""
    if count_per_split < 2:
        raise ValueError("at least two cases per split are required")
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        split = str(row.get("split", ""))
        subject = str(row.get("subject_id", ""))
        if split in {"train", "val", "test"} and subject:
            subjects_by_split[split].add(subject)
        if split not in {"val", "test"}:
            continue
        if str(row.get("ViewPosition", "")).upper() not in {"AP", "PA"}:
            continue
        study = str(row.get("study_id", ""))
        if (not subject or not study or not str(row.get("cxr_path", "")).strip()
                or not str(row.get("report_path", "")).strip()):
            continue
        rank = hashlib.sha256(f"{seed}|{split}|{subject}|{study}|{index}".encode()).hexdigest()
        key = (split, subject)
        candidate = {"row_index": index, "split": split,
                     "subject_id": subject, "cxr_path": row["cxr_path"],
                     "report_path": row["report_path"]}
        if key not in best or rank < best[key][0]:
            best[key] = (rank, candidate)
    if any(subjects_by_split[left] & subjects_by_split[right]
           for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("patient appears in multiple dataset splits")
    selected = []
    for split in ("val", "test"):
        pool = sorted((entry for (role, _), entry in best.items() if role == split),
                      key=lambda pair: pair[0])
        if len(pool) < count_per_split:
            raise ValueError("insufficient distinct eligible patients")
        chosen = [entry[1] for entry in pool[:count_per_split]]
        for position, row in enumerate(chosen):
            donor = chosen[(position + 1) % count_per_split]
            if donor["subject_id"] == row["subject_id"]:
                raise AssertionError("cross-patient donor selection failed")
            selected.append({
                "case_id": f"case_{len(selected):04d}",
                "split": split,
                "row_index": row["row_index"],
                "donor_row_index": donor["row_index"],
                "cxr_path": row["cxr_path"],
                "matched_report_path": row["report_path"],
                "mismatched_report_path": donor["report_path"],
            })
    return selected


def auc_ap(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    if len(labels) != len(scores) or not labels or any(value not in {0, 1} for value in labels):
        raise ValueError("invalid binary score input")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("nonfinite score")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return {"auroc": None, "average_precision": None}
    ordered = sorted(zip(scores, labels), reverse=True)
    true_positive = false_positive = 0
    previous_fpr = previous_tpr = 0.0
    auroc = ap = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        batch = ordered[position:end]
        true_positive += sum(label for _, label in batch)
        false_positive += len(batch) - sum(label for _, label in batch)
        fpr = false_positive / negatives
        tpr = true_positive / positives
        auroc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        ap += (tpr - previous_tpr) * (true_positive / (true_positive + false_positive))
        previous_fpr, previous_tpr = fpr, tpr
        position = end
    return {"auroc": round(auroc, 8), "average_precision": round(ap, 8)}


def summarize(records: list[dict]) -> dict[str, dict]:
    result = {}
    for split in ("val", "test"):
        subset = [row for row in records if row["split"] == split]
        if not subset:
            raise ValueError("missing validation or test records")
        matched = [row["matched_score"] for row in subset]
        mismatched = [row["mismatched_score"] for row in subset]
        metrics = auc_ap([1] * len(subset) + [0] * len(subset), matched + mismatched)
        result[split] = {
            "cases": len(subset),
            "matched_mean": round(sum(matched) / len(matched), 8),
            "mismatched_mean": round(sum(mismatched) / len(mismatched), 8),
            "paired_win_rate": round(sum(a > b for a, b in zip(matched, mismatched)) / len(subset), 8),
            "paired_tie_rate": round(sum(a == b for a, b in zip(matched, mismatched)) / len(subset), 8),
            **metrics,
        }
    return result


def _resolve_source(path_text: str, root: Path) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = root / candidate
    source = candidate.resolve(strict=True)
    if not source.is_file() or not source.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("source artifact outside read-only CARC project")
    return source


def _load_manifest_rows(root: Path) -> list[dict]:
    path = (root / "manifest.csv").resolve(strict=True)
    if not path.is_file():
        raise ValueError("matched manifest unavailable")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("matched manifest schema is incomplete")
        return list(reader)


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real patient-level inputs may be read only inside approved Slurm")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("frozen BioViL-T requires a GPU allocation")
    root = Path(args.dataset_root).resolve(strict=True)
    model_path = Path(args.model_path).resolve(strict=True)
    if not root.is_dir() or not root.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("dataset root outside read-only CARC project")
    if not (model_path / IMAGE_WEIGHT).is_file() or not (model_path / "pytorch_model.bin").is_file():
        raise ValueError("BioViL-T checkpoint incomplete")
    manifest_path = root / "manifest.csv"
    selected = select_cases(_load_manifest_rows(root), count_per_split=args.count_per_split,
                            seed=args.seed)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            image_engine, text_engine = _load_runtime(model_path, torch.device("cuda:0"))
    torch.cuda.reset_peak_memory_stats()
    records = []
    with torch.inference_mode(), open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            for row in selected:
                image_path = _resolve_source(row["cxr_path"], root)
                matched_path = _resolve_source(row["matched_report_path"], root)
                mismatched_path = _resolve_source(row["mismatched_report_path"], root)
                texts = [matched_path.read_text(encoding="utf-8", errors="replace").strip(),
                         mismatched_path.read_text(encoding="utf-8", errors="replace").strip()]
                if not all(texts):
                    raise ValueError("empty real report in selected pair")
                image_embedding = image_engine.get_projected_global_embedding(image_path)
                text_embeddings = text_engine.get_embeddings_from_prompt(
                    texts, normalize=True, verbose=False)
                values = (text_embeddings @ image_embedding).detach().float().cpu().tolist()
                if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
                    raise ValueError("invalid BioViL-T pair scores")
                records.append({
                    "case_id": row["case_id"], "split": row["split"],
                    "source_row_index": row["row_index"],
                    "donor_row_index": row["donor_row_index"],
                    "image_sha256": sha256_file(image_path),
                    "matched_report_sha256": sha256_file(matched_path),
                    "mismatched_report_sha256": sha256_file(mismatched_path),
                    "matched_score": round(float(values[0]), 8),
                    "mismatched_score": round(float(values[1]), 8),
                })
    return {
        "schema_version": SCHEMA,
        "status": "real_matched_vs_random_cross_patient_pilot",
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_schema": "three_modalities/v2_labs_vitals",
        "selection": {"count_per_split": args.count_per_split, "seed": args.seed,
                      "one_study_per_patient": True, "patient_disjoint_splits": True,
                      "allowed_views": ["AP", "PA"],
                      "negative_type": "cross_patient_random_rotation_unadjudicated"},
        "producer": {"model_id": "biovil_t", "frozen": True,
                     "image_checkpoint_sha256": sha256_file(model_path / IMAGE_WEIGHT),
                     "text_checkpoint_sha256": sha256_file(model_path / "pytorch_model.bin")},
        "summary": summarize(records),
        "records": records,
        "clinical_hard_negative_validated": False,
        "probability_calibration_performed": False,
        "interpretation": "Pair retrieval sanity check, not clinical contradiction detection or a validated selection threshold.",
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--count-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2701)
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
                      "counts": {role: item["cases"] for role, item in payload["summary"].items()},
                      "output_sha256": output_hash,
                      "peak_vram_gib": payload["peak_vram_gib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

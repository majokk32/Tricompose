#!/usr/bin/env python3
"""Same-view, shared-finding real CXR/report hard-negative pilot; Slurm only.

Official CheXpert labels are derived from source reports and only PRESELECT
candidate negatives. They are not independent image truth or adjudication.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from biovil_matched_pairs import (IMAGE_WEIGHT, PROJECT, _load_manifest_rows,
                                  _load_runtime, _resolve_source, auc_ap,
                                  select_cases)
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, read_json, require_inside, sha256_file,
                       write_private_json)

SCHEMA = "tricompose-real-biovil-hard-negative-v1"
FINDINGS = {
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax",
}
TARGET_FINDINGS = FINDINGS - {"Lung Opacity"}


def _id(value: str) -> str:
    text = str(value).strip()
    if text.startswith("s") and text[1:].isdigit():
        text = text[1:]
    if not text.isdigit():
        raise ValueError("non-numeric source key in matched-data index")
    return str(int(text))


def _state(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    if text in {"1", "1.0"}:
        return "positive"
    if text in {"0", "0.0"}:
        return "negative"
    if text in {"-1", "-1.0"}:
        return "uncertain"
    if text == "":
        return "unknown"
    raise ValueError("unexpected CheXpert label value")


def load_report_label_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    source = path.resolve(strict=True)
    if not source.is_file() or not source.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("CheXpert label source outside read-only CARC project")
    result = {}
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not ({"subject_id", "study_id"} | FINDINGS).issubset(reader.fieldnames):
            raise ValueError("CheXpert label schema is incomplete")
        for row in reader:
            key = (_id(row["subject_id"]), _id(row["study_id"]))
            if key in result:
                raise ValueError("duplicate CheXpert study key")
            result[key] = {finding: _state(row[finding]) for finding in FINDINGS}
    return result


def select_hard_donor(anchor: dict, pool: list[dict], *, seed: int) -> dict | None:
    """Prefer one explicit report-label conflict amid shared positive findings."""
    valid = []
    own = anchor["labels"]
    if own is None:
        return None
    for donor in pool:
        if donor["subject_id"] == anchor["subject_id"]:
            continue
        if donor["split"] != anchor["split"] or donor["view"] != anchor["view"]:
            continue
        other = donor["labels"]
        if other is None:
            continue
        shared = sorted(finding for finding in FINDINGS
                        if own[finding] == other[finding] == "positive")
        conflicts = sorted(finding for finding in TARGET_FINDINGS
                           if {own[finding], other[finding]} == {"positive", "negative"})
        if not shared or not conflicts:
            continue
        agreements = sum(own[finding] == other[finding] and
                         own[finding] in {"positive", "negative"}
                         for finding in FINDINGS)
        tie = hashlib.sha256(f"{seed}|{anchor['row_index']}|{donor['row_index']}".encode()).hexdigest()
        # A near miss: fewer explicit conflicts, more shared and known agreements.
        priority = (len(conflicts), -len(shared), -agreements, tie)
        valid.append((priority, donor, shared, conflicts))
    if not valid:
        return None
    _, donor, shared, conflicts = min(valid, key=lambda item: item[0])
    return {"donor": donor, "shared_positive_findings": shared,
            "explicit_conflicting_findings": conflicts}


def build_pool(rows: list[dict], labels: dict[tuple[str, str], dict[str, str]]) -> list[dict]:
    pool = []
    seen_studies = set()
    for index, row in enumerate(rows):
        split = str(row.get("split", ""))
        view = str(row.get("ViewPosition", "")).upper()
        if split not in {"val", "test"} or view not in {"AP", "PA"}:
            continue
        if not str(row.get("report_path", "")).strip():
            continue
        key = (_id(row["subject_id"]), _id(row["study_id"]))
        if key in seen_studies:
            continue
        seen_studies.add(key)
        pool.append({"row_index": index, "subject_id": key[0], "study_id": key[1],
                     "split": split, "view": view, "report_path": row["report_path"],
                     "labels": labels.get(key)})
    return pool


def summarize(records: list[dict], unavailable: dict[str, int]) -> dict[str, dict]:
    result = {}
    for split in ("val", "test"):
        current = [row for row in records if row["split"] == split]
        if not current:
            result[split] = {"hard_negative_cases": 0, "unavailable": unavailable.get(split, 0),
                             "auroc": None, "average_precision": None,
                             "paired_win_rate": None}
            continue
        matched = [float(row["matched_score"]) for row in current]
        hard = [float(row["hard_score"]) for row in current]
        random_scores = [float(row["random_score"]) for row in current]
        metrics = auc_ap([1] * len(current) + [0] * len(current), matched + hard)
        random_metrics = auc_ap([1] * len(current) + [0] * len(current),
                                matched + random_scores)
        result[split] = {
            "hard_negative_cases": len(current),
            "unavailable": unavailable.get(split, 0),
            "matched_mean": round(sum(matched) / len(matched), 8),
            "hard_negative_mean": round(sum(hard) / len(hard), 8),
            "random_negative_mean_on_same_subset": round(sum(random_scores) / len(random_scores), 8),
            "paired_win_rate": round(sum(a > b for a, b in zip(matched, hard)) / len(current), 8),
            "random_paired_win_rate_on_same_subset": round(sum(a > b for a, b in zip(matched, random_scores)) / len(current), 8),
            **metrics,
            "random_auroc_on_same_subset": random_metrics["auroc"],
            "random_average_precision_on_same_subset": random_metrics["average_precision"],
        }
    return result


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real patient-level inputs may be read only inside approved Slurm")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BioViL-T requires a GPU allocation")
    dataset = Path(args.dataset_root).resolve(strict=True)
    model = Path(args.model_path).resolve(strict=True)
    labels_path = Path(args.chexpert_labels).resolve(strict=True)
    previous_path = require_inside(args.previous_scores, PROTECTED_ROOT, must_exist=True)
    if not dataset.is_dir() or not dataset.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("dataset root outside read-only CARC project")
    if not (model / IMAGE_WEIGHT).is_file() or not (model / "pytorch_model.bin").is_file():
        raise ValueError("BioViL-T checkpoint incomplete")
    previous = read_json(previous_path)
    if previous.get("schema_version") != "tricompose-real-biovil-pair-check-v1":
        raise ValueError("previous real-pair pilot has the wrong schema")
    if previous.get("source_manifest_sha256") != sha256_file(dataset / "manifest.csv"):
        raise ValueError("matched manifest changed since the previous pilot")
    if previous.get("producer", {}).get("image_checkpoint_sha256") != sha256_file(model / IMAGE_WEIGHT):
        raise ValueError("image checkpoint changed since the previous pilot")
    if previous.get("producer", {}).get("text_checkpoint_sha256") != sha256_file(model / "pytorch_model.bin"):
        raise ValueError("text checkpoint changed since the previous pilot")
    params = previous["selection"]
    rows = _load_manifest_rows(dataset)
    selected = select_cases(rows, count_per_split=params["count_per_split"],
                            seed=params["seed"])
    previous_rows = {row["case_id"]: row for row in previous["records"]}
    if len(previous_rows) != len(selected) or any(
            previous_rows[row["case_id"]]["source_row_index"] != row["row_index"]
            for row in selected):
        raise ValueError("fixed real cohort changed since the previous pilot")
    labels = load_report_label_index(labels_path)
    pool = build_pool(rows, labels)
    by_index = {row["row_index"]: row for row in pool}
    chosen = []
    unavailable = Counter()
    for row in selected:
        anchor = by_index.get(row["row_index"])
        if anchor is None:
            unavailable[row["split"]] += 1
            continue
        negative = select_hard_donor(anchor, pool, seed=params["seed"])
        if negative is None:
            unavailable[row["split"]] += 1
            continue
        chosen.append((row, negative))
    if not chosen:
        raise ValueError("no label-defined hard-negative candidates available")

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            image_engine, text_engine = _load_runtime(model, torch.device("cuda:0"))
    torch.cuda.reset_peak_memory_stats()
    records = []
    with torch.inference_mode(), open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            for row, negative in chosen:
                image_path = _resolve_source(row["cxr_path"], dataset)
                report_path = _resolve_source(negative["donor"]["report_path"], dataset)
                text = report_path.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    raise ValueError("empty hard-negative source report")
                prior = previous_rows[row["case_id"]]
                image_hash = sha256_file(image_path)
                if image_hash != prior["image_sha256"]:
                    raise ValueError("fixed image changed since the previous pilot")
                image_embedding = image_engine.get_projected_global_embedding(image_path)
                text_embedding = text_engine.get_embeddings_from_prompt(
                    [text], normalize=True, verbose=False)
                score = float((text_embedding @ image_embedding).detach().float().cpu().item())
                if not math.isfinite(score):
                    raise ValueError("invalid hard-negative score")
                records.append({
                    "case_id": row["case_id"], "split": row["split"],
                    "source_row_index": row["row_index"],
                    "donor_row_index": negative["donor"]["row_index"],
                    "image_sha256": image_hash,
                    "hard_report_sha256": sha256_file(report_path),
                    "matched_score": prior["matched_score"],
                    "random_score": prior["mismatched_score"],
                    "hard_score": round(score, 8),
                    "shared_positive_findings": negative["shared_positive_findings"],
                    "explicit_conflicting_findings": negative["explicit_conflicting_findings"],
                })
    return {
        "schema_version": SCHEMA,
        "status": "label_selected_hard_negatives_unadjudicated",
        "previous_scores_sha256": sha256_file(previous_path),
        "source_manifest_sha256": sha256_file(dataset / "manifest.csv"),
        "source_chexpert_labels_sha256": sha256_file(labels_path),
        "selection": {"same_split": True, "different_patient": True, "same_view": True,
                      "shared_positive_finding": True,
                      "at_least_one_opposite_explicit_finding": True,
                      "candidate_ranking": "fewest_conflicts_then_most_shared_then_most_agreements",
                      "report_derived_labels_not_image_ground_truth": True},
        "producer": previous["producer"],
        "summary": summarize(records, unavailable),
        "records": records,
        "clinical_hard_negative_validated": False,
        "probability_calibration_performed": False,
        "interpretation": "Same-view and shared-finding report-label challenge only; independent image/report adjudication remains required.",
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chexpert-labels", required=True)
    parser.add_argument("--previous-scores", required=True)
    parser.add_argument("--model-path", required=True)
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
                      "counts": {role: item["hard_negative_cases"]
                                 for role, item in payload["summary"].items()},
                      "output_sha256": output_hash,
                      "peak_vram_gib": payload["peak_vram_gib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

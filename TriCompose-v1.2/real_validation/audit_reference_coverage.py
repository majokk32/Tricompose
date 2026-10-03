#!/usr/bin/env python3
"""Slurm-only metadata coverage audit; no pixels, report text or inference.

Exclude previously scored patients, choose one frontal study per remaining
patient independently of labels, then freeze a label-stratified pilot cohort.
Explicit report-derived labels remain weak references, not image truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from biovil_matched_pairs import PROJECT, _load_manifest_rows
from contracts import (FINDING_STATES, PROTECTED_ROOT, commit_atomic_run,
                       discard_atomic_run, new_atomic_run, read_json,
                       require_inside, sha256_file, write_private_json,
                       write_private_text)
from hard_negative_biovil import FINDINGS, _id, load_report_label_index

SCHEMA = "tricompose-real-reference-coverage-v1"
REQUIRED_FINDINGS = ("Pneumonia",)


def prior_patients(rows, previous):
    if previous.get("schema_version") != "tricompose-real-xrv-weak-finding-check-v1":
        raise ValueError("unexpected prior pilot schema")
    excluded = set()
    seen_indices = set()
    for record in previous["records"]:
        index = record["source_row_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
            raise ValueError("invalid prior source index")
        if index in seen_indices or rows[index]["split"] != record["split"]:
            raise ValueError("duplicate or inconsistent prior source index")
        seen_indices.add(index)
        subject = _id(rows[index]["subject_id"])
        if subject in excluded:
            raise ValueError("prior pilot contains repeated patient")
        excluded.add(subject)
    return excluded


def patient_pool(rows, labels, excluded, *, seed):
    best, patient_splits = {}, {}
    for index, row in enumerate(rows):
        split = str(row.get("split", ""))
        if split not in {"train", "val", "test"}:
            raise ValueError("unexpected dataset split")
        subject = _id(row["subject_id"])
        if subject in patient_splits and patient_splits[subject] != split:
            raise ValueError("patient crosses dataset splits")
        patient_splits[subject] = split
        if (split == "train" or subject in excluded or
                str(row.get("ViewPosition", "")).upper() not in {"AP", "PA"} or
                not str(row.get("cxr_path", "")).strip() or
                not str(row.get("report_path", "")).strip()):
            continue
        study = _id(row["study_id"])
        rank = hashlib.sha256(f"{seed}|{split}|{subject}|{study}|{index}".encode()).hexdigest()
        states = labels.get((subject, study), dict.fromkeys(FINDINGS, "unknown"))
        if set(states) != FINDINGS or any(state not in FINDING_STATES for state in states.values()):
            raise ValueError("invalid reference finding inventory")
        candidate = {"split": split, "source_row_index": index,
                     "reference_states": dict(states), "rank": rank}
        # Select the study without using its labels or any model scores.
        if subject not in best or rank < best[subject]["rank"]:
            best[subject] = candidate
    return sorted(best.values(), key=lambda row: (row["split"], row["rank"]))


def counts(records):
    result = {name: dict.fromkeys(sorted(FINDING_STATES), 0) for name in sorted(FINDINGS)}
    for row in records:
        for name, state in row["reference_states"].items():
            result[name][state] += 1
    return result


def select_coverage_subset(pool, *, min_per_class, max_cases):
    """Cover predeclared binary quotas; scores and synthetic outputs are absent."""
    if min_per_class < 1 or max_cases < 1:
        raise ValueError("positive selection limits required")
    available = counts(pool)
    target_findings = [name for name in sorted(FINDINGS)
                       if min(available[name]["positive"], available[name]["negative"]) >= min_per_class]
    achieved = {name: {state: 0 for state in ("positive", "negative")} for name in target_findings}
    remaining = sorted(pool, key=lambda row: row["rank"])
    chosen = []
    while len(chosen) < max_cases and remaining:
        def gain(row):
            return sum(row["reference_states"][name] in {"positive", "negative"} and
                       achieved[name][row["reference_states"][name]] < min_per_class
                       for name in target_findings)
        position = min(range(len(remaining)), key=lambda i: (-gain(remaining[i]), remaining[i]["rank"]))
        if gain(remaining[position]) == 0:
            break
        row = remaining.pop(position)
        chosen.append(row)
        for name in target_findings:
            state = row["reference_states"][name]
            if state in {"positive", "negative"}:
                achieved[name][state] += 1
    selected_counts = counts(chosen)
    detail = {}
    for name in sorted(FINDINGS):
        enough = min(selected_counts[name]["positive"], selected_counts[name]["negative"]) >= min_per_class
        source_enough = name in target_findings
        detail[name] = {"available": available[name], "selected": selected_counts[name],
                        "quota_satisfied": enough,
                        "status": ("quota_satisfied" if enough else
                                   "source_insufficient_explicit_binary_support" if not source_enough else
                                   "selection_budget_exhausted")}
    return sorted(chosen, key=lambda row: row["rank"]), detail


def build_plan(pool, *, min_per_class, max_cases):
    records, split_summaries = [], {}
    for split in ("val", "test"):
        current = [row for row in pool if row["split"] == split]
        selected, finding_counts = select_coverage_subset(
            current, min_per_class=min_per_class, max_cases=max_cases)
        split_summaries[split] = {"available_distinct_patients": len(current),
                                 "selected_distinct_patients": len(selected),
                                 "findings": finding_counts}
        for row in selected:
            records.append({"case_id": f"case_{len(records):04d}", "split": split,
                            "source_row_index": row["source_row_index"],
                            "reference_states": row["reference_states"]})
    ready = all(split_summaries[split]["findings"][name]["quota_satisfied"]
                for split in ("val", "test") for name in REQUIRED_FINDINGS)
    summary = {"schema_version": SCHEMA,
               "status": ("ready_for_image_side_weak_reference_pilot" if ready else
                          "blocked_insufficient_explicit_pneumonia_reference"),
               "min_per_class": min_per_class, "max_cases_per_split": max_cases,
               "required_findings": list(REQUIRED_FINDINGS), "splits": split_summaries,
               "reference_kind": "report_extracted_weak_labels",
               "label_stratified_cohort": True, "natural_prevalence_estimate": False,
               "primary_metric_eligible": False, "independent_image_ground_truth": False,
               "image_artifact_existence_checked": False, "model_inference_used": False,
               "real_pixels_or_report_text_read": False}
    return records, summary


def markdown(summary):
    lines = ["# New real-reference coverage audit", "", f"Status: {summary['status']}", "",
             "Report-derived weak references only; not independent image truth or clinical validation.",
             "Previously scored patients excluded. One label-independent frontal study per remaining patient.",
             "Then label-stratified selection within the existing patient-disjoint val/test splits.",
             "No pixels, report text, inference, score tuning or synthetic output inspection.", "",
             "| Split | Finding | Available + / - | Selected + / - | Quota met |",
             "|---|---|---:|---:|---|" ]
    for split, subset in summary["splits"].items():
        for name, row in subset["findings"].items():
            available, selected = row["available"], row["selected"]
            lines.append(f"| {split} | {name} | {available['positive']} / {available['negative']} | "
                         f"{selected['positive']} / {selected['negative']} | {row['quota_satisfied']} |")
    lines += ["", f"Minimum: {summary['min_per_class']} explicit positives AND negatives per finding per split.",
              "Unknown and uncertain are excluded from both binary classes, never filled as negative.",
              "The cohort is fixed before any new model scores. It is diagnostic and label-stratified,",
              "not a natural-prevalence sample or paper-primary final cohort. Quotas do not validate a scorer.",
              "Image existence, image hashes and clinical reference validity remain for separate approved checks."]
    return "\n".join(lines) + "\n"


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before real metadata reads")
    dataset = Path(args.dataset_root).resolve(strict=True)
    if not dataset.is_dir() or not dataset.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("dataset outside read-only project")
    labels_path = Path(args.chexpert_labels).resolve(strict=True)
    previous_path = require_inside(args.previous_scores, PROTECTED_ROOT, must_exist=True)
    previous = read_json(previous_path)
    source_hash = sha256_file(dataset / "manifest.csv")
    label_hash = sha256_file(labels_path)
    if (previous["source_manifest_sha256"] != source_hash or
            previous["source_chexpert_labels_sha256"] != label_hash):
        raise ValueError("prior/source fingerprints changed")
    rows = _load_manifest_rows(dataset)
    excluded = prior_patients(rows, previous)
    pool = patient_pool(rows, load_report_label_index(labels_path), excluded, seed=args.seed)
    records, summary = build_plan(pool, min_per_class=args.min_per_class, max_cases=args.max_cases_per_split)
    summary["previous_distinct_patients_excluded"] = len(excluded)
    source_hashes = {"dataset_manifest": source_hash, "chexpert_labels": label_hash,
                     "previous_scores": sha256_file(previous_path), "program": sha256_file(__file__),
                     "selection_source": sha256_file(Path(__file__).with_name("biovil_matched_pairs.py")),
                     "label_parser_source": sha256_file(Path(__file__).with_name("hard_negative_biovil.py"))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "cohort.json", {
            "schema_version": "tricompose-fixed-real-reference-cohort-v1",
            "source_hashes": source_hashes,
            "selection": {"seed": args.seed, "min_per_class": args.min_per_class,
                          "max_cases_per_split": args.max_cases_per_split,
                          "one_study_per_patient": True, "patient_disjoint_splits": True,
                          "previous_patient_exclusion_verified": True, "label_stratified": True,
                          "required_findings": list(REQUIRED_FINDINGS)},
            "primary_metric_eligible": False, "records": records}),
            write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "summary.md", markdown(summary))]
        write_private_json(temporary / "manifest.json", {
            "schema_version": SCHEMA, "run_id": args.run_id, "status": summary["status"],
            "source_hashes": source_hashes,
            "counts": {split: row["selected_distinct_patients"] for split, row in summary["splits"].items()},
            "artifacts": {file.name: {"sha256": sha256_file(file)} for file in files}})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return sha256_file(target / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chexpert-labels", required=True)
    parser.add_argument("--previous-scores", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=2711)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--max-cases-per-split", type=int, default=512)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    os.umask(0o007)
    try:
        digest = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed", "manifest_sha256": digest,
                      "model_inference_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

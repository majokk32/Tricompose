#!/usr/bin/env python3
"""Post-hoc paired analysis of blinded BioViL intervention scores.

This process sees the protected intervention key only after scorer inference.
It measures score sensitivity, not clinical correctness or fault localization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from build_intervention_smoke import file_sha256
from build_targeted_interventions import validate_items
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, require_inside, sha256_file,
                       write_private_json)
from score_intervention_biovil import BANK_SCHEMA, SCHEMA as SCORE_SCHEMA

SCHEMA = "tricompose-v12-intervention-biovil-paired-analysis-v1"


def paired_summary(items: dict[str, list[dict]], scores: list[dict]) -> dict:
    validate_items(items)
    score_by_id = {}
    for row in scores:
        item_id = row["item_id"]
        value = row["biovil_raw_cosine"]
        if item_id in score_by_id or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("duplicate or invalid BioViL item score")
        score_by_id[item_id] = float(value)
    truth = {row["item_id"]: row for row in items["intervention_key"]}
    resolver = {row["item_id"]: row for row in items["resolver"]}
    if set(score_by_id) != set(truth):
        raise ValueError("missing or extra scored intervention item")
    controls = {}
    for item_id, answer in truth.items():
        if answer["intervention_type"] == "no_corruption":
            case = resolver[item_id]["case_id"]
            if case in controls:
                raise ValueError("multiple controls for one EHR case")
            controls[case] = score_by_id[item_id]
    groups = {}
    for item_id, answer in truth.items():
        arm = answer["intervention_type"]
        if arm == "no_corruption":
            continue
        displayed = resolver[item_id]
        case = displayed["case_id"]
        if case not in controls:
            raise ValueError("swap has no untouched control")
        delta = controls[case] - score_by_id[item_id]
        for split in ("overall", displayed["benchmark_split"]):
            groups.setdefault(split, {}).setdefault(arm, []).append(delta)
    result = {}
    for split in ("overall", "development", "calibration", "final_test"):
        result[split] = {}
        for arm in ("report_swap", "cxr_swap"):
            values = groups.get(split, {}).get(arm, [])
            if not values:
                result[split][arm] = {"pairs": 0, "mean_control_minus_swap": None,
                                      "median_control_minus_swap": None,
                                      "fraction_score_decreased": None,
                                      "fraction_score_increased": None}
                continue
            result[split][arm] = {
                "pairs": len(values),
                "mean_control_minus_swap": round(statistics.mean(values), 8),
                "median_control_minus_swap": round(statistics.median(values), 8),
                "fraction_score_decreased": round(sum(x > 0 for x in values) / len(values), 8),
                "fraction_score_increased": round(sum(x < 0 for x in values) / len(values), 8),
            }
    result["coverage"] = {
        "cases": len(controls),
        "arm_counts": dict(Counter(row["intervention_type"] for row in truth.values())),
    }
    return result


def run(args: argparse.Namespace) -> dict:
    bank = require_inside(args.bank_run, PROTECTED_ROOT, must_exist=True)
    scored = require_inside(args.scores_run, PROTECTED_ROOT, must_exist=True)
    manifest_path = require_inside(bank / "manifest.json", bank, must_exist=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BANK_SCHEMA:
        raise ValueError("wrong intervention bank schema")
    items = {}
    for name in ("blind_items", "resolver", "intervention_key"):
        path = require_inside(bank / f"{name}.jsonl", bank, must_exist=True)
        if file_sha256(path) != manifest.get("artifact_sha256", {}).get(name):
            raise ValueError("intervention bank artifact changed")
        items[name] = [json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines() if line]
    score_path = require_inside(scored / "scores.json", scored, must_exist=True)
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCORE_SCHEMA or payload.get("scorer_read_intervention_key") is not False:
        raise ValueError("scores are not from the blinded scorer")
    if payload.get("source", {}).get("bank_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("score source differs from intervention bank")
    # The scorer's item identity and displayed-artifact hashes must still
    # match the protected resolver before the answer key is used.
    indexed = {row["item_id"]: row for row in items["resolver"]}
    for row in payload["records"]:
        source = indexed.get(row["item_id"])
        if source is None or any(
            row[name] != source[name]
            for name in ("case_id", "benchmark_split", "displayed_cxr_sha256",
                         "displayed_report_sha256")
        ):
            raise ValueError("scorer output does not match blinded resolver")
    summary = paired_summary(items, payload["records"])
    return {
        "schema_version": SCHEMA,
        "status": "exploratory_paired_score_sensitivity_only",
        "source_bank_manifest_sha256": file_sha256(manifest_path),
        "source_scores_sha256": file_sha256(score_path),
        "summary": summary,
        "clinical_localization_accuracy": None,
        "clinical_truth_available": False,
        "paper_final_test_independent": False,
        "score_used_for_donor_selection": False,
        "same_modality_labels_used_for_donor_selection": True,
        "interpretation": "Paired BioViL cosine change after artifact swaps. This does not identify the faulty modality or establish clinical mismatch.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-run", required=True)
    parser.add_argument("--scores-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    temporary = None
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = write_private_json(temporary / "analysis.json", payload)
        output_hash = sha256_file(output)
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_exploratory", "run_id": args.run_id,
                      "counts": payload["summary"]["coverage"],
                      "output_sha256": output_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

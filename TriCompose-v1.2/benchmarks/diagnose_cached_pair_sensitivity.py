#!/usr/bin/env python3
"""CPU-only manipulation check using existing frozen per-artifact labels.

This is neither image-text inference nor clinical error localization. It only
asks whether cached CXR/report label agreement changes after artifact swaps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from build_intervention_smoke import file_sha256, inside_protected

WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
sys.path.insert(0, str(WORKSPACE / "TriCompose-v1.0/eval/report_v1_1"))
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       write_private_json)  # noqa: E402
from crossmodal_metrics import score_state_pair  # noqa: E402

SCHEMA = "tricompose-v12-cached-pair-sensitivity-v1"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line]


def _labels(path: Path, *, id_key: str, hash_key: str) -> dict[str, dict]:
    source = inside_protected(path, existing=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("producer", {}).get("frozen") is not True:
        raise ValueError("label bundle must come from a frozen producer")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("label bundle lacks records")
    result = {}
    for row in records:
        item_id = row.get(id_key)
        if not isinstance(item_id, str) or item_id in result:
            raise ValueError("missing or duplicate label candidate ID")
        result[item_id] = {"sha256": row[hash_key],
                           "states": row["finding_states"]}
    return result


def summarize(items: list[dict], truth: list[dict],
              cxr_labels: dict[str, dict], report_labels: dict[str, dict]) -> dict:
    resolver = {row["item_id"]: row for row in items}
    if len(resolver) != len(items) or {row["item_id"] for row in truth} != set(resolver):
        raise ValueError("benchmark item sets do not match")
    scored = {}
    for row in items:
        image = cxr_labels[row["displayed_cxr_candidate_id"]]
        report = report_labels[row["displayed_report_candidate_id"]]
        if (image["sha256"] != row["displayed_cxr_sha256"] or
                report["sha256"] != row["displayed_report_sha256"]):
            raise ValueError("cached label hash does not match intervention artifact")
        scored[row["item_id"]] = score_state_pair(image["states"], report["states"])
    base_by_triple = {}
    for record in truth:
        if record["intervention_type"] == "no_corruption":
            base_by_triple[record["base_triple_candidate_id"]] = scored[record["item_id"]]
    arms = defaultdict(Counter)
    for record in truth:
        arm = arms[record["intervention_type"]]
        score = scored[record["item_id"]]
        base = base_by_triple[record["base_triple_candidate_id"]]
        arm["items"] += 1
        arm["explicit_contradictions"] += score["contradiction_count"]
        arm["support"] += score["support_count"]
        arm["known_reference_facts"] += score["known_reference_fact_count"]
        if score["known_reference_fact_count"] == 0:
            arm["no_known_image_facts"] += 1
        if score["contradiction_count"] > base["contradiction_count"]:
            arm["more_contradictions_than_control"] += 1
        elif score["contradiction_count"] < base["contradiction_count"]:
            arm["fewer_contradictions_than_control"] += 1
        else:
            arm["same_contradictions_as_control"] += 1
        if score["support_recall"] is not None and base["support_recall"] is not None:
            if score["support_recall"] < base["support_recall"]:
                arm["lower_support_recall_than_control"] += 1
            elif score["support_recall"] > base["support_recall"]:
                arm["higher_support_recall_than_control"] += 1
            else:
                arm["same_support_recall_as_control"] += 1
    return {name: dict(sorted(counts.items())) for name, counts in sorted(arms.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-run", required=True)
    parser.add_argument("--cxr-labels", required=True)
    parser.add_argument("--report-labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    benchmark = inside_protected(args.benchmark_run, existing=True)
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tricompose-v12-intervention-smoke-v1":
        raise ValueError("unsupported benchmark manifest")
    for stem in ("resolver", "intervention_key"):
        if file_sha256(benchmark / f"{stem}.jsonl") != manifest["artifact_sha256"][stem]:
            raise ValueError("benchmark file hash mismatch")
    cxr_path = inside_protected(args.cxr_labels, existing=True)
    report_path = inside_protected(args.report_labels, existing=True)
    rows = summarize(
        _jsonl(benchmark / "resolver.jsonl"),
        _jsonl(benchmark / "intervention_key.jsonl"),
        _labels(cxr_path, id_key="cxr_candidate_id", hash_key="image_sha256"),
        _labels(report_path, id_key="report_candidate_id", hash_key="report_sha256"),
    )
    payload = {
        "schema_version": SCHEMA,
        "status": "diagnostic_cached_labels_not_clinical_localization",
        "benchmark_manifest_sha256": file_sha256(benchmark / "manifest.json"),
        "cxr_labels_sha256": file_sha256(cxr_path),
        "report_labels_sha256": file_sha256(report_path),
        "calibrated_primary": False,
        "clinical_mismatch_adjudicated": False,
        "arms": rows,
        "interpretation": "CXR-report label sensitivity only; cannot attribute which modality is wrong or establish clinical truth.",
    }
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / "summary.json", payload)
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(json.dumps({"status": payload["status"], "run": str(target),
                      "arm_counts": {name: data["items"] for name, data in rows.items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

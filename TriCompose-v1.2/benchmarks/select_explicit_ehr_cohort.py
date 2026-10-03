#!/usr/bin/env python3
"""Predeclared, CPU-only eligibility screen for a NEW synthetic EHR pool.

This is unconditional generation followed by selection, not prompted EHR
generation. Source EHRs are never changed. Only aggregate counts leave the
protected output; clinical case-level evidence stays in the protected run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
from tricompose_v11.facts import extract_v11_facts


WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROTECTED = WORKSPACE / "artifacts" / "protected"
OUTPUT_PARENT = PROTECTED / "tricompose_v1_2" / "ehr_cohorts"
SOURCE_SCHEMA = "tricompose.synehrgy_v2.run.v1"
RULE_VERSION = "latest_diagnosis_direct_positive_v1"
CASE_PATTERN = re.compile(r"case_[0-9]{3,6}\Z")
RUN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")

# Frozen before inspecting the new pool. These are direct observable findings,
# not clinical indications, inferred consequences, or device proxies.
ELIGIBLE_FACTS = (
    "cardiomegaly",
    "pleural_effusion",
    "pulmonary_edema",
    "pneumonia",
    "pneumothorax",
    "atelectasis",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _inside(path: Path, root: Path, *, exists: bool) -> Path:
    result = path.resolve(strict=exists)
    if not result.is_relative_to(root.resolve(strict=True)):
        raise ValueError("path escapes protected workspace")
    return result


def eligible_positive_facts(facts: Mapping[str, Any]) -> tuple[str, ...]:
    """Require an explicit positive finding grounded in a diagnosis field."""
    direct = facts.get("direct_facts")
    if not isinstance(direct, Mapping):
        raise ValueError("V1.1 direct facts are missing")
    result = []
    for fact_id in ELIGIBLE_FACTS:
        record = direct.get(fact_id)
        if not isinstance(record, Mapping):
            raise ValueError("required direct fact is missing")
        fields = record.get("source_fields")
        evidence = record.get("evidence")
        if (
            record.get("state") == "positive"
            and isinstance(fields, list)
            and isinstance(evidence, list)
            and evidence
            and any(isinstance(field, str) and field.startswith("diagnoses[") for field in fields)
        ):
            result.append(fact_id)
    return tuple(result)


def first_eligible(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """No quality ranking, post-hoc disease balancing, or retry selection."""
    if target < 1:
        raise ValueError("target must be positive")
    return [row for row in rows if row["eligible"]][:target]


def _write(path: Path, data: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
    os.chmod(path, 0o660)


def screen_pool(
    *, source_run: Path, expected_source_run_id: str, output_run_id: str,
    expected_count: int = 100, expected_base_seed: int = 5200, target: int = 2,
) -> dict[str, Any]:
    if not RUN_PATTERN.fullmatch(expected_source_run_id) or not RUN_PATTERN.fullmatch(output_run_id):
        raise ValueError("run IDs must be opaque and filesystem-safe")
    if target < 1 or expected_count < target:
        raise ValueError("invalid target or expected count")
    source = _inside(source_run, PROTECTED, exists=True)
    manifest_path = source / "run.json"
    manifest = _read_object(manifest_path)
    generation = manifest.get("generation", {})
    if manifest.get("schema") != SOURCE_SCHEMA or manifest.get("run_id") != expected_source_run_id:
        raise ValueError("unexpected source run")
    if manifest.get("generator", {}).get("variant") != "qwen2-40bins":
        raise ValueError("source is not the frozen Qwen2-40bins generator")
    if manifest.get("privacy", {}).get("real_patient_input_used") is not False:
        raise ValueError("source must be fully synthetic")
    if (
        generation.get("input") != "bos_token_only"
        or generation.get("count") != expected_count
        or generation.get("base_seed") != expected_base_seed
    ):
        raise ValueError("generation contract differs from predeclared protocol")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("source case count differs from protocol")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("invalid source case entry")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or not CASE_PATTERN.fullmatch(case_id) or case_id in seen:
            raise ValueError("invalid or duplicated case ID")
        seen.add(case_id)
        relative = entry.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("case path is missing")
        path = _inside(source / relative, source, exists=True)
        if not path.is_file() or path.parent != source / "cases" or path.name != f"{case_id}.json":
            raise ValueError("source case path is not the expected opaque file")
        case_hash = _hash_file(path)
        if case_hash != entry.get("sha256"):
            raise ValueError("source case SHA256 mismatch")
        case = _read_object(path)
        if case.get("case_id") != case_id:
            raise ValueError("source case ID mismatch")
        if case.get("validation", {}).get("strict_valid") is True:
            canonical = canonicalize_synehrgy_case(case, source_model_id="synehrgy_qwen2_40bins")
            positive = eligible_positive_facts(extract_v11_facts(canonical))
            reason = "eligible" if positive else "no_explicit_latest_diagnosis_finding"
        else:
            positive = ()
            reason = "structurally_invalid"
        rows.append({
            "source_index": index,
            "case_id": case_id,
            "source_sha256": case_hash,
            "eligible": bool(positive),
            "eligible_fact_ids": list(positive),
            "screen_reason": reason,
        })

    chosen = first_eligible(rows, target)
    parent = _inside(OUTPUT_PARENT, PROTECTED, exists=False)
    parent.mkdir(parents=True, exist_ok=True, mode=0o2770)
    os.chmod(parent, 0o2770)
    destination = _inside(parent / output_run_id, PROTECTED, exists=False)
    if destination.exists():
        raise FileExistsError("cohort run already exists")
    temporary = parent / f".{output_run_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o2770)
    os.chmod(temporary, 0o2770)
    try:
        counts = Counter(row["screen_reason"] for row in rows)
        fact_counts = Counter(fact for row in rows for fact in row["eligible_fact_ids"])
        result = {
            "schema_version": "tricompose-v1.2-explicit-ehr-screen-v1",
            "run_id": output_run_id,
            "source_run": str(source),
            "source_run_sha256": _hash_file(manifest_path),
            "rule_version": RULE_VERSION,
            "generation_is_prompt_conditioned": False,
            "selection_is_post_generation": True,
            "selection_order": "source_manifest_order_first_eligible",
            "all_generated_cases_retained_in_source": True,
            "source_count": expected_count,
            "target_count": target,
            "eligible_count": counts["eligible"],
            "selected_count": len(chosen),
            "status": "ready_for_two_case_staging" if len(chosen) == target else "insufficient_eligible_cases",
            "screen_reason_counts": dict(sorted(counts.items())),
            "eligible_fact_counts": dict(sorted(fact_counts.items())),
            "selected_case_ids": [row["case_id"] for row in chosen],
            "no_training_or_inference_in_screen": True,
        }
        _write(temporary / "screen_manifest.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        _write(temporary / "screen_rows.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        if len(chosen) == target:
            _write(temporary / "selected_case_ids.txt", "".join(row["case_id"] + "\n" for row in chosen))
        os.rename(temporary, destination)
    except Exception:
        # A failed temporary run contains only synthetic records; keep it for
        # manual audit rather than recursively deleting an uncertain target.
        raise
    return {
        "status": result["status"],
        "source_count": expected_count,
        "eligible_count": result["eligible_count"],
        "selected_count": result["selected_count"],
        "screen_manifest_sha256": _hash_file(destination / "screen_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--expected-source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--expected-base-seed", type=int, default=5200)
    parser.add_argument("--target", type=int, default=2)
    args = parser.parse_args()
    result = screen_pool(**vars(args))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ready_for_two_case_staging" else 2


if __name__ == "__main__":
    raise SystemExit(main())

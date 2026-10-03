#!/usr/bin/env python3
"""Slurm-only frozen SynEHRgy generation with project-group protected output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from tricompose_synehrgy_v2.model import FrozenSynEHRgyRuntime
from tricompose_synehrgy_v2.structure import parse_token_sequence


WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROTECTED = WORKSPACE / "artifacts" / "protected"
OUTPUT_PARENT = PROTECTED / "tricompose_v1_2" / "ehr_pools"
RUN_SCHEMA = "tricompose.synehrgy_v2.run.v1"
CASE_SCHEMA = "tricompose.synehrgy_v2.synthetic_ehr.v1"
RUN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mkdir(path: Path) -> None:
    path.mkdir(mode=0o2770)
    os.chmod(path, 0o2770)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o660)


def generate_pool(
    *, model_dir: Path, run_id: str, count: int = 100,
    base_seed: int = 5200, temperature: float = 1.0, top_k: int = 50,
) -> dict[str, Any]:
    if not RUN_PATTERN.fullmatch(run_id) or count != 100 or base_seed != 5200:
        raise ValueError("run ID or frozen pool size/seed differs from protocol")
    if temperature != 1.0 or top_k != 50:
        raise ValueError("sampling parameters differ from frozen protocol")
    if not PROTECTED.is_dir() or not OUTPUT_PARENT.parent.is_dir():
        raise ValueError("protected project boundary is unavailable")
    OUTPUT_PARENT.mkdir(mode=0o2770, exist_ok=True)
    os.chmod(OUTPUT_PARENT, 0o2770)
    destination = OUTPUT_PARENT / run_id
    if destination.exists():
        raise FileExistsError("pool run already exists")
    temporary = OUTPUT_PARENT / f".{run_id}.{uuid.uuid4().hex}.tmp"
    _mkdir(temporary)
    cases_dir = temporary / "cases"
    _mkdir(cases_dir)

    runtime = FrozenSynEHRgyRuntime(model_dir, "qwen2-40bins")
    started = time.monotonic()
    lengths: list[int] = []
    seconds: list[float] = []
    strict_valid = 0
    entries: list[dict[str, Any]] = []
    for index in range(count):
        case_id = f"case_{index:03d}"
        tokens, elapsed = runtime.generate(
            seed=base_seed + index, temperature=temperature, top_k=top_k,
        )
        parsed = parse_token_sequence(tokens)
        valid = parsed.validation["strict_valid"] is True
        strict_valid += int(valid)
        lengths.append(len(tokens))
        seconds.append(elapsed)
        payload = {
            "schema": CASE_SCHEMA,
            "case_id": case_id,
            "generator": runtime.audit,
            "generation": {
                "cold_start": True,
                "input": "bos_token_only",
                "seed": base_seed + index,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": 1.0,
                "token_count": len(tokens),
                "elapsed_seconds": elapsed,
                "numeric_representation": "official_discrete_bin_tokens",
            },
            "raw_tokens": tokens,
            "structure": parsed.structure,
            "validation": parsed.validation,
        }
        path = cases_dir / f"{case_id}.json"
        _write_json(path, payload)
        entries.append({
            "case_id": case_id,
            "relative_path": f"cases/{case_id}.json",
            "sha256": _hash_file(path),
            "token_count": len(tokens),
            "visit_count": parsed.validation["visit_count"],
            "strict_valid": valid,
        })

    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "generator": runtime.audit,
        "privacy": {
            "real_patient_input_used": False,
            "external_api_used": False,
            "opaque_case_ids": True,
            "protected_output": True,
        },
        "generation": {
            "count": count,
            "base_seed": base_seed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": 1.0,
            "cold_start": True,
            "input": "bos_token_only",
            "numeric_representation": "official_discrete_bin_tokens",
        },
        "aggregate": {
            "strict_valid_count": strict_valid,
            "token_length_min": min(lengths),
            "token_length_median": statistics.median(lengths),
            "token_length_max": max(lengths),
            "generation_seconds_mean": statistics.mean(seconds),
            "total_seconds": time.monotonic() - started,
            "peak_vram_gib": runtime.peak_vram_gib,
        },
        "cases": entries,
    }
    _write_json(temporary / "run.json", manifest)
    os.rename(temporary, destination)
    return {
        "status": "complete",
        "count": count,
        "strict_valid_count": strict_valid,
        "total_seconds": round(manifest["aggregate"]["total_seconds"], 3),
        "peak_vram_gib": round(runtime.peak_vram_gib, 3),
        "manifest_sha256": _hash_file(destination / "run.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    result = generate_pool(
        model_dir=args.model_dir, run_id=args.run_id, count=args.count,
        base_seed=args.seed, temperature=args.temperature, top_k=args.top_k,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

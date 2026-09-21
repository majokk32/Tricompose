#!/usr/bin/env python3
"""Score V1 synthetic CXR candidates with one frozen XRV checkpoint.

This program performs model inference and therefore must only run inside an
approved Slurm allocation.  It never opens a real/source patient image.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import require_private_file, sha256_file, write_private_json
from tricompose_v1.scoring import (
    CALIBRATION_STATUS,
    XRV_SCORE_SCHEMA,
    commit_atomic_protected_run,
    discard_atomic_protected_run,
    ehr_prompt_intent_states,
    load_candidate_bank,
    load_ehr_facts,
    new_atomic_protected_run,
    xrv_ehr_support,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-name", default="densenet121-res224-all")
    parser.add_argument(
        "--weight-filename",
        default=(
            "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-"
            "tw-lr001-rot45-tr15-sc15-seed0-best.pt"
        ),
    )
    return parser


def _nondegeneracy(image_path: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as handle:
        array = np.asarray(handle.convert("L"), dtype=np.float32)
        dimensions = [int(handle.size[0]), int(handle.size[1])]
    finite = bool(np.isfinite(array).all())
    if finite:
        standard_deviation = float(array.std())
        low, high = np.percentile(array, [1.0, 99.0])
        dynamic_range = float(high - low)
    else:
        standard_deviation = 0.0
        dynamic_range = 0.0
    hard_pass = (
        finite
        and min(dimensions) >= 224
        and standard_deviation >= 2.0
        and dynamic_range >= 10.0
    )
    return {
        "metric": "deterministic_image_nondegeneracy_not_realism",
        "hard_gate_pass": hard_pass,
        "score": 1.0 if hard_pass else 0.0,
        "dimensions": dimensions,
        "pixel_standard_deviation": round(standard_deviation, 8),
        "p01_p99_dynamic_range": round(dynamic_range, 8),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from medim_ehr_prompt.consistency import FrozenXRVRuntime

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this program through Slurm")
    bank = load_candidate_bank(args.candidate_bank)
    cache_dir = Path(args.cache_dir).resolve(strict=True)
    weight = (cache_dir / args.weight_filename).resolve(strict=True)
    if not weight.is_file():
        raise ValueError("frozen XRV checkpoint is missing")

    runtime = FrozenXRVRuntime(
        {
            "consistency": {
                "cache_dir": str(cache_dir),
                "weight_filename": args.weight_filename,
                "model_name": args.model_name,
            }
        }
    )
    if runtime.model.training or any(
        parameter.requires_grad for parameter in runtime.model.parameters()
    ):
        raise RuntimeError("XRV verifier is not frozen")

    intent_by_ehr: dict[str, dict[str, str]] = {}
    for ehr in bank.ehr_candidates:
        intent_by_ehr[str(ehr["candidate_id"])] = ehr_prompt_intent_states(
            load_ehr_facts(ehr)
        )

    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    for cxr in sorted(bank.cxr_candidates, key=lambda row: str(row["candidate_id"])):
        artifact = cxr["artifact"]
        image_path = require_private_file(artifact["path"])
        if sha256_file(image_path) != artifact["sha256"]:
            raise ValueError("CXR image hash changed after candidate-bank finalization")
        probabilities = runtime.predict(image_path)
        parent = str(cxr["parent_ids"][0])
        records.append(
            {
                "cxr_candidate_id": cxr["candidate_id"],
                "parent_ehr_candidate_id": parent,
                "image_sha256": artifact["sha256"],
                "xrv_probabilities": probabilities,
                "ehr_cxr_support": xrv_ehr_support(
                    intent_by_ehr[parent], probabilities
                ),
                "cxr_quality": _nondegeneracy(image_path),
            }
        )

    return {
        "schema_version": XRV_SCORE_SCHEMA,
        "status": "completed_uncalibrated",
        "calibration_status": CALIBRATION_STATUS,
        "source_candidate_bank": {
            "path": str(bank.root),
            "manifest_sha256": bank.manifest_sha256,
        },
        "verifier": {
            "model_name": args.model_name,
            "frozen": True,
            "checkpoint_sha256": sha256_file(weight),
            "checkpoint_size_bytes": weight.stat().st_size,
            "probability_contract": (
                "raw frozen XRV outputs; no synthetic-domain operating point "
                "has been calibrated"
            ),
        },
        "records": records,
        "counts": {"cxr_candidates": len(records), "model_calls": len(records)},
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o077)
    started = time.monotonic()
    temp, target = new_atomic_protected_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        path = write_private_json(temp / "scores.json", payload)
        digest = sha256_file(path)
        commit_atomic_protected_run(temp, target)
    except Exception:
        discard_atomic_protected_run(temp)
        raise
    print(
        json.dumps(
            {
                "stage": "tricompose_v1_xrv_score",
                "status": "ok",
                "run_id": args.run_id,
                "candidate_count": payload["counts"]["cxr_candidates"],
                "artifact_sha256": digest,
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_vram_gib": payload["peak_vram_gib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

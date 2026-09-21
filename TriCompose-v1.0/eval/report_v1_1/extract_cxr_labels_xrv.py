#!/usr/bin/env python3
"""Extract 14 finding states from V1.1 CXRs with frozen XRV DenseNet.

This is GPU inference and must run only through an approved Slurm allocation.
Default 0.5 thresholds are explicitly diagnostic and not paper-calibrated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from contracts import (
    CHEXPERT_FINDINGS,
    PROTECTED_ROOT,
    commit_atomic_run,
    discard_atomic_run,
    load_cxr_candidates,
    new_atomic_run,
    read_json,
    require_inside,
    sha256_file,
    write_private_json,
)


SCHEMA_VERSION = "tricompose-cxr-finding-labels-v1.1"
THRESHOLD_SCHEMA = "tricompose-xrv-thresholds-v1"
XRV_LABELS = {
    "atelectasis": ("atelectasis",),
    "cardiomegaly": ("cardiomegaly",),
    "consolidation": ("consolidation",),
    "edema": ("edema",),
    "enlarged_cardiomediastinum": ("enlarged_cardiomediastinum",),
    "fracture": ("fracture",),
    "lung_lesion": ("lung_lesion", "nodule", "mass"),
    "lung_opacity": ("lung_opacity", "infiltration"),
    "pleural_effusion": ("effusion",),
    "pleural_other": ("pleural_thickening",),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax",),
    "support_devices": (),
    "no_finding": (),
}


def _normalized_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


class FrozenXRVRuntime:
    """Self-contained frozen TorchXRayVision runtime for protected CXRs."""

    def __init__(self, *, cache_dir: Path, weight_filename: str, model_name: str) -> None:
        import torch
        import torchxrayvision as xrv

        weight = (cache_dir / weight_filename).resolve(strict=True)
        if not weight.is_file():
            raise ValueError("frozen XRV checkpoint is missing")
        self.torch = torch
        self.xrv = xrv
        self.device = torch.device("cuda:0")
        self.model = xrv.models.DenseNet(
            weights=model_name,
            cache_dir=str(cache_dir),
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        self.crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(224)
        self.pathologies = [_normalized_label(name) for name in self.model.pathologies]

    def predict(self, image_path: Path) -> dict[str, float]:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        array = self.xrv.datasets.normalize(array, 255)
        array = self.crop(array[None, :, :])
        array = self.resize(array)
        tensor = self.torch.from_numpy(array).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)[0].detach().float().cpu().numpy()
        if len(output) != len(self.pathologies) or not np.isfinite(output).all():
            raise ValueError("invalid TorchXRayVision output")
        return {
            label: round(float(value), 8)
            for label, value in zip(self.pathologies, output, strict=True)
            if label
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--weight-filename",
        default=(
            "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-"
            "tw-lr001-rot45-tr15-sc15-seed0-best.pt"
        ),
    )
    parser.add_argument("--model-name", default="densenet121-res224-all")
    parser.add_argument(
        "--thresholds",
        help="Optional protected per-finding calibrated threshold JSON.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _thresholds(path: str | None) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if path is None:
        values = {
            finding: {"negative_max": 0.5, "positive_min": 0.5}
            for finding in CHEXPERT_FINDINGS
        }
        return values, {
            "status": "uncalibrated_default_0_5",
            "primary_metric_eligible": False,
            "source_sha256": None,
        }
    source = require_inside(path, PROTECTED_ROOT, must_exist=True)
    payload = read_json(source)
    if payload.get("schema_version") != THRESHOLD_SCHEMA:
        raise ValueError("unsupported XRV threshold schema")
    findings = payload.get("findings")
    if not isinstance(findings, dict) or set(findings) != set(CHEXPERT_FINDINGS):
        raise ValueError("XRV threshold inventory is incomplete")
    values: dict[str, dict[str, float]] = {}
    for finding in CHEXPERT_FINDINGS:
        row = findings[finding]
        if not isinstance(row, dict):
            raise TypeError("XRV finding threshold is invalid")
        low, high = float(row["negative_max"]), float(row["positive_min"])
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("XRV thresholds are outside [0,1]")
        values[finding] = {"negative_max": low, "positive_min": high}
    eligible = payload.get("calibration_status") == "calibrated_on_matched_real_validation"
    return values, {
        "status": payload.get("calibration_status"),
        "primary_metric_eligible": eligible,
        "source_sha256": sha256_file(source),
    }


def _state(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds["positive_min"]:
        return "positive"
    if probability <= thresholds["negative_max"]:
        return "negative"
    return "uncertain"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run XRV through Slurm")
    cache_dir = Path(args.cache_dir).resolve(strict=True)
    weight = (cache_dir / args.weight_filename).resolve(strict=True)
    if not weight.is_file():
        raise ValueError("frozen XRV checkpoint is missing")
    cxrs = load_cxr_candidates(args.cxr_run)
    threshold_values, calibration = _thresholds(args.thresholds)
    runtime = FrozenXRVRuntime(
        cache_dir=cache_dir,
        weight_filename=args.weight_filename,
        model_name=args.model_name,
    )
    if runtime.model.training or any(
        parameter.requires_grad for parameter in runtime.model.parameters()
    ):
        raise RuntimeError("XRV verifier is not frozen")

    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    for cxr_id, cxr in sorted(cxrs.items()):
        raw = runtime.predict(Path(cxr["artifact"]["path"]))
        probabilities: dict[str, float | None] = {}
        states: dict[str, str] = {}
        for finding in CHEXPERT_FINDINGS:
            labels = XRV_LABELS[finding]
            available = [float(raw[label]) for label in labels if label in raw]
            probability = max(available) if available else None
            probabilities[finding] = None if probability is None else round(probability, 8)
            states[finding] = (
                "unknown"
                if probability is None
                else _state(probability, threshold_values[finding])
            )
        records.append(
            {
                "cxr_candidate_id": cxr_id,
                "image_sha256": cxr["artifact"]["sha256"],
                "finding_probabilities": probabilities,
                "finding_states": states,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": {
            "model_id": args.model_name,
            "frozen": True,
            "checkpoint_sha256": sha256_file(weight),
            "checkpoint_size_bytes": weight.stat().st_size,
        },
        "primary_metric_eligible": calibration["primary_metric_eligible"],
        "calibration": calibration,
        "thresholds": threshold_values,
        "finding_order": list(CHEXPERT_FINDINGS),
        "records": records,
        "counts": {"cxr_candidates": len(records), "model_calls": len(records)},
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o007)
    started = time.monotonic()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = write_private_json(temporary / "cxr_finding_labels.json", payload)
        output_hash = sha256_file(output)
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": args.run_id,
                "cxr_count": payload["counts"]["cxr_candidates"],
                "primary_metric_eligible": payload["primary_metric_eligible"],
                "output_sha256": output_hash,
                "peak_vram_gib": payload["peak_vram_gib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

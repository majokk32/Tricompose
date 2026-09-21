"""Reference-free frozen-XRV selector for a protected CXR candidate sweep.

The selector uses only positive EHR-to-CXR support targets, candidate images,
and a deterministic hard image-quality gate. It never loads a matched real CXR
or a source radiology report.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)

from .candidate_sweep import (
    CANDIDATE_SCHEMA,
    SOURCE_CASE_SCHEMA,
    SWEEP_CASE_SCHEMA,
    SWEEP_RUN_SCHEMA,
)
from .common import (
    PROTECTED_ROENTGEN_ROOT,
    read_private_json,
    require_source_run,
    validate_case_id,
    validate_run_id,
)
from .evaluation import _load_grayscale, _quality_features


CONFIG_SCHEMA = "tricompose.roentgen_v2.candidate_selector_config.v1"
SELECTION_SCHEMA = "tricompose.roentgen_v2.candidate_selection.v1"
CASE_SELECTION_SCHEMA = "tricompose.roentgen_v2.candidate_case_selection.v1"
PROTECTED_SWEEP_ROOT = PROTECTED_ROENTGEN_ROOT / "candidate_sweeps"
PROTECTED_SELECTION_ROOT = PROTECTED_ROENTGEN_ROOT / "candidate_selections"


def _normalized_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported candidate selector config")
    return payload


def _require_sweep_run(run_id: str) -> Path:
    validate_run_id(run_id)
    resolved = require_inside(
        PROTECTED_SWEEP_ROOT / run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    if not resolved.is_dir():
        raise ValueError("candidate sweep run is not a directory")
    enforce_private_directory_mode(resolved)
    return resolved


def positive_ehr_support(
    observable_targets: Mapping[str, Any],
    confidences: Mapping[str, float],
    *,
    operating_point: float,
) -> dict[str, Any]:
    scores: list[float] = []
    for fact in sorted(observable_targets):
        raw_targets = observable_targets[fact]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("invalid positive EHR observable target")
        labels = [_normalized_label(str(label)) for label in raw_targets]
        if any(label not in confidences for label in labels):
            raise ValueError("positive EHR target is unavailable from XRV")
        scores.append(max(float(confidences[label]) for label in labels))
    if not scores:
        raise ValueError("candidate selection requires a positive observable EHR fact")
    values = np.asarray(scores, dtype=np.float64)
    return {
        "fact_count": len(scores),
        "mean_support": float(values.mean()),
        "support_rate": float(np.mean(values >= operating_point)),
    }


def hard_quality_failure(features: Mapping[str, float]) -> bool:
    return bool(
        float(features["intensity_std"]) < 0.03
        or float(features["clipped_fraction"]) > 0.60
        or float(features["entropy_bits"]) < 3.0
    )


def choose_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("candidate rows are empty")
    return min(
        rows,
        key=lambda row: (
            bool(row["hard_quality_failure"]),
            -float(row["ehr_support"]),
            str(row["candidate_id"]),
        ),
    )


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
    }


class FrozenXRVSelector:
    def __init__(self, config: Mapping[str, Any], device: Any) -> None:
        import torch
        import torchxrayvision as xrv

        model_config = config["model"]
        checkpoint = Path(model_config["checkpoint"]).resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError("configured XRV checkpoint is not a file")
        self.torch = torch
        self.xrv = xrv
        self.device = device
        self.model = xrv.models.DenseNet(
            weights=str(model_config["name"]),
            cache_dir=str(checkpoint.parent),
        ).to(device)
        self.model.eval().requires_grad_(False)
        self.pathologies = [_normalized_label(value) for value in self.model.pathologies]
        self.crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(224)
        self.checkpoint_sha256 = sha256_file(checkpoint)

    def predict(self, paths: Sequence[Path], batch_size: int) -> list[dict[str, float]]:
        tensors: list[Any] = []
        for path in paths:
            array = _load_grayscale(path) * 255.0
            normalized = self.xrv.datasets.normalize(array.astype(np.float32), 255)
            normalized = self.crop(normalized[None, :, :])
            normalized = self.resize(normalized)
            tensors.append(self.torch.from_numpy(normalized))

        results: list[dict[str, float]] = []
        with self.torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = self.torch.stack(tensors[start : start + batch_size]).to(self.device)
                output = self.model(batch).detach().float().cpu().numpy()
                for row in output:
                    if len(row) != len(self.pathologies) or not np.isfinite(row).all():
                        raise RuntimeError("invalid XRV selector output")
                    results.append(
                        {
                            label: float(value)
                            for label, value in zip(self.pathologies, row, strict=True)
                        }
                    )
        if len(results) != len(paths):
            raise RuntimeError("XRV selector output count mismatch")
        return results


def _collect_candidates(
    config: Mapping[str, Any], sweep_run_id: str
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    sweep_run = _require_sweep_run(sweep_run_id)
    manifest = read_private_json(sweep_run / "manifest.json")
    if manifest.get("schema_version") != SWEEP_RUN_SCHEMA:
        raise ValueError("unsupported candidate sweep manifest")
    if manifest.get("run_id") != sweep_run_id:
        raise ValueError("candidate sweep run ID mismatch")
    if manifest.get("loads_real_target_cxr") is not False:
        raise ValueError("candidate sweep violated the real-CXR boundary")
    if manifest.get("loads_real_target_report") is not False:
        raise ValueError("candidate sweep violated the report boundary")
    source_run_id = validate_run_id(str(manifest["source_run_id"]))
    source_run = require_source_run(source_run_id)
    case_records = manifest.get("cases")
    if not isinstance(case_records, list) or not case_records:
        raise ValueError("candidate sweep has no cases")
    expected_cases = int(config["selection"]["expected_case_count"])
    expected_per_case = int(config["selection"]["expected_candidates_per_case"])
    if len(case_records) != expected_cases:
        raise ValueError("candidate sweep case count mismatch")

    rows: list[dict[str, Any]] = []
    for case_record in case_records:
        case_id = validate_case_id(str(case_record["case_id"]))
        sweep_input = read_private_json(sweep_run / "cases" / case_id / "input.json")
        if sweep_input.get("schema_version") != SWEEP_CASE_SCHEMA:
            raise ValueError("unsupported candidate sweep case")
        source_case = read_private_json(source_run / "cases" / case_id / "input.json")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case")
        if sweep_input.get("source_prompt_sha256") != source_case.get("prompt_sha256"):
            raise ValueError("candidate sweep source hash mismatch")
        facts = source_case.get("ehr_facts")
        if not isinstance(facts, dict):
            raise TypeError("protected source case has no EHR facts")
        observable_targets = facts.get("cxr_observable_targets")
        if not isinstance(observable_targets, dict) or not observable_targets:
            raise ValueError("candidate case has no positive observable EHR fact")

        plans = sweep_input.get("candidate_plan")
        if not isinstance(plans, list) or len(plans) != expected_per_case:
            raise ValueError("candidate plan count mismatch")
        for plan in plans:
            candidate_id = str(plan["candidate_id"])
            candidate_dir = sweep_run / "cases" / case_id / "candidates" / candidate_id
            generation = read_private_json(candidate_dir / "generation.json")
            if generation.get("schema_version") != CANDIDATE_SCHEMA:
                raise ValueError("unsupported candidate generation artifact")
            if generation.get("prompt_sha256") != sweep_input.get("prompt_sha256"):
                raise ValueError("candidate prompt hash mismatch")
            if generation.get("candidate_id") != candidate_id:
                raise ValueError("candidate ID mismatch")
            image = require_private_file(candidate_dir / "generated_cxr.png")
            if generation.get("output", {}).get("image_sha256") != sha256_file(image):
                raise ValueError("candidate image hash mismatch")
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": candidate_id,
                    "guidance_scale": float(generation["guidance_scale"]),
                    "seed_variant": int(generation["seed_variant"]),
                    "image": image,
                    "image_sha256": str(generation["output"]["image_sha256"]),
                    "observable_targets": observable_targets,
                }
            )
    if len(rows) != expected_cases * expected_per_case:
        raise RuntimeError("candidate sweep total count mismatch")
    return sweep_run, manifest, rows


def run(
    *,
    config_path: str | Path,
    sweep_run_id: str,
    selection_run_id: str,
) -> dict[str, Any]:
    import torch

    os.umask(0o077)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run candidate selection through Slurm")
    validate_run_id(sweep_run_id)
    validate_run_id(selection_run_id)
    config = _load_config(config_path)
    _, manifest, rows = _collect_candidates(config, sweep_run_id)
    output_run = create_private_stage_dir(PROTECTED_SELECTION_ROOT / selection_run_id)
    log_path = output_run / "selection.log"
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    try:
        with log_path.open("x", encoding="utf-8") as protected_log:
            with contextlib.redirect_stdout(protected_log), contextlib.redirect_stderr(protected_log):
                runtime = FrozenXRVSelector(config, device)
                predictions = runtime.predict(
                    [row["image"] for row in rows],
                    int(config["selection"]["batch_size"]),
                )
                operating_point = float(config["selection"]["operating_point"])
                scored: list[dict[str, Any]] = []
                for row, prediction in zip(rows, predictions, strict=True):
                    support = positive_ehr_support(
                        row["observable_targets"],
                        prediction,
                        operating_point=operating_point,
                    )
                    features = _quality_features(_load_grayscale(row["image"]))
                    scored.append(
                        {
                            "case_id": row["case_id"],
                            "candidate_id": row["candidate_id"],
                            "guidance_scale": row["guidance_scale"],
                            "seed_variant": row["seed_variant"],
                            "image_sha256": row["image_sha256"],
                            "ehr_support": support["mean_support"],
                            "support_rate": support["support_rate"],
                            "fact_count": support["fact_count"],
                            "hard_quality_failure": hard_quality_failure(features),
                        }
                    )

                case_rows: list[dict[str, Any]] = []
                for case_record in manifest["cases"]:
                    case_id = validate_case_id(str(case_record["case_id"]))
                    candidates = [row for row in scored if row["case_id"] == case_id]
                    selected = choose_candidate(candidates)
                    baseline = next(
                        row
                        for row in candidates
                        if math.isclose(float(row["guidance_scale"]), 3.0)
                        and int(row["seed_variant"]) == 0
                    )
                    supports = [float(row["ehr_support"]) for row in candidates]
                    case_rows.append(
                        {
                            "schema_version": CASE_SELECTION_SCHEMA,
                            "case_id": case_id,
                            "selected_candidate_id": str(selected["candidate_id"]),
                            "selected_image_sha256": str(selected["image_sha256"]),
                            "selected_guidance_scale": float(selected["guidance_scale"]),
                            "selected_seed_variant": int(selected["seed_variant"]),
                            "selected_ehr_support": round(float(selected["ehr_support"]), 8),
                            "selected_support_rate": round(float(selected["support_rate"]), 8),
                            "selected_hard_quality_failure": bool(
                                selected["hard_quality_failure"]
                            ),
                            "baseline_candidate_id": str(baseline["candidate_id"]),
                            "baseline_ehr_support": round(float(baseline["ehr_support"]), 8),
                            "selected_minus_baseline_support": round(
                                float(selected["ehr_support"])
                                - float(baseline["ehr_support"]),
                                8,
                            ),
                            "candidate_support_std": round(float(np.std(supports)), 8),
                            "candidate_support_range": round(
                                float(max(supports) - min(supports)), 8
                            ),
                            "candidate_count": len(candidates),
                            "candidates": [
                                {
                                    "candidate_id": str(row["candidate_id"]),
                                    "image_sha256": str(row["image_sha256"]),
                                    "guidance_scale": float(row["guidance_scale"]),
                                    "seed_variant": int(row["seed_variant"]),
                                    "ehr_support": round(
                                        float(row["ehr_support"]), 8
                                    ),
                                    "support_rate": round(
                                        float(row["support_rate"]), 8
                                    ),
                                    "hard_quality_failure": bool(
                                        row["hard_quality_failure"]
                                    ),
                                }
                                for row in sorted(
                                    candidates,
                                    key=lambda value: str(value["candidate_id"]),
                                )
                            ],
                        }
                    )

                selection_path = write_private_json(
                    output_run / "selection.json",
                    {
                        "schema_version": SELECTION_SCHEMA,
                        "selection_run_id": selection_run_id,
                        "candidate_sweep_run_id": sweep_run_id,
                        "selector": (
                            "prefer non-hard-failure candidates, then maximize frozen-XRV "
                            "positive EHR support, then deterministic candidate ID tie-break"
                        ),
                        "loads_real_target_cxr": False,
                        "loads_real_target_report": False,
                        "cases": case_rows,
                    },
                )

                selected_support = [
                    float(row["selected_ehr_support"]) for row in case_rows
                ]
                baseline_support = [
                    float(row["baseline_ehr_support"]) for row in case_rows
                ]
                improvements = [
                    float(row["selected_minus_baseline_support"]) for row in case_rows
                ]
                cfg_counts = Counter(
                    str(row["selected_guidance_scale"]) for row in case_rows
                )
                seed_counts = Counter(
                    str(row["selected_seed_variant"]) for row in case_rows
                )
                total_hard = sum(bool(row["hard_quality_failure"]) for row in scored)
                selected_hard = sum(
                    bool(row["selected_hard_quality_failure"]) for row in case_rows
                )
                summary_path = write_private_json(
                    output_run / "summary.json",
                    {
                        "schema_version": SELECTION_SCHEMA,
                        "selection_run_id": selection_run_id,
                        "candidate_sweep_run_id": sweep_run_id,
                        "case_count": len(case_rows),
                        "candidate_count": len(scored),
                        "candidate_count_per_case": int(
                            config["selection"]["expected_candidates_per_case"]
                        ),
                        "scientific_scope": {
                            "reference_free_selection": True,
                            "loads_real_target_cxr": False,
                            "loads_real_target_report": False,
                            "ehr_absence_semantics": "unknown_not_negative",
                        },
                        "selected_ehr_support": _summary_stats(selected_support),
                        "fixed_cfg3_seed0_ehr_support": _summary_stats(baseline_support),
                        "selected_minus_fixed_support": _summary_stats(improvements),
                        "strictly_improved_case_count": int(
                            sum(value > 0 for value in improvements)
                        ),
                        "strictly_improved_case_rate": round(
                            float(np.mean(np.asarray(improvements) > 0)), 8
                        ),
                        "mean_candidate_support_std": round(
                            float(
                                np.mean(
                                    [row["candidate_support_std"] for row in case_rows]
                                )
                            ),
                            8,
                        ),
                        "mean_candidate_support_range": round(
                            float(
                                np.mean(
                                    [row["candidate_support_range"] for row in case_rows]
                                )
                            ),
                            8,
                        ),
                        "selected_guidance_scale_counts": dict(sorted(cfg_counts.items())),
                        "selected_seed_variant_counts": dict(sorted(seed_counts.items())),
                        "candidate_hard_failure_count": int(total_hard),
                        "candidate_hard_failure_rate": round(
                            float(total_hard / len(scored)), 8
                        ),
                        "selected_hard_failure_count": int(selected_hard),
                        "model": {
                            "name": str(config["model"]["name"]),
                            "checkpoint_sha256": runtime.checkpoint_sha256,
                            "torch": torch.__version__,
                            "torchxrayvision": importlib.metadata.version(
                                "torchxrayvision"
                            ),
                        },
                        "selection_sha256": sha256_file(selection_path),
                        "runtime_seconds": round(time.perf_counter() - started, 4),
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    },
                )
        enforce_private_file_mode(log_path)
        return {
            "stage": "roentgen_v2_candidate_selection",
            "status": "completed",
            "case_count": int(config["selection"]["expected_case_count"]),
            "candidate_count": int(config["selection"]["expected_case_count"])
            * int(config["selection"]["expected_candidates_per_case"]),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "peak_vram_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            ),
            "summary_sha256": sha256_file(summary_path),
        }
    except Exception as exc:
        failure = output_run / "failure.json"
        if not failure.exists():
            write_private_json(
                failure,
                {
                    "schema_version": SELECTION_SCHEMA,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep-run-id", required=True)
    parser.add_argument("--selection-run-id", required=True)
    args = parser.parse_args()
    try:
        result = run(
            config_path=args.config,
            sweep_run_id=args.sweep_run_id,
            selection_run_id=args.selection_run_id,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "roentgen_v2_candidate_selection",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

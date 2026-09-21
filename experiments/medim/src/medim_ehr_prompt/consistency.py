"""Score EHR support in matched-real and MeDiM-generated CXRs.

This module must run as a user-approved Slurm job. It never copies the matched
real image and never writes source paths or identifiers to an artifact or log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from medim_ehr_prompt.common import (
    create_case_stage,
    load_config,
    read_private_json,
    require_private_file,
    require_run_directory,
    sha256_file,
    validate_case_id,
    write_private_json,
)
from medim_ehr_prompt.prompting import EHR_TO_CXR_TARGETS


CONSISTENCY_SCHEMA = "medim_ehr_prompt.consistency.v1"
SUMMARY_SCHEMA = "medim_ehr_prompt.consistency_summary.v1"
EXPECTED_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
EXPECTED_GENERATION_SCHEMA = "medim_ehr_prompt.generation.v1"
EXPECTED_SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
EXPECTED_NATIVE_MANIFEST_SCHEMA = "medim_ehr_prompt.official_native_staging.v1"
EXPECTED_NATIVE_GENERATION_SCHEMA = "medim_ehr_prompt.official_native_generation.v1"


def _normalized_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


class FrozenXRVRuntime:
    """Frozen TorchXRayVision DenseNet with a pre-existing local checkpoint."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import torch
        import torchxrayvision as xrv

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run consistency through Slurm")

        consistency = config["consistency"]
        cache_dir = Path(consistency["cache_dir"]).resolve(strict=True)
        weight_path = (cache_dir / str(consistency["weight_filename"])).resolve(strict=True)
        if not weight_path.is_file():
            raise ValueError("configured TorchXRayVision checkpoint is not a file")

        self.torch = torch
        self.xrv = xrv
        self.device = torch.device("cuda:0")
        self.model = xrv.models.DenseNet(
            weights=str(consistency["model_name"]),
            cache_dir=str(cache_dir),
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(224)
        self.pathologies = [_normalized_label(name) for name in self.model.pathologies]
        self.checkpoint = {
            "model_name": str(consistency["model_name"]),
            "weight_size_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
        }

    def predict(self, image_path: Path) -> dict[str, float]:
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


def _load_real_cxr_index(config: Mapping[str, Any]) -> Any:
    """Load only the private path column once; it never leaves process memory."""
    import pandas as pd

    dataset_root = Path(config["dataset"]["root"]).resolve(strict=True)
    return pd.read_csv(
        dataset_root / "manifest.csv",
        usecols=["cxr_path"],
        dtype={"cxr_path": str},
    )["cxr_path"]


def _resolve_real_cxr(paths: Any, source_row_index: int) -> Path:
    """Resolve one matched real CXR internally without returning source metadata."""
    if source_row_index < 0:
        raise ValueError("invalid opaque source row index")
    if source_row_index >= len(paths):
        raise ValueError("opaque source row index is outside the manifest")
    value = paths.iloc[source_row_index]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("matched real CXR path is unavailable")
    real_cxr = Path(value).resolve(strict=True)
    if not real_cxr.is_file():
        raise ValueError("matched real CXR is not a file")
    return real_cxr


def _ehr_support(
    observable_targets: Mapping[str, Any],
    image_confidences: Mapping[str, float],
    *,
    operating_point: float,
) -> dict[str, Any]:
    if not 0.0 < operating_point < 1.0:
        raise ValueError("operating point must be between zero and one")

    per_fact: dict[str, dict[str, Any]] = {}
    fact_scores: list[float] = []
    for fact in sorted(observable_targets):
        raw_targets = observable_targets[fact]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("EHR observable target mapping is invalid")
        targets = [_normalized_label(str(target)) for target in raw_targets]
        missing = [target for target in targets if target not in image_confidences]
        if missing:
            raise ValueError("configured EHR target is unavailable from the CXR labeler")
        support = max(float(image_confidences[target]) for target in targets)
        fact_scores.append(support)
        per_fact[str(fact)] = {
            "candidate_cxr_labels": targets,
            "support_score": round(support, 8),
            "supported_at_operating_point": support >= operating_point,
        }

    if not fact_scores:
        return {
            "status": "not_applicable",
            "observable_fact_count": 0,
            "ehr_support_score": None,
            "support_rate": None,
            "per_fact": {},
        }
    return {
        "status": "scored",
        "observable_fact_count": len(fact_scores),
        "ehr_support_score": round(float(np.mean(fact_scores)), 8),
        "support_rate": round(
            float(np.mean(np.asarray(fact_scores) >= operating_point)),
            8,
        ),
        "per_fact": per_fact,
    }


def _bootstrap_mean_ci(
    values: list[float],
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> list[float] | None:
    if not values:
        return None
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    array = np.asarray(values, dtype=np.float64)
    sampled = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return [round(float(low), 8), round(float(high), 8)]


def _summary_stats(
    values: list[float],
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "bootstrap_mean_95ci": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
        "bootstrap_mean_95ci": _bootstrap_mean_ci(
            values,
            repetitions=repetitions,
            rng=rng,
        ),
    }


def _resolve_evaluation_layout(
    run_dir: Path,
    source_run_id: str | None,
) -> tuple[Path, list[dict[str, Any]], str]:
    """Resolve protected inputs without copying them into a generated run."""

    if source_run_id is None:
        source_run = run_dir
        expected_generation_schema = EXPECTED_GENERATION_SCHEMA
    else:
        source_run = require_run_directory(source_run_id)
        manifest = read_private_json(run_dir / "official_native_manifest.json")
        if manifest.get("schema_version") != EXPECTED_NATIVE_MANIFEST_SCHEMA:
            raise ValueError("unsupported official-native manifest")
        if manifest.get("source_run_id") != source_run_id:
            raise ValueError("official-native source run mismatch")
        expected_generation_schema = EXPECTED_NATIVE_GENERATION_SCHEMA

    selection = read_private_json(source_run / "selection.json")
    if selection.get("schema_version") != EXPECTED_SELECTION_SCHEMA:
        raise ValueError("unsupported selection artifact")
    cases = selection.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("selection does not contain cases")

    if source_run_id is not None:
        manifest_cases = manifest.get("cases")
        if not isinstance(manifest_cases, list):
            raise ValueError("official-native manifest has no cases")
        selected_ids = [validate_case_id(str(record["case_id"])) for record in cases]
        manifest_ids = [
            validate_case_id(str(record["case_id"])) for record in manifest_cases
        ]
        if selected_ids != manifest_ids:
            raise ValueError("official-native case order mismatch")

    return source_run, cases, expected_generation_schema


def _validate_generation_prompt(
    generation: Mapping[str, Any],
    case_input: Mapping[str, Any],
    expected_generation_schema: str,
) -> None:
    if generation.get("schema_version") != expected_generation_schema:
        raise ValueError("unsupported generation artifact")
    hash_field = (
        "source_prompt_sha256"
        if expected_generation_schema == EXPECTED_NATIVE_GENERATION_SCHEMA
        else "prompt_sha256"
    )
    if generation.get(hash_field) != case_input.get("prompt_sha256"):
        raise ValueError("generation prompt hash mismatch")


def score_run(
    config_path: str | Path,
    run_id: str,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    os.umask(0o077)
    config = load_config(config_path)
    run_dir = require_run_directory(run_id)
    source_run, cases, expected_generation_schema = _resolve_evaluation_layout(
        run_dir, source_run_id
    )

    runtime = FrozenXRVRuntime(config)
    operating_point = float(config["consistency"]["operating_point"])
    required_labels = {
        _normalized_label(target)
        for targets in EHR_TO_CXR_TARGETS.values()
        for target in targets
    }
    unavailable = required_labels - set(runtime.pathologies)
    if unavailable:
        raise ValueError("required EHR target is unavailable from the CXR labeler")
    real_cxr_paths = _load_real_cxr_index(config)
    scored_cases: list[dict[str, Any]] = []

    runtime.torch.cuda.reset_peak_memory_stats()
    for record in cases:
        case_id = validate_case_id(str(record["case_id"]))
        case_input = read_private_json(source_run / "cases" / case_id / "input.json")
        if case_input.get("schema_version") != EXPECTED_CASE_SCHEMA:
            raise ValueError("unsupported case input artifact")
        generation_dir = run_dir / "cases" / case_id / "generation"
        generation = read_private_json(generation_dir / "generation.json")
        _validate_generation_prompt(
            generation, case_input, expected_generation_schema
        )

        generated_cxr = require_private_file(generation_dir / "generated_cxr.png")
        real_cxr = _resolve_real_cxr(real_cxr_paths, int(case_input["source_row_index"]))
        real_confidences = runtime.predict(real_cxr)
        generated_confidences = runtime.predict(generated_cxr)

        facts = case_input.get("ehr_facts")
        if not isinstance(facts, dict):
            raise TypeError("protected EHR facts must be an object")
        observable_targets = facts.get("cxr_observable_targets")
        if not isinstance(observable_targets, dict):
            raise TypeError("EHR observable target mapping must be an object")
        real_support = _ehr_support(
            observable_targets,
            real_confidences,
            operating_point=operating_point,
        )
        generated_support = _ehr_support(
            observable_targets,
            generated_confidences,
            operating_point=operating_point,
        )
        if real_support["status"] == "scored":
            gap = round(
                float(generated_support["ehr_support_score"])
                - float(real_support["ehr_support_score"]),
                8,
            )
        else:
            gap = None

        stage_dir = create_case_stage(run_dir, case_id, "consistency")
        case_artifact = write_private_json(
            stage_dir / "consistency.json",
            {
                "schema_version": CONSISTENCY_SCHEMA,
                "run_id": run_id,
                "case_id": case_id,
                "scorer": runtime.checkpoint,
                "score_semantics": {
                    "primary": "mean maximum normalized CXR-label confidence over positive observable EHR facts",
                    "missing_or_zero_ehr": "unknown_not_negative",
                    "device_facts": "not_compared_by_primary_scorer",
                    "operating_point": operating_point,
                },
                "real_cxr": {
                    "ehr_support": real_support,
                    "relevant_label_confidences": {
                        label: real_confidences[label]
                        for label in sorted(required_labels)
                    },
                },
                "generated_cxr": {
                    "ehr_support": generated_support,
                    "relevant_label_confidences": {
                        label: generated_confidences[label]
                        for label in sorted(required_labels)
                    },
                    "image_sha256": sha256_file(generated_cxr),
                },
                "paired": {"generated_minus_real_support": gap},
            },
        )
        scored_cases.append(
            {
                "case_id": case_id,
                "status": real_support["status"],
                "observable_fact_count": int(real_support["observable_fact_count"]),
                "real_support_score": real_support["ehr_support_score"],
                "real_support_rate": real_support["support_rate"],
                "generated_support_score": generated_support["ehr_support_score"],
                "generated_support_rate": generated_support["support_rate"],
                "generated_minus_real_support": gap,
                "artifact_sha256": sha256_file(case_artifact),
            }
        )

    real_values = [float(row["real_support_score"]) for row in scored_cases if row["status"] == "scored"]
    generated_values = [
        float(row["generated_support_score"])
        for row in scored_cases
        if row["status"] == "scored"
    ]
    real_support_rates = [
        float(row["real_support_rate"])
        for row in scored_cases
        if row["status"] == "scored"
    ]
    generated_support_rates = [
        float(row["generated_support_rate"])
        for row in scored_cases
        if row["status"] == "scored"
    ]
    gaps = [
        float(row["generated_minus_real_support"])
        for row in scored_cases
        if row["status"] == "scored"
    ]
    repetitions = int(config["consistency"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(config["consistency"]["bootstrap_seed"]))
    summary_path = write_private_json(
        run_dir / "consistency_summary.json",
        {
            "schema_version": SUMMARY_SCHEMA,
            "run_id": run_id,
            "case_count": len(scored_cases),
            "scored_case_count": len(real_values),
            "not_applicable_case_count": len(scored_cases) - len(real_values),
            "score_semantics": {
                "primary": "positive-observable-EHR support, not a prospective prediction metric",
                "operating_point": operating_point,
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": repetitions,
            },
            "real_cxr_ehr_support": _summary_stats(
                real_values,
                repetitions=repetitions,
                rng=rng,
            ),
            "generated_cxr_ehr_support": _summary_stats(
                generated_values,
                repetitions=repetitions,
                rng=rng,
            ),
            "real_cxr_fact_support_rate": _summary_stats(
                real_support_rates,
                repetitions=repetitions,
                rng=rng,
            ),
            "generated_cxr_fact_support_rate": _summary_stats(
                generated_support_rates,
                repetitions=repetitions,
                rng=rng,
            ),
            "paired_generated_minus_real": _summary_stats(
                gaps,
                repetitions=repetitions,
                rng=rng,
            ),
            "cases": scored_cases,
        },
    )
    return {
        "stage": "ehr_cxr_consistency",
        "status": "ok",
        "run_id": run_id,
        "case_count": len(scored_cases),
        "scored_case_count": len(real_values),
        "not_applicable_case_count": len(scored_cases) - len(real_values),
        "summary_sha256": sha256_file(summary_path),
        "peak_vram_gib": round(
            runtime.torch.cuda.max_memory_allocated() / (1024**3),
            3,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    try:
        with Path(os.devnull).open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                result = score_run(args.config, args.run_id, args.source_run_id)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "ehr_cxr_consistency",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

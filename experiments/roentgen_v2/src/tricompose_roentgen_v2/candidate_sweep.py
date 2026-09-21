"""Generate a protected RoentGen-v2 candidate sweep for composition.

This GPU stage selects opaque source cases that contain at least one positive,
radiographically observable EHR fact and generates multiple frozen-model
candidates per case. It never loads a real target CXR or source report.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tricompose.privacy import create_private_stage_dir

from .common import (
    PROTECTED_ROENTGEN_ROOT,
    create_private_dir,
    enforce_private_file_mode,
    read_private_json,
    require_source_run,
    sha256_file,
    validate_case_id,
    validate_run_id,
    write_private_json,
)
from .model import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_PROMPT_TOKENS,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_REPO_COMMIT,
    FrozenRoentgenRuntime,
    validate_model_snapshot,
)
from .prompting import PROMPT_VERSION, build_roentgen_prompt


SOURCE_SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
SWEEP_RUN_SCHEMA = "tricompose.roentgen_v2.candidate_sweep.v1"
SWEEP_CASE_SCHEMA = "tricompose.roentgen_v2.candidate_sweep_case.v1"
CANDIDATE_SCHEMA = "tricompose.roentgen_v2.candidate.v1"
SUMMARY_SCHEMA = "tricompose.roentgen_v2.candidate_sweep_summary.v1"
PROTECTED_SWEEP_ROOT = PROTECTED_ROENTGEN_ROOT / "candidate_sweeps"
SEED_OFFSET = 27_182_818
MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    guidance_scale: float
    seed_variant: int
    seed: int


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunks(rows: Sequence[Mapping[str, Any]], size: int) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _guidance_token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def build_candidate_plan(
    base_seed: int,
    guidance_scales: Sequence[float],
    seed_count: int,
) -> tuple[CandidatePlan, ...]:
    if seed_count < 1 or seed_count > 8:
        raise ValueError("seed count must be between one and eight")
    normalized = tuple(float(value) for value in guidance_scales)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("guidance scales must be non-empty and unique")
    if any(value < 1.0 or value > 10.0 for value in normalized):
        raise ValueError("guidance scales must be between one and ten")

    plans: list[CandidatePlan] = []
    for guidance_scale in normalized:
        for seed_variant in range(seed_count):
            plans.append(
                CandidatePlan(
                    candidate_id=(
                        f"roentgen_v2_cfg{_guidance_token(guidance_scale)}_"
                        f"seed{seed_variant:02d}"
                    ),
                    guidance_scale=guidance_scale,
                    seed_variant=seed_variant,
                    seed=(int(base_seed) + seed_variant) % MAX_SEED,
                )
            )
    if len({plan.candidate_id for plan in plans}) != len(plans):
        raise RuntimeError("candidate identifiers are not unique")
    return tuple(plans)


def _stage_cases(
    *,
    source_run_id: str,
    output_run_id: str,
    limit: int,
    guidance_scales: Sequence[float],
    seed_count: int,
) -> tuple[Path, list[dict[str, Any]]]:
    if limit < 1:
        raise ValueError("case limit must be positive")
    source_run = require_source_run(source_run_id)
    selection = read_private_json(source_run / "selection.json")
    if selection.get("schema_version") != SOURCE_SELECTION_SCHEMA:
        raise ValueError("unsupported source selection schema")
    source_records = selection.get("cases")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("source selection has no cases")

    output_run = create_private_stage_dir(PROTECTED_SWEEP_ROOT / output_run_id)
    cases_root = create_private_dir(output_run / "cases")
    create_private_dir(output_run / "logs")
    staged: list[dict[str, Any]] = []

    for source_record in source_records:
        if len(staged) >= limit:
            break
        case_id = validate_case_id(str(source_record["case_id"]))
        source_case = read_private_json(source_run / "cases" / case_id / "input.json")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case schema")
        if source_case.get("prompt_sha256") != source_record.get("prompt_sha256"):
            raise ValueError("protected source case hash mismatch")
        facts = source_case.get("ehr_facts")
        if not isinstance(facts, dict):
            raise TypeError("protected source case has no EHR facts")
        observable_targets = facts.get("cxr_observable_targets")
        if not isinstance(observable_targets, dict):
            raise TypeError("protected source case has invalid observable targets")
        if not observable_targets:
            continue

        prompt = build_roentgen_prompt(facts)
        prompt_hash = _hash_text(prompt.text)
        base_seed = (int(source_case["seed"]) + SEED_OFFSET) % MAX_SEED
        plans = build_candidate_plan(base_seed, guidance_scales, seed_count)
        case_dir = create_private_dir(cases_root / case_id)
        create_private_dir(case_dir / "candidates")
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": SWEEP_CASE_SCHEMA,
                "case_id": case_id,
                "source_prompt_sha256": str(source_case["prompt_sha256"]),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "roentgen_prompt": prompt.text,
                "included_diagnoses": list(prompt.included_diagnoses),
                "included_devices": list(prompt.included_devices),
                "omitted_devices": list(prompt.omitted_devices),
                "measurements_used": False,
                "race_used": False,
                "base_seed": base_seed,
                "candidate_plan": [
                    {
                        "candidate_id": plan.candidate_id,
                        "guidance_scale": plan.guidance_scale,
                        "seed_variant": plan.seed_variant,
                        "seed": plan.seed,
                    }
                    for plan in plans
                ],
            },
        )
        staged.append(
            {
                "case_id": case_id,
                "prompt_sha256": prompt_hash,
                "base_seed": base_seed,
            }
        )

    if len(staged) != limit:
        raise ValueError("not enough source cases have observable positive EHR facts")

    write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": SWEEP_RUN_SCHEMA,
            "run_id": output_run_id,
            "source_run_id": source_run_id,
            "producer": "frozen_roentgen_v2_candidate_sweep",
            "scientific_signature": (
                "structured EHR -> deterministic radiology-style prompt -> "
                "multi-seed/multi-CFG text-conditioned CXR candidates"
            ),
            "direct_ehr_to_cxr": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "selection_eligibility": "nonempty_positive_cxr_observable_targets",
            "official_repo_commit": OFFICIAL_REPO_COMMIT,
            "official_model_revision": OFFICIAL_MODEL_REVISION,
            "prompt_version": PROMPT_VERSION,
            "case_count": len(staged),
            "guidance_scales": list(map(float, guidance_scales)),
            "seed_count": int(seed_count),
            "candidate_count_per_case": len(guidance_scales) * seed_count,
            "cases": staged,
        },
    )
    return output_run, staged


def _save_image(image: Any, path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError("generated candidate image already exists")
    if not hasattr(image, "save") or not hasattr(image, "size"):
        raise TypeError("RoentGen-v2 output is not a PIL-compatible image")
    image.convert("RGB").save(path, format="PNG")
    enforce_private_file_mode(path)
    return [int(image.size[0]), int(image.size[1])]


def _write_failure(output_run: Path | None, phase: str, exc: Exception) -> None:
    if output_run is None:
        return
    failure = output_run / "failure.json"
    if not failure.exists():
        write_private_json(
            failure,
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "failed",
                "phase": phase,
                "error_type": type(exc).__name__,
            },
        )


def run(
    *,
    model_dir: str | Path,
    source_run_id: str,
    output_run_id: str,
    limit: int,
    batch_size: int,
    guidance_scales: Sequence[float],
    seed_count: int,
) -> dict[str, Any]:
    os.umask(0o077)
    validate_run_id(source_run_id)
    validate_run_id(output_run_id)
    if batch_size < 1 or batch_size > 4:
        raise ValueError("batch size must be between one and four")
    model_audit = validate_model_snapshot(model_dir)
    output_run: Path | None = None
    phase = "stage"
    started = time.monotonic()
    try:
        output_run, records = _stage_cases(
            source_run_id=source_run_id,
            output_run_id=output_run_id,
            limit=limit,
            guidance_scales=guidance_scales,
            seed_count=seed_count,
        )
        phase = "model_load_and_inference"
        protected_log = output_run / "logs" / "candidate_sweep.log"
        completed: list[dict[str, Any]] = []
        peak_vram = 0.0

        with protected_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = FrozenRoentgenRuntime(model_dir)
                for guidance_scale in map(float, guidance_scales):
                    for seed_variant in range(seed_count):
                        for batch_ordinal, batch in enumerate(_chunks(records, batch_size)):
                            inputs = [
                                read_private_json(
                                    output_run / "cases" / str(record["case_id"]) / "input.json"
                                )
                                for record in batch
                            ]
                            plans = [
                                next(
                                    plan
                                    for plan in item["candidate_plan"]
                                    if float(plan["guidance_scale"]) == guidance_scale
                                    and int(plan["seed_variant"]) == seed_variant
                                )
                                for item in inputs
                            ]
                            result = runtime.generate_batch(
                                [str(item["roentgen_prompt"]) for item in inputs],
                                [int(plan["seed"]) for plan in plans],
                                guidance_scale=guidance_scale,
                                num_inference_steps=DEFAULT_INFERENCE_STEPS,
                                max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
                            )
                            peak_vram = max(peak_vram, result.peak_vram_gib)
                            per_case_elapsed = result.elapsed_seconds / len(batch)
                            for record, case_input, plan, image, token_count in zip(
                                batch,
                                inputs,
                                plans,
                                result.images,
                                result.prompt_token_counts,
                                strict=True,
                            ):
                                case_id = validate_case_id(str(record["case_id"]))
                                candidate_id = str(plan["candidate_id"])
                                candidate_dir = create_private_dir(
                                    output_run / "cases" / case_id / "candidates" / candidate_id
                                )
                                image_path = candidate_dir / "generated_cxr.png"
                                dimensions = _save_image(image, image_path)
                                if dimensions != [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE]:
                                    raise ValueError("candidate CXR dimensions are not 512 by 512")
                                artifact = write_private_json(
                                    candidate_dir / "generation.json",
                                    {
                                        "schema_version": CANDIDATE_SCHEMA,
                                        "case_id": case_id,
                                        "candidate_id": candidate_id,
                                        "modality": "cxr",
                                        "producer": "frozen_roentgen_v2_diffusers",
                                        "frozen_model": True,
                                        "input_signature": "radiology_style_text_to_cxr",
                                        "official_model_revision": OFFICIAL_MODEL_REVISION,
                                        "prompt_version": PROMPT_VERSION,
                                        "prompt_sha256": str(case_input["prompt_sha256"]),
                                        "seed": int(plan["seed"]),
                                        "seed_variant": int(plan["seed_variant"]),
                                        "guidance_scale": guidance_scale,
                                        "num_inference_steps": DEFAULT_INFERENCE_STEPS,
                                        "dtype": "bfloat16",
                                        "prompt_token_count": int(token_count),
                                        "batch_ordinal": batch_ordinal,
                                        "output": {
                                            "image_dimensions": dimensions,
                                            "image_sha256": sha256_file(image_path),
                                        },
                                        "cost": {
                                            "model_calls": 1,
                                            "estimated_case_seconds": round(per_case_elapsed, 3),
                                            "batch_seconds": round(result.elapsed_seconds, 3),
                                            "batch_peak_vram_gib": round(result.peak_vram_gib, 3),
                                        },
                                    },
                                )
                                completed.append(
                                    {
                                        "case_id": case_id,
                                        "candidate_id": candidate_id,
                                        "generation_sha256": sha256_file(artifact),
                                    }
                                )
        enforce_private_file_mode(protected_log)

        expected_candidates = limit * len(guidance_scales) * seed_count
        if len(completed) != expected_candidates:
            raise RuntimeError("candidate sweep completion count mismatch")
        elapsed = time.monotonic() - started
        summary_path = write_private_json(
            output_run / "summary.json",
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "completed",
                "run_id": output_run_id,
                "case_count": limit,
                "candidate_count": len(completed),
                "candidate_count_per_case": len(guidance_scales) * seed_count,
                "guidance_scales": list(map(float, guidance_scales)),
                "seed_count": seed_count,
                "image_dimensions": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
                "elapsed_seconds": round(elapsed, 3),
                "peak_vram_gib": round(peak_vram, 3),
                "model_audit": model_audit,
                "candidates": completed,
            },
        )
        return {
            "stage": "roentgen_v2_candidate_sweep",
            "status": "completed",
            "case_count": limit,
            "candidate_count": len(completed),
            "image_dimensions": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
            "elapsed_seconds": round(elapsed, 3),
            "peak_vram_gib": round(peak_vram, 3),
            "summary_sha256": sha256_file(summary_path),
        }
    except Exception as exc:
        _write_failure(output_run, phase, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--guidance-scales", nargs="+", type=float, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    args = parser.parse_args()
    try:
        result = run(
            model_dir=args.model_dir,
            source_run_id=args.source_run_id,
            output_run_id=args.output_run_id,
            limit=args.limit,
            batch_size=args.batch_size,
            guidance_scales=args.guidance_scales,
            seed_count=args.seed_count,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "roentgen_v2_candidate_sweep",
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

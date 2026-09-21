"""Generate protected EHR-prompt CXR candidates with frozen Liquid 7B."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from tricompose.ehr_prompt_cxr.common import (
    enforce_private_file_mode,
    read_private_json,
    sha256_file,
    validate_case_id,
    validate_run_id,
    write_private_json,
)
from tricompose.ehr_prompt_cxr.staging import stage_cases

from .runtime import (
    DEFAULT_CFG,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    IMAGE_TOKEN_STEPS,
    FrozenLiquidRuntime,
    validate_liquid_deployment,
)
GENERATION_SCHEMA = "tricompose.ehr_prompt_cxr.generation.v1"
SUMMARY_SCHEMA = "tricompose.ehr_prompt_cxr.summary.v1"


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _save_image(image: Any, path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError("generated image already exists")
    image = image.convert("RGB")
    image.save(path, format="PNG")
    enforce_private_file_mode(path)
    return [int(image.size[0]), int(image.size[1])]


def _write_failure(output_run: Path | None, phase: str, exc: Exception) -> None:
    if output_run is None:
        return
    path = output_run / "failure.json"
    if not path.exists():
        write_private_json(
            path,
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
    evaluation_root: str | Path,
    vq_config: str | Path,
    vq_checkpoint: str | Path,
    source_run_id: str,
    output_run_id: str,
    limit: int,
    batch_size: int,
) -> dict[str, Any]:
    os.umask(0o077)
    validate_run_id(source_run_id)
    validate_run_id(output_run_id)
    if batch_size < 1 or batch_size > 4:
        raise ValueError("batch size must be between one and four")
    model_audit = validate_liquid_deployment(model_dir, vq_config, vq_checkpoint)
    output_run: Path | None = None
    phase = "stage"
    total_started = time.monotonic()
    try:
        output_run, records = stage_cases(
            model_name="liquid",
            source_run_id=source_run_id,
            output_run_id=output_run_id,
            limit=limit,
        )
        phase = "model_load_and_inference"
        log_path = output_run / "logs" / "liquid.log"
        completed: list[dict[str, Any]] = []
        peak_vram = 0.0
        with log_path.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = FrozenLiquidRuntime(
                    model_dir=model_dir,
                    evaluation_root=evaluation_root,
                    vq_config=vq_config,
                    vq_checkpoint=vq_checkpoint,
                )
                for batch_ordinal, batch in enumerate(_chunks(records, batch_size)):
                    case_inputs = [
                        read_private_json(
                            output_run / "cases" / record["case_id"] / "input.json"
                        )
                        for record in batch
                    ]
                    result = runtime.generate_batch(
                        [str(item["model_prompt"]) for item in case_inputs],
                        [int(item["seed"]) for item in case_inputs],
                    )
                    peak_vram = max(peak_vram, result.peak_vram_gib)
                    per_case_seconds = result.elapsed_seconds / len(batch)
                    for record, case_input, image, token_count in zip(
                        batch,
                        case_inputs,
                        result.images,
                        result.input_token_counts,
                        strict=True,
                    ):
                        case_id = validate_case_id(str(record["case_id"]))
                        generation_dir = output_run / "cases" / case_id / "generation"
                        generation_dir.mkdir(mode=0o700, exist_ok=False)
                        image_path = generation_dir / "generated_cxr.png"
                        dimensions = _save_image(image, image_path)
                        if dimensions != [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE]:
                            raise ValueError("unexpected Liquid image dimensions")
                        generation_path = write_private_json(
                            generation_dir / "generation.json",
                            {
                                "schema_version": GENERATION_SCHEMA,
                                "case_id": case_id,
                                "candidate_id": "liquid_7b_seed0",
                                "modality": "cxr",
                                "producer": "frozen_liquid_v1_7b",
                                "frozen_model": True,
                                "input_signature": "radiology_style_text_to_cxr",
                                "direct_ehr_to_cxr": False,
                                "model_prompt_version": case_input[
                                    "model_prompt_version"
                                ],
                                "model_prompt_sha256": case_input[
                                    "model_prompt_sha256"
                                ],
                                "seed": int(case_input["seed"]),
                                "image_token_steps": IMAGE_TOKEN_STEPS,
                                "guidance_scale": DEFAULT_CFG,
                                "temperature": DEFAULT_TEMPERATURE,
                                "top_k": DEFAULT_TOP_K,
                                "top_p": DEFAULT_TOP_P,
                                "input_token_count": int(token_count),
                                "batch_ordinal": batch_ordinal,
                                "output": {
                                    "image_dimensions": dimensions,
                                    "image_sha256": sha256_file(image_path),
                                },
                                "cost": {
                                    "model_calls": 1,
                                    "estimated_case_seconds": round(
                                        per_case_seconds, 3
                                    ),
                                    "batch_seconds": round(
                                        result.elapsed_seconds, 3
                                    ),
                                    "batch_peak_vram_gib": round(
                                        result.peak_vram_gib, 3
                                    ),
                                },
                            },
                        )
                        completed.append(
                            {
                                "case_id": case_id,
                                "generation_sha256": sha256_file(generation_path),
                            }
                        )
        enforce_private_file_mode(log_path)
        total_elapsed = time.monotonic() - total_started
        write_private_json(
            output_run / "summary.json",
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "completed",
                "run_id": output_run_id,
                "model_name": "liquid",
                "case_count": len(completed),
                "image_dimensions": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
                "elapsed_seconds": round(total_elapsed, 3),
                "peak_vram_gib": round(peak_vram, 3),
                "batch_size": batch_size,
                "model_audit": model_audit,
                "cases": completed,
            },
        )
        return {
            "status": "completed",
            "model_name": "liquid",
            "case_count": len(completed),
            "image_dimensions": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
            "elapsed_seconds": round(total_elapsed, 3),
            "peak_vram_gib": round(peak_vram, 3),
        }
    except Exception as exc:
        _write_failure(output_run, phase, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--vq-config", required=True)
    parser.add_argument("--vq-checkpoint", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    try:
        result = run(
            model_dir=args.model_dir,
            evaluation_root=args.evaluation_root,
            vq_config=args.vq_config,
            vq_checkpoint=args.vq_checkpoint,
            source_run_id=args.source_run_id,
            output_run_id=args.output_run_id,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate protected EHR-prompt CXR candidates with frozen UniDisc."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tricompose.ehr_prompt_cxr.common import (
    create_private_dir,
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
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLING_STEPS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FrozenUniDiscRuntime,
    validate_unidisc_deployment,
)


GENERATION_SCHEMA = "tricompose.ehr_prompt_cxr.generation.v1"
SUMMARY_SCHEMA = "tricompose.ehr_prompt_cxr.summary.v1"


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
    *, model_root: str | Path, source_run_id: str, output_run_id: str, limit: int
) -> dict[str, Any]:
    os.umask(0o077)
    validate_run_id(source_run_id)
    validate_run_id(output_run_id)
    model_audit = validate_unidisc_deployment(model_root)
    output_run: Path | None = None
    phase = "stage"
    total_started = time.monotonic()
    try:
        output_run, records = stage_cases(
            model_name="unidisc",
            source_run_id=source_run_id,
            output_run_id=output_run_id,
            limit=limit,
        )
        phase = "model_load_and_inference"
        log_path = output_run / "logs" / "unidisc.log"
        bootstrap_dir = create_private_dir(output_run / "logs" / "bootstrap_runtime")
        completed: list[dict[str, Any]] = []
        peak_vram = 0.0
        with log_path.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = FrozenUniDiscRuntime(model_root, bootstrap_dir)
                for record in records:
                    case_id = validate_case_id(str(record["case_id"]))
                    case_input = read_private_json(
                        output_run / "cases" / case_id / "input.json"
                    )
                    generation_dir = create_private_dir(
                        output_run / "cases" / case_id / "generation"
                    )
                    runtime_dir = create_private_dir(
                        output_run / "cases" / case_id / "runtime_debug"
                    )
                    result = runtime.generate(
                        str(case_input["model_prompt"]),
                        int(case_input["seed"]),
                        runtime_dir,
                    )
                    peak_vram = max(peak_vram, result.peak_vram_gib)
                    image_path = generation_dir / "generated_cxr.png"
                    dimensions = _save_image(result.image, image_path)
                    if dimensions != [DEFAULT_RESOLUTION, DEFAULT_RESOLUTION]:
                        raise ValueError("unexpected UniDisc image dimensions")
                    generation_path = write_private_json(
                        generation_dir / "generation.json",
                        {
                            "schema_version": GENERATION_SCHEMA,
                            "case_id": case_id,
                            "candidate_id": "unidisc_seed0",
                            "modality": "cxr",
                            "producer": "frozen_unidisc_interleaved",
                            "frozen_model": True,
                            "input_signature": "radiology_style_text_to_cxr",
                            "direct_ehr_to_cxr": False,
                            "model_prompt_version": case_input["model_prompt_version"],
                            "model_prompt_sha256": case_input["model_prompt_sha256"],
                            "seed": int(case_input["seed"]),
                            "sampling_steps": DEFAULT_SAMPLING_STEPS,
                            "guidance_scale": DEFAULT_CFG,
                            "temperature": DEFAULT_TEMPERATURE,
                            "top_p": DEFAULT_TOP_P,
                            "output": {
                                "image_dimensions": dimensions,
                                "image_sha256": sha256_file(image_path),
                            },
                            "cost": {
                                "model_calls": 1,
                                "case_seconds": round(result.elapsed_seconds, 3),
                                "case_peak_vram_gib": round(result.peak_vram_gib, 3),
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
                "model_name": "unidisc",
                "case_count": len(completed),
                "image_dimensions": [DEFAULT_RESOLUTION, DEFAULT_RESOLUTION],
                "elapsed_seconds": round(total_elapsed, 3),
                "peak_vram_gib": round(peak_vram, 3),
                "model_load_attempts": runtime.load_attempts,
                "model_audit": model_audit,
                "cases": completed,
            },
        )
        return {
            "status": "completed",
            "model_name": "unidisc",
            "case_count": len(completed),
            "image_dimensions": [DEFAULT_RESOLUTION, DEFAULT_RESOLUTION],
            "elapsed_seconds": round(total_elapsed, 3),
            "peak_vram_gib": round(peak_vram, 3),
            "model_load_attempts": runtime.load_attempts,
        }
    except Exception as exc:
        _write_failure(output_run, phase, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    try:
        result = run(
            model_root=args.model_root,
            source_run_id=args.source_run_id,
            output_run_id=args.output_run_id,
            limit=args.limit,
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

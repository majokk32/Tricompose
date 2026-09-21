"""Generate protected CXR candidates with frozen RoentGen-v2.

The source is an existing protected real-anchor EHR selection.  No real CXR or
real report is loaded.  All model calls are local-only and must run via Slurm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .common import (
    PROTECTED_EXPERIMENT_ROOT,
    create_output_run,
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
    DEFAULT_GUIDANCE_SCALE,
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
RUN_SCHEMA = "tricompose.roentgen_v2.run.v1"
CASE_SCHEMA = "tricompose.roentgen_v2.case_input.v1"
GENERATION_SCHEMA = "tricompose.roentgen_v2.generation.v1"
SUMMARY_SCHEMA = "tricompose.roentgen_v2.summary.v1"
SEED_OFFSET = 27_182_818


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _stage_cases(
    *, source_run_id: str, output_run_id: str, limit: int
) -> tuple[Path, list[dict[str, Any]]]:
    source_run = require_source_run(source_run_id)
    selection = read_private_json(source_run / "selection.json")
    if selection.get("schema_version") != SOURCE_SELECTION_SCHEMA:
        raise ValueError("unsupported source selection schema")
    source_records = selection.get("cases")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("source selection has no cases")
    if limit < 1 or limit > len(source_records):
        raise ValueError("requested case limit is outside the source selection")

    output_run = create_output_run(output_run_id)
    cases_root = create_private_dir(output_run / "cases")
    create_private_dir(output_run / "logs")

    staged: list[dict[str, Any]] = []
    for source_record in source_records[:limit]:
        case_id = validate_case_id(str(source_record["case_id"]))
        source_case = read_private_json(source_run / "cases" / case_id / "input.json")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case schema")
        if source_case.get("prompt_sha256") != source_record.get("prompt_sha256"):
            raise ValueError("protected source case hash mismatch")
        facts = source_case.get("ehr_facts")
        if not isinstance(facts, dict):
            raise TypeError("protected source case has no EHR facts")

        prompt = build_roentgen_prompt(facts)
        prompt_hash = _hash_text(prompt.text)
        seed = (int(source_case["seed"]) + SEED_OFFSET) % (2**63 - 1)
        case_dir = create_private_dir(cases_root / case_id)
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": CASE_SCHEMA,
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
                "seed": seed,
            },
        )
        staged.append(
            {
                "case_id": case_id,
                "prompt_sha256": prompt_hash,
                "seed": seed,
            }
        )

    write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": RUN_SCHEMA,
            "run_id": output_run_id,
            "source_run_id": source_run_id,
            "producer": "frozen_roentgen_v2_diffusers",
            "scientific_signature": (
                "structured EHR -> deterministic radiology-style prompt -> "
                "text-conditioned CXR"
            ),
            "direct_ehr_to_cxr": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "official_repo_commit": OFFICIAL_REPO_COMMIT,
            "official_model_revision": OFFICIAL_MODEL_REVISION,
            "prompt_version": PROMPT_VERSION,
            "case_count": len(staged),
            "cases": staged,
        },
    )
    return output_run, staged


def _save_image(image: Any, path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError("generated image already exists")
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
) -> dict[str, Any]:
    os.umask(0o077)
    validate_run_id(source_run_id)
    validate_run_id(output_run_id)
    if batch_size < 1 or batch_size > 4:
        raise ValueError("batch size must be between 1 and 4")

    # Validate the complete local snapshot before creating a protected run.
    model_audit = validate_model_snapshot(model_dir)
    output_run: Path | None = None
    phase = "stage"
    total_started = time.monotonic()
    try:
        output_run, records = _stage_cases(
            source_run_id=source_run_id,
            output_run_id=output_run_id,
            limit=limit,
        )
        phase = "model_load_and_inference"
        protected_log = output_run / "logs" / "roentgen_v2.log"
        completed: list[dict[str, Any]] = []
        peak_vram = 0.0

        with protected_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = FrozenRoentgenRuntime(model_dir)
                for batch_ordinal, batch in enumerate(_chunks(records, batch_size)):
                    inputs = [
                        read_private_json(
                            output_run / "cases" / record["case_id"] / "input.json"
                        )
                        for record in batch
                    ]
                    prompts = [str(item["roentgen_prompt"]) for item in inputs]
                    seeds = [int(item["seed"]) for item in inputs]
                    result = runtime.generate_batch(
                        prompts,
                        seeds,
                        guidance_scale=DEFAULT_GUIDANCE_SCALE,
                        num_inference_steps=DEFAULT_INFERENCE_STEPS,
                        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
                    )
                    peak_vram = max(peak_vram, result.peak_vram_gib)
                    per_case_elapsed = result.elapsed_seconds / len(batch)
                    for record, case_input, image, token_count in zip(
                        batch,
                        inputs,
                        result.images,
                        result.prompt_token_counts,
                        strict=True,
                    ):
                        case_id = validate_case_id(str(record["case_id"]))
                        generation_dir = create_private_dir(
                            output_run / "cases" / case_id / "generation"
                        )
                        image_path = generation_dir / "generated_cxr.png"
                        dimensions = _save_image(image, image_path)
                        if dimensions != [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE]:
                            raise ValueError("generated CXR dimensions are not 512 by 512")
                        generation_path = write_private_json(
                            generation_dir / "generation.json",
                            {
                                "schema_version": GENERATION_SCHEMA,
                                "case_id": case_id,
                                "candidate_id": "roentgen_v2_seed0",
                                "modality": "cxr",
                                "producer": "frozen_roentgen_v2_diffusers",
                                "frozen_model": True,
                                "input_signature": "radiology_style_text_to_cxr",
                                "official_model_revision": OFFICIAL_MODEL_REVISION,
                                "prompt_version": PROMPT_VERSION,
                                "prompt_sha256": str(case_input["prompt_sha256"]),
                                "seed": int(case_input["seed"]),
                                "guidance_scale": DEFAULT_GUIDANCE_SCALE,
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
                                "generation_sha256": sha256_file(generation_path),
                            }
                        )
        enforce_private_file_mode(protected_log)

        total_elapsed = time.monotonic() - total_started
        write_private_json(
            output_run / "summary.json",
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "completed",
                "run_id": output_run_id,
                "case_count": len(completed),
                "image_dimensions": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
                "elapsed_seconds": round(total_elapsed, 3),
                "peak_vram_gib": round(peak_vram, 3),
                "model_audit": model_audit,
                "cases": completed,
            },
        )
        return {
            "status": "completed",
            "case_count": len(completed),
            "image_dimensions": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
            "elapsed_seconds": round(total_elapsed, 3),
            "peak_vram_gib": round(peak_vram, 3),
            "model_revision": OFFICIAL_MODEL_REVISION,
        }
    except Exception as exc:
        _write_failure(output_run, phase, exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(
            model_dir=args.model_dir,
            source_run_id=args.source_run_id,
            output_run_id=args.output_run_id,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

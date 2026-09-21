"""Generate protected CXR candidates with the frozen official RadEdit backbone.

The job consumes an existing protected EHR-fact selection, converts only
radiographically observable facts to short text, and never loads a matched real
CXR or source report. GPU execution is permitted only through Slurm.
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
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_WIDTH,
    RADEDIT_REVISION,
    FrozenRadEditRuntime,
    validate_model_bundle,
)
from .prompting import PROMPT_VERSION, build_radedit_prompt


SOURCE_SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
RUN_SCHEMA = "tricompose.radedit.run.v1"
GENERATION_SCHEMA = "tricompose.radedit.generation.v1"
SUMMARY_SCHEMA = "tricompose.radedit.summary.v1"
SEED_OFFSET = 74_991_263


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

    from .common import PROTECTED_EXPERIMENT_ROOT

    output_run = create_private_dir(PROTECTED_EXPERIMENT_ROOT / output_run_id)
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

        prompt = build_radedit_prompt(facts)
        prompt_hash = _hash_text(prompt.text)
        seed = (int(source_case["seed"]) + SEED_OFFSET) % (2**63 - 1)
        case_dir = create_private_dir(cases_root / case_id)
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": "tricompose.radedit.case_input.v1",
                "case_id": case_id,
                "source_prompt_sha256": str(source_case["prompt_sha256"]),
                "model_prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "radedit_prompt": prompt.text,
                "included_diagnoses": list(prompt.included_diagnoses),
                "included_devices": list(prompt.included_devices),
                "omitted_devices": list(prompt.omitted_devices),
                "underconditioned": prompt.underconditioned,
                "measurements_used": False,
                "demographics_used": False,
                "seed": seed,
            },
        )
        staged.append(
            {
                "case_id": case_id,
                "prompt_sha256": prompt_hash,
                "seed": seed,
                "underconditioned": prompt.underconditioned,
            }
        )

    write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": RUN_SCHEMA,
            "run_id": output_run_id,
            "source_run_id": source_run_id,
            "producer": "frozen_radedit_t2i",
            "scientific_signature": (
                "structured EHR -> deterministic radiology-style text -> "
                "text-conditioned CXR"
            ),
            "direct_ehr_to_cxr": False,
            "editing_mode_used": False,
            "loads_previous_cxr": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "radedit_revision": RADEDIT_REVISION,
            "model_prompt_version": PROMPT_VERSION,
            "case_count": len(staged),
            "cases": staged,
        },
    )
    return output_run, staged


def _save_image(image: Any, path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError("generated image already exists")
    image.convert("RGB").save(path, format="PNG")
    enforce_private_file_mode(path)
    return [int(image.size[0]), int(image.size[1])]


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
    if batch_size < 1 or batch_size > 2:
        raise ValueError("batch size must be one or two")
    model_audit = validate_model_bundle(model_dir)
    output_run, records = _stage_cases(
        source_run_id=source_run_id,
        output_run_id=output_run_id,
        limit=limit,
    )
    total_started = time.monotonic()
    completed: list[dict[str, Any]] = []
    peak_vram = 0.0
    protected_log = output_run / "logs" / "radedit.log"

    try:
        with protected_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = FrozenRadEditRuntime(model_dir)
                for batch_ordinal, batch in enumerate(_chunks(records, batch_size)):
                    inputs = [
                        read_private_json(output_run / "cases" / row["case_id"] / "input.json")
                        for row in batch
                    ]
                    result = runtime.generate_batch(
                        [str(item["radedit_prompt"]) for item in inputs],
                        [int(item["seed"]) for item in inputs],
                    )
                    peak_vram = max(peak_vram, result.peak_vram_gib)
                    per_case_seconds = result.elapsed_seconds / len(batch)
                    for row, case_input, image, token_count in zip(
                        batch, inputs, result.images, result.prompt_token_counts, strict=True
                    ):
                        case_id = validate_case_id(str(row["case_id"]))
                        generation_dir = create_private_dir(
                            output_run / "cases" / case_id / "generation"
                        )
                        image_path = generation_dir / "generated_cxr.png"
                        dimensions = _save_image(image, image_path)
                        if dimensions != [DEFAULT_WIDTH, DEFAULT_HEIGHT]:
                            raise ValueError("generated image dimensions are unexpected")
                        record_path = write_private_json(
                            generation_dir / "generation.json",
                            {
                                "schema_version": GENERATION_SCHEMA,
                                "case_id": case_id,
                                "candidate_id": "radedit_official_t2i_seed0",
                                "modality": "cxr",
                                "producer": "frozen_radedit_t2i",
                                "frozen_model": True,
                                "input_signature": "radiology_style_text_to_cxr",
                                "editing_mode_used": False,
                                "radedit_revision": RADEDIT_REVISION,
                                "prompt_version": PROMPT_VERSION,
                                "prompt_sha256": str(case_input["prompt_sha256"]),
                                "underconditioned": bool(case_input["underconditioned"]),
                                "seed": int(case_input["seed"]),
                                "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                                "num_inference_steps": DEFAULT_INFERENCE_STEPS,
                                "dtype": "float32",
                                "prompt_token_count": int(token_count),
                                "batch_ordinal": batch_ordinal,
                                "output": {
                                    "image_dimensions": dimensions,
                                    "image_sha256": sha256_file(image_path),
                                },
                                "cost": {
                                    "model_calls": 1,
                                    "estimated_case_seconds": round(per_case_seconds, 3),
                                    "batch_seconds": round(result.elapsed_seconds, 3),
                                    "batch_peak_vram_gib": round(result.peak_vram_gib, 3),
                                },
                            },
                        )
                        completed.append(
                            {"case_id": case_id, "generation_sha256": sha256_file(record_path)}
                        )
        enforce_private_file_mode(protected_log)
        elapsed = time.monotonic() - total_started
        summary_path = write_private_json(
            output_run / "summary.json",
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "completed",
                "run_id": output_run_id,
                "case_count": len(completed),
                "image_dimensions": [DEFAULT_WIDTH, DEFAULT_HEIGHT],
                "elapsed_seconds": round(elapsed, 3),
                "peak_vram_gib": round(peak_vram, 3),
                "model_audit": model_audit,
                "cases": completed,
            },
        )
        frozen_cases = []
        for row in records:
            case_id = validate_case_id(str(row["case_id"]))
            generation_path = (
                output_run / "cases" / case_id / "generation" / "generation.json"
            )
            generation = read_private_json(generation_path)
            frozen_cases.append(
                {
                    "case_id": case_id,
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "seed": int(row["seed"]),
                    "generation_sha256": sha256_file(generation_path),
                    "image_sha256": str(generation["output"]["image_sha256"]),
                }
            )
        frozen_path = write_private_json(
            output_run / "frozen.json",
            {
                "schema_version": "tricompose.radedit.frozen_run.v1",
                "status": "frozen",
                "run_id": output_run_id,
                "case_count": len(frozen_cases),
                "radedit_revision": RADEDIT_REVISION,
                "model_snapshot_audit": model_audit,
                "model_prompt_version": PROMPT_VERSION,
                "manifest_sha256": sha256_file(output_run / "manifest.json"),
                "summary_sha256": sha256_file(summary_path),
                "cases": frozen_cases,
                "mutation_policy": "never overwrite or regenerate this run",
            },
        )
        return {
            "status": "completed",
            "case_count": len(completed),
            "image_dimensions": [DEFAULT_WIDTH, DEFAULT_HEIGHT],
            "elapsed_seconds": round(elapsed, 3),
            "peak_vram_gib": round(peak_vram, 3),
            "radedit_revision": RADEDIT_REVISION,
            "frozen_index_sha256": sha256_file(frozen_path),
        }
    except Exception as exc:
        failure_path = output_run / "failure.json"
        if not failure_path.exists():
            write_private_json(
                failure_path,
                {
                    "schema_version": SUMMARY_SCHEMA,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(**vars(args))
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


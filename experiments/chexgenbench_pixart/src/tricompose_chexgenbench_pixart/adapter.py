"""Generate protected CXR candidates with frozen CheXGenBench PixArt-Sigma."""

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
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_WIDTH,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_REPO_COMMIT,
    FrozenPixArtRuntime,
    validate_model_snapshot,
)
from .prompting import CLINICAL_PROMPT_VERSION, PROMPT_VERSION, build_pixart_prompt


SOURCE_SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
RUN_SCHEMA = "tricompose.chexgenbench_pixart.run.v1"
GENERATION_SCHEMA = "tricompose.chexgenbench_pixart.generation.v1"
SUMMARY_SCHEMA = "tricompose.chexgenbench_pixart.summary.v1"
SEED_OFFSET = 88_126_407


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _stage_cases(source_run_id: str, output_run_id: str, limit: int):
    source_run = require_source_run(source_run_id)
    selection = read_private_json(source_run / "selection.json")
    if selection.get("schema_version") != SOURCE_SELECTION_SCHEMA:
        raise ValueError("unsupported source selection schema")
    source_records = selection.get("cases")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("source selection has no cases")
    if limit < 1 or limit > len(source_records):
        raise ValueError("requested case limit is outside the source selection")

    output_run = create_private_dir(PROTECTED_EXPERIMENT_ROOT / output_run_id)
    cases_root = create_private_dir(output_run / "cases")
    create_private_dir(output_run / "logs")
    staged = []
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
        prompt = build_pixart_prompt(facts)
        prompt_hash = _hash_text(prompt.text)
        seed = (int(source_case["seed"]) + SEED_OFFSET) % (2**63 - 1)
        case_dir = create_private_dir(cases_root / case_id)
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": "tricompose.chexgenbench_pixart.case_input.v1",
                "case_id": case_id,
                "source_prompt_sha256": str(source_case["prompt_sha256"]),
                "clinical_prompt_version": CLINICAL_PROMPT_VERSION,
                "model_prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "pixart_prompt": prompt.text,
                "included_diagnoses": list(prompt.included_diagnoses),
                "included_devices": list(prompt.included_devices),
                "omitted_devices": list(prompt.omitted_devices),
                "measurements_used": False,
                "race_used": False,
                "seed": seed,
            },
        )
        staged.append({"case_id": case_id, "prompt_sha256": prompt_hash, "seed": seed})
    write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": RUN_SCHEMA,
            "run_id": output_run_id,
            "source_run_id": source_run_id,
            "producer": "frozen_chexgenbench_pixart_sigma",
            "scientific_signature": (
                "structured EHR -> deterministic radiology-style text -> "
                "text-conditioned CXR"
            ),
            "direct_ehr_to_cxr": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "official_repo_commit": OFFICIAL_REPO_COMMIT,
            "official_model_revision": OFFICIAL_MODEL_REVISION,
            "clinical_prompt_version": CLINICAL_PROMPT_VERSION,
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


def run(*, model_dir, source_run_id, output_run_id, limit, batch_size):
    os.umask(0o077)
    validate_run_id(source_run_id)
    validate_run_id(output_run_id)
    if batch_size != 1:
        raise ValueError("the initial PixArt adapter requires batch size one")
    model_audit = validate_model_snapshot(model_dir)
    output_run, records = _stage_cases(source_run_id, output_run_id, limit)
    started = time.monotonic()
    completed = []
    peak_vram = 0.0
    protected_log = output_run / "logs" / "pixart.log"
    try:
        with protected_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = FrozenPixArtRuntime(model_dir)
                for batch_ordinal, batch in enumerate(_chunks(records, batch_size)):
                    item = read_private_json(
                        output_run / "cases" / batch[0]["case_id"] / "input.json"
                    )
                    result = runtime.generate_batch([str(item["pixart_prompt"])], [int(item["seed"])])
                    peak_vram = max(peak_vram, result.peak_vram_gib)
                    case_id = validate_case_id(str(batch[0]["case_id"]))
                    generation_dir = create_private_dir(
                        output_run / "cases" / case_id / "generation"
                    )
                    image_path = generation_dir / "generated_cxr.png"
                    dimensions = _save_image(result.images[0], image_path)
                    if dimensions != [DEFAULT_WIDTH, DEFAULT_HEIGHT]:
                        raise ValueError("generated image dimensions are unexpected")
                    record_path = write_private_json(
                        generation_dir / "generation.json",
                        {
                            "schema_version": GENERATION_SCHEMA,
                            "case_id": case_id,
                            "candidate_id": "chexgenbench_pixart_sigma_seed0",
                            "modality": "cxr",
                            "producer": "frozen_chexgenbench_pixart_sigma",
                            "frozen_model": True,
                            "input_signature": "radiology_style_text_to_cxr",
                            "official_model_revision": OFFICIAL_MODEL_REVISION,
                            "prompt_version": PROMPT_VERSION,
                            "prompt_sha256": str(item["prompt_sha256"]),
                            "seed": int(item["seed"]),
                            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                            "num_inference_steps": DEFAULT_INFERENCE_STEPS,
                            "dtype": "float16",
                            "prompt_token_count": int(result.prompt_token_counts[0]),
                            "batch_ordinal": batch_ordinal,
                            "output": {
                                "image_dimensions": dimensions,
                                "image_sha256": sha256_file(image_path),
                            },
                            "cost": {
                                "model_calls": 1,
                                "estimated_case_seconds": round(result.elapsed_seconds, 3),
                                "batch_peak_vram_gib": round(result.peak_vram_gib, 3),
                            },
                        },
                    )
                    completed.append(
                        {"case_id": case_id, "generation_sha256": sha256_file(record_path)}
                    )
        enforce_private_file_mode(protected_log)
        elapsed = time.monotonic() - started
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
            generation_dir = output_run / "cases" / case_id / "generation"
            generation_path = generation_dir / "generation.json"
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
                "schema_version": "tricompose.chexgenbench_pixart.frozen_run.v1",
                "status": "frozen",
                "run_id": output_run_id,
                "case_count": len(frozen_cases),
                "model_revision": OFFICIAL_MODEL_REVISION,
                "model_snapshot_audit": model_audit,
                "clinical_prompt_version": CLINICAL_PROMPT_VERSION,
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
            "model_revision": OFFICIAL_MODEL_REVISION,
            "frozen_index_sha256": sha256_file(frozen_path),
        }
    except Exception as exc:
        failure_path = output_run / "failure.json"
        if not failure_path.exists():
            write_private_json(
                failure_path,
                {"schema_version": SUMMARY_SCHEMA, "status": "failed", "error_type": type(exc).__name__},
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
        print(json.dumps(run(**vars(args)), sort_keys=True), flush=True)
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

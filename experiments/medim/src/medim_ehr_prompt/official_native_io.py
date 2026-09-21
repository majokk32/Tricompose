"""Protected I/O staging for MeDiM's unmodified official inference entrypoint.

This module does not implement model inference, tokenization, masking, sampling,
or decoding. Those operations are performed by the external read-only MeDiM
``inference.py`` entrypoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from medim_ehr_prompt.common import (
    create_run_directory,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    read_private_json,
    require_run_directory,
    sha256_file,
    validate_case_id,
    write_private_json,
    write_private_text,
)


STAGING_SCHEMA = "medim_ehr_prompt.official_native_staging.v1"
GENERATION_SCHEMA = "medim_ehr_prompt.official_native_generation.v1"
EXPECTED_INPUT_SCHEMA = "medim_ehr_prompt.case_input.v1"
OFFICIAL_ENTRYPOINT = (
    "/project2/ruishanl_1185/MeDiM/MeDiM_original_infer/inference.py"
)


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    enforce_private_directory_mode(path)
    return path


def _recursive_private_modes(root: Path) -> None:
    enforce_private_directory_mode(root)
    for path in root.rglob("*"):
        if path.is_dir():
            enforce_private_directory_mode(path)
        elif path.is_file():
            enforce_private_file_mode(path)


def stage_official_dataset(
    source_run_id: str, output_run_id: str, sampling_steps: int = 100
) -> dict[str, Any]:
    """Create a protected paired-file shim required by the official dataset.

    The neutral image is only a structural input to the official paired-file
    dataloader. With ``eval.task=mask_image_only``, all image-token positions are
    replaced by the mask token before sampling, so it is not a previous CXR or
    an image condition.
    """

    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")

    os.umask(0o077)
    source_run = require_run_directory(source_run_id)
    output_run = create_run_directory(output_run_id)

    dataset_root = _private_dir(output_run / "dataset")
    mimic_dir = _private_dir(dataset_root / "mimic")
    _private_dir(dataset_root / "path")
    _private_dir(output_run / "native_output")
    _private_dir(output_run / "logs")
    _private_dir(output_run / "work")
    _private_dir(output_run / "hydra")
    cases_root = _private_dir(output_run / "cases")

    selection = read_private_json(source_run / "selection.json")
    records = selection.get("cases")
    if not isinstance(records, list) or not records:
        raise ValueError("source selection has no cases")

    entries: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records):
        case_id = validate_case_id(str(record["case_id"]))
        case_input = read_private_json(
            source_run / "cases" / case_id / "input.json"
        )
        if case_input.get("schema_version") != EXPECTED_INPUT_SCHEMA:
            raise ValueError("unsupported protected case input")
        prompt = str(case_input["medim_prompt"])
        if not prompt.strip():
            raise ValueError("empty protected prompt")

        stem = f"sample_{ordinal:03d}"
        text_path = mimic_dir / f"{stem}.txt"
        image_path = mimic_dir / f"{stem}.jpg"
        write_private_text(text_path, prompt)

        neutral = Image.new("RGB", (512, 512), color=(122, 116, 104))
        neutral.save(image_path, format="JPEG", quality=95)
        enforce_private_file_mode(image_path)

        case_dir = _private_dir(cases_root / case_id)
        write_private_json(
            case_dir / "source.json",
            {
                "schema_version": STAGING_SCHEMA,
                "case_id": case_id,
                "source_run_id": source_run_id,
                "source_prompt_sha256": str(case_input["prompt_sha256"]),
                "dataset_stem": stem,
            },
        )
        entries.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "dataset_stem": stem,
                "source_prompt_sha256": str(case_input["prompt_sha256"]),
            }
        )

    native_glob = [Path(path).stem for path in glob.glob(str(mimic_dir / "*.jpg"))]
    expected_glob = [entry["dataset_stem"] for entry in entries]
    if native_glob != expected_glob:
        raise RuntimeError("official dataset glob order is not deterministic")

    write_private_json(
        output_run / "official_native_manifest.json",
        {
            "schema_version": STAGING_SCHEMA,
            "source_run_id": source_run_id,
            "output_run_id": output_run_id,
            "official_entrypoint": OFFICIAL_ENTRYPOINT,
            "case_count": len(entries),
            "eval_task": "mask_image_only",
            "sampling_steps": sampling_steps,
            "cases": entries,
        },
    )
    _recursive_private_modes(output_run)
    return {"run_id": output_run_id, "staged_case_count": len(entries)}


def _numeric_image_key(path: Path) -> int:
    prefix = "image_"
    if not path.stem.startswith(prefix):
        raise ValueError("unexpected official image filename")
    return int(path.stem[len(prefix) :])


def finalize_official_outputs(output_run_id: str) -> dict[str, Any]:
    """Validate and organize images produced by official MeDiM inference."""

    os.umask(0o077)
    output_run = require_run_directory(output_run_id)
    manifest = read_private_json(output_run / "official_native_manifest.json")
    if manifest.get("schema_version") != STAGING_SCHEMA:
        raise ValueError("unsupported official-native manifest")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("official-native manifest has no cases")
    sampling_steps = int(manifest["sampling_steps"])
    if sampling_steps < 1:
        raise ValueError("invalid sampling_steps in official-native manifest")

    native_test = output_run / "native_output" / "test"
    images = sorted(native_test.glob("image_*.png"), key=_numeric_image_key)
    if len(images) != len(entries):
        raise RuntimeError("official inference output count mismatch")
    if [_numeric_image_key(path) for path in images] != list(range(len(entries))):
        raise RuntimeError("official inference output indices are incomplete")

    completed: list[dict[str, Any]] = []
    for entry, native_image in zip(entries, images, strict=True):
        case_id = validate_case_id(str(entry["case_id"]))
        generation_dir = _private_dir(output_run / "cases" / case_id / "generation")
        output_image = generation_dir / "generated_cxr.png"
        shutil.copyfile(native_image, output_image)
        enforce_private_file_mode(output_image)
        with Image.open(output_image) as image:
            dimensions = [int(image.size[0]), int(image.size[1])]
        if dimensions != [512, 512]:
            raise RuntimeError("official generated image has unexpected dimensions")

        image_hash = sha256_file(output_image)
        write_private_json(
            generation_dir / "generation.json",
            {
                "schema_version": GENERATION_SCHEMA,
                "run_id": output_run_id,
                "case_id": case_id,
                "producer": "official_medim_inference_py",
                "official_entrypoint": OFFICIAL_ENTRYPOINT,
                "official_experiments": ["large_scale_train", "eval_model"],
                "eval_task": "mask_image_only",
                "sampling_steps": sampling_steps,
                "source_prompt_sha256": str(entry["source_prompt_sha256"]),
                "image_dimensions": dimensions,
                "generated_cxr_sha256": image_hash,
            },
        )
        completed.append(
            {"case_id": case_id, "generated_cxr_sha256": image_hash}
        )

    write_private_json(
        output_run / "official_native_summary.json",
        {
            "schema_version": GENERATION_SCHEMA,
            "run_id": output_run_id,
            "completed_case_count": len(completed),
            "eval_task": "mask_image_only",
            "sampling_steps": sampling_steps,
            "cases": completed,
        },
    )
    _recursive_private_modes(output_run)
    return {"run_id": output_run_id, "completed_case_count": len(completed)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--source-run-id", required=True)
    stage.add_argument("--output-run-id", required=True)
    stage.add_argument("--sampling-steps", default=100, type=int)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "stage":
        result = stage_official_dataset(
            args.source_run_id, args.output_run_id, args.sampling_steps
        )
    else:
        result = finalize_official_outputs(args.output_run_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

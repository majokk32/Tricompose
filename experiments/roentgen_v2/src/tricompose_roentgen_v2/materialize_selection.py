"""Materialize one selector choice as a standard protected generation run.

This is a lightweight provenance stage. It creates hard links to already
generated protected synthetic images and converts candidate metadata to the
standard RoentGen-v2 run schema. It never loads a real CXR or source report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
)

from .adapter import CASE_SCHEMA, GENERATION_SCHEMA, RUN_SCHEMA, SUMMARY_SCHEMA
from .candidate_selector import PROTECTED_SELECTION_ROOT, SELECTION_SCHEMA
from .candidate_sweep import (
    CANDIDATE_SCHEMA,
    PROTECTED_SWEEP_ROOT,
    SWEEP_CASE_SCHEMA,
    SWEEP_RUN_SCHEMA,
)
from .common import (
    create_output_run,
    create_private_dir,
    read_private_json,
    sha256_file,
    validate_case_id,
    validate_run_id,
    write_private_json,
)


CHOICES = ("baseline", "selected")


def _require_run(root: Path, run_id: str, kind: str) -> Path:
    validate_run_id(run_id)
    resolved = require_inside(root / run_id, PROTECTED_ROOT, must_exist=True)
    if not resolved.is_dir():
        raise ValueError(f"protected {kind} run is not a directory")
    return resolved


def run(
    *,
    sweep_run_id: str,
    selection_run_id: str,
    output_run_id: str,
    choice: str,
) -> dict[str, Any]:
    os.umask(0o077)
    validate_run_id(sweep_run_id)
    validate_run_id(selection_run_id)
    validate_run_id(output_run_id)
    if choice not in CHOICES:
        raise ValueError("unsupported materialization choice")

    sweep_run = _require_run(
        PROTECTED_SWEEP_ROOT,
        sweep_run_id,
        "candidate sweep",
    )
    selection_run = _require_run(
        PROTECTED_SELECTION_ROOT,
        selection_run_id,
        "candidate selection",
    )
    sweep_manifest = read_private_json(sweep_run / "manifest.json")
    if sweep_manifest.get("schema_version") != SWEEP_RUN_SCHEMA:
        raise ValueError("unsupported candidate sweep manifest")
    selection = read_private_json(selection_run / "selection.json")
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported candidate selection artifact")
    if selection.get("candidate_sweep_run_id") != sweep_run_id:
        raise ValueError("selection and candidate sweep do not match")
    if selection.get("loads_real_target_cxr") is not False:
        raise ValueError("selection violated the real-CXR boundary")
    if selection.get("loads_real_target_report") is not False:
        raise ValueError("selection violated the report boundary")

    output_run = create_output_run(output_run_id)
    cases_root = create_private_dir(output_run / "cases")
    create_private_dir(output_run / "logs")
    case_rows = selection.get("cases")
    if not isinstance(case_rows, list) or not case_rows:
        raise ValueError("candidate selection has no cases")
    materialized: list[dict[str, Any]] = []

    for selection_row in case_rows:
        case_id = validate_case_id(str(selection_row["case_id"]))
        candidate_id = str(
            selection_row[
                "selected_candidate_id"
                if choice == "selected"
                else "baseline_candidate_id"
            ]
        )
        sweep_input = read_private_json(sweep_run / "cases" / case_id / "input.json")
        if sweep_input.get("schema_version") != SWEEP_CASE_SCHEMA:
            raise ValueError("unsupported candidate sweep case")
        candidate_dir = sweep_run / "cases" / case_id / "candidates" / candidate_id
        candidate = read_private_json(candidate_dir / "generation.json")
        if candidate.get("schema_version") != CANDIDATE_SCHEMA:
            raise ValueError("unsupported candidate generation artifact")
        if candidate.get("case_id") != case_id or candidate.get("candidate_id") != candidate_id:
            raise ValueError("candidate provenance mismatch")
        if candidate.get("prompt_sha256") != sweep_input.get("prompt_sha256"):
            raise ValueError("candidate prompt hash mismatch")
        source_image = require_private_file(candidate_dir / "generated_cxr.png")
        source_hash = sha256_file(source_image)
        if candidate.get("output", {}).get("image_sha256") != source_hash:
            raise ValueError("candidate image hash mismatch")

        case_dir = create_private_dir(cases_root / case_id)
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": CASE_SCHEMA,
                "case_id": case_id,
                "source_prompt_sha256": str(sweep_input["source_prompt_sha256"]),
                "prompt_version": str(sweep_input["prompt_version"]),
                "prompt_sha256": str(sweep_input["prompt_sha256"]),
                "roentgen_prompt": str(sweep_input["roentgen_prompt"]),
                "included_diagnoses": list(sweep_input["included_diagnoses"]),
                "included_devices": list(sweep_input["included_devices"]),
                "omitted_devices": list(sweep_input["omitted_devices"]),
                "measurements_used": False,
                "race_used": False,
                "seed": int(candidate["seed"]),
                "materialized_from_sweep_run_id": sweep_run_id,
                "materialized_from_selection_run_id": selection_run_id,
                "materialization_choice": choice,
                "source_candidate_id": candidate_id,
            },
        )
        generation_dir = create_private_dir(case_dir / "generation")
        target_image = generation_dir / "generated_cxr.png"
        os.link(source_image, target_image)
        enforce_private_file_mode(target_image)
        if sha256_file(target_image) != source_hash:
            raise RuntimeError("materialized image hash mismatch")
        generation_path = write_private_json(
            generation_dir / "generation.json",
            {
                "schema_version": GENERATION_SCHEMA,
                "case_id": case_id,
                "candidate_id": candidate_id,
                "modality": "cxr",
                "producer": "frozen_roentgen_v2_diffusers",
                "frozen_model": True,
                "input_signature": "radiology_style_text_to_cxr",
                "official_model_revision": str(candidate["official_model_revision"]),
                "prompt_version": str(candidate["prompt_version"]),
                "prompt_sha256": str(candidate["prompt_sha256"]),
                "seed": int(candidate["seed"]),
                "guidance_scale": float(candidate["guidance_scale"]),
                "num_inference_steps": int(candidate["num_inference_steps"]),
                "dtype": str(candidate["dtype"]),
                "prompt_token_count": int(candidate["prompt_token_count"]),
                "materialization_choice": choice,
                "source_candidate_id": candidate_id,
                "output": {
                    "image_dimensions": list(candidate["output"]["image_dimensions"]),
                    "image_sha256": source_hash,
                },
                "cost": dict(candidate["cost"]),
            },
        )
        materialized.append(
            {
                "case_id": case_id,
                "prompt_sha256": str(candidate["prompt_sha256"]),
                "generation_sha256": sha256_file(generation_path),
            }
        )

    manifest_path = write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": RUN_SCHEMA,
            "run_id": output_run_id,
            "source_run_id": str(sweep_manifest["source_run_id"]),
            "producer": "materialized_roentgen_v2_candidate_selection",
            "scientific_signature": (
                "reference-free frozen-XRV candidate selection -> standard CXR run"
            ),
            "direct_ehr_to_cxr": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "official_repo_commit": str(sweep_manifest["official_repo_commit"]),
            "official_model_revision": str(
                sweep_manifest["official_model_revision"]
            ),
            "prompt_version": str(sweep_manifest["prompt_version"]),
            "materialization_choice": choice,
            "candidate_sweep_run_id": sweep_run_id,
            "selection_run_id": selection_run_id,
            "case_count": len(materialized),
            "cases": materialized,
        },
    )
    summary_path = write_private_json(
        output_run / "summary.json",
        {
            "schema_version": SUMMARY_SCHEMA,
            "status": "completed",
            "run_id": output_run_id,
            "case_count": len(materialized),
            "materialization_choice": choice,
            "manifest_sha256": sha256_file(manifest_path),
            "uses_hard_links_to_protected_candidates": True,
        },
    )
    return {
        "stage": "roentgen_v2_materialize_selection",
        "status": "completed",
        "choice": choice,
        "case_count": len(materialized),
        "summary_sha256": sha256_file(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-run-id", required=True)
    parser.add_argument("--selection-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--choice", choices=CHOICES, required=True)
    args = parser.parse_args()
    try:
        result = run(
            sweep_run_id=args.sweep_run_id,
            selection_run_id=args.selection_run_id,
            output_run_id=args.output_run_id,
            choice=args.choice,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "roentgen_v2_materialize_selection",
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

"""Stage protected prompts without opening a real CXR or radiology report."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .common import (
    create_output_run,
    create_private_dir,
    read_private_json,
    require_source_run,
    validate_case_id,
    validate_model_name,
    write_private_json,
)
from .prompting import build_model_prompt


SOURCE_SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
RUN_SCHEMA = "tricompose.ehr_prompt_cxr.run.v1"
CASE_SCHEMA = "tricompose.ehr_prompt_cxr.case_input.v1"
SEED_OFFSETS = {
    "unidisc": 1_618_033,
    "liquid": 1_414_213,
}


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_cases(
    *, model_name: str, source_run_id: str, output_run_id: str, limit: int
) -> tuple[Path, list[dict[str, Any]]]:
    """Create a fresh protected run and its model-specific prompt inputs."""

    validate_model_name(model_name)
    source_run = require_source_run(source_run_id)
    selection = read_private_json(source_run / "selection.json")
    if selection.get("schema_version") != SOURCE_SELECTION_SCHEMA:
        raise ValueError("unsupported source selection schema")
    source_records = selection.get("cases")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("source selection has no cases")
    if limit < 1 or limit > len(source_records):
        raise ValueError("requested case limit is outside the source selection")

    output_run = create_output_run(model_name, output_run_id)
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

        prompt = build_model_prompt(model_name, facts)
        prompt_hash = hash_text(prompt.text)
        clinical_hash = hash_text(prompt.clinical_text)
        seed = (int(source_case["seed"]) + SEED_OFFSETS[model_name]) % (2**63 - 1)
        case_dir = create_private_dir(cases_root / case_id)
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": CASE_SCHEMA,
                "case_id": case_id,
                "model_name": model_name,
                "source_prompt_sha256": str(source_case["prompt_sha256"]),
                "clinical_prompt_version": prompt.clinical_prompt_version,
                "clinical_prompt_sha256": clinical_hash,
                "model_prompt_version": prompt.model_prompt_version,
                "model_prompt_sha256": prompt_hash,
                "model_prompt": prompt.text,
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
                "clinical_prompt_sha256": clinical_hash,
                "model_prompt_sha256": prompt_hash,
                "seed": seed,
            }
        )

    write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": RUN_SCHEMA,
            "run_id": output_run_id,
            "source_run_id": source_run_id,
            "model_name": model_name,
            "producer": f"frozen_{model_name}_text_to_image",
            "scientific_signature": (
                "structured EHR -> deterministic radiology-style prompt -> "
                "general text-conditioned image generator -> CXR candidate"
            ),
            "direct_ehr_to_cxr": False,
            "clinically_specialized_generator": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "case_count": len(staged),
            "cases": staged,
        },
    )
    return output_run, staged

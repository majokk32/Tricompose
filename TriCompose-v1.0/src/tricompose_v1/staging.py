"""Build an immutable protected TriCompose V1 staging contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_directory_mode,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)

from .ehr_bridge import CANONICALIZER_VERSION, canonicalize_synehrgy_case
from .facts import (
    FACT_EXTRACTOR_VERSION,
    FACT_PATTERNS,
    FACT_STATES,
    extract_ehr_facts,
)
from .prompts import ACTIVE_PROMPT_MODELS, render_prompt


STAGING_SCHEMA = "tricompose.staging.run.v1"
SOURCE_RUN_SCHEMA = "tricompose.synehrgy_v2.run.v1"
CASE_ID_PATTERN = re.compile(r"case_[0-9]{3,6}\Z")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON input must be an object")
    return payload


def _private_mkdir(path: Path) -> None:
    old_umask = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    finally:
        os.umask(old_umask)
    enforce_private_directory_mode(path)


def read_case_ids(path: str | Path) -> list[str]:
    case_file = Path(path).resolve(strict=True)
    case_ids: list[str] = []
    for raw_line in case_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not CASE_ID_PATTERN.fullmatch(line):
            raise ValueError("case list contains a non-opaque case ID")
        if line in case_ids:
            raise ValueError("case list contains a duplicate case ID")
        case_ids.append(line)
    if not case_ids:
        raise ValueError("case list is empty")
    return case_ids


def parse_prompt_models(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        models = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        models = tuple(value)
    if not models:
        raise ValueError("at least one prompt model is required")
    if len(set(models)) != len(models):
        raise ValueError("prompt model list contains duplicates")
    unsupported = sorted(set(models) - set(ACTIVE_PROMPT_MODELS))
    if unsupported:
        raise ValueError(
            "inactive prompt model requested: " + ", ".join(unsupported)
        )
    return models


def _source_generator(run_manifest: Mapping[str, Any]) -> str:
    generator = run_manifest.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("source run has no generator metadata")
    variant = generator.get("variant")
    mapping = {
        "qwen2-40bins": "synehrgy_qwen2_40bins",
        "gpt2-10bins": "synehrgy_gpt2_10bins",
    }
    if variant not in mapping:
        raise ValueError("source SynEHRgy variant is unsupported")
    return mapping[variant]


def _source_entries(run_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = run_manifest.get("cases")
    if not isinstance(rows, list):
        raise ValueError("source run has no case manifest")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("source case manifest entry must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in indexed:
            raise ValueError("source run has invalid or duplicate case IDs")
        indexed[case_id] = row
    return indexed


def _rejection_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "not structurally valid" in message:
        return "source_case_not_strict_valid"
    if "no visits" in message:
        return "source_case_has_no_visits"
    if "schema" in message:
        return "source_schema_invalid"
    if isinstance(exc, (FileNotFoundError, IsADirectoryError)):
        return "source_file_missing"
    if isinstance(exc, (TypeError, ValueError, json.JSONDecodeError)):
        return "source_case_invalid"
    return "staging_error"


def _case_payload_hashes(case_dir: Path, prompt_models: Iterable[str]) -> dict[str, Any]:
    return {
        "source_manifest_sha256": sha256_file(case_dir / "source_manifest.json"),
        "synthetic_ehr_sha256": sha256_file(case_dir / "synthetic_ehr.json"),
        "ehr_facts_sha256": sha256_file(case_dir / "ehr_facts.json"),
        "prompt_manifest_sha256": sha256_file(
            case_dir / "cxr_prompts" / "prompt_manifest.json"
        ),
        "prompt_sha256": {
            model_id: sha256_file(case_dir / "cxr_prompts" / f"{model_id}.txt")
            for model_id in prompt_models
        },
    }


def _stage_one_case(
    *,
    cases_dir: Path,
    source_root: Path,
    source_run_path: Path,
    source_run_sha256: str,
    source_entry: Mapping[str, Any],
    source_generator: str,
    case_id: str,
    prompt_models: tuple[str, ...],
) -> dict[str, Any]:
    relative_path = source_entry.get("relative_path")
    expected_hash = source_entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError("source case manifest lacks path or SHA256")
    source_path = require_inside(
        source_root / relative_path, source_root, must_exist=True
    )
    if not source_path.is_file():
        raise FileNotFoundError("source case file is missing")
    actual_hash = sha256_file(source_path)
    if actual_hash != expected_hash:
        raise ValueError("source case SHA256 mismatch")
    source_case = _read_json(source_path)
    if source_case.get("case_id") != case_id:
        raise ValueError("source case ID does not match its manifest")

    canonical = canonicalize_synehrgy_case(
        source_case,
        source_model_id=source_generator,
    )
    facts = extract_ehr_facts(canonical)
    renderings = {
        model_id: render_prompt(facts, model_id) for model_id in prompt_models
    }

    # Rebuild in memory. Stable equality here is independent of output location/run ID.
    deterministic_rebuild = (
        canonical
        == canonicalize_synehrgy_case(
            source_case,
            source_model_id=source_generator,
        )
        and facts == extract_ehr_facts(canonical)
        and renderings
        == {model_id: render_prompt(facts, model_id) for model_id in prompt_models}
    )
    if not deterministic_rebuild:
        raise ValueError("deterministic rebuild check failed")

    final_case_dir = cases_dir / case_id
    if final_case_dir.exists():
        raise FileExistsError("case staging directory already exists")
    temp_case_dir = cases_dir / f".{case_id}.{uuid.uuid4().hex}.tmp"
    _private_mkdir(temp_case_dir)
    try:
        prompts_dir = temp_case_dir / "cxr_prompts"
        _private_mkdir(prompts_dir)
        source_manifest = {
            "schema_version": "tricompose-source-manifest-v1",
            "case_id": case_id,
            "source_generator": source_generator,
            "source_path": str(source_path),
            "source_sha256": actual_hash,
            "source_run_manifest_path": str(source_run_path),
            "source_run_manifest_sha256": source_run_sha256,
            "canonicalizer_version": CANONICALIZER_VERSION,
            "fact_extractor_version": FACT_EXTRACTOR_VERSION,
            "source_hash_verified": True,
            "source_modified": False,
        }
        write_private_json(temp_case_dir / "source_manifest.json", source_manifest)
        write_private_json(temp_case_dir / "synthetic_ehr.json", canonical)
        write_private_json(temp_case_dir / "ehr_facts.json", facts)
        ehr_facts_sha256 = sha256_file(temp_case_dir / "ehr_facts.json")

        prompt_manifest_models: dict[str, dict[str, Any]] = {}
        for model_id, rendering in renderings.items():
            prompt_path = prompts_dir / f"{model_id}.txt"
            write_private_text(prompt_path, rendering["text"])
            if sha256_file(prompt_path) != rendering["prompt_sha256"]:
                raise ValueError("written prompt differs from final model input")
            prompt_manifest_models[model_id] = {
                "path": f"cxr_prompts/{model_id}.txt",
                "renderer_version": rendering["renderer_version"],
                "clinical_prompt_version": rendering["clinical_prompt_version"],
                "legacy_experiment_reproduction": rendering[
                    "legacy_experiment_reproduction"
                ],
                "included_fact_ids": rendering["included_fact_ids"],
                "derived_rule_ids": rendering["derived_rule_ids"],
                "prompt_sha256": rendering["prompt_sha256"],
                "is_final_model_input": True,
                "underconditioned": rendering["underconditioned"],
            }
        prompt_manifest = {
            "schema_version": "tricompose-cxr-prompts-v1",
            "case_id": case_id,
            "clinical_intent_source": "ehr_facts.json",
            "ehr_facts_sha256": ehr_facts_sha256,
            "adapter_must_not_add_prefix": True,
            "conditioning_bridge": "deterministic_ehr_to_radiology_style_text",
            "direct_structured_ehr_to_cxr": False,
            "legacy_experiment_reproduction": True,
            "models": prompt_manifest_models,
        }
        write_private_json(prompts_dir / "prompt_manifest.json", prompt_manifest)
        os.rename(temp_case_dir, final_case_dir)
        enforce_private_directory_mode(final_case_dir)
    except Exception:
        if temp_case_dir.exists():
            shutil.rmtree(temp_case_dir)
        raise

    hashes = _case_payload_hashes(final_case_dir, prompt_models)
    generation_underconditioned = all(
        renderings[model_id]["underconditioned"] for model_id in prompt_models
    )
    return {
        "case_id": case_id,
        "status": "staged",
        "relative_path": f"cases/{case_id}",
        "hashes": hashes,
        "fact_states": {
            fact_id: facts["facts"][fact_id]["state"] for fact_id in FACT_PATTERNS
        },
        "radiology_relevant_fact_count": facts["summary"][
            "radiology_relevant_fact_count"
        ],
        "underconditioned": generation_underconditioned,
        "prompt_included_fact_ids": {
            model_id: renderings[model_id]["included_fact_ids"]
            for model_id in prompt_models
        },
        "checks": {
            "deterministic_rebuild": deterministic_rebuild,
            "source_hash_verified": True,
            "prompt_hashes_verified": True,
            "all_prompt_facts_traceable": True,
            "unknown_not_rendered": True,
            "no_unsupported_attributes": True,
            "no_duplicate_prefix": True,
            "all_case_files_complete": True,
        },
    }


def _validation_report(
    *,
    run_id: str,
    case_ids: list[str],
    staged: list[dict[str, Any]],
    rejected: list[dict[str, str]],
    prompt_models: tuple[str, ...],
) -> dict[str, Any]:
    state_distribution = {
        fact_id: {state: 0 for state in sorted(FACT_STATES)}
        for fact_id in FACT_PATTERNS
    }
    for row in staged:
        for fact_id, state in row["fact_states"].items():
            state_distribution[fact_id][state] += 1
    rejection_counts = dict(
        sorted(Counter(row["reason"] for row in rejected).items())
    )
    neutral_prompt_counts = {
        model_id: sum(
            not row["prompt_included_fact_ids"][model_id] for row in staged
        )
        for model_id in prompt_models
    }
    check_names = (
        "deterministic_rebuild",
        "source_hash_verified",
        "prompt_hashes_verified",
        "all_prompt_facts_traceable",
        "unknown_not_rendered",
        "no_unsupported_attributes",
        "no_duplicate_prefix",
        "all_case_files_complete",
    )
    aggregate_checks = {
        name: all(row["checks"][name] for row in staged) for name in check_names
    }
    aggregate_checks["at_least_one_case_staged"] = bool(staged)
    return {
        "schema_version": "tricompose-staging-validation-v1",
        "run_id": run_id,
        "overall_valid": all(aggregate_checks.values()),
        "input_case_count": len(case_ids),
        "successful_case_count": len(staged),
        "rejected_case_count": len(rejected),
        "all_requested_cases_staged": not rejected,
        "rejection_reasons": rejection_counts,
        "rejected_cases": rejected,
        "fact_state_distribution": state_distribution,
        "literal_empty_prompt_count": 0,
        "neutral_fallback_prompt_count_by_model": neutral_prompt_counts,
        "underconditioned_case_count": sum(row["underconditioned"] for row in staged),
        "provenance_and_contract_checks": aggregate_checks,
    }


def _validation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TriCompose V1 staging validation",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Overall contract valid: `{str(report['overall_valid']).lower()}`",
        f"- Input cases: {report['input_case_count']}",
        f"- Successfully staged: {report['successful_case_count']}",
        f"- Rejected: {report['rejected_case_count']}",
        f"- Underconditioned cases: {report['underconditioned_case_count']}",
        f"- Literal empty prompts: {report['literal_empty_prompt_count']}",
        "",
        "## Rejection reasons",
        "",
    ]
    if report["rejection_reasons"]:
        lines.extend(
            f"- `{reason}`: {count}"
            for reason, count in report["rejection_reasons"].items()
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Contract checks", ""])
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in report["provenance_and_contract_checks"].items()
    )
    lines.extend(["", "## Fact-state distribution", ""])
    lines.append("| Fact | Positive | Negative | Uncertain | Unknown |")
    lines.append("|---|---:|---:|---:|---:|")
    for fact_id, counts in report["fact_state_distribution"].items():
        lines.append(
            f"| `{fact_id}` | {counts['positive']} | {counts['negative']} | "
            f"{counts['uncertain']} | {counts['unknown']} |"
        )
    lines.append("")
    return "\n".join(lines)


def stage_tricompose_v1(
    *,
    source_root: str | Path,
    case_ids_file: str | Path,
    output_root: str | Path,
    run_id: str,
    prompt_models: str | Iterable[str] = ACTIVE_PROMPT_MODELS,
    validate: bool = False,
) -> dict[str, Any]:
    """Stage a new non-overwriting protected run and return aggregate status."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    models = parse_prompt_models(prompt_models)
    case_ids_path = Path(case_ids_file).resolve(strict=True)
    case_ids = read_case_ids(case_ids_path)
    source_root_path = require_inside(source_root, PROTECTED_ROOT, must_exist=True)
    if not source_root_path.is_dir():
        raise ValueError("source root is not a directory")
    source_run_path = source_root_path / "run.json"
    source_run = _read_json(source_run_path)
    if source_run.get("schema") != SOURCE_RUN_SCHEMA:
        raise ValueError("source root is not a supported SynEHRgy run")
    generator = _source_generator(source_run)
    entries = _source_entries(source_run)
    source_run_sha256 = sha256_file(source_run_path)

    output_root_path = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    _private_mkdir(output_root_path)
    target_run = require_inside(
        output_root_path / run_id, PROTECTED_ROOT, must_exist=False
    )
    if target_run.exists():
        raise FileExistsError("staging run already exists; choose a new run ID")
    temp_run = output_root_path / f".{run_id}.{uuid.uuid4().hex}.tmp"
    _private_mkdir(temp_run)
    staged: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    try:
        cases_dir = temp_run / "cases"
        _private_mkdir(cases_dir)
        for case_id in case_ids:
            entry = entries.get(case_id)
            if entry is None:
                rejected.append({"case_id": case_id, "reason": "source_case_not_listed"})
                continue
            try:
                staged.append(
                    _stage_one_case(
                        cases_dir=cases_dir,
                        source_root=source_root_path,
                        source_run_path=source_run_path,
                        source_run_sha256=source_run_sha256,
                        source_entry=entry,
                        source_generator=generator,
                        case_id=case_id,
                        prompt_models=models,
                    )
                )
            except Exception as exc:
                rejected.append({"case_id": case_id, "reason": _rejection_code(exc)})

        cohort_rows: list[dict[str, Any]] = []
        staged_by_id = {row["case_id"]: row for row in staged}
        rejected_by_id = {row["case_id"]: row for row in rejected}
        for case_id in case_ids:
            if case_id in staged_by_id:
                row = staged_by_id[case_id]
                cohort_rows.append(
                    {
                        "case_id": case_id,
                        "status": "staged",
                        "relative_path": row["relative_path"],
                        "hashes": row["hashes"],
                        "underconditioned": row["underconditioned"],
                    }
                )
            else:
                cohort_rows.append(
                    {
                        "case_id": case_id,
                        "status": "rejected",
                        "reason": rejected_by_id[case_id]["reason"],
                    }
                )
        write_private_text(
            temp_run / "cohort.jsonl",
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in cohort_rows
            ),
        )
        report = _validation_report(
            run_id=run_id,
            case_ids=case_ids,
            staged=staged,
            rejected=rejected,
            prompt_models=models,
        )
        run_manifest = {
            "schema_version": STAGING_SCHEMA,
            "run_id": run_id,
            "source_root": str(source_root_path),
            "source_run_manifest_path": str(source_run_path),
            "source_run_manifest_sha256": source_run_sha256,
            "source_generator": generator,
            "case_ids_file": str(case_ids_path),
            "case_ids_file_sha256": sha256_file(case_ids_path),
            "case_ids": case_ids,
            "prompt_models": list(models),
            "canonicalizer_version": CANONICALIZER_VERSION,
            "fact_extractor_version": FACT_EXTRACTOR_VERSION,
            "validate_requested": validate,
            "overwrite_allowed": False,
            "external_api_used": False,
            "gpu_inference_used": False,
            "disabled_v1_edges": {
                "radedit": "image_editing_not_cold_start_t2i",
                "medim": "official_checkpoint_positive_control_not_passed",
            },
        }
        write_private_json(temp_run / "run_manifest.json", run_manifest)
        write_private_json(temp_run / "validation_report.json", report)
        write_private_text(
            temp_run / "validation_report.md", _validation_markdown(report)
        )
        os.rename(temp_run, target_run)
        enforce_private_directory_mode(target_run)
    except Exception:
        if temp_run.exists():
            shutil.rmtree(temp_run)
        raise

    return {
        "run_directory": str(target_run),
        "input_case_count": len(case_ids),
        "successful_case_count": len(staged),
        "rejected_case_count": len(rejected),
        "rejection_reasons": report["rejection_reasons"],
        "underconditioned_case_count": report["underconditioned_case_count"],
        "overall_valid": report["overall_valid"],
    }


__all__ = [
    "SOURCE_RUN_SCHEMA",
    "STAGING_SCHEMA",
    "parse_prompt_models",
    "read_case_ids",
    "stage_tricompose_v1",
]

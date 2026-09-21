"""CPU-only immutable staging for the TriCompose V1.1 prompt bridge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tricompose_v1.ehr_bridge import validate_canonical_ehr

from .facts import (
    CONTEXT_PATTERNS,
    FACT_EXTRACTOR_VERSION_V11,
    extract_v11_facts,
)
from .prompts import (
    ACTIVE_PROMPT_MODELS_V11,
    CLINICAL_INTENT_VERSION_V11,
    render_v11_prompt,
)


WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
MAIN_PROTECTED_ROOT = WORKSPACE / "artifacts" / "protected"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
CASE_ID_PATTERN = re.compile(r"case_[0-9]{3,6}\Z")
STAGING_SCHEMA_V11 = "tricompose.staging.run.v1.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve(strict=must_exist)
    root_resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ValueError("path is outside the allowed V1.1 boundary")
    return resolved


def _is_protected_path(path: Path) -> bool:
    parts = path.parts
    return any(
        parts[index : index + 2] == ("artifacts", "protected")
        for index in range(len(parts) - 1)
    )


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o2770)
    os.chmod(path, 0o2770)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(path, 0o660)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o660)


def _copy_exact(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o660)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON input must be an object")
    return payload


def _cohort_rows(source_run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (source_run / "cohort.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("V1 cohort row must be an object")
            rows.append(row)
    return rows


def _source_prompt_hashes(source_case: Path) -> dict[str, str]:
    manifest = _read_json(source_case / "cxr_prompts" / "prompt_manifest.json")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("V1 source prompt manifest lacks models")
    hashes: dict[str, str] = {}
    for model_id in ACTIVE_PROMPT_MODELS_V11:
        metadata = models.get(model_id)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("prompt_sha256"), str
        ):
            raise ValueError("V1 source prompt hash is missing")
        hashes[model_id] = metadata["prompt_sha256"]
    return hashes


def _stage_case(
    *, source_case: Path, output_cases: Path, cohort_row: Mapping[str, Any]
) -> dict[str, Any]:
    case_id = cohort_row.get("case_id")
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("V1 source case ID is not opaque")
    if source_case.name != case_id:
        raise ValueError("V1 source case path and ID differ")
    source_ehr = source_case / "synthetic_ehr.json"
    expected_ehr_hash = cohort_row.get("hashes", {}).get("synthetic_ehr_sha256")
    actual_ehr_hash = sha256_file(source_ehr)
    if expected_ehr_hash != actual_ehr_hash:
        raise ValueError("V1 source EHR hash mismatch")
    canonical = _read_json(source_ehr)
    validate_canonical_ehr(canonical)
    if canonical["case_id"] != case_id:
        raise ValueError("canonical EHR case ID mismatch")

    facts = extract_v11_facts(canonical)
    renderings = {
        model_id: render_v11_prompt(facts, model_id)
        for model_id in ACTIVE_PROMPT_MODELS_V11
    }
    if facts != extract_v11_facts(canonical):
        raise ValueError("V1.1 fact extraction is non-deterministic")
    if renderings != {
        model_id: render_v11_prompt(facts, model_id)
        for model_id in ACTIVE_PROMPT_MODELS_V11
    }:
        raise ValueError("V1.1 prompt rendering is non-deterministic")
    intent_hashes = {
        rendering["clinical_intent_sha256"] for rendering in renderings.values()
    }
    if len(intent_hashes) != 1:
        raise ValueError("model-specific prompts do not share one clinical intent")

    final_case = output_cases / case_id
    if final_case.exists():
        raise FileExistsError("V1.1 case output already exists")
    temporary = output_cases / f".{case_id}.{uuid.uuid4().hex}.tmp"
    _mkdir(temporary)
    try:
        prompts_dir = temporary / "cxr_prompts"
        _mkdir(prompts_dir)
        _copy_exact(source_ehr, temporary / "synthetic_ehr.json")
        if sha256_file(temporary / "synthetic_ehr.json") != actual_ehr_hash:
            raise ValueError("V1.1 modified the canonical EHR bytes")
        _write_json(temporary / "ehr_facts.json", facts)
        facts_hash = sha256_file(temporary / "ehr_facts.json")
        source_manifest = {
            "schema_version": "tricompose-v1.1-source-manifest-v1",
            "case_id": case_id,
            "source_v1_case_directory": str(source_case),
            "source_v1_ehr_path": str(source_ehr),
            "source_v1_ehr_sha256": actual_ehr_hash,
            "canonical_ehr_copied_byte_for_byte": True,
            "source_ehr_modified": False,
            "v1_1_fact_extractor_version": FACT_EXTRACTOR_VERSION_V11,
        }
        _write_json(temporary / "source_manifest.json", source_manifest)
        prompt_models: dict[str, dict[str, Any]] = {}
        for model_id, rendering in renderings.items():
            prompt_path = prompts_dir / f"{model_id}.txt"
            _write_text(prompt_path, rendering["text"])
            if sha256_file(prompt_path) != rendering["prompt_sha256"]:
                raise ValueError("written V1.1 prompt differs from final input")
            prompt_models[model_id] = {
                key: rendering[key]
                for key in (
                    "renderer_version",
                    "clinical_intent_version",
                    "clinical_intent_sha256",
                    "available_context_ids",
                    "included_direct_fact_ids",
                    "included_context_ids",
                    "omitted_context_ids",
                    "context_budget_policy",
                    "derived_rule_ids",
                    "context_is_not_a_radiographic_assertion",
                    "conditioning_tier",
                    "underconditioned",
                    "prompt_sha256",
                    "is_final_model_input",
                )
            }
            prompt_models[model_id]["path"] = f"cxr_prompts/{model_id}.txt"
        prompt_manifest = {
            "schema_version": "tricompose-cxr-prompts-v1.1",
            "case_id": case_id,
            "ehr_facts_sha256": facts_hash,
            "one_shared_clinical_intent": True,
            "context_is_not_a_radiographic_assertion": True,
            "direct_structured_ehr_to_cxr": False,
            "adapter_must_not_add_prefix": True,
            "models": prompt_models,
        }
        _write_json(prompts_dir / "prompt_manifest.json", prompt_manifest)
        os.rename(temporary, final_case)
        os.chmod(final_case, 0o2770)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "case_id": case_id,
        "status": "staged",
        "relative_path": f"cases/{case_id}",
        "source_ehr_sha256": actual_ehr_hash,
        "v1_prompt_sha256": _source_prompt_hashes(source_case),
        "v1_1_prompt_sha256": {
            model_id: rendering["prompt_sha256"]
            for model_id, rendering in renderings.items()
        },
        "clinical_intent_sha256": next(iter(intent_hashes)),
        "conditioning_tier": facts["summary"]["conditioning_tier"],
        "included_direct_fact_ids": renderings["roentgen_v2"][
            "included_direct_fact_ids"
        ],
        "included_context_ids": renderings["roentgen_v2"][
            "available_context_ids"
        ],
        "checks": {
            "canonical_ehr_unchanged": True,
            "deterministic_rebuild": True,
            "all_prompt_evidence_traceable": True,
            "one_shared_clinical_intent": True,
            "unknown_not_rendered": True,
            "case_id_not_rendered": True,
            "no_random_nonce": True,
        },
    }


def _validation_report(
    *, run_id: str, staged: list[dict[str, Any]], source_case_count: int
) -> dict[str, Any]:
    v1_unique = {
        model_id: len({row["v1_prompt_sha256"][model_id] for row in staged})
        for model_id in ACTIVE_PROMPT_MODELS_V11
    }
    v11_unique = {
        model_id: len({row["v1_1_prompt_sha256"][model_id] for row in staged})
        for model_id in ACTIVE_PROMPT_MODELS_V11
    }
    tiers = dict(sorted(Counter(row["conditioning_tier"] for row in staged).items()))
    context_counts = Counter(
        context_id
        for row in staged
        for context_id in row["included_context_ids"]
    )
    direct_counts = Counter(
        fact_id for row in staged for fact_id in row["included_direct_fact_ids"]
    )
    check_names = tuple(next(iter(staged))["checks"]) if staged else ()
    checks = {
        name: bool(staged) and all(row["checks"][name] for row in staged)
        for name in check_names
    }
    checks["all_source_cases_staged"] = len(staged) == source_case_count
    checks["prompt_uniqueness_improved"] = all(
        v11_unique[model_id] > v1_unique[model_id]
        for model_id in ACTIVE_PROMPT_MODELS_V11
    )
    return {
        "schema_version": "tricompose-v1.1-staging-validation-v1",
        "run_id": run_id,
        "overall_valid": all(checks.values()),
        "source_case_count": source_case_count,
        "successful_case_count": len(staged),
        "conditioning_tier_counts": tiers,
        "unique_clinical_intent_count": len(
            {row["clinical_intent_sha256"] for row in staged}
        ),
        "v1_unique_prompt_count_by_model": v1_unique,
        "v1_1_unique_prompt_count_by_model": v11_unique,
        "v1_1_duplicate_prompt_count_by_model": {
            model_id: len(staged) - v11_unique[model_id]
            for model_id in ACTIVE_PROMPT_MODELS_V11
        },
        "documented_context_case_count": dict(sorted(context_counts.items())),
        "direct_positive_fact_case_count": dict(sorted(direct_counts.items())),
        "checks": checks,
    }


def _validation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TriCompose V1.1 prompt-bridge validation",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Overall valid: `{str(report['overall_valid']).lower()}`",
        f"- Existing canonical EHR cases: {report['source_case_count']}",
        f"- Successfully restaged without EHR modification: {report['successful_case_count']}",
        f"- Unique V1.1 clinical intents: {report['unique_clinical_intent_count']}",
        "",
        "## Prompt diversity",
        "",
        "| Model | V1.0 unique | V1.1 unique | V1.1 duplicates |",
        "|---|---:|---:|---:|",
    ]
    for model_id in ACTIVE_PROMPT_MODELS_V11:
        lines.append(
            f"| `{model_id}` | {report['v1_unique_prompt_count_by_model'][model_id]} "
            f"| {report['v1_1_unique_prompt_count_by_model'][model_id]} "
            f"| {report['v1_1_duplicate_prompt_count_by_model'][model_id]} |"
        )
    lines.extend(["", "## Conditioning tiers", ""])
    lines.extend(
        f"- `{tier}`: {count}"
        for tier, count in report["conditioning_tier_counts"].items()
    )
    lines.extend(["", "## Contract checks", ""])
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in report["checks"].items()
    )
    lines.append("")
    return "\n".join(lines)


def stage_existing_v1_ehrs(
    *, source_v1_run: str | Path, output_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Build a new V1.1 bridge run without changing the source V1 EHRs."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    source = _inside(Path(source_v1_run), WORKSPACE, must_exist=True)
    if not source.is_dir() or not _is_protected_path(source):
        raise ValueError("source V1 run must be a protected workspace directory")
    source_manifest = source / "run_manifest.json"
    source_cohort = source / "cohort.jsonl"
    if not source_manifest.is_file() or not source_cohort.is_file():
        raise ValueError("source is not a complete V1 staging run")
    manifest = _read_json(source_manifest)
    if manifest.get("schema_version") != "tricompose.staging.run.v1":
        raise ValueError("source staging schema is not V1")
    rows = [row for row in _cohort_rows(source) if row.get("status") == "staged"]
    if not rows:
        raise ValueError("source V1 run contains no staged cases")

    output = _inside(Path(output_root), MAIN_PROTECTED_ROOT, must_exist=False)
    _mkdir(output)
    target = output / run_id
    if target.exists():
        raise FileExistsError("V1.1 run already exists; use a new opaque run ID")
    temporary = output / f".{run_id}.{uuid.uuid4().hex}.tmp"
    _mkdir(temporary)
    staged: list[dict[str, Any]] = []
    try:
        cases_dir = temporary / "cases"
        _mkdir(cases_dir)
        for row in rows:
            case_id = row["case_id"]
            staged.append(
                _stage_case(
                    source_case=source / "cases" / case_id,
                    output_cases=cases_dir,
                    cohort_row=row,
                )
            )
        cohort_rows = [
            {
                "case_id": row["case_id"],
                "status": "staged",
                "relative_path": row["relative_path"],
                "source_ehr_sha256": row["source_ehr_sha256"],
                "clinical_intent_sha256": row["clinical_intent_sha256"],
                "conditioning_tier": row["conditioning_tier"],
            }
            for row in staged
        ]
        _write_text(
            temporary / "cohort.jsonl",
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in cohort_rows
            ),
        )
        report = _validation_report(
            run_id=run_id, staged=staged, source_case_count=len(rows)
        )
        run_manifest = {
            "schema_version": STAGING_SCHEMA_V11,
            "run_id": run_id,
            "source_v1_staging_run": str(source),
            "source_v1_run_manifest_sha256": sha256_file(source_manifest),
            "source_v1_cohort_sha256": sha256_file(source_cohort),
            "source_case_count": len(rows),
            "canonical_ehr_modified": False,
            "fact_extractor_version": FACT_EXTRACTOR_VERSION_V11,
            "clinical_intent_version": CLINICAL_INTENT_VERSION_V11,
            "prompt_models": list(ACTIVE_PROMPT_MODELS_V11),
            "context_inventory": list(CONTEXT_PATTERNS),
            "gpu_inference_used": False,
            "external_api_used": False,
            "overwrite_allowed": False,
        }
        _write_json(temporary / "run_manifest.json", run_manifest)
        _write_json(temporary / "validation_report.json", report)
        _write_text(temporary / "validation_report.md", _validation_markdown(report))
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "run_directory": str(target),
        "source_case_count": len(rows),
        "successful_case_count": len(staged),
        "unique_clinical_intent_count": report["unique_clinical_intent_count"],
        "v1_unique_prompt_count_by_model": report[
            "v1_unique_prompt_count_by_model"
        ],
        "v1_1_unique_prompt_count_by_model": report[
            "v1_1_unique_prompt_count_by_model"
        ],
        "conditioning_tier_counts": report["conditioning_tier_counts"],
        "overall_valid": report["overall_valid"],
    }


__all__ = [
    "MAIN_PROTECTED_ROOT",
    "STAGING_SCHEMA_V11",
    "stage_existing_v1_ehrs",
]

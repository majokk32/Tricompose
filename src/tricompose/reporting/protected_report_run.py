"""Generic protected runner for frozen CXR-to-report models."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_file_mode,
    sha256_file,
    write_private_json,
    write_private_text,
)

from .protected_cxr import load_frozen_cxr_source


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class GeneratedReport:
    canonical_text: str
    findings: str | None
    impression: str | None
    elapsed_seconds: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReportModelSpec:
    name: str
    artifact_namespace: str
    revision: str
    run_schema: str
    generation_schema: str
    frozen_schema: str
    output_sections: tuple[str, ...]


class FrozenReportRuntime(Protocol):
    audit: dict[str, Any]

    def generate(self, image_path: Path) -> GeneratedReport: ...

    @property
    def peak_vram_gib(self) -> float: ...


def _validate_run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid opaque run ID")
    return value


def run_report_generation(
    *,
    spec: ReportModelSpec,
    runtime_factory: Callable[[str | Path], FrozenReportRuntime],
    model_dir: str | Path,
    model_audit: dict[str, Any],
    source_model: str,
    source_run_id: str,
    output_run_id: str,
    limit: int,
) -> dict[str, Any]:
    os.umask(0o077)
    output_run_id = _validate_run_id(output_run_id)
    source = load_frozen_cxr_source(
        source_model=source_model,
        source_run_id=source_run_id,
        limit=limit,
    )
    output_root = PROTECTED_ROOT / spec.artifact_namespace / "runs"
    output_run = create_private_stage_dir(output_root / output_run_id)
    cases_root = create_private_stage_dir(output_run / "cases")
    logs_root = create_private_stage_dir(output_run / "logs")
    manifest_path = write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": spec.run_schema,
            "run_id": output_run_id,
            "producer": spec.name,
            "model_revision": spec.revision,
            "frozen_model": True,
            "input_signature": "single_current_synthetic_cxr_to_report",
            "uses_structured_ehr": False,
            "uses_prior_cxr": False,
            "uses_prior_report": False,
            "loads_real_target_cxr": False,
            "loads_real_target_report": False,
            "source_cxr_model": source.model_name,
            "source_cxr_run_id": source.run_id,
            "source_cxr_frozen_sha256": source.frozen_sha256,
            "source_ehr_run_id": source.source_ehr_run_id,
            "case_count": len(source.cases),
            "cases": [
                {
                    "case_id": case.case_id,
                    "source_image_sha256": case.image_sha256,
                    "source_generation_sha256": case.generation_sha256,
                }
                for case in source.cases
            ],
        },
    )

    protected_log = logs_root / f"{spec.name}.log"
    completed: list[dict[str, str]] = []
    started = time.monotonic()
    try:
        with protected_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = runtime_factory(model_dir)
                if runtime.audit != model_audit:
                    raise ValueError("runtime model audit changed after staging")
                for case in source.cases:
                    result = runtime.generate(case.image_path)
                    if not result.canonical_text.strip():
                        raise ValueError("report model returned empty text")
                    case_dir = create_private_stage_dir(cases_root / case.case_id)
                    report_path = write_private_text(
                        case_dir / "generated_report.txt",
                        result.canonical_text.strip() + "\n",
                    )
                    section_files: dict[str, dict[str, str]] = {}
                    for section_name, section_text in (
                        ("findings", result.findings),
                        ("impression", result.impression),
                    ):
                        if section_text is None or not section_text.strip():
                            continue
                        section_path = write_private_text(
                            case_dir / f"{section_name}.txt",
                            section_text.strip() + "\n",
                        )
                        section_files[section_name] = {
                            "file": section_path.name,
                            "sha256": sha256_file(section_path),
                        }
                    generation_path = write_private_json(
                        case_dir / "generation.json",
                        {
                            "schema_version": spec.generation_schema,
                            "case_id": case.case_id,
                            "candidate_id": f"{spec.name}_greedy",
                            "modality": "radiology_report",
                            "producer": spec.name,
                            "model_revision": spec.revision,
                            "frozen_model": True,
                            "input_signature": "single_current_synthetic_cxr_to_report",
                            "source_image_sha256": case.image_sha256,
                            "source_generation_sha256": case.generation_sha256,
                            "output": {
                                "report_file": report_path.name,
                                "report_sha256": sha256_file(report_path),
                                "sections": section_files,
                            },
                            "decoding": result.metadata,
                            "cost": {
                                "model_calls": 1,
                                "elapsed_seconds": round(result.elapsed_seconds, 3),
                            },
                        },
                    )
                    completed.append(
                        {
                            "case_id": case.case_id,
                            "generation_sha256": sha256_file(generation_path),
                            "report_sha256": sha256_file(report_path),
                        }
                    )
        enforce_private_file_mode(protected_log)
        elapsed = time.monotonic() - started
        summary_path = write_private_json(
            output_run / "summary.json",
            {
                "schema_version": f"{spec.run_schema}.summary",
                "status": "completed",
                "run_id": output_run_id,
                "case_count": len(completed),
                "elapsed_seconds": round(elapsed, 3),
                "peak_vram_gib": round(runtime.peak_vram_gib, 3),
                "model_audit": model_audit,
                "cases": completed,
            },
        )
        frozen_path = write_private_json(
            output_run / "frozen.json",
            {
                "schema_version": spec.frozen_schema,
                "status": "frozen",
                "run_id": output_run_id,
                "model_revision": spec.revision,
                "case_count": len(completed),
                "model_audit": model_audit,
                "source_cxr_frozen_sha256": source.frozen_sha256,
                "manifest_sha256": sha256_file(manifest_path),
                "summary_sha256": sha256_file(summary_path),
                "cases": completed,
                "mutation_policy": "never overwrite or regenerate this run",
            },
        )
        return {
            "status": "completed",
            "model": spec.name,
            "case_count": len(completed),
            "elapsed_seconds": round(elapsed, 3),
            "peak_vram_gib": round(runtime.peak_vram_gib, 3),
            "frozen_index_sha256": sha256_file(frozen_path),
        }
    except Exception as exc:
        failure_path = output_run / "failure.json"
        if not failure_path.exists():
            write_private_json(
                failure_path,
                {
                    "schema_version": f"{spec.run_schema}.failure",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
        raise

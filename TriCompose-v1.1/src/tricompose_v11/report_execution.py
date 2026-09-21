"""Slurm-only execution of hash-bound TriCompose V1.1 report requests."""

from __future__ import annotations

import contextlib
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .cxr_contracts import (
    MAIN_PROTECTED_ROOT,
    canonical_json_sha256,
    private_directory,
    read_json,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)
from .report_contracts import (
    ACTIVE_REPORT_MODELS_V11,
    load_model_requests,
    validate_report_request,
)


REPORT_CANDIDATE_SCHEMA_V11 = "tricompose-report-candidate-v1.1"
REPORT_CANDIDATE_RUN_SCHEMA_V11 = "tricompose-report-candidate-run-v1.1"


class GeneratedReportLike(Protocol):
    canonical_text: str
    elapsed_seconds: float


class FrozenReportRuntime(Protocol):
    audit: dict[str, Any]

    def generate(self, image_path: Path) -> GeneratedReportLike: ...

    @property
    def peak_vram_gib(self) -> float: ...


def validate_report_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("schema_version") != REPORT_CANDIDATE_SCHEMA_V11:
        raise ValueError("unsupported V1.1 report candidate schema")
    if candidate.get("frozen_model") is not True:
        raise ValueError("V1.1 report model is not frozen")
    if candidate.get("model_input_signature") != "single_current_synthetic_cxr":
        raise ValueError("V1.1 report candidate has the wrong input signature")
    if candidate.get("structured_ehr_content_supplied_to_model") is not False:
        raise ValueError("V1.1 CXR-only report candidate unexpectedly used EHR")
    if candidate.get("source_report_or_real_target_supplied") is not False:
        raise ValueError("V1.1 report candidate used a source target")
    artifact = candidate.get("artifact")
    if not isinstance(artifact, dict):
        raise TypeError("V1.1 report artifact is missing")
    path = require_inside(
        str(artifact.get("path", "")), MAIN_PROTECTED_ROOT, must_exist=True
    )
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise ValueError("V1.1 report artifact hash mismatch")
    if artifact.get("mime_type") != "text/plain":
        raise ValueError("V1.1 report artifact is not plain text")


def run_report_request_run(
    *,
    request_run: str | Path,
    output_root: str | Path,
    output_run_id: str,
    model_id: str,
    model_revision: str,
    model_audit: Mapping[str, Any],
    runtime_factory: Callable[[], FrozenReportRuntime],
) -> dict[str, Any]:
    """Run one frozen CXR-to-report expert over exact V1.1 requests."""

    if model_id not in ACTIVE_REPORT_MODELS_V11:
        raise ValueError("inactive V1.1 report model")
    source = require_inside(request_run, MAIN_PROTECTED_ROOT, must_exist=True)
    requests = load_model_requests(source, model_id)
    output = require_inside(output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    target = output / output_run_id
    if target.exists():
        raise FileExistsError("V1.1 report output run already exists")
    temporary = output / f".{output_run_id}.{uuid.uuid4().hex}.tmp"
    private_directory(temporary)
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    try:
        candidates_root = temporary / "candidates"
        logs_root = temporary / "logs"
        private_directory(candidates_root)
        private_directory(logs_root)
        log_path = logs_root / f"{model_id}.log"
        with log_path.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                runtime = runtime_factory()
                if runtime.audit != dict(model_audit):
                    raise ValueError("V1.1 frozen report model audit changed at load time")
                for request in requests:
                    validate_report_request(request)
                    cxr = request["inputs"]["synthetic_cxr"]
                    image_path = require_inside(
                        cxr["path"], MAIN_PROTECTED_ROOT, must_exist=True
                    )
                    if sha256_file(image_path) != cxr["sha256"]:
                        raise ValueError("V1.1 report input CXR changed before inference")
                    result = runtime.generate(image_path)
                    text = str(result.canonical_text).strip()
                    if not text:
                        raise ValueError("V1.1 report runtime returned empty text")

                    parent_cxr_id = request["parent_cxr_candidate_id"]
                    candidate_id = f"report_{parent_cxr_id}_{model_id}"
                    candidate_dir = candidates_root / candidate_id
                    private_directory(candidate_dir)
                    write_private_json(candidate_dir / "request.json", request)
                    report_path = write_private_text(
                        candidate_dir / "synthetic_report.txt", text + "\n"
                    )
                    final_report_path = (
                        target
                        / "candidates"
                        / candidate_id
                        / "synthetic_report.txt"
                    )
                    candidate = {
                        "schema_version": REPORT_CANDIDATE_SCHEMA_V11,
                        "candidate_id": candidate_id,
                        "case_id": request["case_id"],
                        "modality": "report",
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "frozen_model": True,
                        "model_input_signature": "single_current_synthetic_cxr",
                        "structured_ehr_content_supplied_to_model": False,
                        "source_report_or_real_target_supplied": False,
                        "parent_cxr_candidate_id": parent_cxr_id,
                        "input_report_request_sha256": canonical_json_sha256(request),
                        "input_cxr": dict(cxr),
                        "input_cxr_candidate_sha256": request["inputs"][
                            "cxr_candidate_sha256"
                        ],
                        "ehr_sha256_retained_for_lineage": request["inputs"][
                            "ehr_sha256_retained_for_lineage"
                        ],
                        "ehr_facts_sha256_retained_for_lineage": request["inputs"][
                            "ehr_facts_sha256_retained_for_lineage"
                        ],
                        "clinical_intent_sha256_retained_for_lineage": request[
                            "inputs"
                        ]["clinical_intent_sha256_retained_for_lineage"],
                        "artifact": {
                            "path": str(final_report_path),
                            "sha256": sha256_file(report_path),
                            "mime_type": "text/plain",
                        },
                        "cost": {
                            "model_calls": 1,
                            "runtime_seconds": round(
                                float(result.elapsed_seconds), 3
                            ),
                        },
                    }
                    candidate_path = write_private_json(
                        candidate_dir / "candidate.json", candidate
                    )
                    completed.append(
                        {
                            "candidate_id": candidate_id,
                            "case_id": request["case_id"],
                            "parent_cxr_candidate_id": parent_cxr_id,
                            "path": f"candidates/{candidate_id}/candidate.json",
                            "sha256": sha256_file(candidate_path),
                        }
                    )
        os.chmod(log_path, 0o660)
        manifest_path = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": REPORT_CANDIDATE_RUN_SCHEMA_V11,
                "run_id": output_run_id,
                "model_id": model_id,
                "model_revision": model_revision,
                "frozen_model": True,
                "model_input_signature": "single_current_synthetic_cxr",
                "structured_ehr_content_supplied_to_model": False,
                "source_report_or_real_target_supplied": False,
                "source_request_run": str(source),
                "source_request_run_manifest_sha256": sha256_file(
                    source / "manifest.json"
                ),
                "candidate_count": len(completed),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_vram_gib": round(float(runtime.peak_vram_gib), 3),
                "model_audit": dict(model_audit),
                "candidates": completed,
            },
        )
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
        for row in completed:
            candidate = read_json(target / row["path"])
            validate_report_candidate(candidate)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        # Never delete a committed target. This call owns only its UUID temp.
        raise
    return {
        "status": "completed",
        "model_id": model_id,
        "candidate_count": len(completed),
        "run_directory": str(target),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


__all__ = [
    "REPORT_CANDIDATE_RUN_SCHEMA_V11",
    "REPORT_CANDIDATE_SCHEMA_V11",
    "run_report_request_run",
    "validate_report_candidate",
]

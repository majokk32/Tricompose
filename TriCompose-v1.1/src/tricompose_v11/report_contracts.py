"""Hash-bound CXR-to-report contracts for TriCompose V1.1."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .cxr_contracts import (
    MAIN_PROTECTED_ROOT,
    RUN_ID_PATTERN,
    canonical_json_sha256,
    private_directory,
    read_json,
    require_inside,
    sha256_file,
    write_private_json,
)
from .cxr_execution import CXR_CANDIDATE_RUN_SCHEMA_V11, CXR_CANDIDATE_SCHEMA_V11


ACTIVE_REPORT_MODELS_V11 = (
    "maira2",
    "cxrmate_single",
    "llavarad",
    "chexagent2",
)
REPORT_REQUEST_SCHEMA_V11 = "tricompose-report-request-v1.1"
REPORT_REQUEST_RUN_SCHEMA_V11 = "tricompose-report-request-run-v1.1"


def validate_cxr_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("schema_version") != CXR_CANDIDATE_SCHEMA_V11:
        raise ValueError("report input is not a V1.1 CXR candidate")
    if candidate.get("frozen_model") is not True:
        raise ValueError("report input CXR generator was not frozen")
    if candidate.get("adapter_added_prefix") is not False:
        raise ValueError("report input CXR used an untracked adapter prefix")
    artifact = candidate.get("artifact")
    if not isinstance(artifact, dict):
        raise TypeError("CXR candidate artifact is missing")
    path = require_inside(
        str(artifact.get("path", "")), MAIN_PROTECTED_ROOT, must_exist=True
    )
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise ValueError("CXR candidate image hash mismatch")
    if artifact.get("mime_type") != "image/png":
        raise ValueError("CXR candidate is not a PNG")


def load_cxr_candidates(cxr_runs: Iterable[str | Path]) -> list[dict[str, Any]]:
    sources = tuple(
        require_inside(path, MAIN_PROTECTED_ROOT, must_exist=True)
        for path in cxr_runs
    )
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("CXR runs must be non-empty and unique")
    candidates: list[dict[str, Any]] = []
    for source in sources:
        manifest = read_json(source / "manifest.json")
        if manifest.get("schema_version") != CXR_CANDIDATE_RUN_SCHEMA_V11:
            raise ValueError("report source is not a V1.1 CXR candidate run")
        for row in manifest.get("candidates", []):
            if not isinstance(row, dict):
                raise TypeError("CXR manifest entry is invalid")
            path = require_inside(
                source / str(row.get("path", "")), source, must_exist=True
            )
            if sha256_file(path) != row.get("sha256"):
                raise ValueError("CXR candidate JSON hash mismatch")
            candidate = read_json(path)
            validate_cxr_candidate(candidate)
            candidates.append(candidate)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if not candidates or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("CXR candidate bank is empty or contains duplicate IDs")
    return candidates


def build_report_request(
    *, cxr_candidate: Mapping[str, Any], report_model_id: str
) -> dict[str, Any]:
    validate_cxr_candidate(cxr_candidate)
    if report_model_id not in ACTIVE_REPORT_MODELS_V11:
        raise ValueError("inactive V1.1 report model")
    cxr_id = str(cxr_candidate["candidate_id"])
    request = {
        "schema_version": REPORT_REQUEST_SCHEMA_V11,
        "request_id": f"reportreq_{cxr_id}_{report_model_id}",
        "case_id": cxr_candidate["case_id"],
        "model_id": report_model_id,
        "frozen_model_required": True,
        "model_input_signature": "single_current_synthetic_cxr",
        "parent_cxr_candidate_id": cxr_id,
        "inputs": {
            "synthetic_cxr": dict(cxr_candidate["artifact"]),
            "cxr_candidate_sha256": canonical_json_sha256(cxr_candidate),
            "source_cxr_model_id": cxr_candidate["model_id"],
            "source_cxr_seed": cxr_candidate["seed"],
            "ehr_sha256_retained_for_lineage": cxr_candidate["ehr_sha256"],
            "ehr_facts_sha256_retained_for_lineage": cxr_candidate[
                "ehr_facts_sha256"
            ],
            "clinical_intent_sha256_retained_for_lineage": cxr_candidate[
                "clinical_intent_sha256"
            ],
            "structured_ehr_content_supplied_to_model": False,
            "source_report_or_real_target_supplied": False,
        },
    }
    validate_report_request(request)
    return request


def validate_report_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != REPORT_REQUEST_SCHEMA_V11:
        raise ValueError("unsupported V1.1 report request schema")
    model_id = request.get("model_id")
    if model_id not in ACTIVE_REPORT_MODELS_V11:
        raise ValueError("inactive V1.1 report request model")
    if request.get("frozen_model_required") is not True:
        raise ValueError("report request does not require a frozen model")
    if request.get("model_input_signature") != "single_current_synthetic_cxr":
        raise ValueError("report request has the wrong input signature")
    cxr_id = request.get("parent_cxr_candidate_id")
    if request.get("request_id") != f"reportreq_{cxr_id}_{model_id}":
        raise ValueError("report request ID does not match lineage")
    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("report request inputs are missing")
    artifact = inputs.get("synthetic_cxr")
    if not isinstance(artifact, dict):
        raise TypeError("report request CXR artifact is missing")
    path = require_inside(
        str(artifact.get("path", "")), MAIN_PROTECTED_ROOT, must_exist=True
    )
    if sha256_file(path) != artifact.get("sha256"):
        raise ValueError("report request CXR hash mismatch")
    if inputs.get("structured_ehr_content_supplied_to_model") is not False:
        raise ValueError("single-image report request unexpectedly supplies EHR")
    if inputs.get("source_report_or_real_target_supplied") is not False:
        raise ValueError("report request leaks a source target")


def prepare_report_request_run(
    *,
    cxr_runs: Iterable[str | Path],
    output_root: str | Path,
    run_id: str,
    model_ids: Iterable[str] = ACTIVE_REPORT_MODELS_V11,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    models = tuple(model_ids)
    if not models or len(models) != len(set(models)):
        raise ValueError("report model list must be non-empty and unique")
    if any(model not in ACTIVE_REPORT_MODELS_V11 for model in models):
        raise ValueError("report request run contains an inactive model")
    source_values = tuple(cxr_runs)
    candidates = load_cxr_candidates(source_values)
    requests = [
        build_report_request(cxr_candidate=candidate, report_model_id=model)
        for candidate in candidates
        for model in models
    ]
    output = require_inside(output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    target = output / run_id
    if target.exists():
        raise FileExistsError("V1.1 report request run already exists")
    temporary = output / f".{run_id}.{uuid.uuid4().hex}.tmp"
    private_directory(temporary)
    records: list[dict[str, Any]] = []
    try:
        requests_dir = temporary / "requests"
        private_directory(requests_dir)
        for request in requests:
            path = write_private_json(
                requests_dir / f"{request['request_id']}.json", request
            )
            records.append(
                {
                    "request_id": request["request_id"],
                    "case_id": request["case_id"],
                    "model_id": request["model_id"],
                    "parent_cxr_candidate_id": request[
                        "parent_cxr_candidate_id"
                    ],
                    "path": f"requests/{path.name}",
                    "sha256": sha256_file(path),
                }
            )
        source_runs = tuple(
            require_inside(path, MAIN_PROTECTED_ROOT, must_exist=True)
            for path in source_values
        )
        manifest_path = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": REPORT_REQUEST_RUN_SCHEMA_V11,
                "run_id": run_id,
                "source_cxr_runs": [
                    {
                        "path": str(source),
                        "manifest_sha256": sha256_file(
                            source / "manifest.json"
                        ),
                    }
                    for source in source_runs
                ],
                "cxr_candidate_count": len(candidates),
                "model_ids": list(models),
                "request_count": len(records),
                "requests": records,
                "gpu_inference_used": False,
                "raw_source_target_used": False,
            },
        )
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "prepared",
        "run_directory": str(target),
        "cxr_candidate_count": len(candidates),
        "request_count": len(records),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


def load_model_requests(
    request_run: str | Path, model_id: str
) -> list[dict[str, Any]]:
    source = require_inside(request_run, MAIN_PROTECTED_ROOT, must_exist=True)
    manifest = read_json(source / "manifest.json")
    if manifest.get("schema_version") != REPORT_REQUEST_RUN_SCHEMA_V11:
        raise ValueError("unsupported V1.1 report request-run schema")
    requests: list[dict[str, Any]] = []
    for row in manifest.get("requests", []):
        if not isinstance(row, dict) or row.get("model_id") != model_id:
            continue
        path = require_inside(source / str(row.get("path", "")), source, must_exist=True)
        if sha256_file(path) != row.get("sha256"):
            raise ValueError("report request JSON hash mismatch")
        request = read_json(path)
        validate_report_request(request)
        requests.append(request)
    if not requests:
        raise ValueError("report request run has no entries for this model")
    return requests


__all__ = [
    "ACTIVE_REPORT_MODELS_V11",
    "REPORT_REQUEST_RUN_SCHEMA_V11",
    "REPORT_REQUEST_SCHEMA_V11",
    "build_report_request",
    "load_model_requests",
    "prepare_report_request_run",
    "validate_report_request",
]

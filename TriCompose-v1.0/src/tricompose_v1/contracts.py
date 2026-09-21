"""Unified, hash-bound I/O contracts for the three TriCompose modalities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)

from .prompts import ACTIVE_PROMPT_MODELS


EHR_CANDIDATE_SCHEMA = "tricompose-ehr-candidate-v1"
CXR_REQUEST_SCHEMA = "tricompose-cxr-request-v1"
CXR_CANDIDATE_SCHEMA = "tricompose-cxr-candidate-v1"
REPORT_REQUEST_SCHEMA = "tricompose-report-request-v1"
REPORT_CANDIDATE_SCHEMA = "tricompose-report-candidate-v1"
TRIPLE_SCHEMA = "tricompose-synthetic-triple-v1"
STAGING_SCHEMA = "tricompose.staging.run.v1"

ACTIVE_REPORT_MODELS = (
    "maira2",
    "cxrmate_single",
    "llavarad",
    "chexagent2",
)

_CASE_ID = re.compile(r"case_[0-9]{3,6}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    source = require_private_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_case_id(case_id: Any) -> str:
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise ValueError("invalid opaque case ID")
    return case_id


def _valid_seed(seed: Any) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _valid_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"invalid SHA256 for {label}")
    return value


def _require_artifact(ref: Any, label: str) -> Path:
    if not isinstance(ref, dict):
        raise TypeError(f"{label} artifact reference must be an object")
    path = require_private_file(str(ref.get("path", "")))
    expected = _valid_hash(ref.get("sha256"), label)
    if sha256_file(path) != expected:
        raise ValueError(f"{label} artifact hash mismatch")
    return path


def build_ehr_candidate(
    *, staging_run: str | Path, case_id: str
) -> dict[str, Any]:
    """Expose one staged canonical EHR as the first graph node."""

    case_id = _valid_case_id(case_id)
    run_root = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)
    run_manifest_path = require_private_file(run_root / "run_manifest.json")
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("schema_version") != STAGING_SCHEMA:
        raise ValueError("EHR candidate source is not a V1 staging run")
    case_root = require_inside(run_root / "cases" / case_id, run_root, must_exist=True)
    ehr_path = require_private_file(case_root / "synthetic_ehr.json")
    facts_path = require_private_file(case_root / "ehr_facts.json")
    ehr = _read_json(ehr_path)
    facts = _read_json(facts_path)
    if ehr.get("case_id") != case_id or facts.get("case_id") != case_id:
        raise ValueError("staged EHR/facts case lineage is inconsistent")
    source = ehr.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("model_id"), str):
        raise ValueError("canonical EHR generator provenance is missing")
    candidate = {
        "schema_version": EHR_CANDIDATE_SCHEMA,
        "candidate_id": f"ehr_{case_id}",
        "case_id": case_id,
        "modality": "ehr",
        "model_id": source["model_id"],
        "parent_ids": [],
        "artifacts": {
            "synthetic_ehr": {
                "path": str(ehr_path),
                "sha256": sha256_file(ehr_path),
            },
            "ehr_facts": {
                "path": str(facts_path),
                "sha256": sha256_file(facts_path),
            },
        },
        "staging_run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256_file(run_manifest_path),
        },
    }
    validate_ehr_candidate(candidate)
    return candidate


def validate_ehr_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("schema_version") != EHR_CANDIDATE_SCHEMA:
        raise ValueError("unsupported EHR candidate schema")
    case_id = _valid_case_id(candidate.get("case_id"))
    if candidate.get("candidate_id") != f"ehr_{case_id}":
        raise ValueError("EHR candidate ID does not match its case")
    if candidate.get("modality") != "ehr" or candidate.get("parent_ids") != []:
        raise ValueError("EHR candidate modality/parentage is invalid")
    if not isinstance(candidate.get("model_id"), str) or not candidate["model_id"]:
        raise ValueError("EHR candidate model provenance is missing")
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("EHR candidate artifacts are missing")
    _require_artifact(artifacts.get("synthetic_ehr"), "synthetic_ehr")
    _require_artifact(artifacts.get("ehr_facts"), "ehr_facts")
    _require_artifact(candidate.get("staging_run_manifest"), "staging_run_manifest")


def build_cxr_request(
    *,
    staging_run: str | Path,
    case_id: str,
    model_id: str,
    seed: int,
) -> dict[str, Any]:
    """Bind one exact staged prompt to one frozen CXR-model call."""

    case_id = _valid_case_id(case_id)
    seed = _valid_seed(seed)
    if model_id not in ACTIVE_PROMPT_MODELS:
        raise ValueError("CXR request uses an inactive prompt model")
    run_root = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)
    run_manifest_path = require_private_file(run_root / "run_manifest.json")
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("schema_version") != STAGING_SCHEMA:
        raise ValueError("CXR request source is not a V1 staging run")
    case_root = require_inside(run_root / "cases" / case_id, run_root, must_exist=True)
    ehr_path = require_private_file(case_root / "synthetic_ehr.json")
    facts_path = require_private_file(case_root / "ehr_facts.json")
    prompt_manifest_path = require_private_file(
        case_root / "cxr_prompts" / "prompt_manifest.json"
    )
    ehr = _read_json(ehr_path)
    facts = _read_json(facts_path)
    prompt_manifest = _read_json(prompt_manifest_path)
    if ehr.get("case_id") != case_id or facts.get("case_id") != case_id:
        raise ValueError("staged EHR/facts case lineage is inconsistent")
    if prompt_manifest.get("case_id") != case_id:
        raise ValueError("prompt manifest case lineage is inconsistent")
    if prompt_manifest.get("ehr_facts_sha256") != sha256_file(facts_path):
        raise ValueError("prompt manifest is not bound to the staged EHR facts")
    models = prompt_manifest.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model_id), dict):
        raise ValueError("staging run lacks the requested model prompt")
    prompt_meta = models[model_id]
    expected_relative = f"cxr_prompts/{model_id}.txt"
    if prompt_meta.get("path") != expected_relative:
        raise ValueError("model prompt path is not canonical")
    if prompt_meta.get("is_final_model_input") is not True:
        raise ValueError("model prompt is not final tokenizer input")
    prompt_path = require_inside(case_root / expected_relative, case_root, must_exist=True)
    prompt_path = require_private_file(prompt_path)
    prompt_sha256 = sha256_file(prompt_path)
    if prompt_meta.get("prompt_sha256") != prompt_sha256:
        raise ValueError("model prompt hash mismatch")
    ehr_candidate = build_ehr_candidate(staging_run=run_root, case_id=case_id)
    parent_ehr_id = str(ehr_candidate["candidate_id"])
    request = {
        "schema_version": CXR_REQUEST_SCHEMA,
        "request_id": f"cxrreq_{case_id}_{model_id}_s{seed:06d}",
        "case_id": case_id,
        "model_id": model_id,
        "frozen_model_required": True,
        "seed": seed,
        "parent_ehr_candidate_id": parent_ehr_id,
        "parent_ehr_candidate_sha256": _canonical_json_sha256(ehr_candidate),
        "inputs": {
            "staging_run_manifest": {
                "path": str(run_manifest_path),
                "sha256": sha256_file(run_manifest_path),
            },
            "synthetic_ehr": {
                "path": str(ehr_path),
                "sha256": sha256_file(ehr_path),
            },
            "ehr_facts": {
                "path": str(facts_path),
                "sha256": sha256_file(facts_path),
            },
            "final_prompt": {
                "path": str(prompt_path),
                "sha256": prompt_sha256,
                "renderer_version": prompt_meta["renderer_version"],
                "clinical_prompt_version": prompt_meta[
                    "clinical_prompt_version"
                ],
                "legacy_experiment_reproduction": prompt_meta[
                    "legacy_experiment_reproduction"
                ],
                "is_final_model_input": True,
                "adapter_must_not_add_prefix": True,
            },
        },
    }
    validate_cxr_request(request)
    return request


def validate_cxr_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != CXR_REQUEST_SCHEMA:
        raise ValueError("unsupported CXR request schema")
    case_id = _valid_case_id(request.get("case_id"))
    model_id = request.get("model_id")
    seed = _valid_seed(request.get("seed"))
    if model_id not in ACTIVE_PROMPT_MODELS:
        raise ValueError("CXR request model is inactive")
    if request.get("request_id") != f"cxrreq_{case_id}_{model_id}_s{seed:06d}":
        raise ValueError("CXR request ID does not match its lineage")
    if request.get("parent_ehr_candidate_id") != f"ehr_{case_id}":
        raise ValueError("CXR request has the wrong EHR parent")
    _valid_hash(request.get("parent_ehr_candidate_sha256"), "EHR candidate")
    if request.get("frozen_model_required") is not True:
        raise ValueError("CXR request must require a frozen model")
    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("CXR request inputs are missing")
    for label in ("staging_run_manifest", "synthetic_ehr", "ehr_facts"):
        _require_artifact(inputs.get(label), label)
    prompt = inputs.get("final_prompt")
    _require_artifact(prompt, "final_prompt")
    if prompt.get("is_final_model_input") is not True:
        raise ValueError("CXR prompt is not final model input")
    if prompt.get("adapter_must_not_add_prefix") is not True:
        raise ValueError("CXR adapter prefix contract is missing")
    if prompt.get("legacy_experiment_reproduction") is not True:
        raise ValueError("CXR request does not reproduce the validated bridge")
    if not isinstance(prompt.get("clinical_prompt_version"), str):
        raise ValueError("CXR request lacks the shared clinical prompt version")


def build_cxr_candidate(
    *,
    request: Mapping[str, Any],
    image_path: str | Path,
    image_dimensions: tuple[int, int],
    model_revision: str,
    prompt_token_count: int,
    runtime_seconds: float,
    peak_vram_gib: float,
) -> dict[str, Any]:
    """Build the model-independent CXR output record after local inference."""

    validate_cxr_request(request)
    image = require_private_file(image_path)
    width, height = image_dimensions
    if any(not isinstance(value, int) or value < 1 for value in (width, height)):
        raise ValueError("CXR output dimensions are invalid")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("CXR model revision is missing")
    if not isinstance(prompt_token_count, int) or prompt_token_count < 1:
        raise ValueError("CXR prompt token count is invalid")
    if runtime_seconds < 0 or peak_vram_gib < 0:
        raise ValueError("CXR runtime measurements are invalid")
    model_id = str(request["model_id"])
    case_id = str(request["case_id"])
    seed = int(request["seed"])
    candidate = {
        "schema_version": CXR_CANDIDATE_SCHEMA,
        "candidate_id": f"cxr_{case_id}_{model_id}_s{seed:06d}",
        "case_id": case_id,
        "modality": "cxr",
        "model_id": model_id,
        "model_revision": model_revision,
        "frozen_model": True,
        "seed": seed,
        "parent_ids": [request["parent_ehr_candidate_id"]],
        "input_request_sha256": _canonical_json_sha256(request),
        "ehr_sha256": request["inputs"]["synthetic_ehr"]["sha256"],
        "ehr_facts_sha256": request["inputs"]["ehr_facts"]["sha256"],
        "ehr_candidate_sha256": request["parent_ehr_candidate_sha256"],
        "prompt_sha256": request["inputs"]["final_prompt"]["sha256"],
        "artifact": {
            "path": str(image),
            "sha256": sha256_file(image),
            "mime_type": "image/png",
            "dimensions": [width, height],
        },
        "cost": {
            "model_calls": 1,
            "runtime_seconds": round(float(runtime_seconds), 3),
            "peak_vram_gib": round(float(peak_vram_gib), 3),
            "prompt_token_count": prompt_token_count,
        },
    }
    validate_cxr_candidate(candidate, request=request)
    return candidate


def validate_cxr_candidate(
    candidate: Mapping[str, Any], *, request: Mapping[str, Any] | None = None
) -> None:
    if candidate.get("schema_version") != CXR_CANDIDATE_SCHEMA:
        raise ValueError("unsupported CXR candidate schema")
    case_id = _valid_case_id(candidate.get("case_id"))
    model_id = candidate.get("model_id")
    seed = _valid_seed(candidate.get("seed"))
    if model_id not in ACTIVE_PROMPT_MODELS:
        raise ValueError("CXR candidate model is inactive")
    if candidate.get("candidate_id") != f"cxr_{case_id}_{model_id}_s{seed:06d}":
        raise ValueError("CXR candidate ID does not match lineage")
    if candidate.get("modality") != "cxr" or candidate.get("frozen_model") is not True:
        raise ValueError("CXR candidate modality/frozen status is invalid")
    if candidate.get("parent_ids") != [f"ehr_{case_id}"]:
        raise ValueError("CXR candidate has the wrong EHR parent")
    for label in (
        "input_request_sha256",
        "ehr_sha256",
        "ehr_facts_sha256",
        "ehr_candidate_sha256",
        "prompt_sha256",
    ):
        _valid_hash(candidate.get(label), label)
    artifact = candidate.get("artifact")
    _require_artifact(artifact, "synthetic_cxr")
    if artifact.get("mime_type") != "image/png":
        raise ValueError("CXR candidate must be a PNG")
    dimensions = artifact.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 2 or any(
        not isinstance(value, int) or value < 1 for value in dimensions
    ):
        raise ValueError("CXR candidate dimensions are invalid")
    if request is not None:
        validate_cxr_request(request)
        expected = {
            "case_id": request["case_id"],
            "model_id": request["model_id"],
            "seed": request["seed"],
            "input_request_sha256": _canonical_json_sha256(request),
            "ehr_sha256": request["inputs"]["synthetic_ehr"]["sha256"],
            "ehr_facts_sha256": request["inputs"]["ehr_facts"]["sha256"],
            "ehr_candidate_sha256": request["parent_ehr_candidate_sha256"],
            "prompt_sha256": request["inputs"]["final_prompt"]["sha256"],
        }
        if any(candidate.get(key) != value for key, value in expected.items()):
            raise ValueError("CXR candidate does not match its input request")


def build_report_request(
    *,
    cxr_candidate: Mapping[str, Any],
    report_model_id: str,
) -> dict[str, Any]:
    """Build a CXR-only report request while retaining full EHR lineage."""

    validate_cxr_candidate(cxr_candidate)
    if report_model_id not in ACTIVE_REPORT_MODELS:
        raise ValueError("report request uses an inactive model")
    case_id = str(cxr_candidate["case_id"])
    cxr_id = str(cxr_candidate["candidate_id"])
    ehr_id = str(cxr_candidate["parent_ids"][0])
    request = {
        "schema_version": REPORT_REQUEST_SCHEMA,
        "request_id": f"reportreq_{cxr_id}_{report_model_id}",
        "case_id": case_id,
        "model_id": report_model_id,
        "frozen_model_required": True,
        "model_input_signature": "single_current_synthetic_cxr",
        "parent_ehr_candidate_id": ehr_id,
        "parent_cxr_candidate_id": cxr_id,
        "inputs": {
            "synthetic_cxr": dict(cxr_candidate["artifact"]),
            "cxr_candidate_sha256": _canonical_json_sha256(cxr_candidate),
            "ehr_content_supplied_to_model": False,
        },
    }
    validate_report_request(request, cxr_candidate=cxr_candidate)
    return request


def validate_report_request(
    request: Mapping[str, Any], *, cxr_candidate: Mapping[str, Any] | None = None
) -> None:
    if request.get("schema_version") != REPORT_REQUEST_SCHEMA:
        raise ValueError("unsupported report request schema")
    case_id = _valid_case_id(request.get("case_id"))
    model_id = request.get("model_id")
    if model_id not in ACTIVE_REPORT_MODELS:
        raise ValueError("report request model is inactive")
    if request.get("frozen_model_required") is not True:
        raise ValueError("report request must require a frozen model")
    if request.get("model_input_signature") != "single_current_synthetic_cxr":
        raise ValueError("report request has the wrong model input signature")
    cxr_id = request.get("parent_cxr_candidate_id")
    ehr_id = request.get("parent_ehr_candidate_id")
    if not isinstance(cxr_id, str) or not isinstance(ehr_id, str):
        raise ValueError("report request parent lineage is missing")
    if ehr_id != f"ehr_{case_id}":
        raise ValueError("report request has the wrong EHR lineage")
    if request.get("request_id") != f"reportreq_{cxr_id}_{model_id}":
        raise ValueError("report request ID does not match lineage")
    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("report request inputs are missing")
    _require_artifact(inputs.get("synthetic_cxr"), "synthetic_cxr")
    _valid_hash(inputs.get("cxr_candidate_sha256"), "cxr_candidate")
    if inputs.get("ehr_content_supplied_to_model") is not False:
        raise ValueError("CXR-only report request cannot receive EHR content")
    if cxr_candidate is not None:
        validate_cxr_candidate(cxr_candidate)
        if (
            cxr_candidate["candidate_id"] != cxr_id
            or cxr_candidate["case_id"] != case_id
            or cxr_candidate["parent_ids"] != [ehr_id]
            or _canonical_json_sha256(cxr_candidate)
            != inputs["cxr_candidate_sha256"]
        ):
            raise ValueError("report request does not match its CXR candidate")


def build_report_candidate(
    *,
    request: Mapping[str, Any],
    report_path: str | Path,
    model_revision: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    validate_report_request(request)
    report = require_private_file(report_path)
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("report model revision is missing")
    if runtime_seconds < 0:
        raise ValueError("report runtime is invalid")
    cxr_id = str(request["parent_cxr_candidate_id"])
    model_id = str(request["model_id"])
    candidate = {
        "schema_version": REPORT_CANDIDATE_SCHEMA,
        "candidate_id": f"report_{cxr_id}_{model_id}",
        "case_id": request["case_id"],
        "modality": "report",
        "model_id": model_id,
        "model_revision": model_revision,
        "frozen_model": True,
        "parent_ids": [
            request["parent_ehr_candidate_id"],
            request["parent_cxr_candidate_id"],
        ],
        "input_request_sha256": _canonical_json_sha256(request),
        "cxr_candidate_sha256": request["inputs"]["cxr_candidate_sha256"],
        "artifact": {
            "path": str(report),
            "sha256": sha256_file(report),
            "mime_type": "text/plain",
        },
        "cost": {"model_calls": 1, "runtime_seconds": round(float(runtime_seconds), 3)},
    }
    validate_report_candidate(candidate, request=request)
    return candidate


def validate_report_candidate(
    candidate: Mapping[str, Any], *, request: Mapping[str, Any] | None = None
) -> None:
    if candidate.get("schema_version") != REPORT_CANDIDATE_SCHEMA:
        raise ValueError("unsupported report candidate schema")
    case_id = _valid_case_id(candidate.get("case_id"))
    model_id = candidate.get("model_id")
    if model_id not in ACTIVE_REPORT_MODELS:
        raise ValueError("report candidate model is inactive")
    parents = candidate.get("parent_ids")
    if not isinstance(parents, list) or len(parents) != 2:
        raise ValueError("report candidate must retain EHR and CXR parents")
    if parents[0] != f"ehr_{case_id}":
        raise ValueError("report candidate has the wrong EHR parent")
    expected_id = f"report_{parents[1]}_{model_id}"
    if candidate.get("candidate_id") != expected_id:
        raise ValueError("report candidate ID does not match lineage")
    if candidate.get("modality") != "report" or candidate.get("frozen_model") is not True:
        raise ValueError("report candidate modality/frozen status is invalid")
    _valid_hash(candidate.get("input_request_sha256"), "report request")
    _valid_hash(candidate.get("cxr_candidate_sha256"), "CXR candidate")
    artifact = candidate.get("artifact")
    _require_artifact(artifact, "synthetic_report")
    if artifact.get("mime_type") != "text/plain":
        raise ValueError("report candidate must be plain text")
    if request is not None:
        validate_report_request(request)
        if (
            candidate["case_id"] != request["case_id"]
            or candidate["model_id"] != request["model_id"]
            or candidate["parent_ids"]
            != [request["parent_ehr_candidate_id"], request["parent_cxr_candidate_id"]]
            or candidate["input_request_sha256"] != _canonical_json_sha256(request)
            or candidate["cxr_candidate_sha256"]
            != request["inputs"]["cxr_candidate_sha256"]
        ):
            raise ValueError("report candidate does not match its input request")


def _private_dir(path: Path) -> None:
    old_umask = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
    finally:
        os.umask(old_umask)
    enforce_private_directory_mode(path)


def _private_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError("triple output artifact already exists")
    shutil.copyfile(source, target)
    enforce_private_file_mode(target)


def export_selected_triple(
    *,
    staging_run: str | Path,
    cxr_candidate: Mapping[str, Any],
    report_candidate: Mapping[str, Any],
    selection: Mapping[str, Any],
    output_root: str | Path,
    export_id: str,
) -> dict[str, Any]:
    """Atomically publish exactly one complete synthetic EHR-CXR-report triple."""

    validate_cxr_candidate(cxr_candidate)
    validate_report_candidate(report_candidate)
    if not _RUN_ID.fullmatch(export_id):
        raise ValueError("export ID must be opaque and filesystem-safe")
    if selection.get("action") != "stop_and_select":
        raise ValueError("a triple cannot be exported before stop_and_select")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection has no selected triple")
    selection_provenance = selection.get("selection_provenance")
    if selection_provenance is not None:
        if not isinstance(selection_provenance, dict):
            raise TypeError("selection provenance must be an object")
        if not isinstance(selection_provenance.get("method"), str):
            raise ValueError("selection provenance method is missing")
        if not isinstance(selection_provenance.get("scoring_used"), bool):
            raise TypeError("selection provenance scoring flag must be boolean")
    case_id = str(cxr_candidate["case_id"])
    ehr_id = f"ehr_{case_id}"
    if (
        report_candidate["case_id"] != case_id
        or report_candidate["parent_ids"]
        != [ehr_id, cxr_candidate["candidate_id"]]
        or report_candidate["cxr_candidate_sha256"]
        != _canonical_json_sha256(cxr_candidate)
        or selected.get("ehr_candidate_id") != ehr_id
        or selected.get("cxr_candidate_id") != cxr_candidate["candidate_id"]
        or selected.get("report_candidate_id") != report_candidate["candidate_id"]
    ):
        raise ValueError("selected triple has inconsistent cross-modal lineage")
    run_root = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)
    ehr_source = require_private_file(run_root / "cases" / case_id / "synthetic_ehr.json")
    facts_source = require_private_file(run_root / "cases" / case_id / "ehr_facts.json")
    if sha256_file(ehr_source) != cxr_candidate["ehr_sha256"]:
        raise ValueError("selected CXR is not derived from the exported EHR")
    if sha256_file(facts_source) != cxr_candidate["ehr_facts_sha256"]:
        raise ValueError("selected CXR is not derived from the exported EHR facts")
    cxr_source = _require_artifact(cxr_candidate["artifact"], "synthetic_cxr")
    report_source = _require_artifact(
        report_candidate["artifact"], "synthetic_report"
    )

    root = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    if not root.exists():
        old_umask = os.umask(0o077)
        try:
            root.mkdir(parents=True, mode=0o700)
        finally:
            os.umask(old_umask)
        enforce_private_directory_mode(root)
    target = require_inside(root / export_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError("triple export already exists")
    temp = root / f".{export_id}.{uuid.uuid4().hex}.tmp"
    _private_dir(temp)
    try:
        provenance = temp / "provenance"
        _private_dir(provenance)
        ehr_target = temp / "synthetic_ehr.json"
        cxr_target = temp / "synthetic_cxr.png"
        report_target = temp / "synthetic_report.txt"
        facts_target = provenance / "ehr_facts.json"
        _private_copy(ehr_source, ehr_target)
        _private_copy(cxr_source, cxr_target)
        _private_copy(report_source, report_target)
        _private_copy(facts_source, facts_target)
        manifest = {
            "schema_version": TRIPLE_SCHEMA,
            "export_id": export_id,
            "case_id": case_id,
            "triple_id": (
                f"{ehr_id}|{cxr_candidate['candidate_id']}|"
                f"{report_candidate['candidate_id']}"
            ),
            "selection_action": "stop_and_select",
            "selection_provenance": selection_provenance,
            "lineage": {
                "ehr_candidate_id": ehr_id,
                "cxr_candidate_id": cxr_candidate["candidate_id"],
                "report_candidate_id": report_candidate["candidate_id"],
            },
            "artifacts": {
                "synthetic_ehr": {
                    "path": "synthetic_ehr.json",
                    "sha256": sha256_file(ehr_target),
                },
                "synthetic_cxr": {
                    "path": "synthetic_cxr.png",
                    "sha256": sha256_file(cxr_target),
                },
                "synthetic_report": {
                    "path": "synthetic_report.txt",
                    "sha256": sha256_file(report_target),
                },
                "ehr_facts": {
                    "path": "provenance/ehr_facts.json",
                    "sha256": sha256_file(facts_target),
                },
            },
            "candidate_record_sha256": {
                "cxr": _canonical_json_sha256(cxr_candidate),
                "report": _canonical_json_sha256(report_candidate),
            },
            "mutation_policy": "immutable_no_overwrite",
            "complete_three_modality_export": True,
        }
        manifest_path = write_private_json(temp / "triple_manifest.json", manifest)
        os.rename(temp, target)
        enforce_private_directory_mode(target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return {
        "status": "exported",
        "export_directory": str(target),
        "case_count": 1,
        "complete_three_modality_export": True,
        "triple_manifest_sha256": sha256_file(target / manifest_path.name),
    }


__all__ = [
    "ACTIVE_REPORT_MODELS",
    "EHR_CANDIDATE_SCHEMA",
    "CXR_CANDIDATE_SCHEMA",
    "CXR_REQUEST_SCHEMA",
    "REPORT_CANDIDATE_SCHEMA",
    "REPORT_REQUEST_SCHEMA",
    "TRIPLE_SCHEMA",
    "build_cxr_candidate",
    "build_cxr_request",
    "build_ehr_candidate",
    "build_report_candidate",
    "build_report_request",
    "export_selected_triple",
    "validate_cxr_candidate",
    "validate_cxr_request",
    "validate_ehr_candidate",
    "validate_report_candidate",
    "validate_report_request",
]

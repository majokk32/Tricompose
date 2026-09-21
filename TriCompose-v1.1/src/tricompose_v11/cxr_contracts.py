"""Hash-bound, CPU-only CXR request contracts for TriCompose V1.1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .prompts import ACTIVE_PROMPT_MODELS_V11
from .staging import STAGING_SCHEMA_V11


WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
MAIN_PROTECTED_ROOT = WORKSPACE / "artifacts" / "protected"
CXR_REQUEST_SCHEMA_V11 = "tricompose-cxr-request-v1.1"
CXR_REQUEST_RUN_SCHEMA_V11 = "tricompose-cxr-request-run-v1.1"
CASE_ID_PATTERN = re.compile(r"case_[0-9]{3,6}\Z")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_inside(path: str | Path, root: str | Path, *, must_exist: bool) -> Path:
    resolved_root = Path(root).resolve(strict=True)
    resolved = Path(path).resolve(strict=must_exist)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("path is outside the protected V1.1 boundary")
    return resolved


def private_directory(path: Path, *, exist_ok: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=exist_ok, mode=0o2770)
    os.chmod(path, 0o2770)


def write_private_text(path: Path, text: str) -> Path:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o660)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o660)
    return path


def write_private_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return write_private_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def read_json(path: str | Path) -> dict[str, Any]:
    source = require_inside(path, MAIN_PROTECTED_ROOT, must_exist=True)
    if not source.is_file():
        raise ValueError("protected JSON input is not a file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON input must be an object")
    return payload


def read_case_ids(path: str | Path) -> list[str]:
    source = require_inside(path, MAIN_PROTECTED_ROOT, must_exist=True)
    case_ids = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case list must be non-empty and unique")
    if any(not CASE_ID_PATTERN.fullmatch(case_id) for case_id in case_ids):
        raise ValueError("case list contains a non-opaque ID")
    return case_ids


def _validate_seed(seed: Any) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def build_cxr_request(
    *, staging_run: str | Path, case_id: str, model_id: str, seed: int
) -> dict[str, Any]:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("invalid opaque case ID")
    if model_id not in ACTIVE_PROMPT_MODELS_V11:
        raise ValueError("inactive V1.1 CXR model")
    seed = _validate_seed(seed)
    staging = require_inside(staging_run, MAIN_PROTECTED_ROOT, must_exist=True)
    run_manifest_path = staging / "run_manifest.json"
    run_manifest = read_json(run_manifest_path)
    if run_manifest.get("schema_version") != STAGING_SCHEMA_V11:
        raise ValueError("CXR request source is not a V1.1 staging run")
    case_root = require_inside(
        staging / "cases" / case_id, staging, must_exist=True
    )
    ehr_path = case_root / "synthetic_ehr.json"
    facts_path = case_root / "ehr_facts.json"
    prompt_manifest_path = case_root / "cxr_prompts" / "prompt_manifest.json"
    ehr = read_json(ehr_path)
    facts = read_json(facts_path)
    prompt_manifest = read_json(prompt_manifest_path)
    if ehr.get("case_id") != case_id or facts.get("case_id") != case_id:
        raise ValueError("V1.1 EHR/fact lineage mismatch")
    if prompt_manifest.get("case_id") != case_id:
        raise ValueError("V1.1 prompt lineage mismatch")
    if prompt_manifest.get("ehr_facts_sha256") != sha256_file(facts_path):
        raise ValueError("V1.1 prompt is not bound to the EHR facts")
    if prompt_manifest.get("one_shared_clinical_intent") is not True:
        raise ValueError("V1.1 shared clinical-intent contract is missing")
    if prompt_manifest.get("adapter_must_not_add_prefix") is not True:
        raise ValueError("V1.1 no-prefix adapter contract is missing")
    models = prompt_manifest.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model_id), dict):
        raise ValueError("V1.1 prompt manifest lacks the requested model")
    prompt_meta = models[model_id]
    expected_relative = f"cxr_prompts/{model_id}.txt"
    if prompt_meta.get("path") != expected_relative:
        raise ValueError("V1.1 model prompt path is not canonical")
    prompt_path = require_inside(case_root / expected_relative, case_root, must_exist=True)
    prompt_hash = sha256_file(prompt_path)
    if prompt_meta.get("prompt_sha256") != prompt_hash:
        raise ValueError("V1.1 final prompt hash mismatch")
    if prompt_meta.get("is_final_model_input") is not True:
        raise ValueError("V1.1 prompt is not marked as final model input")

    request = {
        "schema_version": CXR_REQUEST_SCHEMA_V11,
        "request_id": f"cxrreq_{case_id}_{model_id}_s{seed:06d}",
        "case_id": case_id,
        "model_id": model_id,
        "seed": seed,
        "frozen_model_required": True,
        "adapter_must_not_add_prefix": True,
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
                "sha256": prompt_hash,
                "renderer_version": prompt_meta["renderer_version"],
                "clinical_intent_sha256": prompt_meta[
                    "clinical_intent_sha256"
                ],
                "conditioning_tier": prompt_meta["conditioning_tier"],
                "underconditioned": prompt_meta["underconditioned"],
                "available_context_ids": prompt_meta["available_context_ids"],
                "included_context_ids": prompt_meta["included_context_ids"],
                "omitted_context_ids": prompt_meta["omitted_context_ids"],
                "included_direct_fact_ids": prompt_meta[
                    "included_direct_fact_ids"
                ],
                "is_final_model_input": True,
            },
        },
    }
    validate_cxr_request(request)
    return request


def validate_cxr_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != CXR_REQUEST_SCHEMA_V11:
        raise ValueError("unsupported V1.1 CXR request schema")
    case_id = request.get("case_id")
    model_id = request.get("model_id")
    seed = _validate_seed(request.get("seed"))
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("invalid V1.1 request case ID")
    if model_id not in ACTIVE_PROMPT_MODELS_V11:
        raise ValueError("inactive V1.1 request model")
    if request.get("request_id") != f"cxrreq_{case_id}_{model_id}_s{seed:06d}":
        raise ValueError("V1.1 request ID does not match lineage")
    if request.get("frozen_model_required") is not True:
        raise ValueError("V1.1 request must require a frozen model")
    if request.get("adapter_must_not_add_prefix") is not True:
        raise ValueError("V1.1 request permits an adapter prefix")
    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("V1.1 request inputs are missing")
    for label in ("staging_run_manifest", "synthetic_ehr", "ehr_facts"):
        artifact = inputs.get(label)
        if not isinstance(artifact, dict):
            raise TypeError(f"V1.1 request {label} is missing")
        path = require_inside(
            str(artifact.get("path", "")), MAIN_PROTECTED_ROOT, must_exist=True
        )
        if sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"V1.1 request {label} hash mismatch")
    prompt = inputs.get("final_prompt")
    if not isinstance(prompt, dict):
        raise TypeError("V1.1 final prompt reference is missing")
    prompt_path = require_inside(
        str(prompt.get("path", "")), MAIN_PROTECTED_ROOT, must_exist=True
    )
    if sha256_file(prompt_path) != prompt.get("sha256"):
        raise ValueError("V1.1 final prompt artifact hash mismatch")
    if prompt.get("is_final_model_input") is not True:
        raise ValueError("V1.1 request prompt is not final")
    if not isinstance(prompt.get("clinical_intent_sha256"), str):
        raise ValueError("V1.1 request lacks clinical-intent lineage")


def prepare_cxr_request_run(
    *,
    staging_run: str | Path,
    case_ids_file: str | Path,
    output_root: str | Path,
    run_id: str,
    model_ids: Iterable[str] = ACTIVE_PROMPT_MODELS_V11,
    seeds: Iterable[int] = (0, 1),
    require_conditioned: bool = True,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    models = tuple(model_ids)
    seed_values = tuple(_validate_seed(seed) for seed in seeds)
    if not models or len(models) != len(set(models)):
        raise ValueError("model list must be non-empty and unique")
    if any(model not in ACTIVE_PROMPT_MODELS_V11 for model in models):
        raise ValueError("request run contains an inactive V1.1 model")
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("seed list must be non-empty and unique")
    staging = require_inside(staging_run, MAIN_PROTECTED_ROOT, must_exist=True)
    case_ids = read_case_ids(case_ids_file)

    requests = [
        build_cxr_request(
            staging_run=staging, case_id=case_id, model_id=model_id, seed=seed
        )
        for case_id in case_ids
        for model_id in models
        for seed in seed_values
    ]
    if require_conditioned and any(
        request["inputs"]["final_prompt"]["underconditioned"] is not False
        for request in requests
    ):
        raise ValueError("conditioned smoke contains an underconditioned EHR")

    output = require_inside(output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    target = output / run_id
    if target.exists():
        raise FileExistsError("V1.1 request run already exists")
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
                    "seed": request["seed"],
                    "path": f"requests/{path.name}",
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": CXR_REQUEST_RUN_SCHEMA_V11,
            "run_id": run_id,
            "staging_run": str(staging),
            "staging_run_manifest_sha256": sha256_file(
                staging / "run_manifest.json"
            ),
            "case_ids": case_ids,
            "model_ids": list(models),
            "seeds": list(seed_values),
            "require_conditioned": require_conditioned,
            "request_count": len(records),
            "requests": records,
            "gpu_inference_used": False,
        }
        manifest_path = write_private_json(temporary / "manifest.json", manifest)
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "prepared",
        "run_directory": str(target),
        "case_count": len(case_ids),
        "request_count": len(records),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


def load_model_requests(request_run: str | Path, model_id: str) -> list[dict[str, Any]]:
    source = require_inside(request_run, MAIN_PROTECTED_ROOT, must_exist=True)
    manifest = read_json(source / "manifest.json")
    if manifest.get("schema_version") != CXR_REQUEST_RUN_SCHEMA_V11:
        raise ValueError("unsupported V1.1 CXR request-run schema")
    requests: list[dict[str, Any]] = []
    for row in manifest.get("requests", []):
        if not isinstance(row, dict) or row.get("model_id") != model_id:
            continue
        path = require_inside(source / str(row.get("path", "")), source, must_exist=True)
        if sha256_file(path) != row.get("sha256"):
            raise ValueError("V1.1 request record hash mismatch")
        request = read_json(path)
        validate_cxr_request(request)
        requests.append(request)
    if not requests:
        raise ValueError("request run has no entries for the selected model")
    return requests


__all__ = [
    "CXR_REQUEST_RUN_SCHEMA_V11",
    "CXR_REQUEST_SCHEMA_V11",
    "MAIN_PROTECTED_ROOT",
    "canonical_json_sha256",
    "load_model_requests",
    "prepare_cxr_request_run",
    "private_directory",
    "read_json",
    "require_inside",
    "sha256_file",
    "validate_cxr_request",
    "write_private_json",
]

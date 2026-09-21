"""Unified protected request preparation and frozen-model execution adapters."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
    write_private_text,
)

from .contracts import (
    ACTIVE_REPORT_MODELS,
    CXR_CANDIDATE_SCHEMA,
    build_cxr_candidate,
    build_cxr_request,
    build_report_candidate,
    build_report_request,
    validate_cxr_candidate,
    validate_cxr_request,
    validate_report_candidate,
)
from .prompts import ACTIVE_PROMPT_MODELS
from .staging import read_case_ids


CXR_REQUEST_RUN_SCHEMA = "tricompose-cxr-request-run-v1"
CXR_CANDIDATE_RUN_SCHEMA = "tricompose-cxr-candidate-run-v1"
REPORT_CANDIDATE_RUN_SCHEMA = "tricompose-report-candidate-run-v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class CXRBatchResult(Protocol):
    images: Sequence[Any]
    prompt_token_counts: Sequence[int]
    elapsed_seconds: float
    peak_vram_gib: float


class FrozenCXRRuntime(Protocol):
    audit: dict[str, Any]

    def generate_batch(
        self, prompts: Sequence[str], seeds: Sequence[int]
    ) -> CXRBatchResult: ...


class GeneratedReportLike(Protocol):
    canonical_text: str
    elapsed_seconds: float


class FrozenReportRuntime(Protocol):
    audit: dict[str, Any]

    def generate(self, image_path: Path) -> GeneratedReportLike: ...

    @property
    def peak_vram_gib(self) -> float: ...


def _read_json(path: str | Path) -> dict[str, Any]:
    source = require_private_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON must be an object")
    return payload


def _private_dir(path: Path, *, exist_ok: bool = False) -> None:
    old_umask = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=exist_ok, mode=0o700)
    finally:
        os.umask(old_umask)
    enforce_private_directory_mode(path)


def _new_atomic_run(output_root: str | Path, run_id: str) -> tuple[Path, Path]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    root = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    if not root.exists():
        _private_dir(root)
    else:
        enforce_private_directory_mode(root)
    target = require_inside(root / run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError("output run already exists")
    temp = root / f".{run_id}.{uuid.uuid4().hex}.tmp"
    _private_dir(temp)
    return temp, target


def _commit_atomic_run(temp: Path, target: Path) -> None:
    os.rename(temp, target)
    enforce_private_directory_mode(target)


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if not values or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in values
    ):
        raise ValueError("seeds must be non-empty non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("seeds must not contain duplicates")
    return values


def prepare_cxr_request_run(
    *,
    staging_run: str | Path,
    case_ids_file: str | Path,
    output_root: str | Path,
    run_id: str,
    model_ids: Iterable[str] = ACTIVE_PROMPT_MODELS,
    seeds: Iterable[int] = (0, 1),
    require_conditioned: bool = True,
) -> dict[str, Any]:
    """Materialize exact CXR requests without importing or running a model."""

    models = tuple(model_ids)
    if not models or len(set(models)) != len(models):
        raise ValueError("CXR model IDs must be non-empty and unique")
    if any(model_id not in ACTIVE_PROMPT_MODELS for model_id in models):
        raise ValueError("CXR request run contains an inactive model")
    seed_values = _validate_seeds(seeds)
    case_ids = read_case_ids(case_ids_file)
    staging_root = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)

    # Eligibility is checked before creating output, so failure leaves no run.
    if require_conditioned:
        underconditioned: list[str] = []
        for case_id in case_ids:
            prompt_manifest = _read_json(
                staging_root
                / "cases"
                / case_id
                / "cxr_prompts"
                / "prompt_manifest.json"
            )
            prompt_rows = prompt_manifest.get("models")
            if not isinstance(prompt_rows, dict) or any(
                not isinstance(prompt_rows.get(model_id), dict)
                or prompt_rows[model_id].get("underconditioned") is not False
                for model_id in models
            ):
                underconditioned.append(case_id)
        if underconditioned:
            raise ValueError(
                f"conditioned CXR request requires eligible facts; rejected_count={len(underconditioned)}"
            )

    temp, target = _new_atomic_run(output_root, run_id)
    records: list[dict[str, Any]] = []
    try:
        requests_root = temp / "requests"
        _private_dir(requests_root)
        for case_id in case_ids:
            for model_id in models:
                for seed in seed_values:
                    request = build_cxr_request(
                        staging_run=staging_root,
                        case_id=case_id,
                        model_id=model_id,
                        seed=seed,
                    )
                    path = write_private_json(
                        requests_root / f"{request['request_id']}.json", request
                    )
                    records.append(
                        {
                            "request_id": request["request_id"],
                            "case_id": case_id,
                            "model_id": model_id,
                            "seed": seed,
                            "path": f"requests/{path.name}",
                            "sha256": sha256_file(path),
                        }
                    )
        manifest_path = write_private_json(
            temp / "manifest.json",
            {
                "schema_version": CXR_REQUEST_RUN_SCHEMA,
                "run_id": run_id,
                "staging_run": str(staging_root),
                "staging_run_manifest_sha256": sha256_file(
                    staging_root / "run_manifest.json"
                ),
                "case_ids": case_ids,
                "model_ids": list(models),
                "seeds": list(seed_values),
                "require_conditioned": require_conditioned,
                "request_count": len(records),
                "requests": records,
                "gpu_inference_used": False,
            },
        )
        _commit_atomic_run(temp, target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return {
        "status": "prepared",
        "run_directory": str(target),
        "request_count": len(records),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


def _load_request_records(request_run: Path, model_id: str) -> list[dict[str, Any]]:
    manifest = _read_json(request_run / "manifest.json")
    if manifest.get("schema_version") != CXR_REQUEST_RUN_SCHEMA:
        raise ValueError("unsupported CXR request-run schema")
    rows = manifest.get("requests")
    if not isinstance(rows, list):
        raise TypeError("CXR request-run manifest has no requests")
    requests: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("model_id") != model_id:
            continue
        path = require_inside(
            request_run / str(row.get("path", "")), request_run, must_exist=True
        )
        if sha256_file(path) != row.get("sha256"):
            raise ValueError("CXR request-run record hash mismatch")
        request = _read_json(path)
        validate_cxr_request(request)
        requests.append(request)
    if not requests:
        raise ValueError("CXR request run has no requests for this model")
    return requests


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def run_cxr_request_run(
    *,
    request_run: str | Path,
    output_root: str | Path,
    output_run_id: str,
    model_id: str,
    model_revision: str,
    model_audit: Mapping[str, Any],
    runtime_factory: Callable[[], FrozenCXRRuntime],
    batch_size: int,
) -> dict[str, Any]:
    """Execute one frozen CXR model and emit only common CXR candidates."""

    if model_id not in ACTIVE_PROMPT_MODELS:
        raise ValueError("inactive CXR model")
    max_batch = {"roentgen_v2": 4, "chexgenbench_sana": 2, "chexgenbench_pixart": 1}[model_id]
    if not isinstance(batch_size, int) or not 1 <= batch_size <= max_batch:
        raise ValueError("batch size is invalid for the selected CXR model")
    source = require_inside(request_run, PROTECTED_ROOT, must_exist=True)
    requests = _load_request_records(source, model_id)
    temp, target = _new_atomic_run(output_root, output_run_id)
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    try:
        candidates_root = temp / "candidates"
        logs_root = temp / "logs"
        _private_dir(candidates_root)
        _private_dir(logs_root)
        log_path = logs_root / f"{model_id}.log"
        with log_path.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = runtime_factory()
                if runtime.audit != dict(model_audit):
                    raise ValueError("CXR runtime model audit changed after validation")
                for batch in _chunks(requests, batch_size):
                    prompts = [
                        require_private_file(row["inputs"]["final_prompt"]["path"])
                        .read_text(encoding="utf-8")
                        for row in batch
                    ]
                    seeds = [int(row["seed"]) for row in batch]
                    result = runtime.generate_batch(prompts, seeds)
                    if len(result.images) != len(batch):
                        raise RuntimeError("CXR runtime returned the wrong image count")
                    per_case_seconds = float(result.elapsed_seconds) / len(batch)
                    for request, image, token_count in zip(
                        batch,
                        result.images,
                        result.prompt_token_counts,
                        strict=True,
                    ):
                        candidate_id = (
                            f"cxr_{request['case_id']}_{model_id}_s{int(request['seed']):06d}"
                        )
                        case_dir = candidates_root / candidate_id
                        _private_dir(case_dir)
                        if not hasattr(image, "save") or not hasattr(image, "size"):
                            raise TypeError("CXR runtime output is not PIL-compatible")
                        image_path = case_dir / "synthetic_cxr.png"
                        image.convert("RGB").save(image_path, format="PNG")
                        enforce_private_file_mode(image_path)
                        write_private_json(case_dir / "request.json", dict(request))
                        candidate = build_cxr_candidate(
                            request=request,
                            image_path=image_path,
                            image_dimensions=(int(image.size[0]), int(image.size[1])),
                            model_revision=model_revision,
                            prompt_token_count=int(token_count),
                            runtime_seconds=per_case_seconds,
                            peak_vram_gib=float(result.peak_vram_gib),
                        )
                        candidate["artifact"]["path"] = str(
                            target
                            / "candidates"
                            / candidate_id
                            / "synthetic_cxr.png"
                        )
                        candidate_path = write_private_json(
                            case_dir / "candidate.json", candidate
                        )
                        completed.append(
                            {
                                "candidate_id": candidate_id,
                                "case_id": request["case_id"],
                                "seed": request["seed"],
                                "path": f"candidates/{candidate_id}/candidate.json",
                                "sha256": sha256_file(candidate_path),
                            }
                        )
        enforce_private_file_mode(log_path)
        manifest_path = write_private_json(
            temp / "manifest.json",
            {
                "schema_version": CXR_CANDIDATE_RUN_SCHEMA,
                "run_id": output_run_id,
                "model_id": model_id,
                "model_revision": model_revision,
                "frozen_model": True,
                "source_request_run": str(source),
                "source_request_run_manifest_sha256": sha256_file(
                    source / "manifest.json"
                ),
                "candidate_count": len(completed),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "model_audit": dict(model_audit),
                "candidates": completed,
            },
        )
        _commit_atomic_run(temp, target)
        for row in completed:
            validate_cxr_candidate(_read_json(target / row["path"]))
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        if target.exists():
            shutil.rmtree(target)
        raise
    return {
        "status": "completed",
        "model_id": model_id,
        "candidate_count": len(completed),
        "run_directory": str(target),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


def _load_cxr_candidates(cxr_run: Path) -> list[dict[str, Any]]:
    manifest = _read_json(cxr_run / "manifest.json")
    if manifest.get("schema_version") != CXR_CANDIDATE_RUN_SCHEMA:
        raise ValueError("unsupported CXR candidate-run schema")
    rows = manifest.get("candidates")
    if not isinstance(rows, list):
        raise TypeError("CXR candidate run has no candidate list")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("CXR candidate manifest entry is invalid")
        path = require_inside(
            cxr_run / str(row.get("path", "")), cxr_run, must_exist=True
        )
        if sha256_file(path) != row.get("sha256"):
            raise ValueError("CXR candidate record hash mismatch")
        candidate = _read_json(path)
        validate_cxr_candidate(candidate)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("CXR candidate run is empty")
    return candidates


def run_report_candidate_run(
    *,
    cxr_run: str | Path | Iterable[str | Path],
    output_root: str | Path,
    output_run_id: str,
    model_id: str,
    model_revision: str,
    model_audit: Mapping[str, Any],
    runtime_factory: Callable[[], FrozenReportRuntime],
) -> dict[str, Any]:
    """Execute one frozen report model over one or more common CXR runs."""

    if model_id not in ACTIVE_REPORT_MODELS:
        raise ValueError("inactive report model")
    if isinstance(cxr_run, (str, Path)):
        source_values = (cxr_run,)
    else:
        source_values = tuple(cxr_run)
    if not source_values:
        raise ValueError("at least one CXR candidate run is required")
    sources = tuple(
        require_inside(value, PROTECTED_ROOT, must_exist=True)
        for value in source_values
    )
    if len(set(sources)) != len(sources):
        raise ValueError("CXR candidate runs must not contain duplicates")
    cxr_candidates = [
        candidate
        for source in sources
        for candidate in _load_cxr_candidates(source)
    ]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in cxr_candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("CXR candidate bank contains duplicate candidate IDs")
    temp, target = _new_atomic_run(output_root, output_run_id)
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        candidates_root = temp / "candidates"
        logs_root = temp / "logs"
        _private_dir(candidates_root)
        _private_dir(logs_root)
        log_path = logs_root / f"{model_id}.log"
        with log_path.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                runtime = runtime_factory()
                if runtime.audit != dict(model_audit):
                    raise ValueError("report runtime model audit changed after validation")
                for cxr_candidate in cxr_candidates:
                    request = build_report_request(
                        cxr_candidate=cxr_candidate,
                        report_model_id=model_id,
                    )
                    result = runtime.generate(
                        require_private_file(cxr_candidate["artifact"]["path"])
                    )
                    text = str(result.canonical_text).strip()
                    if not text:
                        raise ValueError("report runtime returned empty text")
                    candidate_id = f"report_{cxr_candidate['candidate_id']}_{model_id}"
                    case_dir = candidates_root / candidate_id
                    _private_dir(case_dir)
                    write_private_json(case_dir / "request.json", request)
                    report_path = write_private_text(
                        case_dir / "synthetic_report.txt", text + "\n"
                    )
                    candidate = build_report_candidate(
                        request=request,
                        report_path=report_path,
                        model_revision=model_revision,
                        runtime_seconds=float(result.elapsed_seconds),
                    )
                    candidate["artifact"]["path"] = str(
                        target
                        / "candidates"
                        / candidate_id
                        / "synthetic_report.txt"
                    )
                    candidate_path = write_private_json(
                        case_dir / "candidate.json", candidate
                    )
                    completed.append(
                        {
                            "candidate_id": candidate_id,
                            "case_id": cxr_candidate["case_id"],
                            "parent_cxr_candidate_id": cxr_candidate["candidate_id"],
                            "path": f"candidates/{candidate_id}/candidate.json",
                            "sha256": sha256_file(candidate_path),
                        }
                    )
        enforce_private_file_mode(log_path)
        manifest_path = write_private_json(
            temp / "manifest.json",
            {
                "schema_version": REPORT_CANDIDATE_RUN_SCHEMA,
                "run_id": output_run_id,
                "model_id": model_id,
                "model_revision": model_revision,
                "frozen_model": True,
                "source_cxr_runs": [
                    {
                        "path": str(source),
                        "manifest_sha256": sha256_file(source / "manifest.json"),
                    }
                    for source in sources
                ],
                "candidate_count": len(completed),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_vram_gib": round(float(runtime.peak_vram_gib), 3),
                "model_audit": dict(model_audit),
                "candidates": completed,
            },
        )
        _commit_atomic_run(temp, target)
        for row in completed:
            candidate = _read_json(target / row["path"])
            validate_report_candidate(candidate)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        if target.exists():
            shutil.rmtree(target)
        raise
    return {
        "status": "completed",
        "model_id": model_id,
        "candidate_count": len(completed),
        "run_directory": str(target),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


__all__ = [
    "CXR_CANDIDATE_RUN_SCHEMA",
    "CXR_REQUEST_RUN_SCHEMA",
    "REPORT_CANDIDATE_RUN_SCHEMA",
    "prepare_cxr_request_run",
    "run_cxr_request_run",
    "run_report_candidate_run",
]

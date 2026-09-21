"""Slurm-only frozen CXR execution for exact TriCompose V1.1 requests."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .cxr_contracts import (
    MAIN_PROTECTED_ROOT,
    canonical_json_sha256,
    load_model_requests,
    private_directory,
    read_json,
    require_inside,
    sha256_file,
    validate_cxr_request,
    write_private_json,
)
from .prompts import ACTIVE_PROMPT_MODELS_V11


CXR_CANDIDATE_SCHEMA_V11 = "tricompose-cxr-candidate-v1.1"
CXR_CANDIDATE_RUN_SCHEMA_V11 = "tricompose-cxr-candidate-run-v1.1"
MAX_BATCH_SIZE = {
    "roentgen_v2": 4,
    "chexgenbench_sana": 2,
    "chexgenbench_pixart": 1,
}


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


def _chunks(rows: Sequence[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("schema_version") != CXR_CANDIDATE_SCHEMA_V11:
        raise ValueError("unsupported V1.1 CXR candidate schema")
    if candidate.get("frozen_model") is not True:
        raise ValueError("V1.1 CXR candidate model is not frozen")
    artifact = candidate.get("artifact")
    if not isinstance(artifact, dict):
        raise TypeError("V1.1 CXR candidate artifact is missing")
    path = require_inside(
        str(artifact.get("path", "")), MAIN_PROTECTED_ROOT, must_exist=True
    )
    if sha256_file(path) != artifact.get("sha256"):
        raise ValueError("V1.1 CXR candidate image hash mismatch")
    if artifact.get("mime_type") != "image/png":
        raise ValueError("V1.1 CXR candidate must be a PNG")


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
    """Execute one frozen model; the adapter consumes exact staged text."""

    if model_id not in ACTIVE_PROMPT_MODELS_V11:
        raise ValueError("inactive V1.1 CXR model")
    if not 1 <= batch_size <= MAX_BATCH_SIZE[model_id]:
        raise ValueError("batch size is invalid for this V1.1 CXR model")
    source = require_inside(request_run, MAIN_PROTECTED_ROOT, must_exist=True)
    requests = load_model_requests(source, model_id)
    output = require_inside(output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    target = output / output_run_id
    if target.exists():
        raise FileExistsError("V1.1 CXR output run already exists")
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
                    raise ValueError("V1.1 frozen model audit changed at load time")
                for batch in _chunks(requests, batch_size):
                    prompts: list[str] = []
                    for request in batch:
                        validate_cxr_request(request)
                        prompt = request["inputs"]["final_prompt"]
                        prompt_path = require_inside(
                            prompt["path"], MAIN_PROTECTED_ROOT, must_exist=True
                        )
                        if sha256_file(prompt_path) != prompt["sha256"]:
                            raise ValueError("V1.1 final prompt changed before inference")
                        prompts.append(prompt_path.read_text(encoding="utf-8"))
                    seeds = [int(request["seed"]) for request in batch]
                    result = runtime.generate_batch(prompts, seeds)
                    if not (
                        len(result.images)
                        == len(result.prompt_token_counts)
                        == len(batch)
                    ):
                        raise RuntimeError("V1.1 CXR runtime returned an invalid batch")
                    per_image_seconds = float(result.elapsed_seconds) / len(batch)
                    for request, image, token_count in zip(
                        batch,
                        result.images,
                        result.prompt_token_counts,
                        strict=True,
                    ):
                        case_id = request["case_id"]
                        seed = int(request["seed"])
                        candidate_id = (
                            f"cxr_{case_id}_{model_id}_s{seed:06d}"
                        )
                        case_dir = candidates_root / candidate_id
                        private_directory(case_dir)
                        if not hasattr(image, "save") or not hasattr(image, "size"):
                            raise TypeError("V1.1 CXR output is not PIL-compatible")
                        image_path = case_dir / "synthetic_cxr.png"
                        image.convert("RGB").save(image_path, format="PNG")
                        os.chmod(image_path, 0o660)
                        write_private_json(case_dir / "request.json", request)
                        final_image_path = (
                            target
                            / "candidates"
                            / candidate_id
                            / "synthetic_cxr.png"
                        )
                        candidate = {
                            "schema_version": CXR_CANDIDATE_SCHEMA_V11,
                            "candidate_id": candidate_id,
                            "case_id": case_id,
                            "modality": "cxr",
                            "model_id": model_id,
                            "model_revision": model_revision,
                            "frozen_model": True,
                            "seed": seed,
                            "input_request_sha256": canonical_json_sha256(request),
                            "ehr_sha256": request["inputs"]["synthetic_ehr"][
                                "sha256"
                            ],
                            "ehr_facts_sha256": request["inputs"]["ehr_facts"][
                                "sha256"
                            ],
                            "clinical_intent_sha256": request["inputs"][
                                "final_prompt"
                            ]["clinical_intent_sha256"],
                            "prompt_sha256": request["inputs"]["final_prompt"][
                                "sha256"
                            ],
                            "adapter_added_prefix": False,
                            "artifact": {
                                "path": str(final_image_path),
                                "sha256": sha256_file(image_path),
                                "mime_type": "image/png",
                                "dimensions": [
                                    int(image.size[0]),
                                    int(image.size[1]),
                                ],
                            },
                            "cost": {
                                "model_calls": 1,
                                "runtime_seconds": round(per_image_seconds, 3),
                                "peak_vram_gib": round(
                                    float(result.peak_vram_gib), 3
                                ),
                                "prompt_token_count": int(token_count),
                            },
                        }
                        candidate_path = write_private_json(
                            case_dir / "candidate.json", candidate
                        )
                        completed.append(
                            {
                                "candidate_id": candidate_id,
                                "case_id": case_id,
                                "seed": seed,
                                "path": (
                                    f"candidates/{candidate_id}/candidate.json"
                                ),
                                "sha256": sha256_file(candidate_path),
                            }
                        )
        os.chmod(log_path, 0o660)
        manifest_path = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": CXR_CANDIDATE_RUN_SCHEMA_V11,
                "run_id": output_run_id,
                "model_id": model_id,
                "model_revision": model_revision,
                "frozen_model": True,
                "adapter_added_prefix": False,
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
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
        for row in completed:
            candidate = read_json(target / row["path"])
            _validate_candidate(candidate)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        # Never delete a committed target here. Another scheduler task may
        # have completed the same opaque run while this task was loading its
        # model. This execution owns only its UUID-named temporary directory.
        raise
    return {
        "status": "completed",
        "model_id": model_id,
        "candidate_count": len(completed),
        "run_directory": str(target),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


__all__ = ["run_cxr_request_run"]

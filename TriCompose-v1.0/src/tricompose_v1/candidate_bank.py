"""Validate and index a complete protected V1 generation candidate bank."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_directory_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)

from .contracts import (
    ACTIVE_REPORT_MODELS,
    build_ehr_candidate,
    validate_cxr_candidate,
    validate_report_candidate,
)
from .execution import (
    CXR_CANDIDATE_RUN_SCHEMA,
    REPORT_CANDIDATE_RUN_SCHEMA,
)
from .prompts import ACTIVE_PROMPT_MODELS


CANDIDATE_BANK_SCHEMA = "tricompose-generation-candidate-bank-v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


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


def _load_run(
    path: str | Path,
    *,
    schema: str,
    validator: Callable[[Mapping[str, Any]], None],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    root = require_inside(path, PROTECTED_ROOT, must_exist=True)
    manifest_path = require_private_file(root / "manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != schema:
        raise ValueError("candidate run has the wrong schema")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate run is empty")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("candidate manifest entry is invalid")
        candidate_path = require_inside(
            root / str(row.get("path", "")), root, must_exist=True
        )
        if sha256_file(candidate_path) != row.get("sha256"):
            raise ValueError("candidate manifest hash mismatch")
        candidate = _read_json(candidate_path)
        validator(candidate)
        candidates.append(candidate)
    return root, manifest, candidates


def _unique_roots(values: Iterable[str | Path], label: str) -> tuple[str | Path, ...]:
    rows = tuple(values)
    if not rows:
        raise ValueError(f"at least one {label} run is required")
    resolved = tuple(str(Path(value).resolve(strict=True)) for value in rows)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} runs contain duplicates")
    return rows


def finalize_candidate_bank(
    *,
    staging_run: str | Path,
    cxr_runs: Iterable[str | Path],
    report_runs: Iterable[str | Path],
    output_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Require all active generators and atomically index their candidates."""

    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    staging = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)
    staging_manifest = require_private_file(staging / "run_manifest.json")
    cxr_values = _unique_roots(cxr_runs, "CXR")
    report_values = _unique_roots(report_runs, "report")

    cxr_loaded = [
        _load_run(
            value,
            schema=CXR_CANDIDATE_RUN_SCHEMA,
            validator=validate_cxr_candidate,
        )
        for value in cxr_values
    ]
    report_loaded = [
        _load_run(
            value,
            schema=REPORT_CANDIDATE_RUN_SCHEMA,
            validator=validate_report_candidate,
        )
        for value in report_values
    ]

    cxr_by_model = {str(manifest.get("model_id")): (root, manifest, rows)
                    for root, manifest, rows in cxr_loaded}
    report_by_model = {str(manifest.get("model_id")): (root, manifest, rows)
                       for root, manifest, rows in report_loaded}
    if set(cxr_by_model) != set(ACTIVE_PROMPT_MODELS):
        raise ValueError("candidate bank does not contain every active CXR model")
    if set(report_by_model) != set(ACTIVE_REPORT_MODELS):
        raise ValueError("candidate bank does not contain every active report model")
    if len(cxr_by_model) != len(cxr_loaded) or len(report_by_model) != len(report_loaded):
        raise ValueError("candidate bank contains duplicate model runs")

    cxr_candidates = [row for _, _, rows in cxr_loaded for row in rows]
    cxr_ids = [str(row["candidate_id"]) for row in cxr_candidates]
    if len(set(cxr_ids)) != len(cxr_ids):
        raise ValueError("candidate bank contains duplicate CXR candidate IDs")
    cxr_by_id = {str(row["candidate_id"]): row for row in cxr_candidates}
    case_ids = sorted({str(row["case_id"]) for row in cxr_candidates})
    ehr_candidates = [
        build_ehr_candidate(staging_run=staging, case_id=case_id)
        for case_id in case_ids
    ]

    reports = [row for _, _, rows in report_loaded for row in rows]
    report_ids = [str(row["candidate_id"]) for row in reports]
    if len(set(report_ids)) != len(report_ids):
        raise ValueError("candidate bank contains duplicate report candidate IDs")
    expected_cxr_ids = set(cxr_ids)
    for model_id, (_, _, rows) in report_by_model.items():
        covered = {str(row["parent_ids"][1]) for row in rows}
        if covered != expected_cxr_ids or len(rows) != len(cxr_candidates):
            raise ValueError(f"report model does not cover the full CXR bank: {model_id}")
        for report in rows:
            cxr = cxr_by_id[str(report["parent_ids"][1])]
            if report["parent_ids"][0] != cxr["parent_ids"][0]:
                raise ValueError("report and CXR EHR lineage do not match")

    root = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    if not root.exists():
        _private_dir(root)
    target = require_inside(root / run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError("candidate-bank run already exists")
    temp = root / f".{run_id}.{uuid.uuid4().hex}.tmp"
    _private_dir(temp)
    try:
        manifest_path = write_private_json(
            temp / "manifest.json",
            {
                "schema_version": CANDIDATE_BANK_SCHEMA,
                "run_id": run_id,
                "complete_generation_candidate_bank": True,
                "selection_performed": False,
                "staging_run": {
                    "path": str(staging),
                    "manifest_sha256": sha256_file(staging_manifest),
                },
                "models": {
                    "cxr": list(ACTIVE_PROMPT_MODELS),
                    "report": list(ACTIVE_REPORT_MODELS),
                },
                "counts": {
                    "ehr_candidates": len(ehr_candidates),
                    "cxr_candidates": len(cxr_candidates),
                    "report_candidates": len(reports),
                },
                "ehr_candidates": ehr_candidates,
                "cxr_runs": [
                    {
                        "model_id": model_id,
                        "path": str(root_path),
                        "manifest_sha256": sha256_file(root_path / "manifest.json"),
                        "candidate_count": len(rows),
                    }
                    for model_id, (root_path, _, rows) in cxr_by_model.items()
                ],
                "report_runs": [
                    {
                        "model_id": model_id,
                        "path": str(root_path),
                        "manifest_sha256": sha256_file(root_path / "manifest.json"),
                        "candidate_count": len(rows),
                    }
                    for model_id, (root_path, _, rows) in report_by_model.items()
                ],
            },
        )
        os.rename(temp, target)
        enforce_private_directory_mode(target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return {
        "status": "completed",
        "run_directory": str(target),
        "ehr_candidate_count": len(ehr_candidates),
        "cxr_candidate_count": len(cxr_candidates),
        "report_candidate_count": len(reports),
        "manifest_sha256": sha256_file(target / manifest_path.name),
    }


__all__ = ["CANDIDATE_BANK_SCHEMA", "finalize_candidate_bank"]

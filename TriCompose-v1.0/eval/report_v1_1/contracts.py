"""Shared protected-artifact contracts for V1.1 report evaluation.

The loaders in this module validate hashes and lineage without treating a
missing clinical statement as a negative finding.  They intentionally support
only fully synthetic V1.1 artifacts below the workspace protected root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROTECTED_ROOT = WORKSPACE / "artifacts" / "protected"
REPORT_RUN_SCHEMA = "tricompose-report-candidate-run-v1.1"
REPORT_CANDIDATE_SCHEMA = "tricompose-report-candidate-v1.1"
CXR_RUN_SCHEMA = "tricompose-cxr-candidate-run-v1.1"
CXR_CANDIDATE_SCHEMA = "tricompose-cxr-candidate-v1.1"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")

CHEXPERT_FINDINGS = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
    "no_finding",
)
FINDING_STATES = frozenset({"positive", "negative", "uncertain", "unknown"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inside(path: str | Path, root: str | Path, *, must_exist: bool) -> Path:
    resolved_root = Path(root).resolve(strict=True)
    resolved = Path(path).resolve(strict=must_exist)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("path is outside the protected artifact boundary")
    return resolved


def read_json(path: str | Path, *, root: str | Path = PROTECTED_ROOT) -> dict[str, Any]:
    source = require_inside(path, root, must_exist=True)
    if not source.is_file():
        raise ValueError("JSON input is not a file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON input must contain an object")
    return payload


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
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def new_atomic_run(output_root: str | Path, run_id: str) -> tuple[Path, Path]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    root = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(root, exist_ok=True)
    target = require_inside(root / run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError("evaluation run already exists")
    temporary = root / f".{run_id}.{uuid.uuid4().hex}.tmp"
    private_directory(temporary)
    return temporary, target


def commit_atomic_run(temporary: Path, target: Path) -> None:
    os.rename(temporary, target)
    os.chmod(target, 0o2770)


def discard_atomic_run(temporary: Path) -> None:
    if temporary.exists():
        shutil.rmtree(temporary)


def _load_manifest_rows(
    runs: Iterable[str | Path],
    *,
    run_schema: str,
    row_key: str,
) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    run_paths = [require_inside(path, PROTECTED_ROOT, must_exist=True) for path in runs]
    if not run_paths or len(run_paths) != len(set(run_paths)):
        raise ValueError("run list must be non-empty and unique")
    loaded: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    ids: set[str] = set()
    for run_path in run_paths:
        manifest_path = run_path / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != run_schema:
            raise ValueError(f"unsupported run schema: {run_path.name}")
        if manifest.get("frozen_model") is not True:
            raise ValueError(f"run is not marked frozen: {run_path.name}")
        rows = manifest.get(row_key)
        if not isinstance(rows, list):
            raise TypeError(f"run lacks {row_key}: {run_path.name}")
        if manifest.get("candidate_count") != len(rows):
            raise ValueError(f"candidate count mismatch: {run_path.name}")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("manifest candidate entry is invalid")
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in ids:
                raise ValueError("candidate IDs must be non-empty and globally unique")
            candidate_path = require_inside(
                run_path / str(row.get("path", "")), run_path, must_exist=True
            )
            if sha256_file(candidate_path) != row.get("sha256"):
                raise ValueError(f"candidate JSON hash mismatch: {candidate_id}")
            candidate = read_json(candidate_path, root=run_path)
            if candidate.get("candidate_id") != candidate_id:
                raise ValueError("candidate ID does not match its manifest")
            if candidate.get("case_id") != row.get("case_id"):
                raise ValueError("candidate case ID does not match its manifest")
            ids.add(candidate_id)
            loaded.append((run_path, manifest, candidate))
    return loaded


def load_cxr_candidates(runs: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    loaded = _load_manifest_rows(runs, run_schema=CXR_RUN_SCHEMA, row_key="candidates")
    candidates: dict[str, dict[str, Any]] = {}
    for _, manifest, candidate in loaded:
        if candidate.get("schema_version") != CXR_CANDIDATE_SCHEMA:
            raise ValueError("unsupported CXR candidate schema")
        if candidate.get("frozen_model") is not True or candidate.get("modality") != "cxr":
            raise ValueError("invalid frozen CXR candidate")
        if candidate.get("model_id") != manifest.get("model_id"):
            raise ValueError("CXR candidate model lineage mismatch")
        artifact = candidate.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("CXR candidate lacks an artifact")
        image_path = require_inside(
            str(artifact.get("path", "")), PROTECTED_ROOT, must_exist=True
        )
        if sha256_file(image_path) != artifact.get("sha256"):
            raise ValueError("CXR artifact hash mismatch")
        candidates[str(candidate["candidate_id"])] = candidate
    return candidates


def load_report_candidates(
    runs: Iterable[str | Path],
    *,
    cxr_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    loaded = _load_manifest_rows(runs, run_schema=REPORT_RUN_SCHEMA, row_key="candidates")
    candidates: dict[str, dict[str, Any]] = {}
    for _, manifest, candidate in loaded:
        if candidate.get("schema_version") != REPORT_CANDIDATE_SCHEMA:
            raise ValueError("unsupported report candidate schema")
        if candidate.get("frozen_model") is not True or candidate.get("modality") != "report":
            raise ValueError("invalid frozen report candidate")
        if candidate.get("model_id") != manifest.get("model_id"):
            raise ValueError("report candidate model lineage mismatch")
        parent_id = str(candidate.get("parent_cxr_candidate_id", ""))
        parent = cxr_candidates.get(parent_id)
        if parent is None:
            raise ValueError("report parent CXR is absent from the supplied CXR runs")
        if parent.get("case_id") != candidate.get("case_id"):
            raise ValueError("report/CXR case lineage mismatch")
        if candidate.get("ehr_sha256_retained_for_lineage") != parent.get("ehr_sha256"):
            raise ValueError("report/EHR lineage hash mismatch")
        if candidate.get("ehr_facts_sha256_retained_for_lineage") != parent.get(
            "ehr_facts_sha256"
        ):
            raise ValueError("report/EHR-facts lineage hash mismatch")
        artifact = candidate.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("report candidate lacks an artifact")
        report_path = require_inside(
            str(artifact.get("path", "")), PROTECTED_ROOT, must_exist=True
        )
        if sha256_file(report_path) != artifact.get("sha256"):
            raise ValueError("report artifact hash mismatch")
        candidates[str(candidate["candidate_id"])] = candidate
    return candidates


def read_report_text(candidate: Mapping[str, Any]) -> str:
    artifact = candidate.get("artifact")
    if not isinstance(artifact, dict):
        raise TypeError("report candidate lacks an artifact")
    path = require_inside(str(artifact.get("path", "")), PROTECTED_ROOT, must_exist=True)
    if sha256_file(path) != artifact.get("sha256"):
        raise ValueError("report artifact changed after generation")
    return path.read_text(encoding="utf-8", errors="replace")


def validate_finding_states(states: Mapping[str, Any]) -> dict[str, str]:
    if set(states) != set(CHEXPERT_FINDINGS):
        raise ValueError("finding-state vector does not contain exactly 14 findings")
    normalized = {name: str(states[name]) for name in CHEXPERT_FINDINGS}
    if any(state not in FINDING_STATES for state in normalized.values()):
        raise ValueError("finding-state vector contains an invalid state")
    return normalized


__all__ = [
    "CHEXPERT_FINDINGS",
    "FINDING_STATES",
    "PROTECTED_ROOT",
    "commit_atomic_run",
    "discard_atomic_run",
    "load_cxr_candidates",
    "load_report_candidates",
    "new_atomic_run",
    "read_json",
    "read_report_text",
    "require_inside",
    "sha256_file",
    "validate_finding_states",
    "write_private_json",
    "write_private_text",
]

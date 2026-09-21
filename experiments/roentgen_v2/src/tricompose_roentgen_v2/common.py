"""Workspace and protected-artifact helpers for RoentGen-v2."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_EXPERIMENT_ROOT = PROTECTED_ROOT / "medim" / "runs"
PROTECTED_ROENTGEN_ROOT = PROTECTED_ROOT / "roentgen_v2"
PROTECTED_EXPERIMENT_ROOT = PROTECTED_ROENTGEN_ROOT / "runs"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CASE_ID_PATTERN = re.compile(r"case_[0-9]{3}")


def validate_run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid run ID")
    return value


def validate_case_id(value: str) -> str:
    if CASE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid case ID")
    return value


def require_source_run(run_id: str) -> Path:
    validate_run_id(run_id)
    resolved = require_inside(
        SOURCE_EXPERIMENT_ROOT / run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    if not resolved.is_dir():
        raise ValueError("protected source run is not a directory")
    enforce_private_directory_mode(resolved)
    return resolved


def create_output_run(run_id: str) -> Path:
    validate_run_id(run_id)
    return create_private_stage_dir(PROTECTED_EXPERIMENT_ROOT / run_id)


def read_private_json(path: str | Path) -> dict[str, Any]:
    resolved = require_private_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


def create_private_dir(path: str | Path) -> Path:
    return create_private_stage_dir(path)


__all__ = [
    "PROJECT_ROOT",
    "PROTECTED_EXPERIMENT_ROOT",
    "PROTECTED_ROENTGEN_ROOT",
    "create_output_run",
    "create_private_dir",
    "enforce_private_file_mode",
    "read_private_json",
    "require_source_run",
    "sha256_file",
    "validate_case_id",
    "validate_run_id",
    "write_private_json",
]

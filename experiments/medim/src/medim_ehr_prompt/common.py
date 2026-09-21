"""Shared privacy and configuration helpers for the MeDiM experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
    write_private_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "medim"
PROTECTED_MODEL_ROOT = PROTECTED_ROOT / "medim"
PROTECTED_EXPERIMENT_ROOT = PROTECTED_MODEL_ROOT / "runs"
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


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("experiment config must be a mapping")
    if payload.get("schema_version") != "medim_ehr_prompt.config.v1":
        raise ValueError("unsupported experiment config")
    return payload


def create_run_directory(run_id: str) -> Path:
    validate_run_id(run_id)
    return create_private_stage_dir(PROTECTED_EXPERIMENT_ROOT / run_id)


def require_run_directory(run_id: str) -> Path:
    validate_run_id(run_id)
    resolved = require_inside(
        PROTECTED_EXPERIMENT_ROOT / run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    if not resolved.is_dir():
        raise ValueError("protected run path is not a directory")
    enforce_private_directory_mode(resolved)
    return resolved


def create_case_stage(run_dir: Path, case_id: str, stage: str) -> Path:
    validate_case_id(case_id)
    if RUN_ID_PATTERN.fullmatch(stage) is None:
        raise ValueError("invalid stage name")
    case_dir = require_inside(run_dir / "cases" / case_id, PROTECTED_ROOT, must_exist=True)
    if not case_dir.is_dir():
        raise ValueError("case path is not a directory")
    enforce_private_directory_mode(case_dir)
    return create_private_stage_dir(case_dir / stage)


def read_private_json(path: str | Path) -> dict[str, Any]:
    resolved = require_private_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


__all__ = [
    "EXPERIMENT_ROOT",
    "PROJECT_ROOT",
    "PROTECTED_EXPERIMENT_ROOT",
    "PROTECTED_MODEL_ROOT",
    "create_case_stage",
    "create_run_directory",
    "enforce_private_directory_mode",
    "enforce_private_file_mode",
    "load_config",
    "read_private_json",
    "require_private_file",
    "require_run_directory",
    "sha256_file",
    "validate_case_id",
    "validate_run_id",
    "write_private_json",
    "write_private_text",
]

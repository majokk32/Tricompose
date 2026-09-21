"""Small helpers that enforce the Phase-0 local privacy boundary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_ROOT = PROJECT_ROOT / "artifacts" / "protected"


def enforce_private_directory_mode(path: str | Path) -> None:
    """Set and verify owner-only directory permissions."""
    target = Path(path)
    os.chmod(target, 0o700)
    if stat.S_IMODE(target.stat().st_mode) != 0o700:
        raise PermissionError("private directory permissions are not 0700")


def enforce_private_file_mode(path: str | Path) -> None:
    """Set and verify owner-only file permissions."""
    target = Path(path)
    os.chmod(target, 0o600)
    if stat.S_IMODE(target.stat().st_mode) != 0o600:
        raise PermissionError("private file permissions are not 0600")


def require_inside(path: str | Path, root: str | Path, *, must_exist: bool) -> Path:
    """Resolve *path* and require it to remain below *root*."""
    resolved_root = Path(root).resolve(strict=True)
    resolved_path = Path(path).resolve(strict=must_exist)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("path is outside the allowed workspace boundary") from exc
    return resolved_path


def create_private_stage_dir(path: str | Path) -> Path:
    """Create a new non-overwriting stage directory below PROTECTED_ROOT."""
    old_umask = os.umask(0o077)
    try:
        PROTECTED_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        enforce_private_directory_mode(PROTECTED_ROOT)

        stage = require_inside(path, PROTECTED_ROOT, must_exist=False)
        stage.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        enforce_private_directory_mode(stage.parent)
        stage.mkdir(mode=0o700, exist_ok=False)
        enforce_private_directory_mode(stage)
    finally:
        os.umask(old_umask)
    return stage


def require_private_file(path: str | Path) -> Path:
    """Require an existing regular file below PROTECTED_ROOT."""
    resolved = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if not resolved.is_file():
        raise ValueError("protected input is not a regular file")
    return resolved


def _exclusive_private_fd(path: Path) -> int:
    require_inside(path, PROTECTED_ROOT, must_exist=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def write_private_text(path: str | Path, text: str) -> Path:
    """Write a new UTF-8 file with mode 0600 and no overwrite."""
    target = Path(path)
    fd = _exclusive_private_fd(target)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    enforce_private_file_mode(target)
    return target


def write_private_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic private JSON manifest."""
    return write_private_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local artifact without exposing its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

"""Slurm-guarded one-operation dispatch with private durable status journaling.

No model backend is installed here. Backend payloads are validated separately;
receipt status never grants authorization or clinical acceptance.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import stat
import time

from .execution_ledger import CallResult
from .invariant_verification import _ID

PROTECTED_ROOT = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected")


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm allocation required before dispatch")


def _private_parent(path):
    path = Path(path)
    parent = path.parent.resolve(strict=True)
    if (not parent.is_relative_to(PROTECTED_ROOT.resolve(strict=True))
            or not _ID.fullmatch(path.name) or stat.S_IMODE(parent.stat().st_mode) != 0o2770
            or parent.stat().st_gid not in (96293, 65534)):
        raise ValueError("private project-group output boundary required")
    return parent / path.name


def _new_private_handle(path):
    path = _private_parent(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o660)
    os.fchmod(fd, 0o660)
    return os.fdopen(fd, "w", encoding="utf-8")


class ProtectedJournal:
    """Exclusive new journal; flush/fsync each event before returning.

    Actual filesystem resume/reconciliation is deliberately not automatic.
    Process death can leave this journal in an uncommitted private run; retain
    it for reconciliation rather than treating the reserved call as free.
    """

    def __init__(self, path):
        require_slurm()
        self._handle = _new_private_handle(path)

    def append(self, event):
        self._handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self): self._handle.close()

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb): self.close()


def dispatch_operation(ledger, request, backend, validator, *, private_log_path):
    """Reserve → one backend call → validate → commit/fail (never auto-retry).

    Backend must declare `frozen`, `audit_sha256`, `execution_mode` and implement
    `invoke(request)`. The adapter must prove these claims against actual frozen
    model files. This guard doesn't submit a job or replace user approval.
    Standard Python stdout/stderr is protected; a deployed subprocess adapter
    must separately redirect child/native file descriptors as well.
    """
    require_slurm()
    if (backend.frozen is not True or backend.audit_sha256 != request.frozen_model_audit_sha256
            or backend.execution_mode != ledger.mode or not callable(validator)):
        raise ValueError("backend audit/mode differs from frozen call request")
    # Open the private log before reservation; no backend runs if this fails.
    with _new_private_handle(private_log_path) as handle:
        token = ledger.reserve(request)
        started = time.monotonic()
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                payload = backend.invoke(request)
            except TimeoutError:
                ledger.fail(token, error_code="timeout", retryable=True,
                    elapsed_seconds=time.monotonic()-started)
                return None
            except Exception:
                ledger.fail(token, error_code="runtime_exception", retryable=False,
                    elapsed_seconds=time.monotonic()-started)
                return None
            try:
                result = validator(payload, request)
                if (not isinstance(result, CallResult)
                        or (result.verification_receipt_id is not None) != (request.kind in {"xrv", "chexbert"})):
                    raise ValueError("validated CallResult required")
            except Exception:
                ledger.fail(token, error_code="invalid_result", retryable=False,
                    elapsed_seconds=time.monotonic()-started)
                return None
            # Journal failures are NOT model failures and must not trigger a
            # second backend invocation. Keep the pending reservation charged.
            ledger.complete(token, result, elapsed_seconds=time.monotonic()-started)
            return result

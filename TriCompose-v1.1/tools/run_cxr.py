#!/usr/bin/env python3
"""Run one frozen V1.1 CXR model from exact requests (Slurm only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for relative in (
    "TriCompose-v1.1/src",
    "TriCompose-v1.0/src",
    "src",
    "experiments/roentgen_v2/src",
    "experiments/chexgenbench_sana/src",
    "experiments/chexgenbench_pixart/src",
):
    sys.path.insert(0, str(WORKSPACE_ROOT / relative))

from tricompose_v11.cxr_execution import run_cxr_request_run  # noqa: E402


def _model_components(model_id: str, model_dir: str, precision: str | None):
    if model_id == "roentgen_v2":
        from tricompose_roentgen_v2.model import (
            OFFICIAL_MODEL_REVISION,
            FrozenRoentgenRuntime,
            validate_model_snapshot,
        )

        selected_precision = precision or "bfloat16"
        audit = {
            **validate_model_snapshot(model_dir),
            "runtime_precision": selected_precision,
        }

        def factory():
            runtime = FrozenRoentgenRuntime(
                model_dir, precision=selected_precision
            )
            runtime.audit = audit
            return runtime

        return (
            OFFICIAL_MODEL_REVISION,
            audit,
            factory,
        )
    if model_id == "chexgenbench_sana":
        if precision is not None:
            raise ValueError("--precision is only valid for RoentGen-v2")
        from tricompose_chexgenbench_sana.model import (
            OFFICIAL_MODEL_REVISION,
            FrozenSanaRuntime,
            validate_model_snapshot,
        )

        return (
            OFFICIAL_MODEL_REVISION,
            validate_model_snapshot(model_dir),
            lambda: FrozenSanaRuntime(model_dir),
        )
    if model_id == "chexgenbench_pixart":
        if precision is not None:
            raise ValueError("--precision is only valid for RoentGen-v2")
        from tricompose_chexgenbench_pixart.model import (
            OFFICIAL_MODEL_REVISION,
            FrozenPixArtRuntime,
            validate_model_snapshot,
        )

        return (
            OFFICIAL_MODEL_REVISION,
            validate_model_snapshot(model_dir),
            lambda: FrozenPixArtRuntime(model_dir),
        )
    raise ValueError("unsupported V1.1 CXR model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument(
        "--model-id",
        required=True,
        choices=("roentgen_v2", "chexgenbench_sana", "chexgenbench_pixart"),
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("float16", "bfloat16"))
    args = parser.parse_args()
    try:
        revision, audit, factory = _model_components(
            args.model_id, args.model_dir, args.precision
        )
        result = run_cxr_request_run(
            request_run=args.request_run,
            output_root=args.output_root,
            output_run_id=args.output_run_id,
            model_id=args.model_id,
            model_revision=revision,
            model_audit=audit,
            runtime_factory=factory,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

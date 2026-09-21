#!/usr/bin/env python3
"""Run one frozen V1 CXR generator from exact protected requests (Slurm only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "src",
    "TriCompose-v1.0/src",
    "experiments/roentgen_v2/src",
    "experiments/chexgenbench_sana/src",
    "experiments/chexgenbench_pixart/src",
):
    sys.path.insert(0, str(WORKSPACE_ROOT / relative))

# xhuang90 local protected root override for TriCompose-v1.0 runs.
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0/runtime_xhuang90/src"))

from tricompose_v1.execution import run_cxr_request_run  # noqa: E402


def _model_components(model_id: str, model_dir: str):
    if model_id == "roentgen_v2":
        from tricompose_roentgen_v2.model import (
            OFFICIAL_MODEL_REVISION,
            FrozenRoentgenRuntime,
            validate_model_snapshot,
        )

        return (
            OFFICIAL_MODEL_REVISION,
            validate_model_snapshot(model_dir),
            lambda: FrozenRoentgenRuntime(model_dir),
        )
    if model_id == "chexgenbench_sana":
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
    raise ValueError("unsupported V1 CXR model")


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
    args = parser.parse_args()
    try:
        revision, audit, factory = _model_components(args.model_id, args.model_dir)
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
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

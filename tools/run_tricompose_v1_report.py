#!/usr/bin/env python3
"""Run one frozen V1 CXR-to-report model from common candidates (Slurm only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "src",
    "TriCompose-v1.0/src",
    "experiments/maira2_report/src",
    "experiments/cxrmate_single_report/src",
    "experiments/llavarad_report/src",
    "CheXagent-2/src",
):
    sys.path.insert(0, str(WORKSPACE_ROOT / relative))

# xhuang90 local protected root override for TriCompose-v1.0 runs.
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0/runtime_xhuang90/src"))

from tricompose_v1.execution import run_report_candidate_run  # noqa: E402


def _required(value: str | None, option: str) -> str:
    if not value:
        raise ValueError(f"{option} is required for the selected report model")
    return value


def _model_components(args):
    model_id = args.model_id
    model_dir = args.model_dir
    if model_id == "maira2":
        from tricompose_maira2_report.model import (
            MODEL_REVISION,
            FrozenMaira2Runtime,
            validate_model_snapshot,
        )

        return (
            MODEL_REVISION,
            validate_model_snapshot(model_dir),
            lambda: FrozenMaira2Runtime(model_dir),
        )
    if model_id == "cxrmate_single":
        from tricompose_cxrmate_single_report.model import (
            MODEL_REVISION,
            FrozenCXRMateSingleRuntime,
            validate_model_snapshot,
        )

        return (
            MODEL_REVISION,
            validate_model_snapshot(model_dir),
            lambda: FrozenCXRMateSingleRuntime(model_dir),
        )
    if model_id == "llavarad":
        from tricompose_llavarad_report.model import (
            MODEL_REVISION,
            FrozenLlavaRadRuntime,
            validate_model_snapshot,
        )

        model_base = _required(args.model_base, "--model-base")
        biomedbert_dir = _required(args.biomedbert_dir, "--biomedbert-dir")
        external_root = _required(args.external_root, "--external-root")
        runtime_dir = _required(args.runtime_dir, "--runtime-dir")
        return (
            MODEL_REVISION,
            validate_model_snapshot(
                model_dir=model_dir,
                model_base=model_base,
                biomedbert_dir=biomedbert_dir,
                external_root=external_root,
            ),
            lambda: FrozenLlavaRadRuntime(
                model_dir=model_dir,
                model_base=model_base,
                biomedbert_dir=biomedbert_dir,
                external_root=external_root,
                runtime_dir=runtime_dir,
            ),
        )
    if model_id == "chexagent2":
        from tricompose_chexagent2_report.model import (
            MODEL_REVISION,
            FrozenCheXagent2Runtime,
            validate_model_snapshot,
            validate_vision_snapshot,
        )

        vision_dir = _required(args.vision_dir, "--vision-dir")
        audit = {
            "model": validate_model_snapshot(model_dir),
            "vision": validate_vision_snapshot(vision_dir),
        }
        return (
            MODEL_REVISION,
            audit,
            lambda: FrozenCheXagent2Runtime(
                model_dir=model_dir,
                vision_dir=vision_dir,
            ),
        )
    raise ValueError("V1 report runtime is not connected for this model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cxr-run",
        required=True,
        action="append",
        help="Protected common CXR candidate run; repeat for a candidate bank.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument(
        "--model-id",
        required=True,
        choices=("maira2", "cxrmate_single", "llavarad", "chexagent2"),
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-base")
    parser.add_argument("--biomedbert-dir")
    parser.add_argument("--external-root")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--vision-dir")
    args = parser.parse_args()
    try:
        revision, audit, factory = _model_components(args)
        result = run_report_candidate_run(
            cxr_run=args.cxr_run,
            output_root=args.output_root,
            output_run_id=args.output_run_id,
            model_id=args.model_id,
            model_revision=revision,
            model_audit=audit,
            runtime_factory=factory,
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

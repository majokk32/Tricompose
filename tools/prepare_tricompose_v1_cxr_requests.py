#!/usr/bin/env python3
"""Prepare exact protected CXR requests; this command is CPU-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0" / "src"))

from tricompose_v1.execution import prepare_cxr_request_run  # noqa: E402
from tricompose_v1.prompts import ACTIVE_PROMPT_MODELS  # noqa: E402


def _integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-run", required=True)
    parser.add_argument("--case-ids-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default=",".join(ACTIVE_PROMPT_MODELS))
    parser.add_argument("--seeds", type=_integers, default=(0, 1))
    parser.add_argument("--allow-underconditioned", action="store_true")
    args = parser.parse_args()
    result = prepare_cxr_request_run(
        staging_run=args.staging_run,
        case_ids_file=args.case_ids_file,
        output_root=args.output_root,
        run_id=args.run_id,
        model_ids=tuple(item.strip() for item in args.models.split(",") if item.strip()),
        seeds=args.seeds,
        require_conditioned=not args.allow_underconditioned,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a protected TriCompose V1 staging run; never invokes a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0" / "src"))

from tricompose_v1.prompts import ACTIVE_PROMPT_MODELS  # noqa: E402
from tricompose_v1.staging import stage_tricompose_v1  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage canonical synthetic EHR, evidence-grounded facts, and exact "
            "model prompts below artifacts/protected. This command is CPU-only."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--case-ids-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--prompt-models",
        default=",".join(ACTIVE_PROMPT_MODELS),
        help="Comma-separated active cold-start text-to-CXR prompt models.",
    )
    parser.add_argument("--validate", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = stage_tricompose_v1(
        source_root=args.source_root,
        case_ids_file=args.case_ids_file,
        output_root=args.output_root,
        run_id=args.run_id,
        prompt_models=args.prompt_models,
        validate=args.validate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (not args.validate or summary["overall_valid"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

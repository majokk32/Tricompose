#!/usr/bin/env python3
"""Prepare exact protected V1.1 CXR-to-report requests without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0" / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from tricompose_v11.report_contracts import (  # noqa: E402
    ACTIVE_REPORT_MODELS_V11,
    prepare_report_request_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cxr-run",
        required=True,
        action="append",
        help="Protected V1.1 CXR candidate run; repeat to combine runs.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default=",".join(ACTIVE_REPORT_MODELS_V11))
    args = parser.parse_args()
    result = prepare_report_request_run(
        cxr_runs=args.cxr_run,
        output_root=args.output_root,
        run_id=args.run_id,
        model_ids=tuple(
            item.strip() for item in args.models.split(",") if item.strip()
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

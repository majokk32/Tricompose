"""Generate protected reports from frozen synthetic CXRs with CXRMate-single."""

from __future__ import annotations

import argparse
import json
import sys

from tricompose.reporting import ReportModelSpec, run_report_generation

from .model import (
    MODEL_REVISION,
    FrozenCXRMateSingleRuntime,
    validate_model_snapshot,
)


SPEC = ReportModelSpec(
    name="cxrmate_single_tf",
    artifact_namespace="cxrmate_single_report",
    revision=MODEL_REVISION,
    run_schema="tricompose.cxrmate_single_report.run.v1",
    generation_schema="tricompose.cxrmate_single_report.generation.v1",
    frozen_schema="tricompose.cxrmate_single_report.frozen_run.v1",
    output_sections=("findings", "impression"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    try:
        audit = validate_model_snapshot(args.model_dir)
        result = run_report_generation(
            spec=SPEC,
            runtime_factory=FrozenCXRMateSingleRuntime,
            model_dir=args.model_dir,
            model_audit=audit,
            source_model=args.source_model,
            source_run_id=args.source_run_id,
            output_run_id=args.output_run_id,
            limit=args.limit,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


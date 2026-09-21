#!/usr/bin/env python3
"""Build V1.1 facts/prompts from existing canonical V1 EHRs; CPU only."""

from __future__ import annotations

import argparse
import json

from tricompose_v11.staging import stage_existing_v1_ehrs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v1-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = stage_existing_v1_ehrs(
        source_v1_run=args.source_v1_run,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["overall_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

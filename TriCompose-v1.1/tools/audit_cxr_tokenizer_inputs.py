#!/usr/bin/env python3
"""CPU-only reconstruction sidecar; never claim a retrospective runtime capture."""
import argparse
import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parents[2]
for relative in ("src", "TriCompose-v1.0/src", "TriCompose-v1.1/src"):
    sys.path.insert(0, str(workspace / relative))
from tricompose_v11.tokenizer_audit import build_tokenizer_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = build_tokenizer_audit(cxr_runs=args.cxr_run, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

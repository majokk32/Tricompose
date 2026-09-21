#!/usr/bin/env python3
"""Atomically export one complete selected TriCompose V1 synthetic triple."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0" / "src"))

from tricompose.privacy import require_private_file  # noqa: E402
from tricompose_v1.contracts import export_selected_triple  # noqa: E402


def _read_protected_json(path: str) -> dict:
    source = require_private_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON input must be an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-run", required=True)
    parser.add_argument("--cxr-candidate-json", required=True)
    parser.add_argument("--report-candidate-json", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-id", required=True)
    args = parser.parse_args()
    result = export_selected_triple(
        staging_run=args.staging_run,
        cxr_candidate=_read_protected_json(args.cxr_candidate_json),
        report_candidate=_read_protected_json(args.report_candidate_json),
        selection=_read_protected_json(args.selection_json),
        output_root=args.output_root,
        export_id=args.export_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

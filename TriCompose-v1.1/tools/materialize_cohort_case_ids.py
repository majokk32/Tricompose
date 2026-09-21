#!/usr/bin/env python3
"""Materialize a fixed opaque case list from a protected V1.1 cohort."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for relative in ("TriCompose-v1.1/src", "TriCompose-v1.0/src", "src"):
    sys.path.insert(0, str(WORKSPACE_ROOT / relative))

from tricompose_v11.cxr_contracts import (  # noqa: E402
    CASE_ID_PATTERN,
    MAIN_PROTECTED_ROOT,
    RUN_ID_PATTERN,
    private_directory,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    cohort = require_inside(args.cohort_jsonl, MAIN_PROTECTED_ROOT, must_exist=True)
    rows = [json.loads(line) for line in cohort.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("protected cohort JSONL is empty or invalid")
    case_ids = [row.get("case_id") for row in rows]
    if any(
        not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id)
        for case_id in case_ids
    ):
        raise ValueError("cohort contains a non-opaque case ID")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cohort contains duplicate case IDs")
    if any(row.get("status") != "staged" for row in rows):
        raise ValueError("cohort contains a case that was not staged")

    output = require_inside(args.output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    target = output / args.run_id
    if target.exists():
        raise FileExistsError("cohort case-list run already exists")
    temporary = output / f".{args.run_id}.{uuid.uuid4().hex}.tmp"
    private_directory(temporary)
    try:
        case_list = write_private_text(
            temporary / "case_ids.txt", "".join(f"{case_id}\n" for case_id in case_ids)
        )
        manifest = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": "tricompose-v1.1-fixed-cohort-v1",
                "run_id": args.run_id,
                "source_cohort_jsonl": {
                    "path": str(cohort),
                    "sha256": sha256_file(cohort),
                },
                "case_count": len(case_ids),
                "case_ids_sha256": sha256_file(case_list),
                "opaque_case_ids_only": True,
            },
        )
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(
        json.dumps(
            {
                "status": "prepared",
                "run_directory": str(target),
                "case_count": len(case_ids),
                "manifest_sha256": sha256_file(target / manifest.name),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

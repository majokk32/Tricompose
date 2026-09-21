#!/usr/bin/env python3
"""Export a deterministic no-scoring fixed-path baseline from a V1 bank."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "TriCompose-v1.0" / "src"))

from tricompose.privacy import (  # noqa: E402
    enforce_private_directory_mode,
    sha256_file,
    write_private_json,
)
from tricompose_v1.contracts import export_selected_triple  # noqa: E402
from tricompose_v1.scoring import (  # noqa: E402
    commit_atomic_protected_run,
    discard_atomic_protected_run,
    load_candidate_bank,
    new_atomic_protected_run,
)


BASELINE_SCHEMA = "tricompose-v1-fixed-greedy-baseline-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cxr-model", default="roentgen_v2")
    parser.add_argument("--cxr-seed", type=int, default=0)
    parser.add_argument("--report-model", default="maira2")
    return parser


def run(args: argparse.Namespace, temp: Path) -> dict:
    if args.cxr_seed < 0:
        raise ValueError("CXR seed must be non-negative")
    bank = load_candidate_bank(args.candidate_bank)
    triples_root = temp / "triples"
    records: list[dict] = []
    for ehr in sorted(bank.ehr_candidates, key=lambda row: str(row["candidate_id"])):
        ehr_id = str(ehr["candidate_id"])
        cxr_matches = [
            row
            for row in bank.cxr_candidates
            if row["parent_ids"] == [ehr_id]
            and row["model_id"] == args.cxr_model
            and row["seed"] == args.cxr_seed
        ]
        if len(cxr_matches) != 1:
            raise ValueError("fixed CXR baseline path is missing or ambiguous")
        cxr = cxr_matches[0]
        report_matches = [
            row
            for row in bank.report_candidates
            if row["parent_ids"] == [ehr_id, cxr["candidate_id"]]
            and row["model_id"] == args.report_model
        ]
        if len(report_matches) != 1:
            raise ValueError("fixed report baseline path is missing or ambiguous")
        report = report_matches[0]
        selection = {
            "action": "stop_and_select",
            "selected": {
                "ehr_candidate_id": ehr_id,
                "cxr_candidate_id": cxr["candidate_id"],
                "report_candidate_id": report["candidate_id"],
            },
            "selection_provenance": {
                "method": "fixed_greedy_registry_path_v1",
                "scoring_used": False,
                "post_hoc_output_inspection_used": False,
                "rule": (
                    "predeclared CXR model + smallest configured seed + "
                    "predeclared report model"
                ),
            },
        }
        result = export_selected_triple(
            staging_run=bank.staging_root,
            cxr_candidate=cxr,
            report_candidate=report,
            selection=selection,
            output_root=triples_root,
            export_id=str(ehr["case_id"]),
        )
        triple_dir = Path(result["export_directory"])
        manifest_path = triple_dir / "triple_manifest.json"
        records.append(
            {
                "ehr_candidate_id": ehr_id,
                "cxr_candidate_id": cxr["candidate_id"],
                "report_candidate_id": report["candidate_id"],
                "path": str(triple_dir.relative_to(temp)),
                "triple_manifest_sha256": sha256_file(manifest_path),
            }
        )

    return {
        "schema_version": BASELINE_SCHEMA,
        "run_id": args.run_id,
        "status": "completed_fixed_baseline_no_scoring",
        "baseline_name": "fixed_greedy_registry_path",
        "scientific_role": "ordinary fixed-pipeline baseline; not an optimum",
        "scoring_used": False,
        "post_hoc_output_inspection_used": False,
        "source_candidate_bank": {
            "path": str(bank.root),
            "manifest_sha256": bank.manifest_sha256,
        },
        "fixed_path": {
            "cxr_model": args.cxr_model,
            "cxr_seed": args.cxr_seed,
            "report_model": args.report_model,
        },
        "triple_count": len(records),
        "triples": records,
        "mutation_policy": "immutable_no_overwrite",
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o077)
    started = time.monotonic()
    temp, target = new_atomic_protected_run(args.output_root, args.run_id)
    try:
        payload = run(args, temp)
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        manifest_path = write_private_json(temp / "manifest.json", payload)
        manifest_hash = sha256_file(manifest_path)
        commit_atomic_protected_run(temp, target)
        enforce_private_directory_mode(target)
    except Exception:
        discard_atomic_protected_run(temp)
        raise
    print(
        json.dumps(
            {
                "stage": "tricompose_v1_fixed_greedy_baseline",
                "status": "ok",
                "run_id": args.run_id,
                "triple_count": payload["triple_count"],
                "manifest_sha256": manifest_hash,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

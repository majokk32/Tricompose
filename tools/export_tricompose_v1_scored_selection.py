#!/usr/bin/env python3
"""Export every per-EHR stop_and_select result from a V1 score run."""

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

from tricompose.privacy import require_private_file, sha256_file, write_private_json  # noqa: E402
from tricompose_v1.contracts import export_selected_triple  # noqa: E402
from tricompose_v1.scoring import (  # noqa: E402
    AGGREGATE_SCHEMA,
    commit_atomic_protected_run,
    discard_atomic_protected_run,
    load_candidate_bank,
    new_atomic_protected_run,
)


EXPORT_RUN_SCHEMA = "tricompose-v1-scored-selection-export-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _read_json(path: Path) -> dict:
    source = require_private_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("selection artifact must be an object")
    return payload


def run(args: argparse.Namespace, temp: Path) -> dict:
    bank = load_candidate_bank(args.candidate_bank)
    selection_path = require_private_file(
        Path(args.selection_run).resolve(strict=True) / "scores.json"
    )
    selection_bundle = _read_json(selection_path)
    if selection_bundle.get("schema_version") != AGGREGATE_SCHEMA:
        raise ValueError("unsupported selection schema")
    source = selection_bundle.get("source_candidate_bank")
    if not isinstance(source, dict) or source.get("manifest_sha256") != bank.manifest_sha256:
        raise ValueError("selection and candidate bank do not match")
    if selection_bundle.get("calibration_status") != "uncalibrated_engineering_smoke":
        raise ValueError("unexpected selection calibration status")

    cxr_by_id = {str(row["candidate_id"]): row for row in bank.cxr_candidates}
    report_by_id = {
        str(row["candidate_id"]): row for row in bank.report_candidates
    }
    case_results = selection_bundle.get("case_results")
    if not isinstance(case_results, list) or len(case_results) != len(bank.ehr_candidates):
        raise ValueError("selection does not cover every fixed EHR")

    triples_root = temp / "triples"
    exported: list[dict] = []
    for row in sorted(case_results, key=lambda item: str(item["ehr_candidate_id"])):
        decision = row.get("decision")
        if not isinstance(decision, dict) or decision.get("action") != "stop_and_select":
            raise ValueError("a fixed EHR has no accepted scored selection")
        selected = decision.get("selected")
        if not isinstance(selected, dict):
            raise ValueError("accepted decision lacks selected candidate IDs")
        ehr_id = str(selected["ehr_candidate_id"])
        cxr = cxr_by_id[str(selected["cxr_candidate_id"])]
        report = report_by_id[str(selected["report_candidate_id"])]
        if cxr["parent_ids"] != [ehr_id] or report["parent_ids"] != [
            ehr_id,
            cxr["candidate_id"],
        ]:
            raise ValueError("selected triple lineage is inconsistent")

        export_selection = {
            **decision,
            "selection_provenance": {
                "method": "tricompose_v1_static_best_of_n",
                "scoring_used": True,
                "calibration_status": selection_bundle["calibration_status"],
                "policy_id": selection_bundle["policy"]["policy_id"],
                "selection_scores_sha256": sha256_file(selection_path),
            },
        }
        result = export_selected_triple(
            staging_run=bank.staging_root,
            cxr_candidate=cxr,
            report_candidate=report,
            selection=export_selection,
            output_root=triples_root,
            export_id=str(cxr["case_id"]),
        )
        triple_dir = Path(result["export_directory"])
        exported.append(
            {
                "ehr_candidate_id": ehr_id,
                "cxr_candidate_id": cxr["candidate_id"],
                "report_candidate_id": report["candidate_id"],
                "cxr_model": cxr["model_id"],
                "cxr_seed": cxr["seed"],
                "report_model": report["model_id"],
                "global_score": selected["global_score"],
                "path": str(triple_dir.relative_to(temp)),
                "triple_manifest_sha256": result["triple_manifest_sha256"],
            }
        )

    return {
        "schema_version": EXPORT_RUN_SCHEMA,
        "run_id": args.run_id,
        "status": "completed_scored_selection_export",
        "scientific_role": "uncalibrated V1 static best-of-N accepted by engineering policy",
        "source_candidate_bank": {
            "path": str(bank.root),
            "manifest_sha256": bank.manifest_sha256,
        },
        "source_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "triple_count": len(exported),
        "triples": exported,
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
    except Exception:
        discard_atomic_protected_run(temp)
        raise
    print(
        json.dumps(
            {
                "stage": "tricompose_v1_scored_selection_export",
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

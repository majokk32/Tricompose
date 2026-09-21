#!/usr/bin/env python3
"""Merge frozen cross-modal evidence into the V1.1 candidate score table."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from contracts import (
    PROTECTED_ROOT,
    commit_atomic_run,
    discard_atomic_run,
    new_atomic_run,
    read_json,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)


SCHEMA_VERSION = "tricompose-unified-score-table-v1.1"
ROW_SCHEMA_VERSION = "tricompose-unified-score-row-v1.1"
CROSSMODAL_SCHEMA = "tricompose-report-crossmodal-evaluation-v1.1"
POLICY_SCHEMA = "tricompose-static-score-policy-v1.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table-run", required=True)
    parser.add_argument("--crossmodal-run", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row {line_number} is not an object")
            rows.append(row)
    return rows


def _load_policy(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve(strict=True)
    workspace = Path("/project2/ruishanl_1185/inference_3mod").resolve(strict=True)
    if not source.is_relative_to(workspace) or not source.is_file():
        raise ValueError("policy config must be a workspace file")
    policy = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported static score policy")
    edge_weights = policy.get("edge_weights")
    if not isinstance(edge_weights, dict) or set(edge_weights) != {
        "ehr_cxr",
        "ehr_report",
        "report_cxr",
    }:
        raise ValueError("policy must define exactly three edge weights")
    if any(float(value) < 0 for value in edge_weights.values()):
        raise ValueError("edge weights must be non-negative")
    if policy.get("secondary_scores_in_total") != [] or policy.get("cost_in_total") is not False:
        raise ValueError("V1.1 static baseline cannot hide secondary scores or cost in its total")
    return policy, {"path": str(source), "sha256": sha256_file(source)}


def _edge_payload(edge: str, metrics: dict[str, Any]) -> dict[str, Any]:
    required = {
        "known_reference_fact_count",
        "comparable_explicit_fact_count",
        "coverage",
        "support_count",
        "support_recall",
        "contradiction_count",
        "contradiction_rate",
        "agreement_on_comparable",
        "edge_score",
    }
    if set(metrics) != required:
        raise ValueError("candidate edge metrics do not match the registered schema")
    if int(metrics["known_reference_fact_count"]) == 0:
        status = (
            "not_applicable_no_comparable_ehr_facts"
            if edge in {"ehr_cxr", "ehr_report"}
            else "not_applicable_no_classifier_reference_facts"
        )
    else:
        status = "computed_frozen_finding_evidence"
    return {"status": status, **metrics}


def _quality_score(row: dict[str, Any], policy: dict[str, Any]) -> float:
    report = row["modality_quality"]["report"]
    penalties = policy["report_quality_penalties"]
    value = 1.0
    value -= float(penalties["generic_report"]) * int(report["generic_report"])
    value -= float(penalties["unsupported_temporal_comparison_language"]) * int(
        report["unsupported_temporal_comparison_language"]
    )
    value -= float(penalties["repeated_4gram_ratio"]) * float(
        report["repeated_4gram_ratio"]
    )
    return round(min(1.0, max(0.0, value)), 8)


def _static_score(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    weighted = 0.0
    available_weight = 0.0
    utilities: dict[str, float | None] = {}
    for edge, weight_value in policy["edge_weights"].items():
        edge_score = row["cross_modal"][edge]["edge_score"]
        utility = None if edge_score is None else (float(edge_score) + 1.0) / 2.0
        utilities[edge] = None if utility is None else round(utility, 8)
        if utility is not None:
            weight = float(weight_value)
            weighted += weight * utility
            available_weight += weight
    clinical = None if available_weight == 0 else weighted / available_weight
    quality = _quality_score(row, policy)
    total = None
    if clinical is not None:
        total = (
            float(policy["clinical_weight"]) * clinical
            + float(policy["report_quality_weight"]) * quality
        )
    return {
        "policy_id": policy["policy_id"],
        "edge_utilities": utilities,
        "available_edge_weight": round(available_weight, 8),
        "clinical_consistency_score": None if clinical is None else round(clinical, 8),
        "report_quality_score": quality,
        "total_score": None if total is None else round(total, 8),
        "qwen_and_biovil_excluded": True,
        "cost_excluded": True,
    }


def _csv(rows: list[dict[str, Any]]) -> str:
    fields = (
        "triple_candidate_id",
        "case_id",
        "cxr_candidate_id",
        "cxr_model_id",
        "report_model_id",
        "ehr_cxr_score",
        "ehr_report_score",
        "report_cxr_score",
        "report_expert_disagreement",
        "cxr_candidate_disagreement",
        "qwenvl_score",
        "biovil_cosine",
        "report_quality_score",
        "static_total_score",
        "diagnostic_selection_eligible",
        "paper_primary_selection_eligible",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        lineage = row["lineage"]
        writer.writerow(
            {
                "triple_candidate_id": row["triple_candidate_id"],
                "case_id": row["case_id"],
                "cxr_candidate_id": lineage["cxr_candidate_id"],
                "cxr_model_id": lineage["cxr_model_id"],
                "report_model_id": lineage["report_model_id"],
                "ehr_cxr_score": row["cross_modal"]["ehr_cxr"]["edge_score"],
                "ehr_report_score": row["cross_modal"]["ehr_report"]["edge_score"],
                "report_cxr_score": row["cross_modal"]["report_cxr"]["edge_score"],
                "report_expert_disagreement": row["uncertainty"]["report_expert_disagreement"]["disagreement_rate"],
                "cxr_candidate_disagreement": row["uncertainty"]["cxr_candidate_disagreement"]["disagreement_rate"],
                "qwenvl_score": row["secondary_scores"]["qwenvl_report_cxr"]["value"],
                "biovil_cosine": row["secondary_scores"]["biovil_report_cxr"]["raw_cosine"],
                "report_quality_score": row["static_scoring"]["report_quality_score"],
                "static_total_score": row["static_scoring"]["total_score"],
                "diagnostic_selection_eligible": row["selection"]["diagnostic_eligible"],
                "paper_primary_selection_eligible": row["selection"]["paper_primary_eligible"],
            }
        )
    return output.getvalue()


def _markdown(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    return "\n".join(
        [
            "# TriCompose V1.1 Final Unified Score Table",
            "",
            f"- Complete lineages: {counts['rows']}",
            f"- Cases: {counts['cases']}",
            f"- Diagnostic-selection eligible rows: {counts['diagnostic_eligible_rows']}",
            f"- Paper-primary eligible rows: {counts['paper_primary_eligible_rows']}",
            "",
            "The table contains all three primary finding edges: EHR-CXR, EHR-report, and report-CXR.",
            "Unknown EHR facts do not become negative findings. Missing edges are omitted and remaining edge weights are renormalized.",
            "Qwen2.5-VL and BioViL remain separate secondary columns. They are not part of the static total score.",
            "If the CXR threshold bundle is uncalibrated, the table is diagnostic-selection ready but not paper-primary ready.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    skeleton_run = require_inside(args.score_table_run, PROTECTED_ROOT, must_exist=True)
    crossmodal_run = require_inside(args.crossmodal_run, PROTECTED_ROOT, must_exist=True)
    skeleton_summary = read_json(skeleton_run / "score_table_summary.json")
    crossmodal = read_json(crossmodal_run / "crossmodal_details.json")
    if skeleton_summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported score table skeleton")
    if skeleton_summary.get("selection_ready") is not False:
        raise ValueError("input score table is not a pending skeleton")
    if crossmodal.get("schema_version") != CROSSMODAL_SCHEMA:
        raise ValueError("unsupported cross-modal evidence")
    policy, policy_source = _load_policy(args.policy_config)
    skeleton_rows = _read_jsonl(skeleton_run / "score_table.jsonl")
    evidence_rows = crossmodal.get("records")
    if not isinstance(evidence_rows, list):
        raise TypeError("cross-modal evidence lacks candidate records")
    evidence_by_id = {
        str(row["report_candidate_id"]): row
        for row in evidence_rows
        if isinstance(row, dict)
    }
    if len(evidence_by_id) != len(evidence_rows):
        raise ValueError("cross-modal report IDs are duplicated")
    if {str(row["triple_candidate_id"]) for row in skeleton_rows} != set(evidence_by_id):
        raise ValueError("score skeleton and cross-modal evidence do not cover the same candidates")

    primary_eligible = crossmodal.get("primary_metric_status") == "eligible"
    rows: list[dict[str, Any]] = []
    for row in skeleton_rows:
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError("unsupported unified score row")
        evidence = evidence_by_id[str(row["triple_candidate_id"])]
        lineage = row["lineage"]
        if evidence["parent_cxr_candidate_id"] != lineage["cxr_candidate_id"]:
            raise ValueError("cross-modal CXR lineage mismatch")
        if evidence["report_sha256"] != lineage["report_sha256"]:
            raise ValueError("cross-modal report hash mismatch")
        if evidence["image_sha256"] != lineage["cxr_sha256"]:
            raise ValueError("cross-modal CXR hash mismatch")
        row["cross_modal"] = {
            edge: _edge_payload(edge, evidence["edge_metrics"][edge])
            for edge in ("ehr_cxr", "ehr_report", "report_cxr")
        }
        qwen = evidence["secondary_scores"]["qwenvl_report_cxr"]
        biovil = evidence["secondary_scores"]["biovil_report_cxr"]
        row["secondary_scores"] = {
            "qwenvl_report_cxr": {
                "status": "not_available" if qwen is None else "computed_secondary_uncalibrated",
                "value": qwen,
                "calibrated": False,
            },
            "biovil_report_cxr": {
                "status": "not_available" if biovil is None else "computed_secondary_uncalibrated",
                "raw_cosine": biovil,
                "calibrated": False,
            },
        }
        row["uncertainty"] = {
            "status": "computed_from_frozen_finding_vectors",
            **evidence["uncertainty"],
        }
        row["static_scoring"] = _static_score(row, policy)
        hard_failures = row["selection"]["hard_gate_failures"]
        diagnostic = not hard_failures and row["static_scoring"]["total_score"] is not None
        row["selection"] = {
            **row["selection"],
            "eligible": diagnostic and primary_eligible,
            "status": (
                "paper_primary_eligible"
                if diagnostic and primary_eligible
                else "diagnostic_only_uncalibrated_cxr_labels"
                if diagnostic
                else "ineligible"
            ),
            "diagnostic_eligible": diagnostic,
            "paper_primary_eligible": diagnostic and primary_eligible,
        }
        rows.append(row)

    if len(rows) != 960 or len({str(row["case_id"]) for row in rows}) != 80:
        raise ValueError("final score table must preserve the complete 80x12 grid")
    scores = [row["static_scoring"]["total_score"] for row in rows]
    valid_scores = [float(value) for value in scores if value is not None]
    applicability = {
        edge: {
            "applicable_rows": sum(
                row["cross_modal"][edge]["known_reference_fact_count"] > 0
                for row in rows
            ),
            "not_applicable_rows": sum(
                row["cross_modal"][edge]["known_reference_fact_count"] == 0
                for row in rows
            ),
        }
        for edge in ("ehr_cxr", "ehr_report", "report_cxr")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "table_status": "complete_frozen_evidence",
        "diagnostic_selection_ready": all(row["selection"]["diagnostic_eligible"] for row in rows),
        "paper_primary_selection_ready": all(row["selection"]["paper_primary_eligible"] for row in rows),
        "primary_metric_status": crossmodal["primary_metric_status"],
        "counts": {
            "rows": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "diagnostic_eligible_rows": sum(row["selection"]["diagnostic_eligible"] for row in rows),
            "paper_primary_eligible_rows": sum(row["selection"]["paper_primary_eligible"] for row in rows),
            "rows_with_hard_gate_failure": sum(bool(row["selection"]["hard_gate_failures"]) for row in rows),
        },
        "score_distribution": {
            "valid_count": len(valid_scores),
            "mean": None if not valid_scores else round(statistics.fmean(valid_scores), 8),
            "minimum": None if not valid_scores else round(min(valid_scores), 8),
            "maximum": None if not valid_scores else round(max(valid_scores), 8),
        },
        "edge_applicability": applicability,
        "policy": policy,
        "sources": {
            "skeleton": {"path": str(skeleton_run), "manifest_sha256": sha256_file(skeleton_run / "manifest.json")},
            "crossmodal": {"path": str(crossmodal_run), "details_sha256": sha256_file(crossmodal_run / "crossmodal_details.json")},
            "policy": policy_source,
        },
        "model_grid": skeleton_summary["model_grid"],
        "rows": rows,
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o007)
    started = time.monotonic()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        table = write_private_text(
            temporary / "score_table.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["rows"]),
        )
        csv_path = write_private_text(temporary / "score_table.csv", _csv(payload["rows"]))
        summary_payload = {key: value for key, value in payload.items() if key != "rows"}
        summary = write_private_json(temporary / "score_table_summary.json", summary_payload)
        markdown = write_private_text(temporary / "score_table_summary.md", _markdown(payload))
        manifest = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "table_status": payload["table_status"],
                "counts": payload["counts"],
                "artifacts": {
                    path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                    for path in (table, csv_path, summary, markdown)
                },
            },
        )
        manifest_hash = sha256_file(manifest)
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "completed_final_score_table",
                "run_id": args.run_id,
                "rows": payload["counts"]["rows"],
                "diagnostic_selection_ready": payload["diagnostic_selection_ready"],
                "paper_primary_selection_ready": payload["paper_primary_selection_ready"],
                "manifest_sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

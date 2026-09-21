#!/usr/bin/env python3
"""Build and select the 960 TriCompose candidates with edge-specific evidence.

This lightweight, training-free stage replaces the legacy symmetric EHR-edge
scores with the revised EHR-CXR and EHR-report contracts.  It never reads
report text, image pixels, or canonical EHR rows; it consumes only protected,
hash-bound derived metrics and a protected candidate registry.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
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
from crossmodal_metrics import score_state_pair


TABLE_SCHEMA = "tricompose-unified-score-table-v1.1"
EHR_EDGE_SCHEMA = "tricompose-ehr-edge-crossmodal-evaluation-v1.1"
POLICY_SCHEMA = "tricompose-lexicographic-selection-policy-v1.1"
SCHEMA_VERSION = "tricompose-edge-specific-selection-v1.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table-run", required=True)
    parser.add_argument("--ehr-edge-run", required=True)
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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def _load_policy(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = Path("/project2/ruishanl_1185/inference_3mod").resolve(strict=True)
    source = Path(path).resolve(strict=True)
    if not source.is_relative_to(workspace) or not source.is_file():
        raise ValueError("selection policy must be a workspace file")
    policy = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported lexicographic selection policy")
    expected_order = [
        ("hard_gate_failure_count", "ascending"),
        ("total_hard_contradiction_count", "ascending"),
        ("ehr_direct_support_count", "descending"),
        ("report_cxr_support_recall", "descending"),
        ("report_structure_quality_score", "descending"),
        ("known_runtime_seconds", "ascending"),
        ("triple_candidate_id", "ascending"),
    ]
    actual_order = [
        (str(row.get("field")), str(row.get("direction")))
        for row in policy.get("selection_order", [])
        if isinstance(row, dict)
    ]
    if actual_order != expected_order:
        raise ValueError("policy selection order does not match the registered contract")
    fixed = policy.get("operational_fixed_baseline")
    if not isinstance(fixed, dict) or not fixed.get("cxr_model_id") or not fixed.get(
        "report_model_id"
    ):
        raise ValueError("policy lacks the operational fixed baseline")
    diagnostic = policy.get("diagnostic_score")
    if not isinstance(diagnostic, dict) or diagnostic.get("used_for_selection") is not False:
        raise ValueError("diagnostic score must be explicitly excluded from selection")
    return policy, {"path": str(source), "sha256": sha256_file(source)}


def _normalized_edge(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "known_reference_fact_count",
        "comparable_explicit_fact_count",
        "coverage",
        "support_count",
        "support_recall",
        "contradiction_count",
        "contradiction_rate",
        "agreement_on_comparable",
        "edge_score",
    )
    if any(key not in metrics for key in keys):
        raise ValueError("edge evidence does not match the registered candidate schema")
    return {key: metrics[key] for key in keys}


def _weak_payload(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "rule_count",
        "weak_support_count",
        "weak_incompatibility_count",
        "unknown_count",
    }
    if not required.issubset(value):
        raise ValueError("weak EHR evidence is incomplete")
    return {
        "rule_count": int(value["rule_count"]),
        "weak_support_count": int(value["weak_support_count"]),
        "weak_incompatibility_count_not_hard_contradiction": int(
            value["weak_incompatibility_count"]
        ),
        "unknown_count": int(value["unknown_count"]),
        "excluded_from_hard_selection": True,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else round(float(numerator) / float(denominator), 8)


def _report_quality(row: dict[str, Any]) -> float:
    existing = row.get("static_scoring", {}).get("report_quality_score")
    if not isinstance(existing, (int, float)) or isinstance(existing, bool):
        raise ValueError("candidate registry lacks the report structure quality score")
    return round(float(existing), 8)


def _clinical_totals(edges: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known = sum(int(edge["known_reference_fact_count"]) for edge in edges.values())
    comparable = sum(int(edge["comparable_explicit_fact_count"]) for edge in edges.values())
    support = sum(int(edge["support_count"]) for edge in edges.values())
    contradiction = sum(int(edge["contradiction_count"]) for edge in edges.values())
    balance = _ratio(support - contradiction, known)
    return {
        "total_known_reference_fact_count": known,
        "total_comparable_fact_count": comparable,
        "total_direct_support_count": support,
        "ehr_direct_support_count": sum(
            int(edges[edge]["support_count"])
            for edge in ("ehr_cxr", "ehr_report")
        ),
        "total_hard_contradiction_count": contradiction,
        "overall_coverage": _ratio(comparable, known),
        "overall_support_recall": _ratio(support, known),
        "overall_hard_contradiction_rate": _ratio(contradiction, known),
        "clinical_balance_minus1_to1": balance,
        "clinical_balance_score_0_100": (
            None if balance is None else round(50.0 * (1.0 + balance), 8)
        ),
        "diagnostic_score_used_for_selection": False,
    }


def _score_row(
    row: dict[str, Any],
    *,
    ehr_cxr: dict[str, Any],
    ehr_report: dict[str, Any],
) -> dict[str, Any]:
    lineage = row["lineage"]
    if ehr_cxr["case_id"] != row["case_id"] or ehr_report["case_id"] != row["case_id"]:
        raise ValueError("edge-specific case lineage mismatch")
    if ehr_cxr["image_sha256"] != lineage["cxr_sha256"]:
        raise ValueError("edge-specific CXR hash mismatch")
    if ehr_report["report_sha256"] != lineage["report_sha256"]:
        raise ValueError("edge-specific report hash mismatch")
    if ehr_cxr["cxr_model_id"] != lineage["cxr_model_id"]:
        raise ValueError("edge-specific CXR model mismatch")
    if ehr_report["report_model_id"] != lineage["report_model_id"]:
        raise ValueError("edge-specific report model mismatch")
    if ehr_report["source_cxr_model_id"] != lineage["cxr_model_id"]:
        raise ValueError("edge-specific report parent model mismatch")

    edges = {
        "ehr_cxr": _normalized_edge(
            score_state_pair(
                ehr_cxr["ehr_finding_states"], ehr_cxr["cxr_finding_states"]
            )
        ),
        "ehr_report": _normalized_edge(ehr_report["hard_direct_metrics"]),
        "report_cxr": _normalized_edge(row["cross_modal"]["report_cxr"]),
    }
    quality = _report_quality(row)
    hard_failures = list(row["selection"]["hard_gate_failures"])
    runtime = row["cost"].get("known_runtime_seconds")
    if runtime is not None and (
        not isinstance(runtime, (int, float)) or isinstance(runtime, bool)
    ):
        raise TypeError("candidate runtime is not numeric")
    scoring = {
        "edge_metrics": edges,
        "weak_ehr_evidence": {
            "ehr_cxr": _weak_payload(ehr_cxr["weak_clinical_evidence"]),
            "ehr_report": _weak_payload(ehr_report["weak_clinical_evidence"]),
        },
        "clinical_totals": _clinical_totals(edges),
        "modality_quality": {
            "ehr_schema_valid": bool(row["modality_quality"]["ehr"]["schema_valid"]),
            "cxr_basic_validity_pass": not row["modality_quality"]["cxr"]["corrupted"]
            and row["modality_quality"]["cxr"]["blank"] is not True,
            "cxr_realism_scalar": None,
            "cxr_realism_status": "not_available_per_candidate_basic_validity_is_not_realism",
            "report_structure_quality_score_0_1": quality,
        },
        "cost": {
            "known_runtime_seconds": runtime,
            "known_model_calls": row["cost"].get("known_model_calls"),
        },
        "selection": {
            "hard_gate_failures": hard_failures,
            "hard_gate_failure_count": len(hard_failures),
            "eligible": not hard_failures,
            "selection_rank_within_case": None,
            "selected": False,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "triple_candidate_id": row["triple_candidate_id"],
        "case_id": row["case_id"],
        "lineage": lineage,
        "scoring": scoring,
    }


def _descending(value: Any) -> float:
    if value is None:
        return math.inf
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("ranking value is not numeric")
    return -float(value)


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    scoring = row["scoring"]
    totals = scoring["clinical_totals"]
    return (
        scoring["selection"]["hard_gate_failure_count"],
        totals["total_hard_contradiction_count"],
        -totals["ehr_direct_support_count"],
        _descending(scoring["edge_metrics"]["report_cxr"]["support_recall"]),
        _descending(
            scoring["modality_quality"]["report_structure_quality_score_0_1"]
        ),
        math.inf
        if scoring["cost"]["known_runtime_seconds"] is None
        else float(scoring["cost"]["known_runtime_seconds"]),
        str(row["triple_candidate_id"]),
    )


def _pooled_edge(rows: list[dict[str, Any]], edge: str) -> dict[str, Any]:
    metrics = [row["scoring"]["edge_metrics"][edge] for row in rows]
    known = sum(int(value["known_reference_fact_count"]) for value in metrics)
    comparable = sum(int(value["comparable_explicit_fact_count"]) for value in metrics)
    support = sum(int(value["support_count"]) for value in metrics)
    contradiction = sum(int(value["contradiction_count"]) for value in metrics)
    return {
        "known_reference_fact_count": known,
        "comparable_explicit_fact_count": comparable,
        "coverage": _ratio(comparable, known),
        "support_count": support,
        "support_recall": _ratio(support, known),
        "hard_contradiction_count": contradiction,
        "hard_contradiction_rate": _ratio(contradiction, known),
        "clinical_balance": _ratio(support - contradiction, known),
    }


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return None if not valid else round(statistics.fmean(valid), 8)


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edges = {
        edge: _pooled_edge(rows, edge)
        for edge in ("ehr_cxr", "ehr_report", "report_cxr")
    }
    known = sum(edge["known_reference_fact_count"] for edge in edges.values())
    comparable = sum(edge["comparable_explicit_fact_count"] for edge in edges.values())
    support = sum(edge["support_count"] for edge in edges.values())
    contradiction = sum(edge["hard_contradiction_count"] for edge in edges.values())
    balance = _ratio(support - contradiction, known)
    return {
        "selected_count": len(rows),
        "hard_gate_failure_count": sum(
            row["scoring"]["selection"]["hard_gate_failure_count"] for row in rows
        ),
        "edge_metrics": edges,
        "pooled_clinical": {
            "known_reference_fact_count": known,
            "comparable_explicit_fact_count": comparable,
            "coverage": _ratio(comparable, known),
            "support_count": support,
            "support_recall": _ratio(support, known),
            "hard_contradiction_count": contradiction,
            "hard_contradiction_rate": _ratio(contradiction, known),
            "clinical_balance_minus1_to1": balance,
            "clinical_balance_score_0_100": (
                None if balance is None else round(50.0 * (1.0 + balance), 8)
            ),
            "diagnostic_score_used_for_selection": False,
        },
        "mean_report_structure_quality_score_0_1": _mean(
            [
                row["scoring"]["modality_quality"][
                    "report_structure_quality_score_0_1"
                ]
                for row in rows
            ]
        ),
        "mean_known_runtime_seconds": _mean(
            [row["scoring"]["cost"]["known_runtime_seconds"] for row in rows]
        ),
        "cxr_model_selection_count": dict(
            sorted(Counter(row["lineage"]["cxr_model_id"] for row in rows).items())
        ),
        "report_model_selection_count": dict(
            sorted(Counter(row["lineage"]["report_model_id"] for row in rows).items())
        ),
    }


def _selection_record(row: dict[str, Any]) -> dict[str, Any]:
    scoring = row["scoring"]
    return {
        "case_id": row["case_id"],
        "triple_candidate_id": row["triple_candidate_id"],
        "cxr_candidate_id": row["lineage"]["cxr_candidate_id"],
        "cxr_model_id": row["lineage"]["cxr_model_id"],
        "cxr_seed": row["lineage"]["cxr_seed"],
        "report_candidate_id": row["lineage"]["report_candidate_id"],
        "report_model_id": row["lineage"]["report_model_id"],
        "selection_rank_within_case": scoring["selection"][
            "selection_rank_within_case"
        ],
        "hard_gate_failures": scoring["selection"]["hard_gate_failures"],
        "total_hard_contradiction_count": scoring["clinical_totals"][
            "total_hard_contradiction_count"
        ],
        "total_direct_support_count": scoring["clinical_totals"][
            "total_direct_support_count"
        ],
        "ehr_direct_support_count": scoring["clinical_totals"][
            "ehr_direct_support_count"
        ],
        "total_known_reference_fact_count": scoring["clinical_totals"][
            "total_known_reference_fact_count"
        ],
        "clinical_balance_score_0_100": scoring["clinical_totals"][
            "clinical_balance_score_0_100"
        ],
        "report_cxr_support_recall": scoring["edge_metrics"]["report_cxr"][
            "support_recall"
        ],
        "report_structure_quality_score_0_1": scoring["modality_quality"][
            "report_structure_quality_score_0_1"
        ],
        "known_runtime_seconds": scoring["cost"]["known_runtime_seconds"],
    }


def _csv(rows: list[dict[str, Any]]) -> str:
    fields = (
        "case_id",
        "triple_candidate_id",
        "cxr_candidate_id",
        "cxr_model_id",
        "cxr_seed",
        "report_candidate_id",
        "report_model_id",
        "hard_gate_failure_count",
        "ehr_cxr_support_recall",
        "ehr_cxr_hard_contradiction_rate",
        "ehr_cxr_coverage",
        "ehr_report_support_recall",
        "ehr_report_hard_contradiction_rate",
        "ehr_report_coverage",
        "report_cxr_support_recall",
        "report_cxr_hard_contradiction_rate",
        "report_cxr_coverage",
        "total_hard_contradiction_count",
        "total_direct_support_count",
        "ehr_direct_support_count",
        "total_known_reference_fact_count",
        "clinical_balance_score_0_100",
        "report_structure_quality_score_0_1",
        "known_runtime_seconds",
        "selection_rank_within_case",
        "selected",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda value: (value["case_id"], value["scoring"]["selection"]["selection_rank_within_case"])):
        scoring = row["scoring"]
        edge = scoring["edge_metrics"]
        totals = scoring["clinical_totals"]
        writer.writerow(
            {
                "case_id": row["case_id"],
                "triple_candidate_id": row["triple_candidate_id"],
                "cxr_candidate_id": row["lineage"]["cxr_candidate_id"],
                "cxr_model_id": row["lineage"]["cxr_model_id"],
                "cxr_seed": row["lineage"]["cxr_seed"],
                "report_candidate_id": row["lineage"]["report_candidate_id"],
                "report_model_id": row["lineage"]["report_model_id"],
                "hard_gate_failure_count": scoring["selection"]["hard_gate_failure_count"],
                "ehr_cxr_support_recall": edge["ehr_cxr"]["support_recall"],
                "ehr_cxr_hard_contradiction_rate": edge["ehr_cxr"]["contradiction_rate"],
                "ehr_cxr_coverage": edge["ehr_cxr"]["coverage"],
                "ehr_report_support_recall": edge["ehr_report"]["support_recall"],
                "ehr_report_hard_contradiction_rate": edge["ehr_report"]["contradiction_rate"],
                "ehr_report_coverage": edge["ehr_report"]["coverage"],
                "report_cxr_support_recall": edge["report_cxr"]["support_recall"],
                "report_cxr_hard_contradiction_rate": edge["report_cxr"]["contradiction_rate"],
                "report_cxr_coverage": edge["report_cxr"]["coverage"],
                "total_hard_contradiction_count": totals["total_hard_contradiction_count"],
                "total_direct_support_count": totals["total_direct_support_count"],
                "ehr_direct_support_count": totals["ehr_direct_support_count"],
                "total_known_reference_fact_count": totals["total_known_reference_fact_count"],
                "clinical_balance_score_0_100": totals["clinical_balance_score_0_100"],
                "report_structure_quality_score_0_1": scoring["modality_quality"]["report_structure_quality_score_0_1"],
                "known_runtime_seconds": scoring["cost"]["known_runtime_seconds"],
                "selection_rank_within_case": scoring["selection"]["selection_rank_within_case"],
                "selected": scoring["selection"]["selected"],
            }
        )
    return output.getvalue()


def _markdown(payload: dict[str, Any]) -> str:
    fixed = payload["baselines"]["operational_fixed"]["summary"]
    selected = payload["static_reranking"]["summary"]
    fixed_path = payload["baselines"]["operational_fixed"]["path"]
    lines = [
        "# TriCompose V1.1 Edge-specific Static Selection",
        "",
        "This is a diagnostic, training-free selection result. It uses uncalibrated frozen XRV thresholds and is not paper-primary.",
        "",
        "| Method | CXR calls/case | Report calls/case | Balance score (0-100) | Support | Hard contradiction | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Operational fixed `{fixed_path}` | 1 | 1 | {fixed['pooled_clinical']['clinical_balance_score_0_100']} | {fixed['pooled_clinical']['support_recall']} | {fixed['pooled_clinical']['hard_contradiction_rate']} | {fixed['pooled_clinical']['coverage']} |",
        f"| Exhaustive lexicographic reranking | 3 | 12 | {selected['pooled_clinical']['clinical_balance_score_0_100']} | {selected['pooled_clinical']['support_recall']} | {selected['pooled_clinical']['hard_contradiction_rate']} | {selected['pooled_clinical']['coverage']} |",
        "",
        "Selection order: hard validity gates, total hard contradictions, direct EHR-edge support, CXR-report support, report structure quality, known runtime, deterministic ID.",
        "The displayed balance score is diagnostic and is not used to choose candidates.",
        "Weak EHR priors, Qwen2.5-VL, BioViL, and unavailable per-image realism are excluded from primary selection.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_run = require_inside(args.score_table_run, PROTECTED_ROOT, must_exist=True)
    ehr_edge_run = require_inside(args.ehr_edge_run, PROTECTED_ROOT, must_exist=True)
    table_summary = read_json(table_run / "score_table_summary.json")
    if table_summary.get("schema_version") != TABLE_SCHEMA:
        raise ValueError("unsupported unified score table")
    if table_summary.get("table_status") != "complete_frozen_evidence":
        raise ValueError("selection requires a completed candidate registry")
    ehr_edges = read_json(ehr_edge_run / "ehr_edge_details.json")
    if ehr_edges.get("schema_version") != EHR_EDGE_SCHEMA:
        raise ValueError("unsupported edge-specific EHR evidence")
    policy, policy_source = _load_policy(args.policy_config)
    registry_rows = _read_jsonl(table_run / "score_table.jsonl")
    cxr_rows = ehr_edges.get("records", {}).get("ehr_cxr")
    report_rows = ehr_edges.get("records", {}).get("ehr_report")
    if not isinstance(cxr_rows, list) or not isinstance(report_rows, list):
        raise TypeError("edge-specific evidence lacks candidate records")
    cxr_by_id = {str(row["cxr_candidate_id"]): row for row in cxr_rows}
    report_by_id = {str(row["report_candidate_id"]): row for row in report_rows}
    if len(cxr_by_id) != 240 or len(report_by_id) != 960 or len(registry_rows) != 960:
        raise ValueError("selection requires the complete 80 x 3 x 4 candidate grid")

    rows: list[dict[str, Any]] = []
    for row in registry_rows:
        lineage = row["lineage"]
        cxr_id = str(lineage["cxr_candidate_id"])
        report_id = str(lineage["report_candidate_id"])
        if cxr_id not in cxr_by_id or report_id not in report_by_id:
            raise ValueError("candidate registry and edge evidence do not have identical coverage")
        rows.append(
            _score_row(row, ehr_cxr=cxr_by_id[cxr_id], ehr_report=report_by_id[report_id])
        )

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
        by_path[
            f"{row['lineage']['cxr_model_id']}->{row['lineage']['report_model_id']}"
        ].append(row)
    if len(by_case) != 80 or {len(values) for values in by_case.values()} != {12}:
        raise ValueError("every case must have exactly 12 candidates")
    if len(by_path) != 12 or {len(values) for values in by_path.values()} != {80}:
        raise ValueError("every fixed model path must cover all 80 cases")

    selected: list[dict[str, Any]] = []
    for candidates in by_case.values():
        ranked = sorted(candidates, key=selection_key)
        for rank, row in enumerate(ranked, start=1):
            row["scoring"]["selection"]["selection_rank_within_case"] = rank
            row["scoring"]["selection"]["selected"] = rank == 1
        selected.append(ranked[0])
    selected.sort(key=lambda row: str(row["case_id"]))

    fixed_results = {
        path: selection_summary(path_rows) for path, path_rows in sorted(by_path.items())
    }
    fixed_config = policy["operational_fixed_baseline"]
    fixed_path = f"{fixed_config['cxr_model_id']}->{fixed_config['report_model_id']}"
    if fixed_path not in by_path:
        raise ValueError("operational fixed baseline is absent from the model grid")
    descriptive_best_path = min(
        fixed_results,
        key=lambda path: (
            fixed_results[path]["hard_gate_failure_count"],
            fixed_results[path]["pooled_clinical"]["hard_contradiction_count"],
            -sum(
                fixed_results[path]["edge_metrics"][edge]["support_count"]
                for edge in ("ehr_cxr", "ehr_report")
            ),
            _descending(
                fixed_results[path]["edge_metrics"]["report_cxr"]["support_recall"]
            ),
            _descending(
                fixed_results[path]["mean_report_structure_quality_score_0_1"]
            ),
            math.inf
            if fixed_results[path]["mean_known_runtime_seconds"] is None
            else fixed_results[path]["mean_known_runtime_seconds"],
            path,
        ),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "diagnostic_uncalibrated_cxr_labels",
        "counts": {
            "cases": len(by_case),
            "candidate_rows": len(rows),
            "cxr_candidates": len(cxr_by_id),
            "report_candidates": len(report_by_id),
            "fixed_paths": len(by_path),
            "selected_triples": len(selected),
        },
        "policy": policy,
        "sources": {
            "candidate_registry": {
                "path": str(table_run),
                "manifest_sha256": sha256_file(table_run / "manifest.json"),
            },
            "edge_specific_ehr": {
                "path": str(ehr_edge_run),
                "manifest_sha256": sha256_file(ehr_edge_run / "manifest.json"),
            },
            "policy": policy_source,
        },
        "baselines": {
            "all_fixed_paths": fixed_results,
            "operational_fixed": {
                "path": fixed_path,
                "selection_scope": fixed_config["selection_scope"],
                "prospective_model_calls_per_case": {"cxr": 1, "report": 1, "total": 2},
                "summary": fixed_results[fixed_path],
            },
            "same_cohort_descriptive_best": {
                "path": descriptive_best_path,
                "selection_scope": "descriptive_only_not_held_out",
                "prospective_model_calls_per_case": {"cxr": 1, "report": 1, "total": 2},
                "summary": fixed_results[descriptive_best_path],
            },
        },
        "static_reranking": {
            "method": "exhaustive_edge_specific_lexicographic_reranking",
            "prospective_model_calls_per_case": {"cxr": 3, "report": 12, "total": 15},
            "summary": selection_summary(selected),
            "selections": [_selection_record(row) for row in selected],
        },
        "candidate_rows": rows,
        "agent_implemented": False,
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
        candidate_rows = payload.pop("candidate_rows")
        table = write_private_text(
            temporary / "candidate_score_table.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows),
        )
        csv_path = write_private_text(
            temporary / "candidate_score_table.csv", _csv(candidate_rows)
        )
        selections = write_private_text(
            temporary / "selected_triples.jsonl",
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in payload["static_reranking"]["selections"]
            ),
        )
        results = write_private_json(temporary / "selection_results.json", payload)
        markdown = write_private_text(
            temporary / "selection_summary.md", _markdown(payload)
        )
        manifest = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "evaluation_status": payload["evaluation_status"],
                "counts": payload["counts"],
                "artifacts": {
                    path.name: {
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in (table, csv_path, selections, results, markdown)
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
                "status": "completed_edge_specific_static_selection",
                "run_id": args.run_id,
                "candidate_rows": payload["counts"]["candidate_rows"],
                "selected_triples": payload["counts"]["selected_triples"],
                "manifest_sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

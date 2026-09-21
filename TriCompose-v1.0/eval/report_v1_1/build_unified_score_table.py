#!/usr/bin/env python3
"""Build the protected 960-row TriCompose V1.1 candidate score table.

One row represents one exact EHR -> CXR -> report lineage.  Available
deterministic quality evidence is materialized immediately; learned evidence
that has not run remains explicitly pending rather than receiving a proxy zero.
No report text, image pixels, or raw EHR content is written to the table.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from contracts import (
    PROTECTED_ROOT,
    commit_atomic_run,
    discard_atomic_run,
    load_cxr_candidates,
    load_report_candidates,
    new_atomic_run,
    read_json,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)


SCHEMA_VERSION = "tricompose-unified-score-table-v1.1"
ROW_SCHEMA_VERSION = "tricompose-unified-score-row-v1.1"
REQUIRED_CXR_MODELS = frozenset(
    {"chexgenbench_sana", "chexgenbench_pixart", "roentgen_v2"}
)
REQUIRED_REPORT_MODELS = frozenset(
    {"maira2", "cxrmate_single", "llavarad", "chexagent2"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-run", required=True)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--report-run", action="append", required=True)
    parser.add_argument("--cxr-basic-validity", required=True)
    parser.add_argument("--report-unimodal-details", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _workspace_file(path: str | Path) -> Path:
    workspace = Path("/project2/ruishanl_1185/inference_3mod").resolve(strict=True)
    source = Path(path).resolve(strict=True)
    if not source.is_relative_to(workspace) or not source.is_file():
        raise ValueError("evaluation input is outside the workspace")
    return source


def _load_cxr_validity(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = _workspace_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("CXR basic-validity rows must be a list")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise TypeError("CXR basic-validity row is invalid")
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in rows:
            raise ValueError("CXR basic-validity candidate ID is absent or duplicated")
        rows[candidate_id] = {
            "status": "computed",
            "corrupted": bool(row.get("corrupted")),
            "blank": row.get("blank"),
            "width": row.get("width"),
            "height": row.get("height"),
            "mean_intensity": row.get("mean_intensity"),
            "std_intensity": row.get("std_intensity"),
            "min_intensity": row.get("min_intensity"),
            "max_intensity": row.get("max_intensity"),
            "interpretation": "basic_validity_not_image_realism",
        }
    return rows, {"path": str(source), "sha256": sha256_file(source)}


def _load_report_quality(
    path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = require_inside(path, PROTECTED_ROOT, must_exist=True)
    payload = read_json(source)
    if payload.get("schema_version") != "tricompose-report-unimodal-evaluation-v1.1":
        raise ValueError("unsupported report-unimodal evaluation schema")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("report-unimodal details lack records")
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise TypeError("report-unimodal record is invalid")
        report_id = str(row.get("report_candidate_id", ""))
        if not report_id or report_id in rows:
            raise ValueError("report-unimodal candidate ID is absent or duplicated")
        rows[report_id] = {
            "status": "computed_reference_free_structure",
            "empty": row["empty"],
            "token_count": row["token_count"],
            "sentence_count": row["sentence_count"],
            "findings_complete": row["findings_complete"],
            "impression_complete": row["impression_complete"],
            "impression_required_by_model_contract": row[
                "impression_required_by_model_contract"
            ],
            "section_contract_pass": row["section_contract_pass"],
            "generic_report": row["generic_report"],
            "unsupported_temporal_comparison_language": row[
                "unsupported_temporal_comparison_language"
            ],
            "measurement_mention_not_automatically_an_error": row[
                "measurement_mention"
            ],
            "repeated_sentence": row["repeated_sentence"],
            "repeated_4gram_ratio": row["repeated_4gram_ratio"],
            "normalized_template_frequency": row["normalized_template_frequency"],
        }
    return rows, {"path": str(source), "sha256": sha256_file(source)}


def _load_ehr_metadata(
    staging_run: str | Path,
    cxrs: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    staging = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)
    manifest_path = staging / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "tricompose.staging.run.v1.1":
        raise ValueError("unified score table requires V1.1 staging")
    expected_by_case: dict[str, str] = {}
    for cxr in cxrs.values():
        case_id = str(cxr["case_id"])
        facts_hash = str(cxr["ehr_facts_sha256"])
        if case_id in expected_by_case and expected_by_case[case_id] != facts_hash:
            raise ValueError("one case has inconsistent EHR-facts lineage")
        expected_by_case[case_id] = facts_hash
    metadata: dict[str, dict[str, Any]] = {}
    for case_id, expected_hash in sorted(expected_by_case.items()):
        case_root = require_inside(staging / "cases" / case_id, staging, must_exist=True)
        facts_path = case_root / "ehr_facts.json"
        ehr_path = case_root / "synthetic_ehr.json"
        if sha256_file(facts_path) != expected_hash:
            raise ValueError("EHR-facts hash does not match CXR lineage")
        facts = read_json(facts_path, root=staging)
        ehr = read_json(ehr_path, root=staging)
        if facts.get("case_id") != case_id or ehr.get("case_id") != case_id:
            raise ValueError("staging case ID mismatch")
        summary = facts.get("summary")
        if not isinstance(summary, dict):
            raise TypeError("EHR facts lack a summary")
        metadata[case_id] = {
            "schema_valid": True,
            "ehr_sha256": sha256_file(ehr_path),
            "ehr_facts_sha256": expected_hash,
            "conditioning_tier": summary.get("conditioning_tier"),
            "underconditioned": summary.get("underconditioned"),
            "direct_positive_fact_count": summary.get("direct_positive_fact_count"),
            "documented_context_count": summary.get("documented_context_count"),
            "single_case_clinical_validity_score": None,
            "single_case_clinical_validity_status": (
                "not_defined_cohort_and_rule_metrics_remain_separate"
            ),
        }
    return metadata, {
        "path": str(staging),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _pending_edge(name: str) -> dict[str, Any]:
    return {
        "status": "pending_frozen_finding_evidence",
        "edge": name,
        "support": None,
        "contradiction_rate": None,
        "coverage": None,
        "macro_f1": None,
        "micro_f1": None,
        "cohen_kappa": None,
    }


def _cost(candidate: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get("cost")
    if not isinstance(value, dict):
        return {"model_calls": 1, "runtime_seconds": None}
    return {
        "model_calls": value.get("model_calls", 1),
        "runtime_seconds": value.get("runtime_seconds"),
        "peak_vram_gib": value.get("peak_vram_gib"),
    }


def _row(
    report: dict[str, Any],
    cxr: dict[str, Any],
    *,
    ehr: dict[str, Any],
    cxr_quality: dict[str, Any],
    report_quality: dict[str, Any],
) -> dict[str, Any]:
    cxr_cost = _cost(cxr)
    report_cost = _cost(report)
    runtime_values = [
        value
        for value in (cxr_cost["runtime_seconds"], report_cost["runtime_seconds"])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "triple_candidate_id": report["candidate_id"],
        "case_id": report["case_id"],
        "lineage": {
            "ehr_sha256": ehr["ehr_sha256"],
            "ehr_facts_sha256": ehr["ehr_facts_sha256"],
            "cxr_candidate_id": cxr["candidate_id"],
            "cxr_model_id": cxr["model_id"],
            "cxr_seed": cxr["seed"],
            "cxr_sha256": cxr["artifact"]["sha256"],
            "report_candidate_id": report["candidate_id"],
            "report_model_id": report["model_id"],
            "report_sha256": report["artifact"]["sha256"],
        },
        "modality_quality": {
            "ehr": ehr,
            "cxr": cxr_quality,
            "report": report_quality,
        },
        "cross_modal": {
            "ehr_cxr": _pending_edge("ehr_cxr"),
            "ehr_report": _pending_edge("ehr_report"),
            "report_cxr": _pending_edge("report_cxr"),
        },
        "secondary_scores": {
            "qwenvl_report_cxr": {
                "status": "pending_frozen_model_inference",
                "value": None,
                "calibrated": False,
            },
            "biovil_report_cxr": {
                "status": "pending_frozen_model_inference",
                "raw_cosine": None,
                "calibrated": False,
            },
        },
        "uncertainty": {
            "status": "pending_complete_candidate_evidence",
            "report_expert_disagreement": None,
            "cxr_candidate_disagreement": None,
        },
        "cost": {
            "ehr_generation": {"status": "not_available_in_candidate_manifest"},
            "cxr_generation": cxr_cost,
            "report_generation": report_cost,
            "known_runtime_seconds": round(sum(runtime_values), 6),
            "known_model_calls": int(cxr_cost["model_calls"])
            + int(report_cost["model_calls"]),
        },
        "selection": {
            "eligible": False,
            "status": "pending_cross_modal_evidence",
            "hard_gate_failures": [
                name
                for name, failed in (
                    ("corrupted_cxr", cxr_quality["corrupted"]),
                    ("blank_cxr", cxr_quality["blank"] is True),
                    ("empty_report", report_quality["empty"]),
                    ("report_section_contract", not report_quality["section_contract_pass"]),
                )
                if failed
            ],
        },
    }


def _csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = (
        "triple_candidate_id",
        "case_id",
        "cxr_candidate_id",
        "cxr_model_id",
        "cxr_seed",
        "report_model_id",
        "ehr_conditioning_tier",
        "ehr_underconditioned",
        "cxr_corrupted",
        "cxr_blank",
        "report_empty",
        "report_section_contract_pass",
        "report_generic",
        "report_unsupported_temporal",
        "report_repeated_4gram_ratio",
        "ehr_cxr_status",
        "ehr_report_status",
        "report_cxr_status",
        "qwenvl_status",
        "biovil_status",
        "known_runtime_seconds",
        "selection_eligible",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        lineage = row["lineage"]
        quality = row["modality_quality"]
        writer.writerow(
            {
                "triple_candidate_id": row["triple_candidate_id"],
                "case_id": row["case_id"],
                "cxr_candidate_id": lineage["cxr_candidate_id"],
                "cxr_model_id": lineage["cxr_model_id"],
                "cxr_seed": lineage["cxr_seed"],
                "report_model_id": lineage["report_model_id"],
                "ehr_conditioning_tier": quality["ehr"]["conditioning_tier"],
                "ehr_underconditioned": quality["ehr"]["underconditioned"],
                "cxr_corrupted": quality["cxr"]["corrupted"],
                "cxr_blank": quality["cxr"]["blank"],
                "report_empty": quality["report"]["empty"],
                "report_section_contract_pass": quality["report"][
                    "section_contract_pass"
                ],
                "report_generic": quality["report"]["generic_report"],
                "report_unsupported_temporal": quality["report"][
                    "unsupported_temporal_comparison_language"
                ],
                "report_repeated_4gram_ratio": quality["report"][
                    "repeated_4gram_ratio"
                ],
                "ehr_cxr_status": row["cross_modal"]["ehr_cxr"]["status"],
                "ehr_report_status": row["cross_modal"]["ehr_report"]["status"],
                "report_cxr_status": row["cross_modal"]["report_cxr"]["status"],
                "qwenvl_status": row["secondary_scores"]["qwenvl_report_cxr"][
                    "status"
                ],
                "biovil_status": row["secondary_scores"]["biovil_report_cxr"][
                    "status"
                ],
                "known_runtime_seconds": row["cost"]["known_runtime_seconds"],
                "selection_eligible": row["selection"]["eligible"],
            }
        )
    return output.getvalue()


def _markdown(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    lines = [
        "# TriCompose V1.1 Unified Candidate Score Table",
        "",
        f"- Rows / complete candidate lineages: {counts['rows']}",
        f"- Synthetic EHR cases: {counts['cases']}",
        f"- CXR candidates: {counts['cxr_candidates']}",
        f"- Report models: {counts['report_models']}",
        f"- Hard validity failures: {counts['rows_with_hard_gate_failure']}",
        "",
        "## Evidence status",
        "",
        "| Evidence family | Status |",
        "|---|---|",
        "| EHR provenance and conditioning metadata | complete |",
        "| CXR basic validity | complete; not realism |",
        "| Report reference-free structure | complete |",
        "| EHR-CXR finding consistency | pending XRV labels/calibration |",
        "| EHR-Report finding consistency | pending CheXbert labels |",
        "| Report-CXR finding consistency | pending XRV + CheXbert labels |",
        "| Qwen2.5-VL matching | pending; secondary only |",
        "| BioViL-T cosine | pending; secondary only |",
        "| Candidate uncertainty | pending complete evidence |",
        "",
        "All pending values are stored as `null`, never as zero. No candidate is selection-eligible until the required cross-modal evidence is complete.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    cxr_quality, cxr_quality_source = _load_cxr_validity(args.cxr_basic_validity)
    report_quality, report_quality_source = _load_report_quality(
        args.report_unimodal_details
    )
    ehrs, staging_source = _load_ehr_metadata(args.staging_run, cxrs)
    if set(cxr_quality) != set(cxrs):
        raise ValueError("CXR basic-validity evidence does not exactly cover 240 CXRs")
    if set(report_quality) != set(reports):
        raise ValueError("report-unimodal evidence does not exactly cover 960 reports")
    cxr_models = {str(row["model_id"]) for row in cxrs.values()}
    report_models = {str(row["model_id"]) for row in reports.values()}
    if cxr_models != REQUIRED_CXR_MODELS or report_models != REQUIRED_REPORT_MODELS:
        raise ValueError("score table does not contain the registered 3x4 model grid")
    rows = [
        _row(
            report,
            cxrs[str(report["parent_cxr_candidate_id"])],
            ehr=ehrs[str(report["case_id"])],
            cxr_quality=cxr_quality[str(report["parent_cxr_candidate_id"])],
            report_quality=report_quality[report_id],
        )
        for report_id, report in sorted(reports.items())
    ]
    if len(rows) != 960 or len(ehrs) != 80 or len(cxrs) != 240:
        raise ValueError("V1.1 main score table must be exactly 80 EHR / 240 CXR / 960 rows")
    per_case = Counter(str(row["case_id"]) for row in rows)
    if set(per_case.values()) != {12}:
        raise ValueError("each EHR case must have exactly 12 complete candidate rows")
    return {
        "schema_version": SCHEMA_VERSION,
        "table_status": "skeleton_pending_frozen_cross_modal_evidence",
        "selection_ready": False,
        "counts": {
            "rows": len(rows),
            "cases": len(ehrs),
            "cxr_candidates": len(cxrs),
            "cxr_models": len(cxr_models),
            "report_models": len(report_models),
            "rows_per_case": 12,
            "rows_with_hard_gate_failure": sum(
                bool(row["selection"]["hard_gate_failures"]) for row in rows
            ),
        },
        "source_evidence": {
            "staging": staging_source,
            "cxr_basic_validity": cxr_quality_source,
            "report_unimodal": report_quality_source,
        },
        "model_grid": {
            "cxr_models": sorted(cxr_models),
            "report_models": sorted(report_models),
        },
        "missing_evidence": [
            "xrv_cxr_finding_labels_and_calibration",
            "chexbert_report_finding_labels",
            "ehr_cxr_consistency",
            "ehr_report_consistency",
            "report_cxr_consistency",
            "qwenvl_secondary_match",
            "biovil_secondary_cosine",
            "candidate_disagreement_uncertainty",
        ],
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
        jsonl = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in payload["rows"]
        )
        table_jsonl = write_private_text(temporary / "score_table.jsonl", jsonl)
        table_csv = write_private_text(temporary / "score_table.csv", _csv(payload["rows"]))
        summary_payload = {key: value for key, value in payload.items() if key != "rows"}
        summary = write_private_json(temporary / "score_table_summary.json", summary_payload)
        markdown = write_private_text(temporary / "score_table_summary.md", _markdown(payload))
        manifest = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "table_status": payload["table_status"],
                "selection_ready": payload["selection_ready"],
                "counts": payload["counts"],
                "artifacts": {
                    path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                    for path in (table_jsonl, table_csv, summary, markdown)
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
                "status": "completed_score_table_skeleton",
                "run_id": args.run_id,
                "rows": payload["counts"]["rows"],
                "selection_ready": False,
                "manifest_sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

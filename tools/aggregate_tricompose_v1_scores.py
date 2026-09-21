#!/usr/bin/env python3
"""Aggregate frozen V1 verifier outputs and run static per-EHR selection."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    require_private_file,
    sha256_file,
    write_private_json,
    write_private_text,
)
from tricompose_v1.graph import (
    CandidateEdge,
    CandidateGraph,
    CandidateNode,
    SelectionPolicy,
)
from tricompose_v1.scoring import (
    AGGREGATE_SCHEMA,
    BIOVIL_SCORE_SCHEMA,
    CALIBRATION_STATUS,
    QWENVL_SCORE_SCHEMA,
    XRV_SCORE_SCHEMA,
    commit_atomic_protected_run,
    compare_finding_states,
    deterministic_report_quality,
    discard_atomic_protected_run,
    ehr_prompt_intent_states,
    extract_report_finding_states,
    load_candidate_bank,
    load_ehr_facts,
    new_atomic_protected_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--xrv-run", required=True)
    parser.add_argument("--biovil-run", required=True)
    parser.add_argument("--qwenvl-run", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    source = require_private_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("score bundle must contain an object")
    return payload


def _score_bundle(run: str | Path, schema: str, bank_hash: str) -> tuple[dict[str, Any], Path]:
    root = Path(run).resolve(strict=True)
    path = require_private_file(root / "scores.json")
    payload = _read_json(path)
    if payload.get("schema_version") != schema:
        raise ValueError("score bundle has the wrong schema")
    source = payload.get("source_candidate_bank")
    if not isinstance(source, dict) or source.get("manifest_sha256") != bank_hash:
        raise ValueError("score bundle was produced from a different candidate bank")
    if payload.get("calibration_status") != CALIBRATION_STATUS:
        raise ValueError("score bundle calibration status is inconsistent")
    return payload, path


def _model_call_cost(candidate: dict[str, Any]) -> float:
    cost = candidate.get("cost")
    if not isinstance(cost, dict):
        return 0.0
    calls = cost.get("model_calls", 0)
    if not isinstance(calls, (int, float)) or isinstance(calls, bool) or calls < 0:
        raise ValueError("candidate model-call cost is invalid")
    return float(calls)


def _mean(rows: list[float]) -> float | None:
    return round(statistics.fmean(rows), 8) if rows else None


def _summaries(
    bank: Any,
    xrv_by_cxr: dict[str, dict[str, Any]],
    biovil_cxr: dict[str, dict[str, Any]],
    biovil_report: dict[str, dict[str, Any]],
    qwen_by_report: dict[str, dict[str, Any]],
    qualities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cxr_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cxr in bank.cxr_candidates:
        cxr_grouped[str(cxr["model_id"])].append(cxr)
    report_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in bank.report_candidates:
        report_grouped[str(report["model_id"])].append(report)

    cxr_summary: dict[str, Any] = {}
    for model_id, candidates in sorted(cxr_grouped.items()):
        xrv_scores = [
            float(xrv_by_cxr[str(row["candidate_id"])]["ehr_cxr_support"]["score"])
            for row in candidates
            if xrv_by_cxr[str(row["candidate_id"])]["ehr_cxr_support"]["score"] is not None
        ]
        prompt_scores = [
            float(biovil_cxr[str(row["candidate_id"])]["prompt_alignment_raw_cosine"])
            for row in candidates
        ]
        cxr_summary[model_id] = {
            "candidate_count": len(candidates),
            "xrv_ehr_support_mean_uncalibrated": _mean(xrv_scores),
            "biovil_prompt_alignment_mean_raw_cosine": _mean(prompt_scores),
            "nondegeneracy_pass_count": sum(
                bool(xrv_by_cxr[str(row["candidate_id"])]["cxr_quality"]["hard_gate_pass"])
                for row in candidates
            ),
        }

    report_summary: dict[str, Any] = {}
    for model_id, candidates in sorted(report_grouped.items()):
        scored = [
            qwen_by_report[str(row["candidate_id"])]
            for row in candidates
            if qwen_by_report[str(row["candidate_id"])]["status"] == "scored"
        ]
        report_summary[model_id] = {
            "candidate_count": len(candidates),
            "qwen_scored_count": len(scored),
            "qwen_parse_error_count": len(candidates) - len(scored),
            "qwen_finding_contract_complete_count": sum(
                row.get("finding_contract_status") == "complete" for row in scored
            ),
            "qwen_match_mean_uncalibrated": _mean(
                [float(row["qwen_match_score"]) for row in scored]
            ),
            "biovil_image_report_mean_raw_cosine": _mean(
                [
                    float(
                        biovil_report[str(row["candidate_id"])][
                            "image_report_alignment_raw_cosine"
                        ]
                    )
                    for row in candidates
                ]
            ),
            "report_quality_mean_deterministic": _mean(
                [float(qualities[str(row["candidate_id"])]["score"]) for row in candidates]
            ),
            "report_quality_hard_pass_count": sum(
                bool(qualities[str(row["candidate_id"])]["hard_gate_pass"])
                for row in candidates
            ),
        }
    return {"cxr_models": cxr_summary, "report_models": report_summary}


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = load_candidate_bank(args.candidate_bank)
    policy = SelectionPolicy.from_path(args.policy)
    xrv, xrv_path = _score_bundle(args.xrv_run, XRV_SCORE_SCHEMA, bank.manifest_sha256)
    biovil, biovil_path = _score_bundle(
        args.biovil_run, BIOVIL_SCORE_SCHEMA, bank.manifest_sha256
    )
    qwen, qwen_path = _score_bundle(
        args.qwenvl_run, QWENVL_SCORE_SCHEMA, bank.manifest_sha256
    )

    xrv_by_cxr = {str(row["cxr_candidate_id"]): row for row in xrv["records"]}
    biovil_cxr = {str(row["cxr_candidate_id"]): row for row in biovil["cxr_records"]}
    biovil_report = {
        str(row["report_candidate_id"]): row for row in biovil["report_records"]
    }
    qwen_by_report = {
        str(row["report_candidate_id"]): row for row in qwen["records"]
    }
    expected_cxr_ids = {str(row["candidate_id"]) for row in bank.cxr_candidates}
    expected_report_ids = {str(row["candidate_id"]) for row in bank.report_candidates}
    if set(xrv_by_cxr) != expected_cxr_ids or set(biovil_cxr) != expected_cxr_ids:
        raise ValueError("CXR score coverage is incomplete")
    if set(biovil_report) != expected_report_ids or set(qwen_by_report) != expected_report_ids:
        raise ValueError("report score coverage is incomplete")

    intents: dict[str, dict[str, str]] = {}
    ehr_by_id: dict[str, dict[str, Any]] = {}
    for ehr in bank.ehr_candidates:
        ehr_id = str(ehr["candidate_id"])
        ehr_by_id[ehr_id] = ehr
        intents[ehr_id] = ehr_prompt_intent_states(load_ehr_facts(ehr))

    report_qualities: dict[str, dict[str, Any]] = {}
    report_states: dict[str, dict[str, str]] = {}
    for report in bank.report_candidates:
        report_id = str(report["candidate_id"])
        path = require_private_file(report["artifact"]["path"])
        if sha256_file(path) != report["artifact"]["sha256"]:
            raise ValueError("report artifact hash changed before aggregation")
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        report_qualities[report_id] = deterministic_report_quality(text)
        report_states[report_id] = extract_report_finding_states(text)

    cxr_by_id = bank.cxr_by_id
    case_results: list[dict[str, Any]] = []
    score_vectors: list[dict[str, Any]] = []
    for ehr_id, ehr in sorted(ehr_by_id.items()):
        case_cxrs = [
            row for row in bank.cxr_candidates if row["parent_ids"] == [ehr_id]
        ]
        case_cxr_ids = {str(row["candidate_id"]) for row in case_cxrs}
        case_reports = [
            row
            for row in bank.report_candidates
            if row["parent_ids"][0] == ehr_id
            and str(row["parent_ids"][1]) in case_cxr_ids
        ]
        nodes: list[CandidateNode] = [
            CandidateNode(
                candidate_id=ehr_id,
                modality="ehr",
                model_id=str(ehr["model_id"]),
                parent_ids=(),
                artifact_sha256=str(ehr["artifacts"]["synthetic_ehr"]["sha256"]),
                valid=True,
                quality_score=1.0,
                cost_units=1.0,
            )
        ]
        edges: list[CandidateEdge] = []
        for cxr in case_cxrs:
            cxr_id = str(cxr["candidate_id"])
            xrv_row = xrv_by_cxr[cxr_id]
            quality = xrv_row["cxr_quality"]
            nodes.append(
                CandidateNode(
                    candidate_id=cxr_id,
                    modality="cxr",
                    model_id=str(cxr["model_id"]),
                    parent_ids=(ehr_id,),
                    artifact_sha256=str(cxr["artifact"]["sha256"]),
                    valid=bool(quality["hard_gate_pass"]),
                    quality_score=float(quality["score"]),
                    cost_units=_model_call_cost(cxr),
                )
            )
            ehr_cxr = xrv_row["ehr_cxr_support"]
            edges.append(
                CandidateEdge(
                    edge_id=f"edge_ehr_cxr_{cxr_id}",
                    source_id=ehr_id,
                    target_id=cxr_id,
                    kind="ehr_cxr",
                    verifier_model_id="frozen_xrv_densenet121_all",
                    score=(
                        None if ehr_cxr["score"] is None else float(ehr_cxr["score"])
                    ),
                    comparable=ehr_cxr["score"] is not None,
                    strong_contradiction_count=int(
                        ehr_cxr["strong_contradiction_count"]
                    ),
                    cost_units=1.0,
                )
            )

        for report in case_reports:
            report_id = str(report["candidate_id"])
            cxr_id = str(report["parent_ids"][1])
            quality = report_qualities[report_id]
            qwen_row = qwen_by_report[report_id]
            nodes.append(
                CandidateNode(
                    candidate_id=report_id,
                    modality="report",
                    model_id=str(report["model_id"]),
                    parent_ids=(ehr_id, cxr_id),
                    artifact_sha256=str(report["artifact"]["sha256"]),
                    valid=bool(quality["hard_gate_pass"]),
                    quality_score=float(quality["score"]),
                    cost_units=_model_call_cost(report),
                )
            )
            qwen_scored = qwen_row["status"] == "scored"
            qwen_strong = (
                int(qwen_row["finding_comparison"]["strong_contradiction_count"])
                if qwen_scored
                else 0
            )
            edges.append(
                CandidateEdge(
                    edge_id=f"edge_cxr_report_{report_id}",
                    source_id=cxr_id,
                    target_id=report_id,
                    kind="cxr_report",
                    verifier_model_id="frozen_qwen25vl_structured_uncalibrated",
                    score=float(qwen_row["qwen_match_score"]) if qwen_scored else None,
                    comparable=qwen_scored,
                    strong_contradiction_count=qwen_strong,
                    cost_units=1.0,
                )
            )
            ehr_report = compare_finding_states(intents[ehr_id], report_states[report_id])
            edges.append(
                CandidateEdge(
                    edge_id=f"edge_ehr_report_{report_id}",
                    source_id=ehr_id,
                    target_id=report_id,
                    kind="ehr_report",
                    verifier_model_id="deterministic_lexical_finding_v1",
                    score=(
                        None
                        if ehr_report["score"] is None
                        else float(ehr_report["score"])
                    ),
                    comparable=ehr_report["score"] is not None,
                    strong_contradiction_count=int(
                        ehr_report["strong_contradiction_count"]
                    ),
                    cost_units=0.0,
                )
            )
            score_vectors.append(
                {
                    "ehr_candidate_id": ehr_id,
                    "cxr_candidate_id": cxr_id,
                    "report_candidate_id": report_id,
                    "model_ids": {
                        "cxr": cxr_by_id[cxr_id]["model_id"],
                        "report": report["model_id"],
                    },
                    "scores": {
                        "ehr_cxr_xrv_support_uncalibrated": xrv_by_cxr[cxr_id][
                            "ehr_cxr_support"
                        ]["score"],
                        "ehr_cxr_biovil_prompt_raw_cosine": biovil_cxr[cxr_id][
                            "prompt_alignment_raw_cosine"
                        ],
                        "cxr_report_qwen_uncalibrated": (
                            qwen_row.get("qwen_match_score") if qwen_scored else None
                        ),
                        "cxr_report_qwen_finding_contract_status": (
                            qwen_row.get("finding_contract_status")
                            if qwen_scored
                            else "unavailable_parse_error"
                        ),
                        "cxr_report_biovil_raw_cosine": biovil_report[report_id][
                            "image_report_alignment_raw_cosine"
                        ],
                        "ehr_report_deterministic": ehr_report["score"],
                        "cxr_quality_nondegeneracy": xrv_by_cxr[cxr_id][
                            "cxr_quality"
                        ]["score"],
                        "report_quality_deterministic": quality["score"],
                    },
                    "strong_contradictions": {
                        "cxr_report": qwen_strong,
                        "ehr_report": ehr_report["strong_contradiction_count"],
                    },
                }
            )

        graph = CandidateGraph(nodes, edges)
        ranking = graph.rank_triples(policy)
        decision = graph.select_or_route(policy, budget_remaining=True)
        case_results.append(
            {
                "ehr_candidate_id": ehr_id,
                "decision": decision,
                "exploratory_best": ranking["ranked"][0] if ranking["ranked"] else None,
                "ranked_candidate_count": len(ranking["ranked"]),
                "incomplete_candidate_count": len(ranking["incomplete"]),
                "rejected_candidate_count": len(ranking["rejected"]),
                "ranking": ranking,
            }
        )

    return {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "completed_exploratory_uncalibrated",
        "calibration_status": CALIBRATION_STATUS,
        "selection_scope": "one independent selection per fixed synthetic EHR",
        "source_candidate_bank": {
            "path": str(bank.root),
            "manifest_sha256": bank.manifest_sha256,
        },
        "score_bundles": {
            "xrv": {"path": str(xrv_path), "sha256": sha256_file(xrv_path)},
            "biovil": {"path": str(biovil_path), "sha256": sha256_file(biovil_path)},
            "qwenvl": {"path": str(qwen_path), "sha256": sha256_file(qwen_path)},
        },
        "policy": {
            "path": str(Path(args.policy).resolve(strict=True)),
            "sha256": sha256_file(args.policy),
            "policy_id": policy.policy_id,
            "interpretation": "uncalibrated engineering selector, not clinical probability",
        },
        "counts": {
            "ehr_candidates": len(bank.ehr_candidates),
            "cxr_candidates": len(bank.cxr_candidates),
            "report_candidates": len(bank.report_candidates),
            "triple_score_vectors": len(score_vectors),
        },
        "model_summary": _summaries(
            bank,
            xrv_by_cxr,
            biovil_cxr,
            biovil_report,
            qwen_by_report,
            report_qualities,
        ),
        "case_results": case_results,
        "score_vectors": score_vectors,
        "limitations": [
            "No V1 score or threshold is calibrated on matched triples and hard negatives.",
            "BioViL-T raw cosine is retained as evidence and is not collapsed into the selector.",
            "CheXbert is unavailable locally; EHR-report uses a conservative lexical fallback.",
            "CXR quality is only a non-degeneracy gate, not a realism metric.",
            "CXRMate-single is CXR-only and is not treated as an EHR+CXR report path.",
        ],
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TriCompose V1 static score summary",
        "",
        "Status: exploratory and uncalibrated. No generated text or image is embedded here.",
        "",
        f"- Fixed synthetic EHR cases: {payload['counts']['ehr_candidates']}",
        f"- CXR candidates: {payload['counts']['cxr_candidates']}",
        f"- Report candidates: {payload['counts']['report_candidates']}",
        "",
        "## Per-EHR decision",
        "",
        "| EHR candidate | action | ranked | incomplete | rejected |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["case_results"]:
        lines.append(
            "| {ehr} | {action} | {ranked} | {incomplete} | {rejected} |".format(
                ehr=row["ehr_candidate_id"],
                action=row["decision"]["action"],
                ranked=row["ranked_candidate_count"],
                incomplete=row["incomplete_candidate_count"],
                rejected=row["rejected_candidate_count"],
            )
        )
    lines.extend(
        [
            "",
            "The complete score vectors and model-level aggregates are in `scores.json`.",
            "Raw BioViL-T cosine and Qwen/XRV values must not be interpreted as calibrated clinical probabilities.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o077)
    started = time.monotonic()
    temp, target = new_atomic_protected_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        score_path = write_private_json(temp / "scores.json", payload)
        summary_path = write_private_text(temp / "summary.md", _summary_markdown(payload))
        score_hash = sha256_file(score_path)
        summary_hash = sha256_file(summary_path)
        commit_atomic_protected_run(temp, target)
    except Exception:
        discard_atomic_protected_run(temp)
        raise
    print(
        json.dumps(
            {
                "stage": "tricompose_v1_static_selection",
                "status": "ok",
                "run_id": args.run_id,
                "ehr_case_count": payload["counts"]["ehr_candidates"],
                "triple_count": payload["counts"]["triple_score_vectors"],
                "scores_sha256": score_hash,
                "summary_sha256": summary_hash,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

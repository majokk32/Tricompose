"""Select a protected report candidate using a deterministic Phase-0 policy."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROJECT_ROOT,
    create_private_stage_dir,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)


DECISION_SCHEMA_VERSION = "tricompose.agent_decision.v1"
SCORE_SCHEMA_VERSION = "tricompose.score_bundle.v1"
POLICY_SCHEMA_VERSION = "tricompose.report_selector_policy.v1"
PRODUCER_VERSION = "1.0.0"
QWEN_METRIC = "qwen25vl_cxr_report_match"


def _validate_opaque_name(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise ValueError(f"invalid {label}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--qwen-score-bundle", required=True)
    parser.add_argument("--peer-score-bundle", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("CANDIDATE_ID", "MODEL_ID", "REPORT_PATH"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def choose_action(
    scores: dict[str, float],
    *,
    minimum_score: float,
    minimum_margin: float,
    suggested_next_verifier: str,
) -> dict[str, object]:
    """Return an auditable action without reading candidate contents."""
    if len(scores) < 2:
        raise ValueError("at least two scored candidates are required")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum score must be in [0,1]")
    if not 0.0 <= minimum_margin <= 1.0:
        raise ValueError("minimum margin must be in [0,1]")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
        for value in scores.values()
    ):
        raise ValueError("candidate scores must be numeric values in [0,1]")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_candidate_id, top_score = ranked[0]
    runner_up_score = ranked[1][1]
    tied_top_candidate_ids = sorted(
        candidate_id
        for candidate_id, score in scores.items()
        if float(score) == float(top_score)
    )
    margin = round(float(top_score) - float(runner_up_score), 8)
    decision: dict[str, object] = {
        "top_candidate_id": top_candidate_id,
        "tied_top_candidate_ids": tied_top_candidate_ids,
        "top_score": round(float(top_score), 8),
        "runner_up_score": round(float(runner_up_score), 8),
        "margin": margin,
    }
    if top_score >= minimum_score and margin >= minimum_margin:
        decision.update(
            {
                "action": "select",
                "status": "provisional",
                "reason_code": "qwen_score_and_margin_passed",
                "selected_candidate_id": top_candidate_id,
            }
        )
    else:
        reason_code = (
            "top_score_below_threshold"
            if top_score < minimum_score
            else "candidate_margin_below_threshold"
        )
        decision.update(
            {
                "action": "verify_more",
                "status": "insufficient_evidence",
                "reason_code": reason_code,
                "suggested_next_verifier": suggested_next_verifier,
            }
        )
    return decision


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON input must be an object")
    return payload


def _extract_qwen_scores(
    bundle: dict[str, Any],
) -> dict[str, float]:
    if bundle.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported score bundle schema")
    records = bundle.get("records")
    if not isinstance(records, list):
        raise TypeError("score bundle records are missing")

    scores: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("metric") != QWEN_METRIC:
            continue
        candidate_ids = record.get("candidate_ids")
        value = record.get("value")
        if (
            not isinstance(candidate_ids, list)
            or len(candidate_ids) != 1
            or not isinstance(candidate_ids[0], str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError("invalid Qwen candidate score record")
        candidate_id = _validate_opaque_name(
            candidate_ids[0],
            label="scored candidate ID",
        )
        if candidate_id in scores:
            raise ValueError("duplicate Qwen candidate score")
        scores[candidate_id] = float(value)
    return scores


def _extract_peer_scores(
    bundle: dict[str, Any],
    *,
    candidate_ids: set[str],
    candidate_hashes: dict[str, str],
) -> dict[str, float]:
    if bundle.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported peer score bundle schema")
    if bundle.get("comparison_role") != "peer":
        raise ValueError("report score bundle is not a peer comparison")
    bundle_candidates = bundle.get("candidates")
    if not isinstance(bundle_candidates, dict):
        raise TypeError("peer score candidate metadata are missing")
    if set(bundle_candidates) != candidate_ids:
        raise ValueError("peer score and candidate IDs do not match")
    for candidate_id in candidate_ids:
        metadata = bundle_candidates.get(candidate_id)
        if (
            not isinstance(metadata, dict)
            or metadata.get("sha256") != candidate_hashes[candidate_id]
        ):
            raise ValueError("candidate report does not match peer score hash")

    records = bundle.get("records")
    if not isinstance(records, list):
        raise TypeError("peer score records are missing")
    peer_scores: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("scope") != "report_pair":
            continue
        metric = record.get("metric")
        record_candidate_ids = record.get("candidate_ids")
        value = record.get("value")
        if (
            not isinstance(metric, str)
            or not isinstance(record_candidate_ids, list)
            or set(record_candidate_ids) != candidate_ids
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise TypeError("invalid peer score record")
        if metric in peer_scores:
            raise ValueError("duplicate peer score metric")
        peer_scores[metric] = float(value)
    if not peer_scores:
        raise ValueError("no report-pair score records found")
    return peer_scores


def _validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported selector policy schema")
    if policy.get("candidate_metric") != QWEN_METRIC:
        raise ValueError("unsupported candidate metric")
    if policy.get("selection_action") != "select":
        raise ValueError("unsupported selection action")
    if policy.get("fallback_action") != "verify_more":
        raise ValueError("unsupported fallback action")
    for key in ("minimum_score", "minimum_margin"):
        value = policy.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"invalid policy field: {key}")
    for key in (
        "policy_id",
        "calibration_status",
        "suggested_next_verifier",
    ):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise ValueError(f"invalid policy field: {key}")


def _build_decision(args: argparse.Namespace) -> dict[str, object]:
    run_id = _validate_opaque_name(args.run_id, label="run ID")
    policy_path = require_inside(
        args.policy_config,
        PROJECT_ROOT,
        must_exist=True,
    )
    if not policy_path.is_file():
        raise ValueError("policy config is not a regular file")
    qwen_path = require_private_file(args.qwen_score_bundle)
    peer_path = require_private_file(args.peer_score_bundle)
    policy = _load_json(policy_path)
    qwen_bundle = _load_json(qwen_path)
    peer_bundle = _load_json(peer_path)
    _validate_policy(policy)
    if qwen_bundle.get("run_id") != run_id:
        raise ValueError("Qwen score bundle belongs to another run")
    if peer_bundle.get("run_id") != run_id:
        raise ValueError("peer score bundle belongs to another run")

    candidates: dict[str, dict[str, str]] = {}
    for candidate_id_raw, model_id_raw, report_path_raw in args.candidate:
        candidate_id = _validate_opaque_name(
            candidate_id_raw,
            label="candidate ID",
        )
        model_id = _validate_opaque_name(model_id_raw, label="model ID")
        if candidate_id in candidates:
            raise ValueError("candidate IDs must be unique")
        report_path = require_private_file(report_path_raw)
        candidates[candidate_id] = {
            "model_id": model_id,
            "artifact_sha256": sha256_file(report_path),
        }
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")

    scores = _extract_qwen_scores(qwen_bundle)
    if set(scores) != set(candidates):
        raise ValueError("Qwen scores and candidate IDs do not match")
    details = qwen_bundle.get("candidate_details")
    if not isinstance(details, dict):
        raise TypeError("Qwen candidate details are missing")
    for candidate_id, candidate in candidates.items():
        detail = details.get(candidate_id)
        if (
            not isinstance(detail, dict)
            or detail.get("report_sha256") != candidate["artifact_sha256"]
        ):
            raise ValueError("candidate report does not match Qwen input hash")

    peer_scores = _extract_peer_scores(
        peer_bundle,
        candidate_ids=set(candidates),
        candidate_hashes={
            candidate_id: candidate["artifact_sha256"]
            for candidate_id, candidate in candidates.items()
        },
    )
    decision = choose_action(
        scores,
        minimum_score=float(policy["minimum_score"]),
        minimum_margin=float(policy["minimum_margin"]),
        suggested_next_verifier=policy["suggested_next_verifier"],
    )
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "run_id": run_id,
        "producer": {
            "name": "tricompose_phase0_report_selector",
            "version": PRODUCER_VERSION,
        },
        "policy": {
            "policy_id": policy["policy_id"],
            "calibration_status": policy["calibration_status"],
            "minimum_score": policy["minimum_score"],
            "minimum_margin": policy["minimum_margin"],
        },
        "candidates": candidates,
        "evidence": {
            "candidate_metric": QWEN_METRIC,
            "candidate_scores": {
                candidate_id: round(score, 8)
                for candidate_id, score in sorted(scores.items())
            },
            "candidate_score_bundle_sha256": sha256_file(qwen_path),
            "peer_metrics": {
                metric: round(score, 8)
                for metric, score in sorted(peer_scores.items())
            },
            "peer_score_bundle_sha256": sha256_file(peer_path),
        },
        "decision": decision,
    }


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    stage_dir = create_private_stage_dir(args.output_dir)
    decision = _build_decision(args)
    output_path = write_private_json(
        stage_dir / "agent_decision.json",
        decision,
    )
    print(
        json.dumps(
            {
                "stage": "phase0_report_selector",
                "status": "ok",
                "artifact": output_path.name,
                "sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

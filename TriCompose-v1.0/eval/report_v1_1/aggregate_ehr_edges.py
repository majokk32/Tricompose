#!/usr/bin/env python3
"""Evaluate EHR-CXR and EHR-report with edge-specific evidence contracts.

This CPU-only stage consumes previously generated frozen XRV and CheXbert
labels. It never treats clinical context as a required radiographic finding
and never writes EHR or report text to its outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aggregate_crossmodal import (
    CXR_LABEL_SCHEMA,
    REPORT_LABEL_SCHEMA,
    _load_label_bundle,
)
from contracts import (
    CHEXPERT_FINDINGS,
    PROTECTED_ROOT,
    commit_atomic_run,
    discard_atomic_run,
    load_cxr_candidates,
    load_report_candidates,
    new_atomic_run,
    read_json,
    require_inside,
    sha256_file,
    validate_finding_states,
    write_private_json,
    write_private_text,
)
from crossmodal_metrics import ehr_direct_finding_states, summarize_state_pairs
from ehr_edge_metrics import (
    PRIMARY_EHR_CXR_FINDINGS,
    SOURCE_CATEGORIES,
    probability_metrics,
    score_direct_ehr_report,
    score_weak_priors,
    strict_states_by_source,
    weak_clinical_priors,
)


SCHEMA_VERSION = "tricompose-ehr-edge-crossmodal-evaluation-v1.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-run", required=True)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--report-run", action="append", required=True)
    parser.add_argument("--cxr-labels", required=True)
    parser.add_argument("--report-labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _cxr_probabilities(
    path: str | Path,
    cxrs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    source = require_inside(path, PROTECTED_ROOT, must_exist=True)
    payload = read_json(source)
    if payload.get("schema_version") != CXR_LABEL_SCHEMA:
        raise ValueError("unsupported CXR finding-label bundle")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("CXR label bundle lacks records")
    output: dict[str, dict[str, float | None]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise TypeError("CXR label record is invalid")
        candidate_id = str(row.get("cxr_candidate_id", ""))
        candidate = cxrs.get(candidate_id)
        if candidate is None or candidate_id in output:
            raise ValueError("CXR probability candidate is absent or duplicated")
        if row.get("image_sha256") != candidate["artifact"]["sha256"]:
            raise ValueError("CXR probability image hash mismatch")
        values = row.get("finding_probabilities")
        if not isinstance(values, dict) or set(values) != set(CHEXPERT_FINDINGS):
            raise ValueError("CXR probability vector is incomplete")
        normalized: dict[str, float | None] = {}
        for finding in CHEXPERT_FINDINGS:
            value = values[finding]
            if value is None:
                normalized[finding] = None
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("CXR probability is not numeric")
            elif not 0.0 <= float(value) <= 1.0:
                raise ValueError("CXR probability lies outside [0,1]")
            else:
                normalized[finding] = float(value)
        output[candidate_id] = normalized
    if set(output) != set(cxrs):
        raise ValueError("CXR probabilities do not exactly cover the candidate pool")
    return output


def _load_ehr_evidence(
    staging_run: str | Path,
    cxrs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    staging = require_inside(staging_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(staging / "run_manifest.json")
    if manifest.get("schema_version") != "tricompose.staging.run.v1.1":
        raise ValueError("edge-specific evaluation requires V1.1 staging")
    expected_hashes: dict[str, str] = {}
    for cxr in cxrs.values():
        case_id = str(cxr["case_id"])
        facts_hash = str(cxr["ehr_facts_sha256"])
        if case_id in expected_hashes and expected_hashes[case_id] != facts_hash:
            raise ValueError("one case has inconsistent EHR-facts lineage")
        expected_hashes[case_id] = facts_hash
    evidence: dict[str, dict[str, Any]] = {}
    for case_id, expected_hash in sorted(expected_hashes.items()):
        case_root = require_inside(staging / "cases" / case_id, staging, must_exist=True)
        facts_path = case_root / "ehr_facts.json"
        ehr_path = case_root / "synthetic_ehr.json"
        if sha256_file(facts_path) != expected_hash:
            raise ValueError("EHR-facts hash does not match CXR lineage")
        facts = read_json(facts_path, root=staging)
        ehr = read_json(ehr_path, root=staging)
        if facts.get("case_id") != case_id or ehr.get("case_id") != case_id:
            raise ValueError("staging case ID mismatch")
        latest = len(ehr["timeline"]) - 1
        event_counts = {
            category: sum(event.get("visit_index") == latest for event in ehr[field])
            for category, field in (
                ("diagnosis", "diagnoses"),
                ("medication", "medications"),
                ("lab", "labs"),
                ("vital", "vitals"),
            )
        }
        event_counts["other"] = len(ehr["timeline"][latest].get("other_tokens", []))
        evidence[case_id] = {
            "ehr_sha256": sha256_file(ehr_path),
            "ehr_facts_sha256": expected_hash,
            "conditioning_tier": facts["summary"]["conditioning_tier"],
            "strict_states": ehr_direct_finding_states(facts),
            "strict_states_by_source": strict_states_by_source(facts),
            "weak_rules": weak_clinical_priors(facts, ehr),
            "latest_visit_event_counts": event_counts,
        }
    return evidence, {
        "path": str(staging),
        "manifest_sha256": sha256_file(staging / "run_manifest.json"),
    }


def _weak_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rule_count = sum(row[key]["rule_count"] for row in rows)
    support = sum(row[key]["weak_support_count"] for row in rows)
    incompatibility = sum(row[key]["weak_incompatibility_count"] for row in rows)
    unknown = sum(row[key]["unknown_count"] for row in rows)
    probabilities = [
        rule["maximum_target_probability"]
        for row in rows
        for rule in row[key]["rules"]
        if rule["maximum_target_probability"] is not None
    ]
    return {
        "rule_instances": rule_count,
        "weak_support_count": support,
        "weak_incompatibility_count_not_hard_contradiction": incompatibility,
        "unknown_count": unknown,
        "weak_support_rate": None if rule_count == 0 else round(support / rule_count, 8),
        "weak_incompatibility_rate": (
            None if rule_count == 0 else round(incompatibility / rule_count, 8)
        ),
        "mean_maximum_target_probability": (
            None if not probabilities else round(sum(probabilities) / len(probabilities), 8)
        ),
    }


def _hard_contradiction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = sum(row["hard_direct_metrics"]["known_reference_fact_count"] for row in rows)
    comparable = sum(
        row["hard_direct_metrics"]["comparable_explicit_fact_count"] for row in rows
    )
    support = sum(row["hard_direct_metrics"]["support_count"] for row in rows)
    contradiction = sum(
        row["hard_direct_metrics"]["contradiction_count"] for row in rows
    )
    no_finding = sum(
        len(row["hard_direct_metrics"]["no_finding_cross_contradiction_findings"])
        for row in rows
    )
    return {
        "known_direct_fact_count": known,
        "comparable_direct_fact_count": comparable,
        "support_count": support,
        "hard_contradiction_count": contradiction,
        "no_finding_cross_contradiction_count": no_finding,
        "coverage": None if known == 0 else round(comparable / known, 8),
        "support_recall": None if known == 0 else round(support / known, 8),
        "hard_contradiction_rate": (
            None if known == 0 else round(contradiction / known, 8)
        ),
    }


def _source_summary(report_rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    pair_rows = [
        {
            "ehr": row["ehr_finding_states_by_source"][source],
            "report": row["report_finding_states"],
        }
        for row in report_rows
    ]
    summary = summarize_state_pairs(pair_rows, reference_key="ehr", candidate_key="report")
    counts_by_case: dict[str, int] = {}
    for row in report_rows:
        case_id = str(row["case_id"])
        count = int(row["latest_visit_event_counts"][source])
        if case_id in counts_by_case and counts_by_case[case_id] != count:
            raise ValueError("one EHR case has inconsistent event counts across reports")
        counts_by_case[case_id] = count
    summary["latest_visit_event_count_across_unique_cases"] = sum(
        counts_by_case.values()
    )
    summary["cases_with_source_events"] = sum(count > 0 for count in counts_by_case.values())
    summary["status"] = (
        "not_comparable_no_direct_radiographic_fact_from_source"
        if summary["known_reference_fact_count"] == 0
        else "computed_direct_fact_consistency"
    )
    summary["scope_note"] = (
        "Missing mention is unknown, not contradiction. General clinical facts "
        "without a registered radiographic mapping are excluded."
    )
    return summary


def _by_group(
    rows: list[dict[str, Any]],
    group_key: str,
    *,
    reference_key: str,
    candidate_key: str,
    weak_key: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {
        group: {
            "direct_consistency": summarize_state_pairs(
                values, reference_key=reference_key, candidate_key=candidate_key
            ),
            "weak_clinical_evidence": _weak_summary(values, weak_key),
        }
        for group, values in sorted(groups.items())
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    ec = payload["ehr_cxr"]
    er = payload["ehr_report"]
    lines = [
        "# TriCompose V1.1 Edge-specific EHR Cross-modal Evaluation",
        "",
        "## EHR-CXR",
        "",
        f"- Direct known facts: {ec['direct_observable_consistency']['known_reference_fact_count']}",
        f"- Direct coverage: {ec['direct_observable_consistency']['known_fact_coverage']}",
        f"- Direct contradiction rate: {ec['direct_observable_consistency']['explicit_contradiction_rate_over_known_facts']}",
        f"- Weak rule instances: {ec['weak_clinical_evidence']['rule_instances']}",
        "- Temporal consistency: not applicable (cold-start, non-longitudinal cohort)",
        "",
        "## EHR-Report",
        "",
        f"- Direct known facts: {er['shared_direct_finding_agreement']['known_reference_fact_count']}",
        f"- Direct coverage: {er['shared_direct_finding_agreement']['known_fact_coverage']}",
        f"- Hard contradiction rate: {er['hard_contradiction']['hard_contradiction_rate']}",
        f"- Weak incompatibility rate: {er['weak_clinical_evidence']['weak_incompatibility_rate']}",
        "",
        "## Guardrails",
        "",
        "- CheXpert-14 is not treated as a complete EHR ontology.",
        "- CHF and loop diuretics create support-only weak priors, never hard labels.",
        "- Missing report mention is unknown, not contradiction.",
        "- Lab/vital facts without an audited radiographic mapping remain non-comparable.",
        "- AUROC/AUPRC/Brier/ECE exclude weak priors and remain diagnostic until held-out calibration.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    cxr_labels, cxr_label_meta = _load_label_bundle(
        args.cxr_labels,
        schema=CXR_LABEL_SCHEMA,
        id_key="cxr_candidate_id",
        hash_key="image_sha256",
        candidates=cxrs,
        artifact_key="artifact",
    )
    report_labels, report_label_meta = _load_label_bundle(
        args.report_labels,
        schema=REPORT_LABEL_SCHEMA,
        id_key="report_candidate_id",
        hash_key="report_sha256",
        candidates=reports,
        artifact_key="artifact",
    )
    cxr_probabilities = _cxr_probabilities(args.cxr_labels, cxrs)
    ehr, staging_meta = _load_ehr_evidence(args.staging_run, cxrs)

    cxr_rows: list[dict[str, Any]] = []
    for cxr_id, cxr in sorted(cxrs.items()):
        case_id = str(cxr["case_id"])
        evidence = ehr[case_id]
        states = cxr_labels[cxr_id]
        probabilities = cxr_probabilities[cxr_id]
        cxr_rows.append(
            {
                "case_id": case_id,
                "cxr_candidate_id": cxr_id,
                "cxr_model_id": cxr["model_id"],
                "image_sha256": cxr["artifact"]["sha256"],
                "ehr_finding_states": evidence["strict_states"],
                "cxr_finding_states": states,
                "cxr_finding_probabilities": probabilities,
                "weak_clinical_evidence": score_weak_priors(
                    evidence["weak_rules"], states, probabilities
                ),
            }
        )

    report_rows: list[dict[str, Any]] = []
    for report_id, report in sorted(reports.items()):
        case_id = str(report["case_id"])
        evidence = ehr[case_id]
        states = report_labels[report_id]
        report_rows.append(
            {
                "case_id": case_id,
                "report_candidate_id": report_id,
                "report_model_id": report["model_id"],
                "source_cxr_model_id": cxrs[str(report["parent_cxr_candidate_id"])]["model_id"],
                "report_sha256": report["artifact"]["sha256"],
                "ehr_finding_states": evidence["strict_states"],
                "ehr_finding_states_by_source": evidence["strict_states_by_source"],
                "report_finding_states": states,
                "hard_direct_metrics": score_direct_ehr_report(
                    evidence["strict_states"], states
                ),
                "weak_clinical_evidence": score_weak_priors(
                    evidence["weak_rules"], states
                ),
                "latest_visit_event_counts": evidence["latest_visit_event_counts"],
            }
        )

    ehr_cxr_direct = summarize_state_pairs(
        cxr_rows,
        reference_key="ehr_finding_states",
        candidate_key="cxr_finding_states",
    )
    ehr_report_direct = summarize_state_pairs(
        report_rows,
        reference_key="ehr_finding_states",
        candidate_key="report_finding_states",
    )
    profiles = Counter(value["conditioning_tier"] for value in ehr.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_scope": {
            "cohort": "fully_synthetic_cold_start_non_longitudinal",
            "unknown_is_negative": False,
            "weak_prior_is_hard_label": False,
            "real_reference_supplied": False,
        },
        "counts": {
            "cases": len(ehr),
            "cxr_candidates": len(cxrs),
            "report_candidates": len(reports),
            "conditioning_tiers": dict(sorted(profiles.items())),
        },
        "evidence": {
            "staging": staging_meta,
            "cxr_labels": cxr_label_meta,
            "report_labels": report_label_meta,
        },
        "ehr_cxr": {
            "primary_scope": "explicit_observable_ehr_facts_only",
            "direct_observable_consistency": ehr_cxr_direct,
            "disease_specific_consistency": {
                finding: ehr_cxr_direct["per_finding"][finding]
                for finding in PRIMARY_EHR_CXR_FINDINGS
            },
            "weak_clinical_evidence": _weak_summary(
                cxr_rows, "weak_clinical_evidence"
            ),
            "probability_metrics": probability_metrics(cxr_rows),
            "by_cxr_model": _by_group(
                cxr_rows,
                "cxr_model_id",
                reference_key="ehr_finding_states",
                candidate_key="cxr_finding_states",
                weak_key="weak_clinical_evidence",
            ),
            "temporal_consistency": {
                "status": "not_applicable_cold_start_non_longitudinal_cohort",
                "prior_current_progression_agreement": None,
            },
        },
        "ehr_report": {
            "primary_scope": "explicit_direct_facts_and_explicit_opposites",
            "shared_direct_finding_agreement": ehr_report_direct,
            "hard_contradiction": _hard_contradiction_summary(report_rows),
            "weak_clinical_evidence": _weak_summary(
                report_rows, "weak_clinical_evidence"
            ),
            "clinical_fact_consistency": {
                source: _source_summary(report_rows, source)
                for source in SOURCE_CATEGORIES
            },
            "by_report_model": _by_group(
                report_rows,
                "report_model_id",
                reference_key="ehr_finding_states",
                candidate_key="report_finding_states",
                weak_key="weak_clinical_evidence",
            ),
            "llm_based_clinical_consistency": {
                "status": "not_run_secondary_only",
                "score": None,
            },
        },
        "records": {
            "ehr_cxr": cxr_rows,
            "ehr_report": report_rows,
        },
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
        details = write_private_json(temporary / "ehr_edge_details.json", payload)
        summary_payload = {key: value for key, value in payload.items() if key != "records"}
        summary = write_private_json(temporary / "ehr_edge_summary.json", summary_payload)
        markdown = write_private_text(temporary / "ehr_edge_summary.md", _markdown(payload))
        manifest = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "counts": payload["counts"],
                "artifacts": {
                    path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                    for path in (details, summary, markdown)
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
                "status": "completed_edge_specific_ehr_evaluation",
                "run_id": args.run_id,
                "counts": payload["counts"],
                "manifest_sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

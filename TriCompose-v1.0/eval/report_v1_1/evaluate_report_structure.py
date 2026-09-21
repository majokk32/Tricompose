#!/usr/bin/env python3
"""Evaluate reference-free structure and language quality of V1.1 reports.

This is a lightweight CPU evaluation over fully synthetic reports.  It does
not compute reference-based text overlap or clinical factuality and never
writes report text to its output bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from contracts import (
    commit_atomic_run,
    discard_atomic_run,
    load_cxr_candidates,
    load_report_candidates,
    new_atomic_run,
    read_report_text,
    sha256_file,
    write_private_json,
    write_private_text,
)


SCHEMA_VERSION = "tricompose-report-unimodal-evaluation-v1.1"
PRODUCER_VERSION = "1.0.0"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]?", flags=re.MULTILINE)
FINDINGS_HEADING = re.compile(r"(?im)^\s*findings\s*:\s*")
IMPRESSION_HEADING = re.compile(r"(?im)^\s*impression\s*:\s*")
TEMPORAL_PATTERN = re.compile(
    r"\b(?:compared (?:with|to)|comparison (?:with|to)|prior (?:study|exam|film)|"
    r"previous (?:study|exam|film)|since (?:the )?prior|interval (?:change|development)|"
    r"unchanged|no significant change|improved since|worsened since|stable since)\b",
    flags=re.I,
)
MEASUREMENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", flags=re.I)
GENERIC_NORMAL_REPORTS = frozenset(
    {
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary process",
        "no acute cardiopulmonary disease",
        "no acute disease",
        "no acute findings",
        "normal chest radiograph",
    }
)
MODEL_SECTION_CONTRACT = {
    "maira2": {"findings": True, "impression": False},
    "cxrmate_single": {"findings": True, "impression": True},
    "llavarad": {"findings": True, "impression": False},
    "chexagent2": {"findings": True, "impression": False},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--report-run", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _normalize(text: str) -> str:
    without_headings = FINDINGS_HEADING.sub(" ", text)
    without_headings = IMPRESSION_HEADING.sub(" ", without_headings)
    tokens = TOKEN_PATTERN.findall(without_headings.lower())
    return " ".join(tokens)


def _sentences(text: str) -> list[str]:
    return [
        " ".join(match.group(0).split()).strip(" .!?\t\r\n").lower()
        for match in SENTENCE_PATTERN.finditer(text)
        if match.group(0).strip(" .!?\t\r\n")
    ]


def _section_text(text: str, heading: re.Pattern[str], other: re.Pattern[str]) -> str:
    match = heading.search(text)
    if match is None:
        return ""
    end = other.search(text, pos=match.end())
    return text[match.end() : end.start() if end else len(text)].strip()


def _repeated_ngram_ratio(tokens: list[str], order: int = 4) -> float:
    if len(tokens) < order:
        return 0.0
    grams = [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p05": None, "p95": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "p05": round(percentile(0.05), 6),
        "p95": round(percentile(0.95), 6),
    }


def _record(
    report: Mapping[str, Any],
    cxr: Mapping[str, Any],
    *,
    normalized_frequency: Mapping[str, int],
) -> dict[str, Any]:
    text = read_report_text(report)
    stripped = text.strip()
    normalized = _normalize(stripped)
    tokens = TOKEN_PATTERN.findall(stripped.lower())
    sentences = _sentences(stripped)
    findings = _section_text(stripped, FINDINGS_HEADING, IMPRESSION_HEADING)
    impression = _section_text(stripped, IMPRESSION_HEADING, FINDINGS_HEADING)
    findings_heading = FINDINGS_HEADING.search(stripped) is not None
    impression_heading = IMPRESSION_HEADING.search(stripped) is not None
    contract = MODEL_SECTION_CONTRACT.get(
        str(report["model_id"]), {"findings": True, "impression": False}
    )
    findings_complete = bool(findings.strip()) if findings_heading else bool(stripped)
    impression_complete = bool(impression.strip()) if impression_heading else False
    section_contract_pass = (
        (not contract["findings"] or findings_complete)
        and (not contract["impression"] or impression_complete)
    )
    sentence_counts = Counter(sentences)
    duplicate_sentence_count = sum(count - 1 for count in sentence_counts.values() if count > 1)
    normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {
        "report_candidate_id": report["candidate_id"],
        "case_id": report["case_id"],
        "report_model_id": report["model_id"],
        "parent_cxr_candidate_id": report["parent_cxr_candidate_id"],
        "source_cxr_model_id": cxr["model_id"],
        "report_sha256": report["artifact"]["sha256"],
        "image_sha256": cxr["artifact"]["sha256"],
        "empty": not bool(stripped),
        "character_count": len(stripped),
        "token_count": len(tokens),
        "sentence_count": len(sentences),
        "findings_heading_present": findings_heading,
        "findings_complete": findings_complete,
        "impression_heading_present": impression_heading,
        "impression_complete": impression_complete,
        "impression_required_by_model_contract": contract["impression"],
        "section_contract_pass": section_contract_pass,
        "generic_report": normalized in GENERIC_NORMAL_REPORTS,
        "unsupported_temporal_comparison_language": bool(TEMPORAL_PATTERN.search(stripped)),
        "measurement_mention": bool(MEASUREMENT_PATTERN.search(stripped)),
        "repeated_sentence_count": duplicate_sentence_count,
        "repeated_sentence": duplicate_sentence_count > 0,
        "repeated_4gram_ratio": round(_repeated_ngram_ratio(tokens), 8),
        "normalized_report_sha256": normalized_hash,
        "normalized_template_frequency": normalized_frequency.get(normalized, 0),
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("cannot summarize an empty report group")

    def rate(key: str) -> float:
        return round(sum(bool(row[key]) for row in rows) / count, 8)

    normalized_counts = Counter(row["normalized_report_sha256"] for row in rows)
    return {
        "report_count": count,
        "empty_report_rate": rate("empty"),
        "findings_heading_rate": rate("findings_heading_present"),
        "findings_completeness_rate": rate("findings_complete"),
        "impression_heading_rate": rate("impression_heading_present"),
        "impression_completeness_rate": rate("impression_complete"),
        "section_contract_pass_rate": rate("section_contract_pass"),
        "generic_report_rate": rate("generic_report"),
        "no_change_or_prior_hallucination_rate": rate(
            "unsupported_temporal_comparison_language"
        ),
        "measurement_mention_rate_not_automatically_an_error": rate("measurement_mention"),
        "repeated_sentence_rate": rate("repeated_sentence"),
        "token_count": _quantiles([float(row["token_count"]) for row in rows]),
        "sentence_count": _quantiles([float(row["sentence_count"]) for row in rows]),
        "repeated_4gram_ratio": _quantiles(
            [float(row["repeated_4gram_ratio"]) for row in rows]
        ),
        "unique_normalized_report_count": len(normalized_counts),
        "unique_normalized_report_rate": round(len(normalized_counts) / count, 8),
        "largest_exact_template_count": max(normalized_counts.values()),
        "largest_exact_template_share": round(max(normalized_counts.values()) / count, 8),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TriCompose V1.1 Report Unimodal Evaluation",
        "",
        f"- Report candidates: {payload['counts']['reports']}",
        f"- CXR parents: {payload['counts']['cxr_candidates']}",
        "- Cohort type: fully synthetic; no reference report was supplied.",
        "",
        "## Per-report-model summary",
        "",
        "| Model | N | Empty | Findings complete | Impression complete | Contract pass | Generic | Prior/no-change | Unique | Largest template |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, summary in sorted(payload["by_report_model"].items()):
        lines.append(
            f"| {model} | {summary['report_count']} | "
            f"{summary['empty_report_rate']:.3f} | "
            f"{summary['findings_completeness_rate']:.3f} | "
            f"{summary['impression_completeness_rate']:.3f} | "
            f"{summary['section_contract_pass_rate']:.3f} | "
            f"{summary['generic_report_rate']:.3f} | "
            f"{summary['no_change_or_prior_hallucination_rate']:.3f} | "
            f"{summary['unique_normalized_report_rate']:.3f} | "
            f"{summary['largest_exact_template_share']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Metric availability",
            "",
            "- BLEU-1/2/3/4, ROUGE-L, METEOR: N/A without a real reference report.",
            "- BERTScore/ClinicalBERTScore/BioBERTScore: N/A without a real reference report.",
            "- CheXbert-F1, CheXpert-F1, RadGraph-F1, RadCliQ: N/A as unimodal factuality metrics without a real reference report.",
            "- Clinical correctness is evaluated on the EHR-report and report-CXR edges, not inferred from report text alone.",
            "",
            "## Interpretation guardrails",
            "",
            "- `unknown` is never converted to a negative finding.",
            "- Prior/no-change language is counted as unsupported because every report model received one current synthetic CXR and no prior image/report.",
            "- A measurement mention is reported descriptively; it is not automatically labelled a factual error.",
            "- Exact-template statistics measure diversity and genericity, not clinical accuracy.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    texts = {candidate_id: read_report_text(report) for candidate_id, report in reports.items()}
    normalized_frequency = Counter(_normalize(text.strip()) for text in texts.values())
    records = [
        _record(
            report,
            cxrs[str(report["parent_cxr_candidate_id"])],
            normalized_frequency=normalized_frequency,
        )
        for _, report in sorted(reports.items())
    ]
    by_report_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cxr_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_report_model[str(row["report_model_id"])].append(row)
        by_cxr_model[str(row["source_cxr_model_id"])].append(row)
        by_path[f"{row['source_cxr_model_id']}->{row['report_model_id']}"] .append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": {
            "name": "tricompose_report_unimodal_evaluator",
            "version": PRODUCER_VERSION,
            "gpu_inference_used": False,
        },
        "evaluation_scope": {
            "cohort": "fully_synthetic",
            "real_reference_report_supplied": False,
            "raw_source_target_supplied": False,
            "report_text_written_to_output": False,
        },
        "counts": {
            "reports": len(records),
            "cxr_candidates": len(cxrs),
            "cases": len({row["case_id"] for row in records}),
            "report_models": len(by_report_model),
            "cxr_models": len(by_cxr_model),
        },
        "source_runs": {
            "cxr": [str(Path(path).resolve(strict=True)) for path in args.cxr_run],
            "report": [str(Path(path).resolve(strict=True)) for path in args.report_run],
        },
        "metric_availability": {
            "text_overlap": {"status": "not_applicable_without_reference"},
            "semantic_similarity": {"status": "not_applicable_without_reference"},
            "reference_based_clinical_factuality": {
                "status": "not_applicable_without_reference"
            },
            "reference_free_structure": {"status": "computed"},
            "clinical_factuality": {"status": "delegated_to_cross_modal_edges"},
        },
        "overall": _group_summary(records),
        "by_report_model": {
            key: _group_summary(value) for key, value in sorted(by_report_model.items())
        },
        "by_cxr_model": {
            key: _group_summary(value) for key, value in sorted(by_cxr_model.items())
        },
        "by_generation_path": {
            key: _group_summary(value) for key, value in sorted(by_path.items())
        },
        "records": records,
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
        detail_path = write_private_json(temporary / "report_unimodal_details.json", payload)
        summary_payload = {key: value for key, value in payload.items() if key != "records"}
        summary_path = write_private_json(
            temporary / "report_unimodal_summary.json", summary_payload
        )
        markdown_path = write_private_text(
            temporary / "report_unimodal_summary.md", _markdown(payload)
        )
        output_hashes = {
            path.name: sha256_file(path)
            for path in (detail_path, summary_path, markdown_path)
        }
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": args.run_id,
                "report_count": payload["counts"]["reports"],
                "output_hashes": output_hashes,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

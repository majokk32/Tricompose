#!/usr/bin/env python3
"""Audit cached synthetic candidate scores and compare reports on the same CXR.

No model inference or artifact text/pixels are read. All results are descriptive
and retain the number of independent EHR cases separately from candidate rows.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from contracts import (
    CHEXPERT_FINDINGS, FINDING_STATES, PROTECTED_ROOT,
    commit_atomic_run, discard_atomic_run, new_atomic_run, require_inside,
    sha256_file, write_private_json, write_private_text,
)


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite cached score")
    return result


def finding_counts(image, report):
    if set(image) != set(CHEXPERT_FINDINGS) or set(report) != set(CHEXPERT_FINDINGS):
        raise ValueError("finding inventory mismatch")
    if not set(image.values()) <= FINDING_STATES or not set(report.values()) <= FINDING_STATES:
        raise ValueError("invalid finding state")
    result = dict.fromkeys(("cxr_positive", "cxr_negative", "positive_support",
                           "negative_support", "explicit_conflicts",
                           "unresolved_positive", "unresolved_negative",
                           "report_positive_cxr_unknown"), 0)
    for finding in CHEXPERT_FINDINGS:
        left, right = image[finding], report[finding]
        if left in {"positive", "negative"}:
            result[f"cxr_{left}"] += 1
            if left == right:
                result[f"{left}_support"] += 1
            elif right in {"positive", "negative"}:
                result["explicit_conflicts"] += 1
            else:
                result[f"unresolved_{left}"] += 1
        elif right == "positive":
            result["report_positive_cxr_unknown"] += 1
    return result


def ratio(top, bottom):
    return None if bottom == 0 else round(top / bottom, 8)


def sign(value):
    return 0 if abs(value) <= 1e-8 else (1 if value > 0 else -1)


def joined_rows(table, lineages, crossmodal, structures):
    def index(records, field):
        result = {r[field]: r for r in records}
        if len(result) != len(records):
            raise ValueError("duplicate candidate record")
        return result

    indices = [index(table, "triple_candidate_id"),
               index(lineages, "triple_candidate_id"),
               index(crossmodal, "report_candidate_id"),
               index(structures, "report_candidate_id")]
    if any(set(mapping) != set(indices[0]) for mapping in indices[1:]):
        raise ValueError("evidence does not cover exactly the candidate table")
    result = []
    for candidate_id, row in sorted(indices[0].items()):
        lineage = indices[1][candidate_id]["lineage"]
        evidence, structure = indices[2][candidate_id], indices[3][candidate_id]
        checks = (
            (row["case_id"], indices[1][candidate_id]["case_id"],
             evidence["case_id"], structure["case_id"]),
            (row["cxr_candidate_id"], lineage["cxr_candidate_id"],
             evidence["parent_cxr_candidate_id"], structure["parent_cxr_candidate_id"]),
            (lineage["cxr_sha256"], evidence["image_sha256"], structure["image_sha256"]),
            (lineage["report_sha256"], evidence["report_sha256"], structure["report_sha256"]),
            (row["report_model_id"], lineage["report_model_id"], evidence["report_model_id"],
             structure["report_model_id"]),
            (row["cxr_model_id"], lineage["cxr_model_id"], evidence["source_cxr_model_id"]),
            (int(row["cxr_seed"]), lineage["cxr_seed"]),
        )
        if any(len(set(values)) != 1 for values in checks):
            raise ValueError("candidate lineage/hash mismatch")
        counts = finding_counts(evidence["cxr_finding_states"], evidence["report_finding_states"])
        support = ratio(counts["positive_support"] + counts["negative_support"],
                        counts["cxr_positive"] + counts["cxr_negative"])
        if support is None or abs(support - number(row["report_cxr_support_recall"])) > 1e-7:
            raise ValueError("cached support score differs from named finding evidence")
        cosine = number(row["biovil_raw_cosine_secondary"])
        if abs(cosine - number(evidence["secondary_scores"]["biovil_report_cxr"])) > 1e-7:
            raise ValueError("cached BioViL evidence mismatch")
        result.append({
            "candidate_id": candidate_id, "case_id": row["case_id"],
            "cxr_id": row["cxr_candidate_id"], "cxr_model": row["cxr_model_id"],
            "cxr_seed": int(row["cxr_seed"]), "report_model": row["report_model_id"],
            "biovil_cosine": cosine, "support_recall": support,
            "total_hard_conflicts": int(row["total_hard_contradiction_count"]),
            "eligible": int(row["hard_gate_failure_count"]) == 0,
            "existing_rank": int(row["selection_rank_within_case"]),
            "selected": row["selected"].lower() == "true",
            "temporal_flag": bool(structure["unsupported_temporal_comparison_language"]),
            "token_count": int(structure["token_count"]), **counts,
        })
    return result


def summarize(rows):
    by_report, by_image = defaultdict(list), defaultdict(list)
    for row in rows:
        by_report[row["report_model"]].append(row)
        by_image[row["cxr_id"]].append(row)
    report_models = []
    for model, records in sorted(by_report.items()):
        positives = sum(r["cxr_positive"] for r in records)
        negatives = sum(r["cxr_negative"] for r in records)
        pos_support = sum(r["positive_support"] for r in records)
        neg_support = sum(r["negative_support"] for r in records)
        conflicts = sum(r["explicit_conflicts"] for r in records)
        report_models.append({
            "model": model, "reports": len(records),
            "positive_support": pos_support, "cxr_positive_facts": positives,
            "positive_support_recall": ratio(pos_support, positives),
            "negative_support": neg_support, "cxr_negative_facts": negatives,
            "negative_support_recall": ratio(neg_support, negatives),
            "explicit_conflicts": conflicts,
            "conflict_rate_over_known": ratio(conflicts, positives + negatives),
            "mean_biovil": round(statistics.mean(r["biovil_cosine"] for r in records), 8),
            "median_biovil": round(statistics.median(r["biovil_cosine"] for r in records), 8),
            "temporal_flags": sum(r["temporal_flag"] for r in records),
            "mean_tokens": round(statistics.mean(r["token_count"] for r in records), 2),
        })
    image_comparisons, pair_comparisons = [], []
    by_cxr = defaultdict(list)
    for image_id, records in sorted(by_image.items()):
        if len({r["report_model"] for r in records}) != len(records):
            raise ValueError("duplicate report model on one image")
        if len({(r["case_id"], r["cxr_model"], r["cxr_seed"], r["cxr_positive"],
                  r["cxr_negative"]) for r in records}) != 1:
            raise ValueError("one image has inconsistent reference states")
        by_cxr[records[0]["cxr_model"]].append(records[0])
        valid = [r for r in records if r["eligible"]]
        if valid:
            existing = min(valid, key=lambda r: r["existing_rank"])
            best_cosine = max(r["biovil_cosine"] for r in valid)
            best = [r for r in valid if abs(r["biovil_cosine"] - best_cosine) <= 1e-8]
            image_comparisons.append({
                "case_id": existing["case_id"], "cxr_id": image_id,
                "cxr_model": existing["cxr_model"], "cxr_seed": existing["cxr_seed"],
                "existing_top_report": existing["report_model"],
                "biovil_top_reports": ",".join(sorted(r["report_model"] for r in best)),
                "existing_top_cosine": existing["biovil_cosine"],
                "biovil_top_cosine": best_cosine,
                "same_report_top": any(r["candidate_id"] == existing["candidate_id"] for r in best),
                "existing_top_hard_conflicts": existing["total_hard_conflicts"],
                "existing_top_positive_support": existing["positive_support"],
                "existing_top_negative_support": existing["negative_support"],
                "biovil_top_min_hard_conflicts": min(r["total_hard_conflicts"] for r in best),
            })
        for left, right in combinations(sorted(records, key=lambda r: r["report_model"]), 2):
            label_sign = sign(left["support_recall"] - right["support_recall"])
            cosine_sign = sign(left["biovil_cosine"] - right["biovil_cosine"])
            state = ("concordant" if label_sign == cosine_sign else "discordant")
            if label_sign == 0 or cosine_sign == 0:
                state = "tied"
            pair_comparisons.append({
                "case_id": left["case_id"], "cxr_id": image_id,
                "left_report": left["report_model"], "right_report": right["report_model"],
                "support_delta_left_minus_right": round(left["support_recall"] - right["support_recall"], 8),
                "biovil_delta_left_minus_right": round(left["biovil_cosine"] - right["biovil_cosine"], 8),
                "same_total_hard_conflicts": left["total_hard_conflicts"] == right["total_hard_conflicts"],
                "ordering": state,
            })
    cxr_models = [{
        "model": model, "unique_images": len(records),
        "positive_heads": sum(r["cxr_positive"] for r in records),
        "known_heads": sum(r["cxr_positive"] + r["cxr_negative"] for r in records),
        "all_known_heads_positive_images": sum(r["cxr_negative"] == 0 for r in records),
    } for model, records in sorted(by_cxr.items())]
    per_case = []
    for case_id in sorted({r["case_id"] for r in rows}):
        comparisons = [r for r in pair_comparisons if r["case_id"] == case_id]
        strict = [r for r in comparisons if r["ordering"] != "tied"]
        equal_conflicts = [r for r in strict if r["same_total_hard_conflicts"]]
        per_case.append({
            "case_id": case_id, "same_image_report_pairs": len(comparisons),
            "strict_pairs": len(strict),
            "discordant_strict_pairs": sum(r["ordering"] == "discordant" for r in strict),
            "equal_conflict_strict_pairs": len(equal_conflicts),
            "equal_conflict_discordant_pairs": sum(r["ordering"] == "discordant" for r in equal_conflicts),
        })
    return {"counts": {"independent_ehr_cases": len(per_case), "candidate_rows": len(rows),
                       "unique_images": len(by_image), "same_image_report_pairs": len(pair_comparisons)},
            "report_models": report_models, "cxr_models": cxr_models,
            "per_case": per_case, "image_comparisons": image_comparisons,
            "pair_comparisons": pair_comparisons, "candidate_records": rows}


def csv_text(records):
    if not records:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def markdown(payload):
    counts = payload["counts"]
    lines = ["# Cached candidate scorer audit", "",
             f"{counts['candidate_rows']} candidate rows, {counts['unique_images']} synthetic CXRs, "
             f"{counts['independent_ehr_cases']} independent EHR cases. All statistics are descriptive.",
             "", "## Report models on the same image inventory", "",
             "| Model | Positive support / XRV positive facts | Negative support / XRV negative facts | Explicit conflicts | Mean BioViL | Temporal flags |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in payload["report_models"]:
        lines.append(f"| {r['model']} | {r['positive_support']}/{r['cxr_positive_facts']} | "
                     f"{r['negative_support']}/{r['cxr_negative_facts']} | {r['explicit_conflicts']} | "
                     f"{r['mean_biovil']:.3f} | {r['temporal_flags']}/{r['reports']} |")
    lines.extend(["", "## Frozen XRV label saturation", "",
                  "| CXR model | Unique images | Positive / known heads | All known heads positive images |",
                  "|---|---:|---:|---:|"])
    for r in payload["cxr_models"]:
        lines.append(f"| {r['model']} | {r['unique_images']} | {r['positive_heads']}/{r['known_heads']} | "
                     f"{r['all_known_heads_positive_images']} |")
    image_rows = payload["image_comparisons"]
    agree = sum(r["same_report_top"] for r in image_rows)
    lines.extend(["", f"The existing selector and secondary BioViL-T choose the same top report "
                  f"on {agree}/{len(image_rows)} images. A different top report does not identify which scorer is correct.",
                  "", "## Same-image report ordering", "",
                  "| Case | Strictly ordered report pairs | Opposite support/BioViL order | Equal-hard-conflict strict pairs | Opposite order among those |",
                  "|---|---:|---:|---:|---:|"])
    for r in payload["per_case"]:
        lines.append(f"| {r['case_id']} | {r['strict_pairs']} | {r['discordant_strict_pairs']} | "
                     f"{r['equal_conflict_strict_pairs']} | {r['equal_conflict_discordant_pairs']} |")
    lines.extend(["", "Unknown/uncertain labels never become negative or contradictions. Positive support is "
                  "reported separately from negative support. XRV operating-point-normalized scores are not "
                  "calibrated probabilities; label counts are verifier evidence, not image ground truth.",
                  "", f"The {counts['same_image_report_pairs']} within-image report comparisons share "
                  f"{counts['independent_ehr_cases']} EHR cases and {counts['unique_images']} images; "
                  "the pair count is not the number of independent patients.",
                  "The existing protected selections remain the source decisions; this audit adds no new winner."])
    if "historical_swap_context" in payload:
        lines.extend(["", "## Broader cached synthetic context", "",
                      "| Swap arm | Pairs | Score below untouched control | Mean control-minus-swap cosine |",
                      "|---|---:|---:|---:|"])
        for arm, r in sorted(payload["historical_swap_context"]["summary"]["overall"].items()):
            lines.append(f"| {arm} | {r['pairs']} | {100 * r['fraction_score_decreased']:.2f}% | "
                         f"{r['mean_control_minus_swap']:.3f} |")
        lines.extend(["", "The historical BioViL swap sensitivity is separate from within-case report selection. "
                      "Cross-case swaps may remain clinically compatible, and this pilot has no independent clinical truth."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--crossmodal-details", required=True)
    parser.add_argument("--report-structure", required=True)
    parser.add_argument("--historical-analysis")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    run = require_inside(args.selection_run, PROTECTED_ROOT, must_exist=True)
    manifest = json.loads((run / "manifest.json").read_text())
    sources = {}
    for name in ("candidate_score_table.csv", "candidate_score_table.jsonl"):
        path = require_inside(run / name, run, must_exist=True)
        digest = sha256_file(path)
        if digest != manifest["artifacts"][name]["sha256"]:
            raise ValueError("source selection artifact hash mismatch")
        sources[name] = digest
    with (run / "candidate_score_table.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    lineages = [json.loads(line) for line in (run / "candidate_score_table.jsonl").read_text().splitlines() if line]
    if len(table) > 128:
        raise ValueError("bounded cached audit is limited to 128 rows")
    details = require_inside(args.crossmodal_details, PROTECTED_ROOT, must_exist=True)
    structure = require_inside(args.report_structure, PROTECTED_ROOT, must_exist=True)
    records = joined_rows(table, lineages, json.loads(details.read_text())["records"],
                          json.loads(structure.read_text())["records"])
    payload = summarize(records)
    payload.update({"schema_version": "tricompose-candidate-scorer-audit-v1",
                    "primary_metric_eligible": False, "selection_changed": False,
                    "source_hashes": {**sources, "crossmodal_details": sha256_file(details),
                                      "report_structure": sha256_file(structure),
                                      "analysis_code": sha256_file(__file__)}})
    if args.historical_analysis:
        historical = require_inside(args.historical_analysis, PROTECTED_ROOT, must_exist=True)
        old = json.loads(historical.read_text())
        payload["historical_swap_context"] = {k: old[k] for k in (
            "schema_version", "status", "clinical_truth_available",
            "paper_final_test_independent", "summary")}
        payload["source_hashes"]["historical_analysis"] = sha256_file(historical)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        artifacts = [write_private_json(temporary / "scorer_audit.json", payload),
                     write_private_text(temporary / "scorer_audit.md", markdown(payload))]
        for name in ("report_models", "cxr_models", "per_case", "image_comparisons",
                     "pair_comparisons", "candidate_records"):
            artifacts.append(write_private_text(temporary / f"{name}.csv", csv_text(payload[name])))
        write_private_json(temporary / "manifest.json", {
            "schema_version": payload["schema_version"], "run_id": args.run_id,
            "counts": payload["counts"], "selection_changed": False,
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in artifacts},
        })
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(json.dumps({"status": "completed", "counts": payload["counts"]}))


if __name__ == "__main__":
    main()

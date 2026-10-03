#!/usr/bin/env python3
"""Join separated Qwen evidence to an unchanged synthetic static selection.

No new ranking, threshold, clinical attribution or repair action is produced.
Only finding-state diagnostics and original fixed/selected comparisons.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

from contracts import (
    FINDING_STATES, PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, read_json, require_inside, sha256_file,
    write_private_json, write_private_text,
)
from verify_candidate_findings_qwen import FINDINGS, SCHEMA


def ratio(top, bottom):
    return None if bottom == 0 else round(top / bottom, 8)


def compare(reference, candidate):
    if set(reference) != set(FINDINGS) or set(candidate) != set(FINDINGS):
        raise ValueError("finding scope mismatch")
    if any(state not in FINDING_STATES for state in [*reference.values(), *candidate.values()]):
        raise ValueError("invalid comparison state")
    result = {"known": 0, "comparable": 0, "supported": 0,
              "positive_supported": 0, "negative_supported": 0,
              "contradictions": 0, "unresolved": 0}
    for name in FINDINGS:
        left, right = reference[name], candidate[name]
        if left not in {"positive", "negative"}:
            continue
        result["known"] += 1
        if right not in {"positive", "negative"}:
            result["unresolved"] += 1
            continue
        result["comparable"] += 1
        if left == right:
            result["supported"] += 1
            result[f"{left}_supported"] += 1
        else:
            result["contradictions"] += 1
    return {**result, "coverage": ratio(result["comparable"], result["known"]),
            "support_over_known": ratio(result["supported"], result["known"]),
            "contradiction_over_known": ratio(result["contradictions"], result["known"]),
            "agreement_on_comparable": ratio(result["supported"], result["comparable"])}


def subset(states):
    if not isinstance(states, dict) or not set(FINDINGS) <= set(states):
        raise ValueError("source finding inventory is incomplete")
    return {name: states[name] for name in FINDINGS}


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("empty or duplicate records")
    return result


def join_rows(table, evidence, scores):
    if (scores.get("schema_version") != SCHEMA or
            scores.get("primary_metric_eligible") is not False or
            scores.get("targeted_repair_approved") is not False or
            scores.get("selection_changed") is not False or
            scores.get("image_report_separation_enforced") is not True or
            scores.get("verifier_received_ehr_scores_or_winner_flags") is not False or
            scores.get("finding_order") != list(FINDINGS)):
        raise ValueError("unexpected verifier scope")
    source = indexed(evidence, "report_candidate_id")
    index = indexed(scores["records"], "candidate_id")
    table_ids = {row["triple_candidate_id"] for row in table}
    image_ids = {row["cxr_candidate_id"] for row in table}
    if (len(table_ids) != len(table) or set(source) != table_ids or
            set(index) != table_ids | image_ids or table_ids & image_ids):
        raise ValueError("evidence must cover exactly the fixed candidate inventory")
    output, facts_by_case = [], {}
    for old in table:
        report_id, image_id = old["triple_candidate_id"], old["cxr_candidate_id"]
        item, image, report = source[report_id], index[image_id], index[report_id]
        if (old["report_candidate_id"] != report_id or
                old["case_id"] != item["case_id"] or
                image_id != item["parent_cxr_candidate_id"] or
                old["cxr_model_id"] != item["source_cxr_model_id"] or
                old["report_model_id"] != item["report_model_id"] or
                image["input_kind"] != "image" or report["input_kind"] != "report" or
                image["artifact_sha256"] != item["image_sha256"] or
                report["artifact_sha256"] != item["report_sha256"]):
            raise ValueError("candidate/hash lineage mismatch")
        for record in (image, report):
            if record["contract_status"] not in {"complete", "failed_unavailable"}:
                raise ValueError("unknown verifier contract state")
            if (record["contract_status"] == "failed_unavailable" and
                    set(record["states"].values()) != {"unknown"}):
                raise ValueError("failed response must not supply finding evidence")
        ehr = subset(item["ehr_finding_states"])
        if old["case_id"] in facts_by_case and facts_by_case[old["case_id"]] != ehr:
            raise ValueError("fixed EHR evidence changed between candidates")
        facts_by_case[old["case_id"]] = ehr
        row = {**old, "qwen_scope": "secondary_diagnostic_not_clinical_truth",
               "qwen_primary_metric_eligible": False, "qwen_targeted_repair_approved": False,
               "qwen_image_contract_status": image["contract_status"],
               "qwen_report_contract_status": report["contract_status"]}
        edges = {"ehr_cxr": compare(ehr, image["states"]),
                 "ehr_report": compare(ehr, report["states"]),
                 "cxr_report": compare(image["states"], report["states"]),
                 "xrv_image_state_check": compare(subset(item["cxr_finding_states"]), image["states"]),
                 "chexbert_report_state_check": compare(subset(item["report_finding_states"]), report["states"])}
        for name, metrics in edges.items():
            row.update({f"qwen_{name}_{key}": value for key, value in metrics.items()})
        row["qwen_image_states"] = json.dumps(image["states"], sort_keys=True)
        row["qwen_report_states"] = json.dumps(report["states"], sort_keys=True)
        output.append(row)
    return output


def pooled(rows):
    result = {}
    for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
        values = {key: sum(row[f"qwen_{edge}_{key}"] for row in rows)
                  for key in ("known", "comparable", "supported", "positive_supported",
                              "negative_supported", "contradictions", "unresolved")}
        result[edge] = {**values, "coverage": ratio(values["comparable"], values["known"]),
                       "support_over_known": ratio(values["supported"], values["known"]),
                       "contradiction_over_known": ratio(values["contradictions"], values["known"])}
    return result


def summarize(rows):
    selected = [row for row in rows if row["selected"].lower() == "true"]
    fixed = [row for row in rows if row["cxr_model_id"] == "chexgenbench_sana"
             and row["cxr_seed"] == "0" and row["report_model_id"] == "maira2"]
    cases = {row["case_id"] for row in rows}
    if (len(selected) != len(cases) or len(fixed) != len(cases) or
            {row["case_id"] for row in selected} != cases or
            {row["case_id"] for row in fixed} != cases):
        raise ValueError("fixed and selected case denominators differ")
    images = {row["cxr_candidate_id"]: row for row in rows}
    qwen_xrv = {key: sum(row[f"qwen_xrv_image_state_check_{key}"] for row in images.values())
                for key in ("known", "comparable", "supported", "contradictions", "unresolved")}
    return {"independent_ehr_cases": len(cases), "candidate_rows": len(rows),
            "unique_images": len(images), "fixed_path": "sana_seed0_maira2",
            "fixed": pooled(fixed), "original_static_selections": pooled(selected),
            "image_state_disagreement": {**qwen_xrv,
                "disagreement_over_comparable": ratio(qwen_xrv["contradictions"], qwen_xrv["comparable"])},
            "selection_changed": False, "primary_metric_eligible": False,
            "targeted_repair_approved": False, "clinical_adjudication_pending": True,
            "model_pretraining_independence_established": False}


def markdown(summary):
    lines = ["# Separated secondary Qwen evidence / 模态分离的辅助核验", "",
             "The original winner flags, thresholds and candidate artifacts are unchanged.",
             "Qwen saw an image or a report, never both or any EHR/scores/winner flags.",
             "These are auxiliary frozen-model judgments, not clinical ground truth.", "",
             "| Original method | Edge | Known | Comparable | Supported | Contradictions | Coverage |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for method in ("fixed", "original_static_selections"):
        for edge, values in summary[method].items():
            lines.append(f"| {method} | {edge} | {values['known']} | {values['comparable']} | "
                         f"{values['supported']} | {values['contradictions']} | {values['coverage']} |")
    lines += ["", "Unknown/uncertain are never negatives, contradictions or support.",
              "A zero known/comparable denominator remains unavailable, not perfect agreement.",
              "Two EHRs and 48 dependent report candidates are not 48 independent patients.",
              "Qwen-image/XRV and Qwen-report/CheXbert differences diagnose scorer disagreement;",
              "they do not identify which evaluator is correct or authorize targeted repair.",
              "Qwen pretraining/data independence and clinical accuracy are not established.",
              "Use protected per-finding results for review; no new scalar score/ranking was fitted.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run", required=True)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--crossmodal-details", required=True)
    parser.add_argument("--crossmodal-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    os.umask(0o007)
    try:
        review = require_inside(args.review_run, PROTECTED_ROOT, must_exist=True)
        selection = require_inside(args.selection_run, PROTECTED_ROOT, must_exist=True)
        details = require_inside(args.crossmodal_details, PROTECTED_ROOT, must_exist=True)
        paths = {"review_scores": review / "scores.json",
                 "candidate_table": selection / "candidate_score_table.csv",
                 "crossmodal_details": details}
        if sha256_file(details) != args.crossmodal_sha256:
            raise ValueError("crossmodal details hash mismatch")
        for run, name in ((review, "scores.json"), (selection, "candidate_score_table.csv")):
            if sha256_file(run / name) != read_json(run / "manifest.json")["artifacts"][name]["sha256"]:
                raise ValueError("review/selection artifact hash mismatch")
        with paths["candidate_table"].open(newline="", encoding="utf-8") as stream:
            original = list(csv.DictReader(stream))
        crossmodal = read_json(details)
        if crossmodal.get("evaluation_scope", {}).get("cohort") != "fully_synthetic":
            raise ValueError("synthetic evidence required")
        rows = join_rows(original, crossmodal["records"], read_json(paths["review_scores"]))
        summary = summarize(rows)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        files = [write_private_text(temporary / "candidate_evidence.csv", stream.getvalue()),
                 write_private_json(temporary / "summary.json", summary),
                 write_private_text(temporary / "summary.md", markdown(summary))]
        write_private_json(temporary / "manifest.json", {
            "schema_version": "tricompose-secondary-qwen-evidence-analysis-v1",
            "run_id": args.run_id, "selection_changed": False,
            "primary_metric_eligible": False, "targeted_repair_approved": False,
            "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
            "program_sha256": sha256_file(__file__),
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_secondary_evidence_analysis",
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

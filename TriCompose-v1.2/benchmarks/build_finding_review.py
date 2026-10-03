#!/usr/bin/env python3
"""Build a small metadata-only synthetic per-finding evidence/review bundle.

No inference, image pixels, report text, real patients, new scores or actions.
The strict pilot cap permits lightweight cached-metadata work on the login
node. Review flags do not assign clinical fault or authorize regeneration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

from analyze_qwen_evidence import join_rows, subset
from contracts import (
    FINDING_STATES, PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, read_json, require_inside, sha256_file,
    write_private_json, write_private_text,
)
from crossmodal_metrics import DIRECT_EHR_TO_CHEXPERT, ehr_direct_finding_states
from verify_candidate_findings_qwen import FINDINGS

SCHEMA = "tricompose-cached-finding-review-v1"
EXPLICIT = frozenset({"positive", "negative"})
PAIRS = {
    "ehr_xrv": ("ehr", "xrv"),
    "ehr_chexbert": ("ehr", "chexbert"),
    "xrv_chexbert": ("xrv", "chexbert"),
    "ehr_qwen_image": ("ehr", "qwen_image"),
    "ehr_qwen_report": ("ehr", "qwen_report"),
    "qwen_image_report": ("qwen_image", "qwen_report"),
    "image_evaluators": ("xrv", "qwen_image"),
    "report_extractors": ("chexbert", "qwen_report"),
}


def relation(left, right, *, available=True):
    if left not in FINDING_STATES or right not in FINDING_STATES:
        raise ValueError("invalid finding state")
    if not available:
        return "not_comparable", "verifier_contract_unavailable"
    if "uncertain" in (left, right):
        return "not_comparable", "uncertain_state"
    if "unknown" in (left, right):
        return "unknown", "unknown_or_unmentioned"
    if left == right:
        return "support", "explicit_same_state"
    return "contradiction", "explicit_opposite_states_diagnostic_only"


def evidence_id(candidate_id, finding):
    return "fact_" + hashlib.sha256(f"{candidate_id}\n{finding}".encode()).hexdigest()[:24]


def direct_provenance(facts, finding, expected_state):
    mapped = subset(ehr_direct_finding_states(facts))
    if mapped[finding] != expected_state:
        raise ValueError("staged direct EHR state differs from cached reference")
    fact_ids = [name for name, target in DIRECT_EHR_TO_CHEXPERT.items() if target == finding]
    if len(fact_ids) != 1:
        raise ValueError("finding has no unique existing direct mapping")
    fact_id = fact_ids[0]
    fact = facts["direct_facts"][fact_id]
    evidence, fields = fact.get("evidence", []), fact.get("source_fields", [])
    if not isinstance(evidence, list) or not isinstance(fields, list):
        raise ValueError("invalid staged provenance")
    if expected_state != "unknown" and (not evidence or not fields):
        raise ValueError("asserted direct fact lacks evidence")
    if expected_state == "unknown" and (evidence or fields):
        raise ValueError("unknown fact unexpectedly carries evidence")
    return {"fact_id": fact_id, "evidence_ids": evidence, "source_fields": fields,
            "json_pointer": f"/direct_facts/{fact_id}",
            "scope": "existing_direct_finding_proxy_not_image_ground_truth",
            "weak_clinical_context_promoted": False}


def build_rows(table, crossmodal, lineages, staging_facts):
    if not table or len(table) > 48 or len({r["case_id"] for r in table}) > 2:
        raise ValueError("only the bounded two-case/48-candidate pilot is supported")
    if (len({r["triple_candidate_id"] for r in table}) != len(table) or
            set(crossmodal) != {r["triple_candidate_id"] for r in table} or
            set(lineages) != set(crossmodal)):
        raise ValueError("candidate metadata inventory mismatch")
    facts, reviews = [], []
    image_states = {}
    case_hashes = {}
    for candidate in table:
        candidate_id, case = candidate["triple_candidate_id"], candidate["case_id"]
        source, lineage = crossmodal[candidate_id], lineages[candidate_id]
        if (source["case_id"] != case or
                lineage["cxr_candidate_id"] != candidate["cxr_candidate_id"] or
                lineage["report_candidate_id"] != candidate["report_candidate_id"] or
                lineage["cxr_sha256"] != source["image_sha256"] or
                lineage["report_sha256"] != source["report_sha256"]):
            raise ValueError("candidate/hash lineage mismatch")
        ehr_hashes = (lineage["ehr_sha256"], lineage["ehr_facts_sha256"])
        if case in case_hashes and case_hashes[case] != ehr_hashes:
            raise ValueError("fixed EHR hashes differ within a case")
        case_hashes[case] = ehr_hashes
        states = {"ehr": subset(source["ehr_finding_states"]),
                  "xrv": subset(source["cxr_finding_states"]),
                  "chexbert": subset(source["report_finding_states"]),
                  "qwen_image": json.loads(candidate["qwen_image_states"]),
                  "qwen_report": json.loads(candidate["qwen_report_states"])}
        if any(set(value) != set(FINDINGS) or not set(value.values()) <= FINDING_STATES
               for value in states.values()):
            raise ValueError("invalid named state inventory")
        case_facts = staging_facts[case]
        image_id = candidate["cxr_candidate_id"]
        image_scope = (case, lineage["ehr_sha256"], lineage["ehr_facts_sha256"],
                       lineage["cxr_sha256"], states["xrv"], states["qwen_image"])
        if image_id in image_states and image_states[image_id] != image_scope:
            raise ValueError("one image has inconsistent case/hash/verifier states")
        image_states[image_id] = image_scope
        image_ok = candidate["qwen_image_contract_status"] == "complete"
        report_ok = candidate["qwen_report_contract_status"] == "complete"
        if (not image_ok and set(states["qwen_image"].values()) != {"unknown"} or
                not report_ok and set(states["qwen_report"].values()) != {"unknown"}):
            raise ValueError("unavailable verifier must not supply facts")
        candidate_facts = []
        for finding in FINDINGS:
            provenance = direct_provenance(case_facts, finding, states["ehr"][finding])
            relations, reasons = {}, {}
            for name, (left, right) in PAIRS.items():
                available = (image_ok or "qwen_image" not in (left, right)) and (
                    report_ok or "qwen_report" not in (left, right))
                relations[name], reasons[name] = relation(
                    states[left][finding], states[right][finding], available=available)
            record = {"evidence_id": evidence_id(candidate_id, finding),
                      "case_id": case, "triple_candidate_id": candidate_id,
                      "cxr_candidate_id": image_id,
                      "report_candidate_id": candidate["report_candidate_id"],
                      "finding": finding, "original_selected": candidate["selected"].lower() == "true",
                      "artifact_hashes": {key: lineage[key] for key in (
                          "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")},
                      "states": {name: values[finding] for name, values in states.items()},
                      "ehr_provenance": provenance, "relations": relations,
                      "relation_reasons": reasons,
                      "clinical_severity": None, "clinical_error_confirmed": False,
                      "automatic_repair_eligible": False,
                      "dependency_note": "Four reports depend on one CXR; both Qwen passes share one model; no independent votes."}
            candidate_facts.append(record)
        facts.extend(candidate_facts)
        opposed = {name: [record["evidence_id"] for record in candidate_facts
                          if record["relations"][name] == "contradiction"] for name in PAIRS}
        reasons = []
        if not image_ok:
            reasons.append("secondary_image_contract_unavailable")
        if not report_ok:
            reasons.append("secondary_report_contract_unavailable")
        for pair in PAIRS:
            if opposed[pair]:
                reasons.append(f"{pair}_opposition_signal")
        known_ehr = [record for record in candidate_facts if record["states"]["ehr"] in EXPLICIT]
        for pair in ("ehr_qwen_image", "ehr_qwen_report"):
            if known_ehr and not any(record["relations"][pair] in {"support", "contradiction"}
                                     for record in known_ehr):
                reasons.append(f"{pair}_no_comparable_direct_fact")
        reviews.append({"case_id": case, "triple_candidate_id": candidate_id,
                        "cxr_candidate_id": image_id,
                        "original_selected": candidate["selected"].lower() == "true",
                        "fixed_path_control": candidate["cxr_model_id"] == "chexgenbench_sana"
                            and candidate["cxr_seed"] == "0" and candidate["report_model_id"] == "maira2",
                        "review_reasons": reasons, "opposition_evidence_ids": opposed,
                        "finding_evidence_ids": [record["evidence_id"] for record in candidate_facts],
                        "known_direct_ehr_facts": len(known_ehr),
                        "independent_clinical_review_status": "pending",
                        "confirmed_faulty_modality": None, "automatic_repair_eligible": False})
    if len({record["evidence_id"] for record in facts}) != len(facts):
        raise ValueError("evidence ID collision")
    return facts, reviews


def summarize(facts, reviews):
    image_facts = {}
    for record in facts:
        image_facts.setdefault((record["cxr_candidate_id"], record["finding"]), record)
    reason_counts = Counter(reason for record in reviews for reason in record["review_reasons"])
    edge_counts = {}
    for name in PAIRS:
        # Each image-finding contributes once, rather than once per report.
        records = image_facts.values() if name in {"ehr_xrv", "ehr_qwen_image", "image_evaluators"} else facts
        counter = Counter(record["relations"][name] for record in records)
        edge_counts[name] = {key: counter[key] for key in ("support", "contradiction", "unknown", "not_comparable")}
    return {"schema_version": SCHEMA, "counts": {
                "independent_ehr_cases": len({row["case_id"] for row in facts}),
                "candidate_triples": len(reviews), "fact_rows": len(facts),
                "unique_image_finding_pairs": len(image_facts),
                "selected_triples": sum(row["original_selected"] for row in reviews),
                "review_required_triples": sum(bool(row["review_reasons"]) for row in reviews)},
            "finding_order": list(FINDINGS), "relation_counts": edge_counts,
            "review_reason_counts": dict(sorted(reason_counts.items())),
            "selection_changed": False, "primary_metric_eligible": False,
            "targeted_repair_approved": False, "clinical_localization_accuracy": None,
            "policy_executed": False, "clinical_fault_assigned": False,
            "interpretation": "Cached evidence/triage only. Agreement, disagreement and missingness are not clinical truth."}


def csv_text(facts):
    records = []
    for fact in facts:
        records.append({key: value if not isinstance(value, (dict, list)) else json.dumps(value, sort_keys=True)
                        for key, value in fact.items()})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def markdown(summary, facts):
    counts = summary["counts"]
    lines = ["# Cached finding evidence / 逐 finding 核查", "",
             f"{counts['candidate_triples']} triples × {len(FINDINGS)} findings = {counts['fact_rows']} rows; "
             f"only {counts['independent_ehr_cases']} independent synthetic EHRs.",
             "No inference, raw image/report reading, new scores, winner changes or repair.", "",
             "| Relation | Support | Opposite explicit states | Unknown | Not comparable |",
             "|---|---:|---:|---:|---:|"]
    for edge, counts in summary["relation_counts"].items():
        lines.append(f"| {edge} | {counts['support']} | {counts['contradiction']} | "
                     f"{counts['unknown']} | {counts['not_comparable']} |")
    lines += ["", "Image-only rows are deduplicated over images/findings; report edges use candidate/finding rows.",
              "`contradiction` means a frozen-state opposition signal, not a confirmed medical error.",
              "Unknown and uncertainty never become negative or supporting evidence.", "",
              "## Original selected-pair Qwen oppositions", "",
              "| Synthetic case | Finding | Qwen image | Qwen report | Finding evidence ID |",
              "|---|---|---|---|---|"]
    for fact in facts:
        if fact["original_selected"] and fact["relations"]["qwen_image_report"] == "contradiction":
            lines.append(f"| {fact['case_id']} | {fact['finding']} | {fact['states']['qwen_image']} | "
                         f"{fact['states']['qwen_report']} | {fact['evidence_id']} |")
    lines += ["", "## Review boundary", "",
              "All review reasons retain finding IDs and input hashes. No clinical fault was assigned.",
              "`adjudication_template.jsonl` is blank, pending qualified review; it is not a gold-label set.",
              "The cached-state review queue is not blinded clinical evaluation and must not be scored as one.",
              "Generated reports share an image; Qwen passes share a model. Agreement is not independent voting.",
              "Missing/severity/location/device/temporal evidence is not fabricated to enable localization.",
              "Do not change a winner or launch regeneration solely from these unvalidated signals.", ""]
    return "\n".join(lines)


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_inputs(args):
    review = require_inside(args.review_run, PROTECTED_ROOT, must_exist=True)
    analysis = require_inside(args.analysis_run, PROTECTED_ROOT, must_exist=True)
    selection = require_inside(args.selection_run, PROTECTED_ROOT, must_exist=True)
    staging = require_inside(args.staging_run, PROTECTED_ROOT, must_exist=True)
    details = require_inside(args.crossmodal_details, PROTECTED_ROOT, must_exist=True)
    analysis_manifest = read_json(analysis / "manifest.json")
    if (analysis_manifest.get("schema_version") != "tricompose-secondary-qwen-evidence-analysis-v1" or
            analysis_manifest.get("selection_changed") is not False or
            analysis_manifest.get("primary_metric_eligible") is not False or
            analysis_manifest.get("targeted_repair_approved") is not False):
        raise ValueError("unexpected analysis scope")
    sources = {"review_scores": review / "scores.json", "candidate_table": selection / "candidate_score_table.csv",
               "crossmodal_details": details, "candidate_evidence": analysis / "candidate_evidence.csv",
               "lineages": selection / "candidate_score_table.jsonl", "staging_manifest": staging / "run_manifest.json"}
    for name in ("review_scores", "candidate_table", "crossmodal_details"):
        if sha256_file(sources[name]) != analysis_manifest["source_sha256"][name]:
            raise ValueError("source file changed after Qwen analysis")
    for run, name in ((review, "scores.json"), (selection, "candidate_score_table.csv"),
                      (selection, "candidate_score_table.jsonl"), (analysis, "candidate_evidence.csv")):
        if sha256_file(run / name) != read_json(run / "manifest.json")["artifacts"][name]["sha256"]:
            raise ValueError("manifest-bound source hash mismatch")
    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    original, supplied = read_csv(sources["candidate_table"]), read_csv(sources["candidate_evidence"])
    scores, crossmodal = read_json(sources["review_scores"]), read_json(details)
    if crossmodal.get("evaluation_scope", {}).get("cohort") != "fully_synthetic":
        raise ValueError("fully synthetic evidence required")
    joined = join_rows(original, crossmodal["records"], scores)
    if len(joined) != len(supplied) or len(joined) > 48:
        raise ValueError("unexpected pilot size")
    for expected, row in zip(joined, supplied):
        if set(expected) != set(row) or any(row[key] != ("" if value is None else str(value)) for key, value in expected.items()):
            raise ValueError("Qwen companion differs from reproducible cached join")
    lineages = {row["triple_candidate_id"]: row for row in read_lines(sources["lineages"])}
    if len(lineages) != len(original):
        raise ValueError("lineage inventory mismatch")
    facts = {}
    for case in sorted({row["case_id"] for row in original}):
        path = require_inside(staging / "cases" / case / "ehr_facts.json", staging, must_exist=True)
        fingerprint = sha256_file(path)
        expected = {row["lineage"]["ehr_facts_sha256"] for row in lineages.values() if row["case_id"] == case}
        if expected != {fingerprint}:
            raise ValueError("staged EHR facts hash mismatch")
        payload = read_json(path)
        if payload.get("case_id") != case:
            raise ValueError("staging case mismatch")
        facts[case] = payload
        sources[f"staged_facts_{case}"] = path
    return original, supplied, {row["report_candidate_id"]: row for row in crossmodal["records"]}, {
        key: row["lineage"] for key, row in lineages.items()}, facts, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review-run", "analysis-run", "selection-run", "staging-run", "crossmodal-details", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        original, table, source, lineages, staging_facts, inputs = load_inputs(args)
        facts, reviews = build_rows(table, source, lineages, staging_facts)
        summary = summarize(facts, reviews)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        files = [write_private_text(temporary / "fact_evidence.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in facts)),
                 write_private_text(temporary / "fact_evidence.csv", csv_text(facts)),
                 write_private_text(temporary / "review_items.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in reviews)),
                 write_private_json(temporary / "summary.json", summary),
                 write_private_text(temporary / "summary.md", markdown(summary, facts)),
                 write_private_text(temporary / "adjudication_template.jsonl", "".join(json.dumps({
                     "case_id": row["case_id"], "triple_candidate_id": row["triple_candidate_id"],
                     "clinical_review_status": "pending", "reviewer_id": None,
                     "supported_evidence_ids": [], "contradicted_evidence_ids": [],
                     "not_assessable_evidence_ids": [], "confirmed_faulty_modality": None,
                     "independent_artifact_review_complete": False}, sort_keys=True) + "\n" for row in reviews))]
        write_private_json(temporary / "manifest.json", {
            "schema_version": SCHEMA, "run_id": args.run_id, "metadata_only": True,
            "gpu_inference_used": False, "image_pixels_or_report_text_opened": False,
            "selection_changed": False, "targeted_repair_approved": False,
            "counts": summary["counts"], "source_sha256": {key: sha256_file(path) for key, path in inputs.items()},
            "source_paths": {key: str(path) for key, path in inputs.items()},
            "program_sha256": sha256_file(__file__),
            "existing_mapping_sha256": sha256_file(Path(ehr_direct_finding_states.__code__.co_filename)),
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_metadata_only_finding_review",
                      "counts": summary["counts"], "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

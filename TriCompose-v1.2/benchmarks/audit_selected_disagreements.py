#!/usr/bin/env python3
"""Audit selected synthetic Qwen oppositions without inference or text/pixels.

Checks cached extraction, named-state joins, actual artifact byte hashes and
frozen XRV operating points. This is an engineering audit, not adjudication.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

from build_finding_review import SCHEMA as REVIEW_SCHEMA, build_rows, load_inputs, read_lines
from contracts import (
    FINDING_STATES, PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, load_cxr_candidates,
    load_report_candidates, new_atomic_run, read_json, require_inside,
    sha256_file, write_private_json, write_private_text,
)
from extract_cxr_labels_xrv import _state
from verify_candidate_findings_qwen import (
    FINDINGS, IMAGE_PROMPT, REPORT_PROMPT, digest_text, validate_inventory,
)
from xrv_calibration import validate_bundle

SCHEMA = "tricompose-selected-disagreement-engineering-audit-v1"


def index(records, key):
    result = {row[key]: row for row in records}
    if not records or len(result) != len(records):
        raise ValueError("empty or duplicate cached inventory")
    return result


def check_response(record, *, expected_hash, kind, max_tokens):
    if record["input_kind"] != kind or record["artifact_sha256"] != expected_hash:
        raise ValueError("verifier/artifact hash mismatch")
    if set(record["states"]) != set(FINDINGS) or not set(record["states"].values()) <= FINDING_STATES:
        raise ValueError("named finding inventory mismatch")
    complete = record["contract_status"] == "complete"
    if record["contract_status"] not in {"complete", "failed_unavailable"}:
        raise ValueError("unsupported verifier response status")
    tokens = record["output_tokens"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("invalid response token metadata")
    if not isinstance(record["token_limit_reached"], bool) or record["token_limit_reached"] != (tokens >= max_tokens):
        raise ValueError("inconsistent token-limit metadata")
    if complete and record["token_limit_reached"]:
        raise ValueError("token-limited response cannot supply evidence")
    if not complete and set(record["states"].values()) != {"unknown"}:
        raise ValueError("unavailable response cannot supply evidence")
    return {"contract_status": record["contract_status"],
            "input_tokens": record["input_tokens"], "output_tokens": tokens,
            "token_limit_reached": record["token_limit_reached"],
            "raw_response_reparse_available": False}


def audit_records(facts, qwen, cxrs, reports, xrv, chexbert):
    if not facts or len(facts) > 384 or len({row["case_id"] for row in facts}) > 2:
        raise ValueError("only the bounded two-case finding review is supported")
    responses = index(qwen["records"], "candidate_id")
    image_labels = index(xrv["records"], "cxr_candidate_id")
    report_labels = index(chexbert["records"], "report_candidate_id")
    if set(responses) != set(cxrs) | set(reports) or set(cxrs) & set(reports):
        raise ValueError("verifier candidate inventory mismatch")
    if set(image_labels) != set(cxrs) or set(report_labels) != set(reports):
        raise ValueError("label candidate inventory mismatch")
    if len({row["evidence_id"] for row in facts}) != len(facts):
        raise ValueError("duplicate finding evidence")
    max_tokens = qwen["producer"]["max_new_tokens"]
    for row in facts:
        image_id, report_id, finding = row["cxr_candidate_id"], row["report_candidate_id"], row["finding"]
        image, report = cxrs[image_id], reports[report_id]
        ihash, rhash = row["artifact_hashes"]["cxr_sha256"], row["artifact_hashes"]["report_sha256"]
        if (image["artifact"]["sha256"] != ihash or report["artifact"]["sha256"] != rhash
                or report["parent_cxr_candidate_id"] != image_id
                or image["case_id"] != row["case_id"] or report["case_id"] != row["case_id"]
                or image["ehr_sha256"] != row["artifact_hashes"]["ehr_sha256"]
                or image["ehr_facts_sha256"] != row["artifact_hashes"]["ehr_facts_sha256"]):
            raise ValueError("finding/artifact lineage mismatch")
        check_response(responses[image_id], expected_hash=ihash, kind="image", max_tokens=max_tokens)
        check_response(responses[report_id], expected_hash=rhash, kind="report", max_tokens=max_tokens)
        ilabel, rlabel = image_labels[image_id], report_labels[report_id]
        score, threshold = ilabel["finding_probabilities"][finding], xrv["thresholds"][finding]
        recomputed = "unknown" if score is None else _state(score, threshold)
        if (ilabel["image_sha256"] != ihash or rlabel["report_sha256"] != rhash
                or recomputed != ilabel["finding_states"][finding]
                or row["states"]["xrv"] != recomputed
                or row["states"]["chexbert"] != rlabel["finding_states"][finding]
                or row["states"]["qwen_image"] != responses[image_id]["states"][finding]
                or row["states"]["qwen_report"] != responses[report_id]["states"][finding]):
            raise ValueError("cached named-state or threshold replay mismatch")
    selected, peers = [], []
    for row in facts:
        if not row["original_selected"] or row["relations"]["qwen_image_report"] != "contradiction":
            continue
        image_id, report_id, finding = row["cxr_candidate_id"], row["report_candidate_id"], row["finding"]
        labels = image_labels[image_id]
        score, threshold = labels["finding_probabilities"][finding], xrv["thresholds"][finding]
        related = [other for other in facts if other["cxr_candidate_id"] == image_id and other["finding"] == finding]
        for other in related:
            peers.append({"selected_evidence_id": row["evidence_id"], "case_id": row["case_id"],
                          "finding": finding, "cxr_candidate_id": image_id,
                          "report_candidate_id": other["report_candidate_id"],
                          "report_model": reports[other["report_candidate_id"]]["model_id"],
                          "original_selected": other["original_selected"],
                          "chexbert_state": other["states"]["chexbert"],
                          "qwen_report_state": other["states"]["qwen_report"],
                          "dependency_group": image_id})
        selected.append({"case_id": row["case_id"], "finding": finding,
                         "evidence_id": row["evidence_id"], "states": row["states"],
                         "relations": row["relations"], "artifact_hashes": row["artifact_hashes"],
                         "cxr_candidate_id": image_id, "report_candidate_id": report_id,
                         "cxr_path": cxrs[image_id]["artifact"]["path"],
                         "report_path": reports[report_id]["artifact"]["path"],
                         "xrv_operating_point_score": score,
                         "xrv_positive_min": threshold["positive_min"],
                         "xrv_negative_max": threshold["negative_max"],
                         "score_minus_positive_min": round(score - threshold["positive_min"], 8),
                         "distance_is_confidence_or_error_probability": False,
                         "image_response": check_response(responses[image_id], expected_hash=row["artifact_hashes"]["cxr_sha256"], kind="image", max_tokens=max_tokens),
                         "report_response": check_response(responses[report_id], expected_hash=row["artifact_hashes"]["report_sha256"], kind="report", max_tokens=max_tokens),
                         "peer_reports": len(related), "clinical_fault_assigned": False,
                         "confirmed_faulty_modality": None, "automatic_repair_eligible": False,
                         "independent_clinical_review_status": "pending"})
    return selected, peers


def load(finding_run):
    root = require_inside(finding_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != REVIEW_SCHEMA or manifest.get("metadata_only") is not True:
        raise ValueError("unexpected finding review scope")
    for name, spec in manifest["artifacts"].items():
        path = require_inside(root / name, root, must_exist=True)
        if sha256_file(path) != spec["sha256"]:
            raise ValueError("finding review artifact hash mismatch")
    paths = {name: require_inside(path, PROTECTED_ROOT, must_exist=True) for name, path in manifest["source_paths"].items()}
    if any(sha256_file(path) != manifest["source_sha256"][name] for name, path in paths.items()):
        raise ValueError("finding review source changed")
    args = SimpleNamespace(review_run=paths["review_scores"].parent,
        analysis_run=paths["candidate_evidence"].parent, selection_run=paths["candidate_table"].parent,
        staging_run=paths["staging_manifest"].parent, crossmodal_details=paths["crossmodal_details"])
    _, table, source, lineages, staged, _ = load_inputs(args)
    reproduced, _ = build_rows(table, source, lineages, staged)
    facts = read_lines(root / "fact_evidence.jsonl")
    if facts != reproduced:
        raise ValueError("finding review is not reproducible")
    qwen = read_json(paths["review_scores"])
    code = Path(__file__).with_name("verify_candidate_findings_qwen.py")
    if (sha256_file(code) != qwen["producer"]["program_sha256"]
            or qwen["producer"]["image_prompt_sha256"] != digest_text(IMAGE_PROMPT)
            or qwen["producer"]["report_prompt_sha256"] != digest_text(REPORT_PROMPT)):
        raise ValueError("recorded Qwen adapter/prompt changed")
    cxr_runs, report_runs = [], []
    for run, fingerprint in qwen["source_manifest_sha256"].items():
        path = require_inside(Path(run) / "manifest.json", PROTECTED_ROOT, must_exist=True)
        if sha256_file(path) != fingerprint:
            raise ValueError("generation manifest changed after verification")
        payload = read_json(path)
        if payload["schema_version"] == "tricompose-cxr-candidate-run-v1.1":
            cxr_runs.append(path.parent)
        elif payload["schema_version"] == "tricompose-report-candidate-run-v1.1":
            report_runs.append(path.parent)
        else:
            raise ValueError("unsupported synthetic generation manifest")
        paths[f"generation_manifest_{path.parent.name}"] = path
    # Loaders read synthetic candidate metadata and hash artifact bytes only.
    # No read_report_text, PIL, torch, model calls, source EHR or real targets.
    cxrs = load_cxr_candidates(cxr_runs)
    reports = load_report_candidates(report_runs, cxr_candidates=cxrs)
    validate_inventory(cxrs, reports, expected_images=12, expected_reports=48)
    details = read_json(paths["crossmodal_details"])
    labels = {}
    for name in ("cxr_labels", "report_labels"):
        descriptor = details["evidence"][name]
        path = require_inside(descriptor["path"], PROTECTED_ROOT, must_exist=True)
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError("extraction cache hash mismatch")
        paths[name] = path
        labels[name] = read_json(path)
    xrv = labels["cxr_labels"]
    threshold_path = require_inside(paths["cxr_labels"].with_name("thresholds.json"), PROTECTED_ROOT, must_exist=True)
    if sha256_file(threshold_path) != xrv["calibration"]["source_sha256"]:
        raise ValueError("threshold bundle hash mismatch")
    threshold_bundle = read_json(threshold_path)
    thresholds = validate_bundle(threshold_bundle, checkpoint_sha256=xrv["producer"]["checkpoint_sha256"],
        expected_provenance={key: xrv["producer"][key] for key in ("score_space", "preprocessing_sha256", "finding_mapping_sha256")})
    if thresholds != xrv["thresholds"]:
        raise ValueError("cached operating points differ from frozen bundle")
    paths["threshold_bundle"] = threshold_path
    paths["finding_manifest"] = root / "manifest.json"
    selected, peers = audit_records(facts, qwen, cxrs, reports, xrv, labels["report_labels"])
    return selected, peers, paths


def markdown(selected):
    lines = ["# Selected disagreements: engineering audit / 原择优分歧核查", "",
        "Cached metadata and artifact byte hashes only; no model inference, report-text or image-pixel interpretation.",
        "All original scores, artifacts, thresholds and winner flags remain unchanged.", "",
        "| Case | Finding | EHR | XRV | CheXbert | Qwen image | Qwen report | XRV score | Frozen positive cutoff |",
        "|---|---|---|---|---|---|---|---:|---:|"]
    for row in selected:
        states = row["states"]
        lines.append(f"| {row['case_id']} | {row['finding']} | {states['ehr']} | {states['xrv']} | "
            f"{states['chexbert']} | {states['qwen_image']} | {states['qwen_report']} | "
            f"{row['xrv_operating_point_score']:.6f} | {row['xrv_positive_min']:.6f} |")
    lines += ["", "## What passed / 已排除的工程问题", "",
        "Source manifests, synthetic artifact byte hashes, candidate/parent IDs and fixed-EHR hashes match.",
        "Named finding states reproduce the previous join, and frozen XRV threshold decisions replay exactly.",
        "Selected Qwen responses are complete and not token-limited; adapter and prompt fingerprints match.",
        "These checks rule out the checked cache/lineage/join failures, NOT evaluator semantic errors.", "",
        "## What remains unresolved / 尚不能裁决", "",
        "Qwen raw response text was not retained: this audit cannot independently reparse it or audit its reasoning.",
        "EHR unknown is not a negative. CheXbert unknown is not a report denial.",
        "XRV normalized scores and distance from a cutoff are not confidence/probability calibration.",
        "Four reports share an image, and both Qwen passes share a model: no majority-vote adjudication.",
        "Peer report states are supplied for review, not to choose a new winner or assign image/report fault.",
        "No clinical label, localization accuracy, new ranking, or targeted regeneration is authorized.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        selected, peers, sources = load(args.finding_run)
        summary = {"schema_version": SCHEMA, "selected_opposition_rows": len(selected),
            "peer_report_rows": len(peers), "input_contract_audit": "passed",
            "gpu_inference_used": False, "artifact_bytes_hashed": True,
            "report_text_or_image_pixels_interpreted": False, "selection_changed": False,
            "primary_metric_eligible": False, "targeted_repair_approved": False,
            "clinical_fault_assigned": False, "clinical_adjudication_pending": True}
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        stream = io.StringIO()
        if peers:
            writer = csv.DictWriter(stream, fieldnames=list(peers[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(peers)
        files = [write_private_json(temporary / "selected_review.json", {"schema_version": SCHEMA, "records": selected}),
                 write_private_text(temporary / "peer_report_states.csv", stream.getvalue()),
                 write_private_json(temporary / "summary.json", summary),
                 write_private_text(temporary / "summary.md", markdown(selected))]
        write_private_json(temporary / "manifest.json", {**summary, "run_id": args.run_id,
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "source_paths": {name: str(path) for name, path in sources.items()},
            "program_sha256": sha256_file(__file__),
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_selected_disagreement_engineering_audit",
        "selected_opposition_rows": len(selected), "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

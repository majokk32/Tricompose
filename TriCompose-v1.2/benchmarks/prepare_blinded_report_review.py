#!/usr/bin/env python3
"""Freeze a metadata-only, report-only human-review packet for the 48-report pilot.

Never opens report text, images, EHR records or real inputs. Does not run a
parser/model, prefill gold, unblind model identities, change scores or repair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_review import VERSION, make_queue, annotation_template, object_hash, reader_agreement
from tricompose_v12.report_assertions import FINDINGS
from contracts import (PROTECTED_ROOT, REPORT_RUN_SCHEMA, REPORT_CANDIDATE_SCHEMA,
    _load_manifest_rows, require_inside, read_json, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, private_directory, write_private_json,
    write_private_text)

SCHEMA = "tricompose-blinded-report-review-packet-v1"
PINS = {
    "assertion_interface": "00459f4c8c9faf009d60b56529991a690a53ac2b782e9ea1c21604c2789779cd",
    "literal_guard": "7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18",
    "scope_gate": "475596019a21ff386529d358e502a716d8c57d48516391317812aeef6ad060db",
    "literal_readout": "4f0b151543b94c22b6b6b45ae98f8de00d157e92fe56d3ccbed8b485ba084425",
}


def checked_artifact(root, name):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    path = require_inside(root/name, root, must_exist=True)
    manifest = read_json(root/"manifest.json")
    if sha256_file(path) != manifest["artifacts"][name]["sha256"]:
        raise ValueError("review source artifact hash differs")
    return path, manifest


def load_metadata(args):
    table, selection = checked_artifact(args.selection_run, "candidate_score_table.jsonl")
    scores, _ = checked_artifact(args.review_run, "scores.json")
    cross = require_inside(args.crossmodal_details, PROTECTED_ROOT, must_exist=True)
    if sha256_file(cross) != args.crossmodal_sha256:
        raise ValueError("frozen crossmodal hash differs")
    if selection["schema_version"] != "tricompose-edge-specific-selection-v1.1":
        raise ValueError("unsupported source selection")
    rows = [json.loads(line) for line in table.read_text().splitlines() if line]
    details, qwen = read_json(cross), read_json(scores)
    if (details["evaluation_scope"] != {"cohort": "fully_synthetic",
            "raw_source_target_supplied": False, "real_reference_report_supplied": False,
            "unknown_is_negative": False} or len(rows) != 48 or
            len({row["case_id"] for row in rows}) != 2):
        raise ValueError("bounded fully synthetic two-case pilot required")
    if (qwen.get("schema_version") != "tricompose-separated-qwen-finding-review-v1" or
            qwen.get("producer", {}).get("frozen") is not True or
            qwen.get("selection_changed") is not False):
        raise ValueError("frozen existing report review required")
    sources = {"selection_table": table, "selection_manifest": table.parent/"manifest.json",
        "crossmodal_details": cross, "qwen_scores": scores, "qwen_manifest": scores.parent/"manifest.json"}
    report_runs = []
    for raw, h in qwen["source_manifest_sha256"].items():
        path = require_inside(Path(raw)/"manifest.json", PROTECTED_ROOT, must_exist=True)
        if sha256_file(path) != h:
            raise ValueError("generation source manifest differs")
        if read_json(path)["schema_version"] == REPORT_RUN_SCHEMA:
            report_runs.append(path.parent)
            sources[f"generation_manifest_{len(report_runs)}"] = path
    reports = {item["candidate_id"]: item for _, _, item in
        _load_manifest_rows(report_runs, run_schema=REPORT_RUN_SCHEMA, row_key="candidates")}
    reference = {row["report_candidate_id"]: row for row in details["records"]}
    qwen_reports = {row["candidate_id"]: row for row in qwen["records"] if row["input_kind"] == "report"}
    ids = {row["triple_candidate_id"] for row in rows}
    if len(ids) != 48 or set(reports) != ids or set(reference) != ids or set(qwen_reports) != ids:
        raise ValueError("all reports must remain in the blind denominator")
    candidates, predictions = [], []
    fixed_ehrs, parents = {}, {}
    for row in rows:
        candidate_id, lineage = row["triple_candidate_id"], row["lineage"]
        report, ref, secondary = reports[candidate_id], reference[candidate_id], qwen_reports[candidate_id]
        if (report["schema_version"] != REPORT_CANDIDATE_SCHEMA or report["frozen_model"] is not True
                or report["modality"] != "report" or report["source_report_or_real_target_supplied"] is not False
                or report["case_id"] != row["case_id"] or ref["case_id"] != row["case_id"]
                or lineage["report_candidate_id"] != candidate_id
                or lineage["report_sha256"] != report["artifact"]["sha256"]
                or ref["report_sha256"] != lineage["report_sha256"]
                or secondary["artifact_sha256"] != lineage["report_sha256"]
                or ref["image_sha256"] != lineage["cxr_sha256"]
                or ref["report_model_id"] != lineage["report_model_id"]
                or report["parent_cxr_candidate_id"] != lineage["cxr_candidate_id"]
                or ref["parent_cxr_candidate_id"] != lineage["cxr_candidate_id"]
                or report["ehr_sha256_retained_for_lineage"] != lineage["ehr_sha256"]
                or report["ehr_facts_sha256_retained_for_lineage"] != lineage["ehr_facts_sha256"]
                or report["input_cxr"]["sha256"] != lineage["cxr_sha256"]
                or report["model_id"] != lineage["report_model_id"]):
            raise ValueError("synthetic report/case/source lineage differs")
        fixed = (lineage["ehr_sha256"], lineage["ehr_facts_sha256"])
        if row["case_id"] in fixed_ehrs and fixed_ehrs[row["case_id"]] != fixed:
            raise ValueError("fixed EHR changed across candidates")
        fixed_ehrs[row["case_id"]] = fixed
        parent = lineage["cxr_candidate_id"]
        slot = parents.setdefault(parent, {"case": row["case_id"], "hash": lineage["cxr_sha256"], "models": set()})
        if (slot["case"] != row["case_id"] or slot["hash"] != lineage["cxr_sha256"] or
                lineage["report_model_id"] in slot["models"]):
            raise ValueError("shared image lineage or report expert grid differs")
        slot["models"].add(lineage["report_model_id"])
        source_path = require_inside(report["artifact"]["path"], PROTECTED_ROOT, must_exist=True)
        if not source_path.is_file() or source_path.stat().st_size > 32768:
            raise ValueError("bounded source synthetic report required")
        candidates.append({"candidate_id": candidate_id, "case_id": row["case_id"],
            "report_sha256": lineage["report_sha256"], "report_path": str(source_path),
            "ehr_sha256": lineage["ehr_sha256"], "cxr_sha256": lineage["cxr_sha256"],
            "model_id": lineage["report_model_id"]})
        predictions.append({"candidate_id": candidate_id, "report_sha256": lineage["report_sha256"],
            "chexbert": {name: ref["report_finding_states"][name] for name in FINDINGS},
            "qwen": {name: secondary["states"][name] for name in FINDINGS},
            "qwen_contract_status": secondary["contract_status"]})
    if (len(parents) != 12 or len(set(fixed_ehrs.values())) != 2 or
            any(row["models"] != {"maira2", "cxrmate_single", "llavarad", "chexagent2"} for row in parents.values())):
        raise ValueError("fixed two-case twelve-image four-expert grid required")
    return candidates, predictions, sources


def freeze_method():
    root = Path(__file__).resolve().parents[1]
    paths = {"assertion_interface": root/"src/tricompose_v12/report_assertions.py",
        "literal_guard": root/"benchmarks/repair_cached_report_evidence.py",
        "scope_gate": root/"benchmarks/gate_report_assertion_predictions.py",
        "literal_readout": root/"benchmarks/compare_report_assertion_sources.py"}
    if {name: sha256_file(path) for name, path in paths.items()} != PINS:
        raise ValueError("reviewed method freeze differs")
    return {"protocol_version": VERSION, "method_sha256": PINS,
        "finding_order": list(FINDINGS), "context_promoted_into_selection": False,
        "development_design_after_authored_results_seen": True,
        "new_rules_or_thresholds": False, "report_content_truth_from_parsers": False,
        "regeneration_authorized": False}, paths


def instructions():
    return """# Blinded report-only annotation / 报告文字盲审

This packet currently contains metadata and blank templates only. Report
copies have NOT been materialized: do not follow investigator source paths.
Copying synthetic text requires a separately authorized protected step.

Two distinct human readers independently label every finding in every report.
Use opaque reviewer aliases and declare whether model labels/winners were seen.
Do not use an LLM, parser prediction, image, EHR or original reference report
to fill this report-text task. Access to sibling investigator files is possible
for authorized project members; blinding is procedural, not an ACL guarantee.
Intrinsic writing style may also reveal the generator despite hidden IDs.

Labels: positive / negative / uncertain / unknown. Missing is unknown, never
negative. Pending is null, NOT reviewed unknown. Preserve negation, uncertainty,
qualified absence, historical/hypothetical statements and conflicting sections.
A qualified absence does not assert global absence. Keep the original quote.
This four-state scope policy is project-specific, not universal CheXbert gold.

For each reviewed assertion add exact contiguous quote(s), zero-based Unicode
char_start/char_end, quote_sha256 and offset_unit=unicode_codepoint. Unknown
with reason=not_mentioned has no invented evidence. Unreadable/unassessable
reports stay unassessable with state=null; never silently omit their rows.

Copy templates into a NEW protected annotation-return directory before editing.
Do not modify this immutable packet. Adjudicate reader conflicts separately;
no extractor or score may break a tie. Reader agreement alone is not gold.

This is a previously inspected TWO-EHR development pilot, not untouched final
validation. It cannot establish CXR factuality or clinical fault localization.
"""


def prepare(args):
    candidates, predictions, sources = load_metadata(args)
    frozen, methods = freeze_method()
    items, resolver = make_queue(candidates)
    by_hash = {row["report_sha256"]: row["item_id"] for row in items}
    for row in predictions:
        row["item_id"] = by_hash[row["report_sha256"]]
    template = annotation_template(items)
    progress = reader_agreement(items, template, template)
    summary = {"schema_version": SCHEMA, "status": "awaiting_report_copy_and_independent_human_annotations",
        "candidate_reports": len(candidates), "unique_report_text_hashes": len(items),
        "independent_ehr_cases": len({row["case_id"] for row in candidates}),
        "finding_count": len(FINDINGS), "annotation_rows_per_reader": len(template["records"]),
        "annotation_slots_two_readers": 2*len(template["records"]),
        "report_text_opened": False, "image_pixels_opened": False, "raw_ehr_opened": False,
        "report_copies_materialized": False, "model_calls": 0,
        "blindness": "procedural_model_ids_scores_and_winners_omitted_style_may_reveal_model",
        "previously_used_development_cohort": True, "untouched_final_test": False,
        "reader_progress": progress, "primary_metric_eligible": False,
        "selection_changed": False, "regeneration_authorized": False}
    return items, resolver, predictions, template, frozen, summary, {**sources,**methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection-run", "review-run", "crossmodal-details", "crossmodal-sha256", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    temporary = None
    try:
        items, resolver, predictions, template, frozen, summary, sources = prepare(args)
        initial = {name: sha256_file(path) for name,path in sources.items()}
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        private_directory(temporary/"reviewer")
        private_directory(temporary/"investigator")
        files = [write_private_json(temporary/"reviewer/items.json", {"items": items}),
            write_private_json(temporary/"reviewer/reviewer_a_template.json", template),
            write_private_json(temporary/"reviewer/reviewer_b_template.json", template),
            write_private_text(temporary/"reviewer/INSTRUCTIONS.md", instructions()),
            write_private_json(temporary/"investigator/resolver.json", {"records": resolver}),
            write_private_json(temporary/"investigator/frozen_predictions.json", {"records": predictions}),
            write_private_json(temporary/"investigator/method_freeze.json", frozen),
            write_private_json(temporary/"summary.json", summary)]
        if initial != {name: sha256_file(path) for name,path in sources.items()}:
            raise ValueError("review source changed during metadata preparation")
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA,
            "program_sha256": sha256_file(__file__), "run_id": args.run_id,
            "review_library_sha256": sha256_file(Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_review.py"),
            "inventory_sha256": object_hash(items), "source_sha256": initial,
            "artifacts": {str(path.relative_to(temporary)): {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_metadata_only_review_packet", "model_calls": 0,
        "reports": summary["candidate_reports"], "rows_per_reader": summary["annotation_rows_per_reader"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export protected human CSV forms/book or import a human-filled CSV.

No model/API, clinical labels, automatic adjudication or primary-score update.
The read-only source is the approved, copied synthetic blind-report bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_review_sheets import identity_template, sheet_template, import_sheet, report_book
from tricompose_v12.report_review import annotation_template, object_hash
from tricompose_v12.report_assertions import digest
from contracts import (PROTECTED_ROOT,require_inside,read_json,sha256_file,new_atomic_run,
    commit_atomic_run,discard_atomic_run,write_private_json,write_private_text)

SCHEMA = "tricompose-human-review-sheets-v1"
IMPORT_SCHEMA = "tricompose-human-review-sheet-import-v1"


def load_bundle(path):
    root = require_inside(path, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    if manifest.get("schema_version") != "tricompose-blinded-synthetic-report-copies-v1":
        raise ValueError("approved synthetic blind bundle required")
    sources = {"blind_manifest":root/"manifest.json"}
    for name in ("items.json","summary.json"):
        source = require_inside(root/name,root,must_exist=True)
        if sha256_file(source) != manifest["artifacts"][name]["sha256"]:
            raise ValueError("blind bundle metadata hash differs")
        sources[name] = source
    summary,items = read_json(root/"summary.json"),read_json(root/"items.json")["items"]
    if (summary.get("reports") != 48 or len(items) != 48 or
            summary.get("synthetic_only") is not True or summary.get("real_inputs_read") is not False or
            summary.get("copied_reports_byte_identical") is not True or
            summary.get("independent_ehr_cases") != 2 or summary.get("regeneration_authorized") is not False):
        raise ValueError("bounded synthetic development bundle required")
    paths = {}
    for item in items:
        name = item["report_relative_path"]
        if (not re.fullmatch(r"report_[0-9]{4}",item["item_id"]) or
                name != f"reports/{item['item_id']}.txt" or item["report_sha256"] in paths):
            raise ValueError("opaque distinct-text review inventory required")
        path = require_inside(root/name,root,must_exist=True)
        if manifest["artifacts"][name]["sha256"] != item["report_sha256"]:
            raise ValueError("review report manifest binding differs")
        paths[item["report_sha256"]] = path
    def source_reader(h):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("synthetic text source access requires approved Slurm")
        source = paths[h]
        if source.stat().st_size > 32768:
            raise ValueError("review report exceeds source bound")
        value = source.read_bytes().decode("utf-8")
        if not value.strip() or len(value)>8192 or digest(value)!=h:
            raise ValueError("synthetic blind report text/hash differs")
        return value
    return items,source_reader,sources


def instructions(bundle):
    return f"""# CSV human annotation / 人工 CSV 标注

Read only reports_for_review.md or the opaque report files from:
{bundle}

Both humans use the SAME blank form independently; no labels are prefilled.
Create a NEW protected working directory and copy your CSV and identity JSON
there before editing. Do not edit this immutable forms run or source reports.
Use opaque aliases such as reader_a / reader_b, not a patient's or person's name.
Choose your actual role; non_expert is permitted but is NOT clinical gold.
Allowed roles: human_domain_annotator / clinician / radiologist / non_expert.
Fill the independence attestation truthfully. The importer rejects annotations
declared to use model predictions, winner flags or model-generated labels.
Identity JSON needs reviewer_alias, reviewer_role, and this attestation object:
{{"model_predictions_seen": false, "winner_flags_seen": false,
 "labels_generated_by_model": false}}
Do not fill false if it is not true of your review process.

For each of 192 rows:
- Leave pending + blank state/reason/quotes until you have reviewed it.
- Set status=reviewed and choose positive / negative / uncertain / unknown.
- Annotate what the report asserts NOW, not your belief about its image/EHR.
  Positive means asserted presence, negative explicit absence, uncertain a
  qualified/conflicting assertion, unknown no comparable current assertion.
- Add reason: explicit_assertion / not_mentioned /
  historical_or_hypothetical_only / qualified_or_conflicting.
- Copy quote_1 (and optionally quote_2) VERBATIM. Do not paraphrase or trim it.
- If a quote appears more than once, set quote_N_occurrence to its zero-based
  occurrence number. Otherwise leave that field empty.
- Unknown with reason=not_mentioned needs no fabricated quote. Missing is NOT
  negative. Qualified absence is not global absence; retain uncertainty.
- Unreadable: status=unassessable, blank state, reason=unassessable.
- Keep every row and the item_id/report_sha256/finding cells unchanged.

The import command computes exact Unicode offsets and quote hashes only.
It never infers a diagnosis, completes a missing label, or changes your state.
Two quote slots are a convenience; for more evidence use the original JSON
annotation contract. No parser is allowed to resolve disagreement between people.

Import each return into a NEW run with report_review_sheets.py --mode import,
then use audit_blinded_report_review.py on the two reader_annotations.json files.
Exact spans establish traceability, not correct human interpretation. Roles and
independence attestations are not externally verified credentials or clinical gold.
No model/score/winner/EHR/image data or real reference report belongs in this task.
"""


def run(args,temporary):
    if args.mode == "export" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before synthetic book rendering")
    items,source_reader,sources = load_bundle(args.bundle_run)
    if args.mode == "export":
        texts = {row["report_sha256"]:source_reader(row["report_sha256"]) for row in items}
        files = [write_private_text(temporary/"reports_for_review.md",report_book(items,texts)),
            write_private_text(temporary/"reader_a.csv",sheet_template(items)),
            write_private_text(temporary/"reader_b.csv",sheet_template(items)),
            write_private_json(temporary/"reader_a_identity.json",identity_template(items)),
            write_private_json(temporary/"reader_b_identity.json",identity_template(items)),
            write_private_text(temporary/"HOW_TO_REVIEW.md",instructions(str(Path(args.bundle_run).resolve())))]
        summary = {"schema_version":SCHEMA,"reports":len(items),
            "rows_per_reader":len(annotation_template(items)["records"]),"human_annotations_completed":0,
            "anonymous_report_book_rendered":True,"labels_prefilled":False,"model_calls":0,
            "selection_changed":False,"regeneration_authorized":False,"independent_ehr_cases":2,
            "clinical_gold_created":False,"untouched_final_test":False}
    else:
        sheet = require_inside(args.sheet_file,PROTECTED_ROOT,must_exist=True)
        identity = require_inside(args.identity_file,PROTECTED_ROOT,must_exist=True)
        if sheet.stat().st_size > 8*1024*1024:
            raise ValueError("bounded human sheet required")
        sources.update(human_sheet=sheet,human_identity=identity)
        # Preserve CRLF inside quoted evidence cells; universal newline
        # translation would silently change a human's verbatim source quote.
        annotation = import_sheet(items,sheet.read_bytes().decode("utf-8-sig"),read_json(identity),read_source=source_reader)
        files = [write_private_json(temporary/"reader_annotations.json",annotation)]
        counts = dict(Counter(row["status"] for row in annotation["records"]))
        summary = {"schema_version":IMPORT_SCHEMA,"annotation_rows":len(annotation["records"]),
            "status_counts":counts,"human_reviewed_rows":counts.get("reviewed",0),
            "human_states_changed_by_importer":False,"evidence_offsets_hashes_computed":True,
            "model_calls":0,"selection_changed":False,"regeneration_authorized":False,
            "clinical_gold_created":False,"independence_attestation_externally_verified":False}
    summary["inventory_sha256"] = object_hash(items)
    return summary,files,sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("export","import"),required=True)
    for name in ("bundle-run","output-root","run-id"):
        parser.add_argument(f"--{name}",required=True)
    for name in ("sheet-file","identity-file"):
        parser.add_argument(f"--{name}")
    args=parser.parse_args()
    if args.mode == "import" and (args.sheet_file is None or args.identity_file is None):
        parser.error("--mode import requires --sheet-file and --identity-file")
    if args.mode == "export" and (args.sheet_file is not None or args.identity_file is not None):
        parser.error("human return paths apply only to --mode import")
    os.umask(0o007)
    temporary=None
    try:
        temporary,target=new_atomic_run(args.output_root,args.run_id)
        summary,files,sources=run(args,temporary)
        files.append(write_private_json(temporary/"summary.json",summary))
        write_private_json(temporary/"manifest.json",{"schema_version":summary["schema_version"],
            "program_sha256":sha256_file(__file__),"run_id":args.run_id,
            "sheet_library_sha256":sha256_file(Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_review_sheets.py"),
            "review_library_sha256":sha256_file(Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_review.py"),
            "span_library_sha256":sha256_file(Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_assertions.py"),
            "source_sha256":{name:sha256_file(path) for name,path in sources.items()},
            "artifacts":{str(path.relative_to(temporary)):{"sha256":sha256_file(path)} for path in files}})
        commit_atomic_run(temporary,target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}))
        return 1
    print(json.dumps({"status":"completed_human_review_sheet_"+args.mode,"model_calls":0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

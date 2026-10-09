#!/usr/bin/env python3
"""CPU Slurm: recompute native n-best receipts, decisions and measurement.

Only synthetic metadata, cached labels, schemas and file hashes are consumed.
No report bodies, image pixels, model factories, real inputs or targets.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, load_cxr_candidates,
    load_report_candidates, new_atomic_run, write_private_json, write_private_text, commit_atomic_run, discard_atomic_run)
from run_report_nbest import load, single_view
from tricompose_v12.report_nbest import SCHEMA, freeze, measure
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v12.invariant_verification import _digest


def audit(args):
    require_slurm(); began = time.monotonic()
    root = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    mp = root/"manifest.json"; manifest_sha = sha256_file(mp); m = read_json(mp)
    if (m.get("schema_version") != SCHEMA or m.get("status") != "completed_engineering_nbest_control_unvalidated"
            or m.get("clinical_acceptance") is not False or m.get("adaptive_repair_executed") is not False
            or m.get("biovil_used_for_selection") is not False or m.get("original_ehr_cxr_or_winners_changed") is not False):
        raise ValueError("completed unvalidated frozen n-best control required")
    args.plan_manifest_sha256 = m["plan_manifest_sha256"]; plan = load(args)
    for name, entry in m["artifacts"].items():
        p = require_inside(root/name, root, must_exist=True)
        if sha256_file(p) != entry["sha256"]: raise ValueError("published artifact changed")
    for p in (root, *root.rglob("*")):
        info = p.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_gid not in (96293, 65534)
                or stat.S_IMODE(info.st_mode) != (0o2770 if p.is_dir() else 0o660)):
            raise ValueError("new-run project-private modes/group differ")
    cxrs = load_cxr_candidates([plan["cxr_run"]])
    reports = load_report_candidates([root/"reports"], cxr_candidates=cxrs)
    gm = read_json(root/"reports/manifest.json")
    if sha256_file(root/"reports/manifest.json") != m["generator_manifest_sha256"]:
        raise ValueError("generator manifest changed")
    ip = root/"xrv/scored/cxr_finding_labels.json"; tp = root/"chexbert/scored/report_finding_labels.json"
    if sha256_file(ip) != m["xrv_labels_sha256"] or sha256_file(tp) != m["chexbert_labels_sha256"]:
        raise ValueError("scorer bundle changed")
    il, tl = read_json(ip), read_json(tp)
    rows = read_json(root/"score_rows.json")["records"]
    if len(cxrs) != 2 or len(reports) != 6 or len(rows) != 6:
        raise ValueError("two-image six-sequence inventory differs")
    contexts = {c["image"]["candidate_id"]: c for c in plan["fixed_cases"]}
    for row in rows:
        image = cxrs[row["cxr_candidate_id"]]; report = reports[row["report_candidate_id"]]
        context = contexts[image["candidate_id"]]; anchor = anchor_from_record(context["anchor"])
        if image != context["image"]: raise ValueError("original fixed image metadata changed")
        image_labels = single_view(il, "cxr_candidates", "cxr_candidate_id", image["candidate_id"])
        partial = image_receipt(anchor, image, image_labels, label_sha256=sha256_file(ip),
            thresholds_sha256=plan["xrv"]["thresholds_sha256"], checkpoint_sha256=plan["xrv"]["checkpoint_sha256"])
        rebuilt = completed_receipt(anchor, partial, image, image_labels, report,
            single_view(tl, "reports", "report_candidate_id", report["candidate_id"]),
            image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
            thresholds_sha256=plan["xrv"]["thresholds_sha256"], xrv_checkpoint_sha256=plan["xrv"]["checkpoint_sha256"],
            chexbert_checkpoint_sha256=plan["chexbert"]["checkpoint_sha256"])
        if (row["receipt"] != rebuilt or row["raw_edge_readouts"] != rebuilt["raw_edge_readouts"]
                or row["beam_rank"] != report["beam_rank"]
                or row["structure"] != read_json(Path(report["artifact"]["path"]).parent/"structure.json")
                or row["ehr_sha256"] != anchor.ehr_sha256 or row["ehr_facts_sha256"] != anchor.ehr_facts_sha256):
            raise ValueError("fresh source-bound receipt/structure metadata differs")
    selection = read_json(root/"selection.json"); endpoint = read_json(root/"secondary.json")
    if (freeze(rows) != selection or measure(selection, rows, endpoint) != read_json(root/"comparison.json")
            or sha256_file(root/"selection.json") != m["selection_sha256_before_endpoint"]
            or sha256_file(root/"secondary_request/request.json") != endpoint["request_sha256"]):
        raise ValueError("sealed choice/independent measurement differs")
    table = list(csv.DictReader(io.StringIO((root/"score_table.csv").read_text())))
    table_index = {r["triple_candidate_id"]: r for r in table}
    if len(table_index) != 6 or len(table) != 6: raise ValueError("complete score table required")
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    for row in rows:
        out = table_index[row["triple_candidate_id"]]
        for edge, values in row["raw_edge_readouts"].items():
            for name, value in values.items():
                expected = "NA" if value is None else str(value)
                if out[edge+"_"+name] != expected: raise ValueError("CSV raw count/rate/NA changed")
        if out["biovil_status"] != scores[row["triple_candidate_id"]]["status"]:
            raise ValueError("CSV endpoint status changed")
    calls = gm["native_calls"]
    if (len(calls) != 2 or gm["native_generate_invocations"] != 2 or gm["returned_sequences"] != 6
            or sum(r["cost"]["model_calls"] for r in reports.values()) != 2):
        raise ValueError("native shared invocation accounting differs")
    for call in calls:
        group = [r for r in reports.values() if r["native_invocation_id"] == call["native_invocation_id"]]
        sr = [r for r in rows if r["report_candidate_id"] in {x["candidate_id"] for x in group}]
        if (len(group) != 3 or {r["beam_rank"] for r in group} != {0,1,2}
                or call["unique_exact_reports"] != len({r["artifact"]["sha256"] for r in group})
                or call["unique_normalized_reports"] != len({r["structure"]["normalized_report_sha256"] for r in sr})):
            raise ValueError("native beam/duplicate counts differ")
    native_journal = root/"reports/native_call_journal.jsonl"
    if sha256_file(native_journal) != gm["native_call_journal_sha256"]: raise ValueError("native journal changed")
    events = [json.loads(line) for line in native_journal.read_text().splitlines()]
    if len(events) != 4 or [e["status"] for e in events] != ["reserved_before_generate", "validated"]*2:
        raise ValueError("native reservations/completions differ")
    costs = [json.loads(line) for line in (root/"cost_journal.jsonl").read_text().splitlines()]
    if ([e["stage"] for e in costs if e["status"] == "reserved_before_spawn"] != ["generate","xrv","chexbert","biovil"]
            or any(e["status"] == "failed_or_interrupted_retained" for e in costs)):
        raise ValueError("bounded stage journal differs")
    if sha256_file(mp) != manifest_sha: raise ValueError("source manifest changed during audit")
    result = {"schema_version": "tricompose-report-nbest-audit-v1", "status": "metadata_hash_receipt_audit_passed",
        "source_manifest_sha256": manifest_sha, "source_plan_manifest_sha256": m["plan_manifest_sha256"],
        "fixed_ehr_cases": 2, "fixed_images": 2, "native_generate_invocations": 2, "returned_sequences": 6,
        "gate_passing_alternatives": sum(p["exploratory_gate_pass"] for p in selection["comparisons"]),
        "changed_selected_images": sum(c["baseline_triple_id"] != c["selected_triple_id"] for c in selection["choices"]),
        "direct_ehr_known_facts_per_baseline": [r["receipt"]["known_ehr_facts"] for r in rows if r["beam_rank"] == 0],
        "raw_edge_csv_verified": True, "decision_and_receipts_recomputed": True, "duplicate_counts_recomputed": True,
        "structure_scope": "authenticated_cached_structure_not_body_reexecution",
        "source_bodies_or_image_pixels_opened": False, "new_model_calls": 0,
        "clinical_acceptance": False, "clinical_repair_success": False,
        "runtime_seconds": round(time.monotonic()-began, 3)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(temporary/"audit.json", result)
        write_private_json(temporary/"manifest.json", {"schema_version": result["schema_version"],
            "audit_sha256": sha256_file(path), "source_manifest_sha256": manifest_sha})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root", "run-id"): p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try: root, result = audit(args)
    except Exception as exc:
        print(json.dumps({"status":"audit_failed","error_type":type(exc).__name__})); return 1
    print(json.dumps({"status":result["status"],"runtime_seconds":result["runtime_seconds"],
        "manifest_sha256":sha256_file(root/"manifest.json")})); return 0


if __name__ == "__main__": raise SystemExit(main())

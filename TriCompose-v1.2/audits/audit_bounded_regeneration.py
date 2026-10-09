#!/usr/bin/env python3
"""CPU Slurm metadata audit; never read report text or image pixels."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, load_cxr_candidates,
    load_report_candidates, new_atomic_run, write_private_json, commit_atomic_run, discard_atomic_run)
from tricompose_v12.bounded_regeneration import SCHEMA, POLICY, decision, validate_row
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v12.execution_ledger import restore_ledger
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.invariant_verification import _digest
from run_bounded_regeneration import load, tables
from audit_full_pool_report_control import verify_csv


def audit(args):
    require_slurm()
    plan = load(args)
    root = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root / "manifest.json")
    if (manifest["schema_version"] != SCHEMA or manifest["status"] != "completed_bounded_retry_engineering_unvalidated"
            or manifest["plan_manifest_sha256"] != args.plan_manifest_sha256
            or any(manifest[k] is not False for k in ("clinical_acceptance", "clinical_repair_success", "original_winners_changed",
                "historical_pool_is_untouched_test", "automatic_resume", "same_budget_static_reranking_control_available"))):
        raise ValueError("completed nonclinical immutable bounded retry required")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(require_inside(root / name, root, must_exist=True)) != expected["sha256"]:
            raise ValueError("published retry artifact changed")
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_gid not in (96293, 65534)
                or stat.S_IMODE(info.st_mode) != (0o2770 if path.is_dir() else 0o660)):
            raise ValueError("project-private mode/group boundary differs")
    rows = read_json(root / "score_rows.json")["records"]
    indexed = {r["triple_candidate_id"]: r for r in rows}
    if len(indexed) != len(rows) or not 2 <= len(rows) <= 6:
        raise ValueError("unique bounded original/static/new row inventory required")
    selection = read_json(root / "selection.json")
    if (selection["schema_version"] != SCHEMA or selection["policy"] != POLICY or selection["cohort"] != plan["cohort"]
            or selection["used_biovil"] is not False or selection["clinical_acceptance"] is not False
            or selection["original_winners_changed"] is not False
            or sha256_file(root / "selection.json") != manifest["selection_sha256_before_endpoint"]):
        raise ValueError("sealed choice differs")
    triples = read_json(root / "completed_chains.json")["records"]
    books = read_json(root / "execution_summary.json")["case_ledgers"]
    bycase = {t["case_id"]: t for t in triples}
    if len(bycase) != len(triples) or len(triples) > 2:
        raise ValueError("one completed retry chain per fixed case required")
    expected_choices, expected_row_ids = [], set()
    for case in plan["cases"]:
        anchor = anchor_from_record(case["anchor"])
        base, static = case["baseline"], case["static"]
        for row in (base, static):
            if indexed.get(row["triple_candidate_id"]) != row:
                raise ValueError("original fixed/static reference changed")
            expected_row_ids.add(row["triple_candidate_id"])
        triple = bycase.get(case["case_id"])
        alt = None
        if triple:
            cr = root / "cases" / case["case_id"]
            images = load_cxr_candidates([triple["cxr_run"]])
            reports = load_report_candidates([triple["report_run"]], cxr_candidates=images)
            if len(images) != 1 or len(reports) != 1:
                raise ValueError("one authenticated image/report per retry required")
            image, report = next(iter(images.values())), next(iter(reports.values()))
            request = case["requests"][0]
            if (image["model_id"] != "roentgen_v2" or image["seed"] != 1
                    or image["ehr_sha256"] != anchor.ehr_sha256 or image["ehr_facts_sha256"] != anchor.ehr_facts_sha256
                    or image["prompt_sha256"] != request["request"]["inputs"]["final_prompt"]["sha256"]
                    or image["input_request_sha256"] != request["canonical_request_sha256"]
                    or report["model_id"] != "cxrmate_single"):
                raise ValueError("new image/report lost original EHR/text contract")
            ip = cr / "operations/xrv_0_a1/scored/cxr_finding_labels.json"
            tp = cr / "operations/chexbert_0_0_a1/scored/report_finding_labels.json"
            il, tl = read_json(ip), read_json(tp)
            xrv, cb = plan["workers"]["xrv"], plan["workers"]["chexbert"]
            partial = image_receipt(anchor, image, il, label_sha256=sha256_file(ip),
                thresholds_sha256=xrv["thresholds_sha256"], checkpoint_sha256=xrv["checkpoint_sha256"])
            receipt = completed_receipt(anchor, partial, image, il, report, tl,
                image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
                thresholds_sha256=xrv["thresholds_sha256"], xrv_checkpoint_sha256=xrv["checkpoint_sha256"],
                chexbert_checkpoint_sha256=cb["checkpoint_sha256"])
            pair_id = "retrypair_" + _digest([image["candidate_id"], report["candidate_id"]])[:32]
            alt = indexed[pair_id]
            validate_row(alt)
            if (alt["receipt"] != receipt or alt["seed"] != 1 or alt["cxr_model_id"] != "roentgen_v2"
                    or alt["report_model_id"] != "cxrmate_single"
                    or sha256_file(triple["receipt_path"]) != triple["receipt_sha256"]
                    or read_json(triple["receipt_path"]) != receipt):
                raise ValueError("fresh source-bound receipt/raw row differs")
            expected_row_ids.add(pair_id)
        failure = "bounded_generation_or_verification_incomplete" if case["frozen_action"]["action"] == "regenerate_cxr" and alt is None else None
        expected_choices.append(decision(base, static, alt, failure))
    if expected_choices != selection["choices"] or expected_row_ids != set(indexed):
        raise ValueError("unrequested candidate or changed route/decision detected")
    routed = [c for c in plan["cases"] if c["frozen_action"]["action"] == "regenerate_cxr"]
    if {b["case_id"] for b in books} != {c["case_id"] for c in routed} or len(books) != len(routed):
        raise ValueError("missing/duplicate case budget ledger")
    for book in books:
        case = next(c for c in plan["cases"] if c["case_id"] == book["case_id"])
        if book["ehr_anchor_sha256"] != case["ehr_anchor_sha256"]:
            raise ValueError("case cost ledger changed EHR anchor")
        rebuilt = restore_ledger(book["events"], case_id=book["case_id"], ehr_anchor_sha256=book["ehr_anchor_sha256"],
            call_budget=4, max_retries=0, execution_mode="approved_slurm_backend", sink=lambda event: None)
        if rebuilt.snapshot() != book or book["charged_model_attempts"] > 4 or book["pending_attempts"]:
            raise ValueError("durable attempt accounting/budget differs")
        cr = root / "cases" / book["case_id"]
        actual = [json.loads(line) for line in (cr / "execution.journal.jsonl").read_text().splitlines() if line]
        if actual != book["events"] or read_json(cr / "ledger_snapshot.json") != book:
            raise ValueError("case journal/snapshot differs")
        for event in book["events"]:
            if event["event"] != "attempt_reserved":
                continue
            request = event["request"]
            spec = plan["workers"].get(request["model_id"])
            if (spec is None or request["kind"] != spec["kind"] or request["seed"] != 1
                    or request["frozen_model_audit_sha256"] != _digest(spec)):
                raise ValueError("reserved worker, native seed or frozen audit differs")
    if (manifest["generation_verification_attempts_including_failures"] != sum(b["charged_model_attempts"] for b in books)
            or manifest["failed_generation_verification_attempts"] != sum(b["failed_attempts"] for b in books)
            or manifest["completed_new_chains"] != len(triples)
            or manifest["routed_retry_cases"] != len(routed)
            or manifest["exploratory_gate_pass_cases"] != sum(c["status"] == "exploratory_retry_gate_pass" for c in expected_choices)):
        raise ValueError("published cost/outcome counts differ")
    endpoint = read_json(root / "endpoint.json")
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    if set(scores) != set(indexed) or len(scores) != len(endpoint["records"]) or endpoint["used_for_routing"] is not False:
        raise ValueError("complete secondary-only endpoint required")
    from tricompose_v12.full_pool_report_control import validate_endpoint
    # Validation does not select or re-run models. It preserves NA semantics.
    validate_endpoint(rows, endpoint)
    table, case_rows = tables(rows, selection["choices"], endpoint)
    verify_csv(root / "score_table.csv", table)
    verify_csv(root / "case_comparison.csv", case_rows)
    if manifest["endpoint_encodings_separate_from_retry_budget"] != endpoint["counts"]:
        raise ValueError("separate endpoint cost differs")
    if endpoint["counts"] is not None:
        native = read_json(root / "secondary/scores.json")
        native["historical_pool_is_untouched_test"] = False
        if (native != endpoint or endpoint["counts"]["requested_pairs"] != len(rows)
                or endpoint["counts"]["image_encoder_calls"] > 4
                or endpoint["counts"]["text_encoder_calls"] > 6):
            raise ValueError("authenticated bounded fresh endpoint differs")
    events = [json.loads(line) for line in (root / "controller.journal.jsonl").read_text().splitlines() if line]
    seal = next(i for i, e in enumerate(events) if e["stage"] == "choice")
    reservation = next(i for i, e in enumerate(events) if e["stage"] == "secondary_endpoint")
    if (seal >= reservation or events[seal]["selection_sha256"] != manifest["selection_sha256_before_endpoint"]):
        raise ValueError("endpoint was accessed before choice seal")
    result = {"schema_version": "tricompose-bounded-regeneration-metadata-audit-v1", "status": "passed_metadata_not_clinical",
        "source_manifest_sha256": sha256_file(root / "manifest.json"),
        "plan_manifest_sha256": args.plan_manifest_sha256, "audited_fixed_ehr_cases": 2,
        "recomputed_new_receipts": len(triples), "charged_attempts": sum(b["charged_model_attempts"] for b in books),
        "clinical_accuracy_verified": False, "new_model_calls": 0, "source_bodies_or_pixels_opened": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(temporary / "audit.json", result)
        write_private_json(temporary / "manifest.json", {"schema_version": result["schema_version"],
            "audit_sha256": sha256_file(path), "status": result["status"], "new_model_calls": 0})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "plan-manifest-sha256", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    try:
        root = audit(args)
    except Exception as exc:
        print(json.dumps({"status": "metadata_audit_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "passed_metadata_not_clinical", "runtime_seconds": round(time.monotonic() - started, 3),
        "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

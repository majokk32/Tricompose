"""Fixed operational smoke: real workers, fixed EHR, bounded durable attempts.

This does not implement a quality-based router, clinical acceptance or repair.
Outer run paths stay stable: child adapters embed absolute artifact paths.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import time

from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
                       private_directory, write_private_json, write_private_text)
from .execution_ledger import BoundedCallLedger, CallRequest, CallResult, BudgetExhausted, RetryExhausted
from .invariant_verification import _digest, _ID
from .live_plan import load_plan, anchor_from_record
from .live_receipts import image_receipt, completed_receipt, PROFILE
from .live_workers import LocalFrozenBackend, single_generated, require_gpu_slurm, check_pins
from .runtime_dispatch import ProtectedJournal, dispatch_operation

SCHEMA = "tricompose-live-frozen-worker-smoke-v1"


def copy_private_bytes(source, target, expected):
    # Only authenticated, fully synthetic inputs. Never parse/print their bodies.
    source = require_inside(source, PROTECTED_ROOT, must_exist=True)
    if sha256_file(source) != expected: raise ValueError("fixed input changed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o660)
    with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
        os.fchmod(dst.fileno(), 0o660)
        for chunk in iter(lambda: src.read(1024*1024), b""): dst.write(chunk)
        dst.flush(); os.fsync(dst.fileno())
    if sha256_file(target) != expected: raise ValueError("copied fixed input hash differs")


def clone_cxr_request(row, output):
    from tricompose_v11.cxr_contracts import validate_cxr_request, canonical_json_sha256
    request = row["request"]; validate_cxr_request(request)
    if canonical_json_sha256(request) != row["canonical_request_sha256"]:
        raise ValueError("final original input request changed")
    private_directory(output); private_directory(output/"requests")
    path = write_private_json(output/"requests"/(request["request_id"]+".json"), request)
    staging = Path(request["inputs"]["staging_run_manifest"]["path"])
    write_private_json(output/"manifest.json", {"schema_version": "tricompose-cxr-request-run-v1.1",
        "run_id": "request_pack", "staging_run": str(staging.parent),
        "staging_run_manifest_sha256": request["inputs"]["staging_run_manifest"]["sha256"],
        "case_ids": [request["case_id"]], "model_ids": [request["model_id"]], "seeds": [request["seed"]],
        "require_conditioned": False, "request_count": 1, "gpu_inference_used": False,
        "requests": [{"request_id": request["request_id"], "case_id": request["case_id"],
            "model_id": request["model_id"], "seed": request["seed"],
            "path": "requests/"+path.name, "sha256": sha256_file(path)}],
        "original_request_sha256": row["source_sha256"], "prompt_or_ehr_changed": False})
    return output


def validate_cxr_binding(candidate, row, anchor):
    request = row["request"]
    if (candidate["case_id"] != anchor.case_id or candidate["seed"] != request["seed"]
            or candidate["model_id"] != request["model_id"] or candidate["ehr_sha256"] != anchor.ehr_sha256
            or candidate["ehr_facts_sha256"] != anchor.ehr_facts_sha256
            or candidate["input_request_sha256"] != row["canonical_request_sha256"]
            or candidate["prompt_sha256"] != request["inputs"]["final_prompt"]["sha256"]
            or candidate["clinical_intent_sha256"] != request["inputs"]["final_prompt"]["clinical_intent_sha256"]
            or candidate.get("adapter_added_prefix") is not False
            or candidate["cost"].get("model_calls") != 1):
        raise ValueError("generated image did not consume the exact fixed EHR-derived prompt")


def validate_report_binding(candidate, cxr, request_run, anchor):
    from tricompose_v11.cxr_contracts import canonical_json_sha256
    manifest = read_json(Path(request_run)/"manifest.json")
    if manifest["request_count"] != 1 or len(manifest["requests"]) != 1:
        raise ValueError("single report request required")
    row = manifest["requests"][0]
    rp = require_inside(Path(request_run)/row["path"], request_run, must_exist=True)
    if sha256_file(rp) != row["sha256"]: raise ValueError("report request changed")
    request = read_json(rp)
    if (candidate["case_id"] != anchor.case_id or candidate["parent_cxr_candidate_id"] != cxr["candidate_id"]
            or candidate["input_cxr"]["sha256"] != cxr["artifact"]["sha256"]
            or candidate["ehr_sha256_retained_for_lineage"] != anchor.ehr_sha256
            or candidate["ehr_facts_sha256_retained_for_lineage"] != anchor.ehr_facts_sha256
            or candidate["input_report_request_sha256"] != canonical_json_sha256(request)
            or candidate.get("model_input_signature") != "single_current_synthetic_cxr"
            or candidate.get("structured_ehr_content_supplied_to_model") is not False
            or candidate.get("source_report_or_real_target_supplied") is not False):
        raise ValueError("report input/lineage changed or extra source/target supplied")


def attempt(ledger, request, spec, case_root, policy, validator, **inputs):
    # Only identical-operation timeout retry. No quality-based regeneration here.
    last = None
    for number in range(1, policy["max_retries_per_operation"]+2):
        if ledger.charged_attempts >= ledger.budget: return None
        op_root = case_root/"operations"/f"{request.operation_id}_a{number}"
        private_directory(op_root)
        backend = LocalFrozenBackend(spec, operation_root=op_root,
            timeout=policy["timeout_seconds_per_worker_process"], **inputs)
        try:
            last = dispatch_operation(ledger, request, backend, validator,
                private_log_path=op_root/"dispatch.log")
        except (BudgetExhausted, RetryExhausted): return None
        if last is not None: return last
        event = ledger.snapshot()["events"][-1]
        if event.get("event") != "attempt_failed" or not event.get("retryable"): return None
    return last


def one_case(case, plan, root):
    from tricompose_v11.report_contracts import prepare_report_request_run
    anchor = anchor_from_record(case["anchor"])
    policy = plan["policy"]; workers = plan["workers"]
    cr = root/"cases"/anchor.case_id
    private_directory(cr); private_directory(cr/"inputs"); private_directory(cr/"inputs/cxr_prompts")
    private_directory(cr/"operations")
    write_private_json(cr/"ehr_anchor.json", anchor.record())
    for key, filename in (("synthetic_ehr", "synthetic_ehr.json"), ("ehr_facts", "ehr_facts.json")):
        source = case["requests"][0]["request"]["inputs"][key]
        copy_private_bytes(source["path"], cr/"inputs"/filename, source["sha256"])
    for row in case["requests"]:
        source = row["request"]["inputs"]["final_prompt"]
        copy_private_bytes(source["path"], cr/"inputs/cxr_prompts"/(row["request"]["model_id"]+".txt"), source["sha256"])
    triples = []; partials = []; successful = []
    thresholds = Path(plan["workers"]["xrv"]["thresholds_path"])
    with ProtectedJournal(cr/"execution.journal.jsonl") as journal:
        ledger = BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
            call_budget=policy["call_budget_per_case"], max_retries=policy["max_retries_per_operation"],
            execution_mode="approved_slurm_backend", sink=journal.append)

        def call(kind, model, op, validator, seed, parent=None, image=None, report=None, **inputs):
            spec = workers[model]
            request = CallRequest(op, anchor.case_id, anchor.sha256, kind, model, _digest(spec), seed, parent, image, report)
            result = attempt(ledger, request, spec, cr, policy, validator, **inputs)
            if result is not None: successful.append({"operation_id": op, "kind": kind, "result": result.record()})
            return result

        for slot, row in enumerate(case["requests"]):
            req = row["request"]; model = req["model_id"]; seed = req["seed"]
            request_run = clone_cxr_request(row, cr/"operations"/f"request_cxr_{slot}")
            state = {}

            def image_validator(payload, operation):
                run, candidate = single_generated(payload, workers[model], request_run)
                validate_cxr_binding(candidate, row, anchor)
                state.update(cxr_run=run, cxr=candidate)
                return CallResult(candidate["artifact"]["sha256"])

            image_op = f"cxr_{slot}"
            result = call("cxr_generator", model, image_op, image_validator, seed, request_run=request_run)
            if result is None: continue
            image_hash = result.output_artifact_sha256

            def xrv_validator(payload, operation):
                if payload["worker_audit_sha256"] != operation.frozen_model_audit_sha256:
                    raise ValueError("classifier worker audit differs")
                op_root = Path(payload["output_root"])
                file = require_inside(op_root/"scored/cxr_finding_labels.json", op_root, must_exist=True)
                labels = read_json(file); spec = workers["xrv"]
                if (labels["thresholds"] != spec["thresholds"]
                        or labels["calibration"]["status"] != spec["calibration_status"]
                        or any(labels["producer"].get(k) != v for k,v in spec["scorer_provenance"].items())):
                    raise ValueError("frozen XRV threshold/preprocessing profile changed")
                receipt = image_receipt(anchor, state["cxr"], labels, label_sha256=sha256_file(file),
                    thresholds_sha256=spec["thresholds_sha256"], checkpoint_sha256=spec["checkpoint_sha256"])
                rp = write_private_json(op_root/"image_receipt.json", receipt)
                state.update(image_labels=labels, image_labels_path=file, partial=receipt, partial_path=rp)
                partials.append({"path": str(rp), "sha256": sha256_file(rp), "receipt_id": receipt["receipt_id"]})
                return CallResult(sha256_file(file), receipt["receipt_id"])

            xrv_op = f"xrv_{slot}"
            if call("xrv", "xrv", xrv_op, xrv_validator, seed, parent=image_op, image=image_hash,
                    cxr_run=state["cxr_run"], thresholds=thresholds) is None: continue

            for r_index, report_model in enumerate(policy["report_models"]):
                pack = prepare_report_request_run(cxr_runs=[state["cxr_run"]], output_root=cr/"operations",
                    run_id=f"request_report_{slot}_{r_index}", model_ids=[report_model])
                report_request_run = Path(pack["run_directory"])
                rs = {}

                def report_validator(payload, operation):
                    run, candidate = single_generated(payload, workers[report_model], report_request_run,
                        cxr_candidates={state["cxr"]["candidate_id"]: state["cxr"]})
                    validate_report_binding(candidate, state["cxr"], report_request_run, anchor)
                    rs.update(report_run=run, report=candidate)
                    return CallResult(candidate["artifact"]["sha256"])

                report_op = f"report_{slot}_{r_index}"
                result = call("report_generator", report_model, report_op, report_validator, seed, parent=xrv_op,
                    image=image_hash, request_run=report_request_run)
                if result is None: continue

                def chexbert_validator(payload, operation):
                    if payload["worker_audit_sha256"] != operation.frozen_model_audit_sha256:
                        raise ValueError("report-label worker audit differs")
                    op_root = Path(payload["output_root"])
                    file = require_inside(op_root/"scored/report_finding_labels.json", op_root, must_exist=True)
                    labels = read_json(file)
                    receipt = completed_receipt(anchor, state["partial"], state["cxr"], state["image_labels"],
                        rs["report"], labels, image_labels_sha256=sha256_file(state["image_labels_path"]),
                        report_labels_sha256=sha256_file(file), thresholds_sha256=workers["xrv"]["thresholds_sha256"],
                        xrv_checkpoint_sha256=workers["xrv"]["checkpoint_sha256"],
                        chexbert_checkpoint_sha256=workers["chexbert"]["checkpoint_sha256"])
                    rp = write_private_json(op_root/"completed_receipt.json", receipt)
                    triple = {"schema_version": SCHEMA, "case_id": anchor.case_id,
                        "ehr_anchor_sha256": anchor.sha256, "ehr_path": str(cr/"inputs/synthetic_ehr.json"),
                        "ehr_sha256": anchor.ehr_sha256, "ehr_facts_sha256": anchor.ehr_facts_sha256,
                        "cxr_model_id": model, "seed": seed, "report_model_id": report_model,
                        "cxr_path": state["cxr"]["artifact"]["path"], "cxr_sha256": image_hash,
                        "report_path": rs["report"]["artifact"]["path"], "report_sha256": result.output_artifact_sha256,
                        "cxr_run": str(state["cxr_run"]), "report_run": str(rs["report_run"]),
                        "receipt_path": str(rp), "receipt_sha256": sha256_file(rp), "receipt_id": receipt["receipt_id"],
                        "profile": PROFILE, "raw_edge_readouts": receipt["raw_edge_readouts"],
                        "verification_status": receipt["verification_status"],
                        "clinical_acceptance": False, "selected_as_best": False}
                    triples.append(triple)
                    return CallResult(sha256_file(file), receipt["receipt_id"])

                call("chexbert", "chexbert", f"chexbert_{slot}_{r_index}", chexbert_validator, seed,
                    parent=report_op, image=image_hash, report=result.output_artifact_sha256,
                    cxr_run=state["cxr_run"], report_run=rs["report_run"])
        snapshot = ledger.snapshot()
    write_private_json(cr/"ledger_snapshot.json", snapshot)
    write_private_json(cr/"case_manifest.json", {"schema_version": SCHEMA, "case_id": anchor.case_id,
        "ehr_anchor_sha256": anchor.sha256, "completed_triplets": triples, "partial_image_receipts": partials,
        "successful_operations": successful, "charged_model_attempts": ledger.charged_attempts,
        "validated_model_calls": len(successful), "clinical_acceptance": False})
    return triples, snapshot, successful


def score_csv(triples):
    columns = ["case_id", "cxr_model_id", "seed", "report_model_id", "verification_status"]
    names = ("known_reference_facts", "comparable_facts", "supported_positive", "supported_negative",
             "proxy_opposition_facts", "missing_comparisons", "support_over_known", "coverage_over_known")
    columns += [edge+"_"+n for edge in ("ehr_cxr", "ehr_report", "cxr_report") for n in names]
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=columns); writer.writeheader()
    for triple in triples:
        row = {k: triple[k] for k in columns[:5]}
        for edge, metrics in triple["raw_edge_readouts"].items():
            for n in names: row[edge+"_"+n] = "NA" if metrics[n] is None else metrics[n]
        writer.writerow(row)
    return out.getvalue()


def run(args):
    require_gpu_slurm()  # No real-worker execution or run directory in current CPU allocation.
    plan = load_plan(args.plan_run, args.plan_manifest_sha256)
    if not _ID.fullmatch(args.run_id): raise ValueError("opaque output run ID required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = output/args.run_id
    private_directory(root)  # Exclusive and stable. Never rename after child artifacts are created.
    private_directory(root/"cases")
    started = time.monotonic()
    write_private_json(root/"start_manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
        "plan_run": str(Path(args.plan_run).resolve()), "plan_manifest_sha256": args.plan_manifest_sha256,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "policy": plan["policy"],
        "status": "in_progress", "clinical_acceptance": False})
    triples = []; books = []; successful = []
    try:
        for case in plan["cases"]:
            rows, ledger, calls = one_case(case, plan, root)
            triples += rows; books.append(ledger); successful += calls
        check_pins(plan["source_pins"]); check_pins(plan["source_artifact_pins"])
        files = [write_private_text(root/"completed_triplets.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in triples)),
            write_private_text(root/"score_table.csv", score_csv(triples)),
            write_private_json(root/"execution_summary.json", {"case_ledgers": books, "successful_operations": successful})]
        summary = {"schema_version": SCHEMA, "run_id": args.run_id,
            "status": "completed_fixed_paths_unvalidated" if len(triples)==4 else "partial_fixed_paths_unvalidated",
            "completed_triplets": len(triples), "charged_model_attempts": sum(b["charged_model_attempts"] for b in books),
            "validated_model_calls": len(successful), "failed_attempts": sum(b["failed_attempts"] for b in books),
            "runtime_seconds_includes_startup_and_io": round(time.monotonic()-started, 3),
            "plan_manifest_sha256": args.plan_manifest_sha256,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "selection_or_adaptive_repair_executed": False, "clinical_acceptance": False,
            "clinical_accuracy": None, "gpu_seconds": None, "original_ehr_prompts_or_winners_changed": False}
        write_private_json(root/"manifest.json", summary)
    except Exception as exc:
        # Keep charged/in-flight journals and private outputs; do not discard or resume blindly.
        write_private_json(root/"failure_manifest.json", {"schema_version": SCHEMA, "status": "interrupted_run_retained",
            "error_type": type(exc).__name__, "plan_manifest_sha256": args.plan_manifest_sha256,
            "clinical_acceptance": False, "automatic_resume_allowed": False})
        raise
    return root, summary

"""Real same-image report generation plus unchanged frozen CheXbert worker.

Call only after local planning subprocess exits, in an approved GPU Slurm job.
The original live adapters and consumed smoke entry points are not modified.
"""
from __future__ import annotations

from pathlib import Path

from contracts import (PROTECTED_ROOT, load_cxr_candidates, private_directory,
                       read_json, require_inside, sha256_file, write_private_json)
from tricompose_v11.report_contracts import prepare_report_request_run
from tricompose_v12.execution_ledger import CallRequest, CallResult, restore_ledger
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import attempt, validate_report_binding
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import completed_receipt, image_receipt
from tricompose_v12.live_workers import require_gpu_slurm, single_generated
from tricompose_v12.runtime_dispatch import ProtectedJournal


def additional_report(case, plan, run_root, baseline_triple, prefix, *, model, step_index):
    require_gpu_slurm()  # Before inputs, directories, workers or imports of ML frameworks.
    anchor = anchor_from_record(case["anchor"])
    cr = Path(run_root) / "cases" / anchor.case_id
    workers = plan["workers"]
    images = load_cxr_candidates([baseline_triple["cxr_run"]])
    if len(images) != 1:
        raise ValueError("one_fixed_synthetic_image_required")
    image = next(iter(images.values()))
    if image["artifact"]["sha256"] != baseline_triple["cxr_sha256"]:
        raise ValueError("fixed_image_changed")
    if (workers[model]["kind"] != "report_generator" or type(step_index) is not int
            or not 0 <= step_index < 3 or prefix["pending_attempts"]):
        raise ValueError("bounded_report_expert_action_required")
    requests = {e["request"]["operation_id"]: e["request"] for e in prefix["events"]
        if e["event"] == "attempt_reserved"}
    matches = [e for e in prefix["events"] if e["event"] == "attempt_completed"
        and requests[e["operation_id"]]["kind"] == "xrv"
        and requests[e["operation_id"]]["input_image_sha256"] == image["artifact"]["sha256"]]
    if len(matches) != 1:
        raise ValueError("unique_completed_frozen_image_verification_required")
    event = matches[0]; parent = event["operation_id"]
    ip = require_inside(cr / "operations" / (parent + "_a1") / "scored/cxr_finding_labels.json",
                        PROTECTED_ROOT, must_exist=True)
    if sha256_file(ip) != event["result"]["output_artifact_sha256"]:
        raise ValueError("fixed_image_labels_changed")
    il = read_json(ip)
    partial = image_receipt(anchor, image, il, label_sha256=sha256_file(ip),
        thresholds_sha256=workers["xrv"]["thresholds_sha256"],
        checkpoint_sha256=workers["xrv"]["checkpoint_sha256"])
    if partial["receipt_id"] != event["result"]["verification_receipt_id"]:
        raise ValueError("fixed_partial_receipt_changed")
    pack = prepare_report_request_run(cxr_runs=[baseline_triple["cxr_run"]],
        output_root=cr / "operations", run_id=f"request_live_report_{step_index}", model_ids=[model])
    pack_root = Path(pack["run_directory"])
    state, triple = {}, None
    with ProtectedJournal(cr / f"continuation_{step_index}.journal.jsonl") as journal:
        ledger = restore_ledger(prefix["events"], case_id=anchor.case_id,
            ehr_anchor_sha256=anchor.sha256, call_budget=prefix["call_budget"],
            max_retries=prefix["max_retries"], execution_mode=prefix["execution_mode"], sink=journal.append)
        if ledger.snapshot() != prefix:
            raise ValueError("unchanged_cost_prefix_required")

        def report_validator(payload, operation):
            report_run, report = single_generated(payload, workers[model], pack_root, cxr_candidates=images)
            validate_report_binding(report, image, pack_root, anchor)
            state.update(report=report, report_run=report_run)
            return CallResult(report["artifact"]["sha256"])

        report_op = f"live_report_{step_index}"
        request = CallRequest(report_op, anchor.case_id, anchor.sha256, "report_generator", model,
            _digest(workers[model]), image["seed"], parent, image["artifact"]["sha256"])
        result = attempt(ledger, request, workers[model], cr, plan["policy"], report_validator, request_run=pack_root)
        if result is not None:
            def label_validator(payload, operation):
                nonlocal triple
                if payload["worker_audit_sha256"] != operation.frozen_model_audit_sha256:
                    raise ValueError("frozen_report_scorer_audit_required")
                op_root = require_inside(payload["output_root"], PROTECTED_ROOT, must_exist=True)
                tp = op_root / "scored/report_finding_labels.json"
                receipt = completed_receipt(anchor, partial, image, il, state["report"], read_json(tp),
                    image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
                    thresholds_sha256=workers["xrv"]["thresholds_sha256"],
                    xrv_checkpoint_sha256=workers["xrv"]["checkpoint_sha256"],
                    chexbert_checkpoint_sha256=workers["chexbert"]["checkpoint_sha256"])
                rp = write_private_json(op_root / "completed_receipt.json", receipt)
                triple = {"case_id": anchor.case_id, "ehr_sha256": anchor.ehr_sha256,
                    "ehr_facts_sha256": anchor.ehr_facts_sha256, "ehr_path": baseline_triple["ehr_path"],
                    "cxr_run": baseline_triple["cxr_run"], "report_run": str(state["report_run"]),
                    "receipt_path": str(rp), "receipt_sha256": sha256_file(rp),
                    "cxr_sha256": image["artifact"]["sha256"],
                    "report_sha256": state["report"]["artifact"]["sha256"],
                    "cxr_path": image["artifact"]["path"], "report_path": state["report"]["artifact"]["path"],
                    "cxr_model_id": image["model_id"], "report_model_id": model, "seed": image["seed"],
                    "raw_edge_readouts": receipt["raw_edge_readouts"], "clinical_acceptance": False}
                return CallResult(sha256_file(tp), receipt["receipt_id"])

            labels = CallRequest(f"live_chexbert_{step_index}", anchor.case_id, anchor.sha256,
                "chexbert", "chexbert", _digest(workers["chexbert"]), image["seed"], report_op,
                image["artifact"]["sha256"], result.output_artifact_sha256)
            label_result = attempt(ledger, labels, workers["chexbert"], cr, plan["policy"], label_validator,
                cxr_run=baseline_triple["cxr_run"], report_run=state["report_run"])
            if label_result is None:
                triple = None
        book = ledger.snapshot()
    write_private_json(cr / f"continuation_{step_index}_snapshot.json", book)
    return triple, book

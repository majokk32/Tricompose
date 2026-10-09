"""Invented receipts and fake backends only; no model or clinical-body reads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
import run_guarded_requested_reports as worker
from test_fresh_output_acceptance import fixture
from test_fresh_report_agent_bridge import base_ledger
from test_guarded_fresh_policy import case_fixture, decision
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher
from tricompose_v12.execution_ledger import CallRequest, CallResult
from tricompose_v12.invariant_verification import _digest


def ingredients():
    packet_case = case_fixture()
    packet = packet_case["packet"]
    d = decision(packet)
    dispatch, _ = GuardedActionDispatcher(sink=lambda event: None).dispatch(d,
        packet["observation"], packet["image_guard"], backend_factory=None)
    baseline, ctx = fixture(image_state="negative", report_state="unknown", seed=1,
        image_id="invented_trial_image", report_id="invented_trial_report", model="cxrmate_single")
    book = base_ledger(baseline, ctx).snapshot()
    # Create a genuinely four-budget fixture, rather than changing the old chain.
    from tricompose_v12.execution_ledger import BoundedCallLedger
    small = BoundedCallLedger(case_id=book["case_id"], ehr_anchor_sha256=book["ehr_anchor_sha256"],
        call_budget=4, max_retries=0, execution_mode="invented_fixture_no_models", sink=lambda e: None)
    for e in book["events"]:
        if e["event"] == "attempt_reserved": token = small.reserve(CallRequest(**e["request"]))
        else: small.complete(token, CallResult(**e["result"]), elapsed_seconds=e["backend_elapsed_seconds"])
    book = small.snapshot()
    specs = {"xrv": {"thresholds": {n:{"enabled":n in ctx["enabled_xrv_findings"]}
        for n in worker.gate.FINDINGS}, "thresholds_sha256":ctx["thresholds_sha256"],
        "checkpoint_sha256":ctx["xrv_checkpoint_sha256"]},
        "chexbert":{"checkpoint_sha256":ctx["chexbert_checkpoint_sha256"]}}
    old = {"case_id":baseline["case_id"], "anchor":ctx["anchor"], "reference":baseline, "workers":specs}
    triple = {k:baseline[k] for k in ("case_id","ehr_sha256","ehr_facts_sha256","cxr_sha256","seed","report_model_id")}
    result = {"case_id":baseline["case_id"], "source":"new_frozen_qwen_generate_call", "decision":d,
        "dispatch":dispatch, "policy_input_sha256":packet_case["policy_input_sha256"],
        "clinical_acceptance":False, "original_winner_changed":False}
    pending = {"case_id":baseline["case_id"], "request":packet_case["tool_catalog"][d["target_id"]],
        "decision":d, "clinical_acceptance":False}
    return packet_case,result,pending,old,baseline,triple,book


def case_fixture_bound():
    return worker.bind_request(*ingredients())


def request(case, kind="report_generator", *, report_hash=None):
    anchor=worker.gate.anchor_from_record(case["anchor"])
    return CallRequest("requested_report" if kind=="report_generator" else "requested_chexbert",
        anchor.case_id, anchor.sha256, kind, case["requested_model"] if kind=="report_generator" else "chexbert",
        _digest("fixture_audit"), 1, case["cached_xrv_operation_id"] if kind=="report_generator" else "requested_report",
        case["baseline_row"]["cxr_sha256"], report_hash)


def backend(req, callback=lambda request: {}):
    return SimpleNamespace(frozen=True, audit_sha256=req.frozen_model_audit_sha256,
        execution_mode="approved_slurm_backend", invoke=callback)


class RequestedReportTests(unittest.TestCase):
    def test_binding_keeps_exhausted_historical_ledger_and_fixed_image(self):
        parts=ingredients(); before=deepcopy(parts)
        case=worker.bind_request(*parts)
        self.assertEqual(parts,before)
        self.assertEqual(case["historical_ledger"]["call_budget"],4)
        self.assertEqual(case["historical_ledger"]["charged_model_attempts"],4)
        self.assertEqual(case["historical_ledger_sha256"],_digest(parts[-1]))
        self.assertEqual(case["requested_model"],"maira2")
        self.assertFalse(case["original_selection_change_allowed"])

    def test_changed_pending_model_or_image_is_rejected(self):
        for key,value in (("model_id","llavarad"),("input_image_sha256",_digest("wrong_image")),("seed",0)):
            parts=list(ingredients()); parts[2]["request"][key]=value
            with self.assertRaises(ValueError): worker.bind_request(*parts)

    def test_different_ehr_is_rejected(self):
        parts=list(ingredients()); parts[4]["ehr_sha256"]=_digest("wrong_ehr")
        with self.assertRaises(ValueError): worker.bind_request(*parts)

    def test_two_new_calls_are_reserved_before_factory_and_old_cost_stays_four(self):
        case=case_fixture_bound(); events=[]
        calls=worker.RequestedReportCalls(case,sink=events.append)
        r=request(case)
        def factory():
            self.assertEqual(events[-1]["event"],"new_worker_reserved")
            return backend(r)
        first=calls.invoke(r,factory=factory,validator=lambda payload,op:CallResult(_digest("new_report")))
        c=request(case,"chexbert",report_hash=first.output_artifact_sha256)
        calls.invoke(c,factory=lambda:backend(c),validator=lambda payload,op:CallResult(_digest("new_labels"),_digest("new_receipt")))
        self.assertEqual(calls.snapshot()["charged_new_worker_attempts"],2)
        self.assertEqual(calls.snapshot()["completed_new_worker_operations"],2)
        self.assertEqual(case["historical_ledger"]["charged_model_attempts"],4)
        with self.assertRaises(ValueError): calls.invoke(c,factory=lambda:backend(c),validator=lambda *x:None)

    def test_failed_report_is_charged_and_prevents_labels_or_retry(self):
        case=case_fixture_bound(); events=[]; calls=worker.RequestedReportCalls(case,sink=events.append)
        r=request(case)
        def fail(): raise TimeoutError("invented")
        self.assertIsNone(calls.invoke(r,factory=fail,validator=lambda *x:None))
        self.assertEqual(calls.charged,1); self.assertEqual(calls.failed,1)
        self.assertEqual(events[-1]["error_code"],"timeout")
        with self.assertRaises(ValueError): calls.invoke(r,factory=lambda:backend(r),validator=lambda *x:None)

    def test_failed_chexbert_retains_both_charges(self):
        case=case_fixture_bound(); calls=worker.RequestedReportCalls(case,sink=lambda event:None)
        r=request(case); h=_digest("new_report")
        calls.invoke(r,factory=lambda:backend(r),validator=lambda *x:CallResult(h))
        c=request(case,"chexbert",report_hash=h)
        calls.invoke(c,factory=lambda:backend(c),validator=lambda *x:None)
        self.assertEqual(calls.charged,2); self.assertEqual(calls.completed,1); self.assertEqual(calls.failed,1)

    def test_durable_reservation_failure_never_constructs_backend(self):
        case=case_fixture_bound()
        def bad_sink(event): raise OSError("invented_fsync_failure")
        calls=worker.RequestedReportCalls(case,sink=bad_sink)
        factory=unittest.mock.Mock()
        with self.assertRaises(OSError): calls.invoke(request(case),factory=factory,validator=lambda *x:None)
        factory.assert_not_called()
        self.assertTrue(calls.pending)
        with self.assertRaises(ValueError): calls.invoke(request(case),factory=factory,validator=lambda *x:None)

    def test_completed_journal_failure_cannot_repeat_a_model_call(self):
        case=case_fixture_bound(); invocations=[]
        def sink(event):
            if event["event"]=="new_worker_completed": raise OSError("invented_fsync_failure")
        calls=worker.RequestedReportCalls(case,sink=sink); r=request(case)
        with self.assertRaises(OSError): calls.invoke(r,factory=lambda:backend(r,lambda op:invocations.append(op)),
            validator=lambda *x:CallResult(_digest("new_report")))
        self.assertEqual(len(invocations),1)
        with self.assertRaises(ValueError): calls.invoke(r,factory=lambda:backend(r),validator=lambda *x:None)

    def test_label_dependency_cannot_use_another_report_hash(self):
        case=case_fixture_bound(); calls=worker.RequestedReportCalls(case,sink=lambda event:None)
        r=request(case); calls.invoke(r,factory=lambda:backend(r),validator=lambda *x:CallResult(_digest("new_report")))
        c=request(case,"chexbert",report_hash=_digest("wrong_report"))
        with self.assertRaises(ValueError): calls.invoke(c,factory=lambda:backend(c),validator=lambda *x:None)
        self.assertEqual(calls.charged,1)

    def test_cosmetic_unknown_report_is_not_improvement(self):
        case=case_fixture_bound()
        alt,_=fixture(image_state="negative",report_state="unknown",seed=1,image_id="invented_trial_image",
            report_id="invented_alternative",model="maira2")
        old=case["baseline_row"]
        result=worker.assess_pair(case,alt,ingredients()[3]["workers"])
        self.assertFalse(result["comparison"]["exploratory_gate_pass"])
        self.assertEqual(result["branch_selected_candidate_id"],old["triple_candidate_id"])
        self.assertFalse(result["clinical_acceptance"])

    def test_strict_gain_can_only_change_local_branch_not_original_winner(self):
        case=case_fixture_bound()
        baseline,_=fixture(image_state="positive",report_state="negative",seed=1,image_id="invented_trial_image",
            report_id="invented_baseline",model="cxrmate_single")
        case["baseline_row"]=baseline
        alt,_=fixture(image_state="positive",report_state="positive",seed=1,image_id="invented_trial_image",
            report_id="invented_alternative",model="maira2")
        result=worker.assess_pair(case,alt,ingredients()[3]["workers"])
        self.assertTrue(result["comparison"]["exploratory_gate_pass"])
        self.assertEqual(result["branch_selected_candidate_id"],alt["triple_candidate_id"])
        self.assertFalse(result["original_winner_changed"])
        self.assertFalse(result["clinical_repair_success"])

    def test_conflict_silenced_to_unknown_cannot_pass(self):
        case=case_fixture_bound()
        baseline,_=fixture(image_state="positive",report_state="negative",seed=1,image_id="invented_trial_image",
            report_id="invented_baseline",model="cxrmate_single")
        case["baseline_row"]=baseline
        alt,_=fixture(image_state="positive",report_state="unknown",seed=1,image_id="invented_trial_image",
            report_id="invented_alternative",model="maira2")
        result=worker.assess_pair(case,alt,ingredients()[3]["workers"])
        self.assertFalse(result["comparison"]["exploratory_gate_pass"])
        self.assertIn("conflict_silenced_not_corrected",result["comparison"]["reasons"])

    def test_wrong_image_cannot_be_compared_as_report_only(self):
        case=case_fixture_bound()
        alt,_=fixture(seed=1,image_id="invented_different_image",report_id="invented_alt",model="maira2")
        with self.assertRaises(ValueError): worker.assess_pair(case,alt,ingredients()[3]["workers"])

    def test_gpu_guard_precedes_plan_or_input_reads(self):
        with patch.object(worker.producer,"gpu_guard",side_effect=RuntimeError("no_gpu")), \
             patch.object(worker,"read_json") as read:
            with self.assertRaises(RuntimeError): worker.run(SimpleNamespace())
            read.assert_not_called()
            with self.assertRaises(RuntimeError): worker.run_case({}, {}, Path("."))
            read.assert_not_called()

    def test_cpu_guard_precedes_preparation_reads(self):
        with patch.object(worker.gate,"cpu_guard",side_effect=RuntimeError("no_cpu_allocation")), \
             patch.object(worker,"Reader") as reader:
            with self.assertRaises(RuntimeError): worker.prepare(SimpleNamespace())
            reader.assert_not_called()


if __name__ == "__main__": unittest.main()

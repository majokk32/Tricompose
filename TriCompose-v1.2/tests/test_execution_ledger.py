"""Invented call metadata: no models, images, reports or clinical outcomes."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tricompose_v12.execution_ledger import (
    BoundedCallLedger, CallRequest, CallResult, BudgetExhausted, RetryExhausted, restore_ledger,
)

ANCHOR = "a" * 64; AUDIT = "b" * 64; IMAGE = "c" * 64
LABELS = "d" * 64; REPORT = "e" * 64; RECEIPT = "f" * 64


def ledger(*, budget=5, retries=1, sink=None):
    return BoundedCallLedger(case_id="fixture_execution_000", ehr_anchor_sha256=ANCHOR,
        call_budget=budget, max_retries=retries, execution_mode="invented_fixture_no_models",
        sink=sink if sink is not None else lambda event: None)


def requests():
    base = {"case_id":"fixture_execution_000", "ehr_anchor_sha256":ANCHOR,
        "frozen_model_audit_sha256":AUDIT, "seed":0}
    return [
        CallRequest(operation_id="image_0", kind="cxr_generator", model_id="chexgenbench_sana", **base),
        CallRequest(operation_id="xrv_0", kind="xrv", model_id="xrv_densenet121",
            parent_operation_id="image_0", input_image_sha256=IMAGE, **base),
        CallRequest(operation_id="report_0", kind="report_generator", model_id="maira2",
            parent_operation_id="xrv_0", input_image_sha256=IMAGE, **base),
        CallRequest(operation_id="chexbert_0", kind="chexbert", model_id="chexbert",
            parent_operation_id="report_0", input_image_sha256=IMAGE, input_report_sha256=REPORT, **base),
    ]


def results():
    return [CallResult(IMAGE), CallResult(LABELS, RECEIPT), CallResult(REPORT), CallResult(LABELS, RECEIPT)]


def complete_first(book, n=4):
    for req, result in zip(requests()[:n], results()[:n]):
        token = book.reserve(req); book.complete(token, result, elapsed_seconds=0.125)


def restore(book, **overrides):
    opts = dict(case_id=book.case_id, ehr_anchor_sha256=book.anchor, call_budget=book.budget,
        max_retries=book.max_retries, execution_mode=book.mode, sink=lambda event: None)
    opts.update(overrides)
    return restore_ledger(book.snapshot()["events"], **opts)


class ExecutionLedgerTests(unittest.TestCase):
    def test_four_distinct_generation_scoring_attempts_are_not_two(self):
        book = ledger(); complete_first(book); s = book.snapshot()
        self.assertEqual(s["charged_model_attempts"], 4)
        self.assertEqual(s["charged_attempts_by_kind"], {"cxr_generator":1,"xrv":1,"report_generator":1,"chexbert":1})
        self.assertEqual(s["actual_model_calls"], 0)
        self.assertFalse(s["clinical_acceptance"])

    def test_frozen_budget_anchor_mode_and_retry_properties(self):
        book = ledger()
        for name, value in (("budget",100), ("anchor","f"*64), ("max_retries",10), ("mode","approved_slurm_backend")):
            with self.assertRaises(AttributeError): setattr(book, name, value)

    def test_reservation_is_journaled_before_caller_can_start(self):
        seen = []; book = ledger(sink=seen.append)
        token = book.reserve(requests()[0])
        self.assertEqual(seen[0]["reservation_id"], token)
        self.assertEqual(book.snapshot()["pending_attempts"], 1)
        self.assertEqual(book.charged_attempts, 1)

    def test_journal_failure_returns_no_reservation_and_does_not_commit_state(self):
        def bad_sink(event): raise OSError("invented journal failure")
        book = ledger(sink=bad_sink)
        with self.assertRaises(OSError): book.reserve(requests()[0])
        self.assertEqual(book.charged_attempts, 0)
        self.assertEqual(book.snapshot()["events"], [])

    def test_completion_journal_failure_keeps_pending_charge(self):
        calls = []
        def sink(event):
            calls.append(event)
            if len(calls) == 2: raise OSError("invented completion sink failure")
        book = ledger(sink=sink); token = book.reserve(requests()[0])
        with self.assertRaises(OSError): book.complete(token, results()[0], elapsed_seconds=1)
        self.assertEqual(book.snapshot()["pending_attempts"], 1)
        self.assertEqual(book.charged_attempts, 1)
        with self.assertRaises(ValueError): book.reuse(requests()[0])

    def test_failed_retry_is_charged_again_without_refund(self):
        book = ledger(); r = requests()[0]; t = book.reserve(r)
        book.fail(t, error_code="timeout", retryable=True, elapsed_seconds=1)
        t = book.reserve(r); book.complete(t, results()[0], elapsed_seconds=1)
        self.assertEqual(book.charged_attempts, 2)
        self.assertEqual(book.snapshot()["failed_attempts"], 1)
        self.assertFalse(book.snapshot()["events"][1]["failure_is_clinical_contradiction"])

    def test_retry_limit_and_nonretryable_failures_block_launch(self):
        for retries, retryable in ((0,True),(1,False)):
            book = ledger(retries=retries); t = book.reserve(requests()[0])
            book.fail(t,error_code="invalid_result",retryable=retryable,elapsed_seconds=0)
            with self.assertRaises(RetryExhausted): book.reserve(requests()[0])
        book = ledger()
        for _ in range(2):
            t=book.reserve(requests()[0]); book.fail(t,error_code="runtime_exception",retryable=True,elapsed_seconds=0)
        with self.assertRaises(RetryExhausted): book.reserve(requests()[0])
        self.assertEqual(book.charged_attempts, 2)

    def test_retry_cannot_change_seed_model_audit_or_input(self):
        book = ledger(); r=requests()[0]; t=book.reserve(r)
        book.fail(t,error_code="timeout",retryable=True,elapsed_seconds=0)
        for fields in ({"seed":1},{"model_id":"roentgen_v2"},{"frozen_model_audit_sha256":"c"*64}):
            with self.assertRaises(ValueError): book.reserve(replace(r,**fields))
        self.assertEqual(book.charged_attempts,1)

    def test_budget_accounts_failed_and_inflight_attempts(self):
        book=ledger(budget=1); t=book.reserve(requests()[0])
        book.fail(t,error_code="timeout",retryable=True,elapsed_seconds=1)
        with self.assertRaises(BudgetExhausted): book.reserve(requests()[0])
        self.assertEqual(book.charged_attempts,1)
        with self.assertRaises(BudgetExhausted): ledger(budget=0).reserve(requests()[0])

    def test_report_not_launched_before_image_scoring_success(self):
        book=ledger(); complete_first(book,1)
        with self.assertRaises(ValueError): book.reserve(requests()[2])
        t=book.reserve(requests()[1]); book.fail(t,error_code="runtime_exception",retryable=False,elapsed_seconds=0)
        with self.assertRaises(ValueError): book.reserve(requests()[2])

    def test_missing_report_generation_blocks_report_scoring(self):
        book=ledger(); complete_first(book,2)
        with self.assertRaises(ValueError): book.reserve(requests()[3])

    def test_changed_case_anchor_or_image_hash_is_refused_before_charge(self):
        book=ledger(); complete_first(book,2)
        for fields in ({"case_id":"foreign_case"},{"ehr_anchor_sha256":"f"*64},{"input_image_sha256":"d"*64}):
            with self.assertRaises(ValueError): book.reserve(replace(requests()[2],**fields))
        self.assertEqual(book.charged_attempts,2)

    def test_changed_report_hash_or_wrong_parent_kind_refused(self):
        book=ledger(); complete_first(book,3)
        for fields in ({"input_report_sha256":"c"*64},{"parent_operation_id":"image_0"}):
            with self.assertRaises(ValueError): book.reserve(replace(requests()[3],**fields))

    def test_one_inflight_call_prevents_duplicate_or_parallel_launch(self):
        book=ledger(); book.reserve(requests()[0])
        with self.assertRaises(RuntimeError): book.reserve(requests()[0])
        self.assertEqual(book.charged_attempts,1)

    def test_successful_image_and_scorer_reuse_is_zero_new_cost(self):
        book=ledger(); complete_first(book,4)
        self.assertEqual(book.reuse(requests()[0]),results()[0])
        self.assertEqual(book.reuse(requests()[1]),results()[1])
        self.assertEqual(book.charged_attempts,4)
        with self.assertRaises(ValueError): book.reserve(requests()[0])
        with self.assertRaises(ValueError): book.reuse(replace(requests()[0],seed=1))

    def test_second_report_reuses_image_and_stops_unscored_at_budget(self):
        book=ledger(); complete_first(book,4)
        r=replace(requests()[2],operation_id="report_1",model_id="cxrmate_single")
        t=book.reserve(r); book.complete(t,CallResult("a"*64),elapsed_seconds=0)
        s=replace(requests()[3],operation_id="chexbert_1",parent_operation_id="report_1",input_report_sha256="a"*64)
        with self.assertRaises(BudgetExhausted): book.reserve(s)
        self.assertEqual(book.snapshot()["charged_attempts_by_kind"]["cxr_generator"],1)
        self.assertEqual(book.snapshot()["completed_operations"],5)

    def test_scorer_requires_receipt_and_generator_cannot_claim_receipt(self):
        book=ledger(); t=book.reserve(requests()[0])
        with self.assertRaises(ValueError): book.complete(t,CallResult(IMAGE,RECEIPT),elapsed_seconds=0)
        book.complete(t,CallResult(IMAGE),elapsed_seconds=0); t=book.reserve(requests()[1])
        with self.assertRaises(ValueError): book.complete(t,CallResult(LABELS),elapsed_seconds=0)
        self.assertEqual(book.snapshot()["pending_attempts"],1)

    def test_failure_logs_fixed_codes_not_free_text(self):
        book=ledger(); t=book.reserve(requests()[0])
        with self.assertRaises(ValueError): book.fail(t,error_code="sensitive free text",retryable=True,elapsed_seconds=0)
        self.assertEqual(book.snapshot()["pending_attempts"],1)

    def test_invalid_elapsed_times_and_double_completion_refused(self):
        book=ledger(); t=book.reserve(requests()[0])
        for elapsed in (-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError): book.complete(t,results()[0],elapsed_seconds=elapsed)
        book.complete(t,results()[0],elapsed_seconds=0)
        with self.assertRaises(ValueError): book.complete(t,results()[0],elapsed_seconds=0)

    def test_journal_restores_exactly_without_writing_old_events(self):
        book=ledger(); complete_first(book,4); book.reuse(requests()[0]); seen=[]
        resumed=restore(book,sink=seen.append)
        self.assertEqual(resumed.snapshot(),book.snapshot()); self.assertEqual(seen,[])
        resumed.reuse(requests()[1]); self.assertEqual(len(seen),1)

    def test_pending_crash_restoration_preserves_charge_and_blocks_blind_retry(self):
        book=ledger(); t=book.reserve(requests()[0]); resumed=restore(book)
        self.assertEqual(resumed.charged_attempts,1)
        self.assertEqual(resumed.snapshot()["pending_attempts"],1)
        with self.assertRaises(RuntimeError): resumed.reserve(requests()[0])
        resumed.fail(t,error_code="interrupted",retryable=True,elapsed_seconds=0)
        resumed.reserve(requests()[0]); self.assertEqual(resumed.charged_attempts,2)

    def test_restore_cannot_increase_budget_change_retries_or_mode(self):
        book=ledger(); complete_first(book,1)
        for fields in ({"call_budget":50},{"max_retries":2},{"execution_mode":"approved_slurm_backend"}):
            with self.assertRaises(ValueError): restore(book,**fields)

    def test_tampered_removed_reordered_or_python_equal_journal_fields_refused(self):
        book=ledger(); complete_first(book,2); events=book.snapshot()["events"]
        altered=copy.deepcopy(events); altered[0]["charged_model_attempts"]=1.0
        self.assertEqual(altered,events)
        for trial in (altered,events[1:],list(reversed(events))):
            with self.assertRaises(ValueError):
                restore_ledger(trial,case_id=book.case_id,ehr_anchor_sha256=book.anchor,
                    call_budget=book.budget,max_retries=book.max_retries,execution_mode=book.mode,sink=lambda e:None)

    def test_snapshot_and_sink_mutation_cannot_corrupt_internal_events(self):
        def mutate(event): event["charged_model_attempts"]=999
        book=ledger(sink=mutate); book.reserve(requests()[0]); before=book.snapshot()
        changed=book.snapshot(); changed["events"][0]["charged_model_attempts"]=100
        self.assertEqual(book.snapshot(),before)
        self.assertEqual(before["events"][0]["charged_model_attempts"],1)

    def test_no_body_paths_or_raw_patient_fields_in_journal(self):
        book=ledger(); complete_first(book,4); text=json.dumps(book.snapshot())
        for field in ("subject_id","patient_id","report_text","image_path","source_statement"):
            self.assertNotIn(field,text)

    def test_invalid_operation_kind_hash_seed_and_dependency_shapes(self):
        r=requests()[0]
        for fields in ({"kind":"train"},{"seed":True},{"ehr_anchor_sha256":"bad"},
                       {"input_image_sha256":IMAGE},{"parent_operation_id":"report_0"}):
            with self.assertRaises(ValueError): replace(r,**fields)


if __name__ == "__main__": unittest.main()

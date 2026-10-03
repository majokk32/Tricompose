"""Invented callbacks only; guards and fsync are tested without model calls."""
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import runtime_dispatch as dispatch
from tricompose_v12.execution_ledger import CallResult
from test_execution_ledger import ledger, requests, results, AUDIT
import smoke_bounded_execution as cli


def backend(callback=None):
    value=Mock(frozen=True,audit_sha256=AUDIT,execution_mode="invented_fixture_no_models")
    value.invoke.side_effect=callback
    value.invoke.return_value=results()[0]
    return value


class RuntimeDispatchTests(unittest.TestCase):
    def invoke(self, book, runtime, validator=lambda payload,request:payload):
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented_fixture"}), \
                patch.object(dispatch,"_new_private_handle",return_value=io.StringIO()):
            return dispatch.dispatch_operation(book,requests()[0],runtime,validator,private_log_path="fixture.log")

    def test_slurm_guard_precedes_private_logs_reservation_and_backend(self):
        book=ledger(); runtime=backend()
        with patch.dict(os.environ,{},clear=True),patch.object(dispatch,"_new_private_handle") as handle:
            with self.assertRaisesRegex(RuntimeError,"Slurm"):
                dispatch.dispatch_operation(book,requests()[0],runtime,lambda p,r:p,private_log_path="unused")
            handle.assert_not_called(); runtime.invoke.assert_not_called()
        self.assertEqual(book.charged_attempts,0)

    def test_backend_audit_freeze_and_mode_mismatch_prevents_reservation(self):
        for field,value in (("frozen",False),("audit_sha256","f"*64),("execution_mode","approved_slurm_backend")):
            runtime=backend(); setattr(runtime,field,value); book=ledger()
            with self.assertRaises(ValueError): self.invoke(book,runtime)
            runtime.invoke.assert_not_called(); self.assertEqual(book.charged_attempts,0)

    def test_backend_sees_already_journaled_reservation(self):
        recorded=[]; book=ledger(sink=recorded.append)
        def call(request):
            self.assertEqual(recorded[0]["event"],"attempt_reserved")
            self.assertEqual(book.charged_attempts,1)
            return results()[0]
        self.assertEqual(self.invoke(book,backend(call)),results()[0])
        self.assertEqual(book.snapshot()["completed_operations"],1)

    def test_sink_or_private_log_failure_prevents_model_callback(self):
        def broken(event): raise OSError("invented journal error")
        runtime=backend()
        with self.assertRaises(OSError): self.invoke(ledger(sink=broken),runtime)
        runtime.invoke.assert_not_called()
        book=ledger()
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented"}),patch.object(dispatch,"_new_private_handle",side_effect=OSError):
            with self.assertRaises(OSError):
                dispatch.dispatch_operation(book,requests()[0],runtime,lambda p,r:p,private_log_path="unused")
        self.assertEqual(book.charged_attempts,0)

    def test_timeout_returns_unresolved_and_charged_without_automatic_retry(self):
        runtime=backend(lambda request:(_ for _ in ()).throw(TimeoutError("invented secret exception text")))
        book=ledger(); self.assertIsNone(self.invoke(book,runtime))
        runtime.invoke.assert_called_once()
        self.assertEqual(book.charged_attempts,1)
        self.assertEqual(book.snapshot()["events"][-1]["error_code"],"timeout")
        self.assertNotIn("secret",json.dumps(book.snapshot()))

    def test_generic_runtime_error_not_clinical_contradiction(self):
        runtime=backend(lambda request:(_ for _ in ()).throw(RuntimeError("invented sensitive error")))
        book=ledger(); self.assertIsNone(self.invoke(book,runtime))
        event=book.snapshot()["events"][-1]
        self.assertEqual(event["error_code"],"runtime_exception")
        self.assertFalse(event["retryable"])
        self.assertFalse(event["failure_is_clinical_contradiction"])

    def test_malformed_result_or_validator_exception_is_charged_failure(self):
        for validator in (lambda payload,request:None, lambda payload,request:CallResult("c"*64,"f"*64),
                lambda payload,request:(_ for _ in ()).throw(ValueError("invented sensitive payload"))):
            book=ledger(); self.assertIsNone(self.invoke(book,backend(),validator))
            self.assertEqual(book.charged_attempts,1)
            self.assertEqual(book.snapshot()["events"][-1]["error_code"],"invalid_result")

    def test_stdout_stderr_stay_out_of_public_capture(self):
        def call(request):
            print("invented_private_fixture_output")
            print("invented_private_error",file=sys.stderr)
            return results()[0]
        public=io.StringIO()
        from contextlib import redirect_stdout,redirect_stderr
        with redirect_stdout(public),redirect_stderr(public): self.invoke(ledger(),backend(call))
        self.assertEqual(public.getvalue(),"")

    def test_completion_journal_error_keeps_pending_not_retried(self):
        seen=[]
        def broken(event):
            seen.append(event)
            if len(seen)>1: raise OSError("invented fsync failure")
        runtime=backend(); book=ledger(sink=broken)
        with self.assertRaises(OSError): self.invoke(book,runtime)
        runtime.invoke.assert_called_once()
        self.assertEqual(book.snapshot()["pending_attempts"],1)
        self.assertEqual(book.charged_attempts,1)

    def test_durable_journal_flushes_fsyncs_and_closes(self):
        handle=Mock(); handle.fileno.return_value=123
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented"}), \
                patch.object(dispatch,"_new_private_handle",return_value=handle),patch.object(dispatch.os,"fsync") as sync:
            with dispatch.ProtectedJournal("fixture.journal.jsonl") as journal:
                journal.append({"invented_event":1})
            handle.flush.assert_called_once(); sync.assert_called_once_with(123)
            handle.close.assert_called_once()
            self.assertEqual(json.loads(handle.write.call_args.args[0]),{"invented_event":1})

    def test_journal_creation_also_refuses_before_private_write_outside_slurm(self):
        with patch.dict(os.environ,{},clear=True),patch.object(dispatch,"_new_private_handle") as handle:
            with self.assertRaises(RuntimeError): dispatch.ProtectedJournal("unused")
            handle.assert_not_called()

    def test_fixture_cli_guard_precedes_config_and_output_paths(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"require_inside") as read:
            with self.assertRaisesRegex(RuntimeError,"Slurm"): cli.run(None)
            read.assert_not_called()

    def test_fixture_config_cannot_be_relabelled_prospective_generation(self):
        config=json.loads((ROOT/"configs/bounded_execution_fixture_v1.json").read_text())
        for name,value in (("execution_mode","approved_slurm_backend"),("inference_backend_installed",True),
                           ("fixture_results_are_generation_outputs",True)):
            altered=dict(config); altered[name]=value
            with self.assertRaises(ValueError): cli.validate_config(altered)


if __name__=="__main__": unittest.main()

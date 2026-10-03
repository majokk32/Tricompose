"""Mocked processes only. Never spawn a real worker or load a model."""
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import live_workers as workers
from tricompose_v12 import live_execution as execution
from tricompose_v12.execution_ledger import BoundedCallLedger,CallRequest,CallResult
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.runtime_dispatch import dispatch_operation

GPU={"SLURM_JOB_ID":"invented_fixture","SLURM_GPUS_ON_NODE":"1","CUDA_VISIBLE_DEVICES":"0"}


class WorkerTests(unittest.TestCase):
    def test_gpu_guard_before_any_subprocess_or_new_log(self):
        for env in ({},{"SLURM_JOB_ID":"invented_cpu"},
            {**GPU,"CUDA_VISIBLE_DEVICES":"-1"},{**GPU,"SLURM_GPUS_ON_NODE":"0"}):
            with patch.dict(os.environ,env,clear=True),patch.object(workers.subprocess,"Popen") as proc, \
                    patch.object(workers,"_new_private_handle") as handle,patch.object(workers,"worker_environment") as cache:
                with self.assertRaises(RuntimeError): workers.run_private_process(["not_executed"],Path("unused"),180)
                proc.assert_not_called(); handle.assert_not_called(); cache.assert_not_called()

    def test_controller_cpu_guard_precedes_plan_load_and_run_creation(self):
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented_cpu"},clear=True), \
                patch.object(execution,"load_plan") as load,patch.object(execution,"private_directory") as mkdir:
            with self.assertRaises(RuntimeError): execution.run(Mock())
            load.assert_not_called(); mkdir.assert_not_called()

    def test_registry_has_all_existing_three_image_four_report_experts(self):
        r=workers.registry()
        self.assertEqual(sum(v["kind"]=="cxr_generator" for v in r.values()),3)
        self.assertEqual(sum(v["kind"]=="report_generator" for v in r.values()),4)
        for value in r.values():
            if value["kind"] in {"cxr_generator","report_generator"}:
                self.assertFalse(value["native_structured_ehr_conditioning"])
        self.assertGreater(r["maira2"]["minimum_planning_vram_gib"],16)

    def test_argv_exact_single_case_batch_no_shell_training_or_sampling_override(self):
        r=workers.registry()
        for name,spec in r.items():
            args=workers.argv_for(spec,request_run="pack",cxr_run="image",report_run="report",
                output_root="output",thresholds="thresholds")
            self.assertEqual(args[0],spec["python"]); self.assertEqual(args[1],spec["script"])
            if "--batch-size" in args: self.assertEqual(args[args.index("--batch-size")+1],"1")
            self.assertNotIn("--train",args); self.assertNotIn("torchrun",args)
            if name=="roentgen_v2": self.assertEqual(args[args.index("--precision")+1],"float16")
            if name=="llavarad": self.assertIn("--runtime-dir",args)
            if name=="xrv": self.assertIn("--thresholds",args)

    def test_no_missing_input_string_passes_to_worker(self):
        with self.assertRaises(ValueError): workers.argv_for(workers.registry()["roentgen_v2"],output_root="output")

    def invoke_process(self,process):
        with patch.dict(os.environ,GPU,clear=True),patch.object(workers,"worker_environment",return_value={}), \
                patch.object(workers,"_new_private_handle",side_effect=[io.StringIO(),io.StringIO()]), \
                patch.object(workers.subprocess,"Popen",return_value=process) as create:
            workers.run_private_process(["invented_argv_not_executed"],Path("unused"),180)
        return create

    def test_private_native_capture_owned_session_and_no_shell(self):
        process=Mock(); process.wait.return_value=0
        create=self.invoke_process(process); kwargs=create.call_args.kwargs
        self.assertFalse(kwargs["shell"]); self.assertTrue(kwargs["start_new_session"])
        self.assertIsNotNone(kwargs["stdout"]); self.assertIsNotNone(kwargs["stderr"])
        process.wait.assert_called_once_with(timeout=180)

    def test_timeout_terminates_owned_group_before_returning_retryable(self):
        process=Mock(pid=123456); process.wait.side_effect=[subprocess.TimeoutExpired("invented",180),0]
        with patch.object(workers.os,"killpg") as kill:
            with self.assertRaises(TimeoutError): self.invoke_process(process)
        self.assertEqual(kill.call_args_list[0].args,(123456,signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args,(123456,signal.SIGKILL))
        self.assertEqual(process.wait.call_count,2)

    def test_nonzero_exit_is_operational_failure_not_clinical_error(self):
        process=Mock(); process.wait.return_value=1
        with self.assertRaises(RuntimeError): self.invoke_process(process)

    def test_backend_refuses_not_preflighted_descriptor(self):
        spec=workers.registry()["roentgen_v2"]
        with self.assertRaises(ValueError): workers.LocalFrozenBackend(spec,operation_root="unused",timeout=180)

    def test_cpu_preflight_does_not_instantiate_runtime_factory(self):
        spec=workers.registry()["roentgen_v2"]; factory=Mock()
        module=Mock(); module._model_components.return_value=("fixture_revision",{},factory)
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented_cpu"},clear=True), \
                patch.object(workers,"load_script",return_value=module),patch.object(Path,"exists",return_value=True), \
                patch.object(Path,"rglob",return_value=[]),patch.object(workers.os,"access",return_value=True):
            result=workers.preflight_generator(spec)
        factory.assert_not_called(); self.assertFalse(result["factory_instantiated"])
        self.assertEqual(result["status"],"preflighted")

    def test_offline_private_cache_environment_removes_api_credentials(self):
        env={"HF_TOKEN":"invented", "OPENAI_API_KEY":"invented", "AWS_ACCESS_KEY_ID":"invented"}
        runtime=workers.PROTECTED_ROOT/"invented_runtime_not_created"
        with patch.dict(os.environ,env,clear=True),patch.object(workers,"require_inside",return_value=runtime), \
                patch.object(workers,"private_directory"):
            actual=workers.worker_environment(runtime)
        for key in env: self.assertNotIn(key,actual)
        self.assertEqual(actual["HF_HUB_OFFLINE"],"1")
        self.assertEqual(actual["HF_HUB_DISABLE_IMPLICIT_TOKEN"],"1")
        self.assertTrue(actual["TMPDIR"].startswith(str(workers.PROTECTED_ROOT)))
        self.assertEqual(actual["PYTHONDONTWRITEBYTECODE"],"1")

    def test_real_backend_observes_reservation_before_mocked_process_spawn(self):
        spec={**workers.registry()["roentgen_v2"],"status":"preflighted","asset_pins":{}}
        audit=_digest(spec); anchor="a"*64; events=[]
        ledger=BoundedCallLedger(case_id="fixture_live",ehr_anchor_sha256=anchor,call_budget=1,max_retries=0,
            execution_mode="approved_slurm_backend",sink=events.append)
        request=CallRequest("cxr_0","fixture_live",anchor,"cxr_generator","roentgen_v2",audit,0)
        backend=workers.LocalFrozenBackend(spec,operation_root=Path("unused"),timeout=180,request_run="pack")
        def spawn(*args): self.assertEqual(events[-1]["event"],"attempt_reserved")
        with patch.dict(os.environ,GPU,clear=True),patch.object(workers,"check_pins"), \
                patch.object(workers,"private_directory"),patch.object(workers,"run_private_process",side_effect=spawn) as run, \
                patch("tricompose_v12.runtime_dispatch._new_private_handle",return_value=io.StringIO()):
            result=dispatch_operation(ledger,request,backend,lambda p,r:CallResult("c"*64),private_log_path="unused")
        run.assert_called_once(); self.assertEqual(result.output_artifact_sha256,"c"*64)
        self.assertEqual(ledger.charged_attempts,1)


if __name__=="__main__": unittest.main()

"""Synthetic metadata and source-contract tests; no models or bodies."""
import argparse
import ast
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))
from complete_bounded_regeneration_endpoint import prepare, run


class EndpointRecoveryTests(unittest.TestCase):
    def test_gpu_guard_before_plan_reads_or_model_loading(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "invented_cpu", "CUDA_VISIBLE_DEVICES": ""}, clear=True), \
             patch("complete_bounded_regeneration_endpoint.read_json") as reader:
            with self.assertRaises(RuntimeError): run(argparse.Namespace())
            reader.assert_not_called()

    def test_cpu_prepare_guard_before_protected_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch("complete_bounded_regeneration_endpoint.sha256_file") as hasher:
            with self.assertRaises(RuntimeError): prepare(argparse.Namespace())
            hasher.assert_not_called()

    def test_existing_vendor_bootstrap_before_frozen_endpoint_call(self):
        source = (ROOT / "audits/complete_bounded_regeneration_endpoint.py").read_text()
        self.assertLess(source.index('sys.path.insert(0, plan["vendor_path"])'), source.index('scores = score('))
        self.assertIn('"runtime/vendor/hi_ml_multimodal_0_2_2"', source)
        self.assertIn('require_gpu_slurm()  # BEFORE inputs', source)

    def test_no_generator_or_choice_policy_update(self):
        source = (ROOT / "audits/complete_bounded_regeneration_endpoint.py").read_text()
        calls = {n.func.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn("one_case", calls)
        self.assertNotIn("decision", calls)
        self.assertNotIn("candidate_row", calls)
        self.assertNotIn("read_report_text", calls)
        self.assertNotIn("import torch", source)
        self.assertIn('"new_generation_calls": 0', source)
        self.assertIn('"selection_or_threshold_update_allowed": False', source)

    def test_failure_is_retained_and_not_automatically_retried(self):
        source = (ROOT / "audits/complete_bounded_regeneration_endpoint.py").read_text()
        self.assertIn('"failed_endpoint_attempt_retained"', source)
        self.assertIn('"original_failed_endpoint_attempt_is_not_free": True', source)
        self.assertIn('"automatic_resume": False', source)

    def test_source_result_and_selection_not_overwritten(self):
        source = (ROOT / "audits/complete_bounded_regeneration_endpoint.py").read_text()
        self.assertNotIn("write_private_json(SOURCE", source)
        self.assertNotIn("write_private_text(SOURCE", source)
        self.assertIn('validate_endpoint(rows, scores)', source)
        self.assertIn('tables(rows, selection["choices"], scores)', source)
        self.assertIn('"endpoint_used_for_selection": False', source)


if __name__ == "__main__":
    unittest.main()

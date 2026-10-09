"""Tiny invented model-file fixtures; no real checkpoint/model/body reads."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import run_fresh_report_agent_v2 as bridge


class SupplementalAssetTests(unittest.TestCase):
    def tiny_root(self, root, name):
        p = root / name; p.mkdir()
        for filename in ("config.json", "pytorch_model.bin", "model.safetensors", "builder.py"):
            (p / filename).write_text("invented_fixture_not_model_or_data")
        return p

    def test_explicit_base_vision_text_weights_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.tiny_root(root, "base"); vision = self.tiny_root(root, "vision")
            text = self.tiny_root(root, "text")
            workers = {"fixture_report": {"extra_args": ["--model-base", str(base),
                "--vision-dir", str(vision), "--biomedbert-dir", str(text)]}}
            inventory = bridge.supplemental_inventory(workers)["fixture_report"]
            self.assertEqual(len(inventory), 12)
            self.assertIn(str(base / "pytorch_model.bin"), inventory)
            self.assertIn(str(vision / "model.safetensors"), inventory)

    def test_external_source_only_includes_python_not_json_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.tiny_root(Path(directory), "source")
            (root / "token").write_text("invented_not_a_secret")
            inventory = bridge.supplemental_inventory({"fixture_report": {
                "extra_args": ["--external-root", str(root)]}})["fixture_report"]
            self.assertEqual(inventory, [str(root / "builder.py")])

    def test_chexbert_bert_weights_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.tiny_root(Path(directory), "bert")
            with patch.object(bridge, "CHEXBERT_BERT", root):
                inventory = bridge.supplemental_inventory({"chexbert": {"extra_args": []}})
            self.assertIn(str(root / "pytorch_model.bin"), inventory["chexbert"])

    def test_empty_and_missing_secondary_directories_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for p in (root, root / "missing"):
                with self.assertRaises(ValueError):
                    bridge.supplemental_inventory({"fixture": {"extra_args": ["--vision-dir", str(p)]}})

    def test_missing_secondary_weight_pin_is_rejected(self):
        plan = {"supplemental_asset_version": bridge.VERSION,
            "preflight_source_manifest_sha256": bridge.SOURCE_SHA,
            "supplemental_asset_inventory": {"fixture": ["fixture_weight.bin"]},
            "workers": {"fixture": {"asset_pins": {}}}}
        with patch.object(bridge.bridge, "validate_plan"), patch.object(bridge, "supplemental_inventory",
                return_value=plan["supplemental_asset_inventory"]):
            with self.assertRaises(ValueError): bridge.validate(plan)

    def test_present_secondary_pin_is_accepted_with_upstream_authentication(self):
        plan = {"supplemental_asset_version": bridge.VERSION,
            "preflight_source_manifest_sha256": bridge.SOURCE_SHA,
            "supplemental_asset_inventory": {"fixture": ["fixture_weight.bin"]},
            "workers": {"fixture": {"asset_pins": {"fixture_weight.bin": {"sha256": "0" * 64}}}}}
        with patch.object(bridge.bridge, "validate_plan") as upstream, patch.object(bridge, "supplemental_inventory",
                return_value=plan["supplemental_asset_inventory"]):
            bridge.validate(plan)
            upstream.assert_called_once_with(plan)

    def test_gpu_guard_precedes_plan_reads_and_execution(self):
        with patch.object(bridge.bridge, "gpu_guard", side_effect=RuntimeError("fixture_no_gpu")), \
             patch.object(bridge, "read_json") as read, patch.object(bridge.bridge, "run") as run:
            with self.assertRaises(RuntimeError): bridge.run(argparse.Namespace())
            read.assert_not_called(); run.assert_not_called()

    def test_main_reports_only_fixed_failure_code(self):
        argv = ["fixture", "run", "--plan-run", "fixture", "--plan-manifest-sha256", "0" * 64,
                "--output-root", "fixture", "--run-id", "fixture"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(bridge, "run", side_effect=RuntimeError("invented_private_body")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(bridge.main(), 1)
        self.assertNotIn("invented_private_body", output.getvalue())
        self.assertIn('"status": "failed"', output.getvalue())


if __name__ == "__main__":
    unittest.main()

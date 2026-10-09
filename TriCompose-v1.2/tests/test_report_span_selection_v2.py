"""Invented/mock fixtures. Prompt structure tests are not model qualification."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('span_v2_driver_fixture', ROOT/'tools/verify_report_span_selection_v2.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
module = driver.interface


class SpanV2Tests(unittest.TestCase):
    def test_finding_object_and_polarity_array_levels_explicit(self):
        self.assertIn('VALUE OF EACH FINDING KEY must be a JSON OBJECT, NOT an array', module.PROMPT)
        self.assertIn('VALUES OF THOSE POLARITY KEYS are arrays', module.PROMPT)
        self.assertIn('Keep all twelve polarity keys', module.PROMPT)
        self.assertNotIn('Each value is an array', module.PROMPT)

    def test_template_has_exactly_four_objects_twelve_arrays(self):
        self.assertEqual(set(module.JSON_TEMPLATE), set(module.FINDINGS))
        for value in module.JSON_TEMPLATE.values():
            self.assertIsInstance(value, dict)
            self.assertEqual(set(value), set(module.POLARITIES))
            self.assertTrue(all(array == [] for array in value.values()))

    def test_full_shape_and_original_text_present_in_actual_request(self):
        text = 'No effusion.\r\nNo pneumothorax.'
        inventory = module.build_inventory(text)
        messages = module.request_messages(text, inventory)
        request = messages[0]['content'][0]['text']
        self.assertIn(text, request)
        self.assertIn(json.dumps(module.JSON_TEMPLATE, separators=(',', ':')), request)
        self.assertNotIn('{required_json_shape}', request)
        self.assertEqual([c['type'] for c in messages[0]['content']], ['text'])

    def test_all_nonformat_instructions_identical(self):
        self.assertEqual(module.PROMPT.replace(module._V2_FORMAT, module._V1_FORMAT), module.v1.PROMPT)

    def test_same_inventory_offsets_hashes_and_decoder(self):
        text = 'No effusion. No effusion.\nHeart is enlarged.'
        self.assertEqual(module.build_inventory(text), module.v1.build_inventory(text))
        self.assertIs(module.decode_response, module.v1.decode_response)
        self.assertEqual(module.validate_inventory(text, module.build_inventory(text)), module.v1.validate_inventory(text, module.v1.build_inventory(text)))

    def test_flat_finding_arrays_still_fail_no_polarity_invention(self):
        text = 'No effusion.'
        flat = {f: [] for f in module.FINDINGS}; flat['pleural_effusion'] = ['span_0000']
        result = module.decode_response(json.dumps(flat), text, module.build_inventory(text))
        self.assertEqual(result['contract_status'], 'failed_unavailable')
        self.assertEqual(result['contract_failure_reason'], 'polarity_inventory_mismatch')
        self.assertTrue(all(f['state'] == 'unknown' for f in result['findings'].values()))

    def test_valid_nested_structure_matches_v1_decode(self):
        text = 'No effusion.'
        template = json.loads(json.dumps(module.JSON_TEMPLATE))
        template['pleural_effusion']['negative'] = ['span_0000']
        response = json.dumps(template)
        self.assertEqual(module.decode_response(response, text, module.build_inventory(text)), module.v1.decode_response(response, text, module.v1.build_inventory(text)))

    def test_empty_template_is_unknown_not_binary_agreement(self):
        text = 'Generic invented text.'
        result = module.decode_response(json.dumps(module.JSON_TEMPLATE), text, module.build_inventory(text))
        self.assertEqual(result['contract_status'], 'complete')
        self.assertTrue(all(f['state'] == 'unknown' for f in result['findings'].values()))

    def test_driver_records_v1_reuse_and_v2_sources(self):
        paths = driver.program_sources()
        self.assertEqual(paths['span_worker'], driver.WORKER_V1)
        self.assertEqual(paths['span_interface'], driver.INTERFACE_V2)
        self.assertEqual(paths['span_v1_inventory_decoder'], driver.INTERFACE_V1)
        self.assertNotEqual(driver.worker.SCHEMA, 'tricompose-frozen-report-span-selection-v1')

    def test_driver_cpu_guard_before_input_or_model(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(driver.worker, 'require_inside') as resolve:
            with self.assertRaises(RuntimeError):
                driver.worker.execute(SimpleNamespace())
            resolve.assert_not_called()

    def test_driver_overwrite_guard_before_gpu(self):
        target = MagicMock(); target.exists.return_value = True
        args = SimpleNamespace(run_id='fixture_existing', output_root='unused', mode='run')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(driver.worker, 'require_inside', return_value=target), patch.object(driver.worker, 'verify') as verify:
            with self.assertRaises(FileExistsError):
                driver.worker.execute(args)
            verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()

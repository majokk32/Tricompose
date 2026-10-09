"""Worker guards/request staging with invented data and mocked CUDA only."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report_span_worker_fixture', ROOT/'tools/verify_report_span_selection.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture():
    texts = {cli.interface.digest(f'Invented source {index}.'): f'Invented source {index}.' for index in range(22)}
    hashes = sorted(texts)
    inputs = [{'report_sha256': h} for h in hashes]+[{'report_sha256': h} for h in hashes[:2]]
    return inputs, texts


class SpanWorkerTests(unittest.TestCase):
    def test_same_24_slots_deduplicated_to_22_exact_requests(self):
        inputs, texts = fixture()
        before = copy.deepcopy((inputs, texts))
        requests = cli.build_requests(inputs, texts, {})
        self.assertEqual((inputs, texts), before)
        self.assertEqual(len(requests), 22)
        self.assertTrue(all(r['input_status'] == 'ready' for r in requests))
        self.assertEqual([r['report_sha256'] for r in requests], sorted(texts))

    def test_request_hash_is_actual_user_content_before_chat_template(self):
        inputs, texts = fixture()
        for request in cli.build_requests(inputs, texts, {}):
            self.assertEqual(request['input_text_sha256'], cli.interface.digest(request['messages'][0]['content'][0]['text']))
            self.assertEqual(request['inventory']['report_sha256'], request['report_sha256'])
            self.assertEqual(request['source_text'], texts[request['report_sha256']])
            self.assertEqual([c['type'] for c in request['messages'][0]['content']], ['text'])

    def test_repeat_staging_is_deterministic(self):
        inputs, texts = fixture()
        self.assertEqual(cli.build_requests(inputs, texts, {}), cli.build_requests(inputs, texts, {}))

    def test_missing_extra_or_incorrect_slot_denominator_refused(self):
        inputs, texts = fixture()
        fewer = dict(texts); fewer.pop(next(iter(fewer)))
        for source, body, unavailable in ((inputs[:-1], texts, {}), (inputs, fewer, {}),
                (inputs, texts, {next(iter(texts)): 'invalid_utf8'})):
            with self.assertRaises(ValueError):
                cli.build_requests(source, body, unavailable)

    def test_changed_text_hash_refused(self):
        inputs, texts = fixture()
        texts[next(iter(texts))] += ' modified'
        with self.assertRaisesRegex(ValueError, 'exact_source'):
            cli.build_requests(inputs, texts, {})

    def test_unavailable_source_retains_hash_without_model_message(self):
        inputs, texts = fixture()
        h = next(iter(texts)); del texts[h]
        requests = cli.build_requests(inputs, texts, {h: 'invalid_utf8'})
        unavailable = next(r for r in requests if r['report_sha256'] == h)
        self.assertEqual(len(requests), 22)
        self.assertEqual(unavailable['input_status'], 'unavailable')
        self.assertIsNone(unavailable['messages'])

    def test_span_bound_failure_not_truncated_or_case_replaced(self):
        inputs, texts = fixture()
        old_h = inputs[0]['report_sha256']
        text = 'x'*2049; h = cli.interface.digest(text)
        texts.pop(old_h); texts[h] = text
        for item in inputs:
            if item['report_sha256'] == old_h:
                item['report_sha256'] = h
        requests = cli.build_requests(inputs, texts, {})
        refused = next(r for r in requests if r['report_sha256'] == h)
        self.assertEqual(refused['input_failure_reason'], 'span_character_limit_exceeded')
        self.assertEqual(refused['source_text'], text)
        self.assertIsNone(refused['messages'])

    def test_empty_protocol_rejected_before_paths(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, 'require_inside') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'slurm_required'):
                cli.execute(SimpleNamespace())
            resolve.assert_not_called()

    def test_existing_output_rejected_before_preparation_or_inference(self):
        target = MagicMock(); target.exists.return_value = True
        args = SimpleNamespace(mode='run', output_root='unused', run_id='fixture_existing')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(cli, 'require_inside', return_value=target), patch.object(cli, 'verify') as verify:
            with self.assertRaises(FileExistsError):
                cli.execute(args)
            verify.assert_not_called()

    def test_nonopaque_run_id_rejected_before_paths(self):
        args = SimpleNamespace(run_id='../bad')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(cli, 'require_inside') as resolve:
            with self.assertRaisesRegex(ValueError, 'opaque_run'):
                cli.execute(args)
            resolve.assert_not_called()

    def test_no_cuda_cannot_load_or_run_model(self):
        torch = MagicMock(); torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {'torch': torch}), patch.object(cli, 'load_staged', return_value=({}, [], {})), patch('tricompose.verifiers.qwenvl_cxr_report._load_model') as loader, patch.object(cli, 'infer') as infer:
            with self.assertRaisesRegex(RuntimeError, '24_gib'):
                cli.verify(SimpleNamespace(plan_run='invented'))
            loader.assert_not_called(); infer.assert_not_called()

    def test_low_vram_cannot_load_or_run_model(self):
        torch = MagicMock(); torch.cuda.is_available.return_value = True
        torch.cuda.get_device_properties.return_value.total_memory = 16*1024**3
        with patch.dict(sys.modules, {'torch': torch}), patch.object(cli, 'load_staged', return_value=({}, [], {})), patch('tricompose.verifiers.qwenvl_cxr_report._load_model') as loader, patch.object(cli, 'infer') as infer:
            with self.assertRaisesRegex(RuntimeError, '24_gib'):
                cli.verify(SimpleNamespace(plan_run='invented'))
            loader.assert_not_called(); infer.assert_not_called()


if __name__ == '__main__':
    unittest.main()

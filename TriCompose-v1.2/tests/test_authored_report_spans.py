"""Invented texts/metadata and mocks only; no dataset/checkpoint/model reads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('authored_span_test_worker', ROOT/'tools/verify_authored_report_spans.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(state='positive', status='complete'):
    reference = {'item_id': 'invented_000', 'report_sha256': 'a'*64,
        'expected_states': dict.fromkeys(m.interface.FINDINGS, 'unknown'),
        'evaluation_findings': ['cardiomegaly'], 'family': 'invented_family'}
    reference['expected_states']['cardiomegaly'] = 'positive'
    record = {'item_id': reference['item_id'], 'report_sha256': reference['report_sha256'],
        'status': status, 'finding_states': None, 'contract_failure_reason': None}
    if status == 'complete':
        record['finding_states'] = {**dict.fromkeys(m.interface.FINDINGS, 'unknown'), 'cardiomegaly': state}
    return [reference], [record]


class AuthoredReportSpanTests(unittest.TestCase):
    def test_messages_have_original_text_but_no_reference_metadata(self):
        text = 'Invented test: no pneumothorax.'
        messages = m.interface.request_messages(text, m.interface.build_inventory(text))
        self.assertIn(text, json.dumps(messages))
        for name in ('expected_states', 'evaluation_findings', 'report_sha256', 'family'):
            self.assertNotIn(name, json.dumps(messages))

    def test_decoder_exports_hash_offsets_not_quotes_or_raw_response(self):
        text = 'No pneumothorax.'
        response = copy.deepcopy(m.interface.JSON_TEMPLATE)
        response['pneumothorax']['negative'] = ['span_0000']
        decoded = m.sanitized_decode(json.dumps(response), text)
        self.assertEqual(decoded['finding_states']['pneumothorax'], 'negative')
        self.assertIn('quote_sha256', json.dumps(decoded))
        self.assertNotIn('"quote":', json.dumps(decoded))
        self.assertNotIn(text, json.dumps(decoded))
        self.assertFalse(decoded['semantic_correctness_independently_verified'])

    def test_bad_shape_empty_response_and_token_cap_are_unavailable(self):
        text = 'Invented example.'
        for response in ('', '{}', '{"cardiomegaly":[]}'):
            decoded = m.sanitized_decode(response, text)
            self.assertEqual(decoded['status'], 'failed_unavailable')
            self.assertIsNone(decoded['finding_states'])
            self.assertIsNone(decoded['source_span_references'])
        decoded = m.sanitized_decode(json.dumps(m.interface.JSON_TEMPLATE), text, token_limit_reached=True)
        self.assertEqual(decoded['status'], 'failed_unavailable')

    def test_nonexistent_span_and_same_id_for_opposed_polarities_rejected(self):
        response = copy.deepcopy(m.interface.JSON_TEMPLATE)
        response['cardiomegaly']['positive'] = ['span_9999']
        self.assertEqual(m.sanitized_decode(json.dumps(response), 'Cardiomegaly.')['status'], 'failed_unavailable')
        response['cardiomegaly']['positive'] = response['cardiomegaly']['negative'] = ['span_0000']
        self.assertEqual(m.sanitized_decode(json.dumps(response), 'Cardiomegaly.')['status'], 'failed_unavailable')

    def test_failed_prediction_not_normalized_into_successful_unknown(self):
        references, records = fixture(status='failed_unavailable')
        before = copy.deepcopy(records)
        metrics, _ = m.metrics_for_records(references, records)
        self.assertEqual(records, before)
        self.assertEqual(metrics['designated_targets']['checks'], 1)
        self.assertEqual(metrics['designated_targets']['unavailable'], 1)
        self.assertEqual(metrics['designated_targets']['accuracy'], 0)

    def test_failed_unknown_vector_persisted_as_if_success_rejected(self):
        references, records = fixture(status='failed_unavailable')
        records[0]['finding_states'] = dict.fromkeys(m.interface.FINDINGS, 'unknown')
        with self.assertRaisesRegex(ValueError, 'unavailable_is_not_unknown_prediction'):
            m.metrics_for_records(references, records)

    def test_omission_flip_and_uncertainty_kept_separate(self):
        for state, flips in (('unknown', 0), ('uncertain', 0), ('negative', 1)):
            metrics, _ = m.metrics_for_records(*fixture(state=state))
            self.assertEqual(metrics['designated_targets']['hard_positive_negative_flips'], flips)
            self.assertEqual(metrics['designated_targets']['accuracy'], 0)

    def test_missing_duplicate_or_wrong_hash_predictions_rejected(self):
        references, records = fixture()
        for bad in ([], records+records, [{**records[0], 'report_sha256': 'b'*64}]):
            with self.assertRaises(ValueError):
                m.metrics_for_records(references, bad)

    def test_uncertain_reference_is_not_negative(self):
        references, records = fixture(state='negative')
        references[0]['expected_states']['cardiomegaly'] = 'uncertain'
        metrics, _ = m.metrics_for_records(references, records)
        self.assertEqual(metrics['designated_targets']['unsafe_commit_on_unknown_or_uncertain'], 1)
        self.assertEqual(metrics['designated_targets']['hard_positive_negative_flips'], 0)

    def test_completed_all_unknown_is_available_but_not_annotated_recovery(self):
        references, records = fixture(state='unknown')
        metrics, _ = m.metrics_for_records(references, records)
        self.assertEqual(metrics['designated_targets']['unavailable'], 0)
        self.assertEqual(metrics['designated_targets']['correct'], 0)

    def test_run_guard_before_any_plan_model_or_reference_access(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(m, 'load_plan') as access:
            with self.assertRaises(RuntimeError):
                m.run('/invented', Path('/invented'))
            access.assert_not_called()

    def test_spoofed_slurm_env_without_cgroup_is_refused(self):
        with patch.dict('os.environ', {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(Path, 'read_text', return_value='invented_non_slurm_cgroup'):
            with self.assertRaisesRegex(RuntimeError, 'actual_slurm_cgroup_required'):
                m.require_gpu_approval(True)

    def test_cuda_guard_before_loading_plan_or_blinded_text(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(m, 'require_gpu_approval'), patch.dict('sys.modules', {'torch': torch}), \
                patch.object(m, 'load_plan') as access:
            with self.assertRaisesRegex(RuntimeError, 'gpu_with_at_least_24_gib_required'):
                m.run('/invented', Path('/invented'), approved=True)
            access.assert_not_called()

    def test_existing_run_refused_before_preparation_or_text_read(self):
        args = SimpleNamespace(mode='prepare', output_root='/invented', run_id='invented_001')
        with patch.object(m, 'new_atomic_run', side_effect=FileExistsError), patch.object(m, 'prepare') as prepare:
            with self.assertRaises(FileExistsError):
                m.execute(args)
            prepare.assert_not_called()

    def test_model_run_requires_explicit_flag_before_output_creation(self):
        args = SimpleNamespace(mode='run', allow_authored_language_benchmark=False)
        with patch.object(m, 'new_atomic_run') as output:
            with self.assertRaises(RuntimeError):
                m.execute(args)
            output.assert_not_called()


if __name__ == '__main__':
    unittest.main()

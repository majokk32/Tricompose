"""Invented reports/labels only; no real-source/model access in these tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('official_span_test_worker', ROOT/'tools/verify_official_report_spans.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(truth='positive', prediction='positive', status='complete', ready=True):
    item = {'item_id': 'item_0000', 'status': 'ready_metadata',
        'reference': {m.gold_method.SOURCE_NAMES[f]: 'unknown' for f in m.FINDINGS}}
    item['reference']['Cardiomegaly'] = truth
    record = {'item_id': 'item_0000', 'status': status, 'request_ready': ready,
        'selected_input_sha256': 'a'*64,
        'finding_states': dict.fromkeys(m.FINDINGS, 'unknown') if status == 'complete' else None}
    if status == 'complete':
        record['finding_states']['cardiomegaly'] = prediction
    baseline = {'item_id': 'item_0000', 'status': 'complete', 'selected_input_sha256': 'a'*64,
        'finding_states': {**dict.fromkeys(m.gold_method.HEADS, 'unknown'), 'cardiomegaly': truth}}
    return [item], [record], [baseline]


class OfficialSpanDiagnosticTests(unittest.TestCase):
    def test_sanitized_reference_exports_offsets_hashes_not_quotes_or_responses(self):
        text = 'Invented example: no pleural effusion.'
        payload = copy.deepcopy(m.interface.JSON_TEMPLATE)
        payload['pleural_effusion']['negative'] = ['span_0000']
        decoded = m.sanitized_decode(json.dumps(payload), text, m.interface.build_inventory(text))
        self.assertEqual(decoded['finding_states']['pleural_effusion'], 'negative')
        public = json.dumps(decoded)
        self.assertNotIn(text, public)
        self.assertNotIn('"quote"', public)
        self.assertNotIn('"response"', public)
        self.assertIn('span_sha256', public)
        self.assertFalse(decoded['semantic_correctness_independently_verified'])

    def test_bad_shape_or_raw_text_fails_closed_without_echoing_input(self):
        text = 'Invented secret string.'
        for response in (text, '{"cardiomegaly":["span_0000"]}', '{}'):
            decoded = m.sanitized_decode(response, text, m.interface.build_inventory(text))
            self.assertEqual(decoded['status'], 'contract_failed_unavailable')
            self.assertIsNone(decoded['finding_states'])
            self.assertIsNone(decoded['source_span_references'])
            self.assertNotIn(text, json.dumps(decoded))

    def test_token_cap_does_not_accept_syntactically_complete_response(self):
        text = 'Invented example.'
        decoded = m.sanitized_decode(json.dumps(m.interface.JSON_TEMPLATE), text,
            m.interface.build_inventory(text), token_limit_reached=True)
        self.assertEqual(decoded['contract_failure_reason'], 'token_limit_reached')

    def test_unknown_prediction_is_omission_not_negative_or_agreement(self):
        summary = m.summarize(*fixture(prediction='unknown'))
        finding = summary['per_finding']['cardiomegaly']
        self.assertEqual(finding['omitted_annotated_assertions'], 1)
        self.assertEqual(finding['hard_polarity_flips'], 0)
        self.assertEqual(finding['annotated_state_accuracy'], 0)
        self.assertEqual(summary['failure_aware_annotated_recovery'], 0)

    def test_known_positive_negative_flip_is_not_missing(self):
        summary = m.summarize(*fixture(prediction='negative'))
        self.assertEqual(summary['per_finding']['cardiomegaly']['hard_polarity_flips'], 1)
        self.assertEqual(summary['per_finding']['cardiomegaly']['omitted_annotated_assertions'], 0)

    def test_uncertain_reference_not_converted_to_negative(self):
        summary = m.summarize(*fixture(truth='uncertain', prediction='positive'))
        finding = summary['per_finding']['cardiomegaly']
        self.assertEqual(finding['determinate_promotions_on_uncertain_unknown'], 1)
        self.assertEqual(finding['hard_polarity_flips'], 0)
        self.assertEqual(finding['confusion_matrix']['uncertain']['positive'], 1)

    def test_all_unknown_reference_cannot_inflate_annotated_recovery(self):
        summary = m.summarize(*fixture(truth='unknown', prediction='unknown'))
        self.assertEqual(summary['per_finding']['cardiomegaly']['exact_state_accuracy'], 1)
        self.assertIsNone(summary['per_finding']['cardiomegaly']['annotated_state_accuracy'])
        self.assertIsNone(summary['failure_aware_annotated_recovery'])
        self.assertEqual(summary['complete_all_unknown_reports'], 1)

    def test_contract_failed_response_counts_in_failure_aware_denominator(self):
        summary = m.summarize(*fixture(status='contract_failed_unavailable'))
        self.assertEqual(summary['request_ready_reports'], 1)
        self.assertEqual(summary['completed_reports'], 0)
        self.assertEqual(summary['failed_ready_requests'], 1)
        self.assertIsNone(summary['per_finding']['cardiomegaly']['exact_state_accuracy'])
        self.assertEqual(summary['failure_aware_annotated_recovery'], 0)
        self.assertEqual(summary['paired_complete_reports'], 0)

    def test_non_test_unavailable_item_remains_in_inventory_without_fake_prediction(self):
        items, records, baseline = fixture()
        items.append({'item_id': 'item_0001', 'status': 'linked_non_test',
            'reference': dict.fromkeys(m.gold_method.SOURCE_NAMES.values(), 'unknown')})
        missing = {'item_id': 'item_0001', 'status': 'linked_non_test', 'request_ready': False,
            'finding_states': None, 'selected_input_sha256': None}
        records.append(missing)
        baseline.append(copy.deepcopy(missing))
        summary = m.summarize(items, records, baseline)
        self.assertEqual(summary['annotation_inventory'], 2)
        self.assertEqual(summary['completed_reports'], 1)
        self.assertEqual(summary['metadata_status_counts']['linked_non_test'], 1)

    def test_different_impression_hashes_cannot_be_paired(self):
        items, records, baseline = fixture()
        baseline[0]['selected_input_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'same_impression_bytes_required'):
            m.summarize(items, records, baseline)

    def test_missing_duplicate_and_nonready_completed_rows_rejected(self):
        items, records, baseline = fixture()
        for bad in ([], records+records):
            with self.assertRaises(ValueError):
                m.summarize(items, bad, baseline)
        records[0]['request_ready'] = False
        with self.assertRaises(ValueError):
            m.summarize(items, records, baseline)

    def test_unavailable_may_not_be_serialized_as_four_unknown_predictions(self):
        items, records, baseline = fixture()
        records[0]['status'] = 'report_file_missing'
        with self.assertRaisesRegex(ValueError, 'unavailable_is_not_unknown_prediction'):
            m.summarize(items, records, baseline)

    def test_summary_and_markdown_contain_no_reference_rows_or_ids(self):
        items, records, baseline = fixture()
        items[0]['patient_key'] = 'invented_private_patient_key'
        items[0]['report_path'] = 'invented_private_report_path'
        summary = m.summarize(items, records, baseline)
        output = json.dumps(summary)+m.markdown(summary)
        for value in ('invented_private_patient_key', 'invented_private_report_path', 'item_0000', '"reference"'):
            self.assertNotIn(value, output)
        self.assertFalse(summary['primary_metric_eligible'])
        self.assertFalse(summary['selection_changed'])
        self.assertFalse(summary['regeneration_authorized'])

    def test_no_real_reads_without_explicit_approval_and_slurm(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(m, 'load_plan') as access:
            with self.assertRaises(RuntimeError):
                m.run('invented_plan')
            access.assert_not_called()

    def test_cuda_guard_before_real_input_read(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(m.gold_runtime, 'require_approved_slurm'), patch.object(m, 'load_plan', return_value=({}, {})), \
                patch.dict(sys.modules, {'torch': fake_torch}), patch.object(m, 'sha256_file') as read:
            with self.assertRaisesRegex(RuntimeError, 'gpu_with_at_least_24_gib_required'):
                m.run('invented_plan', allow_real_report_benchmark=True)
            read.assert_not_called()

    def test_existing_output_refused_before_preparation_or_real_input(self):
        args = SimpleNamespace(mode='prepare', allow_real_report_benchmark=False,
            output_root='/invented', run_id='invented_001')
        with patch.object(m.gold_runtime, 'require_approved_slurm'), \
                patch.object(m, 'require_inside', return_value=Path('/invented/existing')), \
                patch.object(Path, 'exists', return_value=True), patch.object(m, 'prepare') as prepare:
            with self.assertRaises(FileExistsError):
                m.execute(args)
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()

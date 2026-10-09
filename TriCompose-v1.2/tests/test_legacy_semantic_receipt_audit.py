"""Invented text only; cache audit never invokes a model."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('legacy_receipt_audit_fixture', ROOT/'tools/audit_legacy_semantic_receipts.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
from verify_report_evidence_qwen import FINDINGS


def fixture(text='No pleural effusion.', quote=None):
    payload = {name: {'positive': [], 'negative': [], 'uncertain': []} for name in FINDINGS}
    payload['pleural_effusion']['negative'] = [text if quote is None else quote]
    response = json.dumps(payload)
    raw = {'report_sha256': cli.digest_text(text), 'response': response,
        'input_tokens': 100, 'output_tokens': 80, 'token_limit_reached': False}
    record = {'report_sha256': raw['report_sha256'], 'response_sha256': cli.digest_text(response),
        **{k: raw[k] for k in ('input_tokens', 'output_tokens', 'token_limit_reached')},
        **cli.decode_evidence(response, text)}
    return record, raw, text


class ReceiptAuditTests(unittest.TestCase):
    def test_exact_receipt_stays_complete(self):
        result = cli.audit_response(*fixture())
        self.assertEqual(result['strict_contract_status'], 'complete')
        self.assertEqual(result['whitespace_contract_status'], 'complete')
        self.assertFalse(result['clinical_validation'])
        self.assertFalse(result['selection_changed'])

    def test_only_whitespace_is_recoverable_without_changing_strict_result(self):
        result = cli.audit_response(*fixture('No\tpleural\n effusion.', 'No pleural effusion.'))
        self.assertEqual(result['strict_contract_status'], 'failed_unavailable')
        self.assertEqual(result['whitespace_contract_status'], 'complete')
        self.assertEqual(result['aligned_source_offsets_and_hashes'][0]['alignment_mode'], 'whitespace_equivalent')
        self.assertEqual(result['strict_states']['pleural_effusion'], 'unknown')

    def test_changed_clinical_word_is_not_recovered(self):
        result = cli.audit_response(*fixture('No pleural effusion.', 'Small pleural effusion.'))
        self.assertEqual(result['whitespace_contract_status'], 'failed_unavailable')
        self.assertEqual(result['whitespace_states']['pleural_effusion'], 'unknown')

    def test_ambiguous_quote_not_disambiguated(self):
        result = cli.audit_response(*fixture('No pleural effusion. No pleural effusion.', 'No pleural effusion.'))
        self.assertEqual(result['whitespace_failure_reason'], 'quote_location_ambiguous')

    def test_modified_response_or_source_refused(self):
        record, raw, text = fixture()
        raw['response'] += ' changed'
        with self.assertRaises(ValueError):
            cli.audit_response(record, raw, text)
        record, raw, text = fixture()
        with self.assertRaises(ValueError):
            cli.audit_response(record, raw, text+' changed')

    def test_forged_original_label_refused(self):
        record, raw, text = fixture()
        record['findings']['pleural_effusion']['state'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'reparse'):
            cli.audit_response(record, raw, text)

    def test_output_contains_offsets_not_quotes(self):
        result = cli.audit_response(*fixture())
        serialized = json.dumps(result)
        self.assertNotIn('No pleural effusion.', serialized)
        self.assertNotIn('"quote"', serialized)

    def test_no_slurm_refuses_before_input_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, 'load_plan') as load:
            with self.assertRaises(RuntimeError):
                cli.run(SimpleNamespace())
            load.assert_not_called()

    def test_overwrite_refused_before_text_access(self):
        target = MagicMock()
        target.exists.return_value = True
        args = SimpleNamespace(output_root='unused', run_id='fixture_existing')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(cli, 'require_inside', return_value=target), patch.object(cli, 'load_plan') as load:
            with self.assertRaises(FileExistsError):
                cli.run(args)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()

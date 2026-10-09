"""Invented text/mock metadata only; never model or patient-data access."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'benchmarks'))
from test_legacy_report_scope import scoped_fixture
from repair_cached_report_evidence import scope_check
from tricompose_v12.legacy_report_scope import build_scope_audit
from tricompose_v12.legacy_report_semantics import HEADS, compare_extractions
from verify_report_evidence_qwen import request_messages, decode_evidence, digest_text
import verify_legacy_report_semantics as cli


def fixture(raw_state='positive', qwen_state='positive', duplicate=False):
    text = 'Pleural effusion is present.' if qwen_state == 'positive' else 'No pleural effusion.'
    count = 2 if duplicate else 1
    rows, facts, groups, texts = scoped_fixture(texts=(text,)*count, states=(raw_state,)*count)
    scoped, _, _, _ = build_scope_audit(rows, facts, groups, texts, scope_check)
    assertions = {f: {'positive': [], 'negative': [], 'uncertain': []} for f in HEADS}
    if qwen_state != 'unknown':
        assertions['pleural_effusion'][qwen_state] = [text]
    record = {'report_sha256': digest_text(text), 'model_call_attempted': True,
        **decode_evidence(json.dumps(assertions), text)}
    return rows, scoped, [record]


class SemanticAuditTests(unittest.TestCase):
    def test_positive_evidence_can_be_recorded_without_changing_original(self):
        values = fixture()
        before = copy.deepcopy(values)
        table, summary = compare_extractions(*values)
        self.assertEqual(values, before)
        finding = next(r for r in table if r['finding'] == 'pleural_effusion')
        self.assertEqual(finding['qwen_assertion_state'], 'positive')
        self.assertEqual(finding['chexbert_qwen_comparison'], 'same_explicit_state')
        self.assertFalse(summary['selection_changed'])
        self.assertIsNone(summary['clinical_accuracy'])
        self.assertFalse(summary['primary_metric_eligible'])

    def test_opposition_not_fault_localization_or_new_score(self):
        table, summary = compare_extractions(*fixture(qwen_state='negative'))
        finding = next(r for r in table if r['finding'] == 'pleural_effusion')
        self.assertEqual(finding['raw_chexbert_state'], 'positive')
        self.assertEqual(finding['qwen_assertion_state'], 'negative')
        self.assertEqual(finding['chexbert_qwen_comparison'], 'opposed_explicit_state')
        self.assertIsNone(finding['clinical_selection_score'])
        self.assertIsNone(finding['confirmed_faulty_modality'])
        self.assertFalse(summary['regeneration_authorized'])

    def test_unknown_unknown_is_not_agreement(self):
        table, _ = compare_extractions(*fixture(raw_state='unknown', qwen_state='unknown'))
        self.assertTrue(all(r['chexbert_qwen_comparison'] == 'not_comparable' for r in table))

    def test_uncertain_not_positive_or_comparable(self):
        table, _ = compare_extractions(*fixture(qwen_state='uncertain'))
        finding = next(r for r in table if r['finding'] == 'pleural_effusion')
        self.assertEqual(finding['qwen_assertion_state'], 'uncertain')
        self.assertEqual(finding['chexbert_qwen_comparison'], 'not_comparable')

    def test_all_fourteen_heads_retained_ten_unavailable(self):
        table, summary = compare_extractions(*fixture())
        self.assertEqual(len(table), 14)
        outside = [r for r in table if r['finding'] not in HEADS]
        self.assertEqual(len(outside), 10)
        self.assertEqual(summary['outside_scope_candidate_rows'], 10)
        self.assertTrue(all(r['qwen_contract_status'] == 'outside_scope_inventory' for r in outside))

    def test_shared_report_inference_not_independent_votes(self):
        table, summary = compare_extractions(*fixture(duplicate=True))
        self.assertEqual(len(table), 28)
        self.assertEqual(summary['distinct_report_texts'], 1)
        effusion = summary['same_four_heads']['pleural_effusion']
        self.assertEqual(effusion['qwen_states_candidate_slots'], {'positive': 2})
        self.assertEqual(effusion['qwen_states_distinct_texts'], {'positive': 1})

    def test_failed_parse_unknown_and_counted(self):
        rows, scoped, records = fixture()
        records[0].update(decode_evidence('{}', 'invented source'))
        table, summary = compare_extractions(rows, scoped, records)
        self.assertEqual(summary['failed_distinct_responses'], 1)
        self.assertTrue(all(r['chexbert_qwen_comparison'] == 'not_comparable' for r in table))

    def test_duplicate_or_missing_receipts_refused(self):
        rows, scoped, records = fixture()
        for bad in ([], records*2):
            with self.assertRaises(ValueError):
                compare_extractions(rows, scoped, bad)

    def test_missing_or_duplicate_fact_inventory_refused(self):
        rows, scoped, records = fixture()
        for bad in (scoped[:-1], scoped+[scoped[0]]):
            with self.assertRaises(ValueError):
                compare_extractions(rows, bad, records)

    def test_lineage_mismatch_refused(self):
        rows, scoped, records = fixture()
        scoped[0]['artifact_hashes']['report_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'lineage'):
            compare_extractions(rows, scoped, records)

    def test_forged_clinical_validation_refused(self):
        rows, scoped, records = fixture()
        records[0]['findings']['pleural_effusion']['semantic_correctness_independently_verified'] = True
        with self.assertRaises(ValueError):
            compare_extractions(rows, scoped, records)

    def test_failed_record_cannot_have_positive_finding(self):
        rows, scoped, records = fixture()
        records[0]['contract_status'] = 'failed_unavailable'
        with self.assertRaises(ValueError):
            compare_extractions(rows, scoped, records)

    def test_no_quotes_in_comparison_table(self):
        table, summary = compare_extractions(*fixture())
        serialized = json.dumps([table, summary])
        self.assertNotIn('Pleural effusion is present.', serialized)
        self.assertNotIn('"quote"', serialized)

    def test_request_contains_report_only_and_preserves_exact_text(self):
        text = 'Invented clinical text.\r\nLine two: α.'
        messages = request_messages(text)
        self.assertEqual(len(messages), 1)
        self.assertEqual([c['type'] for c in messages[0]['content']], ['text'])
        self.assertIn(text, messages[0]['content'][0]['text'])
        self.assertNotIn('image', messages[0]['content'][0])

    def test_invented_quote_fails_closed(self):
        payload = {f: {'positive': [], 'negative': [], 'uncertain': []} for f in HEADS}
        payload['pleural_effusion']['positive'] = ['Invented unseen effusion.']
        result = decode_evidence(json.dumps(payload), 'No pleural effusion.')
        self.assertEqual(result['contract_status'], 'failed_unavailable')
        self.assertEqual(result['contract_failure_reason'], 'quote_not_in_source')

    def test_token_limit_fails_even_for_complete_json(self):
        payload = {f: {'positive': [], 'negative': [], 'uncertain': []} for f in HEADS}
        result = decode_evidence(json.dumps(payload), 'Invented text.', token_limit_reached=True)
        self.assertEqual(result['contract_failure_reason'], 'token_limit_reached')

    def test_slurm_guard_before_path_model_or_report_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, 'require_inside') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'slurm_required'):
                cli.execute(SimpleNamespace())
            resolve.assert_not_called()

    def test_existing_run_refused_before_model_or_text(self):
        target = MagicMock()
        target.exists.return_value = True
        args = SimpleNamespace(run_id='invented_existing', output_root='unused', mode='run')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(cli, 'require_inside', return_value=target), patch.object(cli, 'verify') as verify:
            with self.assertRaises(FileExistsError):
                cli.execute(args)
            verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()

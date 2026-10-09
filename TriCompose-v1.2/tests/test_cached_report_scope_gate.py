"""Invented source text/metadata only; no real data, models or GPU."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cached_scope_test_worker', ROOT/'tools/gate_cached_report_spans.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prediction(text, state='positive', status='complete'):
    h = m.challenge.text_hash(text)
    resolver = [{'item_id': 'invented_000', 'report_sha256': h}]
    row = {'item_id': 'invented_000', 'report_sha256': h, 'status': status,
        'finding_states': {**dict.fromkeys(m.FINDINGS, 'unknown'), 'cardiomegaly': state}}
    return resolver, {h: text}, {'invented': [row]}


def fact(text, finding='cardiomegaly', state='positive'):
    row = dict.fromkeys(m.progress.FACT_FIELDS)
    row.update(finding=finding, qwen_assertion_state=state,
        qwen_contract_status='complete' if finding in m.FINDINGS else 'outside_scope_inventory',
        report_sha256=m.challenge.text_hash(text), report_candidate_id='invented_report',
        triple_candidate_id='invented_triple', independent_clinical_validation=False,
        selection_changed=False, regeneration_authorized=False)
    return row


class CachedReportScopeGateTests(unittest.TestCase):
    def test_keep_explicit_proposal_but_not_clinical_truth(self):
        row = m.source_gate('Cardiomegaly is present.', 'cardiomegaly', 'positive')
        self.assertEqual(row['scopegate_retained_state'], 'positive')
        self.assertEqual(row['scopegate_decision'], 'scope_commit')
        self.assertFalse(row['scopegate_independent_clinical_validation'])
        self.assertFalse(row['scopegate_regeneration_authorized'])

    def test_incorrect_polarity_abstains_without_flipping(self):
        row = m.source_gate('Cardiomegaly is absent.', 'cardiomegaly', 'positive')
        self.assertEqual(row['scopegate_decision'], 'abstain')
        self.assertIsNone(row['scopegate_retained_state'])

    def test_unknown_does_not_become_negative_even_with_explicit_source(self):
        row = m.source_gate('No cardiomegaly.', 'cardiomegaly', 'unknown')
        self.assertEqual(row['scopegate_decision'], 'no_model_assertion')
        self.assertIsNone(row['scopegate_retained_state'])
        self.assertEqual(row['scopegate_evidence'], [])

    def test_uncertainty_does_not_support_determinate_proposal(self):
        for text in ('Cardiomegaly may be present.', 'No large cardiomegaly.'):
            self.assertEqual(m.source_gate(text, 'cardiomegaly', 'positive')['scopegate_decision'], 'abstain')
            self.assertEqual(m.source_gate(text, 'cardiomegaly', 'uncertain')['scopegate_retained_state'], 'uncertain')

    def test_synonym_and_historical_context_not_unearned_pass(self):
        for text in ('The heart is enlarged.', 'History of cardiomegaly is present.'):
            self.assertEqual(m.source_gate(text, 'cardiomegaly', 'positive')['scopegate_decision'], 'abstain')

    def test_failure_and_outside_scope_distinct_from_missing_assertion(self):
        row = m.source_gate('No cardiomegaly.', 'cardiomegaly', 'unknown', contract_status='failed_unavailable')
        self.assertEqual(row['scopegate_decision'], 'verifier_unavailable')
        self.assertEqual(m.source_gate('No edema.', 'edema', 'unknown')['scopegate_decision'], 'outside_verifier_scope')

    def test_evidence_export_contains_hash_not_quote(self):
        row = m.source_gate('No cardiomegaly.', 'cardiomegaly', 'negative')
        self.assertTrue(row['scopegate_evidence'])
        self.assertTrue(all('quote' not in span for span in row['scopegate_evidence']))

    def test_authored_all_four_finding_rows_and_original_prediction_preserved(self):
        args = prediction('Cardiomegaly is present.')
        before = copy.deepcopy(args)
        rows = m.authored_rows(*args)
        self.assertEqual(args, before)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['raw_state'], 'positive')

    def test_authored_duplicate_missing_or_hash_mismatch_refused(self):
        resolver, texts, predictions = prediction('Cardiomegaly is present.')
        for records in ([], predictions['invented']*2, [{**predictions['invented'][0], 'report_sha256': 'b'*64}]):
            with self.assertRaises(ValueError):
                m.authored_rows(resolver, texts, {'invented': records})

    def test_candidate_every_original_cell_preserved(self):
        text = 'No cardiomegaly.'
        original = fact(text, state='positive')
        inputs = [{'report_candidate_id': original['report_candidate_id'], 'report_sha256': original['report_sha256']}]
        output = m.synthetic_rows([original], {original['report_sha256']: text}, inputs)[0]
        self.assertEqual({k: output[k] for k in original}, original)
        self.assertIsNone(output['scopegate_retained_state'])

    def test_candidate_exact_report_identity_required(self):
        text = 'No cardiomegaly.'
        row = fact(text)
        with self.assertRaises(ValueError):
            m.synthetic_rows([row], {row['report_sha256']: text},
                [{'report_candidate_id': row['report_candidate_id'], 'report_sha256': 'b'*64}])

    def test_abstention_does_not_count_as_correct_unknown(self):
        text = 'Cardiomegaly is present.'
        rows = m.authored_rows(*prediction(text, state='unknown'))
        ref = {'item_id': 'invented_000', 'report_sha256': m.challenge.text_hash(text),
            'evaluation_findings': ['cardiomegaly'], 'expected_states': {'cardiomegaly': 'unknown'},
            'family': 'invented_family'}
        summary, _ = m.analyze_authored(rows, [ref])
        value = summary['invented']['designated_targets']
        self.assertEqual(value['raw_exact_matches'], 1)
        self.assertEqual(value['scope_commits'], 0)
        self.assertIsNone(value['conditional_authored_match'])
        self.assertEqual(value['noncommitted_checks'], 1)

    def test_wrong_raw_proposal_removed_not_corrected(self):
        text = 'Cardiomegaly is absent.'
        rows = m.authored_rows(*prediction(text))
        ref = {'item_id': 'invented_000', 'report_sha256': m.challenge.text_hash(text),
            'evaluation_findings': ['cardiomegaly'], 'expected_states': {'cardiomegaly': 'negative'},
            'family': 'invented_family'}
        summary, _ = m.analyze_authored(rows, [ref])
        value = summary['invented']['designated_targets']
        self.assertEqual(value['raw_errors_not_committed'], 1)
        self.assertEqual(value['correct_commits'], 0)

    def test_reference_inventory_hash_and_duplicates_checked(self):
        text = 'Cardiomegaly is present.'
        rows = m.authored_rows(*prediction(text))
        with self.assertRaises(ValueError):
            m.analyze_authored(rows, [])
        with self.assertRaises(ValueError):
            m.analyze_authored(rows+rows, [])

    def test_non_slurm_refused_before_any_input_read(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(m, 'load_fixed_inputs') as access:
            with self.assertRaises(RuntimeError):
                m.execute('/invented', 'invented_001')
            access.assert_not_called()

    def test_spoofed_slurm_variable_without_cgroup_refused(self):
        with patch.dict('os.environ', {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(Path, 'read_text', return_value='invented_non_slurm'):
            with self.assertRaises(RuntimeError):
                m.require_cpu_slurm()

    def test_existing_run_refused_before_input_read(self):
        with patch.object(m, 'require_cpu_slurm'), patch.object(m, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(m, 'load_fixed_inputs') as access:
            with self.assertRaises(FileExistsError):
                m.execute('/invented', 'invented_001')
            access.assert_not_called()


if __name__ == '__main__':
    unittest.main()

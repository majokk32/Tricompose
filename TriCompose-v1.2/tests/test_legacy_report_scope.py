"""Invented source text and metadata only; no original reports/images/models."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'benchmarks'))
from test_scorer_reliability import fixture
from tricompose_v12 import scorer_reliability as source
from tricompose_v12 import legacy_report_scope as module
from repair_cached_report_evidence import scope_check

spec = importlib.util.spec_from_file_location('legacy_report_scope_cli_fixture', ROOT/'benchmarks/check_legacy_report_scope.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def scoped_fixture(texts=('Pleural effusion is present.',), states=('positive',), ehr='positive', image='positive'):
    rows, bank = fixture(experts=states)
    for index, text in enumerate(texts):
        h = module.digest(text)
        row, candidate = rows[index], bank['fixture_case'][index]
        row['report_sha256'] = h
        candidate['score_record']['lineage']['report_sha256'] = h
        for fact in candidate['facts']:
            fact['artifact_hashes']['report_sha256'] = h
            fact['states'] = {'ehr': 'unknown', 'xrv': 'unknown', 'chexbert': 'unknown'}
            fact['source_categories'] = []
            if fact['finding'] == 'pleural_effusion':
                fact['states'] = {'ehr': ehr, 'xrv': image, 'chexbert': states[index]}
                fact['source_categories'] = ['diagnoses'] if ehr != 'unknown' else []
        for edge, (left, right) in source.EDGES.items():
            values = source._edge([f['states'] for f in candidate['facts']], left, right)
            row.update({edge+'_'+key: '' if value is None else str(value) for key, value in values.items()})
    annotated, facts, groups, _ = source.build_overlay(rows, bank)
    return annotated, facts, groups, {module.digest(text): text for text in texts}


class LegacyScopeTests(unittest.TestCase):
    def test_scope_commit_retains_original_state_not_new_clinical_truth(self):
        inputs = scoped_fixture()
        before = copy.deepcopy(inputs)
        facts, _, _, summary = module.build_scope_audit(*inputs, scope_check)
        self.assertEqual(inputs, before)
        checked = next(f for f in facts if f['finding'] == 'pleural_effusion')
        self.assertEqual(checked['scope_decision'], 'scope_commit')
        self.assertEqual(checked['scope_report_state'], checked['states']['chexbert'])
        self.assertFalse(checked['independent_clinical_validation'])
        self.assertFalse(summary['selection_changed'])
        self.assertIsNone(summary['clinical_accuracy'])

    def test_negation_veto_withdraws_to_unknown_does_not_flip(self):
        facts, _, _, _ = module.build_scope_audit(*scoped_fixture(texts=('No pleural effusion.',)), scope_check)
        checked = next(f for f in facts if f['finding'] == 'pleural_effusion')
        self.assertEqual(checked['scope_decision'], 'abstain')
        self.assertEqual(checked['scope_report_state'], 'unknown')
        self.assertEqual(checked['states']['chexbert'], 'positive')

    def test_missing_report_assertion_not_recovered_from_positive_text(self):
        facts, _, _, _ = module.build_scope_audit(*scoped_fixture(states=('unknown',)), scope_check)
        checked = next(f for f in facts if f['finding'] == 'pleural_effusion')
        self.assertEqual(checked['scope_decision'], 'no_model_assertion')
        self.assertEqual(checked['scope_report_state'], 'unknown')

    def test_qualified_absence_not_explicit_global_negative(self):
        facts, _, _, _ = module.build_scope_audit(*scoped_fixture(texts=('No large pleural effusion.',), states=('negative',)), scope_check)
        checked = next(f for f in facts if f['finding'] == 'pleural_effusion')
        self.assertEqual(checked['scope_decision'], 'abstain')
        self.assertEqual(checked['scope_report_state'], 'unknown')

    def test_synonym_not_added_to_frozen_guard(self):
        facts, _, _, _ = module.build_scope_audit(*scoped_fixture(texts=('Fluid blunts the costophrenic angle.',)), scope_check)
        checked = next(f for f in facts if f['finding'] == 'pleural_effusion')
        self.assertEqual(checked['scope_decision'], 'abstain')

    def test_outside_scope_heads_all_preserved_as_unavailable(self):
        facts, _, _, summary = module.build_scope_audit(*scoped_fixture(), scope_check)
        outside = [f for f in facts if not f['scope_supported_finding']]
        self.assertEqual(len(facts), 14)
        self.assertEqual(len(outside), 10)
        self.assertTrue(all(f['scope_decision'] == 'outside_scope_inventory' and not f['scope_available'] for f in outside))
        self.assertIn('pneumonia', {f['finding'] for f in outside})
        self.assertEqual(summary['scope_finding_inventory'], list(module.SCOPE_FINDINGS))

    def test_offsets_and_hashes_without_report_quotes(self):
        facts, _, _, _ = module.build_scope_audit(*scoped_fixture(), scope_check)
        checked = next(f for f in facts if f['finding'] == 'pleural_effusion')
        self.assertEqual(checked['evidence_offsets_and_hashes'][0]['char_start'], 0)
        self.assertNotIn('quote', checked['evidence_offsets_and_hashes'][0])
        self.assertFalse(checked['report_quote_copied_to_output'])

    def test_ehr_image_states_and_relation_are_never_changed(self):
        facts, _, _, summary = module.build_scope_audit(*scoped_fixture(texts=('No pleural effusion.',)), scope_check)
        self.assertTrue(all(f['relations']['raw']['ehr_cxr'] == f['relations']['scoped']['ehr_cxr'] for f in facts))
        self.assertEqual(summary['relations']['raw']['ehr_cxr'], summary['relations']['scoped']['ehr_cxr'])

    def test_lost_opposition_reported_as_abstention_not_repair(self):
        facts, _, _, summary = module.build_scope_audit(*scoped_fixture(texts=('No large pleural effusion.',), states=('negative',)), scope_check)
        self.assertEqual(summary['relations']['raw']['cxr_report']['opposition'], 1)
        self.assertEqual(summary['relations']['scoped']['cxr_report']['comparable_facts'], 0)
        self.assertEqual(summary['withdrawn_proxy_comparisons']['cxr_report']['abstain'], 1)
        self.assertFalse(summary['regeneration_authorized'])

    def test_duplicate_report_artifact_scope_checker_invoked_once(self):
        inputs = scoped_fixture(texts=('Pleural effusion is present.',)*2, states=('positive',)*2)
        calls = []
        def counted(*args):
            calls.append(args[2])
            return scope_check(*args)
        _, _, _, summary = module.build_scope_audit(*inputs, counted)
        self.assertEqual(calls, ['pleural_effusion'])
        self.assertEqual(summary['unique_report_artifacts'], 1)
        self.assertEqual(summary['candidate_rows'], 2)

    def test_unknown_ehr_coverage_na_not_negative(self):
        _, _, _, summary = module.build_scope_audit(*scoped_fixture(ehr='unknown'), scope_check)
        self.assertIsNone(summary['relations']['scoped']['ehr_report']['coverage_over_explicit_reference'])
        self.assertEqual(summary['relations']['scoped']['ehr_report']['explicit_reference_facts'], 0)

    def test_unavailable_text_keeps_full_inventory(self):
        rows, facts, groups, texts = scoped_fixture()
        h = next(iter(texts))
        result, _, _, summary = module.build_scope_audit(rows, facts, groups, {}, scope_check,
            text_unavailable={h: 'invalid_utf8'})
        self.assertEqual(len(result), 14)
        self.assertEqual(summary['unavailable_text_artifacts'], 1)
        self.assertEqual(sum(f['scope_decision'] == 'source_text_unavailable' for f in result), 4)

    def test_text_hash_inventory_empty_or_overlength_refused(self):
        rows, facts, groups, texts = scoped_fixture()
        h = next(iter(texts))
        for bad in ({}, {h: 'different'}, {h: ''}, {h: 'a'*8193}):
            with self.assertRaises(ValueError):
                module.build_scope_audit(rows, facts, groups, bad, scope_check)

    def test_selection_keeps_complete_case_and_ignores_scores(self):
        rows, facts, groups, _ = scoped_fixture()
        first = module.select_cases(rows, facts, groups, count=1)
        rows[0]['biovil_raw_cosine'] = '0.99'
        second = module.select_cases(rows, facts, groups, count=1)
        self.assertEqual([r['triple_candidate_id'] for r in first[0]], [r['triple_candidate_id'] for r in second[0]])
        self.assertEqual(len(first[1]), 14)

    def test_bad_scope_batch_size_or_case_count_refused(self):
        rows, facts, groups, _ = scoped_fixture()
        for count in (0, 3, True):
            with self.assertRaises(ValueError):
                module.select_cases(rows, facts, groups, count=count)
        with self.assertRaises(ValueError):
            module.select_cases(rows, facts, groups, count=2)

    def test_invalid_shared_states_or_forged_clinical_eligibility_refused(self):
        rows, facts, groups, texts = scoped_fixture()
        facts[0]['clinical_truth_available'] = True
        with self.assertRaises(ValueError):
            module.build_scope_audit(rows, facts, groups, texts, scope_check)

    def test_cpu_guard_before_paths_and_text_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, 'require_inside') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'existing_cpu_slurm_required'):
                cli.run(SimpleNamespace())
            resolve.assert_not_called()

    def test_existing_run_refused_before_text_access(self):
        target = MagicMock()
        target.exists.return_value = True
        args = SimpleNamespace(run_id='existing_fixture', output_root='unused')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(cli, 'require_inside', return_value=target), patch.object(cli, 'read_text_inputs') as reader:
            with self.assertRaises(FileExistsError):
                cli.run(args)
            reader.assert_not_called()

    def test_exact_byte_reader_preserves_crlf_and_utf8(self):
        raw = 'Pleural effusion is present.\r\nUnicode: α'.encode('utf-8')
        path = MagicMock()
        path.stat.return_value.st_size = len(raw)
        path.read_bytes.return_value = raw
        h = hashlib.sha256(raw).hexdigest()
        with patch.object(cli, 'require_inside', return_value=path):
            texts, unavailable, _, count = cli.read_text_inputs([{'path': 'invented', 'report_sha256': h}])
        self.assertEqual(texts[h], raw.decode('utf-8'))
        self.assertEqual(unavailable, {})
        self.assertEqual(count, 1)

    def test_malformed_or_empty_text_returns_unavailable_without_truncation(self):
        for raw, reason in ((b'', 'empty_report'), (b'\xff', 'invalid_utf8')):
            path = MagicMock()
            path.stat.return_value.st_size = len(raw)
            path.read_bytes.return_value = raw
            h = hashlib.sha256(raw).hexdigest()
            with patch.object(cli, 'require_inside', return_value=path):
                texts, unavailable, _, _ = cli.read_text_inputs([{'path': 'invented', 'report_sha256': h}])
            self.assertEqual(texts, {})
            self.assertEqual(unavailable[h], reason)

    def test_changed_bytes_refused_without_printing_body(self):
        path = MagicMock()
        path.stat.return_value.st_size = 5
        path.read_bytes.return_value = b'wrong'
        with patch.object(cli, 'require_inside', return_value=path):
            with self.assertRaisesRegex(ValueError, 'sealed_synthetic_report_bytes_changed'):
                cli.read_text_inputs([{'path': 'invented', 'report_sha256': 'a'*64}])


if __name__ == '__main__':
    unittest.main()

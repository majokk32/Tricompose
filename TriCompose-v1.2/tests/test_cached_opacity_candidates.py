"""Invented metadata/score fixtures only; no image bodies or model inference."""
import copy
import csv
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cached_opacity_fixture', ROOT / 'tools/score_cached_opacity_candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(experts=('positive', 'negative', 'unknown', 'uncertain'), ehr='unknown', exact=.2, infiltration=.9):
    rows, records = [], []
    for index, report in enumerate(experts):
        row = {'case_id': 'case_fixture', 'triple_candidate_id': f'triple_{index}',
            'cxr_candidate_id': 'image_fixture', 'report_candidate_id': f'report_{index}',
            'report_sha256': str(index) * 64, 'source_score': '0.7000', 'source_missing': ''}
        rows.append(row)
        records.append({**row, 'ehr_sha256': 'a' * 64, 'ehr_facts_sha256': 'b' * 64,
            'cxr_sha256': 'c' * 64, 'cxr_model_id': 'fixture_model', 'cxr_seed': 0,
            'report_model_id': 'fixture_expert', 'ehr_opacity_state': ehr,
            'report_opacity_state': report, 'cached_legacy_max_opacity_state': 'positive',
            'cached_ehr_source_categories': ['diagnosis'] if ehr in module.EXPLICIT else []})
    results = {'image_fixture': {'cxr_candidate_id': 'image_fixture', 'case_id': 'case_fixture',
        'cxr_sha256': 'c' * 64, 'status': 'scored', 'model_calls': 1,
        'exact_lung_opacity_score': exact, 'infiltration_score': infiltration}}
    return rows, records, results


def flatten_fixture():
    rows, records, _ = fixture()
    bank = {'case_fixture': {}}
    for i, (row, record) in enumerate(zip(rows, records)):
        lineage = {k: record[k] for k in ('cxr_candidate_id', 'report_candidate_id', 'report_model_id',
            'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256', 'cxr_model_id', 'cxr_seed')}
        row.update({k: lineage[k] for k in ('report_model_id', 'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256')},
                   profile=module.PROFILE)
        original = {'case_id': row['case_id'], 'triple_candidate_id': row['triple_candidate_id'], 'lineage': lineage}
        fact = {'finding': 'lung_opacity', 'clinical_truth_verified': False, 'weak_context_promoted': False,
            'states': {'ehr': 'unknown', 'xrv': 'positive', 'chexbert': record['report_opacity_state']}, 'source_categories': []}
        bank['case_fixture'][i] = {'score_record': original, 'facts': [fact]}
    return rows, bank


class OpacityScoreTests(unittest.TestCase):
    def test_exact_head_not_historical_max(self):
        rows, records, scores = fixture()
        output, _ = module.overlay(rows, records, scores)
        self.assertEqual(output[0]['opacity_exact_score'], .2)
        self.assertEqual(output[0]['opacity_exact_state_0_5'], 'negative')
        self.assertEqual(output[0]['opacity_same_call_max_score'], .9)
        self.assertEqual(output[0]['opacity_same_call_max_state_0_5'], 'positive')
        self.assertTrue(output[0]['opacity_exact_vs_same_call_max_state_changed'])
        self.assertFalse(module.POLICY['probability_semantics'])

    def test_boundary_frozen_at_point_five(self):
        self.assertEqual(module.state(.5), 'positive')
        self.assertEqual(module.state(.49999999), 'negative')
        self.assertEqual(module.state(0), 'negative')
        self.assertEqual(module.state(1), 'positive')
        self.assertEqual(module.state(None), 'unknown')

    def test_nonfinite_out_of_range_boolean_scores_refused(self):
        for score in (float('nan'), float('inf'), -.1, 1.1, True, False, '0.5'):
            with self.subTest(score=score), self.assertRaises(ValueError):
                module.state(score)

    def test_ehr_unknown_not_inferred_from_image_report_or_pneumonia(self):
        rows, records, scores = fixture()
        records[0]['pneumonia_state'] = 'positive'
        output, _ = module.overlay(rows, records, scores)
        self.assertTrue(all(r['opacity_cached_ehr_state'] == 'unknown' for r in output))
        self.assertTrue(all(r['opacity_ehr_cxr_relation'] == 'not_comparable' for r in output))
        self.assertTrue(all(r['opacity_ehr_report_relation'] == 'not_comparable' for r in output))
        self.assertFalse(module.POLICY['pneumonia_is_opacity'])
        self.assertFalse(module.POLICY['global_no_finding_expands_unknown'])

    def test_uncertain_and_unknown_report_not_negative(self):
        output, summary = module.overlay(*fixture())
        self.assertEqual([r['opacity_cxr_report_relation'] for r in output],
                         ['proxy_opposition', 'proxy_support', 'not_comparable', 'not_comparable'])
        self.assertEqual(summary[0]['comparable_pairs'], 2)
        self.assertEqual(summary[0]['candidate_rows'], 4)
        self.assertEqual(summary[0]['agreement_over_comparable'], .5)
        self.assertEqual(summary[0]['comparable_coverage_over_all_rows'], .5)

    def test_zero_comparable_is_na_not_perfect(self):
        _, summary = module.overlay(*fixture(experts=('unknown', 'uncertain')))
        self.assertIsNone(summary[0]['agreement_over_comparable'])
        self.assertEqual(summary[0]['comparable_coverage_over_all_rows'], 0)

    def test_raw_opposition_not_confirmed_fault_or_clinical_accuracy(self):
        output, summary = module.overlay(*fixture(experts=('positive',)))
        self.assertEqual(output[0]['opacity_cxr_report_relation'], 'proxy_opposition')
        self.assertIsNone(output[0]['opacity_clinical_accuracy'])
        self.assertIsNone(output[0]['opacity_confirmed_faulty_modality'])
        self.assertFalse(output[0]['opacity_selector_used'])
        self.assertIsNone(summary[0]['clinical_accuracy'])
        self.assertFalse(summary[0]['independent_patients'])

    def test_original_cells_order_and_sources_unchanged(self):
        inputs = fixture(); before = copy.deepcopy(inputs)
        output, _ = module.overlay(*inputs)
        self.assertEqual(inputs, before)
        for old, new in zip(inputs[0], output):
            self.assertTrue(all(old[k] == new[k] for k in old))
        loaded = list(csv.DictReader(io.StringIO(module.csv_text(output))))
        for old, new in zip(inputs[0], loaded):
            self.assertTrue(all(old[k] == new[k] for k in old))

    def test_failed_image_preserves_all_report_slots_and_denominator(self):
        rows, records, scores = fixture()
        scores['image_fixture'].update(status='failed_without_replacement',
            exact_lung_opacity_score=None, infiltration_score=None)
        output, summary = module.overlay(rows, records, scores)
        self.assertEqual(len(output), 4)
        self.assertTrue(all(r['opacity_exact_state_0_5'] == 'unknown' for r in output))
        self.assertTrue(all(r['opacity_cxr_report_relation'] == 'not_comparable' for r in output))
        self.assertIsNone(summary[0]['agreement_over_comparable'])
        self.assertIsNone(output[0]['opacity_exact_vs_same_call_max_state_changed'])

    def test_failed_outcome_cannot_smuggle_score(self):
        rows, records, scores = fixture()
        scores['image_fixture']['status'] = 'failed_without_replacement'
        with self.assertRaisesRegex(ValueError, 'explicit_scored_or_failed'):
            module.overlay(rows, records, scores)

    def test_scored_outcome_requires_both_original_heads(self):
        rows, records, scores = fixture()
        scores['image_fixture']['infiltration_score'] = None
        with self.assertRaisesRegex(ValueError, 'both_raw_heads'):
            module.overlay(rows, records, scores)

    def test_no_missing_foreign_image_outcomes(self):
        rows, records, scores = fixture()
        for bad in ({}, {**scores, 'foreign_image': scores['image_fixture']}):
            with self.assertRaisesRegex(ValueError, 'all_image_outcomes'):
                module.overlay(rows, records, bad)

    def test_artifact_hash_or_case_change_refused(self):
        for field, value in (('case_id', 'foreign_case'), ('cxr_sha256', 'f' * 64), ('cxr_candidate_id', 'foreign_image')):
            rows, records, scores = fixture(); scores['image_fixture'][field] = value
            with self.assertRaisesRegex(ValueError, 'new_score_image_lineage'):
                module.overlay(rows, records, scores)

    def test_swapped_csv_order_refused(self):
        rows, records, scores = fixture()
        with self.assertRaisesRegex(ValueError, 'original_row_order'):
            module.overlay(list(reversed(rows)), records, scores)

    def test_second_annotation_refused(self):
        rows, records, scores = fixture(); rows[0]['opacity_selector_used'] = False
        with self.assertRaisesRegex(ValueError, 'already_annotated'):
            module.overlay(rows, records, scores)

    def test_inventory_uses_image_case_and_artifact_denominators(self):
        _, records, _ = fixture()
        counts = module.inventory(records)
        self.assertEqual(counts['fixed_ehr_cases'], 1)
        self.assertEqual(counts['image_slots'], 1)
        self.assertEqual(counts['candidate_rows'], 4)
        self.assertEqual(counts['ehr_opacity_explicit_cases'], 0)
        self.assertEqual(counts['ehr_opacity_states_one_per_case'], {'unknown': 1})
        self.assertEqual(counts['cached_legacy_max_states_one_per_image'], {'positive': 1})

    def test_repeated_report_hash_dedup_not_independent_votes(self):
        _, records, _ = fixture(experts=('positive', 'positive'))
        records[1]['report_sha256'] = records[0]['report_sha256']
        counts = module.inventory(records)
        self.assertEqual(counts['unique_report_artifact_hashes'], 1)
        self.assertEqual(counts['report_opacity_states_unique_artifact_hashes'], {'positive': 1})
        self.assertEqual(counts['report_opacity_states_candidate_rows'], {'positive': 2})

    def test_shared_hash_cannot_have_different_report_states(self):
        _, records, _ = fixture(experts=('positive', 'negative'))
        records[1]['report_sha256'] = records[0]['report_sha256']
        with self.assertRaisesRegex(ValueError, 'shared_artifact'):
            module.inventory(records)

    def test_all_cases_need_fixed_ehr_and_shared_image(self):
        for field in ('ehr_opacity_state', 'cxr_sha256', 'cxr_seed'):
            _, records, _ = fixture(); records[1][field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'shared_artifact'):
                module.inventory(records)

    def test_incomplete_full_grid_rejected(self):
        _, records, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'full_fixed_eighty'):
            module.inventory(records, full=True)

    def test_invalid_four_state_evidence_rejected(self):
        for left, right in (('not mentioned', 'positive'), ('negative', None)):
            with self.assertRaises(ValueError):
                module.relation(left, right)

    def test_flatten_reads_only_cached_opacity_and_retains_order(self):
        inputs = flatten_fixture(); before = copy.deepcopy(inputs)
        result = module.flatten(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual([r['triple_candidate_id'] for r in result], [r['triple_candidate_id'] for r in inputs[0]])
        self.assertTrue(all(r['ehr_opacity_state'] == 'unknown' for r in result))
        self.assertEqual([r['report_opacity_state'] for r in result], ['positive', 'negative', 'unknown', 'uncertain'])

    def test_flatten_missing_duplicate_or_foreign_triple_rejected(self):
        rows, bank = flatten_fixture()
        for bad in (rows[:1], [rows[0]] * len(rows), [{**r, 'triple_candidate_id': 'foreign'} for r in rows]):
            with self.assertRaisesRegex(ValueError, 'exact_unique_candidate'):
                module.flatten(bad, bank)

    def test_flatten_mixed_profile_or_changed_lineage_rejected(self):
        for key in ('profile', 'case_id', 'ehr_sha256', 'report_model_id', 'cxr_candidate_id'):
            rows, bank = flatten_fixture(); rows[0][key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'source_row_lineage'):
                module.flatten(rows, bank)

    def test_flatten_unqualified_cached_fact_only(self):
        for key in ('clinical_truth_verified', 'weak_context_promoted'):
            rows, bank = flatten_fixture(); bank['case_fixture'][0]['facts'][0][key] = True
            with self.assertRaisesRegex(ValueError, 'one_unqualified'):
                module.flatten(rows, bank)

    def test_flatten_explicit_ehr_needs_source_category(self):
        rows, bank = flatten_fixture(); bank['case_fixture'][0]['facts'][0]['states']['ehr'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'source_category'):
            module.flatten(rows, bank)

    def test_cached_baseline_is_max_not_exact_and_unknown_safe(self):
        _, records, _ = fixture()
        self.assertEqual(module.cached_relations(records), {'proxy_support': 1, 'proxy_opposition': 1, 'not_comparable': 2})

    def test_evaluate_requires_explicit_plan_sha_before_metadata(self):
        for digest in (None, '', 'f' * 63, 'g' * 64):
            args = SimpleNamespace(allow_synthetic_xrv=True, plan_manifest_sha256=digest)
            with patch.object(module, 'guard'), patch.object(module, 'metadata') as metadata:
                with self.assertRaisesRegex(ValueError, 'explicit_frozen_plan'):
                    module.evaluate(args)
                metadata.assert_not_called()

    def test_explicit_ehr_negative_supported_not_missing(self):
        output, _ = module.overlay(*fixture(ehr='negative'))
        self.assertTrue(all(r['opacity_ehr_cxr_relation'] == 'proxy_support' for r in output))
        self.assertEqual(output[0]['opacity_ehr_report_relation'], 'proxy_opposition')

    def test_gpu_guard_runs_before_protected_metadata(self):
        args = SimpleNamespace(allow_synthetic_xrv=False)
        with patch.object(module, 'guard', side_effect=RuntimeError('refused')) as guard, \
                patch.object(module, 'metadata') as metadata:
            with self.assertRaisesRegex(RuntimeError, 'refused'):
                module.evaluate(args)
            guard.assert_called_once_with(gpu=True, approved=False)
            metadata.assert_not_called()

    def test_actual_allocation_guard_not_just_env_job(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(Path, 'read_text', return_value='not in that job'):
            with self.assertRaisesRegex(RuntimeError, 'actual_slurm'):
                module.guard()

    def test_gpu_requires_both_explicit_flag_and_visible_device(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(Path, 'read_text', return_value='/slurm/job_123/task_0'):
            with self.assertRaisesRegex(RuntimeError, 'separately_approved'):
                module.guard(gpu=True, approved=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True), \
                patch.object(Path, 'read_text', return_value='/slurm/job_123/task_0'):
            with self.assertRaisesRegex(RuntimeError, 'separately_approved'):
                module.guard(gpu=True, approved=False)

    def test_prepare_guard_runs_before_metadata_and_output(self):
        with patch.object(module, 'guard', side_effect=RuntimeError('refused')), \
                patch.object(module, 'load_preparation') as load, patch.object(module, 'new_atomic_run') as atomic:
            with self.assertRaises(RuntimeError):
                module.prepare(SimpleNamespace())
            load.assert_not_called(); atomic.assert_not_called()

    def test_cli_never_prints_error_message_or_patient_data(self):
        output = io.StringIO()
        with patch.object(module, 'prepare', side_effect=ValueError('secret patient contents')), \
                context_stdout(output):
            status = module.main(['prepare', '--output-root', 'unused', '--run-id', 'fixture'])
        self.assertEqual(status, 2)
        self.assertNotIn('secret', output.getvalue())
        self.assertIn('ValueError', output.getvalue())


def context_stdout(output):
    import contextlib
    return contextlib.redirect_stdout(output)


if __name__ == '__main__':
    unittest.main()

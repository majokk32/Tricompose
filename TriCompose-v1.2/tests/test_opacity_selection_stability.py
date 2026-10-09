"""Invented choices/states/hashes only. No clinical bodies, images or models."""
import copy
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('opacity_selection_fixture', ROOT / 'tools/audit_opacity_selection_stability.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CAPS = (12,)


def fixture():
    original, annotated, trials = {}, [], []
    for case_index in range(2):
        case = f'case_fixture_{case_index}'; ehr = str(case_index) * 64
        for suffix, exact, report in [('a', .8, 'negative'), ('b', .2, 'unknown'), ('c', .2, 'positive')]:
            cid = case + '_' + suffix
            source = {'case_id': case, 'triple_candidate_id': cid, 'ehr_sha256': ehr,
                'cxr_sha256': ('a' if suffix == 'a' else 'b') * 64,
                'report_sha256': suffix * 64, 'source_primary_prefix': '[0, 0, 0, -1]'}
            original[cid] = source
            annotated.append({**source, 'opacity_cached_ehr_state': 'unknown',
                'opacity_cached_report_state': report,
                'opacity_exact_state_0_5': 'positive' if exact >= .5 else 'negative',
                'opacity_same_call_max_state_0_5': 'positive', 'opacity_exact_score': str(exact),
                'opacity_infiltration_score': '0.9', 'opacity_clinical_accuracy': '',
                'opacity_confirmed_faulty_modality': '', 'opacity_selector_used': 'False',
                'opacity_scoring_status': 'scored'})
        for method in module.METHODS:
            settings = [(a, b) for a in range(5) for b in range(5)] if method == module.frozen.NEW_METHOD else \
                       [(a, None) for a in range(5)] if method == 'random' else [(None, None)]
            for acquisition, final in settings:
                suffix = 'a' if method == 'fixed' else 'b' if method == 'targeted_heuristic' else 'c'
                if method == module.frozen.NEW_METHOD:
                    suffix = 'b' if final < 2 else 'c'
                trial = {'case_id': case, 'method': method, 'model_call_budget': 12,
                    'selected_candidate_id': case + '_' + suffix, 'selected_ehr_sha256': ehr,
                    'simulated_model_calls': 4 if method == 'fixed' else 10}
                if method == module.frozen.NEW_METHOD:
                    trial.update(acquisition_seed=acquisition, final_choice_seed=final)
                else:
                    trial['random_seed'] = acquisition
                trials.append(trial)
    return original, annotated, trials


def parent_fixture():
    original, _, _ = fixture(); case = 'case_fixture_0'
    parent = {'method': 'random', 'case_id': case, 'model_call_budget': 12, 'random_seed': 0,
        'selected_candidate_id': case + '_c', 'selected_ehr_sha256': '0' * 64,
        'input_ehr_assessment_scope': 'no_direct_comparable_ehr_facts', 'simulated_model_calls': 10,
        'simulated_calls': {'cxr_generator': 2, 'xrv': 2, 'report_generator': 3, 'chexbert': 3},
        'observed_candidates': 3, 'observed_images': 2, 'terminal_reason': 'budget_exhausted',
        'action_trace': [{'observed_candidate_id': case + '_' + suffix} for suffix in ('a', 'c', 'b')]}
    control = {k: parent[k] for k in ('case_id', 'model_call_budget', 'selected_ehr_sha256',
        'input_ehr_assessment_scope', 'simulated_calls', 'simulated_model_calls', 'observed_candidates', 'observed_images')}
    control.update(method=module.frozen.NEW_METHOD, acquisition_seed=0, final_choice_seed=0,
        selected_candidate_id=case + '_b', source_terminal_reason=parent['terminal_reason'],
        parent_acquisition_trial_sha256=module.frozen.digest(parent), clinical_scores_used_for_choice=False,
        endpoint_used_for_choice=False, clinical_acceptance=False, actual_regeneration_executed=False)
    return parent, control, original


class SelectionStabilityTests(unittest.TestCase):
    def test_both_definitions_keep_exact_same_choice(self):
        _, candidates, trials = fixture(); index = {r['triple_candidate_id']: r for r in candidates}
        before = copy.deepcopy((trials, index)); rows = module.joined_readouts(trials, index)
        self.assertEqual((trials, index), before)
        self.assertEqual(len(rows), 2 * len(trials))
        for a, b in zip(rows[::2], rows[1::2]):
            self.assertEqual(a['selected_candidate_id'], b['selected_candidate_id'])
            self.assertEqual(a['frozen_trial_sha256'], b['frozen_trial_sha256'])
            self.assertEqual(a['simulated_calls'], b['simulated_calls'])

    def test_no_new_selection_or_uniform_draw(self):
        _, candidates, trials = fixture()
        with patch.object(module.frozen, 'uniform_choice', side_effect=AssertionError('must not draw')):
            module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})

    def test_unknown_report_never_negative_or_success(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        unknown = [r for r in rows if r['method'] == 'targeted_heuristic']
        self.assertTrue(all(r['report_state'] == 'unknown' and r['comparable'] == 0 and r['proxy_support'] == 0
                            and r['proxy_opposition'] == 0 and r['not_comparable'] == 1 for r in unknown))

    def test_raw_opposition_not_clinical_error_or_repair(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        self.assertTrue(any(r['proxy_opposition'] for r in rows))
        self.assertTrue(all(r['clinical_accuracy'] is None and r['ehr_cxr_clinical_score'] is None and
                            r['ehr_report_clinical_score'] is None for r in rows))

    def test_no_selected_output_retained_not_dropped(self):
        _, candidates, trials = fixture(); trials[0]['selected_candidate_id'] = None
        rows = module.joined_readouts(trials[:1], {r['triple_candidate_id']: r for r in candidates})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['selected'] == 0 and r['not_comparable'] == 1 and r['simulated_calls'] == 4 for r in rows))

    def test_choice_case_or_fixed_ehr_change_refused(self):
        for field, value in [('case_id', 'foreign'), ('selected_ehr_sha256', 'f' * 64), ('selected_candidate_id', 'foreign')]:
            _, candidates, trials = fixture(); trials[0][field] = value
            with self.assertRaisesRegex(ValueError, 'fixed_chosen_candidate'):
                module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})

    def test_all_25_final_and_5_acquisition_seed_settings_average_within_case(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        means = module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)
        self.assertEqual(len(means), 20)
        control = next(r for r in means if r['method'] == module.frozen.NEW_METHOD and r['head_definition'] == module.HEADS[0])
        self.assertEqual(control['seed_replicates'], 25)
        self.assertEqual(control['comparable'], .6)
        self.assertEqual(control['proxy_opposition'], .6)
        self.assertEqual(control['simulated_calls'], 10)

    def test_missing_case_head_method_or_replicate_refused(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        for bad in (rows[:2], rows[1:]):
            with self.assertRaises(ValueError):
                module.case_means(bad, ['case_fixture_0', 'case_fixture_1'], CAPS)

    def test_duplicate_seed_setting_refused(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        rr = [r for r in rows if r['method'] == 'random' and r['case_id'] == 'case_fixture_0' and r['head_definition'] == module.HEADS[0]]
        rr[1]['acquisition_seed'] = rr[0]['acquisition_seed']
        with self.assertRaisesRegex(ValueError, 'all_unique_seed'):
            module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)

    def test_case_weighting_not_13200_independent_patients(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        means = module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)
        table, pairs = module.comparisons(means, ['case_fixture_0', 'case_fixture_1'], CAPS)
        self.assertEqual(len(table), 10); self.assertEqual(len(pairs), 10)
        self.assertTrue(all(r['fixed_ehr_cases'] == 2 for r in table + pairs))
        self.assertTrue(all(r['mean_selected'] == 1 for r in table))

    def test_no_comparable_is_na_not_perfect(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        means = module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)
        table, _ = module.comparisons(means, ['case_fixture_0', 'case_fixture_1'], CAPS)
        missing = [r for r in table if r['method'] == 'targeted_heuristic']
        self.assertTrue(all(r['conditional_proxy_agreement'] is None and r['mean_comparable'] == 0 for r in missing))

    def test_opposition_disappearing_into_unknown_is_lost_evidence_not_repair(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        means = module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)
        _, pairs = module.comparisons(means, ['case_fixture_0', 'case_fixture_1'], CAPS)
        lost = [r for r in pairs if r['method'] == 'targeted_heuristic']
        self.assertTrue(all(r['opposition_reduction_with_lost_all_comparison_ehr_cases'] == 2 for r in lost))
        self.assertTrue(all(r['clinical_repair_success'] is None for r in lost))

    def test_same_cap_not_equal_expenditure(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        means = module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)
        _, pairs = module.comparisons(means, ['case_fixture_0', 'case_fixture_1'], CAPS)
        static = next(r for r in pairs if r['method'] == 'static_rerank')
        self.assertEqual(static['mean_delta_simulated_calls'], 6)
        self.assertFalse(static['same_acquisition_and_expenditure'])

    def test_only_selector_contrast_requires_matching_per_case_cost(self):
        _, candidates, trials = fixture(); rows = module.joined_readouts(trials, {r['triple_candidate_id']: r for r in candidates})
        means = module.case_means(rows, ['case_fixture_0', 'case_fixture_1'], CAPS)
        _, pairs = module.comparisons(means, ['case_fixture_0', 'case_fixture_1'], CAPS)
        selector = [r for r in pairs if r['reference'] == module.frozen.NEW_METHOD]
        self.assertTrue(all(r['same_acquisition_and_expenditure'] and r['mean_delta_simulated_calls'] == 0 for r in selector))
        next(r for r in means if r['method'] == 'random')['simulated_calls'] += 1
        with self.assertRaisesRegex(ValueError, 'expenditure_must_match'):
            module.comparisons(means, ['case_fixture_0', 'case_fixture_1'], CAPS)

    def test_control_parent_hash_observed_choice_and_cost_bound(self):
        parent, control, candidates = parent_fixture(); before = copy.deepcopy((parent, control, candidates))
        with patch.object(module.frozen, 'uniform_choice', side_effect=AssertionError('no new draw')):
            module.controls_checked([parent], [control], candidates, full=False)
        self.assertEqual((parent, control, candidates), before)

    def test_control_parent_unknown_changed_cost_or_deterministic_scope_refused(self):
        for field, value in [('parent_acquisition_trial_sha256', 'f' * 64), ('simulated_model_calls', 11),
                             ('acquisition_seed', 2), ('clinical_scores_used_for_choice', True)]:
            parent, control, candidates = parent_fixture(); control[field] = value
            with self.assertRaises(ValueError):
                module.controls_checked([parent], [control], candidates, full=False)

    def test_control_foreign_unobserved_or_failed_gate_choice_refused(self):
        parent, control, candidates = parent_fixture(); control['selected_candidate_id'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'observed_only_control'):
            module.controls_checked([parent], [control], candidates, full=False)
        parent, control, candidates = parent_fixture()
        candidates[control['selected_candidate_id']]['source_primary_prefix'] = '[1, 0, 0, -1]'
        with self.assertRaisesRegex(ValueError, 'observed_only_control'):
            module.controls_checked([parent], [control], candidates, full=False)

    def test_control_duplicate_or_incomplete_inventory_refused(self):
        parent, control, candidates = parent_fixture()
        with self.assertRaisesRegex(ValueError, 'unique_all_frozen'):
            module.controls_checked([parent], [control, control], candidates, full=False)
        with self.assertRaisesRegex(ValueError, 'all_10000'):
            module.controls_checked([parent], [control], candidates)

    def test_original_csv_cells_and_id_inventory_preserved(self):
        original, rows, _ = fixture(); before = copy.deepcopy((original, rows))
        index = module.validate_table(original, rows)
        self.assertEqual((original, rows), before)
        self.assertEqual(set(index), set(original))
        rows[0]['report_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'all_original_csv'):
            module.validate_table(original, rows)

    def test_ehr_unknown_clinical_na_and_no_selector_required(self):
        for key, value in [('opacity_cached_ehr_state', 'positive'), ('opacity_clinical_accuracy', '1'),
                           ('opacity_selector_used', 'True'), ('opacity_cached_report_state', 'missing')]:
            original, rows, _ = fixture(); rows[0][key] = value
            with self.assertRaisesRegex(ValueError, 'unqualified_opacity'):
                module.validate_table(original, rows)

    def test_threshold_changed_or_score_nonfinite_refused(self):
        for key, value in [('opacity_exact_score', 'nan'), ('opacity_exact_state_0_5', 'negative'),
                           ('opacity_infiltration_score', '1.2')]:
            original, rows, _ = fixture(); rows[0][key] = value
            with self.assertRaises(ValueError):
                module.validate_table(original, rows)

    def test_failed_image_unknown_preserved(self):
        original, rows, _ = fixture(); rows[0].update(opacity_scoring_status='failed_without_replacement',
            opacity_exact_score='', opacity_infiltration_score='', opacity_exact_state_0_5='unknown',
            opacity_same_call_max_state_0_5='unknown')
        module.validate_table(original, rows)
        rows[0]['opacity_exact_score'] = '0.8'
        with self.assertRaisesRegex(ValueError, 'failed_evidence_retained_unknown'):
            module.validate_table(original, rows)

    def test_allocation_guard_precedes_output_or_input_reads(self):
        with patch.object(module.frozen, 'cpu_guard', side_effect=RuntimeError('not CPU Slurm')), \
                patch.object(module, 'new_atomic_run') as atomic, patch.object(module.opacity, 'source_pins') as sources:
            with self.assertRaises(RuntimeError):
                module.execute(SimpleNamespace())
            atomic.assert_not_called(); sources.assert_not_called()

    def test_public_error_does_not_include_secret_message(self):
        import contextlib
        output = io.StringIO()
        with patch.object(module, 'execute', side_effect=ValueError('secret clinical contents')), contextlib.redirect_stdout(output):
            result = module.main(['--run-id', 'fixture'])
        self.assertEqual(result, 2); self.assertNotIn('secret', output.getvalue())
        self.assertIn('ValueError', output.getvalue())


if __name__ == '__main__':
    unittest.main()

"""Invented selector/counter/trace fixtures, never model or patient artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('score_free_control_fixture', ROOT / 'tools/score_free_random_control.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def candidate(cid='c1', case='invented_case', image='image1', report='maira2', gate=0, cosine=.5):
    row = {'case_id': case, 'triple_candidate_id': cid, 'cxr_candidate_id': image,
        'report_model_id': report, 'ehr_sha256': 'a' * 64,
        'source_primary_prefix': json.dumps([gate, 0, 0, -.5]), 'biovil_raw_cosine': str(cosine)}
    for edge in w.EDGES:
        values = {'known_reference_facts': 1, 'comparable_facts': 1, 'supported_facts': 1,
            'supported_positive': 1, 'supported_negative': 0, 'proxy_opposition_facts': 0}
        row.update({edge + '_' + k: str(v) for k, v in values.items()})
    return row


def trace_row():
    return {'case_id': 'invented_case', 'method': 'random', 'model_call_budget': 8, 'random_seed': 0,
        'selected_ehr_sha256': 'a' * 64, 'selected_candidate_id': 'c1',
        'simulated_calls': {'cxr_generator': 1, 'xrv': 1, 'report_generator': 2, 'chexbert': 2},
        'simulated_model_calls': 6, 'observed_candidates': 2, 'observed_images': 1,
        'terminal_reason': 'invented_terminal', 'input_ehr_assessment_scope': 'explicit_fact_proxy',
        'clinical_acceptance': False, 'clinical_fault_confirmed': False, 'actual_regeneration_executed': False,
        'action_trace': [{'step': 0, 'request_slot': ['sana', 0, 'maira2'], 'observed_candidate_id': 'c1',
            'charged_model_calls': 4, 'cumulative_model_calls': 4},
            {'step': 1, 'request_slot': ['sana', 0, 'cxrmate_single'], 'observed_candidate_id': 'c2',
                'charged_model_calls': 2, 'cumulative_model_calls': 6}]}


class ScoreFreeRandomControlTests(unittest.TestCase):
    def test_selector_accepts_only_ids_and_artifact_gates(self):
        valid = [{'candidate_id': 'a', 'artifact_gate_pass': True}]
        self.assertEqual(w.uniform_choice('case', 4, 0, 0, valid), 'a')
        for field in ('biovil_raw_cosine', 'proxy_support', 'rank', 'model', 'runtime'):
            visible = [{**valid[0], field: 1}]
            with self.assertRaises(ValueError):
                w.uniform_choice('case', 4, 0, 0, visible)

    def test_no_eligible_candidates_remain_unavailable(self):
        self.assertIsNone(w.uniform_choice('case', 4, 0, 0, []))
        self.assertIsNone(w.uniform_choice('case', 4, 0, 0,
            [{'candidate_id': 'a', 'artifact_gate_pass': False}]))

    def test_candidate_order_irrelevant_and_selection_deterministic(self):
        values = [{'candidate_id': s, 'artifact_gate_pass': True} for s in ('a', 'b', 'c')]
        before = copy.deepcopy(values)
        a = w.uniform_choice('case', 12, 2, 1, values)
        self.assertEqual(a, w.uniform_choice('case', 12, 2, 1, list(reversed(values))))
        self.assertEqual(values, before)

    def test_unseen_and_failed_gate_candidates_never_selected(self):
        values = [{'candidate_id': 'a', 'artifact_gate_pass': True},
            {'candidate_id': 'b', 'artifact_gate_pass': False}]
        for seed in range(30):
            self.assertEqual(w.uniform_choice('case', 30, 0, seed, values), 'a')

    def test_duplicate_ids_nonboolean_gates_or_bad_seeds_refused(self):
        good = [{'candidate_id': 'a', 'artifact_gate_pass': True}]
        for values in (good * 2, [{'candidate_id': 'a', 'artifact_gate_pass': 1}]):
            with self.assertRaises(ValueError):
                w.uniform_choice('case', 4, 0, 0, values)
        for cap, acq, final in ((True, 0, 0), (4, -1, 0), (4, 0, True)):
            with self.assertRaises(ValueError):
                w.uniform_choice('case', cap, acq, final, good)

    def test_all_seed_settings_preserved_without_best_seed(self):
        row = trace_row()
        candidates = {'c1': candidate(), 'c2': candidate('c2', report='cxrmate_single')}
        controls = w.make_controls([row], candidates)
        self.assertEqual([r['final_choice_seed'] for r in controls], list(w.SEEDS))
        self.assertEqual(len(controls), 5)
        for result in controls:
            self.assertEqual(result['simulated_calls'], row['simulated_calls'])
            self.assertEqual(result['simulated_model_calls'], row['simulated_model_calls'])
            self.assertEqual(result['parent_acquisition_trial_sha256'], w.digest(row))
            self.assertFalse(result['clinical_scores_used_for_choice'])
            self.assertFalse(result['endpoint_used_for_choice'])

    def test_scores_can_change_without_changing_choices(self):
        row = trace_row()
        first = {'c1': candidate(cosine=.9), 'c2': candidate('c2', report='cxrmate_single', cosine=-.9)}
        second = copy.deepcopy(first)
        second['c1']['biovil_raw_cosine'] = '-.9'
        second['c2']['biovil_raw_cosine'] = '.9'
        second['c1']['source_primary_prefix'] = '[0,99,-99,null]'
        second['c2']['source_primary_prefix'] = '[0,0,0,-1]'
        self.assertEqual(w.make_controls([row], first), w.make_controls([row], second))

    def test_make_choices_does_not_parse_endpoint_or_clinical_readouts(self):
        row = trace_row()
        candidates = {'c1': candidate(), 'c2': candidate('c2', report='cxrmate_single')}
        with patch.object(w, 'candidate_readout', side_effect=AssertionError('no scores while choosing')):
            self.assertEqual(len(w.make_controls([row], candidates)), 5)

    def test_shared_image_cost_charged_once(self):
        candidates = {'c1': candidate(), 'c2': candidate('c2', report='cxrmate_single')}
        self.assertEqual(set(w.validate_trace(trace_row(), candidates)), {'c1', 'c2'})

    def test_wrong_charge_duplicate_trace_or_budget_overrun_refused(self):
        candidates = {'c1': candidate(), 'c2': candidate('c2', report='cxrmate_single')}
        for mode in ('charge', 'duplicate', 'budget', 'count'):
            row = trace_row()
            if mode == 'charge':
                row['action_trace'][1]['charged_model_calls'] = 4
            elif mode == 'duplicate':
                row['action_trace'][1]['observed_candidate_id'] = 'c1'
            elif mode == 'budget':
                row['model_call_budget'] = 4
            else:
                row['observed_candidates'] = 3
            with self.assertRaises(ValueError):
                w.validate_trace(row, candidates)

    def test_cross_case_or_changed_ehr_refused(self):
        for field, value in (('case_id', 'other'), ('ehr_sha256', 'b' * 64)):
            candidates = {'c1': candidate(), 'c2': candidate('c2', report='cxrmate_single')}
            candidates['c2'][field] = value
            with self.assertRaises(ValueError):
                w.validate_trace(trace_row(), candidates)

    def test_unknown_ehr_edges_are_null_not_perfect(self):
        row = candidate()
        for edge in ('ehr_cxr', 'ehr_report'):
            for field in w.COUNT_FIELDS:
                row[edge + '_' + field] = '0'
        result = w.candidate_readout(row)
        self.assertIsNone(result['ehr_cxr_support_over_known'])
        self.assertIsNone(result['ehr_report_opposition_over_known'])
        self.assertIsNone(result['ehr_cxr_coverage_over_known'])
        self.assertEqual(result['cxr_report_support_over_known'], 1.)

    def test_four_state_counter_arithmetic_and_bounds_refused(self):
        for field, value in (('comparable_facts', '2'), ('supported_positive', '0'),
                ('proxy_opposition_facts', '1'), ('known_reference_facts', '15')):
            row = candidate()
            row['ehr_cxr_' + field] = value
            with self.assertRaises(ValueError):
                w.candidate_readout(row)

    def test_gate_is_not_a_clinical_prefix_score(self):
        row = candidate()
        row['source_primary_prefix'] = '[0,99999,-99999,null]'
        self.assertTrue(w.artifact_gate(row))
        row['source_primary_prefix'] = '[1,0,0,0]'
        self.assertFalse(w.artifact_gate(row))

    def test_invalid_endpoint_not_probability_or_zero(self):
        for value in ('nan', 'inf', '1.1'):
            with self.assertRaises(ValueError):
                w.candidate_readout(candidate(cosine=value))
        row = candidate()
        row['biovil_raw_cosine'] = ''
        self.assertIsNone(w.candidate_readout(row)['biovil_raw_cosine'])

    def test_seed_replicates_averaged_within_case(self):
        row = trace_row()
        candidates = {'c1': candidate(cosine=.8), 'c2': candidate('c2', report='cxrmate_single', cosine=.2)}
        outcomes = []
        for i in range(5):
            result = copy.deepcopy(row);result['random_seed'] = i
            result['selected_candidate_id'] = 'c1' if i < 2 else 'c2';outcomes.append(result)
        means = w.case_means(outcomes, candidates)
        self.assertEqual(len(means), 1)
        self.assertAlmostEqual(means[0]['biovil_raw_cosine'], .44)
        self.assertEqual(means[0]['seed_replicates'], 5)

    def test_missing_seed_and_missing_endpoint_not_cherry_picked(self):
        row = trace_row()
        candidates = {'c1': candidate(), 'c2': candidate('c2', report='cxrmate_single')}
        with self.assertRaises(ValueError):
            w.case_means([row], candidates)
        outcomes = [copy.deepcopy(row) for _ in range(5)]
        outcomes[0]['selected_candidate_id'] = None
        result = w.case_means(outcomes, candidates)[0]
        self.assertEqual(result['selected_seed_replicates'], 4)
        self.assertIsNone(result['biovil_raw_cosine'])
        self.assertEqual(result['mean_simulated_calls'], 6)

    def test_guard_before_loading_inputs(self):
        with patch.dict(w.os.environ, {}, clear=True), patch.object(w, 'load') as load:
            with self.assertRaises(RuntimeError):
                w.execute(Path('/invented'), 'new')
            load.assert_not_called()

    def test_forged_cpu_or_gpu_allocation_rejected(self):
        with patch.dict(w.os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(w.Path, 'read_text', return_value='/other'):
            with self.assertRaises(RuntimeError):
                w.cpu_guard()
        with patch.dict(w.os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_GPUS': '0'}, clear=True), \
                patch.object(w.Path, 'read_text', return_value='/job_123/step_batch'):
            with self.assertRaises(RuntimeError):
                w.cpu_guard()


if __name__ == '__main__':
    unittest.main()

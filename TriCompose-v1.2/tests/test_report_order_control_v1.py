"""Invented fixtures for schedule controls, unchanged guards and missingness."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_report_order_v1 as m
from test_probe_repair_v1 import observation


def fixture_grid():
    reports = ['maira', 'cxrmate', 'llava', 'chexagent']
    grid = {('sana', 0, r): observation(report=r, number=n,
        text='positive' if r == 'chexagent' else 'negative') for n, r in enumerate(reports)}
    return grid, {'image_order': [['sana', 0]], 'report_order': reports}


def run_fixture(cap, order=None, grid=None):
    original, policy = fixture_grid()
    with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
        return m.replay(grid if grid is not None else original, policy, cap,
            'fixed_report_order_001', order or policy['report_order'])


class ReportOrderControlTests(unittest.TestCase):
    def test_exact_six_complete_orders_same_initial(self):
        protocol = json.loads(m.PROTOCOL.read_text())
        policy = {'report_order': protocol['orders'][0], 'image_order': [protocol['initial_image']]}
        orders = m.declared_orders(policy, protocol)
        self.assertEqual(len(orders), 6)
        self.assertEqual(len({tuple(order) for _, order in orders}), 6)
        self.assertTrue(all(order[0] == 'maira2' for _, order in orders))

    def test_missing_duplicate_reordered_protocol_not_accepted(self):
        original = json.loads(m.PROTOCOL.read_text())
        policy = {'report_order': original['orders'][0], 'image_order': [original['initial_image']]}
        for mode in ('missing', 'duplicate', 'reordered'):
            value = deepcopy(original)
            if mode == 'missing': value['orders'].pop()
            elif mode == 'duplicate': value['orders'][-1] = value['orders'][0]
            else: value['orders'].reverse()
            with self.assertRaises(ValueError): m.declared_orders(policy, value)

    def test_secondary_training_or_clinical_claim_cannot_enter_protocol(self):
        original = json.loads(m.PROTOCOL.read_text())
        policy = {'report_order': original['orders'][0], 'image_order': [original['initial_image']]}
        for field in ('choose_order_using_secondary', 'choose_order_on_final_test', 'training_allowed',
                'clinical_qualified', 'image_regeneration_allowed', 'actual_regeneration_executed'):
            value = deepcopy(original); value[field] = True
            with self.assertRaises(ValueError): m.declared_orders(policy, value)

    def test_cap_four_has_no_probes(self):
        result = run_fixture(4)
        self.assertEqual(result['selected_candidate_id'], 'triple_0')
        self.assertEqual(result['history'], [])
        self.assertEqual(result['simulated_calls'], 4)

    def test_reordering_changes_acquisition_not_commit_rule(self):
        source = run_fixture(8)
        changed = run_fixture(8, ['maira', 'chexagent', 'cxrmate', 'llava'])
        self.assertEqual(source['selected_candidate_id'], 'triple_0')
        self.assertEqual(changed['selected_candidate_id'], 'triple_3')
        self.assertEqual(source['simulated_calls'], changed['simulated_calls'])
        self.assertEqual(source['same_commit_guard_version'], changed['same_commit_guard_version'])

    def test_all_reports_affordable_at_cap_twelve(self):
        result = run_fixture(12)
        self.assertEqual(result['selected_candidate_id'], 'triple_3')
        self.assertEqual(len(result['history']), 3)
        self.assertEqual(result['simulated_calls'], 10)

    def test_missing_worker_slot_still_charged(self):
        grid, _ = fixture_grid(); del grid['sana', 0, 'cxrmate']
        result = run_fixture(6, grid=grid)
        self.assertEqual(result['simulated_calls'], 6)
        self.assertEqual(result['history'][0]['failure_type'], 'unavailable_cache_slot')
        self.assertFalse(result['history'][0]['accepted'])

    def test_quality_loss_veto_not_relaxed_for_earlier_expert(self):
        grid, _ = fixture_grid(); grid['sana', 0, 'chexagent']['quality']['report_structure_quality_score_0_1'] = .1
        result = run_fixture(8, ['maira', 'chexagent', 'cxrmate', 'llava'], grid=grid)
        self.assertEqual(result['selected_candidate_id'], 'triple_0')
        self.assertIn('structure_proxy_regressed', result['history'][0]['credit']['reasons'])

    def test_ehr_and_image_not_mutated(self):
        grid, _ = fixture_grid(); before = deepcopy(grid)
        run_fixture(12, grid=grid)
        self.assertEqual(grid, before)

    def test_secondary_payload_rejected_by_controller(self):
        grid, _ = fixture_grid(); grid['sana', 0, 'maira']['biovil_raw_cosine'] = .9
        with self.assertRaises(ValueError): run_fixture(8, grid=grid)

    def test_complete_group_mapping(self):
        rows = [{'case_id': 'a', 'conditioning_group_id': 'g1'}, {'case_id': 'b', 'conditioning_group_id': 'g2'}]
        self.assertEqual(m.groups_mapping(rows, {'a', 'b'}, 2), {'a': 'g1', 'b': 'g2'})

    def test_duplicate_missing_and_wrong_group_count_rejected(self):
        rows = [{'case_id': 'a', 'conditioning_group_id': 'g1'}, {'case_id': 'b', 'conditioning_group_id': 'g2'}]
        for bad in (rows[:1], rows + [rows[0]], [dict(rows[0], case_id='z'), rows[1]]):
            with self.assertRaises(ValueError): m.groups_mapping(bad, {'a', 'b'}, 2)
        with self.assertRaises(ValueError): m.groups_mapping(rows, {'a', 'b'}, 1)

    def test_equal_group_means_not_case_means(self):
        rows = [{'case_id': cid, 'method': 'm', 'model_call_budget': 8, 'replicates': 1, 'score': value}
            for cid, value in (('a', 0), ('b', 0), ('c', 1))]
        result = m.grouped_metrics(rows, {'a': 'g1', 'b': 'g1', 'c': 'g2'})[0]
        self.assertAlmostEqual(result['case_weighted_mean'], 1/3)
        self.assertEqual(result['equal_conditioning_mean'], .5)
        self.assertEqual(result['available_ehr_cases'], 3)

    def test_missing_ehr_metric_stays_na_with_explicit_denominators(self):
        rows = [{'case_id': 'a', 'method': 'm', 'model_call_budget': 8, 'replicates': 1, 'score': None}]
        result = m.grouped_metrics(rows, {'a': 'g1'})[0]
        self.assertIsNone(result['case_weighted_mean']); self.assertIsNone(result['equal_conditioning_mean'])
        self.assertEqual(result['attempted_ehr_cases'], 1); self.assertEqual(result['available_ehr_cases'], 0)

    def test_exact_trial_grid_rejects_duplicates_and_missing(self):
        rows = [dict(run_fixture(cap), model_call_budget=cap) for cap in m.b.CAPS]
        m.exact_grid(rows, {'case_000'}, ['fixed_report_order_001'])
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError): m.exact_grid(bad, {'case_000'}, ['fixed_report_order_001'])

    def test_bool_or_overbudget_calls_not_valid(self):
        rows = [dict(run_fixture(cap), model_call_budget=cap) for cap in m.b.CAPS]
        for value in (True, 100):
            changed = deepcopy(rows); changed[0]['simulated_calls'] = value
            with self.assertRaises(ValueError): m.exact_grid(changed, {'case_000'}, ['fixed_report_order_001'])

    def test_runtime_guard_precedes_inputs_and_creation(self):
        with patch.object(m.b, 'cpu_guard', side_effect=RuntimeError('invented')), \
                patch.object(m.b, 'load_primary') as read, patch.object(m, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): m.run(None)
            read.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()

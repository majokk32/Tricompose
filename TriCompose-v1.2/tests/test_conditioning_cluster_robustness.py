"""Invented identities and numeric fixtures only."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'tools/check_conditioning_cluster_robustness.py'
spec = importlib.util.spec_from_file_location('test_conditioning_robustness', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def index_fixture():
    rows = []
    for case in range(4):
        condition = (0, 0, 1, 2)[case]
        for model_index, model in enumerate(worker.MODELS):
            for expert_index, expert in enumerate(worker.EXPERTS):
                rows.append({'case_id': f'case_{case:03d}', 'cxr_model_id': model, 'report_model_id': expert,
                    'cxr_seed': '0', 'triple_candidate_id': f'fixture_{case}_{model_index}_{expert_index}',
                    'ehr_sha256': f'{case+1:064x}', 'ehr_facts_sha256': f'{case+100:064x}',
                    'prompt_sha256': f'{condition+200+model_index*10:064x}',
                    'cxr_sha256': f'{condition+500+model_index*10:064x}', 'ignored_outcome': None})
    return rows


class ClusterRobustnessTests(unittest.TestCase):
    def test_metadata_grouping_keeps_all_cases_and_duplicate_condition(self):
        rows = index_fixture()
        mapping, inventory = worker.groups(rows, expected_cases=4)
        self.assertEqual(len(mapping), 4)
        self.assertEqual(inventory['exact_conditioning_groups'], 3)
        self.assertEqual(inventory['distinct_ehr_hashes'], 4)
        self.assertEqual(mapping[0]['conditioning_group_id'], mapping[1]['conditioning_group_id'])
        self.assertEqual(inventory['group_size_histogram'], {1: 2, 2: 1})

    def test_grouping_order_deterministic_outcomes_unused(self):
        rows = index_fixture()
        expected = worker.groups(rows, expected_cases=4)
        for row in rows:
            row['ignored_outcome'] = object()
        self.assertEqual(worker.groups(list(reversed(rows)), expected_cases=4), expected)

    def test_shared_prompt_or_grid_mismatch_refused(self):
        for mutation in ('prompt', 'slot', 'hash', 'seed'):
            rows = index_fixture()
            if mutation == 'prompt': rows[0]['prompt_sha256'] = 'b'*64
            if mutation == 'slot': rows[0]['report_model_id'] = rows[1]['report_model_id']
            if mutation == 'hash': rows[0]['ehr_sha256'] = 'broken'
            if mutation == 'seed': rows[0]['cxr_seed'] = '1'
            with self.assertRaises(ValueError):
                worker.groups(rows, expected_cases=4)

    def test_estimands_distinguish_case_and_equal_condition_weighting(self):
        result, _ = worker.aggregate([('a', 0), ('a', 0), ('b', 1), ('c', 1)], 4)
        self.assertEqual(result['case_weighted_mean'], .5)
        self.assertAlmostEqual(result['equal_conditioning_mean'], 2/3)
        self.assertEqual(result['available_conditioning_groups'], 3)

    def test_deterministic_whole_group_bootstrap(self):
        pairs = [('a', 0), ('a', 2), ('b', 4), ('c', 8)]
        result = worker.paired_interval(pairs, 4, draws=100)
        self.assertEqual(result, worker.paired_interval(pairs, 4, draws=100))
        self.assertEqual(result['bootstrap_resamples_used'], 100)
        self.assertEqual(result['case_weighted_mean'], 3.5)
        self.assertAlmostEqual(result['equal_conditioning_mean'], 13/3)

    def test_no_available_or_too_few_groups_stay_unavailable(self):
        for pairs in ([], [('a', 1)], [('a', 1), ('a', 1), ('b', 2)]):
            result = worker.paired_interval(pairs, 4)
            self.assertIsNone(result['case_weighted_interval_95'])
            self.assertEqual(result['bootstrap_resamples_used'], 0)
            self.assertEqual(result['available_ehr_cases'], len(pairs))
        self.assertIsNone(worker.paired_interval([], 4)['case_weighted_mean'])

    def test_zero_effect_not_silently_dropped(self):
        result = worker.paired_interval([('a', 0), ('b', 0), ('c', 0)], 3, draws=30)
        self.assertEqual(result['case_weighted_interval_95'], [0, 0])
        self.assertTrue(result['constant_observed_values'])

    def test_number_nan_bool_rejected_and_missing_stays_null(self):
        for value in (True, float('inf'), 'nan'):
            with self.assertRaises(ValueError): worker.number(value)
        self.assertIsNone(worker.number(''))
        self.assertEqual(worker.number(0), 0)

    def test_all_methods_caps_and_seed_averages_required(self):
        mapping, _ = worker.groups(index_fixture(), expected_cases=4)
        rows = []
        for item in mapping:
            for method in worker.METHODS:
                for cap in worker.CAPS:
                    row = {'case_id': item['case_id'], 'method': method, 'model_call_budget': str(cap),
                        'seed_replicates': '25' if method == worker.METHODS[-1] else '5' if method == 'random' else '1'}
                    row.update({metric: '0' for metric in worker.METRICS})
                    row['ehr_cxr_support_over_known'] = row['ehr_cxr_opposition_over_known'] = row['ehr_cxr_coverage_over_known'] = ''
                    rows.append(row)
        parsed = worker.parse_means(rows, mapping)
        self.assertEqual(len(parsed), 100)
        self.assertIsNone(next(iter(parsed.values()))['ehr_cxr_support_over_known'])
        with self.assertRaises(ValueError): worker.parse_means(rows[:-1], mapping)
        changed = copy.deepcopy(rows); changed[0]['seed_replicates'] = '25'
        with self.assertRaises(ValueError): worker.parse_means(changed, mapping)

    def test_no_slurm_refused_before_any_source_read(self):
        with patch.object(worker.guard, 'guard', side_effect=RuntimeError('no_slurm')), \
                patch.object(worker.guard, 'metadata') as read:
            with self.assertRaises(RuntimeError): worker.source_pins()
            read.assert_not_called()

    def test_proxies_do_not_promote_or_authorize_repair(self):
        for flag in ('clinical_qualified', 'primary_metric_eligible', 'selection_changed', 'regeneration_authorized'):
            self.assertIs(worker.POLICY[flag], False)


if __name__ == '__main__':
    unittest.main()

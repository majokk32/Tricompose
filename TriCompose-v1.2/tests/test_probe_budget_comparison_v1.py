"""Authored scalar fixtures; no protected generation or clinical data read."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_probe_budgets_v1 as compare


def fixture():
    groups = [{'case_id': 'case_000', 'conditioning_group_id': 'group_000'}]
    rows = []
    for method in compare.METHODS:
        for cap in compare.CAPS:
            row = {'case_id': 'case_000', 'method': method, 'model_call_budget': str(cap),
                'replicates': 25 if method == compare.OLD_METHODS[-1] else 5 if method == 'random' else 1,
                'mean_simulated_calls': '4', 'biovil_raw_cosine': '.5', 'cxr_report_known': '2',
                'cxr_report_positive_support': '1'}
            for metric in compare.grouped.METRICS:
                if metric not in row:
                    row[metric.replace('coverage_over_known', 'comparable_over_known')] = (
                        '.5' if 'support_' in metric or 'coverage_' in metric else '1' if metric == 'artifact_valid' else '0')
            rows.append(row)
    return rows, groups


class BudgetComparisonTests(unittest.TestCase):
    def test_complete_grid_is_parsed_without_mutation(self):
        args = fixture(); original = deepcopy(args)
        r = compare.parse(*args, expected_cases=1)
        self.assertEqual(len(r), 35); self.assertEqual(args, original)
        self.assertEqual(r['case_000', 'fixed', 4]['cxr_report_positive_support_over_known'], .5)

    def test_unknown_edges_remain_unknown_not_zero(self):
        rows, groups = fixture()
        for k in ('ehr_cxr_support_over_known', 'ehr_cxr_opposition_over_known', 'ehr_cxr_comparable_over_known'):
            rows[0][k] = ''
        r = compare.parse(rows, groups, expected_cases=1)
        self.assertIsNone(r['case_000', 'fixed', 4]['ehr_cxr_support_over_known'])

    def test_duplicates_and_missing_rows_fail(self):
        for delta in (-1, 1):
            rows, groups = fixture()
            rows = rows[:-1] if delta < 0 else rows + [deepcopy(rows[0])]
            with self.assertRaises(ValueError): compare.parse(rows, groups, expected_cases=1)

    def test_wrong_replicate_count_fails(self):
        rows, groups = fixture(); rows[0]['replicates'] = 5
        with self.assertRaises(ValueError): compare.parse(rows, groups, expected_cases=1)

    def test_over_budget_nonfinite_and_invalid_coverage_fail(self):
        for field, value in (('mean_simulated_calls', '5'), ('biovil_raw_cosine', 'nan'),
                ('cxr_report_comparable_over_known', '.9'), ('cxr_report_positive_support', '3')):
            rows, groups = fixture(); rows[0][field] = value
            with self.assertRaises(ValueError): compare.parse(rows, groups, expected_cases=1)

    def test_same_cap_is_not_same_calls_or_real_gpu_cost(self):
        args = fixture(); means = compare.parse(*args, expected_cases=1)
        methods, contrasts = compare.analyze(means, args[1])
        self.assertEqual(len(methods), 420); self.assertEqual(len(contrasts), 240)
        for row in contrasts:
            self.assertTrue(row['same_budget_cap'])
            self.assertFalse(row['equal_realized_calls_claimed'])
            self.assertFalse(row['measured_gpu_savings_claimed'])
            self.assertIsNone(row['case_weighted_interval_95'])

    def test_grouped_estimands_preserve_duplicate_conditionings(self):
        r, _ = compare.grouped.aggregate([('a', 0), ('a', 0), ('b', 1)], 3)
        self.assertAlmostEqual(r['case_weighted_mean'], 1/3)
        self.assertEqual(r['equal_conditioning_mean'], .5)

    def test_guard_precedes_reads_and_writes(self):
        with patch.object(compare, 'cpu_guard', side_effect=RuntimeError('invented_guard')), \
                patch.object(compare, 'checked') as read, patch.object(compare, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): compare.run(None)
            read.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()

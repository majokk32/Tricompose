"""Invented, nonclinical numeric fixtures; no protected input reads."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from test_probe_repair_v1 import observation

spec = importlib.util.spec_from_file_location('probe_benchmark',
    Path(__file__).resolve().parents[1]/'tools/benchmark_probe_repair_v1.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def baseline_rows(uniform=False):
    methods = ('random_acquisition_score_free_final',) if uniform else b.OLD_METHODS[:4]
    rows = []
    for cap in b.CAPS:
        for method in methods:
            seeds = [(a, c) for a in range(5) for c in range(5)] if uniform else \
                [(s, None) for s in range(5)] if method == 'random' else [(None, None)]
            for a, c in seeds:
                r = {'case_id': 'case_000', 'method': method, 'model_call_budget': cap,
                    'simulated_model_calls': 4, 'actual_regeneration_executed': False,
                    'clinical_acceptance': False, 'selected_candidate_id': 'triple_0',
                    'selected_ehr_sha256': '1'*64}
                if uniform: r.update(acquisition_seed=a, final_choice_seed=c)
                else: r.update(random_seed=a)
                rows.append(r)
    return rows


class BaselineGridTests(unittest.TestCase):
    def validate(self, rows, uniform=False):
        b.validate_baseline_grid(rows, {'case_000': {}}, {'triple_0': observation()}, uniform=uniform)

    def test_complete_exact_case_budget_seed_grid(self):
        self.validate(baseline_rows())
        self.validate(baseline_rows(True), True)

    def test_duplicate_cannot_replace_missing_row_even_when_count_matches(self):
        for uniform in (False, True):
            rows = baseline_rows(uniform); rows[-1] = deepcopy(rows[0])
            with self.assertRaises(ValueError): self.validate(rows, uniform)

    def test_missing_or_extra_seed_rejected(self):
        for uniform in (False, True):
            rows = baseline_rows(uniform)
            with self.assertRaises(ValueError): self.validate(rows[:-1], uniform)
            rows[-1]['acquisition_seed' if uniform else 'random_seed'] = 10
            with self.assertRaises(ValueError): self.validate(rows, uniform)

    def test_no_cross_case_or_changed_ehr_selection(self):
        for field, value in [('case_id', 'case_001'), ('selected_ehr_sha256', '9'*64),
                             ('selected_candidate_id', 'foreign')]:
            rows = baseline_rows(); rows[0][field] = value
            with self.assertRaises(ValueError): self.validate(rows)

    def test_proxy_baseline_cannot_claim_clinical_or_executed_success(self):
        for field in ('actual_regeneration_executed', 'clinical_acceptance'):
            rows = baseline_rows(); rows[0][field] = True
            with self.assertRaises(ValueError): self.validate(rows)

    def test_failed_attempt_not_free_or_over_budget(self):
        for calls in (-1, True, 4.5, 5):
            rows = baseline_rows(); rows[0]['simulated_model_calls'] = calls
            with self.assertRaises(ValueError): self.validate(rows)


class AggregationTests(unittest.TestCase):
    def trial(self, candidate, case='case_000', calls=4):
        return {'selected_candidate_id': candidate, 'case_id': case, 'method': 'fixture',
            'model_call_budget': 8, 'calls': calls}

    def test_seed_replicates_averaged_within_ehr_before_cohort(self):
        a, c = observation(), observation(number=1)
        c['case_id'] = 'case_001'
        trials = [self.trial('triple_0') for _ in range(5)] + [self.trial('triple_1', 'case_001', 8)]
        cases, table = b.aggregate(trials, {'triple_0': a, 'triple_1': c}, {'triple_0': 0, 'triple_1': 1})
        self.assertEqual(len(cases), 2)
        self.assertEqual(table[0]['biovil_raw_cosine_mean'], .5)
        self.assertEqual(table[0]['mean_simulated_calls'], 6)

    def test_missing_secondary_stays_na_not_zero(self):
        _, table = b.aggregate([self.trial('triple_0')], {'triple_0': observation()}, {'triple_0': None})
        self.assertIsNone(table[0]['biovil_raw_cosine_mean'])
        self.assertEqual(table[0]['biovil_raw_cosine_available_ehr_cases'], 0)

    def test_unknown_ehr_has_zero_known_and_na_ratio(self):
        row = b.numeric_readout(observation(ehr='unknown'), .1)
        self.assertEqual(row['ehr_cxr_known'], 0)
        self.assertIsNone(row['ehr_cxr_support_over_known'])

    def test_no_selected_candidate_keeps_case_and_selection_failure(self):
        cases, table = b.aggregate([self.trial(None)], {'triple_0': observation()}, {'triple_0': .1})
        self.assertEqual(table[0]['fixed_ehr_cases'], 1)
        self.assertEqual(table[0]['selected_fraction'], 0)
        self.assertIsNone(cases[0]['artifact_valid'])
        self.assertIsNone(table[0]['clinical_accuracy'])

    def test_partial_secondary_replicates_cannot_drop_missing_value(self):
        trials = [self.trial('triple_0'), self.trial('triple_1')]
        cases, _ = b.aggregate(trials, {'triple_0': observation(), 'triple_1': observation(number=1)},
            {'triple_0': .1, 'triple_1': None})
        self.assertIsNone(cases[0]['biovil_raw_cosine'])

    def test_activity_distinguishes_cases_from_transitions(self):
        outcomes = [{'method': 'observed_probe_repair', 'model_call_budget': 8,
            'history': [{'accepted': True, 'action': 'report_probe', 'failure_type': None}]*2}]
        row = b.probe_activity(outcomes)[1]
        self.assertEqual(row['cases_with_accepted_proxy_transition'], 1)
        self.assertEqual(row['accepted_proxy_transitions'], 2)
        self.assertIsNone(row['clinical_repair_success'])


if __name__ == '__main__':
    unittest.main()

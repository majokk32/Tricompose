"""Invented counts/scores only; no models, datasets or patient text."""
import copy
import math
import unittest

from tricompose_v12 import report_metric_alignment as alignment


def fixture(n=5, benchmark='radevalx-1.0.0'):
    width = alignment.BENCHMARKS[benchmark]
    predictions = {'schema_version': alignment.VERSION + '-predictions', 'benchmark': benchmark,
        'metric_definitions': {'example_quality': {'orientation': 'higher_is_better', 'provenance': 'published_cached'}},
        'records': []}
    references = {'schema_version': alignment.VERSION + '-references', 'benchmark': benchmark,
        'reference_policy': 'two_reader_consensus' if width == 8 else 'mean_of_all_released_readers', 'records': []}
    for i in range(n):
        identity = {'item_id': 'pair_%04d' % i, 'source_group_id': 'group_%04d' % i}
        predictions['records'].append({**identity, 'scores': {'example_quality': {'status': 'complete', 'value': n-i}}})
        references['records'].append({**identity, 'errors': {'clinically_significant': [i]+[0]*(width-1),
            'clinically_insignificant': [0]*width}})
    return predictions, references


class AlignmentTests(unittest.TestCase):
    def run_fixture(self, p=None, r=None):
        if p is None:
            p, r = fixture()
        return alignment.evaluate(p, r, bootstrap_resamples=0)

    def primary(self, result):
        return result['metrics']['example_quality']['outcomes']['clinically_significant_total']

    def test_positive_alignment_for_quality_decreasing_with_errors(self):
        result = self.run_fixture()
        self.assertAlmostEqual(self.primary(result)['spearman'], 1)
        self.assertAlmostEqual(self.primary(result)['kendall_tau_b'], 1)
        self.assertEqual(result['all_attempted_rows'], 5)

    def test_lower_is_better_orientation_is_explicit(self):
        p, r = fixture()
        p['metric_definitions']['example_quality']['orientation'] = 'lower_is_better'
        for i, row in enumerate(p['records']):
            row['scores']['example_quality']['value'] = i
        self.assertAlmostEqual(self.primary(self.run_fixture(p, r))['spearman'], 1)

    def test_wrong_direction_is_not_automatically_flipped(self):
        p, r = fixture()
        for i, row in enumerate(p['records']):
            row['scores']['example_quality']['value'] = i
        self.assertAlmostEqual(self.primary(self.run_fixture(p, r))['spearman'], -1)

    def test_tie_aware_ranks_and_tau_b(self):
        result = alignment.correlation([(1,1), (2,3), (2,2), (3,4)])
        self.assertEqual(alignment.ranks([1,2,2,3]), [1,2.5,2.5,4])
        self.assertAlmostEqual(result['spearman'], math.sqrt(.9))
        self.assertAlmostEqual(result['kendall_tau_b'], 5/math.sqrt(30))
        self.assertEqual(result['kendall_counts']['left_only_ties'], 1)

    def test_joint_ties_not_double_counted(self):
        result = alignment.correlation([(1,1), (1,1), (2,2), (3,3)])
        self.assertEqual(result['kendall_tau_b'], 1)
        self.assertEqual(result['kendall_counts']['concordant_pairs'], 5)

    def test_missing_score_is_null_and_retains_denominator(self):
        p, r = fixture()
        p['records'][0]['scores']['example_quality'] = {'status': 'failed_unavailable', 'value': None}
        result = self.run_fixture(p, r)
        self.assertEqual(result['all_attempted_rows'], 5)
        self.assertEqual(self.primary(result)['paired_rows'], 4)
        self.assertEqual(self.primary(result)['paired_coverage'], .8)
        self.assertEqual(result['metrics']['example_quality']['score_status_counts']['failed_unavailable'], 1)

    def test_missing_annotation_is_not_zero(self):
        p, r = fixture()
        r['records'][0]['errors']['clinically_significant'][0] = None
        result = self.run_fixture(p, r)
        self.assertEqual(self.primary(result)['paired_rows'], 4)
        unaffected = result['metrics']['example_quality']['outcomes']['clinically_significant_category_2']
        self.assertEqual(unaffected['paired_rows'], 5)
        self.assertEqual(unaffected['status'], 'constant_score_or_reference')

    def test_zero_error_category_has_no_fabricated_correlation(self):
        result = self.run_fixture()['metrics']['example_quality']['outcomes']['clinically_insignificant_total']
        self.assertIsNone(result['spearman'])
        self.assertIsNone(result['kendall_tau_b'])

    def test_constant_metric_has_no_fabricated_correlation(self):
        p, r = fixture()
        for row in p['records']:
            row['scores']['example_quality']['value'] = 1
        self.assertIsNone(self.primary(self.run_fixture(p, r))['spearman'])

    def test_insufficient_pairs_are_unavailable(self):
        p, r = fixture(2)
        self.assertEqual(self.primary(self.run_fixture(p, r))['status'], 'insufficient_pairs')

    def test_reader_means_allowed_only_for_rexval(self):
        p, r = fixture(benchmark='rexval-1.0.0')
        r['records'][0]['errors']['clinically_significant'][0] = .5
        self.run_fixture(p, r)
        p, r = fixture()
        r['records'][0]['errors']['clinically_significant'][0] = .5
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)

    def test_shared_group_bootstrap_is_deterministic(self):
        pairs = [('group_%04d' % i, 10-i, -i) for i in range(10)]
        first = alignment.clustered_spearman_interval(pairs, 40, 0)
        self.assertEqual(first, alignment.clustered_spearman_interval(pairs, 40, 0))
        self.assertEqual(first['usable_resamples'], 40)
        self.assertEqual(first['spearman_interval_95'], [1,1])

    def test_dependent_candidates_are_not_independent_bootstrap_groups(self):
        pairs = [('group_0000', 3,0), ('group_0000',2,-1), ('group_0001',1,-2)]
        result = alignment.clustered_spearman_interval(pairs, 40, 0)
        self.assertEqual(result['source_groups'], 2)
        self.assertIsNone(result['spearman_interval_95'])

    def test_insufficient_nonconstant_bootstrap_resamples_explicit(self):
        result = alignment.clustered_spearman_interval([('group_%04d' % i,1,-i) for i in range(5)], 40, 0)
        self.assertEqual(result['usable_resamples'], 0)
        self.assertEqual(result['status'], 'insufficient_nonconstant_resamples')

    def test_ci_only_on_predeclared_primary_outcome(self):
        p, r = fixture(10)
        result = alignment.evaluate(p, r, bootstrap_resamples=40, seed=0)
        values = result['metrics']['example_quality']['outcomes']
        self.assertEqual([key for key, value in values.items() if 'cluster_bootstrap' in value], ['clinically_significant_total'])

    def test_raw_text_and_patient_keys_refused(self):
        for target, key in (('prediction','report'), ('reference','patient_id')):
            p, r = fixture()
            (p if target == 'prediction' else r)['records'][0][key] = 'invented_never_loaded'
            with self.assertRaises(ValueError):
                self.run_fixture(p, r)

    def test_real_style_identifiers_refused(self):
        p, r = fixture()
        p['records'][0]['item_id'] = r['records'][0]['item_id'] = 'invented_external_record_key'
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)

    def test_duplicate_reordered_or_wrong_group_join_refused(self):
        for change in ('duplicate','order','group'):
            p, r = fixture()
            if change == 'duplicate':
                p['records'][1]['item_id'] = r['records'][1]['item_id'] = p['records'][0]['item_id']
            elif change == 'order':
                r['records'].reverse()
            else:
                r['records'][0]['source_group_id'] = 'group_9999'
            with self.assertRaises(ValueError):
                self.run_fixture(p, r)

    def test_radevalx_one_candidate_per_group_enforced(self):
        p, r = fixture()
        p['records'][1]['source_group_id'] = r['records'][1]['source_group_id'] = 'group_0000'
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)

    def test_metric_inventory_missingness_never_silent_drop(self):
        p, r = fixture()
        p['records'][0]['scores'] = {}
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)

    def test_nonfinite_bool_and_negative_counts_refused(self):
        for value in (float('nan'), float('inf'), True):
            p, r = fixture()
            p['records'][0]['scores']['example_quality']['value'] = value
            with self.assertRaises(ValueError):
                self.run_fixture(p, r)
        for value in (-1, True, float('nan')):
            p, r = fixture()
            r['records'][0]['errors']['clinically_significant'][0] = value
            with self.assertRaises(ValueError):
                self.run_fixture(p, r)

    def test_failed_score_must_have_null_not_zero(self):
        p, r = fixture()
        p['records'][0]['scores']['example_quality'] = {'status':'failed_unavailable','value':0}
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)

    def test_wrong_width_and_reader_policy_refused(self):
        p, r = fixture()
        r['records'][0]['errors']['clinically_significant'].pop()
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)
        p, r = fixture()
        r['reference_policy'] = 'best_reader'
        with self.assertRaises(ValueError):
            self.run_fixture(p, r)

    def test_no_input_mutation_model_calls_or_clinical_promotion(self):
        p, r = fixture()
        before = copy.deepcopy((p, r))
        result = self.run_fixture(p, r)
        self.assertEqual((p, r), before)
        self.assertEqual(result['new_model_calls'], 0)
        self.assertFalse(result['local_implementation_qualified'])
        for key in ('clinical_qualified','primary_metric_eligible','selection_changed','regeneration_authorized',
                    'weight_or_threshold_fitting','published_score_qualifies_local_implementation'):
            self.assertFalse(result['policy'][key])

    def test_bootstrap_bounds_refused(self):
        p, r = fixture()
        for bad in (-1,2001,True):
            with self.assertRaises(ValueError):
                alignment.evaluate(p, r, bootstrap_resamples=bad)

    def test_separate_total_significance_and_error_category_outcomes(self):
        result = self.run_fixture()
        self.assertEqual(len(result['metrics']['example_quality']['outcomes']), 19)
        self.assertEqual(result['correlation_direction'], 'quality_score_vs_negative_expert_error_burden')


if __name__ == '__main__':
    unittest.main()

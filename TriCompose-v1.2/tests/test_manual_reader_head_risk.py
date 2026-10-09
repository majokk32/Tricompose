"""Invented per-head reference and decision metadata only."""
import copy
import unittest

import numpy as np

from tricompose_v12.manual_reader_risk_coverage import aggregate_reports, analyze
from tricompose_v12.manual_reader_head_risk import (
    FINDINGS, IX, analyze_heads, aggregate_heads, support_status,
)
from test_manual_reader_risk_coverage import fixtures


class HeadRiskCounterTests(unittest.TestCase):
    def test_every_original_report_counter_conserved(self):
        refs, details = fixtures()
        _, _, heads = aggregate_heads(refs, details)
        _, _, pooled = aggregate_reports(refs, details)
        np.testing.assert_array_equal(heads.sum(axis=2), pooled)
        self.assertTrue(np.all(heads[..., IX['attempted']] == 1))

    def test_positive_negative_uncertain_unknown_not_pooled(self):
        result = analyze_heads(*fixtures(), repetitions=100)
        rows = [r for r in result['rows'] if r['domain'] == 'all' and r['readout'] == 'native_labels'
                and r['policy'] == 'radgraph']
        risks = [r['metrics']['conditional_literal_assertion_error']['estimate'] for r in rows]
        self.assertEqual(risks, [0., 0., 1., None])

    def test_unknown_only_head_does_not_get_perfect_risk(self):
        r = next(r for r in analyze_heads(*fixtures(), repetitions=100)['rows']
                 if r['domain'] == 'all' and r['finding'] == FINDINGS[-1])
        self.assertEqual(r['unknown'], 2)
        self.assertEqual(r['commit_unknown'], 2)
        self.assertIsNone(r['metrics']['conditional_literal_assertion_error']['estimate'])
        self.assertIsNone(r['metrics']['known_recovery']['estimate'])
        self.assertEqual(r['metrics']['conditional_literal_assertion_error']['zero_denominator_draws'], 100)

    def test_uncertain_commitment_separate_from_polarity_flip(self):
        r = next(r for r in analyze_heads(*fixtures(), repetitions=100)['rows']
                 if r['domain'] == 'all' and r['finding'] == FINDINGS[2])
        self.assertEqual(r['flips'], 0)
        self.assertEqual(r['commit_uncertain'], 2)
        self.assertEqual(r['metrics']['conditional_literal_assertion_error']['numerator'], 2)

    def test_missing_head_or_duplicate_detail_rejected(self):
        refs, details = fixtures()
        for changed in (details[:-1], details + [details[0]]):
            with self.assertRaises(ValueError):
                aggregate_heads(refs, changed)

    def test_abstention_not_success(self):
        refs, details = fixtures()
        for d in details:
            if d['finding'] == FINDINGS[0]:
                d.update(state=None, status='abstain_unknown')
        r = next(r for r in analyze_heads(refs, details, repetitions=100)['rows']
                 if r['domain'] == 'all' and r['finding'] == FINDINGS[0])
        self.assertEqual(r['metrics']['known_recovery']['estimate'], 0.)
        self.assertIsNone(r['metrics']['conditional_literal_assertion_error']['estimate'])

    def test_failure_kept_in_head_denominator(self):
        refs, details = fixtures()
        for d in details:
            if d['report_id'] == 'report_0001':
                d.update(state=None, status='unavailable_reader')
        r = next(r for r in analyze_heads(refs, details, repetitions=100)['rows']
                 if r['domain'] == 'all' and r['finding'] == FINDINGS[0])
        self.assertEqual(r['attempted'], 2)
        self.assertEqual(r['unavailable'], 1)
        self.assertEqual(r['metrics']['positive_recovery']['estimate'], .5)

    def test_reference_metadata_and_inputs_not_mutated(self):
        refs, details = fixtures()
        original = copy.deepcopy((refs, details))
        analyze_heads(refs, details, repetitions=100)
        self.assertEqual((refs, details), original)


class HeadRiskSamplingTests(unittest.TestCase):
    def test_same_report_draw_hash_as_pooled_parent(self):
        f = fixtures()
        self.assertEqual(analyze_heads(*f, repetitions=100)['sampling']['weights_sha256'],
                         analyze(*f, repetitions=100)['sampling']['weights_sha256'])

    def test_all_heads_views_masks_domains_kept(self):
        r = analyze_heads(*fixtures(), repetitions=100)
        self.assertEqual(len(r['rows']), 168)
        self.assertEqual(len(r['paired_contrasts']), 72)
        self.assertEqual(len(r['reference_inventory']), 12)
        self.assertEqual(sum(len(row['metrics']) for row in r['rows']), 1512)
        self.assertEqual(sum(len(row['differences']) for row in r['paired_contrasts']), 648)

    def test_domains_do_not_supply_each_others_support(self):
        refs, details = fixtures()
        result = analyze_heads(refs, details, repetitions=100)
        inventory = [r for r in result['reference_inventory'] if r['finding'] == FINDINGS[0]]
        self.assertEqual([r['positive'] for r in inventory], [2, 1, 1])
        self.assertTrue(all(r['negative'] == 0 for r in inventory))

    def test_output_order_deterministic(self):
        refs, details = fixtures()
        self.assertEqual(analyze_heads(refs, details, repetitions=100),
                         analyze_heads(list(reversed(refs)), list(reversed(details)), repetitions=100))

    def test_undefined_paired_endpoint_not_zero(self):
        r = next(r for r in analyze_heads(*fixtures(), repetitions=100)['paired_contrasts']
                 if r['finding'] == FINDINGS[-1])
        self.assertIsNone(r['differences']['conditional_literal_assertion_error']['right_minus_left'])
        self.assertIsNone(r['differences']['conditional_literal_assertion_error']['percentile_interval'])

    def test_draw_arguments_bounded_and_typed(self):
        for kwargs in ({'seed': -1}, {'seed': True}, {'repetitions': True}, {'repetitions': 10001}):
            with self.assertRaises(ValueError):
                analyze_heads(*fixtures(), **kwargs)

    def test_no_clinical_qualification_or_new_score(self):
        result = analyze_heads(*fixtures(), repetitions=100)
        for flag in ('clinical_qualified', 'new_untouched_test_set', 'unknown_is_negative',
                     'best_policy_selected', 'thresholds_fitted', 'selection_changed',
                     'regeneration_authorized', 'synthetic_domain_transport_validated'):
            self.assertFalse(result[flag])
        self.assertFalse(result['sampling']['patient_groups_verified'])

    def test_support_status_is_count_description_not_minimum_n_claim(self):
        self.assertEqual(support_status(1, 0), 'no_negative_reference_support')
        self.assertEqual(support_status(0, 1), 'no_positive_reference_support')
        self.assertEqual(support_status(0, 0), 'no_positive_or_negative_reference_support')
        self.assertEqual(support_status(1, 1), 'both_reference_polarities_present_not_sufficient_validation')


if __name__ == '__main__':
    unittest.main()

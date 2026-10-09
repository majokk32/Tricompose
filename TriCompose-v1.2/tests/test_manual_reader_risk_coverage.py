"""Invented four-state counters and clustered draws only, no private data."""
import copy
import unittest

import numpy as np

from tricompose_v12.manual_reader_risk_coverage import (
    CONTRASTS, FIELDS, FINDINGS, IX, POLICIES, VIEWS,
    aggregate_reports, analyze, draws, fractions, interval, ratio,
)


def fixtures():
    truths = dict(zip(FINDINGS, ('positive', 'negative', 'uncertain', 'unknown')))
    refs = [{'report_id': f'report_{i:04d}', 'source_domain': domain,
             'source_sha256': str(i) * 64, 'finding_states': dict(truths)}
            for i, domain in enumerate(('mimic', 'chexpert'))]
    details = [{'readout': view, 'policy': policy, 'report_id': ref['report_id'],
                'finding': f, 'human_literal_reference_state': truth,
                'state': truth if truth in ('positive', 'negative') else 'positive',
                'status': 'accepted_determinate_proposal'}
               for view in VIEWS for policy in POLICIES for ref in refs for f, truth in truths.items()]
    return refs, details


class RiskCounterTests(unittest.TestCase):
    def test_uncertain_is_error_but_literal_unknown_is_not(self):
        refs, details = fixtures()
        _, _, a = aggregate_reports(refs, details)
        f = fractions(a.sum(axis=2))
        n, d = f['conditional_literal_assertion_error']
        np.testing.assert_array_equal(n, np.full((2, 7), 2))
        np.testing.assert_array_equal(d, np.full((2, 7), 6))

    def test_correct_known_recovery_separate_from_coverage(self):
        refs, details = fixtures()
        _, _, a = aggregate_reports(refs, details)
        f = fractions(a.sum(axis=2))
        np.testing.assert_array_equal(ratio(*f['known_recovery']), np.ones((2, 7)))
        np.testing.assert_array_equal(ratio(*f['proposal_coverage']), np.ones((2, 7)))
        np.testing.assert_array_equal(ratio(*f['uncertain_commitment_rate']), np.ones((2, 7)))

    def test_failed_report_remains_in_reference_denominator(self):
        refs, details = fixtures()
        for row in details:
            if row['report_id'] == 'report_0001':
                row.update(state=None, status='unavailable_reader')
        _, _, a = aggregate_reports(refs, details)
        f = fractions(a.sum(axis=2))
        np.testing.assert_array_equal(ratio(*f['known_recovery']), np.full((2, 7), .5))
        np.testing.assert_array_equal(ratio(*f['unavailable_check_rate']), np.full((2, 7), .5))

    def test_polarity_flip_joins_uncertain_error(self):
        refs, details = fixtures()
        for row in details:
            if row['human_literal_reference_state'] == 'negative':
                row['state'] = 'positive'
        _, _, a = aggregate_reports(refs, details)
        np.testing.assert_array_equal(ratio(*fractions(a.sum(axis=2))['conditional_literal_assertion_error']),
                                      np.full((2, 7), 2/3))

    def test_abstention_not_success_or_error(self):
        refs, details = fixtures()
        for row in details:
            row.update(state=None, status='abstain_unknown')
        _, _, a = aggregate_reports(refs, details)
        f = fractions(a.sum(axis=2))
        self.assertTrue(np.isnan(ratio(*f['conditional_literal_assertion_error'])).all())
        np.testing.assert_array_equal(ratio(*f['known_recovery']), np.zeros((2, 7)))

    def test_zero_denominator_not_zero_risk(self):
        value = interval(ratio(np.array([0, 1]), np.array([0, 2])))
        self.assertEqual(value['zero_denominator_draws'], 1)
        self.assertEqual(value['valid_draws'], 1)
        self.assertEqual(value['percentile_interval'], [.5, .5])

    def test_all_missing_interval_null(self):
        self.assertIsNone(interval([np.nan, np.nan])['percentile_interval'])

    def test_impossible_counts_fail(self):
        a = np.zeros(len(FIELDS), dtype=int)
        a[IX['attempted']] = 1
        with self.assertRaises(ValueError):
            fractions(a)

    def test_negative_and_fractional_counts_fail(self):
        for value in (-1, .5):
            a = np.zeros(len(FIELDS))
            a[IX['attempted']] = value
            with self.assertRaises(ValueError):
                fractions(a)

    def test_invalid_confidence_and_infinity_fail(self):
        with self.assertRaises(ValueError):
            interval([.2], confidence=1)
        with self.assertRaises(ValueError):
            interval([np.inf])


class FixedInventoryTests(unittest.TestCase):
    def test_duplicate_detail_rejected(self):
        refs, rows = fixtures()
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows + [rows[0]])

    def test_missing_head_rejected(self):
        refs, rows = fixtures()
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows[:-1])

    def test_changed_reference_rejected(self):
        refs, rows = fixtures()
        rows[0]['human_literal_reference_state'] = 'unknown'
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows)

    def test_duplicate_source_report_cluster_rejected(self):
        refs, rows = fixtures()
        refs[1]['source_sha256'] = refs[0]['source_sha256']
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows)

    def test_nonopaque_index_rejected(self):
        refs, rows = fixtures()
        refs[0]['report_id'] = 'external_source_key'
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows)

    def test_bad_acceptance_receipt_rejected(self):
        refs, rows = fixtures()
        rows[0]['state'] = None
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows)

    def test_unrecognized_state_rejected(self):
        refs, rows = fixtures()
        rows[0]['state'] = 'uncertain'
        with self.assertRaises(ValueError):
            aggregate_reports(refs, rows)

    def test_old_four_state_inventory_not_mutated(self):
        refs, rows = fixtures()
        original = copy.deepcopy((refs, rows))
        aggregate_reports(refs, rows)
        self.assertEqual((refs, rows), original)


class ClusteredDrawTests(unittest.TestCase):
    def test_same_seed_identical_draws(self):
        d = ['mimic', 'mimic', 'chexpert', 'chexpert']
        np.testing.assert_array_equal(draws(d, repetitions=100), draws(d, repetitions=100))

    def test_each_stratum_size_preserved(self):
        w = draws(['mimic', 'mimic', 'chexpert', 'chexpert'], repetitions=100)
        np.testing.assert_array_equal(w[:, :2].sum(axis=1), np.full(100, 2))
        np.testing.assert_array_equal(w[:, 2:].sum(axis=1), np.full(100, 2))

    def test_missing_stratum_rejected(self):
        with self.assertRaises(ValueError):
            draws(['mimic'], repetitions=100)

    def test_unbounded_draws_rejected(self):
        with self.assertRaises(ValueError):
            draws(['mimic', 'chexpert'], repetitions=10001)

    def test_identical_policies_paired_difference_exact_zero(self):
        refs, details = fixtures()
        result = analyze(refs, details, repetitions=100)
        self.assertEqual(len(result['rows']), 42)
        self.assertEqual(len(result['paired_contrasts']), 2 * 3 * len(CONTRASTS))
        for row in result['paired_contrasts']:
            for value in row['differences'].values():
                self.assertEqual(value['right_minus_left'], 0.)
                self.assertEqual(value['percentile_interval'], [0., 0.])

    def test_manifest_order_does_not_change_draws_or_results(self):
        refs, details = fixtures()
        a = analyze(refs, details, repetitions=100)
        b = analyze(list(reversed(refs)), list(reversed(details)), repetitions=100)
        self.assertEqual(a, b)

    def test_no_patient_independence_or_clinical_qualification(self):
        refs, details = fixtures()
        result = analyze(refs, details, repetitions=100)
        self.assertFalse(result['sampling']['patient_groups_verified'])
        self.assertFalse(result['clinical_qualified'])
        self.assertFalse(result['best_policy_selected'])
        self.assertFalse(result['selection_changed'])
        self.assertFalse(result['regeneration_authorized'])


if __name__ == '__main__':
    unittest.main()

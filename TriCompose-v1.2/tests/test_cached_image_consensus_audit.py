"""Independent arithmetic fixtures; no cached real input or pixel IO."""
import unittest

from audit_cached_image_consensus import replay_decision, fractions, percentile, check_interval


class ImageConsensusIndependentAuditTests(unittest.TestCase):
    def test_exact_threshold_and_different_reference_not_used(self):
        r = {'xrv_score': .5, 'margins': [.1, .2, .3], 'reference_state': 'negative'}
        self.assertEqual(replay_decision(r, 'xrv_exact_0_5'), ('positive', 'accepted_determinate'))

    def test_disagreement_and_template_sensitive_are_distinct(self):
        r = {'xrv_score': .2, 'margins': [.1, -.2, .4]}
        self.assertEqual(replay_decision(r, 'agree_fixed_mean'), (None, 'abstain_reader_disagreement'))
        self.assertEqual(replay_decision(r, 'agree_all_templates'), (None, 'abstain_template_sensitive'))

    def test_failure_not_negative(self):
        r = {'xrv_score': None, 'margins': [.1, .2, .3]}
        self.assertEqual(replay_decision(r, 'agree_fixed_mean'), (None, 'unavailable_reader'))

    def test_mean_and_template_ties_are_not_acceptance(self):
        r = {'xrv_score': .8, 'margins': [1., -1., 0.]}
        self.assertEqual(replay_decision(r, 'biovil_fixed_mean'), (None, 'abstain_mean_tie'))
        self.assertEqual(replay_decision(r, 'agree_all_templates'), (None, 'abstain_template_tie'))

    def test_separate_full_reference_and_accepted_denominators(self):
        c = {'false_positive': 2, 'false_negative': 1, 'true_positive': 20, 'true_negative': 17,
             'accepted': 40, 'attempted': 50, 'reference_positive': 25, 'reference_negative': 25, 'unavailable': 0}
        r = fractions(c)
        self.assertEqual(r['accepted_error_risk'], (3, 40))
        self.assertEqual(r['negative_reference_recovery'], (17, 25))
        self.assertEqual(r['conditional_negative_recovery'], (17, 19))

    def test_separate_linear_percentile(self):
        value, count = percentile([0., 1., 2., 3.])
        self.assertEqual(count, 4)
        self.assertAlmostEqual(value[0], .075)
        self.assertAlmostEqual(value[1], 2.925)

    def test_missing_draw_not_zero(self):
        self.assertEqual(percentile([float('nan')]), (None, 0))
        check_interval({'valid_draws': 0, 'zero_denominator_draws': 1, 'percentile_interval': None}, [float('nan')])
        with self.assertRaises(ValueError):
            check_interval({'valid_draws': 0, 'zero_denominator_draws': 1, 'percentile_interval': [0., 0.]}, [float('nan')])


if __name__ == '__main__':
    unittest.main()

"""Output plumbing tests with invented aggregates; no source or model IO."""
import unittest
from unittest.mock import patch

from analyze_manual_reader_risk_coverage import tables, main


def metric():
    return {'numerator': 2, 'denominator': 143, 'estimate': 2 / 143,
            'percentile_interval': [0., .1], 'valid_draws': 100, 'zero_denominator_draws': 0}


def fixture():
    keys = ('known_recovery', 'conditional_literal_assertion_error', 'proposal_coverage')
    row = {'readout': 'native_labels', 'domain': 'all', 'policy': 'invented', 'reports': 100,
           'correct_positive': 20, 'positive': 21, 'correct_negative': 121, 'negative': 125,
           'flips': 0, 'commit_uncertain': 2, 'uncertain': 10, 'accepted': 145,
           'attempted': 400, 'unavailable': 0, 'commit_unknown': 2, 'unknown': 244,
           'metrics': {k: metric() for k in keys}}
    return {'rows': [row], 'paired_contrasts': [{'readout': 'native_labels', 'domain': 'all',
        'left': 'invented_a', 'right': 'invented_b',
        'differences': {'known_recovery': {'right_minus_left': -.1,
            'percentile_interval': [-.2, 0.], 'valid_draws': 100, 'zero_denominator_draws': 0}}}]}


class RiskCoverageTablePlumbingTests(unittest.TestCase):
    def test_raw_fraction_and_denominator_kept(self):
        short, long, pairs = tables(fixture())
        self.assertEqual(short[0]['literal_state_error_numerator'], 2)
        self.assertEqual(short[0]['adjudicable_determinate_proposals'], 143)
        self.assertEqual(short[0]['conditional_literal_assertion_error'], 2/143)
        self.assertEqual(len(long), 3)
        self.assertEqual(len(pairs), 1)

    def test_missing_interval_not_zero(self):
        f = fixture()
        m = f['rows'][0]['metrics']['conditional_literal_assertion_error']
        m.update(estimate=None, percentile_interval=None, valid_draws=0, zero_denominator_draws=100)
        short, _, _ = tables(f)
        self.assertIsNone(short[0]['risk_interval_lower'])
        self.assertIsNone(short[0]['conditional_literal_assertion_error'])

    def test_no_patient_independence_or_qualification(self):
        short, long, pairs = tables(fixture())
        self.assertFalse(short[0]['patient_cluster_verified'])
        self.assertTrue(all(r['clinical_qualified'] is False for r in short + long + pairs))

    def test_difference_direction_preserved(self):
        _, _, pairs = tables(fixture())
        self.assertEqual(pairs[0]['right_minus_left'], -.1)
        self.assertTrue(pairs[0]['shared_paired_draws'])

    def test_requires_actual_slurm_cgroup_before_loading(self):
        with patch('analyze_manual_reader_risk_coverage.sys.argv', ['worker']), \
             patch('analyze_manual_reader_risk_coverage.os.environ', {'SLURM_JOB_ID': '12784259'}), \
             patch('analyze_manual_reader_risk_coverage.Path.read_text', return_value='/login_node'), \
             patch('analyze_manual_reader_risk_coverage.execute') as execute, \
             patch('builtins.print'):
            self.assertEqual(main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

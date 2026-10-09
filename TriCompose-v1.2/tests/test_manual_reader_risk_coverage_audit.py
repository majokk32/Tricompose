"""Independent scalar percentile arithmetic with invented arrays only."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('independent_manual_reader_risk_audit',
    Path(__file__).resolve().parents[1] / 'audits/audit_manual_reader_risk_coverage.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class IndependentPercentileTests(unittest.TestCase):
    def test_empty_percentile_is_null(self):
        self.assertEqual(AUDIT.percentile([np.nan, np.nan]), (None, 0))

    def test_linear_interpolation(self):
        values, valid = AUDIT.percentile([0., 10.])
        np.testing.assert_allclose(values, [.25, 9.75])
        self.assertEqual(valid, 2)

    def test_unordered_values_and_missing(self):
        values, valid = AUDIT.percentile([10., np.nan, 0.])
        np.testing.assert_allclose(values, [.25, 9.75])
        self.assertEqual(valid, 2)

    def test_mismatched_interval_fails(self):
        with self.assertRaises(ValueError):
            AUDIT.compare_interval({'percentile_interval': [0., 0.], 'valid_draws': 2,
                                    'zero_denominator_draws': 0}, [0., 10.])

    def test_missing_draw_counter_fails(self):
        with self.assertRaises(ValueError):
            AUDIT.compare_interval({'percentile_interval': [0., 0.], 'valid_draws': 2,
                                    'zero_denominator_draws': 0}, [np.nan, 0.])

    def test_risk_counts_uncertain_not_unknown(self):
        fields = dict.fromkeys(AUDIT.FIELDS, 0)
        fields.update(flips=1, commit_uncertain=2, commit_unknown=4, accepted=10)
        n, d = AUDIT.rates(np.array([fields[f] for f in AUDIT.FIELDS]))['conditional_literal_assertion_error']
        self.assertEqual((n, d), (3, 6))


if __name__ == '__main__':
    unittest.main()

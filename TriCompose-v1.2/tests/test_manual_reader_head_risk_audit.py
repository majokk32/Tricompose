"""Independent head-audit scalar helpers; invented values only."""
import unittest

from audit_manual_reader_head_risk import estimate, csv_number, verify_numeric_cell


class IndependentHeadAuditTests(unittest.TestCase):
    def test_zero_denominator_has_no_risk(self):
        self.assertIsNone(estimate(0, 0))

    def test_raw_fraction_kept(self):
        self.assertEqual(estimate(2, 61), 2 / 61)

    def test_csv_blank_not_zero(self):
        self.assertIsNone(csv_number(''))
        self.assertEqual(csv_number('0'), 0.)

    def test_missing_cannot_pass_as_zero(self):
        with self.assertRaises(ValueError):
            verify_numeric_cell('0', None)
        with self.assertRaises(ValueError):
            verify_numeric_cell('', 0.)

    def test_finite_scalar_tolerance(self):
        verify_numeric_cell(str(2 / 61), 2 / 61)
        with self.assertRaises(ValueError):
            verify_numeric_cell('.2', .3)

    def test_nan_not_valid_csv_number(self):
        with self.assertRaises(ValueError):
            verify_numeric_cell('nan', 0.)


if __name__ == '__main__':
    unittest.main()

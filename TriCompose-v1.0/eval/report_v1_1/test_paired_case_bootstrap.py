import unittest

from paired_case_bootstrap import paired_case_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_paired_constant_improvement(self):
        rows = [{"case_id": f"case_{i:03d}", "group_id": f"group_{i // 2}",
                 "baseline": i, "method": i + 1} for i in range(6)]
        result = paired_case_bootstrap(rows, repetitions=100)
        self.assertEqual(result["independent_groups"], 3)
        self.assertEqual(result["confidence_interval"], [1.0, 1.0])
        self.assertEqual(result, paired_case_bootstrap(list(reversed(rows)), repetitions=100))

    def test_candidates_not_independent_cases(self):
        row = {"case_id": "case_000", "group_id": "group_000", "baseline": 0, "method": 1}
        with self.assertRaises(ValueError):
            paired_case_bootstrap([row, row])
        result = paired_case_bootstrap([row])
        self.assertIsNone(result["confidence_interval"])

    def test_missing_retained_in_denominator(self):
        rows = [{"case_id": f"case_{i:03d}", "group_id": f"group_{i}",
                 "baseline": 0, "method": 1 if i else None} for i in range(3)]
        result = paired_case_bootstrap(rows, repetitions=100)
        self.assertEqual(result["missing_pairs"], 1)
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertFalse(result["missing_is_success"])

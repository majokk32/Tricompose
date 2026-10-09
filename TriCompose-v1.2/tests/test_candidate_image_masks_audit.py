"""Independent proxy-state readout fixtures, no private/source IO."""
import unittest

from audit_candidate_image_masks import proxy_relation


class CandidateMaskIndependentAuditTests(unittest.TestCase):
    def test_positive_support_and_opposition(self):
        self.assertEqual(proxy_relation('positive', 'positive'), 'proxy_support')
        self.assertEqual(proxy_relation('positive', 'negative'), 'proxy_opposition')

    def test_negative_support_and_opposition(self):
        self.assertEqual(proxy_relation('negative', 'negative'), 'proxy_support')
        self.assertEqual(proxy_relation('negative', 'positive'), 'proxy_opposition')

    def test_unknown_report_is_not_negative(self):
        self.assertEqual(proxy_relation('negative', 'unknown'), 'not_comparable')

    def test_uncertain_report_not_opposition(self):
        self.assertEqual(proxy_relation('positive', 'uncertain'), 'not_comparable')

    def test_withheld_image_is_not_support(self):
        self.assertEqual(proxy_relation(None, 'negative'), 'not_comparable')

    def test_invalid_report_or_image_state_refused(self):
        for a, b in (('uncertain', 'negative'), ('negative', None), ('unknown', 'positive')):
            with self.assertRaises(ValueError):
                proxy_relation(a, b)


if __name__ == '__main__':
    unittest.main()

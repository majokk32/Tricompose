"""Synthetic rule-parser output contracts only, no real source or native IO."""
import unittest

from score_manual_three_readers import FINDINGS, convert


def fixture():
    labels = {f: {'state': 'positive'} for f in FINDINGS}
    return [{'item_id': 'report_0000', 'report_sha256': 'a' * 64, 'output': {
        'status': 'complete', 'original_text_sha256': 'a' * 64, 'hard_action_eligible': False,
        'clinical_score': None, 'failure_reason': None, 'native_labels': labels,
        'mention_conflict_view': {f: {'state': 'uncertain'} for f in FINDINGS}}}]


class NativeManualConversionTests(unittest.TestCase):
    def test_native_and_conflict_views_not_conflated(self):
        rows = fixture()
        self.assertEqual(set(convert(rows, 'native_labels')[0]['finding_states'].values()), {'positive'})
        self.assertEqual(set(convert(rows, 'mention_conflict_view')[0]['finding_states'].values()), {'uncertain'})

    def test_source_hash_checked(self):
        rows = fixture()
        rows[0]['output']['original_text_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            convert(rows, 'native_labels')

    def test_failed_output_null_not_unknown(self):
        rows = fixture()
        rows[0]['output'].update(status='failed_unavailable', native_labels=None,
                                 mention_conflict_view=None, failure_reason='detector_error')
        r = convert(rows, 'native_labels')[0]
        self.assertIsNone(r['finding_states'])
        self.assertEqual(r['status'], 'failed_unavailable')

    def test_failure_cannot_have_fabricated_vector(self):
        rows = fixture()
        rows[0]['output']['status'] = 'failed_unavailable'
        with self.assertRaises(ValueError):
            convert(rows, 'native_labels')

    def test_unknown_stays_unknown(self):
        rows = fixture()
        rows[0]['output']['native_labels'] = {f: {'state': 'unknown'} for f in FINDINGS}
        self.assertEqual(set(convert(rows, 'native_labels')[0]['finding_states'].values()), {'unknown'})

    def test_qualified_or_action_eligible_output_rejected(self):
        rows = fixture()
        rows[0]['output']['hard_action_eligible'] = True
        with self.assertRaises(ValueError):
            convert(rows, 'native_labels')


if __name__ == '__main__':
    unittest.main()

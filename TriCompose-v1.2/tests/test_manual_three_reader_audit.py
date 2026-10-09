"""Independent audit arithmetic on invented states; no model/source IO."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('manual_three_reader_numeric_audit',
    Path(__file__).resolve().parents[1] / 'audits/audit_manual_three_readers.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def row(state='positive', status='complete'):
    return {'status': status, 'finding_states': {'invented_finding': state} if status == 'complete' else None}


class IndependentMaskArithmeticTests(unittest.TestCase):
    def test_seven_masks_no_extra_independent_view(self):
        self.assertEqual(len(AUDIT.MASKS), 7)
        self.assertEqual(AUDIT.MASKS['agree_all_three'], AUDIT.READERS)

    def test_positive_agreement(self):
        self.assertEqual(AUDIT.proposal([row(), row()], 'invented_finding'),
                         ('positive', 'accepted_determinate_proposal'))

    def test_negative_agreement(self):
        self.assertEqual(AUDIT.proposal([row('negative')] * 3, 'invented_finding')[0], 'negative')

    def test_opposite_abstains(self):
        self.assertEqual(AUDIT.proposal([row(), row('negative')], 'invented_finding'),
                         (None, 'abstain_disagreement'))

    def test_unknown_never_negative(self):
        self.assertEqual(AUDIT.proposal([row('unknown')] * 3, 'invented_finding'),
                         (None, 'abstain_unknown'))

    def test_uncertain_never_positive(self):
        self.assertEqual(AUDIT.proposal([row(), row('uncertain')], 'invented_finding'),
                         (None, 'abstain_uncertain'))

    def test_failure_is_unavailable_before_unknown(self):
        self.assertEqual(AUDIT.proposal([row('unknown'), row(status='failed_unavailable')], 'invented_finding'),
                         (None, 'unavailable_reader'))

    def test_invalid_state_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.proposal([row('invented_state')], 'invented_finding')

    def test_invalid_availability_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.proposal([row(status='omitted')], 'invented_finding')

    def test_unavailable_counts_in_known_denominator(self):
        matrix, known, correct, flips = AUDIT.counts([
            ('negative', 'unavailable'), ('positive', 'positive'), ('negative', 'positive'),
            ('uncertain', 'negative'), ('unknown', 'positive')])
        self.assertEqual((known, correct, flips), (3, 1, 1))
        self.assertEqual(matrix['negative']['unavailable'], 1)
        self.assertEqual(matrix['uncertain']['negative'], 1)

    def test_both_flip_directions(self):
        self.assertEqual(AUDIT.counts([('positive', 'negative'), ('negative', 'positive')])[1:], (2, 0, 2))

    def test_unsupported_truth_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.counts([('unavailable', 'positive')])


if __name__ == '__main__':
    unittest.main()

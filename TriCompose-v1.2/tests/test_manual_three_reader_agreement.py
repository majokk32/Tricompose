"""Invented four-state receipts only; no parser, patient source or model IO."""
from copy import deepcopy
import unittest

from tricompose_v12.manual_three_reader_agreement import READERS, POLICIES, FINDINGS, evaluate


def fixtures():
    refs = [{'report_id': 'report_0000', 'source_sha256': 'a' * 64,
             'finding_states': dict.fromkeys(FINDINGS, 'positive')}]
    preds = {name: [{'report_id': 'report_0000', 'source_sha256': 'a' * 64,
        'status': 'complete', 'finding_states': dict.fromkeys(FINDINGS, 'positive')}]
        for name in READERS}
    return refs, preds


class ThreeReaderAgreementTests(unittest.TestCase):
    def test_all_seven_masks_fixed(self):
        self.assertEqual(len(POLICIES), 7)
        refs, pred = fixtures()
        result, rows = evaluate(refs, pred)
        self.assertEqual(set(result['masks']), set(POLICIES))
        self.assertEqual(len(rows), 28)
        self.assertFalse(result['best_policy_selected'])

    def test_unknown_agreement_not_support(self):
        refs, pred = fixtures()
        for records in pred.values():
            records[0]['finding_states'] = dict.fromkeys(FINDINGS, 'unknown')
        result, _ = evaluate(refs, pred)
        self.assertEqual(result['masks']['agree_all_three']['accepted_determinate_proposals'], 0)

    def test_uncertain_agreement_abstains(self):
        refs, pred = fixtures()
        for records in pred.values():
            records[0]['finding_states'] = dict.fromkeys(FINDINGS, 'uncertain')
        result, _ = evaluate(refs, pred)
        self.assertEqual(result['masks']['agree_all_three']['accepted_determinate_proposals'], 0)

    def test_failed_reader_is_unavailable(self):
        refs, pred = fixtures()
        pred['chexpert_negbio'][0].update(status='failed_unavailable', finding_states=None)
        result, _ = evaluate(refs, pred)
        self.assertEqual(result['masks']['agree_all_three']['unavailable_reader_checks'], 4)
        self.assertEqual(result['reader_metrics']['chexpert_negbio']['overall']['unavailable_checks'], 4)

    def test_reader_not_in_mask_does_not_change_pair(self):
        refs, pred = fixtures()
        pred['chexpert_negbio'][0].update(status='failed_unavailable', finding_states=None)
        result, _ = evaluate(refs, pred)
        self.assertEqual(result['masks']['agree_radgraph_chexbert']['correct_known_proposals'], 4)

    def test_missing_prediction_cannot_leave_denominator(self):
        refs, pred = fixtures()
        pred['chexbert'] = []
        with self.assertRaises(ValueError):
            evaluate(refs, pred)

    def test_duplicate_predictions_rejected(self):
        refs, pred = fixtures()
        pred['chexbert'].append(deepcopy(pred['chexbert'][0]))
        with self.assertRaises(ValueError):
            evaluate(refs, pred)

    def test_failed_not_unknown_success(self):
        refs, pred = fixtures()
        pred['chexbert'][0]['status'] = 'failed_unavailable'
        with self.assertRaises(ValueError):
            evaluate(refs, pred)

    def test_wrong_source_hash_rejected(self):
        refs, pred = fixtures()
        pred['chexbert'][0]['source_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            evaluate(refs, pred)

    def test_agreement_can_be_wrong_on_uncertainty(self):
        refs, pred = fixtures()
        refs[0]['finding_states'] = dict.fromkeys(FINDINGS, 'uncertain')
        result, _ = evaluate(refs, pred)
        self.assertEqual(result['masks']['agree_all_three']['determinate_on_uncertain_reference'], 4)
        self.assertEqual(result['masks']['agree_all_three']['hard_positive_negative_flips'], 0)

    def test_unknown_reference_promotions_not_clinical_errors(self):
        refs, pred = fixtures()
        refs[0]['finding_states'] = dict.fromkeys(FINDINGS, 'unknown')
        result, _ = evaluate(refs, pred)
        row = result['masks']['agree_all_three']
        self.assertEqual(row['determinate_on_unknown_literal_reference'], 4)
        self.assertFalse(row['unknown_promotions_are_clinical_errors'])

    def test_opposition_withholds_pair_but_preserves_single(self):
        refs, pred = fixtures()
        pred['chexbert'][0]['finding_states'][FINDINGS[0]] = 'negative'
        result, _ = evaluate(refs, pred)
        self.assertEqual(result['masks']['chexbert']['hard_positive_negative_flips'], 1)
        self.assertEqual(result['masks']['agree_all_three']['accepted_determinate_proposals'], 3)

    def test_no_clinical_or_fault_authorization(self):
        refs, pred = fixtures()
        result, _ = evaluate(refs, pred)
        for key in ('clinical_qualified', 'selection_changed', 'regeneration_authorized', 'thresholds_fitted'):
            self.assertFalse(result[key])


if __name__ == '__main__':
    unittest.main()

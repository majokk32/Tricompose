"""Invented state/hash receipts only. No clinical source or inference."""
from copy import deepcopy
import unittest

from tricompose_v12.cached_manual_agreement import FINDINGS, STATES, READERS, evaluate, matrix


def fixtures():
    truths = ('positive', 'negative', 'unknown', 'uncertain')
    qwen = ('positive', 'positive', 'negative', 'uncertain')
    predictions = {name: [] for name in READERS}
    confusions = {name: {f: matrix() for f in FINDINGS} for name in READERS}
    for i, (truth, q) in enumerate(zip(truths, qwen)):
        for name, state in (('chexbert', truth), ('qwen_span_v2', q)):
            predictions[name].append({'item_id': f'item_{i:04d}', 'status': 'complete',
                'finding_states': dict.fromkeys(FINDINGS, state),
                'source_report_sha256': 'a' * 64, 'selected_input_sha256': 'b' * 64})
            for f in FINDINGS:
                confusions[name][f][truth][state] += 1
    for rows in predictions.values():
        rows.append({'item_id': 'item_0004', 'status': 'unlinked', 'finding_states': None})
    return predictions, confusions


class CachedManualAgreementTests(unittest.TestCase):
    def test_three_fixed_masks_all_evaluated(self):
        p, c = fixtures()
        r = evaluate(p, c)
        self.assertEqual(len(r['policies']), 3)
        self.assertFalse(r['best_policy_selected'])

    def test_skip_inventory_retained_not_fact_denominator(self):
        p, c = fixtures()
        r = evaluate(p, c)
        self.assertEqual((r['annotation_inventory'], r['shared_complete_reports'],
                          r['not_comparable_inventory_entries']), (5, 4, 1))
        self.assertEqual(r['policies']['chexbert']['overall']['attempted_shared_complete_checks'], 16)

    def test_known_flips_separate_from_unknown_promotions(self):
        p, c = fixtures()
        row = evaluate(p, c)['policies']['qwen_span_v2']['overall']
        self.assertEqual(row['hard_positive_negative_flips'], 4)
        self.assertEqual(row['determinate_on_unknown_reference'], 4)
        self.assertFalse(row['unknown_promotions_are_clinical_errors'])
        self.assertEqual(row['conditional_known_error_rate'], 0.5)

    def test_pair_count_proof_no_per_record_gold_export(self):
        p, c = fixtures()
        r = evaluate(p, c)
        row = r['policies']['agree_chexbert_qwen_span_v2']['overall']
        self.assertEqual(row['accepted_determinate_proposals'], 4)
        self.assertEqual(row['hard_positive_negative_flips'], 0)
        self.assertEqual(row['correct_known_reference_recall'], 0.5)
        self.assertFalse(r['per_record_gold_reconstructed'])

    def test_uncertain_agreement_not_determinate(self):
        p, c = fixtures()
        row = evaluate(p, c)['policies']['agree_chexbert_qwen_span_v2']['overall']
        self.assertEqual(row['decision_status_counts']['abstain_uncertain'], 4)
        self.assertEqual(row['uncertain_reference_checks'], 4)

    def test_unknown_agreement_not_support(self):
        p, c = fixtures()
        for f in FINDINGS:
            p['qwen_span_v2'][2]['finding_states'][f] = 'unknown'
            c['qwen_span_v2'][f]['unknown']['negative'] -= 1
            c['qwen_span_v2'][f]['unknown']['unknown'] += 1
        r = evaluate(p, c)
        self.assertFalse(r['unknown_agreement_is_factual_support'])
        self.assertEqual(r['policies']['agree_chexbert_qwen_span_v2']['overall']['accepted_determinate_proposals'], 4)

    def test_imperfect_baseline_cannot_stand_in_for_reference(self):
        p, c = fixtures()
        for f in FINDINGS:
            p['chexbert'][0]['finding_states'][f] = 'negative'
            c['chexbert'][f]['positive']['positive'] -= 1
            c['chexbert'][f]['positive']['negative'] += 1
        r = evaluate(p, c)
        row = r['policies']['agree_chexbert_qwen_span_v2']['overall']
        self.assertIsNone(row['hard_positive_negative_flips'])
        self.assertIsNone(row['correct_known_reference_recall'])
        self.assertFalse(any(r['identity_count_proof_by_finding'].values()))

    def test_no_proposals_error_rate_not_zero(self):
        p, c = fixtures()
        for name in READERS:
            for row in p[name][:-1]:
                row['finding_states'] = dict.fromkeys(FINDINGS, 'unknown')
            for f in FINDINGS:
                c[name][f] = matrix()
                for state in STATES:
                    c[name][f][state]['unknown'] = 1
        r = evaluate(p, c)
        self.assertIsNone(r['policies']['chexbert']['overall']['conditional_known_error_rate'])
        self.assertIsNone(r['policies']['agree_chexbert_qwen_span_v2']['overall']['hard_positive_negative_flips'])

    def test_duplicate_records_fail(self):
        p, c = fixtures()
        p['chexbert'].append(deepcopy(p['chexbert'][0]))
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_removed_skipped_record_fails(self):
        p, c = fixtures()
        p['chexbert'].pop()
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_different_completed_cohort_fails(self):
        p, c = fixtures()
        p['chexbert'][0].update(status='failed_unavailable', finding_states=None)
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_failed_does_not_become_unknown_success(self):
        p, c = fixtures()
        p['chexbert'][-1]['finding_states'] = dict.fromkeys(FINDINGS, 'unknown')
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_same_impression_hash_required(self):
        p, c = fixtures()
        p['chexbert'][0]['selected_input_sha256'] = 'c' * 64
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_same_original_report_hash_required(self):
        p, c = fixtures()
        p['chexbert'][0]['source_report_sha256'] = 'c' * 64
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_invalid_state_fails(self):
        p, c = fixtures()
        p['chexbert'][0]['finding_states'][FINDINGS[0]] = 'absent'
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_bool_count_not_integer(self):
        p, c = fixtures()
        c['chexbert'][FINDINGS[0]]['positive']['positive'] = True
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_changed_reference_support_fails(self):
        p, c = fixtures()
        f = FINDINGS[0]
        c['qwen_span_v2'][f]['positive']['positive'] -= 1
        c['qwen_span_v2'][f]['negative']['positive'] += 1
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_changed_prediction_support_fails(self):
        p, c = fixtures()
        f = FINDINGS[0]
        c['qwen_span_v2'][f]['positive']['positive'] -= 1
        c['qwen_span_v2'][f]['positive']['negative'] += 1
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_joint_matrix_proof_catches_same_marginal_wrong_assignment(self):
        p, c = fixtures()
        f = FINDINGS[0]
        c['qwen_span_v2'][f]['positive']['positive'] -= 1
        c['qwen_span_v2'][f]['positive']['negative'] += 1
        c['qwen_span_v2'][f]['unknown']['negative'] -= 1
        c['qwen_span_v2'][f]['unknown']['positive'] += 1
        with self.assertRaises(ValueError):
            evaluate(p, c)

    def test_no_truth_votes_or_clinical_promotion(self):
        p, c = fixtures()
        r = evaluate(p, c)
        for field in ('independent_truth_votes', 'clinical_qualified', 'selection_changed',
                      'regeneration_authorized', 'thresholds_fitted', 'untouched_final_test'):
            self.assertFalse(r[field])


if __name__ == '__main__':
    unittest.main()

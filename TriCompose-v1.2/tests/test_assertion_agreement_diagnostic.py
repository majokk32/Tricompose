"""Invented prediction metadata only; no source files or model calls."""
from copy import deepcopy
import unittest

from tricompose_v12.assertion_agreement_diagnostic import READERS, POLICIES, decision, evaluate


def fixtures():
    refs = [{'item_id': 'fixture_000', 'report_sha256': 'a' * 64,
        'evaluation_findings': ['cardiomegaly'], 'expected_states': {'cardiomegaly': 'positive'}}]
    pred = {reader: [{'item_id': 'fixture_000', 'report_sha256': 'a' * 64,
        'status': 'complete', 'finding_states': {'cardiomegaly': 'positive'}}] for reader in READERS}
    return refs, pred


class AgreementDiagnosticTests(unittest.TestCase):
    def test_seven_fixed_masks(self):
        self.assertEqual(len(POLICIES), 7)

    def test_all_agree_positive(self):
        _, preds = fixtures()
        rows = {k: v[0] for k, v in preds.items()}
        self.assertEqual(decision(rows, READERS, 'cardiomegaly')['state'], 'positive')

    def test_unknown_agreement_cannot_authorize_statement(self):
        _, preds = fixtures()
        rows = {k: v[0] for k, v in preds.items()}
        for row in rows.values():
            row['finding_states']['cardiomegaly'] = 'unknown'
        self.assertEqual(decision(rows, READERS, 'cardiomegaly')['status'], 'abstain_unknown')

    def test_uncertain_agreement_cannot_authorize_statement(self):
        _, preds = fixtures()
        rows = {k: v[0] for k, v in preds.items()}
        for row in rows.values():
            row['finding_states']['cardiomegaly'] = 'uncertain'
        self.assertEqual(decision(rows, READERS, 'cardiomegaly')['status'], 'abstain_uncertain')

    def test_opposition_abstains(self):
        _, preds = fixtures()
        rows = {k: v[0] for k, v in preds.items()}
        rows['radgraph']['finding_states']['cardiomegaly'] = 'negative'
        self.assertEqual(decision(rows, READERS, 'cardiomegaly')['status'], 'abstain_disagreement')

    def test_failed_readers_not_unknown_success(self):
        _, preds = fixtures()
        rows = {k: v[0] for k, v in preds.items()}
        rows['radgraph'].update(status='failed_unavailable', finding_states=None)
        self.assertEqual(decision(rows, READERS, 'cardiomegaly')['status'], 'unavailable_reader')

    def test_reader_not_in_mask_does_not_change_decision(self):
        _, preds = fixtures()
        rows = {k: v[0] for k, v in preds.items()}
        rows['qwen_span_v2']['finding_states']['cardiomegaly'] = 'negative'
        self.assertEqual(decision(rows, ('radgraph', 'chexbert'), 'cardiomegaly')['state'], 'positive')

    def test_all_fixed_masks_evaluated_no_best_choice(self):
        refs, preds = fixtures()
        result, details = evaluate(refs, preds)
        self.assertEqual(set(result['policies']), set(POLICIES))
        self.assertEqual(len(details), 7)
        self.assertFalse(result['best_policy_selected'])

    def test_uncertain_reference_determinate_proposal_is_unsafe(self):
        refs, preds = fixtures()
        refs[0]['expected_states']['cardiomegaly'] = 'uncertain'
        result, _ = evaluate(refs, preds)
        row = result['policies']['agree_all_three']
        self.assertEqual(row['accepted_authored_state_errors'], 1)
        self.assertEqual(row['determinate_on_authored_uncertain_or_unknown'], 1)
        self.assertEqual(row['hard_positive_negative_flips'], 0)

    def test_abstention_does_not_count_correct(self):
        refs, preds = fixtures()
        for records in preds.values():
            records[0]['finding_states']['cardiomegaly'] = 'unknown'
        result, _ = evaluate(refs, preds)
        row = result['policies']['agree_all_three']
        self.assertEqual(row['correct_determinate_recall'], 0)
        self.assertEqual(row['proposal_coverage'], 0)
        self.assertIsNone(row['accepted_authored_error_rate'])

    def test_missing_prediction_cannot_leave_denominator(self):
        refs, preds = fixtures()
        preds['radgraph'] = []
        with self.assertRaises(ValueError):
            evaluate(refs, preds)

    def test_wrong_hash_rejected(self):
        refs, preds = fixtures()
        preds['radgraph'][0]['report_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            evaluate(refs, preds)

    def test_hard_flip_is_separate_from_unsupported_commit(self):
        refs, preds = fixtures()
        for records in preds.values():
            records[0]['finding_states']['cardiomegaly'] = 'negative'
        result, _ = evaluate(refs, preds)
        row = result['policies']['agree_all_three']
        self.assertEqual(row['hard_positive_negative_flips'], 1)
        self.assertEqual(row['determinate_on_authored_uncertain_or_unknown'], 0)


if __name__ == '__main__':
    unittest.main()

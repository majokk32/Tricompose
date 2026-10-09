"""Independent cache audit, only invented state/hash records."""
import ast
from copy import deepcopy
import unittest
from unittest.mock import patch

import audit_manual_opacity_reader as audit


def fixture():
    references, predictions = [], []
    for i, (truth, pred) in enumerate((('positive', 'positive'), ('positive', None),
        ('negative', 'positive'), ('unknown', 'negative'), (None, 'unknown'))):
        key = f'report_{i:04d}'
        references.append({'report_id': key, 'source_index': i, 'source_sha256': 'a' * 64,
            'status': 'complete' if truth else 'failed_unavailable', 'failure_type': None if truth else 'ValueError',
            'projection': {'source_sha256': 'a' * 64, 'manual_literal_state': truth,
                'current_lung_opacity_reference': None, 'clinical_qualified': False} if truth else None})
        labels = dict.fromkeys(audit.HEADS, 'unknown')
        labels['lung_opacity'] = pred
        predictions.append({'report_id': key, 'source_sha256': 'a' * 64,
            'status': 'complete' if pred else 'failed_unavailable', 'failure_type': None if pred else 'RuntimeError',
            'finding_states': labels if pred else None})
    return references, predictions


class ManualOpacityReaderAuditTests(unittest.TestCase):
    def test_full_denominator_failure_and_scope_difference(self):
        numeric, states, details = audit.replay(*fixture())
        self.assertEqual(numeric['attempted_reports'], 5)
        self.assertEqual(numeric['jointly_available'], 3)
        self.assertEqual(numeric['reference_unavailable'], 1)
        self.assertEqual(numeric['known_literal_reference_support'], 3)
        self.assertEqual(numeric['known_literal_match_fraction_all_supported'], 1 / 3)
        self.assertEqual(numeric['known_literal_match_fraction_prediction_available'], .5)
        self.assertEqual(numeric['known_literal_polarity_flips'], 1)
        self.assertEqual(numeric['determinate_predictions_on_literal_unknown'], 1)
        self.assertIsNone(states['uncertain']['match_fraction_all_reference_supported'])
        self.assertEqual(len(details), 5)

    def test_hash_mismatch_or_duplicate_slots_rejected(self):
        refs, predictions = fixture()
        changed = deepcopy(predictions)
        changed[0]['source_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            audit.replay(refs, changed)
        changed[0] = deepcopy(changed[1])
        with self.assertRaises(ValueError):
            audit.replay(refs, changed)

    def test_clinical_reference_promotion_rejected(self):
        refs, predictions = fixture()
        refs[0]['projection']['current_lung_opacity_reference'] = 'positive'
        with self.assertRaises(ValueError):
            audit.replay(refs, predictions)

    def test_failed_prediction_cannot_be_an_unknown_prediction(self):
        refs, predictions = fixture()
        predictions[1]['finding_states'] = dict.fromkeys(audit.HEADS, 'unknown')
        with self.assertRaises(ValueError):
            audit.replay(refs, predictions)

    def test_tampered_score_detected_without_production_import(self):
        refs, preds = fixture()
        numeric, states, details = audit.replay(refs, preds)
        evaluation = {**numeric, 'per_reference_state': states}
        for flag in ('clinical_qualified', 'primary_metric_eligible', 'selection_changed', 'regeneration_authorized',
            'untouched_final_test', 'thresholds_fitted', 'vocabulary_and_scope_equivalence_verified',
            'independent_image_truth', 'literal_unknown_determinate_is_clinical_hallucination'):
            evaluation[flag] = False
        comparison = [dict(row, clinical_error_adjudicated=False) for row in details]
        self.assertEqual(audit.check_evaluation(evaluation, preds, refs, comparison), numeric)
        evaluation['known_literal_match_fraction_all_supported'] = 1
        with self.assertRaises(ValueError):
            audit.check_evaluation(evaluation, preds, refs, comparison)
        imports = [n.module for n in ast.walk(ast.parse(audit.Path(audit.__file__).read_text()))
                   if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any('manual_opacity_reader_diagnostic' in (module or '') for module in imports))

    def test_no_source_or_inference_and_login_execution_guard(self):
        self.assertTrue(any(name.endswith('dev.json') for name in audit.NO_REOPEN))
        self.assertTrue(any(name.endswith('chexbert.pth') for name in audit.NO_REOPEN))
        with patch.object(audit.sys, 'argv', ['audit']), patch.object(audit.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(audit.Path, 'read_text', return_value='/login'), patch.object(audit, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(audit.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

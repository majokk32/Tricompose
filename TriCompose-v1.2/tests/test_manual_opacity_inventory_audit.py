"""Independent replay tests: invented numeric/coordinate metadata only."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import audit_manual_opacity_inventory as audit


def row(index, state='positive'):
    has_head = state != 'unknown'
    projection = {'source_sha256': 'a' * 64, 'current_lung_opacity_reference': None,
        'manual_literal_state': state,
        'literal_mentions': [{'head_start': 2, 'head_end_exclusive': 9, 'annotated_observation_count': 1}] if has_head else [],
        'human_span_evidence': [{'head_start': 2, 'head_end_exclusive': 9, 'char_start': 2,
            'char_end_exclusive': 9, 'human_native_state': state}] if has_head else [],
        'literal_mention_count': int(has_head), 'unannotated_literal_mentions': 0}
    for f in ('current_patient_scope_verified', 'qualifier_scope_verified', 'pulmonary_anatomy_scope_verified',
        'generic_normality_expanded', 'synonyms_added', 'clinical_qualified', 'selection_changed', 'regeneration_authorized'):
        projection[f] = False
    return {'report_id': f'report_{index:04d}', 'source_index': index, 'source_sha256': 'a' * 64,
        'status': 'complete', 'failure_type': None, 'projection': projection}


class ManualOpacityAuditTests(unittest.TestCase):
    def test_states_failures_duplicates_overlap_all_retained(self):
        rows = [row(i, s) for i, s in enumerate(audit.STATES)]
        failed = row(4)
        failed.update(status='failed_unavailable', failure_type='ValueError', projection=None)
        rows.append(failed)
        counts = audit.replay(rows, {'a' * 64})
        self.assertEqual(counts['manual_literal_state_counts'], dict.fromkeys(audit.STATES, 1))
        self.assertEqual(counts['attempted_reports'], 5)
        self.assertEqual(counts['failed_reports'], 1)
        self.assertEqual(counts['prior_test_source_hash_overlap_slots'], 5)
        self.assertEqual(counts['duplicate_source_hash_slots_retained'], 4)

    def test_state_flip_or_clinical_promotion_rejected(self):
        for field, value in (('manual_literal_state', 'negative'), ('current_lung_opacity_reference', 'positive'),
                             ('regeneration_authorized', True), ('literal_mention_count', 3)):
            r = row(0)
            r['projection'][field] = value
            with self.assertRaises(ValueError):
                audit.replay([r], set())

    def test_bad_evidence_coordinates_or_counts_rejected(self):
        for field, value in (('head_start', 1), ('char_end_exclusive', 8), ('human_native_state', 'unknown')):
            r = row(0)
            r['projection']['human_span_evidence'][0][field] = value
            with self.assertRaises(ValueError):
                audit.replay([r], set())

    def test_original_key_or_bad_order_rejected(self):
        for r in (row(1), {**row(0), 'original_patient_key': 'invented-not-real'}):
            with self.assertRaises(ValueError):
                audit.replay([r], set())

    def test_unannotated_mention_stays_unknown(self):
        r = row(0, 'unknown')
        r['projection'].update(literal_mentions=[{'head_start': 2, 'head_end_exclusive': 9, 'annotated_observation_count': 0}],
                              literal_mention_count=1, unannotated_literal_mentions=1)
        self.assertEqual(audit.replay([r], set())['manual_literal_state_counts']['unknown'], 1)

    def test_no_production_projection_import_and_login_guard(self):
        import ast
        code = audit.Path(audit.__file__).read_text()
        imports = [n.module for n in ast.walk(ast.parse(code)) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any('manual_opacity_inventory' in (n or '') for n in imports))
        with patch.object(audit.sys, 'argv', ['audit']), patch.object(audit.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(audit.Path, 'read_text', return_value='/login'), patch.object(audit, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(audit.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

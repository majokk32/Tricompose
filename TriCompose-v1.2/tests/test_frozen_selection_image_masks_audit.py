"""Independent audit helper checks with invented, non-patient metadata."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import audit_frozen_selection_masks as audit
from test_frozen_selection_image_masks import fixtures


class FrozenMaskAuditTests(unittest.TestCase):
    def test_independent_counts_known_unknown_and_abstention(self):
        t, rows, _ = fixtures()
        value = audit.trial_counts(t[0], rows[0])
        self.assertEqual(value[audit.POLICIES[0]]['proxy_support'], 1)
        self.assertEqual(value[audit.POLICIES[-1]]['not_comparable'], 1)
        self.assertEqual(audit.trial_counts(t[-1], rows[1])[audit.POLICIES[0]]['comparable'], 0)

    def test_missing_choice_retains_uncomparable_count(self):
        t, _, _ = fixtures()
        row = deepcopy(t[0])
        row['selected_candidate_id'] = None
        value = audit.trial_counts(row, None)
        self.assertTrue(all(x['selected'] == x['accepted_image'] == 0 and x['not_comparable'] == 1 for x in value.values()))

    def test_artifact_substitution_rejected(self):
        t, rows, _ = fixtures()
        with self.assertRaises(ValueError):
            audit.trial_counts(t[0], rows[1])
        with self.assertRaises(ValueError):
            audit.trial_counts(t[0], None)

    def test_null_is_not_zero_and_numeric_tolerance(self):
        audit.close('', None)
        audit.close('0.5', .5)
        for actual, expected in ((0, None), ('', 0), ('nan', 0), (True, 1), (.1, .2)):
            with self.assertRaises(ValueError):
                audit.close(actual, expected)

    def test_production_comparison_module_not_imported(self):
        import ast
        tree = ast.parse(audit.Path(audit.__file__).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any('frozen_selection_image_masks' in (n or '') or 'candidate_image_masks' in (n or '') for n in imports))

    def test_guard_before_metadata_read(self):
        with patch.object(audit.sys, 'argv', ['audit']), \
             patch.object(audit.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(audit.Path, 'read_text', return_value='/login'), \
             patch.object(audit, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(audit.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

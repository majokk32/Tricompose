"""Head identities and launcher boundaries; invented metadata only."""
import unittest
from unittest.mock import patch

import analyze_manual_reader_head_risk as worker
from test_manual_reader_risk_coverage import fixtures


class HeadRiskWorkerTests(unittest.TestCase):
    def test_source_inventory_has_existing_code_paths(self):
        code = [p for p in worker.sources() if p.suffix == '.py']
        self.assertEqual(len(code), 15)
        self.assertTrue(all(p.is_file() and p.resolve().is_relative_to(worker.WORKSPACE) for p in code))
        self.assertEqual(len(set(worker.sources())), len(worker.sources()))

    def test_finding_survives_every_table(self):
        result = worker.analyze_heads(*fixtures(), repetitions=100)
        short, long, pairs = worker.tables(result)
        self.assertEqual((len(short), len(long), len(pairs)), (168, 1512, 648))
        for rows in (short, long, pairs):
            self.assertEqual({r['finding'] for r in rows}, set(worker.FINDINGS))
            self.assertTrue(all(r['clinical_qualified'] is False for r in rows))

    def test_missing_risk_not_zero_in_csv_row(self):
        result = worker.analyze_heads(*fixtures(), repetitions=100)
        short, _, _ = worker.tables(result)
        r = next(r for r in short if r['finding'] == worker.FINDINGS[-1])
        self.assertIsNone(r['conditional_literal_assertion_error'])
        self.assertIsNone(r['risk_interval_lower'])
        self.assertEqual(r['risk_valid_draws'], 0)

    def test_no_real_read_on_login_node(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(worker.Path, 'read_text', return_value='/login_node'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_no_cli_scope_expansion(self):
        with patch.object(worker.sys, 'argv', ['worker', '--choose-model']), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

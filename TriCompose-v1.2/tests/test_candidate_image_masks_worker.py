"""Launcher guards and local code inventory, no candidate/source IO."""
import unittest
from unittest.mock import patch

import annotate_candidate_image_masks as worker


class CandidateMaskWorkerTests(unittest.TestCase):
    def test_code_dependencies_exist(self):
        code = [p for p in worker.sources() if p.suffix == '.py']
        self.assertEqual(len(code), 14)
        self.assertTrue(all(p.is_file() and p.resolve().is_relative_to(worker.WORKSPACE) for p in code))
        self.assertEqual(len(set(worker.sources())), len(worker.sources()))

    def test_actual_slurm_cgroup_before_source_load(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(worker.Path, 'read_text', return_value='/login'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_cannot_change_mask_or_candidate_scope_by_argument(self):
        with patch.object(worker.sys, 'argv', ['worker', '--pick-best']), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

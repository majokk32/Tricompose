"""Launcher boundary and local code inventory; no score/patient IO."""
import unittest
from unittest.mock import patch

import benchmark_cached_image_consensus_v2 as worker


class CachedImageLauncherRevisionTests(unittest.TestCase):
    def test_local_code_dependency_paths_exist(self):
        code = [p for p in worker.sources() if p.suffix == '.py']
        self.assertEqual(len(code), 15)
        self.assertTrue(all(p.is_file() and p.resolve().is_relative_to(worker.WORKSPACE) for p in code))
        self.assertEqual(len(set(worker.sources())), len(worker.sources()))

    def test_actual_cgroup_required_before_source_access(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(worker.Path, 'read_text', return_value='/login_node'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_no_scope_expansion_by_argument(self):
        with patch.object(worker.sys, 'argv', ['worker', '--change-threshold']), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

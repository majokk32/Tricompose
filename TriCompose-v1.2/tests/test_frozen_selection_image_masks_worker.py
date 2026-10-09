import unittest
from unittest.mock import patch

import compare_frozen_selection_masks as worker


class FrozenMaskWorkerTests(unittest.TestCase):
    def test_local_source_inventory_complete(self):
        code = [p for p in worker.sources() if p.suffix == '.py']
        self.assertEqual(len(code), 15)
        self.assertTrue(all(p.is_file() and p.resolve().is_relative_to(worker.WORKSPACE) for p in code))
        self.assertEqual(len(set(worker.sources())), len(worker.sources()))

    def test_login_node_cannot_load_inputs(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(worker.Path, 'read_text', return_value='/login'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_arguments_cannot_choose_favorable_mask(self):
        with patch.object(worker.sys, 'argv', ['worker', '--mask', 'best']), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_other_job_or_gpu_cannot_load_inputs(self):
        for environ in ({'SLURM_JOB_ID': '1'}, {'SLURM_JOB_ID': '12784259', 'SLURM_JOB_GPUS': '0'}):
            with patch.object(worker.sys, 'argv', ['worker']), patch.object(worker.os, 'environ', environ), \
                 patch.object(worker.Path, 'read_text', return_value='/slurm/job_' + environ['SLURM_JOB_ID'] + '/step_batch'), \
                 patch.object(worker, 'execute') as execute, patch('builtins.print'):
                self.assertEqual(worker.main(), 1)
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

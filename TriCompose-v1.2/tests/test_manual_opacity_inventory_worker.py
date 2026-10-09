"""Invented documents and launcher guards only; no real source consumption."""
import json
import unittest
from unittest.mock import patch

import inventory_manual_opacity_dev as worker


def document(state='Observation-Present'):
    return {'doc_key': 'fictional-fixture-only', 'sentences': [['A', 'lung', 'opacity', 'is', 'described.']],
        'ner': [[[2, 2, state]]], 'relations': [[]], 'entity_attributes': [[]]}


class ManualOpacityWorkerTests(unittest.TestCase):
    def test_authored_native_document_projects_without_source_key_or_words(self):
        row = worker.project_document(document(), 0)
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['projection']['manual_literal_state'], 'positive')
        self.assertNotIn('fictional-fixture-only', json.dumps(row))
        self.assertNotIn('described.', json.dumps(row))

    def test_invalid_native_schema_retained_as_failure(self):
        doc = document()
        doc['ner'][0][0][2] = 'Unsupported'
        row = worker.project_document(doc, 3)
        self.assertEqual(row['status'], 'failed_unavailable')
        self.assertIsNone(row['projection'])
        self.assertEqual(row['report_id'], 'report_0003')
        self.assertEqual(row['failure_type'], 'ValueError')

    def test_request_and_dependencies_exist(self):
        config = json.loads(worker.CONFIG.read_text())
        self.assertEqual(config['literal_pattern'], worker.PATTERN)
        self.assertEqual(config['source_file'], 'source/manual_data/dev.json')
        self.assertTrue(all(p.is_file() and p.resolve().is_relative_to(worker.WORKSPACE) for p in worker.sources()))
        self.assertFalse(config['regeneration_authorized'])

    def test_current_cache_only_allocation_cannot_read_raw_input(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '12784259',
                 'TRICOMPOSE_MANUAL_OPACITY_REQUEST': 'manual-opacity-inventory-request-v1'}), \
             patch.object(worker.Path, 'read_text', return_value='/slurm/job_12784259/step_batch'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_login_gpu_missing_scope_or_arguments_cannot_load_source(self):
        attempts = [(['worker'], {}, '/login'), (['worker'], {'SLURM_JOB_ID': '123'}, '/slurm/job_123/step_batch'),
            (['worker', '--choose-positive'], {'SLURM_JOB_ID': '123'}, '/slurm/job_123/step_batch'),
            (['worker'], {'SLURM_JOB_ID': '123', 'SLURM_JOB_GPUS': '0',
                'TRICOMPOSE_MANUAL_OPACITY_REQUEST': 'manual-opacity-inventory-request-v1'}, '/slurm/job_123/step_batch')]
        for argv, env, cgroup in attempts:
            with patch.object(worker.sys, 'argv', argv), patch.object(worker.os, 'environ', env), \
                 patch.object(worker.Path, 'read_text', return_value=cgroup), patch.object(worker, 'execute') as execute, patch('builtins.print'):
                self.assertEqual(worker.main(), 1)
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()

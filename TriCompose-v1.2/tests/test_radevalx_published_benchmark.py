"""Mocked Slurm/access gates; no source dataset or network."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'tools/run_radevalx_published_benchmark.py'
spec=importlib.util.spec_from_file_location('test_published_radevalx_worker',PATH)
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerGuardTests(unittest.TestCase):
    def test_no_slurm_refused_before_any_source_or_directory_access(self):
        with patch.object(worker.guard,'guard',side_effect=RuntimeError('not_slurm')), \
                patch.object(worker.guard,'metadata') as loaded, patch.object(worker,'new_atomic_run') as created:
            with self.assertRaises(RuntimeError):
                worker.evaluate(Path('invented'),'a'*64,'invented',True)
            loaded.assert_not_called()
            created.assert_not_called()

    def test_no_scope_approval_refused_before_source_access(self):
        with patch.object(worker.guard,'guard'), patch.object(worker.guard,'metadata') as loaded, \
                patch.object(worker,'new_atomic_run') as created:
            with self.assertRaises(ValueError):
                worker.evaluate(Path('invented'),'a'*64,'invented',False)
            loaded.assert_not_called()
            created.assert_not_called()

    def test_only_fixed_release_files_accepted(self):
        with self.assertRaises(ValueError):
            worker.csv_rows('../outside.csv')

    def test_fixed_metric_and_reference_field_inventory(self):
        self.assertEqual(len(worker.EXPECTED_HEADERS),3)
        self.assertEqual(worker.EXPECTED_HEADERS['metrcis_scores_m2tr.csv'],list(worker.adapter.METRIC_FIELDS))
        for name in worker.EXPECTED_HEADERS:
            if name!='metrcis_scores_m2tr.csv':
                self.assertEqual(worker.EXPECTED_HEADERS[name],list(worker.adapter.ANNOTATION_FIELDS))


if __name__=='__main__':
    unittest.main()

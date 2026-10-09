"""Mocked worker access gates only, no clinical data or network."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'tools/run_radevalx_published_benchmark_v2.py'
spec=importlib.util.spec_from_file_location('test_published_radevalx_worker_v2',PATH)
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerV2GuardTests(unittest.TestCase):
    def test_no_slurm_or_scope_approval_refused_before_sources(self):
        for allowed,failure in ((True,RuntimeError('not_slurm')),(False,None)):
            with patch.object(worker.base.guard,'guard',side_effect=failure), \
                    patch.object(worker.base.guard,'metadata') as read:
                with self.assertRaises((RuntimeError,ValueError)):
                    worker.evaluate(Path('invented'),'a'*64,'invented',allowed)
                read.assert_not_called()

    def test_unknown_source_file_refused_before_csv_read(self):
        with self.assertRaises(ValueError):
            worker.rows('../outside.csv')


if __name__=='__main__':
    unittest.main()

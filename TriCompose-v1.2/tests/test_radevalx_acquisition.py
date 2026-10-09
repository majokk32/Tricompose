"""Mock/invented release metadata; no network or clinical input."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'tools/acquire_radevalx_benchmark.py'
spec = importlib.util.spec_from_file_location('test_radevalx_acquisition_worker', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class AcquisitionTests(unittest.TestCase):
    def test_fixed_checksum_inventory(self):
        text = ''.join('a'*64 + '  ' + name + '\n' for name in worker.FILES)
        self.assertEqual(set(worker.released_digests(text.encode())), set(worker.FILES))

    def test_bad_duplicate_path_or_extra_checksum_refused(self):
        valid = ''.join('a'*64 + '  ' + name + '\n' for name in worker.FILES)
        for extra in ('a'*64 + '  ../outside\n', 'a'*64 + '  unexpected\n',
                      'a'*64 + '  ' + worker.FILES[0] + '\n', 'not_checksum\n'):
            with self.assertRaises(ValueError):
                worker.released_digests((valid+extra).encode())

    def test_only_header_inspected(self):
        self.assertEqual(worker.header_only(b'report_id,ground_truth,1,2\ninvalid_never_read_\xff'),
                         ['report_id','ground_truth','1','2'])

    def test_duplicate_unsafe_or_oversized_header_refused(self):
        for data in (b'id,id\n',b'id,unsafe<column>\n',b'a'*8193+b'\n'):
            with self.assertRaises(ValueError):
                worker.header_only(data)

    def test_unknown_remote_filename_refused_before_network(self):
        with patch.object(worker.urllib.request, 'urlopen') as opened:
            with self.assertRaises(ValueError):
                worker.retrieve('../outside')
            opened.assert_not_called()

    def test_actual_slurm_guard_precedes_network_or_output_creation(self):
        with patch.object(worker,'guard',side_effect=RuntimeError('not_slurm')), \
                patch.object(worker,'retrieve') as opened, patch.object(worker,'new_atomic_run') as created:
            with self.assertRaises(RuntimeError):
                worker.acquire('invented_run',True)
            opened.assert_not_called()
            created.assert_not_called()

    def test_approval_precedes_network_or_output_creation(self):
        with patch.object(worker,'guard'), patch.object(worker,'retrieve') as opened, \
                patch.object(worker,'new_atomic_run') as created:
            with self.assertRaises(ValueError):
                worker.acquire('invented_run',False)
            opened.assert_not_called()
            created.assert_not_called()


if __name__ == '__main__':
    unittest.main()

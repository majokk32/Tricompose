"""Only authored checksum metadata; no clinical input or network."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'tools/acquire_radevalx_benchmark_v2.py'
spec = importlib.util.spec_from_file_location('test_radevalx_acquisition_v2', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ChecksumSyntaxTests(unittest.TestCase):
    def test_single_space_double_space_and_tab_metadata(self):
        for separator in (' ', '  ', '\t'):
            payload = ''.join('a'*64 + separator + name + '\n' for name in worker.base.FILES).encode()
            self.assertEqual(set(worker.released_digests(payload)), set(worker.base.FILES))

    def test_checksum_values_unchanged(self):
        payload = ''.join(('a' if i == 0 else 'b')*64+' '+name+'\n'
                          for i,name in enumerate(worker.base.FILES)).encode()
        self.assertEqual(worker.released_digests(payload)[worker.base.FILES[0]], 'a'*64)

    def test_unknown_and_traversal_names_refused(self):
        for name in ('../outside','unknown','/absolute'):
            with self.assertRaises(ValueError):
                worker.released_digests(('a'*64+' '+name+'\n').encode())

    def test_missing_and_duplicate_files_refused(self):
        valid = ''.join('a'*64+' '+name+'\n' for name in worker.base.FILES)
        for text in (valid.splitlines()[0], valid+valid.splitlines()[0]+'\n'):
            with self.assertRaises(ValueError):
                worker.released_digests(text.encode())


if __name__ == '__main__':
    unittest.main()

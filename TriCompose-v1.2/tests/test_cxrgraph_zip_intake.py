"""ZIP/checksum safety tests on authored metadata, never clinical JSON."""

import importlib.util
import io
from pathlib import Path
import stat
import unittest
import zipfile


WORKER = Path(__file__).resolve().parents[2] / 'tools/intake_cxrgraph_zip.py'
SPEC = importlib.util.spec_from_file_location('cxrgraph_intake_worker', WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class MetadataTests(unittest.TestCase):
    def test_safe_fixed_names(self):
        for name in worker.FILES:
            self.assertEqual(worker.safe_name(name), name)

    def test_reject_unsafe_names(self):
        for name in ('../x', '/x', 'a/../x', 'a\\x', 'a//x', './x', '', 'a\x00x'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                worker.safe_name(name)

    def test_checksum_formats(self):
        for separator in (' ', '  ', ' *', '\t'):
            payload = ('a' * 64 + separator + './manual_data/test.json\n').encode()
            self.assertEqual(worker.parse_checksums(payload),
                             {'manual_data/test.json': 'a' * 64})

    def test_checksum_path_rejected(self):
        with self.assertRaises(ValueError):
            worker.parse_checksums(('a' * 64 + '  ../outside\n').encode())

    def test_duplicate_checksum_rejected(self):
        row = 'a' * 64 + '  manual_data/test.json\n'
        with self.assertRaises(ValueError):
            worker.parse_checksums((row + row).encode())

    def test_invalid_checksum_rejected(self):
        with self.assertRaises(ValueError):
            worker.parse_checksums(b'not a checksum\n')

    def inventory_for(self, info):
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w') as archive:
            archive.writestr(info, b'authored metadata fixture')
        memory.seek(0)
        with zipfile.ZipFile(memory) as archive:
            return worker.member_inventory(archive)

    def test_regular_member(self):
        self.assertIn('manual_data/test.json',
                      self.inventory_for('manual_data/test.json'))

    def test_traversal_member_rejected(self):
        with self.assertRaises(ValueError):
            self.inventory_for('../outside')

    def test_link_member_rejected(self):
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            self.inventory_for(info)

    def test_special_file_rejected(self):
        info = zipfile.ZipInfo('pipe')
        info.create_system = 3
        info.external_attr = (stat.S_IFIFO | 0o600) << 16
        with self.assertRaises(ValueError):
            self.inventory_for(info)


if __name__ == '__main__':
    unittest.main()

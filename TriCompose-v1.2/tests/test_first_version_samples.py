"""Invented case-order/byte-copy fixtures, no real or generated clinical data."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('first_version_samples_fixture', ROOT / 'tools/export_first_version_samples.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class FirstVersionSampleTests(unittest.TestCase):
    def test_numeric_opaque_case_order_not_score(self):
        rows = [{'case_id': 'case_' + str(i), 'score': 100 - i} for i in reversed(range(80))]
        before = copy.deepcopy(rows)
        self.assertEqual([r['case_id'] for r in w.choose_cases(rows)], ['case_0', 'case_1'])
        self.assertEqual(rows, before)

    def test_extra_cases_missing_duplicate_wrong_count_refused(self):
        rows = [{'case_id': 'case_%03d' % i} for i in range(80)]
        for value in (rows[:-1], rows + [rows[0]], [rows[0]] * 80):
            with self.assertRaises(ValueError): w.choose_cases(value)
        for count in (1, 3, True):
            with self.assertRaises(ValueError): w.choose_cases(rows, count)

    def test_nonopaque_identifier_refused(self):
        rows = [{'case_id': 'case_%03d' % i} for i in range(80)]
        rows[0]['case_id'] = '../invented_patient'
        with self.assertRaises(ValueError): w.choose_cases(rows)

    def test_existing_destination_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw); source = p / 'source.bin'; target = p / 'target.bin'
            source.write_bytes(b'wholly invented bytes'); target.write_bytes(b'preserve')
            with patch.object(w.delivery, 'bounded', return_value=source):
                with self.assertRaises(FileExistsError): w.copy_bytes(source, 'a' * 64, target, {}, 'fixture')
            self.assertEqual(target.read_bytes(), b'preserve')

    def test_byte_copy_has_private_mode_and_exact_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw); source = p / 'source.bin'; target = p / 'target.bin'
            source.write_bytes(b'wholly invented non-clinical bytes')
            digest = w.sha256_file(source)
            with patch.object(w.delivery, 'bounded', return_value=source):
                w.copy_bytes(source, digest, target, {}, 'fixture')
            self.assertEqual(w.sha256_file(target), digest)
            self.assertEqual(target.stat().st_mode & 0o777, 0o660)

    def test_source_hash_guard_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / 'target.bin'
            with patch.object(w.delivery, 'bounded', side_effect=ValueError('invented_hash_failure')):
                with self.assertRaises(ValueError): w.copy_bytes('/invented/source', 'a' * 64, target, {}, 'fixture')
            self.assertFalse(target.exists())

    def test_cpu_guard_before_private_inputs(self):
        with patch.object(w.delivery.cached, 'cpu_guard', side_effect=RuntimeError('guard')), \
                patch.object(w.delivery, 'manifest') as reader:
            with self.assertRaises(RuntimeError): w.export('fixture')
            reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()

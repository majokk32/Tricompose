"""Legacy wrapper contract tests, invented words only; no parser or data IO."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'interfaces/chexpert_negbio_manual100_v1.py'
spec = importlib.util.spec_from_file_location('_test_manual_negbio_request_contract', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixtures():
    text = 'Invented finding .'
    docs = [{'sentences': [['Invented', 'finding', '.']], 'doc_key': 'DO_NOT_SEND',
             'ner': 'DO_NOT_SEND'} for _ in range(100)]
    cohort = [{'report_id': f'report_{i:04d}', 'source_index': i, 'status': 'validated',
               'source_report_sha256': hashlib.sha256(text.encode()).hexdigest()} for i in range(100)]
    return docs, cohort


class ManualNegBioWorkerTests(unittest.TestCase):
    def test_exact_hundred_requests_keys_only(self):
        docs, cohort = fixtures()
        requests = worker.input_items(docs, cohort)
        self.assertEqual(len(requests), 100)
        self.assertTrue(all(set(r) == {'item_id', 'text', 'report_sha256'} for r in requests))
        self.assertNotIn('DO_NOT_SEND', str(requests))

    def test_subset_cannot_replace_cohort(self):
        docs, cohort = fixtures()
        with self.assertRaises(ValueError):
            worker.input_items(docs[:10], cohort[:10])

    def test_fixed_release_order_required(self):
        docs, cohort = fixtures()
        cohort.reverse()
        with self.assertRaises(ValueError):
            worker.input_items(docs, cohort)

    def test_unvalidated_case_fails_closed(self):
        docs, cohort = fixtures()
        cohort[0]['status'] = 'unavailable'
        with self.assertRaises(ValueError):
            worker.input_items(docs, cohort)

    def test_input_hash_required(self):
        docs, cohort = fixtures()
        cohort[0]['source_report_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            worker.input_items(docs, cohort)

    def test_no_source_token_whitespace_rewrite(self):
        docs, cohort = fixtures()
        docs[0]['sentences'][0][0] = 'Invented extra'
        with self.assertRaises(ValueError):
            worker.input_items(docs, cohort)

    def test_bounded_parser_text(self):
        docs, cohort = fixtures()
        docs[0]['sentences'][0] = ['X' * 32769]
        cohort[0]['source_report_sha256'] = hashlib.sha256(('X' * 32769).encode()).hexdigest()
        with self.assertRaises(ValueError):
            worker.input_items(docs, cohort)

    def test_legacy_syntax_compatible(self):
        tree = ast.parse(PATH.read_text(), feature_version=(3, 6))
        self.assertFalse(any(isinstance(n, ast.AnnAssign) for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()

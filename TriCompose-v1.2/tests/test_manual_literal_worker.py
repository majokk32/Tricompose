"""Source request plumbing using invented documents only; no model or IO."""
import hashlib
import unittest

from benchmark_manual_literal_readers import requests_from_documents


def fixtures():
    text = 'Invented cardiomegaly .'
    documents = [{'sentences': [['Invented', 'cardiomegaly', '.']],
                  'doc_key': 'DO_NOT_EXPORT', 'ner': 'DO_NOT_PASS_TO_MODEL'}]
    cohort = [{'report_id': 'report_0000', 'source_index': 0, 'status': 'validated',
               'source_report_sha256': hashlib.sha256(text.encode()).hexdigest()}]
    return documents, cohort


class ManualLiteralWorkerTests(unittest.TestCase):
    def test_source_keys_annotations_not_in_request(self):
        docs, cohort = fixtures()
        texts = requests_from_documents(docs, cohort)
        self.assertEqual(texts, {'report_0000': 'Invented cardiomegaly .'})
        self.assertNotIn('DO_NOT', str(texts))

    def test_hash_changed_fails(self):
        docs, cohort = fixtures()
        cohort[0]['source_report_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            requests_from_documents(docs, cohort)

    def test_release_order_cannot_be_reselected(self):
        docs, cohort = fixtures()
        cohort[0]['source_index'] = 1
        with self.assertRaises(ValueError):
            requests_from_documents(docs, cohort)

    def test_no_missing_source_case(self):
        docs, cohort = fixtures()
        with self.assertRaises(ValueError):
            requests_from_documents([], cohort)

    def test_unvalidated_reference_retained_not_used(self):
        docs, cohort = fixtures()
        cohort[0]['status'] = 'unavailable'
        with self.assertRaises(ValueError):
            requests_from_documents(docs, cohort)

    def test_whitespace_inside_source_token_not_rewritten(self):
        docs, cohort = fixtures()
        docs[0]['sentences'][0][0] = 'Invented\nextra'
        with self.assertRaises(ValueError):
            requests_from_documents(docs, cohort)


if __name__ == '__main__':
    unittest.main()

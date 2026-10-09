"""Authored fixtures only; source reports and checkpoints are never opened."""
from copy import deepcopy
import json
import unittest

from prepare_cxrgraph_manual_cohort import read_documents, split_membership, normalize_documents
from score_cxrgraph_manual_xl import aggregates
from test_cxrgraph_gold_adapter import document
from tricompose_v12.cxrgraph_gold_adapter import compare_to_native_prediction
from tricompose_v12.entity_gold_contract import digest


class ManualCohortTests(unittest.TestCase):
    def docs(self):
        a, b = document(), document()
        a['doc_key'], b['doc_key'] = 'authored_a', 'authored_b'
        return [a, b]

    def test_json_array(self):
        docs = self.docs()
        loaded, fmt = read_documents(json.dumps(docs).encode())
        self.assertEqual(loaded, docs)
        self.assertEqual(fmt, 'json_array')

    def test_json_lines(self):
        docs = self.docs()
        loaded, fmt = read_documents(('\n'.join(json.dumps(d) for d in docs) + '\n').encode())
        self.assertEqual(loaded, docs)
        self.assertEqual(fmt, 'json_lines')

    def test_single_document(self):
        doc = document()
        self.assertEqual(read_documents(json.dumps(doc).encode()), ([doc], 'single_json_document'))

    def test_empty_file_rejected(self):
        with self.assertRaises(ValueError):
            read_documents(b'')

    def test_empty_inventory_rejected(self):
        with self.assertRaises(ValueError):
            read_documents(b'[]')

    def test_duplicate_internal_keys_rejected(self):
        with self.assertRaises(ValueError):
            read_documents(json.dumps([document(), document()]).encode())

    def test_non_string_key_rejected(self):
        doc = document()
        doc['doc_key'] = 1
        with self.assertRaises(ValueError):
            read_documents(json.dumps([doc]).encode())

    def test_partition_has_opaque_groups_only(self):
        a, b = self.docs()
        groups = split_membership([a, b], [a], [b])
        self.assertEqual(groups, {0: 'mimic', 1: 'chexpert'})
        self.assertNotIn(a['doc_key'], json.dumps(groups))

    def test_overlapping_domains_rejected(self):
        a, b = self.docs()
        with self.assertRaises(ValueError):
            split_membership([a, b], [a], [a, b])

    def test_incomplete_partition_rejected(self):
        a, b = self.docs()
        with self.assertRaises(ValueError):
            split_membership([a, b], [a], [])

    def test_changed_split_annotation_rejected(self):
        a, b = self.docs()
        changed = deepcopy(b)
        changed['entity_attributes'] = [[], []]
        with self.assertRaises(ValueError):
            split_membership([a, b], [a], [changed])

    def test_normalization_keeps_fixed_source_order(self):
        docs = self.docs()
        rows, gold = normalize_documents(docs, {0: 'mimic', 1: 'chexpert'})
        self.assertEqual([r['report_id'] for r in rows], ['report_0000', 'report_0001'])
        self.assertEqual([r['source_index'] for r in rows], [0, 1])
        self.assertEqual([r['status'] for r in rows], ['validated', 'validated'])
        for doc in docs:
            self.assertNotIn(doc['doc_key'], json.dumps(rows))
        self.assertEqual(len(gold), 2)

    def test_failure_not_dropped_or_imputed(self):
        a, b = self.docs()
        b['ner'][0][0][2] = 'invalid_authored_type'
        rows, gold = normalize_documents([a, b], {0: 'mimic', 1: 'chexpert'})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(gold), 1)
        self.assertEqual(rows[1]['status'], 'schema_unavailable')
        self.assertEqual(rows[1]['failure_type'], 'ValueError')
        self.assertIsNone(rows[1]['adapter_sha256'])
        self.assertNotIn('invalid_authored_type', json.dumps(rows))

    def test_deterministic_normalization(self):
        docs = self.docs()
        groups = {0: 'mimic', 1: 'chexpert'}
        self.assertEqual(normalize_documents(docs, groups), normalize_documents(docs, groups))

    def test_aggregate_preserves_missing_domain_and_denominator(self):
        cohort, gold = normalize_documents(self.docs(), {0: 'mimic', 1: 'chexpert'})
        prediction = deepcopy(gold[0]['common_label_record'])
        prediction.update(origin='declared_frozen_prediction', reader_id=None)
        prediction['record_sha256'] = digest({k: v for k, v in prediction.items() if k != 'record_sha256'})
        comparison = compare_to_native_prediction(gold[0], prediction)
        result = aggregates([comparison], cohort)
        self.assertEqual(result['all']['attempted_reports'], 2)
        self.assertEqual(result['all']['eligible_reports'], 1)
        self.assertEqual(result['all']['comparison_availability'], 0.5)
        self.assertEqual(result['mimic']['entity_micro']['f1'], 1)
        self.assertIsNone(result['chexpert']['entity_micro'])
        self.assertEqual(result['chexpert']['unavailable_report_ids'], ['report_0001'])


if __name__ == '__main__':
    unittest.main()

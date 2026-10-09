"""Invented schemas/numbers, with poison report fields that must not be read."""
import importlib.util
import json
from pathlib import Path
import unittest

from tricompose_v12.report_metric_alignment import evaluate

PATH = Path(__file__).resolve().parents[1] / 'real_validation/radevalx_published_adapter.py'
spec = importlib.util.spec_from_file_location('test_radevalx_published_adapter', PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class PoisonRows(dict):
    def __getitem__(self, name):
        if name in ('ground_truth','M2Tr-Generation','id'):
            raise AssertionError('source_text_or_secondary_key_accessed')
        return super().__getitem__(name)


def invented():
    scores, counts = [], []
    for i in range(4):
        row = {'report_id':'invented_private_key_'+str(i), 'id': 'never_read'}
        row.update({name: str(4-i) if name != 'radcliq' else str(i) for name in adapter.METRICS})
        scores.append(PoisonRows(row))
        counts.append(PoisonRows({'report_id': row['report_id'], 'ground_truth': 'never_read',
            'M2Tr-Generation': 'never_read', **{str(n): str(i if n == 1 else 0) for n in range(1,9)}}))
    return scores, counts


class PublishedAdapterTests(unittest.TestCase):
    def test_only_numeric_fields_and_opaque_ids_exported(self):
        scores, counts = invented()
        pred, keys = adapter.predictions(scores, expected_count=4)
        ref = adapter.references(counts, counts, keys)
        serialized = json.dumps((pred,ref))
        self.assertNotIn('invented_private_key',serialized)
        self.assertNotIn('never_read',serialized)
        self.assertNotIn('ground_truth',serialized)
        result = evaluate(pred,ref,bootstrap_resamples=0)
        for row in result['metrics'].values():
            self.assertEqual(row['outcomes']['clinically_significant_total']['spearman'],1)

    def test_missing_scores_are_unavailable_not_zero(self):
        scores, _ = invented()
        scores[0]['bleu4']='NaN'
        pred,_=adapter.predictions(scores,expected_count=4)
        self.assertEqual(pred['records'][0]['scores']['published_bleu4'],
                         {'status':'not_available_in_source','value':None})

    def test_missing_counts_are_null_not_zero(self):
        scores,counts=invented()
        counts[0]['1']=''
        _,keys=adapter.predictions(scores,expected_count=4)
        ref=adapter.references(counts,counts,keys)
        self.assertIsNone(ref['records'][0]['errors']['clinically_significant'][0])

    def test_all_released_inventory_is_fixed(self):
        scores,_=invented()
        with self.assertRaises(ValueError):
            adapter.predictions(scores)

    def test_duplicate_or_unmatched_keys_refused(self):
        scores,counts=invented()
        _,keys=adapter.predictions(scores,expected_count=4)
        counts[1]['report_id']=counts[0]['report_id']
        with self.assertRaises(ValueError):
            adapter.references(counts,counts,keys)
        scores,counts=invented()
        counts[0]['report_id']='invented_unmatched'
        with self.assertRaises(ValueError):
            adapter.references(counts,counts,keys)

    def test_reference_order_is_joined_to_frozen_metric_order(self):
        scores,counts=invented()
        _,keys=adapter.predictions(scores,expected_count=4)
        self.assertEqual(adapter.references(counts,counts,keys),adapter.references(counts[::-1],counts[::-1],keys))

    def test_schema_drift_and_malformed_counts_refused(self):
        scores,counts=invented()
        counts[0]['extra']='0'
        _,keys=adapter.predictions(scores,expected_count=4)
        with self.assertRaises(ValueError):
            adapter.references(counts,counts,keys)
        for value in ('-1','1.5','inf','text','10001'):
            with self.assertRaises(ValueError):
                adapter.cell(value,count=True)

    def test_radcliq_orientation_not_fitted_from_values(self):
        self.assertEqual(adapter.METRICS['radcliq'],('published_radcliq','lower_is_better'))
        self.assertEqual(set(adapter.METRICS),set(adapter.METRIC_FIELDS)-{'report_id','id'})


if __name__=='__main__':
    unittest.main()

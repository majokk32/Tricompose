"""Invented numeric fixtures only; failed GPU jobs are never marked complete."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import read_paired_endpoint_cache_v1 as cache


class PairedCacheTests(unittest.TestCase):
    def test_na_counted_not_replaced_with_zero(self):
        rows=[{'case_id':'case_000','report_model_id':'invented','biovil_raw_cosine':.4},
            {'case_id':'case_001','report_model_id':'invented','biovil_raw_cosine':None}]
        old=deepcopy(rows);result=cache.groups(rows,[])[0]
        self.assertEqual(result['available_scores'],1)
        self.assertEqual(result['unavailable_scores'],1)
        self.assertEqual(result['mean_raw_cosine'],.4);self.assertEqual(rows,old)
        self.assertEqual(result['distinct_ehr_cases'],2);self.assertIsNone(result['clinical_accuracy'])

    def test_slots_are_not_independent_patients(self):
        rows=[{'case_id':f'case_{i//2:03d}','report_model_id':'invented','biovil_raw_cosine':.4} for i in range(4)]
        result=cache.groups(rows,[])[0]
        self.assertEqual(result['candidate_or_selection_slots'],4)
        self.assertEqual(result['distinct_ehr_cases'],2)

    def test_all_na_mean_is_na(self):
        r=cache.groups([],[{'case_id':'case_000','method':'invented','biovil_raw_cosine':None}])[0]
        self.assertIsNone(r['mean_raw_cosine']);self.assertIsNone(r['minimum_raw_cosine'])

    def test_guard_before_any_source_or_model_access(self):
        with patch.object(cache,'cpu_guard',side_effect=RuntimeError('invented_guard')), \
                patch.object(cache.secondary,'load_metadata') as load:
            with self.assertRaises(RuntimeError):cache.run(None)
            load.assert_not_called()

    def test_fixed_source_pins_are_explicit_and_no_trace_loader_used(self):
        self.assertEqual(cache.inputs().source_manifest_sha256,
            'e5979cddf6b2075ea5a3e5ddef2c282a6278e89493144714b3afd1fd4dac0713')
        self.assertEqual(len(cache.NATIVE_SHA),64)
        self.assertEqual(len(cache.FAILURE_SHA),64)
        self.assertNotIn('load_cxr_candidates',cache.run.__code__.co_names)
        self.assertNotIn('read_report_text',cache.run.__code__.co_names)


if __name__=='__main__':unittest.main()

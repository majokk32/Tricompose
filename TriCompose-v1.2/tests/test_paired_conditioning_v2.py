"""Invented traces only; no existing synthetic text or real data opened."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import check_paired_conditioning_v2 as trace
from test_paired_probe_secondary_v1 import conditioning


def data():
    image,request,anchor,observed=conditioning()
    return image,request,observed


class PairedTraceTests(unittest.TestCase):
    def test_normal_capture_matches_old_gate_without_payload_output(self):
        args=data();r=trace.diagnostic(*args)
        self.assertTrue(r['v1_gate_pass']);self.assertEqual(r['matched_phrase_count'],1)
        self.assertNotIn('positive_tokenizer_text',r)
        self.assertFalse(r['clinical_truth_available']);self.assertFalse(r['text_encoder_hook_observed'])

    def test_optional_mask_is_unavailable_not_zero_or_false_negative(self):
        args=data();args[-1]['attention_token_count']=None;r=trace.diagnostic(*args)
        self.assertEqual(r['count_check_status'],'not_available')
        self.assertIsNone(r['attention_equals_adapter_length'])
        self.assertFalse(r['v1_gate_pass']);self.assertFalse(r['scores_or_source_inputs_modified'])

    def test_mismatched_length_not_declared_untruncated(self):
        args=data();args[-1]['attention_token_count']=19;r=trace.diagnostic(*args)
        self.assertEqual(r['count_check_status'],'mismatch')
        self.assertIn('attention_equals_adapter_length',r['v1_failed_checks'])

    def test_bool_is_not_valid_token_count(self):
        args=data();args[-1]['attention_token_count']=True;r=trace.diagnostic(*args)
        self.assertEqual(r['count_check_status'],'not_available')
        self.assertFalse(r['v1_gate_pass'])

    def test_changed_runtime_text_exposed_without_logging_it(self):
        args=data();args[-1]['pipeline_changed_text']=True
        r=trace.diagnostic(*args);self.assertIn('unchanged_text_flag',r['v1_failed_checks'])

    def test_invalid_context_shape_is_visible_not_rewritten(self):
        args=data();args[-1]['padded_token_count']=76;original=deepcopy(args)
        r=trace.diagnostic(*args);self.assertIn('padded_context_length_77',r['v1_failed_checks'])
        self.assertEqual(args,original)

    def test_old_cache_only_allocation_cannot_open_trace(self):
        with patch.object(trace,'cpu_guard'),patch.dict(trace.os.environ,{
            'TRICOMPOSE_SCOPE':'approved_synthetic_trace_diagnostic','SLURM_JOB_ID':'12784259'}):
            with self.assertRaises(RuntimeError):trace.require_trace_allocation()

    def test_guard_precedes_metadata_or_payload_reads(self):
        with patch.object(trace,'require_trace_allocation',side_effect=RuntimeError('invented_guard')), \
                patch.object(trace.cache.secondary,'load_metadata') as load:
            with self.assertRaises(RuntimeError):trace.run(None)
            load.assert_not_called()


if __name__=='__main__':unittest.main()

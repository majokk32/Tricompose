"""Authored legacy-metadata fixtures only; no trace source body is opened."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import check_paired_conditioning_v3 as trace
from test_paired_probe_secondary_v1 import conditioning


def legacy():
    image,request,anchor,observed=conditioning()
    request['inputs']['final_prompt'].update(renderer_version=trace.LEGACY,
        included_direct_fact_ids=['pneumonia','congestive_heart_failure'])
    return image,request,observed


class VersionAwareTraceTests(unittest.TestCase):
    def test_exact_legacy_role_bug_is_reproduced_in_old_checker(self):
        with self.assertRaisesRegex(ValueError,'recognized_unique_rendered_fact_ids_required'):
            trace.previous.diagnostic(*legacy())

    def test_legacy_context_is_not_relabelled_as_edema(self):
        _,request,_=legacy();original=deepcopy(request);r=trace.roles(request)
        self.assertEqual(r['radiographic_ids'],['pneumonia'])
        self.assertEqual(r['legacy_context_ids'],['congestive_heart_failure'])
        self.assertEqual(request,original);self.assertFalse(r['clinical_context_promoted_to_image_finding'])

    def test_valid_capture_and_legacy_schema_failure_are_separate(self):
        args=legacy();original=deepcopy(args);r=trace.diagnostic(*args)
        self.assertTrue(r['length_and_hash_checks_pass'])
        self.assertFalse(r['old_combined_guard_would_pass'])
        self.assertEqual(r['matched_phrase_count'],1)
        self.assertEqual(r['original_declared_direct_id_count'],2)
        self.assertEqual(r['legacy_context_id_count'],1);self.assertEqual(args,original)
        self.assertNotIn('positive_tokenizer_text',r)

    def test_current_schema_cannot_reintroduce_context_as_direct(self):
        _,r,_=legacy();r['inputs']['final_prompt']['renderer_version']=trace.CURRENT
        with self.assertRaisesRegex(ValueError,'context_id_in_current_direct_fact_list'):trace.roles(r)

    def test_unknown_ids_or_renderer_versions_fail_closed(self):
        for key,value in [('renderer_version','unreviewed_version'),
                ('included_direct_fact_ids',['pneumonia','unrecognized_code']),
                ('included_direct_fact_ids',['pneumonia','pneumonia'])]:
            _,r,_=legacy();r['inputs']['final_prompt'][key]=value
            with self.assertRaises(ValueError):trace.roles(r)

    def test_absent_attention_mask_still_unavailable(self):
        args=legacy();args[-1]['attention_token_count']=None;r=trace.diagnostic(*args)
        self.assertEqual(r['count_check_status'],'not_available')
        self.assertFalse(r['length_and_hash_checks_pass'])

    def test_radiographic_and_context_term_counts_are_not_merged(self):
        args=legacy();r=trace.diagnostic(*args)
        self.assertEqual(r['radiographic_id_count'],1)
        self.assertEqual(r['legacy_context_id_count'],1)
        self.assertFalse(r['clinical_context_promoted_to_image_finding'])

    def test_guard_before_source_bodies_and_directory_creation(self):
        with patch.object(trace.previous,'require_trace_allocation',side_effect=RuntimeError('invented_guard')), \
                patch.object(trace,'private_directory') as write:
            with self.assertRaises(RuntimeError):trace.run(None)
            write.assert_not_called()


if __name__=='__main__':unittest.main()

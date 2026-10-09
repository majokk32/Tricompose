"""Invented fixtures only; no model, source payload, credentials or image IO."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import score_paired_probes_v1 as endpoint
from test_paired_probe_audit_v1 import data as invented_data


def conditioning():
    text = 'PA chest radiograph. Findings: airspace opacity compatible with pneumonia.'
    digest = endpoint.text_sha256(text)
    image = {'case_id':'case_000','candidate_id':'invented_image','model_id':'roentgen_v2',
        'seed':3,'prompt_sha256':digest,'cost':{'prompt_token_count':20}}
    request = {'inputs':{'final_prompt':{'sha256':digest,'included_direct_fact_ids':['pneumonia']}}}
    anchor = {'findings':[{'finding':'pneumonia','state':'positive'},
        {'finding':'edema','state':'unknown'},{'finding':'pneumothorax','state':'negative'}]}
    trace = {'positive_tokenizer_text':text,'runtime_observed':True,'supplied_prompt_sha256':digest,
        'pipeline_changed_text':False,'official_generation_arguments_changed':False,
        'positive_tokenizer_text_sha256':digest,'attention_token_count':20,'padded_token_count':77}
    return image,request,anchor,trace


def bank():
    plan,rows,triples = invented_data()
    readouts = []
    for case in plan['cases']:
        ctx = endpoint.audit.paired.fresh.context(case['anchor'],plan['workers']['xrv'],plan['workers']['chexbert'])
        readouts.append({'case_id':case['case_id'],'readout':endpoint.audit.paired.paired_readout(
            [r for r in rows if r['case_id']==case['case_id']],ctx)})
    books = [{'case_id':c['case_id'],'charged_model_attempts':6} for c in plan['cases'] for _ in range(2)]
    records = [{**{k:r[k] for k in endpoint.PAIR_FIELDS}, 'biovil_raw_cosine':.2,
        'status':'computed_secondary_uncalibrated','reason':None,'calibrated':False} for r in rows]
    result = {'schema_version':endpoint.SCORE_SCHEMA,'status':'completed_secondary_biovil',
        'request_sha256':endpoint.REQUEST_SHA,'original_selection_changed':False,
        'primary_clinical_metric':False,'producer':{'frozen':True,'model_id':'biovil_t',
            'checkpoint_sha256':endpoint.MODEL_HASHES.copy(),'text_policy':endpoint.TEXT_POLICY},
        'used_for_routing':False,'clinical_truth_available':False,'historical_pool_is_untouched_test':False,
        'records':records,'counts':{'requested_pairs':8,'image_encoder_calls':4,
            'text_encoder_calls':8,'unavailable_reports':0}}
    return dict(plan=plan, rows=rows, triples=triples, readouts=readouts, books=books),result


class ConditioningTests(unittest.TestCase):
    def test_transfer_receipt_contains_no_text_and_no_clinical_truth(self):
        r = endpoint.conditioning_row(*conditioning())
        self.assertEqual(r['matched_phrase_count'],1)
        self.assertEqual(r['positive_anchor_phrase_match_count'],1)
        self.assertEqual(r['known_positive_anchor_count'],1)
        self.assertFalse(r['text_encoder_hook_observed'])
        self.assertFalse(r['lexical_checks_are_clinical_truth'])
        self.assertNotIn('positive_tokenizer_text',r)

    def test_recorded_truncation_or_prefix_change_refused(self):
        for key,value in [('attention_token_count',19),('padded_token_count',76),
                          ('pipeline_changed_text',True),('runtime_observed',False)]:
            args = conditioning();args[-1][key]=value
            with self.assertRaises(ValueError):endpoint.conditioning_row(*args)

    def test_changed_trace_text_does_not_keep_old_hash(self):
        args = conditioning();args[-1]['positive_tokenizer_text']='invented_changed_text'
        with self.assertRaises(ValueError):endpoint.conditioning_row(*args)

    def test_absent_phrase_is_diagnostic_not_silent_prompt_fix(self):
        args = conditioning();args[1]['inputs']['final_prompt']['included_direct_fact_ids'].append('cardiomegaly')
        original = deepcopy(args)
        r = endpoint.conditioning_row(*args)
        self.assertEqual(r['missing_phrase_count'],1);self.assertEqual(args,original)

    def test_unknown_and_negative_do_not_become_positive_anchors(self):
        args = conditioning();args[2]['findings'][0]['state']='uncertain'
        r = endpoint.conditioning_row(*args)
        self.assertEqual(r['known_positive_anchor_count'],0)
        self.assertEqual(r['positive_anchor_phrase_match_count'],0)

    def test_unmapped_known_head_counted_not_invented(self):
        args = conditioning();args[2]['findings'].append({'finding':'lung_lesion','state':'positive'})
        r = endpoint.conditioning_row(*args)
        self.assertEqual(r['known_positive_anchor_count'],2)
        self.assertEqual(r['mappable_positive_anchor_count'],1)


class EndpointTests(unittest.TestCase):
    def test_join_does_not_modify_candidates_or_selected_methods(self):
        data,result = bank();before = deepcopy(data)
        candidates,methods,actions = endpoint.endpoint_tables(data,result)
        self.assertEqual(data,before);self.assertEqual(len(candidates),8)
        self.assertEqual(len(methods),6);self.assertEqual(len(actions),16)
        self.assertTrue(all(r['biovil_raw_cosine']==.2 for r in candidates))
        self.assertTrue(all(r['actual_shared_collection_attempts']==12 for r in methods))
        self.assertTrue(all(r['clinical_accuracy'] is None for r in methods))
        self.assertTrue(all(r['biovil_delta']==0 for r in actions))
        self.assertTrue(all(r['secondary_used_to_change_acceptance'] is False for r in actions))

    def test_unavailable_full_report_kept_without_zero_imputation(self):
        data,result = bank();r = result['records'][0]
        r.update(biovil_raw_cosine=None,status='not_available',reason='full_report_exceeds_text_context_no_truncation')
        result['counts'].update(text_encoder_calls=7,unavailable_reports=1)
        candidates,methods,actions = endpoint.endpoint_tables(data,result)
        self.assertEqual(len(candidates),8);self.assertIsNone(candidates[0]['biovil_raw_cosine'])
        self.assertEqual(candidates[0]['secondary_status'],'not_available')
        self.assertTrue(any(a['biovil_delta'] is None for a in actions))

    def test_bad_lineage_or_missing_or_duplicate_pair_rejected(self):
        for kind in ('lineage','missing','duplicate'):
            data,result = bank()
            if kind=='lineage':result['records'][0]['cxr_sha256']='9'*64
            elif kind=='missing':result['records'].pop()
            else:result['records'][-1]=deepcopy(result['records'][0])
            with self.assertRaises(ValueError):endpoint.check_endpoint(data['rows'],result)

    def test_wrong_or_unbounded_encoder_counts_rejected(self):
        for key,value in [('image_encoder_calls',5),('text_encoder_calls',9),('requested_pairs',7),
                          ('unavailable_reports',False)]:
            data,result = bank();result['counts'][key]=value
            with self.assertRaises(ValueError):endpoint.check_endpoint(data['rows'],result)

    def test_endpoint_never_promoted_to_clinical_or_router_evidence(self):
        for key in ('used_for_routing','clinical_truth_available','primary_clinical_metric','original_selection_changed'):
            data,result = bank();result[key]=True
            with self.assertRaises(ValueError):endpoint.check_endpoint(data['rows'],result)

    def test_changed_checkpoint_or_request_refused(self):
        data,result = bank();result['producer']['checkpoint_sha256']['pytorch_model.bin']='9'*64
        with self.assertRaises(ValueError):endpoint.check_endpoint(data['rows'],result)
        data,result = bank();result['request_sha256']='9'*64
        with self.assertRaises(ValueError):endpoint.check_endpoint(data['rows'],result)

    def test_gpu_guard_before_any_reads_or_model_imports(self):
        with patch.object(endpoint,'require_gpu_slurm',side_effect=RuntimeError('invented_guard')), \
                patch.object(endpoint,'load_metadata') as load:
            with self.assertRaises(RuntimeError):endpoint.run(None)
            load.assert_not_called()


if __name__=='__main__':unittest.main()

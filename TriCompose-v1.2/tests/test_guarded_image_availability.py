"""Invented metadata only: exact joins, honest availability and lossless tables."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('live_image_availability_fixture',ROOT/'tools/attach_guarded_image_availability.py')
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
g=w.guard


def receipt(uniform=False):
    sha='c'*64 if uniform else 'a'*64
    pixel='d'*64 if uniform else 'b'*64
    value=g.metadata_guard(sha,width=64,height=64,mode='RGB',format_name='PNG',frames=1,
        extrema=((0,0),)*3 if uniform else ((0,255),)*3)
    return g.receipt(sha,value['status'],value['reason'],evidence=value['native_image_metadata'],pixel_sha256=pixel)


def raw(states=None,failed=False):
    return {'contract_status':'failed_unavailable' if failed else 'complete',
        'states':None if failed else dict(states or dict.fromkeys(g.HEADS,'negative')),
        'failure_reason':'invalid_eight_state_json' if failed else None,'token_limit_reached':False,
        'response_sha256':'2'*64,'input_tokens':600,'output_tokens':85,'elapsed_seconds':1.2,
        'independent_clinical_validation':False}


def fixture(reports=1,unchecked=False,old_state='positive',new_state='uncertain'):
    current=receipt()
    old=dict.fromkeys(g.HEADS,'negative')
    fresh=dict(old)
    old['pneumonia']=old_state
    fresh['pneumonia']=new_state
    records=[{'observation_id':'b'*64,'artifact_sha256':'a'*64,'guard':current,'model_called':True,**raw(fresh)},
        {'observation_id':'d'*64,'artifact_sha256':'c'*64,'guard':receipt(True),'model_called':False,
            'contract_status':'blocked_before_model_call','states':None,'failure_reason':'spatially_uniform',
            'token_limit_reached':None,'response_sha256':None,'input_tokens':None,'output_tokens':None,
            'elapsed_seconds':None,'independent_clinical_validation':False}]
    prompt='1'*64
    live={'schema_version':'tricompose-prospective-guarded-image-only-v1','records':records,
        'frozen':True,'image_only':True,'image_prompt_sha256':prompt}
    baseline={'schema_version':'tricompose-image-no-information-control-v1',
        'records':[{'observation_id':'b'*64,**raw(old)}, {'observation_id':'d'*64,**raw(dict.fromkeys(g.HEADS,'unknown'))}],
        'frozen':True,'image_only':True,'image_prompt_sha256':prompt,
        'model_received_arm_names_ehr_reports_ids_scores_or_expected_answers':False}
    slots=[{'slot_id':'original0','cxr_candidate_id':'image0','cxr_sha256':'a'*64,
        'observation_id':'b'*64,'arm':'original'},
        {'slot_id':'control0','cxr_candidate_id':'image0','cxr_sha256':'a'*64,
        'observation_id':'d'*64,'arm':'uniform_black'}]
    rows=[]
    for i in range(reports+int(unchecked)):
        un=unchecked and i==reports
        rows.append({'case_id':'invented_case','triple_candidate_id':'triple'+str(i),
            'cxr_candidate_id':'same_hash_unchecked' if un else 'image0','cxr_sha256':'a'*64,
            'ehr_sha256':'e'*64,'ehr_facts_sha256':'f'*64,'report_candidate_id':'report'+str(i),
            'report_sha256':'3'*64,'old_score':'0.217','old_unavailable_score':'',
            'imageguard_status':'not_checked' if un else current['status'],
            'imageguard_receipt_sha256':'' if un else current['receipt_sha256']})
    reference={k:rows[0][k] for k in ('case_id','cxr_candidate_id','cxr_sha256','ehr_sha256','ehr_facts_sha256')}
    original={'image_inputs':[reference],'image_prompt_sha256':prompt}
    facts=[]
    for row in rows:
        for head in w.CHEXPERT_FINDINGS:
            known=row['cxr_candidate_id']=='image0'
            facts.append({**{k:row[k] for k in ('case_id','triple_candidate_id','cxr_candidate_id','report_candidate_id')},
                'artifact_hashes':{k:row[k] for k in ('ehr_sha256','ehr_facts_sha256','cxr_sha256','report_sha256')},
                'finding':head,'states':{'ehr':'unknown','xrv':'negative','chexbert':'unknown'},
                'imageverify_qwen_state':old[head] if known and head in w.HEADS else None,
                'imageverify_exact_report_comparison_available':known,
                'reportgate_retained_state':None,'old_diagnostic':'unqualified'})
    summary={'actual_model_calls':1,'blocked_before_model_call':1,'unique_inputs':2,'logical_slots':2,
        'clinical_acceptance':False,'primary_metric_eligible':False,'selection_changed':False,
        'fixed_ehr_changed':False,'regeneration_authorized':False,
        'old_prediction_states_parsed_after_prediction_fsync':True,'clinical_requests_resolved':0,
        'model_received_ehr_reports_ids_scores_arm_names_or_expected_answers':False}
    return rows,facts,original,live,slots,baseline,summary


def attach(data):
    rows,facts,original,live,slots,baseline,summary=data
    candidates,records=w.index_evidence(rows,original,live,slots,baseline,summary)
    return w.attach(rows,facts,candidates,records)


class GuardedAvailabilityTests(unittest.TestCase):
    def test_exact_checked_scope_and_no_hash_only_spread(self):
        data=fixture(unchecked=True)
        rows,facts,unique=attach(data)
        self.assertEqual(rows[0]['liveimage_status'],'checked_clinically_unqualified')
        self.assertEqual(rows[1]['liveimage_status'],'not_checked')
        self.assertEqual(rows[0]['liveimage_state_pneumonia'],'uncertain')
        self.assertIsNone(rows[1]['liveimage_state_pneumonia'])
        for key in ('liveimage_complete_state_count','liveimage_changed_readout_count',
                'liveimage_xrv_explicit_opposition_unqualified_count'):
            self.assertIsNone(rows[1][key])
        self.assertEqual(len(unique),14)
        self.assertTrue(all(r['liveimage_status']=='not_checked' for r in facts[14:]))

    def test_every_original_value_and_order_preserved(self):
        data=fixture(reports=4,unchecked=True)
        original=copy.deepcopy(data)
        rows,facts,_=attach(data)
        self.assertEqual(data,original)
        for old,new in zip(data[0],rows):self.assertEqual({k:new[k] for k in old},old)
        for old,new in zip(data[1],facts):self.assertEqual({k:new[k] for k in old},old)

    def test_report_repetition_is_not_independent_image_count(self):
        data=fixture(reports=4)
        rows,facts,unique=attach(data)
        summary=w.summarize(rows,facts,unique,data[-1])
        self.assertEqual(summary['checked_image_slots'],1)
        self.assertEqual(summary['checked_candidate_slots'],4)
        self.assertEqual(summary['unique_in_scope_readout_slots'],8)
        self.assertEqual(summary['unique_changed_readout_slots'],1)
        self.assertEqual(summary['changed_readout_candidate_occurrences'],4)
        self.assertEqual(summary['source_guarded_run_model_calls'],1)
        self.assertEqual(summary['new_model_calls'],0)

    def test_uncertain_readout_is_not_hard_contradiction(self):
        data=fixture()
        for f in data[1]:
            if f['finding']=='pneumonia':f['reportgate_retained_state']='positive'
        rows,facts,_=attach(data)
        pneumonia=next(f for f in facts if f['finding']=='pneumonia')
        self.assertEqual(pneumonia['liveimage_xrv_relation'],'uncertainty_not_comparable')
        self.assertEqual(pneumonia['liveimage_retained_report_relation'],'uncertainty_not_comparable')
        self.assertTrue(pneumonia['liveimage_readout_changed'])
        self.assertEqual(rows[0]['liveimage_changed_readout_count'],1)
        self.assertFalse(pneumonia['liveimage_regeneration_authorized'])

    def test_same_unknown_is_readout_match_not_clinical_agreement(self):
        data=fixture(old_state='unknown',new_state='unknown')
        for f in data[1]:
            if f['finding']=='pneumonia':f['states']['xrv']='unknown'
        _,facts,_=attach(data)
        f=next(r for r in facts if r['finding']=='pneumonia')
        self.assertFalse(f['liveimage_readout_changed'])
        self.assertEqual(f['liveimage_xrv_relation'],'both_unmentioned_or_unassessable')
        self.assertFalse(f['liveimage_primary_metric_eligible'])

    def test_unknown_does_not_become_negative(self):
        data=fixture(new_state='unknown')
        rows,facts,_=attach(data)
        self.assertEqual(rows[0]['liveimage_state_pneumonia'],'unknown')
        f=next(r for r in facts if r['finding']=='pneumonia')
        self.assertEqual(f['liveimage_xrv_relation'],'single_source_assertion_unqualified')

    def test_six_unsupported_heads_have_null_states_and_repeatability(self):
        _,facts,unique=attach(fixture())
        outside=[r for r in unique if r['finding'] not in w.HEADS]
        self.assertEqual(len(outside),6)
        for r in outside:
            self.assertEqual(r['liveimage_status'],'outside_image_verifier_scope')
            self.assertIsNone(r['liveimage_qwen_state'])
            self.assertIsNone(r['liveimage_readout_changed'])

    def test_missing_report_assertion_is_not_negative(self):
        _,facts,_=attach(fixture())
        self.assertTrue(all(f['liveimage_retained_report_relation']=='report_assertion_not_retained'
            for f in facts if f['finding'] in w.HEADS))

    def test_report_relation_requires_exact_cached_check(self):
        data=fixture()
        for f in data[1]:
            f['imageverify_exact_report_comparison_available']=False
            f['reportgate_retained_state']='positive'
        _,facts,_=attach(data)
        self.assertTrue(all(f['liveimage_retained_report_relation']=='no_exact_report_check'
            for f in facts if f['finding'] in w.HEADS))

    def test_failed_new_readout_remains_unavailable_not_unknown(self):
        data=fixture()
        record=data[3]['records'][0]
        record.update(raw(failed=True))
        rows,facts,_=attach(data)
        self.assertEqual(rows[0]['liveimage_status'],'image_verifier_unavailable')
        self.assertEqual(rows[0]['liveimage_complete_state_count'],0)
        self.assertIsNone(rows[0]['liveimage_state_pneumonia'])
        self.assertEqual(next(f for f in facts if f['finding']=='pneumonia')['liveimage_readout_repeatability'],
            'image_verifier_unavailable')

    def test_failed_baseline_has_no_fabricated_repeatability(self):
        data=fixture()
        data[5]['records'][0].update(raw(failed=True))
        rows,facts,_=attach(data)
        self.assertEqual(rows[0]['liveimage_repeat_readout_slots'],0)
        for f in facts:
            if f['finding'] in w.HEADS:
                self.assertIsNone(f['liveimage_baseline_state'])
                self.assertIsNone(f['liveimage_readout_changed'])
                self.assertEqual(f['liveimage_readout_repeatability'],'old_verifier_unavailable')

    def test_guard_checksum_or_clinical_flag_tampering_refused(self):
        for key,value in (('receipt_sha256','0'*64),('clinical_acceptance',True),('regeneration_authorized',True)):
            data=fixture()
            data[3]['records'][0]['guard'][key]=value
            with self.assertRaises(ValueError):attach(data)

    def test_blocked_control_cannot_call_or_have_labels(self):
        for reason in ('call','labels','tokens'):
            data=fixture()
            control=data[3]['records'][1]
            if reason=='call':control['model_called']=True
            elif reason=='labels':control['states']=dict.fromkeys(g.HEADS,'negative')
            else:control['input_tokens']=2
            with self.assertRaises(ValueError):attach(data)

    def test_control_cannot_be_relabelled_as_original(self):
        data=fixture()
        data[4][1]['arm']='original'
        with self.assertRaises(ValueError):attach(data)

    def test_changed_image_hash_or_ehr_anchor_refused(self):
        for key in ('cxr_sha256','ehr_sha256','ehr_facts_sha256','case_id'):
            data=fixture()
            data[0][0][key]='9'*64
            with self.assertRaises(ValueError):attach(data)

    def test_same_image_id_must_share_fixed_lineage(self):
        data=fixture(reports=2)
        data[0][1]['ehr_sha256']='9'*64
        with self.assertRaises(ValueError):attach(data)

    def test_old_cached_state_must_match_baseline(self):
        data=fixture()
        next(f for f in data[1] if f['finding']=='pneumonia')['imageverify_qwen_state']='negative'
        with self.assertRaises(ValueError):attach(data)

    def test_finding_and_report_lineage_tampering_refused(self):
        for key in ('report_candidate_id','triple_candidate_id','case_id'):
            data=fixture()
            data[1][0][key]='invented_other'
            with self.assertRaises(ValueError):attach(data)
        data=fixture()
        data[1][0]['artifact_hashes']['report_sha256']='9'*64
        with self.assertRaises(ValueError):attach(data)

    def test_complete_inventory_and_unique_facts_required(self):
        for reason in ('missing','duplicate','unsupported'):
            data=fixture()
            if reason=='missing':data[1].pop()
            elif reason=='duplicate':data[1].append(copy.deepcopy(data[1][0]))
            else:data[1][0]['finding']='not_supported_invented_head'
            with self.assertRaises(ValueError):attach(data)

    def test_shared_image_readout_cannot_vary_by_report(self):
        data=fixture(reports=2)
        data[1][14]['states']['xrv']='positive'
        with self.assertRaises(ValueError):attach(data)

    def test_new_model_and_baseline_contracts_must_be_blind_and_frozen(self):
        for reason in ('context','prompt','frozen','duplicate','incomplete'):
            data=fixture()
            if reason=='context':data[5]['model_received_arm_names_ehr_reports_ids_scores_or_expected_answers']=True
            elif reason=='prompt':data[3]['image_prompt_sha256']='9'*64
            elif reason=='frozen':data[3]['frozen']=False
            elif reason=='duplicate':data[3]['records'].append(copy.deepcopy(data[3]['records'][0]))
            else:data[3]['records'][0]['states'].pop('edema')
            with self.assertRaises(ValueError):attach(data)

    def test_call_accounting_and_clinical_eligibility_cannot_change(self):
        for key,value in (('actual_model_calls',2),('blocked_before_model_call',0),('logical_slots',9),
                ('primary_metric_eligible',True),('selection_changed',True),('regeneration_authorized',True)):
            data=fixture()
            data[-1][key]=value
            with self.assertRaises(ValueError):attach(data)

    def test_repeat_attachment_refused(self):
        data=fixture()
        data[0][0]['liveimage_status']='old'
        with self.assertRaises(ValueError):attach(data)
        data=fixture()
        data[1][0]['liveimage_status']='old'
        with self.assertRaises(ValueError):attach(data)

    def test_slurm_guard_before_any_new_directory_or_input_read(self):
        with patch.object(w.tables,'require_cpu_slurm',side_effect=RuntimeError),patch.object(w,'new_atomic_run') as new:
            with self.assertRaises(RuntimeError):w.execute('/invented','test')
            new.assert_not_called()

    def test_existing_run_refused_before_loading_inputs(self):
        with patch.object(w.tables,'require_cpu_slurm'),patch.object(w,'new_atomic_run',side_effect=FileExistsError), \
                patch.object(w,'load_inputs') as load:
            with self.assertRaises(FileExistsError):w.execute('/invented','test')
            load.assert_not_called()

    def test_input_failure_discards_only_new_temporary(self):
        with patch.object(w.tables,'require_cpu_slurm'), \
                patch.object(w,'new_atomic_run',return_value=(Path('/invented/tmp'),Path('/invented/new'))), \
                patch.object(w,'load_inputs',side_effect=ValueError),patch.object(w,'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):w.execute('/invented','test')
            discard.assert_called_once_with(Path('/invented/tmp'))

    def test_summary_claims_stay_diagnostic(self):
        data=fixture()
        rows,facts,unique=attach(data)
        summary=w.summarize(rows,facts,unique,data[-1])
        for key in ('primary_metric_eligible','regeneration_authorized','selection_changed',
                'same_checkpoint_readout_agreement_is_clinical_truth','repeatability_change_is_confirmed_modality_error'):
            self.assertFalse(summary[key])
        self.assertIsNone(summary['independent_clinical_accuracy'])
        self.assertEqual(summary['clinically_resolved_requests'],0)


if __name__=='__main__':unittest.main()

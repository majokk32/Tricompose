#!/usr/bin/env python3
"""Append prospective guard/readout availability without replacing old scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import image_validity_guard as guard
import attach_report_gate_availability as tables
import attach_image_verification_availability as image_tables
import verify_cached_image_findings as cached
from contracts import (PROTECTED_ROOT,CHEXPERT_FINDINGS,require_inside,sha256_file,
    new_atomic_run,commit_atomic_run,discard_atomic_run,write_private_json,write_private_text)

BASE=PROTECTED_ROOT/'tricompose_v1_2'
SCHEMA='tricompose-full-bank-prospective-guarded-image-availability-v1'
LIVE_SHA='850f90f5b4624bc15d26aedcdc50585238805a1cd3c28c3b276055ba3d046149'
HEADS=frozenset(guard.HEADS)
PARENTS={
    'rows':(BASE/'image_validity_guards/pngguard_scope2_12645021_001',
        '0c9654d2147b7268890527c5db5f85f0c455217f501f17484ddf23365983ccba',0,('candidate_score_table.csv',)),
    'facts':(BASE/'image_verification_availability/imagecheck_pool960_12645021_001',
        'd933d9ba37181e2eed31f3f92fd8831d514a313fc225c6a6bb6f4260ad08f1bf',0,('fact_verification_availability.jsonl',)),
    'original':(BASE/'image_only_plans/image_only_scope2_12645021_001',
        '5fbcd984de88e7f776f3ef52081a98d978d620848f15feb63cfc19b4bedbcdfe',0,('plan.json',)),
    'live':(BASE/'guarded_image_runs/guardhook_scope2_12651080',LIVE_SHA,6,
        ('predictions.json','logical_slots.json','summary.json')),
    'baseline':(BASE/'image_control_runs/noinfo_scope2_12650073',
        '87d1e211e21fe497bda8647ae356a5c5a61246801bd5a288a828263f4f908ecd',10,('predictions.json',)),
}


def load_inputs():
    sources={'worker':Path(__file__),'tests':ROOT/'tests/test_guarded_image_availability.py',
        'protocol':ROOT.parent/'docs/guarded_image_availability_protocol.md'}
    for module in (guard,tables,image_tables,cached,cached.image_interface,cached.frozen,sys.modules['contracts']):
        sources[module.__name__]=Path(module.__file__)
    for name in ('reliability_preview','scorer_reliability','legacy_replay_adapter','invariant_verification','decision_preview'):
        sources[name]=ROOT/'src/tricompose_v12'/f'{name}.py'
    values={}
    for label,(root,expected,calls,names) in PARENTS.items():
        mp=require_inside(root/'manifest.json',PROTECTED_ROOT,must_exist=True)
        if sha256_file(mp)!=expected:raise ValueError('fixed_parent_manifest_required')
        manifest=cached.bounded_json(mp)
        if (manifest['new_model_calls']!=calls or any(manifest[k] is not False
                for k in ('primary_metric_eligible','selection_changed','regeneration_authorized'))):
            raise ValueError('unchanged_unqualified_parent_required')
        sources[label+'_manifest']=mp
        for name in names:
            path=require_inside(root/name,root,must_exist=True)
            if (not path.is_file() or path.stat().st_size>32*1024*1024
                    or sha256_file(path)!=manifest['artifacts'][name]['sha256']):
                raise ValueError('bounded_quote_free_fixed_artifact_required')
            if name.endswith('.csv'):
                with path.open(encoding='utf-8',newline='') as handle:value=list(csv.DictReader(handle))
                if any(None in r or None in r.values() for r in value):raise ValueError('complete_csv_required')
            elif name.endswith('.jsonl'):
                value=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
            else:value=json.loads(path.read_text(encoding='utf-8'))
            values[label,name]=value
            sources[label+'_'+name]=path
        if sha256_file(mp)!=expected:raise ValueError('parent_changed_during_read')
    return values,sources


def validate_readout(record):
    if record['contract_status']=='complete':
        if (not isinstance(record['states'],dict) or set(record['states'])!=HEADS
                or not set(record['states'].values())<=guard.STATES
                or record['token_limit_reached'] is not False or record['failure_reason'] is not None):
            raise ValueError('complete_named_four_state_readout_required')
    elif record['contract_status']!='failed_unavailable' or record['states'] is not None:
        raise ValueError('failed_readout_must_remain_null')
    if record['independent_clinical_validation'] is not False:
        raise ValueError('no_independent_clinical_truth_promotion')


def index_evidence(rows,original,predictions,slots,baseline,summary):
    candidates={r['triple_candidate_id']:r for r in rows}
    if len(candidates)!=len(rows):raise ValueError('unique_candidate_slots_required')
    images={}
    for row in rows:
        identity=tuple(row[k] for k in ('case_id','cxr_sha256','ehr_sha256','ehr_facts_sha256'))
        if images.setdefault(row['cxr_candidate_id'],identity)!=identity:
            raise ValueError('fixed_shared_image_anchor_required')
    references={r['cxr_candidate_id']:r for r in original['image_inputs']}
    if len(references)!=len(original['image_inputs']):raise ValueError('unique_original_image_inventory_required')
    if (predictions['schema_version']!='tricompose-prospective-guarded-image-only-v1'
            or baseline['schema_version']!='tricompose-image-no-information-control-v1'
            or any(p['frozen'] is not True or p['image_only'] is not True for p in (predictions,baseline))
            or any(p['image_prompt_sha256']!=original['image_prompt_sha256'] for p in (predictions,baseline))
            or baseline['model_received_arm_names_ehr_reports_ids_scores_or_expected_answers'] is not False
            or summary['model_received_ehr_reports_ids_scores_arm_names_or_expected_answers'] is not False):
        raise ValueError('same_frozen_blind_image_prompt_required')
    live={r['observation_id']:r for r in predictions['records']}
    previous={r['observation_id']:r for r in baseline['records']}
    if (len(live)!=len(predictions['records']) or len(previous)!=len(baseline['records'])
            or set(live)!=set(previous) or set(live)!={s['observation_id'] for s in slots}
            or len({s['slot_id'] for s in slots})!=len(slots)):
        raise ValueError('exact_unique_live_baseline_and_slot_inventory_required')
    for old in previous.values():validate_readout(old)
    for key,rec in live.items():
        current=guard.validate_receipt(rec['guard'])
        if (rec['artifact_sha256']!=current['artifact_sha256']
                or current['normalized_pixel_sha256'] not in (None,key)
                or type(rec['model_called']) is not bool):
            raise ValueError('exact_guarded_artifact_binding_required')
        if rec['model_called']:
            if current['basic_comparison_permitted'] is not True:raise ValueError('blocked_image_cannot_call_model')
            validate_readout(rec)
        elif (current['basic_comparison_permitted'] is not False
                or rec['contract_status']!='blocked_before_model_call'
                or any(rec[k] is not None for k in ('states','input_tokens','output_tokens','response_sha256','token_limit_reached'))):
            raise ValueError('blocked_before_call_must_have_null_readouts')
    original_keys={s['observation_id'] for s in slots if s['arm']=='original'}
    controls={s['observation_id'] for s in slots if s['arm'] in ('uniform_black','uniform_white')}
    if original_keys & controls or any(s['arm'] not in ('original','uniform_black','uniform_white') for s in slots):
        raise ValueError('isolated_controls_required')
    result={}
    for slot in slots:
        if slot['arm']!='original':continue
        cid=slot['cxr_candidate_id']
        ref=references.get(cid)
        rec=live[slot['observation_id']]
        if (ref is None or cid not in images or slot['cxr_sha256']!=ref['cxr_sha256']
                or rec['artifact_sha256']!=ref['cxr_sha256']
                or images[cid]!=tuple(ref[k] for k in ('case_id','cxr_sha256','ehr_sha256','ehr_facts_sha256'))):
            raise ValueError('exact_original_candidate_identity_and_hash_required')
        value={'record':rec,'baseline':previous[slot['observation_id']],'reference':ref}
        if result.setdefault(cid,value)!=value:raise ValueError('unchanged_original_image_slot_required')
        for row in rows:
            if row['cxr_candidate_id']==cid and (row['imageguard_status']!=rec['guard']['status']
                    or row['imageguard_receipt_sha256']!=rec['guard']['receipt_sha256']):
                raise ValueError('same_previously_checked_guard_receipt_required')
    if set(result)!=set(references):raise ValueError('all_and_only_original_images_may_join')
    calls=sum(r['model_called'] for r in live.values())
    if (summary['actual_model_calls']!=calls or summary['blocked_before_model_call']!=len(live)-calls
            or summary['unique_inputs']!=len(live) or summary['logical_slots']!=len(slots)
            or summary.get('old_prediction_states_parsed_after_prediction_fsync') is not True
            or summary.get('clinical_requests_resolved')!=0
            or any(summary[k] is not False for k in ('clinical_acceptance','primary_metric_eligible',
                'selection_changed','fixed_ehr_changed','regeneration_authorized'))):
        raise ValueError('unchanged_actual_call_and_guard_accounting_required')
    return candidates,result


def annotate_fact(fact,candidate,pair):
    if any(k.startswith('liveimage_') for k in fact):raise ValueError('repeat_attachment_refused')
    if (any(fact[k]!=candidate[k] for k in ('case_id','triple_candidate_id','cxr_candidate_id','report_candidate_id'))
            or any(fact['artifact_hashes'][k]!=candidate[k]
                for k in ('ehr_sha256','ehr_facts_sha256','cxr_sha256','report_sha256'))
            or fact['finding'] not in CHEXPERT_FINDINGS or not set(fact['states'].values())<=guard.STATES):
        raise ValueError('exact_original_finding_and_artifact_lineage_required')
    rec=None if pair is None else pair['record']
    fresh=old=changed=None
    status=repeat=xrv_relation=report_relation='not_checked'
    exact=False
    if rec is not None:
        if fact['finding'] not in HEADS:
            status=repeat=xrv_relation=report_relation='outside_image_verifier_scope'
        elif rec['contract_status']!='complete':
            status=repeat=xrv_relation=report_relation='image_verifier_unavailable'
        else:
            fresh=rec['states'][fact['finding']]
            status='checked_clinically_unqualified'
            base=pair['baseline']
            if base['contract_status']=='complete':
                old=base['states'][fact['finding']]
                if fact['imageverify_qwen_state']!=old:
                    raise ValueError('old_full_bank_state_must_match_frozen_baseline')
                changed=old!=fresh
                repeat='changed_state_unqualified' if changed else 'same_state_unqualified'
            else:repeat='old_verifier_unavailable'
            xrv_relation=cached.relation(fact['states']['xrv'],fresh)
            exact=fact['imageverify_exact_report_comparison_available']
            retained=fact['reportgate_retained_state']
            report_relation=('no_exact_report_check' if not exact else 'report_assertion_not_retained'
                if retained is None else cached.relation(fresh,retained))
    return {**fact,'liveimage_status':status,'liveimage_guard_status':None if rec is None else rec['guard']['status'],
        'liveimage_guard_receipt_sha256':None if rec is None else rec['guard']['receipt_sha256'],
        'liveimage_run_manifest_sha256':None if rec is None else LIVE_SHA,
        'liveimage_response_sha256':None if rec is None else rec['response_sha256'],
        'liveimage_baseline_state':old,'liveimage_qwen_state':fresh,
        'liveimage_readout_repeatability':repeat,'liveimage_readout_changed':changed,
        'liveimage_xrv_relation':xrv_relation,'liveimage_retained_report_relation':report_relation,
        'liveimage_exact_report_comparison_available':exact,
        'liveimage_independent_clinical_validation':False,'liveimage_primary_metric_eligible':False,
        'liveimage_regeneration_authorized':False}


def attach(rows,facts,candidates,records):
    annotated,groups,unique=[],defaultdict(list),{}
    seen=set()
    for fact in facts:
        key=fact['triple_candidate_id'],fact['finding']
        if key in seen or key[0] not in candidates:raise ValueError('unique_existing_finding_required')
        seen.add(key)
        pair=records.get(fact['cxr_candidate_id'])
        new=annotate_fact(fact,candidates[key[0]],pair)
        annotated.append(new)
        groups[key[0]].append(new)
        if pair is not None:
            image_key=fact['cxr_candidate_id'],fact['finding']
            value={'cxr_candidate_id':image_key[0],'cxr_sha256':fact['artifact_hashes']['cxr_sha256'],
                'finding':image_key[1],'raw_xrv_state':fact['states']['xrv'],
                **{k:v for k,v in new.items() if k.startswith('liveimage_') and k not in
                    ('liveimage_retained_report_relation','liveimage_exact_report_comparison_available')}}
            if unique.setdefault(image_key,value)!=value:raise ValueError('shared_image_readout_must_not_vary_by_report')
    result=[]
    for row in rows:
        if any(k.startswith('liveimage_') for k in row):raise ValueError('repeat_candidate_attachment_refused')
        own=groups[row['triple_candidate_id']]
        if len(own)!=14 or {f['finding'] for f in own}!=set(CHEXPERT_FINDINGS):
            raise ValueError('complete_14_finding_inventory_required')
        pair=records.get(row['cxr_candidate_id'])
        rec=None if pair is None else pair['record']
        counts=Counter(f['liveimage_xrv_relation'] for f in own)
        report_counts=Counter(f['liveimage_retained_report_relation'] for f in own)
        fields={'liveimage_status':'not_checked' if rec is None else 'checked_clinically_unqualified'
                if rec['contract_status']=='complete' else 'image_verifier_unavailable',
            'liveimage_guard_status':None if rec is None else rec['guard']['status'],
            'liveimage_guard_receipt_sha256':None if rec is None else rec['guard']['receipt_sha256'],
            'liveimage_run_manifest_sha256':None if rec is None else LIVE_SHA,
            'liveimage_shared_image_dependency_id':None if rec is None else row['cxr_candidate_id'],
            'liveimage_complete_state_count':None if rec is None else sum(f['liveimage_qwen_state'] is not None for f in own),
            'liveimage_explicit_state_count':None if rec is None else sum(f['liveimage_qwen_state'] in ('positive','negative') for f in own),
            'liveimage_uncertain_state_count':None if rec is None else sum(f['liveimage_qwen_state']=='uncertain' for f in own),
            'liveimage_unknown_state_count':None if rec is None else sum(f['liveimage_qwen_state']=='unknown' for f in own),
            'liveimage_repeat_readout_slots':None if rec is None else sum(f['liveimage_readout_changed'] is not None for f in own),
            'liveimage_changed_readout_count':None if rec is None else sum(f['liveimage_readout_changed'] is True for f in own),
            'liveimage_independent_clinical_validation':False,'liveimage_primary_metric_eligible':False,
            'liveimage_regeneration_authorized':False}
        for head in guard.HEADS:
            fields['liveimage_state_'+head]=next(f['liveimage_qwen_state'] for f in own if f['finding']==head)
        for relation in image_tables.RELATIONS:
            fields['liveimage_xrv_'+relation+'_count']=None if rec is None else counts[relation]
        for relation in ('explicit_agreement_unqualified','explicit_opposition_unqualified',
                'uncertainty_not_comparable','report_assertion_not_retained','no_exact_report_check'):
            fields['liveimage_report_'+relation+'_count']=None if rec is None else report_counts[relation]
        result.append({**row,**fields})
    image_tables.preserve(rows,result)
    image_tables.preserve(facts,annotated)
    return result,annotated,[unique[k] for k in sorted(unique)]


def summarize(rows,facts,unique,live_summary):
    checked=[r for r in rows if r['liveimage_status']!='not_checked']
    changed=[r for r in unique if r['liveimage_readout_changed'] is True]
    transition=Counter((r['finding'],r['liveimage_baseline_state'],r['liveimage_qwen_state']) for r in changed)
    return {'schema_version':SCHEMA,'candidate_rows':len(rows),'fact_rows':len(facts),
        'fixed_ehr_cases':len({r['case_id'] for r in rows}),'image_slots':len({r['cxr_candidate_id'] for r in rows}),
        'checked_candidate_slots':len(checked),'unchecked_candidate_slots':len(rows)-len(checked),
        'checked_image_slots':len({r['cxr_candidate_id'] for r in checked}),
        'candidate_slots_with_changed_readout':sum((r['liveimage_changed_readout_count'] or 0)>0 for r in rows),
        'unique_image_finding_rows':len(unique),'unique_in_scope_readout_slots':sum(r['finding'] in HEADS for r in unique),
        'unique_same_readout_slots':sum(r['liveimage_readout_changed'] is False for r in unique),
        'unique_changed_readout_slots':len(changed),'unique_changed_image_slots':len({r['cxr_candidate_id'] for r in changed}),
        'changed_readout_candidate_occurrences':sum(f['liveimage_readout_changed'] is True for f in facts),
        'changed_readout_transitions':[{'finding':h,'old_state':a,'new_state':b,'unique_slots':n}
            for (h,a,b),n in sorted(transition.items())],
        'unique_xrv_relation_counts':dict(sorted(Counter(r['liveimage_xrv_relation'] for r in unique).items())),
        'candidate_finding_report_relation_counts':dict(sorted(Counter(f['liveimage_retained_report_relation'] for f in facts).items())),
        'source_guarded_run_model_calls':live_summary['actual_model_calls'],
        'source_guarded_run_blocked_controls':live_summary['blocked_before_model_call'],
        'new_model_calls':0,'new_slurm_submissions':0,'clinically_resolved_requests':0,
        'original_cells_and_order_preserved':True,'selection_changed':False,'fixed_ehr_changed':False,
        'primary_metric_eligible':False,'regeneration_authorized':False,'independent_clinical_accuracy':None,
        'report_image_ehr_bodies_opened':False,'raw_patient_inputs_opened':False,
        'reference_keys_opened':False,'models_or_weights_opened':False,
        'same_checkpoint_readout_agreement_is_clinical_truth':False,
        'repeatability_change_is_confirmed_modality_error':False,'development_availability_not_heldout':True}


def execute(output_root,run_id):
    tables.require_cpu_slurm()
    temporary,target=new_atomic_run(output_root,run_id)
    try:
        inputs,sources=load_inputs()
        before={k:sha256_file(p) for k,p in sources.items()}
        rows=inputs['rows','candidate_score_table.csv']
        facts=inputs['facts','fact_verification_availability.jsonl']
        if (len(rows),len(facts))!=(960,13440):raise ValueError('fixed_full_bank_inventory_required')
        candidates,records=index_evidence(rows,inputs['original','plan.json'],inputs['live','predictions.json'],
            inputs['live','logical_slots.json']['slots'],inputs['baseline','predictions.json'],inputs['live','summary.json'])
        annotated,finding_rows,unique=attach(rows,facts,candidates,records)
        summary=summarize(annotated,finding_rows,unique,inputs['live','summary.json'])
        if (summary['fixed_ehr_cases'],summary['image_slots'],summary['checked_image_slots'],
                summary['checked_candidate_slots'],summary['unique_image_finding_rows'],
                summary['unique_in_scope_readout_slots'])!=(80,240,6,24,84,48):
            raise ValueError('fixed_six_image_twentyfour_triple_scope_required')
        if (summary['unique_same_readout_slots'],summary['unique_changed_readout_slots'])!=(46,2):
            raise ValueError('completed_readout_denominators_must_match')
        summary['existing_cpu_job_id']=os.environ['SLURM_JOB_ID']
        write_private_text(temporary/'candidate_score_table.csv',tables.csv_text(annotated))
        write_private_text(temporary/'fact_verification_availability.jsonl',tables.jsonl_text(finding_rows))
        write_private_text(temporary/'unique_image_finding_table.jsonl',tables.jsonl_text(unique))
        write_private_json(temporary/'summary.json',summary)
        write_private_text(temporary/'RESULTS_CN_EN.md',
            '# Prospective image evidence availability / 新护栏读数接回候选表\n\n'
            '旧分数、标签、择优及 EHR 均保留；只追加 liveimage_ 字段。\n\n'
            '24 slots reuse six images. Unchecked/null, unknown and uncertain are not negatives.\n\n'
            '两个读数变化不是临床错误或修复授权；同一图的四份报告不是独立投票。\n\n'
            'No new model/API/GPU/submission, clinical score, selection or regeneration.\n\n'
            '```json\n'+json.dumps(summary,sort_keys=True,indent=2)+'\n```\n')
        if any(sha256_file(p)!=before[k] for k,p in sources.items()):raise ValueError('consumed_source_changed')
        write_private_json(temporary/'manifest.json',{'schema_version':SCHEMA,'run_id':run_id,
            'source_paths':{k:str(p.resolve()) for k,p in sources.items()},'source_sha256':before,
            'artifacts':{p.name:{'sha256':sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls':0,'primary_metric_eligible':False,'selection_changed':False,
            'regeneration_authorized':False,'report_image_ehr_bodies_opened':False})
        for p in (temporary,*temporary.iterdir()):
            st=p.stat()
            if st.st_gid not in (96293,65534) or st.st_mode & 0o7777!=(0o2770 if p.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary,target)
        return target,summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=BASE/'guarded_image_availability')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    try:target,summary=execute(args.output_root,args.run_id)
    except Exception as error:
        print(json.dumps({'status':'failed_closed','error_type':type(error).__name__}))
        return 2
    print(json.dumps({'status':'guarded_image_availability_attached','candidate_rows':summary['candidate_rows'],
        'checked_candidate_slots':summary['checked_candidate_slots'],'unique_changed_readout_slots':summary['unique_changed_readout_slots'],
        'new_model_calls':0,'manifest_sha256':sha256_file(target/'manifest.json')}))
    return 0


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Version-aware, trace-only diagnostic; no generator or source-data edits.

The archived v1_1_3 prompt metadata calls CHF a direct fact. Current v1_1_4
instead separates clinical context from radiographic findings. Reconcile only
this explicitly supported legacy metadata role; never infer edema or enlarge
the EHR reference inventory. Old worker/job failures remain recorded failures.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import check_paired_conditioning_v2 as previous
import read_paired_endpoint_cache_v1 as cache
from contracts import (PROTECTED_ROOT, RUN_ID_PATTERN, require_inside, sha256_file,
    private_directory, write_private_json, write_private_text)
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text
from tricompose_v11.facts import CONTEXT_LABELS
from tricompose_v11.tokenizer_trace import validate_tokenizer_trace
from tricompose_v12.live_execution import validate_cxr_binding
from tricompose_v12.live_workers import check_pins
from run_fixed_image_reports import normalize_owned_modes

VERSION='tricompose-paired-conditioning-diagnostic-v3'
LEGACY='roentgen_v2.ehr_context_prompt.v1_1_3'
CURRENT='roentgen_v2.ehr_context_prompt.v1_1_4'
LEGACY_CONTEXT_IDS={'congestive_heart_failure'}
SAFE_ERRORS={'recognized_unique_rendered_fact_ids_required',
    'candidate tokenizer observation status is invalid','candidate tokenizer artifact hash mismatch',
    'candidate tokenizer trace status is invalid','tokenizer trace is bound to another supplied prompt',
    'tokenizer trace text hash mismatch','tokenizer input ID hash mismatch',
    'source_metadata_version_not_supported','context_id_in_current_direct_fact_list',
    'unknown_legacy_or_current_direct_fact_id','duplicate_or_invalid_declared_fact_id'}


def roles(request):
    """Pure saved-metadata adapter. No original input/request is changed."""
    prompt=request['inputs']['final_prompt'];version=prompt['renderer_version']
    if version not in (LEGACY,CURRENT):raise ValueError('source_metadata_version_not_supported')
    ids=prompt['included_direct_fact_ids']
    if (not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(ids)!=len(set(ids))):
        raise ValueError('duplicate_or_invalid_declared_fact_id')
    image_ids=[];contexts=[]
    for item in ids:
        if item in cache.secondary.ROENTGEN_FINDING_PHRASES:image_ids.append(item)
        elif item in LEGACY_CONTEXT_IDS:
            if version!=LEGACY:raise ValueError('context_id_in_current_direct_fact_list')
            contexts.append(item)
        else:raise ValueError('unknown_legacy_or_current_direct_fact_id')
    return {'source_renderer_version':version,'source_declared_direct_ids':list(ids),
        'radiographic_ids':image_ids,'legacy_context_ids':contexts,
        'runtime_request_changed':False,'ehr_labels_or_scores_changed':False,
        'clinical_context_promoted_to_image_finding':False}


def diagnostic(image,request,trace):
    mapping=roles(request)
    # Explicit readout-only projection for the old length/hash checker, not
    # an input sent to a model. Source request and its hash remain untouched.
    projection=deepcopy(request)
    projection['inputs']['final_prompt']['included_direct_fact_ids']=mapping['radiographic_ids']
    row=previous.diagnostic(image,projection,trace)
    length_hash_pass=row.pop('v1_gate_pass')
    row.update(source_renderer_version=mapping['source_renderer_version'],
        original_declared_direct_id_count=len(mapping['source_declared_direct_ids']),
        radiographic_id_count=len(mapping['radiographic_ids']),
        legacy_context_id_count=len(mapping['legacy_context_ids']),
        legacy_context_mention_count=sum(CONTEXT_LABELS[i] in trace['positive_tokenizer_text'].lower()
            for i in mapping['legacy_context_ids']),
        source_role_mismatch_identified=bool(mapping['legacy_context_ids']),
        old_combined_guard_would_pass=length_hash_pass and not mapping['legacy_context_ids'],
        length_and_hash_checks_pass=length_hash_pass,
        metadata_projection_is_not_runtime_request=True,
        clinical_context_promoted_to_image_finding=False)
    if mapping['legacy_context_ids']:
        row['v1_failed_checks'].append('legacy_context_not_in_current_radiographic_phrase_inventory')
    return row


def run(args):
    previous.require_trace_allocation()  # Before private body opens.
    if not RUN_ID_PATTERN.fullmatch(args.run_id):raise ValueError('new_opaque_run_id_required')
    output=require_inside(args.output_root,PROTECTED_ROOT,must_exist=False)
    private_directory(output,exist_ok=True);root=output/args.run_id;private_directory(root)
    phase='load_sealed_metadata';record_index=None;records=[]
    write_private_json(root/'start_manifest.json',{'schema_version':VERSION,'status':'in_progress',
        'new_model_calls':0,'source_inputs_changed':False})
    try:
        data=cache.secondary.load_metadata(cache.inputs());sources=data['sources']
        code_pins={str(p):sha256_file(p) for p in (Path(__file__),ROOT/'tests/test_paired_conditioning_v3.py',
            ROOT/'tools/check_paired_conditioning_v2.py',ROOT/'tools/read_paired_endpoint_cache_v1.py',
            ROOT/'tools/score_paired_probes_v1.py')}
        cm=json.loads(checked(previous.CACHE_RUN/'manifest.json',previous.CACHE_SHA,1024**2,sources))
        if cm['source_gpu_job_completed_successfully'] is not False or cm['original_selection_changed'] is not False:
            raise ValueError('recovered_partial_source_not_relabelled_required')
        for p in sorted({t['cxr_run'] for t in data['triples']}):
            branch=require_inside(p,PROTECTED_ROOT,must_exist=True);phase='load_candidate_metadata'
            m=json.loads(checked(branch/'manifest.json',sha256_file(branch/'manifest.json'),1024**2,sources))
            if m['schema_version']!='tricompose-cxr-candidate-run-v1.1' or m['candidate_count']!=1:
                raise ValueError('single_frozen_source_image_branch_required')
            for entry in m['candidates']:
                image=json.loads(checked(branch/entry['path'],entry['sha256'],1024**2,sources))
                case=next(c for c in data['plan']['cases'] if c['case_id']==image['case_id'])
                req=next(r for r in case['requests'] if r['request']['seed']==image['seed'])
                phase='validate_image_request_binding'
                validate_cxr_binding(image,req,cache.secondary.audit.paired.anchor_from_record(case['anchor']))
                if not any(r['cxr_candidate_id']==image['candidate_id'] and r['cxr_sha256']==image['artifact']['sha256']
                        for r in data['rows']):raise ValueError('registered_source_image_required')
                record_index=len(records);phase='reconcile_saved_metadata_roles';roles(req['request'])
                phase='validate_runtime_trace_integrity';validate_tokenizer_trace(image,PROTECTED_ROOT)
                observed=image['tokenizer_input'];phase='read_authenticated_synthetic_trace'
                trace=json.loads(checked(Path(observed['trace_path']),observed['trace_sha256'],1024**2,sources))
                phase='derive_text_free_length_and_role_diagnostics'
                records.append(diagnostic(image,req['request'],trace))
        if len(records)!=4 or len({r['cxr_candidate_id'] for r in records})!=4:
            raise ValueError('four_unique_registered_runtime_traces_required')
        phase='write_private_readout'
        write_private_json(root/'trace_diagnostics.json',{'records':records})
        table=[{k:','.join(v) if isinstance(v,list) else v for k,v in r.items() if k!='checks'} for r in records]
        write_private_text(root/'conditioning_transfer.csv',csv_text(table))
        summary={'schema_version':VERSION,'status':'version_aware_diagnostic_completed_not_clinical',
            'runtime_traces':4,'fixed_ehr_cases':len({r['case_id'] for r in records}),
            'length_hash_checks_passed':sum(r['length_and_hash_checks_pass'] for r in records),
            'old_combined_guard_passed':sum(r['old_combined_guard_would_pass'] for r in records),
            'legacy_role_mismatch_traces':sum(r['source_role_mismatch_identified'] for r in records),
            'radiographic_phrase_count':sum(r['included_phrase_count'] for r in records),
            'matched_radiographic_phrase_count':sum(r['matched_phrase_count'] for r in records),
            'legacy_context_mention_count':sum(r['legacy_context_mention_count'] for r in records),
            'attention_count_unavailable':sum(not r['attention_count_available'] for r in records),
            'new_model_calls':0,'source_inputs_or_scores_changed':False,'clinical_truth_available':False,
            'original_gpu_and_cpu_job_failures_remain_failures':True,'text_encoder_hook_observed':False}
        write_private_json(root/'summary.json',summary);phase='final_source_pins'
        if any(sha256_file(p)!=pin for p,pin in sources.items()):raise ValueError('source_changed')
        check_pins(code_pins)
        write_private_json(root/'manifest.json',{'schema_version':VERSION,'sources':sources,'code_pins':code_pins,
            'artifacts':{p.name:{'sha256':sha256_file(p)} for p in root.iterdir() if p.is_file()},
            'original_selection_changed':False,'clinical_qualified':False})
    except BaseException as exc:
        write_private_json(root/'failure_manifest.json',{'status':'failed_retained_no_automatic_resume',
            'phase':phase,'trace_index':record_index,'error_type':type(exc).__name__,
            'error_code':str(exc) if str(exc) in SAFE_ERRORS else 'unlisted_error_redacted',
            'new_model_calls':0,'source_inputs_changed':False})
        raise
    finally:normalize_owned_modes(root)
    return root,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('inspect','run'))
    p.add_argument('--output-root',default=str(cache.BASE/'paired_conditioning_diagnostics'))
    p.add_argument('--run-id');args=p.parse_args()
    try:
        if args.mode=='inspect':
            cpu_guard();data=cache.secondary.load_metadata(cache.inputs())
            maps=[roles(r['request']) for c in data['plan']['cases'] for r in c['requests']]
            print(json.dumps({'status':'metadata_role_reconciliation_ready_trace_job_not_run',
                'request_count':len(maps),'radiographic_ids':sum(len(m['radiographic_ids']) for m in maps),
                'legacy_context_ids':sum(len(m['legacy_context_ids']) for m in maps),'new_model_calls':0}))
        else:
            root,summary=run(args)
            print(json.dumps({'status':summary['status'],'manifest_sha256':sha256_file(root/'manifest.json'),'new_model_calls':0}))
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())

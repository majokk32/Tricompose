#!/usr/bin/env python3
"""Separately approved CPU Slurm: inspect four protected synthetic traces.

No model, tokenizer, report/EHR input body, image pixel or API is invoked.
Only text-free diagnostics are persisted; nullable masks stay unavailable.
The consumed V1 worker, failed GPU run and recovered scores stay immutable.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import read_paired_endpoint_cache_v1 as cache
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text
from tricompose_v12.live_execution import validate_cxr_binding
from tricompose_v12.live_workers import check_pins
from tricompose_v11.tokenizer_trace import validate_tokenizer_trace, text_sha256

VERSION='tricompose-paired-conditioning-diagnostic-v2'
CACHE_RUN=cache.BASE/'paired_probe_endpoint_readouts/paired2_biovil_12792810_001'
CACHE_SHA='48b472d148ccffc94ce6e48fbb0d3ed34bac4430fed07a1a824218cba3753c52'


def require_trace_allocation():
    cpu_guard()
    if (os.environ.get('TRICOMPOSE_SCOPE')!='approved_synthetic_trace_diagnostic'
            or os.environ.get('SLURM_JOB_ID') in ('12784259','12784711')):
        raise RuntimeError('separately_approved_trace_cpu_allocation_required')


def diagnostic(image,request,trace):
    """Exact checks of the old gate, not a change to model/clinical acceptance."""
    prompt=request['inputs']['final_prompt'];count=image['cost']['prompt_token_count']
    attention=trace['attention_token_count'];padded=trace['padded_token_count']
    valid_count=type(count) is int and 1<=count<=77
    valid_attention=type(attention) is int and 1<=attention<=77
    checks={'supported_model':image['model_id']=='roentgen_v2',
        'candidate_prompt_hash':image['prompt_sha256']==prompt['sha256'],
        'runtime_observed':trace['runtime_observed'] is True,
        'supplied_prompt_hash':trace['supplied_prompt_sha256']==prompt['sha256'],
        'unchanged_text_flag':trace['pipeline_changed_text'] is False,
        'unchanged_generation_arguments':trace['official_generation_arguments_changed'] is False,
        'positive_prompt_hash':trace['positive_tokenizer_text_sha256']==prompt['sha256'],
        'positive_text_bytes_hash':text_sha256(trace['positive_tokenizer_text'])==prompt['sha256'],
        'adapter_length_within_limit':valid_count,
        'attention_count_available':valid_attention,
        'padded_context_length_77':type(padded) is int and padded==77,
        'attention_equals_adapter_length':valid_count and valid_attention and count==attention}
    # No attention mask is a supported trace-schema possibility, not a reason
    # to erase successfully validated CXR/report endpoint measurements.
    agreement=count==attention if valid_count and valid_attention else None
    phrase_ids=prompt['included_direct_fact_ids']
    phrases=cache.secondary.ROENTGEN_FINDING_PHRASES
    if len(phrase_ids)!=len(set(phrase_ids)) or any(f not in phrases for f in phrase_ids):
        raise ValueError('recognized_unique_rendered_fact_ids_required')
    return {'case_id':image['case_id'],'cxr_candidate_id':image['candidate_id'],'seed':image['seed'],
        'adapter_token_count':count,'attention_token_count':attention,'padded_token_count':padded,
        'attention_count_available':valid_attention,
        'attention_equals_adapter_length':agreement,
        'count_check_status':'matched' if agreement is True else 'mismatch' if agreement is False else 'not_available',
        'count_unavailable_reason':'attention_mask_not_recorded' if attention is None else
            'invalid_length_metadata' if agreement is None else None,
        'v1_gate_pass':all(checks.values()),'v1_failed_checks':[k for k,v in checks.items() if not v],
        'included_phrase_count':len(phrase_ids),
        'matched_phrase_count':sum(phrases[f] in trace['positive_tokenizer_text'].lower() for f in phrase_ids),
        'checks':checks,'text_encoder_hook_observed':False,'clinical_truth_available':False,
        'scores_or_source_inputs_modified':False}


def run(args):
    require_trace_allocation()  # Before any protected trace/body open.
    data=cache.secondary.load_metadata(cache.inputs());sources=data['sources']
    cm=json.loads(checked(CACHE_RUN/'manifest.json',CACHE_SHA,1024**2,sources))
    if (cm['schema_version']!=cache.VERSION or cm['source_gpu_job_completed_successfully'] is not False
            or cm['original_selection_changed'] is not False):
        raise ValueError('recovered_partial_source_not_relabelled_required')
    records=[]
    for p in sorted({t['cxr_run'] for t in data['triples']}):
        root=require_inside(p,PROTECTED_ROOT,must_exist=True)
        manifest=json.loads(checked(root/'manifest.json',sha256_file(root/'manifest.json'),1024**2,sources))
        if manifest['schema_version']!='tricompose-cxr-candidate-run-v1.1' or manifest['candidate_count']!=1:
            raise ValueError('single_frozen_source_image_branch_required')
        for entry in manifest['candidates']:
            image=json.loads(checked(root/entry['path'],entry['sha256'],1024**2,sources))
            case=next(c for c in data['plan']['cases'] if c['case_id']==image['case_id'])
            req=next(r for r in case['requests'] if r['request']['seed']==image['seed'])
            validate_cxr_binding(image,req,cache.secondary.audit.paired.anchor_from_record(case['anchor']))
            if not any(r['cxr_candidate_id']==image['candidate_id'] and r['cxr_sha256']==image['artifact']['sha256']
                    for r in data['rows']):
                raise ValueError('source_image_must_belong_to_sealed_score_rows')
            validate_tokenizer_trace(image,PROTECTED_ROOT)
            observed=image['tokenizer_input']
            trace=json.loads(checked(Path(observed['trace_path']),observed['trace_sha256'],1024**2,sources))
            records.append(diagnostic(image,req['request'],trace))
    if len(records)!=4 or len({r['cxr_candidate_id'] for r in records})!=4:
        raise ValueError('four_unique_registered_runtime_traces_required')
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    code_pins={str(p):sha256_file(p) for p in (Path(__file__),ROOT/'tests/test_paired_conditioning_v2.py',
        ROOT/'tools/read_paired_endpoint_cache_v1.py',ROOT/'tools/score_paired_probes_v1.py')}
    try:
        write_private_json(temporary/'trace_diagnostics.json',{'records':records})
        table=[{k:','.join(v) if isinstance(v,list) else v for k,v in r.items() if k!='checks'} for r in records]
        write_private_text(temporary/'conditioning_transfer.csv',csv_text(table))
        summary={'schema_version':VERSION,'status':'diagnostic_completed_not_clinical',
            'runtime_traces':4,'fixed_ehr_cases':len({r['case_id'] for r in records}),
            'v1_gate_passed_traces':sum(r['v1_gate_pass'] for r in records),
            'attention_count_unavailable':sum(not r['attention_count_available'] for r in records),
            'included_phrase_count':sum(r['included_phrase_count'] for r in records),
            'matched_phrase_count':sum(r['matched_phrase_count'] for r in records),
            'new_model_calls':0,'model_or_tokenizer_invoked':False,'source_gpu_job_stays_failed':True,
            'scores_or_ehr_or_prompts_changed':False,'clinical_truth_available':False,
            'positive_text_transfer_is_not_radiographic_truth':True}
        write_private_json(temporary/'summary.json',summary)
        if any(sha256_file(p)!=pin for p,pin in sources.items()):raise ValueError('source_changed')
        check_pins(code_pins)
        write_private_json(temporary/'manifest.json',{'schema_version':VERSION,'sources':sources,'code_pins':code_pins,
            'artifacts':{p.name:{'sha256':sha256_file(p)} for p in temporary.iterdir() if p.is_file()},
            'original_selection_changed':False,'clinical_qualified':False})
        commit_atomic_run(temporary,target)
    except BaseException:
        discard_atomic_run(temporary);raise
    return target,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default=str(cache.BASE/'paired_conditioning_diagnostics'))
    p.add_argument('--run-id',required=True);args=p.parse_args()
    try:
        target,summary=run(args)
        print(json.dumps({'status':summary['status'],'manifest_sha256':sha256_file(target/'manifest.json'),'new_model_calls':0}))
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())

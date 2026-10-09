#!/usr/bin/env python3
"""Recover authenticated secondary scores, NOT the failed optional trace gate.

Existing cache-only CPU Slurm; no report, prompt, EHR, pixel or weight opens.
The failed GPU run remains immutable and is always identified as failed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import score_paired_probes_v1 as secondary
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text
from tricompose_v12.live_workers import check_pins

VERSION = 'tricompose-paired-secondary-cache-readout-v1'
BASE = secondary.audit.paired.BASE
GPU_RUN = BASE/'paired_probe_endpoints/paired2_biovil_12792810'
NATIVE_SHA = '7762615ed873d0d99ccde04c39371b66d975439fe6cea7ae37a0f02f2e03f648'
FAILURE_SHA = '24514b0b88b912bf872db9edb8598237ad51ee8d5e0e5403c20a625d164b54f2'


def inputs():
    return argparse.Namespace(plan_run=BASE/'paired_probe_plans/paired2_12784259_001',
        plan_manifest_sha256='ea4a1faf17b9d22a063bccd6d9482c4f598d017acb63920a59c2a0934ab0d37a',
        source_run=BASE/'paired_probe_runs/paired2_12791443',
        source_manifest_sha256='e5979cddf6b2075ea5a3e5ddef2c282a6278e89493144714b3afd1fd4dac0713',
        audit_run=BASE/'paired_probe_audits/paired2_12791443_001',
        audit_manifest_sha256='548ee53017c7a46fe836098e2be5bc00dc2cc8a1b2d7c8a70aa171ffb35fd195')


def groups(candidates, methods):
    """Keep NA/denominators visible; never average eight slots as eight cases."""
    result = []
    for field, rows in (('report_model_id',candidates),('method',methods)):
        for name in sorted({r[field] for r in rows}):
            part = [r for r in rows if r[field]==name]
            valid = [r['biovil_raw_cosine'] for r in part if r['biovil_raw_cosine'] is not None]
            result.append({'group_type':field, 'group':name, 'candidate_or_selection_slots':len(part),
                'distinct_ehr_cases':len({r['case_id'] for r in part}),
                'available_scores':len(valid), 'unavailable_scores':len(part)-len(valid),
                'mean_raw_cosine':sum(valid)/len(valid) if valid else None,
                'minimum_raw_cosine':min(valid) if valid else None,
                'maximum_raw_cosine':max(valid) if valid else None,
                'clinical_accuracy':None, 'development_only':True})
    return result


def run(args):
    cpu_guard()  # Before cache reads or writes; no model factory is imported.
    data = secondary.load_metadata(inputs())
    sources = data['sources']
    failure = json.loads(checked(GPU_RUN/'failure_manifest.json',FAILURE_SHA,1024**2,sources))
    if (failure['status']!='failed_retained_no_automatic_resume'
            or failure['endpoint_reserved_before_spawn'] is not True
            or failure['new_generation_calls']!=0 or (GPU_RUN/'manifest.json').exists()):
        raise ValueError('immutable_failed_source_not_complete_job_required')
    endpoint = json.loads(checked(GPU_RUN/'native_endpoint.json',NATIVE_SHA,1024**2,sources))
    endpoint['historical_pool_is_untouched_test'] = False
    candidates, methods, actions = secondary.endpoint_tables(data,endpoint)
    code_pins = {str(p):sha256_file(p) for p in (Path(__file__),ROOT/'tests/test_paired_endpoint_cache_v1.py',
        ROOT/'tools/score_paired_probes_v1.py')}
    temporary,target = new_atomic_run(args.output_root,args.run_id)
    try:
        for name, rows in (('candidate_scores.csv',candidates),('method_comparison.csv',methods),
            ('paired_action_endpoints.csv',actions),('group_summary.csv',groups(candidates,methods))):
            write_private_text(temporary/name,csv_text(rows))
        write_private_json(temporary/'scores.json',endpoint)
        summary = {'schema_version':VERSION,'status':'secondary_scores_validated_optional_trace_unresolved',
            'source_gpu_job':'12792810','source_gpu_job_state':'FAILED',
            'source_gpu_elapsed_seconds':33,'source_exit_code':'1:0',
            'source_status_is_scheduler_observation_not_promoted':True,
            'source_native_endpoint_sha256':NATIVE_SHA,'source_failure_manifest_sha256':FAILURE_SHA,
            'fixed_ehr_cases':2,'candidate_pairs':8,'available_scores':sum(r['biovil_raw_cosine'] is not None for r in candidates),
            'encoder_counts':endpoint['counts'],'new_model_calls':0,'new_generation_calls':0,
            'source_bodies_pixels_or_weights_read':False,'clinical_qualified':False,
            'clinical_repair_success':None,'selection_changed':False,
            'source_selection_sha256':data['seal'],'optional_trace_check_passed':None,
            'source_run_unchanged':True,'groups':groups(candidates,methods)}
        write_private_json(temporary/'summary.json',summary)
        if any(sha256_file(p)!=pin for p,pin in sources.items()):raise ValueError('source_cache_changed')
        check_pins(code_pins)
        artifacts={p.name:{'sha256':sha256_file(p)} for p in temporary.iterdir() if p.is_file()}
        write_private_json(temporary/'manifest.json',{'schema_version':VERSION,'sources':sources,
            'artifacts':artifacts,'code_pins':code_pins,'original_selection_changed':False,
            'clinical_qualified':False,'source_gpu_job_completed_successfully':False})
        commit_atomic_run(temporary,target)
    except BaseException:
        discard_atomic_run(temporary);raise
    return target,summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default=str(BASE/'paired_probe_endpoint_readouts'))
    p.add_argument('--run-id',required=True)
    args = p.parse_args()
    try:
        target,summary = run(args)
        print(json.dumps({'status':summary['status'],'manifest_sha256':sha256_file(target/'manifest.json'),
            'available_pairs':summary['available_scores'],'new_model_calls':0}))
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())

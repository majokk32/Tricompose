#!/usr/bin/env python3
"""Exact all-annotated cohort over broader published score release; no models."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import time

OLD=Path(__file__).with_name('run_radevalx_published_benchmark.py')
spec=importlib.util.spec_from_file_location('immutable_published_metric_worker_v1',OLD)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ADAPTER=base.ROOT/'real_validation/radevalx_published_adapter_v2.py'
spec=importlib.util.spec_from_file_location('frozen_published_metric_cohort_v2',ADAPTER)
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
SCHEMA='tricompose-radevalx-published-benchmark-v2'
PROTOCOL=base.WORKSPACE/'docs/report_metric_alignment_v2_protocol.md'
TESTS=(base.ROOT/'tests/test_radevalx_published_adapter_v2.py',
       base.ROOT/'tests/test_radevalx_published_benchmark_v2.py')


def rows(name):
    if name not in base.EXPECTED_HEADERS:
        raise ValueError('fixed_release_file_required')
    expected=590 if name=='metrcis_scores_m2tr.csv' else 100
    with (base.SOURCE/name).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=base.EXPECTED_HEADERS[name]:
            raise ValueError('unchanged_release_header_required')
        result=[]
        for row in reader:
            result.append(row)
            if len(result)>expected:
                raise ValueError('fixed_release_row_bound_exceeded')
    if len(result)!=expected:
        raise ValueError('fixed_release_row_inventory_required')
    return result


def metadata_cohort():
    published=rows('metrcis_scores_m2tr.csv')
    significant=rows('RadEval_clinically_significant_errors.csv')
    insignificant=rows('RadEval-clinically_insignificant_errors.csv')
    selected,inventory=adapter.cohort(published,significant,insignificant)
    return selected,significant,insignificant,inventory


def prepare(run_id):
    base.guard.guard()
    paths=(Path(__file__),ADAPTER,Path(adapter.base.__file__),PROTOCOL,OLD,base.PROTOCOL,*TESTS,*base.TESTS,
        Path(base.alignment.__file__),base.ROOT/'tools/score_cached_opacity_candidates.py',
        base.WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py')
    pins={str(p.resolve()):base.sha256_file(p) for p in paths}
    base.sources(pins)
    _,_,_,inventory=metadata_cohort()  # Keys/schema only; no numeric decode.
    temporary,target=base.new_atomic_run(base.BASE/'report_metric_alignment_plans',run_id)
    try:
        base.dump(temporary/'plan.json',{'schema_version':SCHEMA+'-plan','benchmark':'radevalx-1.0.0',
            'source_manifest_sha256':base.SOURCE_SHA,'inventory':inventory,
            'metric_columns':adapter.METRICS,'bootstrap_resamples':1000,'seed':0,'policy':base.alignment.POLICY,
            'annotation_key_metadata_used_before_predictions':True,
            'score_or_error_semantics_decoded_during_preparation':False,
            'published_results_available_to_investigator':True,'new_model_calls':0})
        base.guard.verify_pins(pins)
        base.dump(temporary/'manifest.json',{'schema_version':SCHEMA+'-plan-manifest','sources':pins,
            'artifacts':{p.name:base.sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified':False,'selection_changed':False,'regeneration_authorized':False})
        base.commit_atomic_run(temporary,target)
    except BaseException:
        base.discard_atomic_run(temporary)
        raise
    return target


def evaluate(plan_root,expected,run_id,allowed):
    base.require_analysis(allowed)
    pins={}
    manifest=base.guard.metadata(Path(plan_root)/'manifest.json',pins,expected)
    if manifest['schema_version']!=SCHEMA+'-plan-manifest':
        raise ValueError('frozen_v2_plan_required')
    base.guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan=base.guard.metadata(Path(plan_root)/'plan.json',pins,manifest['artifacts']['plan.json'])
    if plan['policy']!=base.alignment.POLICY or plan['metric_columns']!={k:list(v) for k,v in adapter.METRICS.items()} or \
            plan['bootstrap_resamples']!=1000 or plan['seed']!=0:
        raise ValueError('unchanged_metric_and_statistics_required')
    base.sources(pins)
    if base.sha256_file(base.guard.BANK/'manifest.json')!=base.guard.BANK_SHA:
        raise ValueError('original_bank_metadata_changed')
    temporary,target=base.new_atomic_run(base.BASE/'report_metric_alignment_runs',run_id)
    started=time.monotonic()
    try:
        selected,significant,insignificant,inventory=metadata_cohort()
        if inventory!=plan['inventory']:
            raise ValueError('frozen_all_annotated_inventory_required')
        predictions,internal_keys=adapter.predictions(selected)
        base.dump(temporary/'published_predictions.json',predictions)
        frozen=base.sha256_file(temporary/'published_predictions.json')
        base.dump(temporary/'prediction_freeze_receipt.json',{'prediction_sha256':frozen,
            'annotation_key_metadata_decoded_before_prediction_fsync':True,
            'reference_error_semantics_decoded_before_prediction_fsync':False,
            'source_reference_bytes_hashed_before_prediction_fsync':True,
            'published_results_available_to_investigator':True,'new_model_calls':0,'clinical_blinding_claimed':False})
        references=adapter.references(significant,insignificant,internal_keys)
        base.dump(temporary/'references.json',references)
        summary=base.alignment.evaluate(predictions,references,bootstrap_resamples=1000,seed=0)
        summary.update(elapsed_seconds=round(time.monotonic()-started,6),release_inventory=inventory,
            source_manifest_sha256=base.SOURCE_SHA,plan_manifest_sha256=expected,
            actual_slurm_job_id=os.environ['SLURM_JOB_ID'],source_csv_rows_read_internally=True,
            report_text_fields_accessed=False,original_source_keys_exported=False,
            real_mimic_inputs_read=False,image_or_ehr_inputs_read=False,
            published_predictions_not_new_local_model_outputs=True,new_gpu_calls=0,new_slurm_submissions=0)
        base.dump(temporary/'summary.json',summary)
        report=base.markdown(summary)+'\nRelease inventory: 590 score rows, all100 annotated pairs evaluated; 490 unannotated score rows not treated as zero-error cases.\n'
        base.write_private_text(temporary/'RESULTS_CN_EN.md',report)
        base.guard.verify_pins(pins)
        if base.sha256_file(temporary/'published_predictions.json')!=frozen or \
                base.sha256_file(base.guard.BANK/'manifest.json')!=base.guard.BANK_SHA:
            raise ValueError('frozen_prediction_or_original_bank_changed')
        base.dump(temporary/'manifest.json',{'schema_version':SCHEMA+'-run-manifest','sources':pins,
            'source_manifest_sha256':base.SOURCE_SHA,'plan_manifest_sha256':expected,
            'artifacts':{p.name:base.sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified':False,'selection_changed':False,'regeneration_authorized':False})
        base.commit_atomic_run(temporary,target)
    except BaseException:
        base.discard_atomic_run(temporary)
        raise
    return target


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('prepare','evaluate'))
    cli.add_argument('--run-id',required=True)
    cli.add_argument('--plan-root',type=Path)
    cli.add_argument('--plan-manifest-sha256')
    cli.add_argument('--allow-open-published-score-benchmark',action='store_true')
    args=cli.parse_args()
    os.umask(0o007)
    started=time.monotonic()
    try:
        target=prepare(args.run_id) if args.action=='prepare' else evaluate(args.plan_root,
            args.plan_manifest_sha256,args.run_id,args.allow_open_published_score_benchmark)
    except Exception as error:
        print(json.dumps({'status':'published_metric_benchmark_v2_failed_closed','error_type':type(error).__name__,
            'source_text_exposed':False}))
        return 2
    print(json.dumps({'status':'published_metric_v2_'+args.action+'_complete','model_calls':0,
        'elapsed_seconds':round(time.monotonic()-started,3),'manifest_sha256':base.sha256_file(target/'manifest.json')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

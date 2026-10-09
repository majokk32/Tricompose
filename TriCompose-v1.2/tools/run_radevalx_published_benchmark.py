#!/usr/bin/env python3
"""Released numeric metric/expert alignment, existing CPU Slurm, no models.

CSV report columns are unused. Only opaque derived numeric records and
aggregate summaries are exported protected. stdout is sanitized status/hash.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'),
               str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text, sha256_file)
from tricompose_v12 import report_metric_alignment as alignment
import score_cached_opacity_candidates as guard

ADAPTER_PATH = ROOT/'real_validation/radevalx_published_adapter.py'
spec = importlib.util.spec_from_file_location('frozen_published_radevalx_adapter', ADAPTER_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
BASE = PROTECTED_ROOT/'tricompose_v1_2'
SOURCE = BASE/'report_metric_sources/radevalx_source_12714150_002'
SOURCE_SHA = '7073624ae36b71b1581dc5e288902659513761b4051f0cd5b571821aaf6467ee'
PROTOCOL = WORKSPACE/'docs/report_metric_alignment_protocol.md'
TESTS = (ROOT/'tests/test_report_metric_alignment.py', ROOT/'tests/test_radevalx_published_adapter.py',
         ROOT/'tests/test_radevalx_published_benchmark.py')
SCHEMA = 'tricompose-radevalx-published-benchmark-v1'
EXPECTED_HEADERS = {'metrcis_scores_m2tr.csv': list(adapter.METRIC_FIELDS),
    'RadEval_clinically_significant_errors.csv': list(adapter.ANNOTATION_FIELDS),
    'RadEval-clinically_insignificant_errors.csv': list(adapter.ANNOTATION_FIELDS)}


def dump(path, payload):
    write_private_json(path, payload)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def sources(pins):
    manifest = guard.metadata(SOURCE/'manifest.json', pins, SOURCE_SHA)
    guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    summary = guard.metadata(SOURCE/'summary.json', pins, manifest['artifacts']['summary.json'])
    if summary['csv_headers'] != EXPECTED_HEADERS or summary['benchmark'] != 'radevalx-1.0.0':
        raise ValueError('fixed_release_source_schemas_required')
    for name in EXPECTED_HEADERS:
        path = SOURCE/name
        if sha256_file(path) != manifest['artifacts'][name] or not 0 < path.stat().st_size <= 256*1024:
            raise ValueError('bounded_unchanged_source_bytes_required')
        pins[str(path)] = manifest['artifacts'][name]
    return manifest


def require_analysis(allowed):
    guard.guard()
    if not allowed:
        raise ValueError('explicit_open_published_numeric_analysis_required')


def prepare(run_id):
    guard.guard()
    pins = {str(p.resolve()): sha256_file(p) for p in (Path(__file__), Path(alignment.__file__), ADAPTER_PATH,
        PROTOCOL, *TESTS, ROOT/'tools/score_cached_opacity_candidates.py',
        WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py')}
    sources(pins)
    temporary, target = new_atomic_run(BASE/'report_metric_alignment_plans', run_id)
    try:
        dump(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'benchmark': 'radevalx-1.0.0',
            'source_manifest_sha256': SOURCE_SHA, 'expected_pairs': 100,
            'metric_columns': adapter.METRICS, 'bootstrap_resamples': 1000, 'seed': 0,
            'reference_columns': list(range(1,9)), 'policy': alignment.POLICY,
            'source_rows_decoded_in_preparation': False, 'published_results_available_to_investigator': True,
            'new_model_calls': 0, 'threshold_fitting': False})
        guard.verify_pins(pins)
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def csv_rows(name):
    if name not in EXPECTED_HEADERS:
        raise ValueError('fixed_numeric_source_file_required')
    with (SOURCE/name).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_HEADERS[name]:
            raise ValueError('exact_csv_header_schema_required')
        result = []
        for row in reader:
            result.append(row)
            if len(result) > 100:
                raise ValueError('released_hundred_pair_bound_exceeded')
    return result


def markdown(summary):
    lines = ['# Published RadEvalX scores vs expert error counts / 独立数值诊断', '',
        'Published cached scores only. No local scorer inference or clinical qualification.',
        f"Attempted pairs: {summary['all_attempted_rows']}; source groups: {summary['source_groups']}.", '',
        '| Released metric | Paired rows | Coverage | Spearman quality vs -significant errors | Kendall tau-b | 95% group-bootstrap Spearman interval |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    render = lambda v: 'null' if v is None else f'{v:.4f}'
    for name, metric in summary['metrics'].items():
        outcome = metric['outcomes']['clinically_significant_total']
        interval = outcome['cluster_bootstrap']['spearman_interval_95']
        ci = 'null' if interval is None else '['+', '.join(render(v) for v in interval)+']'
        lines.append(f"| {name} | {outcome['paired_rows']} | {outcome['paired_coverage']:.1%} | "
            f"{render(outcome['spearman'])} | {render(outcome['kendall_tau_b'])} | {ci} |")
    lines += ['', 'All 8 error categories and significant/insignificant/all-error totals retained in summary.json.',
        'Constant/missing categories stay null. No rank direction, threshold, weight or scorer priority fitted.',
        'Text-reference errors are not image/EHR truth; released column identities do not qualify local implementations.',
        'No current bank, source EHR, report/image candidate, score, winner or regeneration gate changed.', '']
    return '\n'.join(lines)


def evaluate(plan_root, expected, run_id, allowed):
    require_analysis(allowed)
    pins = {}
    plan_manifest = guard.metadata(Path(plan_root)/'manifest.json', pins, expected)
    if plan_manifest['schema_version'] != SCHEMA+'-plan-manifest':
        raise ValueError('versioned_frozen_plan_required')
    guard.verify_pins(plan_manifest['sources'])
    pins.update(plan_manifest['sources'])
    plan = guard.metadata(Path(plan_root)/'plan.json', pins, plan_manifest['artifacts']['plan.json'])
    expected_metrics = {k: list(v) for k,v in adapter.METRICS.items()}
    if plan['policy'] != alignment.POLICY or plan['metric_columns'] != expected_metrics or \
            plan['expected_pairs'] != 100 or plan['bootstrap_resamples'] != 1000 or plan['seed'] != 0:
        raise ValueError('unchanged_predeclared_metric_statistics_required')
    sources(pins)
    if sha256_file(guard.BANK/'manifest.json') != guard.BANK_SHA:
        raise ValueError('original_bank_metadata_changed')
    temporary, target = new_atomic_run(BASE/'report_metric_alignment_runs', run_id)
    started = time.monotonic()
    try:
        predictions, internal_keys = adapter.predictions(csv_rows('metrcis_scores_m2tr.csv'))
        dump(temporary/'published_predictions.json', predictions)
        closed_prediction = sha256_file(temporary/'published_predictions.json')
        dump(temporary/'prediction_freeze_receipt.json', {'prediction_sha256': closed_prediction,
            'reference_annotation_semantics_decoded_before_prediction_fsync': False,
            'source_reference_bytes_hashed_before_prediction_fsync': True,
            'published_results_available_to_investigator': True, 'new_model_calls': 0,
            'clinical_blinding_claimed': False})
        references = adapter.references(csv_rows('RadEval_clinically_significant_errors.csv'),
            csv_rows('RadEval-clinically_insignificant_errors.csv'), internal_keys)
        dump(temporary/'references.json', references)
        summary = alignment.evaluate(predictions, references, bootstrap_resamples=1000, seed=0)
        summary.update(elapsed_seconds=round(time.monotonic()-started,6),
            source_manifest_sha256=SOURCE_SHA, plan_manifest_sha256=expected,
            actual_slurm_job_id=os.environ['SLURM_JOB_ID'], source_csv_rows_read_internally=True,
            report_text_fields_accessed=False, original_source_keys_exported=False,
            real_mimic_inputs_read=False, image_or_ehr_inputs_read=False,
            published_predictions_not_new_local_model_outputs=True, new_gpu_calls=0, new_slurm_submissions=0)
        dump(temporary/'summary.json',summary)
        write_private_text(temporary/'RESULTS_CN_EN.md',markdown(summary))
        guard.verify_pins(pins)
        if sha256_file(temporary/'published_predictions.json') != closed_prediction or \
                sha256_file(guard.BANK/'manifest.json') != guard.BANK_SHA:
            raise ValueError('sealed_prediction_or_original_bank_changed')
        dump(temporary/'manifest.json',{'schema_version':SCHEMA+'-run-manifest','sources':pins,
            'source_manifest_sha256':SOURCE_SHA,'plan_manifest_sha256':expected,
            'artifacts':{p.name:sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified':False,'selection_changed':False,'regeneration_authorized':False})
        commit_atomic_run(temporary,target)
    except BaseException:
        discard_atomic_run(temporary)
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
        print(json.dumps({'status':'published_metric_benchmark_failed_closed','error_type':type(error).__name__,
            'source_text_exposed':False}))
        return 2
    print(json.dumps({'status':'published_metric_'+args.action+'_complete','model_calls':0,
        'elapsed_seconds':round(time.monotonic()-started,3),'manifest_sha256':sha256_file(target/'manifest.json')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

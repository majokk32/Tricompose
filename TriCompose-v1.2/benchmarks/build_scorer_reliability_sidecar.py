#!/usr/bin/env python3
"""Append reliability scope to all 960 cached synthetic candidates, CPU only.

Reads hash-bound score/finding caches and real-benchmark AGGREGATES only. Never
loads image pixels, source EHR/report text, patient keys, models or a selector.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from tricompose_v12.scorer_reliability import SCHEMA, build_overlay

BANK_MANIFEST_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
XRV_MANIFEST_SHA = '92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0'
BIOVIL_MANIFEST_SHA = 'a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a'


def bounded_json(path, limit=1024*1024):
    if path.stat().st_size > limit:
        raise ValueError('metadata_size_limit_exceeded')
    return read_json(path)


def benchmark_aggregate(root, manifest_sha, schema):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    manifest = root / 'manifest.json'
    if sha256_file(manifest) != manifest_sha:
        raise ValueError('benchmark_manifest_changed')
    receipt = bounded_json(manifest)
    if receipt['schema_version'] != schema:
        raise ValueError('benchmark_schema_changed')
    summary = root / 'summary.json'
    if sha256_file(summary) != receipt['artifacts']['summary.json']:
        raise ValueError('benchmark_aggregate_changed')
    result = bounded_json(summary)
    if 'records' in result:
        raise ValueError('aggregate_only_no_patient_records')
    return result, manifest, summary


def reliability_profile(xrv, biovil, recorded_biovil_weights):
    if 'records' in xrv or 'records' in biovil:
        raise ValueError('aggregate_only_no_patient_records')
    if (xrv['frozen'] is not True or biovil['frozen'] is not True
            or xrv['selection_changed'] is not False or biovil['selection_changed'] is not False
            or xrv['primary_metric_eligible'] is not False or biovil['primary_metric_eligible'] is not False
            or xrv['thresholds_fitted_on_this_cohort'] is not False or biovil['thresholds_fitted'] is not False
            or biovil['template_choice_fitted'] is not False
            or biovil['primary_reduction'] != 'mean_all_three_predeclared'
            or xrv['probability_semantics'] is not False or biovil['probability_semantics'] is not False
            or xrv['reference_kind'] != biovil['reference_kind']
            or xrv['model_calls'] != 50 or biovil['execution']['source_cases'] != 50):
        raise ValueError('unchanged_diagnostic_benchmark_required')
    return {'schema_version': SCHEMA + '-profile',
        'reference_kind': xrv['reference_kind'], 'published_cohort_proxy_not_adjudicated_image_truth': True,
        'primary_metric_eligible': False, 'numeric_benchmark_metric_transferred_to_candidates': False,
        'masks_thresholds_weights_or_reliability_bands_fitted': False,
        'xrv_pneumonia_diagnostic': {'default_0_5': xrv['default_0_5'],
            'unchanged_weak_reference_threshold': xrv['unchanged_weak_reference_threshold_transport'],
            'score_space': xrv['score_space'], 'checkpoint_sha256': xrv['checkpoint_sha256'],
            'thresholds_sha256': xrv['thresholds_sha256'],
            'historical_candidate_checkpoint_match_verified': None,
            'fresh_threshold_profile_applied_to_legacy_cache': False,
            'interpretation': 'Finding/task diagnostic only; a negative proxy does not confirm clinical absence.'},
        'biovil_pneumonia_polarity_diagnostic': {'all_predeclared_templates': biovil['templates'],
            'primary_reduction': biovil['primary_reduction'], 'cross_scorer': biovil['cross_scorer'],
            'recorded_image_checkpoint_hash_match': recorded_biovil_weights['image'] == biovil['producer']['image_checkpoint_sha256'],
            'recorded_text_checkpoint_hash_match': recorded_biovil_weights['text'] == biovil['producer']['text_checkpoint_sha256'],
            'task_matches_candidate_full_report_endpoint': False,
            'interpretation': 'Global report cosine remains retrieval evidence, not a clinical finding/polarity label.'},
        'candidate_same_fact_image_scorer_disagreement': None,
        'candidate_same_fact_image_scorer_disagreement_status': 'not_measured_on_synthetic_bank',
        'report_expert_disagreement_source': 'cached_chexbert_proposals_on_correlated_same_image_reports',
        'clinical_fault_localization': None, 'automatic_regeneration_authorized': False}


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: ('' if value is None else str(value).lower() if isinstance(value, bool)
                            else json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value)
                     for key, value in row.items()} for row in rows)
    return stream.getvalue()


def markdown(summary):
    lines = ['# 候选评分可靠性 Sidecar / Candidate scorer reliability', '',
        f"{summary['fixed_ehr_cases']} fixed EHRs, {summary['image_slots']} image slots, "
        f"{summary['candidate_rows']} candidate triples, {summary['fact_rows']} cached fact rows.", '',
        '原始 CSV 每个字段及行顺序保留；只添加 reliability_ 字段，不重新排名或改旧 winner。',
        'All original cells and row order are preserved. No new scoring, selection or repair.', '',
        '## Evidence availability / 证据可用性', '',
        '| Cached evidence status | EHR–CXR candidate rows | EHR–Report candidate rows | CXR–Report candidate rows |',
        '|---|---:|---:|---:|']
    from tricompose_v12.scorer_reliability import REASONS
    for label in REASONS:
        values = summary['edge_evidence_status_candidate_counts']
        lines.append(f"| {label} | {values['ehr_cxr'][label]} | {values['ehr_report'][label]} | {values['cxr_report'][label]} |")
    lines += ['', f"Fixed cases with explicit cached EHR facts: {summary['fixed_ehr_cases_with_explicit_cached_facts']}; "
        f"without: {summary['fixed_ehr_cases_without_explicit_cached_facts']}.",
        f"Same-image/finding groups with explicit positive–negative report-proposal disagreement: "
        f"{summary['same_image_report_expert_proxy_disagreement_groups']}/{summary['image_finding_dependency_groups']}.", '',
        '## Interpretation / 如何使用', '',
        '- 原始 proxy support/opposition 不等于已确认的临床支持/错误；临床 score 和错误模态保持 unavailable。',
        '- Unknown/uncertain are retained, never converted into negative or perfect consistency.',
        '- RSUA 只评估公开肺炎/正常队列代理；其指标不是 synthetic 候选的新标签、概率、权重或阈值。',
        '- BioViL full-report cosine stays secondary retrieval evidence. A pneumonia polarity probe is a different task.',
        '- Report disagreements are cached CheXbert proposal disagreements; shared-image reports are correlated, not independent votes.',
        '- 尚未在这些 synthetic 图上运行第二个同 finding 的图像评分器；image scorer disagreement 保持 NA，不填零。',
        '- Same-image group counts and repeated fact rows are not independent patient/sample counts.',
        '- No new models, body/pixel reads, training, API, submission, ranking, threshold or winner change.', '',
        '## Files / 文件', '',
        '`candidate_score_table.csv`: original table plus reliability annotations.',
        '`fact_reliability.jsonl`: lossless four-state facts and explicit evidence scope.',
        '`report_dependency_groups.jsonl`: deduplicated report-artifact state groups.',
        '`scorer_reliability_profile.json`: benchmark aggregates and non-transfer boundary.',
        '`summary.json`, `manifest.json`: inventory, invariants and hashes.', '',
        'This is a diagnostic overlay, not a validated Agent or a new clinical selection policy.']
    return '\n'.join(lines) + '\n'


def run(args):
    # Existing CPU allocation only; no new Slurm submission is performed.
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('existing_cpu_slurm_required')
    started = time.monotonic()
    root = require_inside(args.candidate_run, PROTECTED_ROOT, must_exist=True)
    mp = root / 'manifest.json'
    if sha256_file(mp) != BANK_MANIFEST_SHA:
        raise ValueError('full_bank_manifest_changed')
    manifest = bounded_json(mp)
    if (manifest['schema_version'] != 'tricompose-full-bank-secondary-coverage-v1'
            or manifest['original_selection_changed'] is not False or manifest['clinical_acceptance'] is not False):
        raise ValueError('unchanged_complete_bank_required')
    source = root / 'candidate_score_table.csv'
    if source.stat().st_size > 4*1024*1024 or sha256_file(source) != manifest['artifacts'][source.name]['sha256']:
        raise ValueError('candidate_csv_changed_or_unbounded')
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    names = ('historical_source_scores', 'historical_source_edges', 'historical_policy_config')
    paths = {name: require_inside(manifest['source_paths'][name], PROTECTED_ROOT, must_exist=True) for name in names}
    if any(sha256_file(path) != manifest['source_sha256'][name] for name, path in paths.items()):
        raise ValueError('cached_state_source_changed')
    if paths['historical_source_scores'].stat().st_size > 4*1024*1024:
        raise ValueError('cached_score_source_unbounded')
    scores = [json.loads(line) for line in paths['historical_source_scores'].read_text().splitlines() if line]
    details = bounded_json(paths['historical_source_edges'], 8*1024*1024)
    policy = bounded_json(paths['historical_policy_config'], 16*1024)
    bank = make_legacy_bank(scores, details, policy)
    if len(rows) != 960 or len(bank) != 80 or any(len(grid) != 12 for grid in bank.values()):
        raise ValueError('complete_80_by_3_by_4_inventory_required')
    xrv, xm, xs = benchmark_aggregate(args.xrv_run, XRV_MANIFEST_SHA, 'tricompose-rsua-xrv-pilot-v1-manifest')
    biovil, bm, bs = benchmark_aggregate(args.biovil_run, BIOVIL_MANIFEST_SHA, 'tricompose-rsua-biovil-polarity-v1-manifest')
    profile = reliability_profile(xrv, biovil, {
        'image': manifest['source_sha256']['model_biovil_t_image_model_proj_size_128.pt'],
        'text': manifest['source_sha256']['model_pytorch_model.bin']})
    sources = {**paths, 'candidate_manifest': mp, 'candidate_csv': source,
        'xrv_manifest': xm, 'xrv_aggregate': xs, 'biovil_manifest': bm, 'biovil_aggregate': bs,
        'program': Path(__file__), 'overlay_contract': ROOT/'src/tricompose_v12/scorer_reliability.py',
        'cache_adapter': ROOT/'src/tricompose_v12/legacy_replay_adapter.py',
        'edge_contract': ROOT/'src/tricompose_v12/invariant_verification.py'}
    before = {key: sha256_file(path) for key, path in sources.items()}
    annotated, facts, groups, summary = build_overlay(rows, bank)
    summary.update(status='completed_lossless_candidate_reliability_sidecar',
        source_original_column_count=len(rows[0]), output_column_count=len(annotated[0]),
        input_profile_unchanged=True, benchmark_profile_pooled_with_candidate_scores=False,
        existing_cpu_job_id=os.environ['SLURM_JOB_ID'], new_slurm_submissions=0,
        report_bodies_or_image_pixels_opened=False, raw_patient_inputs_opened=False,
        runtime_seconds=round(time.monotonic()-started, 6))
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_text(temporary/'candidate_score_table.csv', csv_text(annotated)),
            write_private_text(temporary/'fact_reliability.jsonl', ''.join(json.dumps(row, sort_keys=True)+'\n' for row in facts)),
            write_private_text(temporary/'report_dependency_groups.jsonl', ''.join(json.dumps(row, sort_keys=True)+'\n' for row in groups)),
            write_private_json(temporary/'scorer_reliability_profile.json', profile),
            write_private_json(temporary/'summary.json', summary),
            write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))]
        if before != {key: sha256_file(path) for key, path in sources.items()}:
            raise ValueError('immutable_sources_changed_during_annotation')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': args.run_id,
            'source_paths': {key: str(path) for key, path in sources.items()}, 'source_sha256': before,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in files},
            'original_selection_changed': False, 'thresholds_or_weights_changed': False,
            'new_model_calls': 0, 'regeneration_authorized': False, 'primary_metric_eligible': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-run', 'xrv-run', 'biovil-run', 'output-root', 'run-id'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    try:
        target, summary = run(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': summary['status'], 'candidate_rows': summary['candidate_rows'],
        'fact_rows': summary['fact_rows'], 'new_model_calls': 0,
        'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

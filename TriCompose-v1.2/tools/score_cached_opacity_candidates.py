#!/usr/bin/env python3
"""Separate exact-opacity diagnostic for the existing 80 x 3 x 4 synthetic bank.

Prepare reads bounded metadata only. Evaluate needs separately approved GPU
Slurm and opens only hash-bound synthetic CXR files. No generator, selector,
patient source, EHR/report body, threshold fitting or benchmark row is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (PROTECTED_ROOT, WORKSPACE, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.legacy_replay_adapter import make_legacy_bank, PROFILE

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
BENCHMARK = BASE / 'real_validation/ricord_xrv_pilots/ricord_xrv50_12667524'
BENCHMARK_SHA = 'dc14a2ba1aed3823f0752a73dda297e874f638b31644d2992227eb3f1cf697a9'
WEIGHT_DIR = WORKSPACE / 'artifacts/protected/tricompose_v1_1/evaluation/smoke48/scoring_12257177/cache/xrv'
WEIGHT_NAME = 'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
WEIGHT_SHA = '56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899'
SCHEMA = 'tricompose-cached-exact-opacity-sidecar-v1'
STATES = frozenset(('positive', 'negative', 'uncertain', 'unknown'))
EXPLICIT = frozenset(('positive', 'negative'))
MODELS = frozenset(('roentgen_v2', 'chexgenbench_sana', 'chexgenbench_pixart'))
EXPERTS = frozenset(('maira2', 'cxrmate_single', 'llavarad', 'chexagent2'))
POLICY = {
    'finding': 'lung_opacity', 'exact_head': 'lung_opacity', 'exact_threshold': 0.5,
    'secondary_same_call_max_heads': ['lung_opacity', 'infiltration'],
    'secondary_same_call_max_threshold': 0.5,
    'score_space': 'xrv_op_norm_0_1', 'probability_semantics': False,
    'unknown_is_negative': False, 'pneumonia_is_opacity': False,
    'global_no_finding_expands_unknown': False, 'new_ehr_fact_extraction': False,
    'clinical_fault_localization': False, 'selector_or_repair_enabled': False,
    'benchmark_accuracy_transferred_to_synthetic': False,
    'threshold_fitting': False, 'primary_metric_eligible': False,
}


def guard(*, gpu=False, approved=False):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('actual_slurm_allocation_required')
    if gpu and (not approved or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('separately_approved_gpu_required')
    if not gpu and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')):
        raise RuntimeError('existing_cpu_metadata_allocation_required')


def metadata(path, pins, expected=None, *, cap=8 * 1024**2, workspace=False):
    path = require_inside(path, WORKSPACE if workspace else PROTECTED_ROOT, must_exist=True)
    if path.suffix not in ('.json', '.jsonl', '.csv') or path.name in (
            'synthetic_ehr.json', 'ehr_facts.json', 'tokenizer_trace.json'):
        raise ValueError('metadata_only_no_clinical_bodies')
    if not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('bounded_metadata_required')
    actual = sha256_file(path)
    if expected is not None and expected != actual:
        raise ValueError('metadata_hash_changed')
    text = path.read_text(encoding='utf-8')
    if sha256_file(path) != actual:
        raise ValueError('metadata_changed_during_read')
    pins[str(path)] = actual
    if path.suffix == '.csv':
        return list(csv.DictReader(io.StringIO(text)))
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line]
    return json.loads(text)


def verify_pins(pins):
    for path, digest in pins.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('immutable_source_pin_changed')


def state(score):
    if score is None:
        return 'unknown'
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('finite_operating_point_score_required')
    return 'positive' if score >= POLICY['exact_threshold'] else 'negative'


def relation(left, right):
    if left not in STATES or right not in STATES:
        raise ValueError('four_state_evidence_required')
    if left not in EXPLICIT or right not in EXPLICIT:
        return 'not_comparable'
    return 'proxy_support' if left == right else 'proxy_opposition'


def flatten(rows, bank):
    """One cached opacity assertion per triple, retaining original CSV order."""
    candidates = {c['score_record']['triple_candidate_id']: c for grid in bank.values() for c in grid.values()}
    if len(candidates) != sum(len(g) for g in bank.values()) or len(rows) != len(candidates) or \
            len({r['triple_candidate_id'] for r in rows}) != len(rows) or \
            set(r['triple_candidate_id'] for r in rows) != set(candidates):
        raise ValueError('exact_unique_candidate_inventory_required')
    result = []
    for row in rows:
        c = candidates[row['triple_candidate_id']]; original = c['score_record']; lineage = original['lineage']
        if row.get('profile') != PROFILE or row['case_id'] != original['case_id'] or \
                any(row[k] != lineage[k] for k in ('cxr_candidate_id', 'report_candidate_id', 'report_model_id',
                    'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')):
            raise ValueError('source_row_lineage_changed')
        facts = [f for f in c['facts'] if f['finding'] == 'lung_opacity']
        if len(facts) != 1 or facts[0]['clinical_truth_verified'] is not False or facts[0]['weak_context_promoted'] is not False:
            raise ValueError('one_unqualified_cached_opacity_fact_required')
        fact = facts[0]
        if any(s not in STATES for s in fact['states'].values()) or \
                fact['states']['ehr'] in EXPLICIT and not fact['source_categories']:
            raise ValueError('cached_state_and_source_category_required')
        result.append({k: row[k] for k in ('case_id', 'triple_candidate_id', 'cxr_candidate_id',
            'report_candidate_id', 'report_model_id', 'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')} | {
            'cxr_model_id': lineage['cxr_model_id'], 'cxr_seed': lineage['cxr_seed'],
            'ehr_opacity_state': fact['states']['ehr'], 'report_opacity_state': fact['states']['chexbert'],
            'cached_legacy_max_opacity_state': fact['states']['xrv'],
            'cached_ehr_source_categories': list(fact['source_categories'])})
    return result


def inventory(records, *, full=False):
    cases, images, reports = {}, {}, {}
    for r in records:
        for index, key, value in (
            (cases, r['case_id'], (r['ehr_sha256'], r['ehr_facts_sha256'], r['ehr_opacity_state'])),
            (images, r['cxr_candidate_id'], (r['case_id'], r['cxr_sha256'], r['cxr_model_id'], r['cxr_seed'], r['cached_legacy_max_opacity_state'])),
            (reports, r['report_sha256'], r['report_opacity_state'])):
            if index.setdefault(key, value) != value:
                raise ValueError('shared_artifact_or_fixed_ehr_changed')
    if full and (len(records) != 960 or len(cases) != 80 or len(images) != 240 or
            any(sum(r['case_id'] == c for r in records) != 12 for c in cases) or
            any(sum(r['cxr_candidate_id'] == i for r in records) != 4 for i in images) or
            {r['cxr_model_id'] for r in records} != MODELS or {r['report_model_id'] for r in records} != EXPERTS):
        raise ValueError('full_fixed_eighty_three_four_grid_required')
    return {'fixed_ehr_cases': len(cases), 'image_slots': len(images), 'candidate_rows': len(records),
        'unique_report_artifact_hashes': len(reports),
        'ehr_opacity_states_one_per_case': dict(Counter(v[2] for v in cases.values())),
        'cached_legacy_max_states_one_per_image': dict(Counter(v[4] for v in images.values())),
        'report_opacity_states_candidate_rows': dict(Counter(r['report_opacity_state'] for r in records)),
        'report_opacity_states_unique_artifact_hashes': dict(Counter(reports.values())),
        'exact_head_scores_available_in_old_cache': False,
        'ehr_opacity_explicit_cases': sum(v[2] in EXPLICIT for v in cases.values())}


def cached_relations(records):
    """Old definition, explicitly secondary; no new exact-head value is implied."""
    return dict(Counter(relation(r['cached_legacy_max_opacity_state'], r['report_opacity_state']) for r in records))


def resolve_images(records, manifest, pins):
    expected = {r['cxr_candidate_id']: {k: r[k] for k in ('case_id', 'cxr_candidate_id',
        'cxr_sha256', 'cxr_model_id', 'cxr_seed', 'ehr_sha256', 'ehr_facts_sha256')} for r in records}
    images = {}
    for key in sorted(k for k in manifest['source_paths'] if k.startswith('cxr_manifest_')):
        mp = Path(manifest['source_paths'][key]); m = metadata(mp, pins, manifest['source_sha256'][key])
        if m['schema_version'] != 'tricompose-cxr-candidate-run-v1.1' or m['frozen_model'] is not True or \
                m['model_id'] not in MODELS or m['candidate_count'] != 80 or len(m['candidates']) != 80:
            raise ValueError('fixed_frozen_image_run_required')
        requests = metadata(Path(m['source_request_run']) / 'manifest.json', pins, m['source_request_run_manifest_sha256'])
        staged = metadata(Path(requests['staging_run']) / 'run_manifest.json', pins, requests['staging_run_manifest_sha256'])
        if staged['schema_version'] != 'tricompose.staging.run.v1.1' or staged['canonical_ehr_modified'] is not False:
            raise ValueError('unchanged_synthetic_ehr_staging_required')
        origin = metadata(Path(staged['source_v1_staging_run']) / 'run_manifest.json', pins,
                          staged['source_v1_run_manifest_sha256'], workspace=True)
        if origin['schema_version'] != 'tricompose.staging.run.v1' or origin['source_generator'] not in (
                'synehrgy_gpt2_10bins', 'synehrgy_qwen2_40bins', 'synehrgy_v2'):
            raise ValueError('fully_synthetic_origin_required')
        for entry in m['candidates']:
            cp = require_inside(mp.parent / entry['path'], mp.parent, must_exist=True)
            c = metadata(cp, pins, entry['sha256']); cid = c['candidate_id']
            if cid not in expected or cid in images:
                raise ValueError('unique_expected_image_required')
            r = expected[cid]
            if c['schema_version'] != 'tricompose-cxr-candidate-v1.1' or c['modality'] != 'cxr' or c['frozen_model'] is not True or \
                    c['case_id'] != entry['case_id'] or c['case_id'] != r['case_id'] or \
                    c['model_id'] != m['model_id'] or c['model_id'] != r['cxr_model_id'] or \
                    c['seed'] != r['cxr_seed'] or c['seed'] != 0 or \
                    c['ehr_sha256'] != r['ehr_sha256'] or c['ehr_facts_sha256'] != r['ehr_facts_sha256'] or \
                    c['artifact']['sha256'] != r['cxr_sha256']:
                raise ValueError('exact_frozen_synthetic_image_lineage_required')
            path = require_inside(c['artifact']['path'], mp.parent, must_exist=True)
            if path.suffix.lower() != '.png' or not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024**2:
                raise ValueError('bounded_existing_synthetic_png_required')
            images[cid] = {**r, 'path': str(path), 'file_size_bytes': path.stat().st_size,
                           'file_mtime_ns': path.stat().st_mtime_ns}
    if set(images) != set(expected) or len(images) != 240:
        raise ValueError('full_240_image_inventory_required')
    return [images[cid] for cid in sorted(images)]


def versions():
    return {name: importlib.metadata.version(name) for name in (
        'torch', 'torchxrayvision', 'numpy', 'Pillow', 'scikit-image')}


def source_pins():
    files = [Path(__file__), ROOT / 'tests/test_cached_opacity_candidates.py',
        ROOT.parent / 'docs/cached_opacity_candidates_protocol.md',
        ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
        ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1/extract_cxr_labels_xrv.py',
        ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1/xrv_calibration.py']
    # Pin the read-only legacy adapter import closure; do not add/edit these files.
    files += list((ROOT / 'src/tricompose_v12').glob('*.py'))
    return {str(p): sha256_file(p) for p in files}


def load_preparation():
    pins = source_pins()
    m = metadata(BANK / 'manifest.json', pins, BANK_SHA)
    if m['schema_version'] != 'tricompose-full-bank-secondary-coverage-v1' or \
            m['original_selection_changed'] is not False or m['clinical_acceptance'] is not False:
        raise ValueError('unchanged_complete_bank_required')
    rows = metadata(BANK / 'candidate_score_table.csv', pins, m['artifacts']['candidate_score_table.csv']['sha256'])
    values = {k: metadata(Path(m['source_paths'][k]), pins, m['source_sha256'][k]) for k in (
        'historical_source_scores', 'historical_source_edges', 'historical_policy_config')}
    details = values['historical_source_edges']
    bank = make_legacy_bank(values['historical_source_scores'], details, values['historical_policy_config'])
    records = flatten(rows, bank); counts = inventory(records, full=True)
    cxr = details['evidence']['cxr_labels']; report = details['evidence']['report_labels']
    xrv = metadata(Path(cxr['path']), pins, cxr['sha256'])
    chexbert = metadata(Path(report['path']), pins, report['sha256'])
    if xrv['producer']['frozen'] is not True or xrv['producer']['checkpoint_sha256'] != WEIGHT_SHA or \
            xrv['producer']['model_id'] != 'densenet121-res224-all' or \
            xrv['calibration']['status'] != 'uncalibrated_default_0_5' or chexbert['producer']['frozen'] is not True:
        raise ValueError('same_checkpoint_legacy_label_caches_required')
    xi = {r['cxr_candidate_id']: r for r in xrv['records']}; ri = {r['report_candidate_id']: r for r in chexbert['records']}
    if len(xi) != 240 or len(ri) != 960 or len(xrv['records']) != 240 or len(chexbert['records']) != 960:
        raise ValueError('complete_original_label_cache_required')
    for r in records:
        a, b = xi[r['cxr_candidate_id']], ri[r['report_candidate_id']]
        if a['image_sha256'] != r['cxr_sha256'] or a['finding_states']['lung_opacity'] != r['cached_legacy_max_opacity_state'] or \
                b['report_sha256'] != r['report_sha256'] or b['finding_states']['lung_opacity'] != r['report_opacity_state']:
            raise ValueError('original_label_cache_lineage_changed')
    bm = metadata(BENCHMARK / 'manifest.json', pins, BENCHMARK_SHA)
    summary = metadata(BENCHMARK / 'summary.json', pins, bm['artifacts']['summary.json'])
    if 'records' in summary or summary['planned_cases'] != 50 or summary['scored_cases'] != 50 or summary['failed_cases'] != 0 or \
            summary['frozen'] is not True or summary['primary_metric_eligible'] is not False or \
            summary['scorer']['checkpoint_sha256'] != WEIGHT_SHA or summary['scorer']['exact_head_threshold'] != 0.5:
        raise ValueError('completed_aggregate_only_opacity_benchmark_required')
    for path, digest in summary['scorer']['external_readonly_sources'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('benchmark_external_xrv_source_changed')
        pins[path] = digest
    weight = WEIGHT_DIR / WEIGHT_NAME
    if sha256_file(weight) != WEIGHT_SHA:
        raise ValueError('unchanged_xrv_weight_required')
    pins[str(weight)] = WEIGHT_SHA
    plan = {'schema_version': SCHEMA + '-plan', 'modality_source': 'fully_synthetic',
        'policy': POLICY, 'original_rows': rows, 'cached_records': records,
        'images': resolve_images(records, m, pins), 'inventory': counts,
        'environment_versions': versions(), 'cached_xrv_checkpoint_matches_benchmark': True,
        'historical_xrv_runtime_preprocessing_fingerprint_available': False,
        'benchmark_summary_only': {'metrics': summary['metrics'],
            'reference_kind': summary['reference_kind'], 'scorer': summary['scorer']},
        'new_exact_head_scores_available': False, 'no_patient_source_or_clinical_body_reads': True,
        'image_pixels_opened': False, 'model_calls': 0, 'old_scores_and_choices_changed': False}
    verify_pins(pins)
    return plan, pins


def csv_text(rows):
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def overlay(rows, records, results):
    """Lossless CSV append, with failure/unknown retained in all denominators."""
    if len(rows) != len(records) or not rows:
        raise ValueError('matching_nonempty_rows_required')
    if any(any(k.startswith('opacity_') for k in r) for r in rows):
        raise ValueError('already_annotated_source_refused')
    expected = {r['cxr_candidate_id'] for r in records}
    if set(results) != expected:
        raise ValueError('all_image_outcomes_required_including_failures')
    output, by_model = [], defaultdict(list)
    for row, r in zip(rows, records, strict=True):
        if any(row[k] != r[k] for k in ('triple_candidate_id', 'case_id', 'cxr_candidate_id', 'report_sha256')):
            raise ValueError('original_row_order_and_lineage_required')
        s = results[r['cxr_candidate_id']]
        if s['cxr_candidate_id'] != r['cxr_candidate_id'] or s['cxr_sha256'] != r['cxr_sha256'] or s['case_id'] != r['case_id']:
            raise ValueError('new_score_image_lineage_changed')
        exact, infiltration = s['exact_lung_opacity_score'], s['infiltration_score']
        if s['status'] == 'scored':
            if exact is None or infiltration is None:
                raise ValueError('both_raw_heads_required')
            xstate = state(exact); state(infiltration); maximum = max(exact, infiltration)
        elif s['status'] == 'failed_without_replacement' and exact is None and infiltration is None:
            xstate, maximum = 'unknown', None
        else:
            raise ValueError('explicit_scored_or_failed_outcome_required')
        values = {'opacity_exact_score': exact, 'opacity_exact_state_0_5': xstate,
            'opacity_infiltration_score': infiltration, 'opacity_same_call_max_score': maximum,
            'opacity_same_call_max_state_0_5': state(maximum),
            'opacity_cached_legacy_max_state': r['cached_legacy_max_opacity_state'],
            'opacity_exact_vs_same_call_max_state_changed': None if maximum is None else xstate != state(maximum),
            'opacity_cached_ehr_state': r['ehr_opacity_state'], 'opacity_cached_report_state': r['report_opacity_state'],
            'opacity_ehr_cxr_relation': relation(r['ehr_opacity_state'], xstate),
            'opacity_ehr_report_relation': relation(r['ehr_opacity_state'], r['report_opacity_state']),
            'opacity_cxr_report_relation': relation(xstate, r['report_opacity_state']),
            'opacity_scoring_status': s['status'], 'opacity_clinical_accuracy': None,
            'opacity_confirmed_faulty_modality': None, 'opacity_selector_used': False}
        output.append({**row, **values})
        by_model[r['cxr_model_id'], r['report_model_id']].append(values['opacity_cxr_report_relation'])
    model_readouts = []
    for (image, expert), relations in sorted(by_model.items()):
        counts = Counter(relations); comparable = counts['proxy_support'] + counts['proxy_opposition']
        model_readouts.append({'cxr_model_id': image, 'report_model_id': expert,
            'candidate_rows': len(relations), 'comparable_pairs': comparable,
            'proxy_support_pairs': counts['proxy_support'], 'proxy_opposition_pairs': counts['proxy_opposition'],
            'not_comparable_pairs': counts['not_comparable'],
            'agreement_over_comparable': counts['proxy_support'] / comparable if comparable else None,
            'comparable_coverage_over_all_rows': comparable / len(relations),
            'clinical_accuracy': None, 'independent_patients': False})
    return output, model_readouts


def prepare(args):
    guard(); started = time.monotonic(); plan, pins = load_preparation()
    report = {'schema_version': SCHEMA + '-preparation', 'status': 'frozen_plan_no_inference',
        **plan['inventory'], 'new_model_calls': 0, 'new_slurm_submissions': 0,
        'clinical_bodies_or_pixels_opened': False, 'old_scores_and_choices_changed': False,
        'next_gpu_images': 240, 'new_triple_rows_after_scoring': 960,
        'cached_max_cannot_recover_exact_head': True, 'primary_metric_eligible': False,
        'cached_legacy_cxr_report_relations_candidate_rows': cached_relations(plan['cached_records']),
        'elapsed_cpu_seconds': round(time.monotonic() - started, 6)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'plan.json', plan)
        write_private_json(temporary / 'preparation.json', report)
        write_private_text(temporary / 'RESULTS_CN_EN.md', '# Exact opacity sidecar preparation / 固定候选准备\n\n'
            'No inference yet. Exact head cannot be recovered from the historical maximum.\n'
            'EHR unknowns remain unknown; no pneumonia-to-opacity label conversion or winner update.\n\n'
            + json.dumps(report, indent=2, sort_keys=True) + '\n')
        verify_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-plan-manifest',
            'sources': pins, 'artifacts': {n: sha256_file(temporary / n) for n in (
                'plan.json', 'preparation.json', 'RESULTS_CN_EN.md')}, 'new_model_calls': 0,
            'old_scores_and_choices_changed': False, 'primary_metric_eligible': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, report


def evaluate(args):
    guard(gpu=True, approved=args.allow_synthetic_xrv)
    if not isinstance(args.plan_manifest_sha256, str) or len(args.plan_manifest_sha256) != 64 or \
            any(c not in '0123456789abcdef' for c in args.plan_manifest_sha256):
        raise ValueError('explicit_frozen_plan_sha256_required')
    # Refuse an existing output before imports/model loading/image reads.
    prospective = require_inside(Path(args.output_root) / args.run_id, PROTECTED_ROOT, must_exist=False)
    if prospective.exists():
        raise FileExistsError('output_run_exists')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('offline_scoring_required')
    started = time.monotonic(); pins = {}
    plan_root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    m = metadata(plan_root / 'manifest.json', pins, args.plan_manifest_sha256)
    if m['schema_version'] != SCHEMA + '-plan-manifest':
        raise ValueError('frozen_opacity_plan_required')
    verify_pins(m['sources']); pins.update(m['sources'])
    plan = metadata(plan_root / 'plan.json', pins, m['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA + '-plan' or plan['policy'] != POLICY or \
            plan['modality_source'] != 'fully_synthetic' or plan['old_scores_and_choices_changed'] is not False or \
            plan['new_exact_head_scores_available'] is not False or plan['environment_versions'] != versions():
        raise ValueError('unchanged_policy_and_environment_required')
    inventory(plan['cached_records'], full=True)
    expected = {r['cxr_candidate_id']: (r['case_id'], r['cxr_sha256'], r['cxr_model_id']) for r in plan['cached_records']}
    images = plan['images']
    if len(images) != 240 or len({r['cxr_candidate_id'] for r in images}) != 240 or \
            {r['cxr_candidate_id'] for r in images} != set(expected) or \
            any((r['case_id'], r['cxr_sha256'], r['cxr_model_id']) != expected[r['cxr_candidate_id']] for r in images):
        raise ValueError('fixed_full_image_inventory_required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('cuda_slurm_required')
    from extract_cxr_labels_xrv import FrozenXRVRuntime
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        outcomes = []; torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            runtime = FrozenXRVRuntime(cache_dir=WEIGHT_DIR, weight_filename=WEIGHT_NAME,
                                       model_name='densenet121-res224-all')
            if runtime.model.training or any(p.requires_grad for p in runtime.model.parameters()):
                raise RuntimeError('xrv_not_frozen')
            for r in images:
                outcome = {k: r[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_sha256', 'cxr_model_id')}
                outcome.update(status='failed_without_replacement', model_calls=0,
                    exact_lung_opacity_score=None, infiltration_score=None)
                try:
                    path = require_inside(r['path'], PROTECTED_ROOT, must_exist=True)
                    if sha256_file(path) != r['cxr_sha256']:
                        raise ValueError('synthetic_image_hash_changed')
                    outcome['model_calls'] = 1; raw = runtime.predict(path)
                    exact, infiltration = raw.get('lung_opacity'), raw.get('infiltration')
                    if exact is None or infiltration is None:
                        raise ValueError('exact_heads_missing')
                    state(exact); state(infiltration)
                    if sha256_file(path) != r['cxr_sha256']:
                        raise ValueError('synthetic_image_changed_during_scoring')
                    outcome.update(status='scored', exact_lung_opacity_score=exact, infiltration_score=infiltration)
                except Exception as error:
                    outcome['error_type'] = type(error).__name__
                outcomes.append(outcome)
            torch.cuda.synchronize()
        results = {r['cxr_candidate_id']: r for r in outcomes}
        annotated, model_readouts = overlay(plan['original_rows'], plan['cached_records'], results)
        scored = [r for r in outcomes if r['status'] == 'scored']
        summary = {'schema_version': SCHEMA, 'status': 'completed_opacity_sidecar_unverified' if len(scored) == 240
            else 'incomplete_original_denominators_retained', **plan['inventory'],
            'planned_images': 240, 'scored_images': len(scored), 'failed_images': 240 - len(scored),
            'model_calls': sum(r['model_calls'] for r in outcomes), 'generation_calls': 0, 'training_calls': 0,
            'exact_state_counts_one_per_image': dict(Counter(state(r['exact_lung_opacity_score']) for r in outcomes)),
            'same_call_max_state_counts_one_per_image': dict(Counter(state(max(r['exact_lung_opacity_score'],
                r['infiltration_score'])) for r in scored)),
            'same_call_exact_vs_max_state_changes_images': sum(state(r['exact_lung_opacity_score']) !=
                state(max(r['exact_lung_opacity_score'], r['infiltration_score'])) for r in scored),
            'old_cache_vs_new_same_call_max_state_changes_images': sum(results[r['cxr_candidate_id']]['status'] == 'scored' and
                r['cached_legacy_max_opacity_state'] != state(max(results[r['cxr_candidate_id']]['exact_lung_opacity_score'],
                results[r['cxr_candidate_id']]['infiltration_score'])) for r in images_with_legacy(plan)),
            'three_edge_relations_candidate_rows': {edge: dict(Counter(r['opacity_' + edge + '_relation'] for r in annotated))
                for edge in ('ehr_cxr', 'ehr_report', 'cxr_report')},
            'secondary_cached_legacy_cxr_report_relations_candidate_rows': cached_relations(plan['cached_records']),
            'secondary_same_call_max_cxr_report_relations_candidate_rows': dict(Counter(
                relation(r['opacity_same_call_max_state_0_5'], r['opacity_cached_report_state']) for r in annotated)),
            'model_pair_readouts': model_readouts, 'policy': POLICY,
            'original_csv_cells_and_order_preserved': True, 'old_scores_and_choices_changed': False,
            'clinical_truth_available': False, 'benchmark_to_synthetic_domain_transport_validated': False,
            'report_chexbert_labels_independently_qualified': False,
            'real_or_synthetic_clinical_accuracy_established': False,
            'no_ehr_report_body_reads': True, 'no_raw_patient_inputs': True,
            'elapsed_seconds': round(time.monotonic() - started, 6),
            'peak_allocated_tensor_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
        write_private_json(temporary / 'image_scores.json', {'schema_version': SCHEMA + '-scores', 'records': outcomes})
        write_private_text(temporary / 'candidate_score_table.csv', csv_text(annotated))
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md', '# Exact opacity candidate diagnostic / 候选阴影诊断\n\n'
            'Raw operating-point scores, not calibrated probabilities. Same-image reports are correlated.\n'
            'Not pneumonia/EHR fidelity, clinical truth, error localization, repair or a new selection policy.\n'
            'Unknown/uncertain pairs remain not comparable; all original 960 CSV rows/cells retained.\n\n'
            + json.dumps(summary, indent=2, sort_keys=True) + '\n')
        verify_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'plan_manifest_sha256': args.plan_manifest_sha256, 'sources': pins,
            'artifacts': {n: sha256_file(temporary / n) for n in ('image_scores.json', 'candidate_score_table.csv',
                'summary.json', 'RESULTS_CN_EN.md')}, 'original_selection_changed': False,
            'primary_metric_eligible': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, summary


def images_with_legacy(plan):
    values = {}
    for r in plan['cached_records']:
        values[r['cxr_candidate_id']] = r
    return list(values.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    for name in ('output-root', 'run-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--plan-root'); parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--allow-synthetic-xrv', action='store_true')
    args = parser.parse_args(argv); os.umask(0o007)
    try:
        target, summary = prepare(args) if args.action == 'prepare' else evaluate(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__})); return 2
    print(json.dumps({'status': summary['status'], 'manifest_sha256': sha256_file(target / 'manifest.json'),
        'fixed_ehr_cases': summary['fixed_ehr_cases'], 'image_slots': summary['image_slots'],
        'candidate_rows': summary['candidate_rows'], 'model_calls': summary.get('model_calls', 0)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

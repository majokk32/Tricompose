#!/usr/bin/env python3
"""Lossless opacity evidence overlay; CPU metadata prepare, separately approved GPU scoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'real_validation'),
               str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
import score_cached_opacity_candidates as opacity
import ricord_biovil50 as frozen
from contracts import (PROTECTED_ROOT, WORKSPACE, commit_atomic_run, discard_atomic_run,
    new_atomic_run, require_inside, sha256_file, write_private_json, write_private_text)

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SOURCE = BASE / 'candidate_opacity_runs/opacity_pool240_12668204'
SOURCE_SHA = '778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94'
OLD_PLAN = BASE / 'candidate_opacity_plans/opacity_pool240_12666569_001'
OLD_PLAN_SHA = '650f129ce6b20e6e278bd937accfd348cef4e68328fcf793ab722caa55d7323a'
REFERENCE = BASE / 'real_validation/ricord_biovil_pilots/ricord_biovil50_12669671'
REFERENCE_SHA = 'd50e5594e43f7979f385f0e14745d66103714b16693e7656ddcf7812e85480ef'
REFERENCE_PLAN = BASE / 'real_validation/ricord_biovil_plans/opacity50_12666569_001'
REFERENCE_PLAN_SHA = '91560c8407d638979d665e3fc4717dff80e68289e58d9036ce1258a8f96687e1'
SCHEMA = 'tricompose-cached-opacity-biovil-evidence-v1'
PREFIX = 'opacity_biovil_'
PROTOCOL = ROOT.parent / 'docs/cached_opacity_biovil_protocol.md'
TESTS = ROOT / 'tests/test_cached_opacity_biovil.py'
POLICY = {'finding': 'lung_opacity', 'primary_reduction': frozen.reuse.MEAN,
    'preference_zero': 'unknown', 'positive_rule': 'mean_margin_gt_zero',
    'negative_rule': 'mean_margin_lt_zero', 'templates_fitted': False,
    'probability_semantics': False, 'independent_model_votes': False,
    'report_text_encoded': False, 'ehr_fact_enrichment': False,
    'clinical_localization': False, 'regeneration_authorized': False,
    'selector_enabled': False, 'primary_metric_eligible': False}


def preference(value):
    if value is None:
        return 'unknown'
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 2.00002:
        raise ValueError('finite_margin_required')
    return 'positive' if value > 0 else 'negative' if value < 0 else 'unknown'


def template_pattern(margins):
    states = [preference(margins[f]) for f in frozen.reuse.FAMILIES]
    if all(s == 'positive' for s in states):
        return 'all_positive_preference'
    if all(s == 'negative' for s in states):
        return 'all_negative_preference'
    return 'all_tied' if all(s == 'unknown' for s in states) else 'mixed_or_tied'


def joint_pattern(xrv, biovil, report):
    if any(s not in opacity.STATES for s in (xrv, biovil, report)):
        raise ValueError('four_state_evidence_required')
    if xrv not in opacity.EXPLICIT or biovil not in opacity.EXPLICIT:
        return 'image_evidence_unavailable'
    if xrv != biovil:
        return 'image_sources_disagree'
    if report not in opacity.EXPLICIT:
        return 'report_evidence_unavailable'
    return 'three_proxy_sources_agree' if report == xrv else 'report_proposal_opposes_two_image_sources'


def validate_inventory(rows, images, *, full=True):
    if not rows or len({r['triple_candidate_id'] for r in rows}) != len(rows) or \
            len({i['cxr_candidate_id'] for i in images}) != len(images):
        raise ValueError('unique_nonempty_inventory_required')
    image_index = {i['cxr_candidate_id']: i for i in images}
    cases, report_states = {}, {}
    for row in rows:
        image = image_index.get(row['cxr_candidate_id'])
        if image is None or any(row[k] != image[k] for k in ('case_id', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')):
            raise ValueError('same_fixed_ehr_image_lineage_required')
        value = row['ehr_sha256'], row['ehr_facts_sha256']
        if cases.setdefault(row['case_id'], value) != value:
            raise ValueError('fixed_ehr_required')
        if row['opacity_cached_ehr_state'] != 'unknown' or \
                any(row[k] not in opacity.STATES for k in ('opacity_cached_report_state', 'opacity_exact_state_0_5')) or \
                row['opacity_clinical_accuracy'] != '' or row['opacity_confirmed_faulty_modality'] != '' or \
                row['opacity_selector_used'] != 'False':
            raise ValueError('original_unqualified_states_required')
        key = row['report_sha256']
        if report_states.setdefault(key, row['opacity_cached_report_state']) != row['opacity_cached_report_state']:
            raise ValueError('shared_report_artifact_state_changed')
    if set(image_index) != {r['cxr_candidate_id'] for r in rows}:
        raise ValueError('all_existing_images_required')
    if full and (len(rows) != 960 or len(images) != 240 or len(cases) != 80 or
            any(sum(r['cxr_candidate_id'] == i for r in rows) != 4 for i in image_index) or
            any(sum(r['case_id'] == c for r in rows) != 12 for c in cases)):
        raise ValueError('complete_80_3_4_grid_required')
    return image_index


def score_values(outcome):
    if outcome['status'] == 'failed_without_replacement' and outcome['score_pairs'] is None:
        return None
    if outcome['status'] != 'scored' or outcome['score_pairs'] is None:
        raise ValueError('explicit_scored_or_failed_outcome_required')
    return frozen.reuse.margins(outcome)


def overlay(rows, images, outcomes):
    index = validate_inventory(rows, images, full=False)
    if len(outcomes) != len(index) or {r['cxr_candidate_id'] for r in outcomes} != set(index):
        raise ValueError('all_unique_image_outcomes_required')
    scored = {}
    for outcome in outcomes:
        image = index[outcome['cxr_candidate_id']]
        if any(outcome[k] != image[k] for k in ('case_id', 'cxr_sha256', 'cxr_model_id')):
            raise ValueError('scored_image_lineage_changed')
        values = score_values(outcome)
        scored[outcome['cxr_candidate_id']] = outcome, values
    result = []
    for row in rows:
        if any(k.startswith(PREFIX) for k in row):
            raise ValueError('already_annotated_source_refused')
        outcome, margins = scored[row['cxr_candidate_id']]
        added = {}
        for family in frozen.reuse.FAMILIES:
            pair = outcome['score_pairs'][family] if margins is not None else None
            for polarity in ('positive', 'negative'):
                added[PREFIX + family + '_' + polarity + '_cosine'] = pair[polarity + '_cosine'] if pair else None
            added[PREFIX + family + '_margin'] = margins[family] if margins else None
        mean = margins[frozen.reuse.MEAN] if margins else None
        state = preference(mean)
        values = [margins[f] for f in frozen.reuse.FAMILIES] if margins else []
        added.update({PREFIX + 'mean_margin': mean, PREFIX + 'preference_state': state,
            PREFIX + 'scoring_status': outcome['status'],
            PREFIX + 'template_pattern': template_pattern(margins) if margins else 'unavailable',
            PREFIX + 'template_min_margin': min(values) if values else None,
            PREFIX + 'template_max_margin': max(values) if values else None,
            PREFIX + 'report_proxy_relation': opacity.relation(state, row['opacity_cached_report_state']),
            PREFIX + 'exact_xrv_proxy_relation': opacity.relation(state, row['opacity_exact_state_0_5']),
            PREFIX + 'joint_proxy_pattern': joint_pattern(row['opacity_exact_state_0_5'], state, row['opacity_cached_report_state']),
            PREFIX + 'clinical_accuracy': None, PREFIX + 'selector_used': False})
        result.append({**row, **added})
    return result


def aggregate(rows, outcomes):
    image_states = [preference(score_values(r)[frozen.reuse.MEAN]) if score_values(r) is not None else 'unknown' for r in outcomes]
    groups = defaultdict(list)
    for row in rows:
        groups[row['report_model_id']].append(row)
    def relations(subset):
        counts = Counter(r[PREFIX + 'report_proxy_relation'] for r in subset)
        comparable = counts['proxy_support'] + counts['proxy_opposition']
        return {'candidate_rows': len(subset), 'proxy_support': counts['proxy_support'],
            'proxy_opposition': counts['proxy_opposition'], 'not_comparable': counts['not_comparable'],
            'comparable_coverage': comparable/len(subset),
            'agreement_over_comparable': counts['proxy_support']/comparable if comparable else None}
    return {'fixed_ehr_cases': len({r['case_id'] for r in rows}), 'image_slots': len(outcomes),
        'candidate_rows': len(rows), 'scored_images': sum(r['status'] == 'scored' for r in outcomes),
        'failed_images': sum(r['status'] != 'scored' for r in outcomes),
        'biovil_preference_states_one_per_image': dict(Counter(image_states)),
        'biovil_report_proxy_relations': relations(rows),
        'joint_proxy_patterns_candidate_rows': dict(Counter(r[PREFIX + 'joint_proxy_pattern'] for r in rows)),
        'by_report_expert': {name: relations(values) for name, values in sorted(groups.items())},
        'ehr_opacity_clinical_edges_available': False, 'clinical_accuracy': None}


def source_inputs():
    pins = {str(p): sha256_file(p) for p in (Path(__file__), TESTS, PROTOCOL,
        Path(frozen.__file__), Path(opacity.__file__))}
    old = opacity.metadata(OLD_PLAN / 'manifest.json', pins, OLD_PLAN_SHA)
    opacity.verify_pins(old['sources']); pins.update(old['sources'])
    previous = opacity.metadata(OLD_PLAN / 'plan.json', pins, old['artifacts']['plan.json'])
    if previous['modality_source'] != 'fully_synthetic' or previous['old_scores_and_choices_changed'] is not False:
        raise ValueError('fully_synthetic_fixed_source_required')
    manifest = opacity.metadata(SOURCE / 'manifest.json', pins, SOURCE_SHA)
    opacity.verify_pins(manifest['sources']); pins.update(manifest['sources'])
    if manifest['plan_manifest_sha256'] != OLD_PLAN_SHA or manifest['original_selection_changed'] is not False:
        raise ValueError('unchanged_opacity_sidecar_required')
    rows = opacity.metadata(SOURCE / 'candidate_score_table.csv', pins, manifest['artifacts']['candidate_score_table.csv'])
    xrv = opacity.metadata(SOURCE / 'image_scores.json', pins, manifest['artifacts']['image_scores.json'])
    original, _ = opacity.overlay(previous['original_rows'], previous['cached_records'], {r['cxr_candidate_id']: r for r in xrv['records']})
    if rows != list(csv.DictReader(io.StringIO(opacity.csv_text(original)))):
        raise ValueError('all_66_original_columns_and_cells_required')
    images = previous['images']; validate_inventory(rows, images)
    for image in images:
        path = require_inside(image['path'], PROTECTED_ROOT, must_exist=True)
        if sha256_file(path) != image['cxr_sha256']:
            raise ValueError('immutable_synthetic_png_changed')
        pins[str(path)] = image['cxr_sha256']
    bm = opacity.metadata(REFERENCE / 'manifest.json', pins, REFERENCE_SHA)
    if bm['schema_version'] != frozen.SCHEMA + '-manifest' or bm['plan_manifest_sha256'] != REFERENCE_PLAN_SHA:
        raise ValueError('completed_bound_biovil_benchmark_required')
    for path, digest in bm['sources'].items():
        pins[str(WORKSPACE / path)] = digest
    summary = opacity.metadata(REFERENCE / 'summary.json', pins, bm['artifacts']['summary.json'])
    bp = opacity.metadata(REFERENCE_PLAN / 'manifest.json', pins, REFERENCE_PLAN_SHA)
    bplan = opacity.metadata(REFERENCE_PLAN / 'plan.json', pins, bp['artifacts']['plan.json'])
    if summary['scored_cases'] != 50 or summary['failed_cases'] != 0 or summary['primary_metric_eligible'] is not False or \
            bplan['authored_probes'] != frozen.catalog() or bplan['primary_reduction'] != POLICY['primary_reduction'] or \
            bm['runtime'] != frozen.runtime_metadata():
        raise ValueError('same_completed_frozen_probe_contract_required')
    opacity.verify_pins(pins)
    return rows, images, bm['runtime'], pins


def prepare(args):
    opacity.guard(); rows, images, runtime, pins = source_inputs()
    plan = {'schema_version': SCHEMA + '-plan', 'source_manifest_sha256': SOURCE_SHA,
        'benchmark_manifest_sha256': REFERENCE_SHA, 'rows': rows, 'images': images,
        'runtime': runtime, 'policy': POLICY, 'authored_probes': frozen.catalog(),
        'preprocessing': frozen.PREPROCESSING, 'replay_image_indices': [0, 1],
        'expected_image_encoder_calls': 242, 'expected_text_encoder_batches': 1,
        'expected_text_inputs': 8, 'modality_source': 'fully_synthetic',
        'pending_image_encoder_calls': 242, 'new_model_calls': 0, 'pixels_opened': False,
        'old_scores_and_choices_changed': False, 'post_hoc_development': True}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'plan.json', plan)
        opacity.verify_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-plan-manifest',
            'sources': pins, 'artifacts': {'plan.json': sha256_file(temporary / 'plan.json')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, {'status': 'sealed_metadata_only_new_scores_pending', 'new_model_calls': 0}


def evaluate(args):
    opacity.guard(gpu=True, approved=args.allow_synthetic_biovil)
    target = require_inside(Path(args.output_root) / args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('output_run_exists')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('offline_frozen_scoring_required')
    pins = {}
    plan_root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    manifest = opacity.metadata(plan_root / 'manifest.json', pins, args.plan_manifest_sha256)
    opacity.verify_pins(manifest['sources']); pins.update(manifest['sources'])
    plan = opacity.metadata(plan_root / 'plan.json', pins, manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA + '-plan' or plan['policy'] != POLICY or \
            plan['authored_probes'] != frozen.catalog() or plan['runtime'] != frozen.runtime_metadata() or \
            plan['preprocessing'] != frozen.PREPROCESSING or plan['replay_image_indices'] != [0, 1] or \
            plan['modality_source'] != 'fully_synthetic' or plan['old_scores_and_choices_changed'] is not False:
        raise ValueError('frozen_source_model_probe_contract_required')
    validate_inventory(plan['rows'], plan['images'])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('allocated_cuda_required')
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started, outcomes, replay = time.monotonic(), [], []
    image_calls = text_batches = 0
    try:
        torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
        prompts = [p[k] for p in frozen.catalog() for k in ('positive_text', 'negative_text')]
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            image_engine, text_engine = frozen.reuse._load_runtime(frozen.reuse.MODEL, torch.device('cuda:0'))
            if any(e.model.training or any(p.requires_grad for p in e.model.parameters()) for e in (image_engine, text_engine)):
                raise RuntimeError('model_not_frozen')
            with torch.inference_mode():
                text_batches += 1
                embeddings = text_engine.get_embeddings_from_prompt(prompts + [prompts[0]] * 2, normalize=True, verbose=False)
                if tuple(embeddings.shape) != (8, 128) or not torch.isfinite(embeddings).all() or not torch.allclose(
                        embeddings.norm(dim=1), torch.ones(8, device=embeddings.device), atol=1e-4, rtol=1e-4):
                    raise ValueError('normalized_text_embedding_required')
                duplicate_delta = float((embeddings[-2] - embeddings[-1]).abs().max().item())
                for index, image in enumerate(plan['images']):
                    outcome = {k: image[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_sha256', 'cxr_model_id')}
                    outcome.update(status='failed_without_replacement', score_pairs=None, image_calls=0)
                    try:
                        path = require_inside(image['path'], PROTECTED_ROOT, must_exist=True)
                        if sha256_file(path) != image['cxr_sha256']:
                            raise ValueError('fixed_image_changed')
                        image_calls += 1; outcome['image_calls'] += 1
                        vector = image_engine.get_projected_global_embedding(path)
                        if tuple(vector.shape) != (128,) or not torch.isfinite(vector).all() or not torch.allclose(
                                vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4):
                            raise ValueError('normalized_image_embedding_required')
                        values = (embeddings[:6] @ vector).detach().float().cpu().tolist()
                        pairs = {family: {'positive_cosine': values[2*i], 'negative_cosine': values[2*i+1]}
                                 for i, family in enumerate(frozen.reuse.FAMILIES)}
                        frozen.reuse.margins({'score_pairs': pairs})
                        outcome.update(status='scored', score_pairs=pairs)
                        if index in plan['replay_image_indices']:
                            check = {'cxr_candidate_id': image['cxr_candidate_id'], 'status': 'failed_replay_without_retry'}
                            try:
                                image_calls += 1; outcome['image_calls'] += 1
                                repeated = image_engine.get_projected_global_embedding(path)
                                delta = float(((embeddings[:6] @ repeated) - (embeddings[:6] @ vector)).abs().max().item())
                                if not math.isfinite(delta):
                                    raise ValueError('finite_replay_required')
                                check.update(status='replayed', max_abs_score_delta=delta)
                            except Exception as error:
                                check['error_type'] = type(error).__name__
                            replay.append(check)
                    except Exception as error:
                        outcome['error_type'] = type(error).__name__
                    outcomes.append(outcome)
            torch.cuda.synchronize()
        rows = overlay(plan['rows'], plan['images'], outcomes)
        summary = {'schema_version': SCHEMA, **aggregate(rows, outcomes),
            'status': 'completed_proxy_evidence_overlay_unqualified' if all(r['status'] == 'scored' for r in outcomes)
                      else 'incomplete_all_candidates_retained', 'policy': POLICY,
            'old_columns_preserved': len(plan['rows'][0]), 'new_columns_added': len(rows[0])-len(plan['rows'][0]),
            'old_scores_and_choices_changed': False, 'source_ehr_report_text_read': False,
            'new_selector_calls': 0, 'generation_calls': 0, 'training_calls': 0,
            'image_encoder_calls': image_calls, 'text_encoder_batches': text_batches, 'text_inputs': 8,
            'duplicate_text_embedding_delta': duplicate_delta, 'first_two_image_replay': replay,
            'clinical_localization_established': False, 'benchmark_transport_validated': False,
            'report_proposals_independently_qualified': False, 'primary_metric_eligible': False,
            'elapsed_seconds_before_serialization': round(time.monotonic()-started, 6),
            'peak_allocated_vram_gib_including_load': round(torch.cuda.max_memory_allocated()/1024**3, 3)}
        opacity.verify_pins(pins)
        if frozen.runtime_metadata() != plan['runtime']:
            raise ValueError('runtime_changed_during_scoring')
        write_private_json(temporary / 'image_scores.json', {'schema_version': SCHEMA, 'records': outcomes})
        write_private_text(temporary / 'candidate_evidence_table.csv', opacity.csv_text(rows))
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md', '# Opacity evidence overlay / 候选证据表\n\n'
            'Diagnostic only, not a total triple score, fault attribution or a new selected output.\n'
            'Same-image scoring proposals are dependent. Template variations are not clinical uncertainty probabilities.\n'
            'All existing columns/rows stay; unknown EHR and report states are not negatives.\n\n'
            + json.dumps(summary, indent=2, sort_keys=True) + '\n')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'sources': pins, 'runtime': plan['runtime'], 'plan_manifest_sha256': args.plan_manifest_sha256,
            'source_manifest_sha256': SOURCE_SHA, 'benchmark_manifest_sha256': REFERENCE_SHA,
            'artifacts': {name: sha256_file(temporary / name) for name in (
                'image_scores.json', 'candidate_evidence_table.csv', 'summary.json', 'RESULTS_CN_EN.md')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, {k: summary[k] for k in ('status', 'elapsed_seconds_before_serialization', 'peak_allocated_vram_gib_including_load')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('--plan-root')
    parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--allow-synthetic-biovil', action='store_true')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        target, status = prepare(args) if args.action == 'prepare' else evaluate(args)
        status['manifest_sha256'] = sha256_file(target / 'manifest.json')
        print(json.dumps(status, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

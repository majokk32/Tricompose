#!/usr/bin/env python3
"""Prepare numeric lineage; separately approved GPU reads synthetic images only.

Six named findings, not whole-prompt retrieval or opacity used as effusion.
No EHR, report or generation-prompt bodies; no benchmark intervention key.
Readouts are diagnostic evidence, never calibrated clinical truth or actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'src'),
               str(ROOT / 'real_validation'),
               str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
import score_prompt_image_conditioning_v1 as lineage
import benchmark_probe_repair_v1 as cached
from contracts import (PROTECTED_ROOT, sha256_file, require_inside,
    new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

SCHEMA = 'tricompose-finding-matched-biovil-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
PROTOCOL = ROOT / 'configs/finding_matched_biovil_v1.json'
TESTS = ROOT / 'tests/test_finding_matched_biovil_v1.py'
FINDINGS = ('edema', 'pleural_effusion', 'cardiomegaly', 'pneumonia',
            'pneumothorax', 'support_devices')
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
STATES = frozenset(('positive', 'negative', 'uncertain', 'unknown'))
EXPLICIT = frozenset(('positive', 'negative'))
IMAGE_FIELDS = ('case_id', 'cxr_candidate_id', 'cxr_model_id',
                'ehr_sha256', 'ehr_facts_sha256', 'cxr_path', 'cxr_sha256')
HASH_FIELDS = ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')
FACT_FIELDS = ('case_id', 'triple_candidate_id', 'cxr_candidate_id',
               'report_candidate_id', 'report_model_id', *HASH_FIELDS)
FALSE_FLAGS = ('training_allowed', 'threshold_fitting', 'template_fitting',
    'unknown_is_negative', 'weak_ehr_context_promoted', 'external_api_allowed',
    'probability_semantics', 'independent_template_votes',
    'independent_clinical_reference_available', 'clinical_qualified',
    'regeneration_authorized', 'selection_changed', 'untouched_final_test')


def protocol():
    value = json.loads(PROTOCOL.read_text())
    if (value['schema_version'] != SCHEMA + '-protocol'
            or value['findings'] != list(FINDINGS)
            or value['template_families'] != list(FAMILIES)
            or (value['cases'], value['image_slots'], value['candidate_triples']) != (80, 240, 960)
            or (value['text_batch_size'], value['text_replays'], value['image_replays'], value['seed']) != (16, 2, 2, 0)
            or value['support_devices_xrv_head'] is not None
            or value['preference_rule'] != 'positive_mean_margin_positive_negative_mean_margin_negative_exact_zero_unknown'
            or value['template_reduction'] != 'mean_all_three_predeclared'
            or value['xrv_role'] != 'unchanged_cached_operating_point_states_not_calibrated_probabilities'
            or value['report_role'] != 'unchanged_cached_chexbert_proposals_current_assertion_scope_unqualified'
            or value['post_hoc_development'] is not True
            or any(value[k] is not False for k in FALSE_FLAGS)):
        raise ValueError('fixed_scope_unqualified_protocol_required')
    return value


def catalog():
    # Authored generic probes, not any patient's report or EHR-derived prompt.
    result = []
    for finding in FINDINGS:
        name = finding.replace('_', ' ')
        pairs = (
            (f'The chest X-ray shows {name}.', f'The chest X-ray shows no {name}.'),
            (f'There is evidence of {name}.', f'There is no evidence of {name}.'),
            (f'{name.capitalize()} is present.', f'{name.capitalize()} is absent.'),
        )
        for family, (positive, negative) in zip(FAMILIES, pairs, strict=True):
            result.append({'finding': finding, 'family': family,
                           'positive_text': positive, 'negative_text': negative})
    return result


def project_facts(rows, bank):
    candidates = {c['score_record']['triple_candidate_id']: c
                  for grid in bank.values() for c in grid.values()}
    if (len(candidates) != sum(len(g) for g in bank.values())
            or len(rows) != len(candidates)
            or len({r['triple_candidate_id'] for r in rows}) != len(rows)
            or set(candidates) != {r['triple_candidate_id'] for r in rows}):
        raise ValueError('exact_unique_original_candidate_inventory_required')
    result = []
    for row in rows:
        observation = cached.policy_module.snapshot(candidates[row['triple_candidate_id']])
        if (observation['case_id'] != row['case_id']
                or any(observation['lineage'][k] != row[k] for k in HASH_FIELDS)
                or any(observation['lineage'][k] != row[k]
                       for k in ('cxr_candidate_id', 'report_candidate_id', 'report_model_id', 'cxr_model_id'))
                or str(observation['lineage']['cxr_seed']) != str(row['cxr_seed'])):
            raise ValueError('exact_original_candidate_lineage_required')
        for finding in FINDINGS:
            state = {name: observation['states'][name][finding]
                     for name in ('ehr', 'xrv', 'chexbert')}
            sources = observation['ehr_sources'][finding]
            if state['ehr'] != 'unknown' and not sources:
                raise ValueError('asserted_ehr_requires_cached_source_categories')
            if finding == 'support_devices' and state['xrv'] != 'unknown':
                raise ValueError('unavailable_device_head_cannot_be_filled')
            result.append({**{k: row[k] for k in FACT_FIELDS}, 'finding': finding,
                'states': state, 'ehr_source_categories': list(sources),
                'report_assertion_scope_qualified': False})
    return result


def validate_inventory(rows, images, facts, *, full=True):
    if not rows or len({r['triple_candidate_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty_unique_original_rows_required')
    expected_images = [{k: r[k] for k in IMAGE_FIELDS}
                       for r in lineage.images_from_index(rows, full=full)]
    if images != expected_images:
        raise ValueError('exact_original_image_projection_required')
    by_candidate = {r['triple_candidate_id']: r for r in rows}
    seen, shared_ehr, shared_image, shared_report = set(), {}, {}, {}
    for fact in facts:
        key = fact['triple_candidate_id'], fact['finding']
        original = by_candidate.get(key[0])
        if (original is None or key in seen or key[1] not in FINDINGS
                or set(fact) != set(FACT_FIELDS) | {'finding', 'states', 'ehr_source_categories',
                                                   'report_assertion_scope_qualified'}
                or any(fact[k] != original[k] for k in FACT_FIELDS)
                or set(fact['states']) != {'ehr', 'xrv', 'chexbert'}
                or any(s not in STATES for s in fact['states'].values())
                or fact['report_assertion_scope_qualified'] is not False
                or not isinstance(fact['ehr_source_categories'], list)
                or len(set(fact['ehr_source_categories'])) != len(fact['ehr_source_categories'])
                or not set(fact['ehr_source_categories']) <= {'diagnosis', 'medication', 'lab', 'vital', 'other'}):
            raise ValueError('exact_named_fact_projection_required')
        if fact['states']['ehr'] != 'unknown' and not fact['ehr_source_categories']:
            raise ValueError('asserted_ehr_requires_cached_source_categories')
        if key[1] == 'support_devices' and fact['states']['xrv'] != 'unknown':
            raise ValueError('unavailable_device_head_cannot_be_filled')
        seen.add(key)
        for index, identity, value in (
                (shared_ehr, (fact['case_id'], key[1]), (fact['ehr_sha256'], fact['ehr_facts_sha256'],
                    fact['states']['ehr'], fact['ehr_source_categories'])),
                (shared_image, (fact['cxr_sha256'], key[1]), fact['states']['xrv']),
                (shared_report, (fact['report_sha256'], key[1]), fact['states']['chexbert'])):
            if index.setdefault(identity, value) != value:
                raise ValueError('shared_artifact_or_fixed_ehr_changed')
    if seen != {(cid, finding) for cid in by_candidate for finding in FINDINGS}:
        raise ValueError('complete_six_finding_inventory_required')


def preference(value):
    if value is None:
        return 'unknown'
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 2.00002:
        raise ValueError('finite_cosine_margin_required')
    return 'positive' if value > 0 else 'negative' if value < 0 else 'unknown'


def score_readout(pairs):
    if pairs is None:
        return {'mean_margin': None, 'min_margin': None, 'max_margin': None,
                'preference_state': 'unknown', 'template_pattern': 'unavailable'}
    if set(pairs) != set(FAMILIES):
        raise ValueError('all_three_template_families_required')
    margins = []
    for family in FAMILIES:
        pair = pairs[family]
        if (set(pair) != {'positive_cosine', 'negative_cosine'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1.00001
                       for v in pair.values())):
            raise ValueError('finite_named_cosine_pair_required')
        margins.append(pair['positive_cosine'] - pair['negative_cosine'])
    mean = sum(margins) / len(margins)
    states = {preference(v) for v in margins}
    pattern = ('all_positive_preference' if states == {'positive'} else
               'all_negative_preference' if states == {'negative'} else
               'all_tied' if states == {'unknown'} else 'mixed_or_tied')
    return {'mean_margin': mean, 'min_margin': min(margins), 'max_margin': max(margins),
            'preference_state': preference(mean), 'template_pattern': pattern}


def relation(left, right):
    if left not in STATES or right not in STATES:
        raise ValueError('four_state_evidence_required')
    if left not in EXPLICIT or right not in EXPLICIT:
        return 'not_comparable'
    return 'proxy_support' if left == right else 'proxy_opposition'


def build_tables(rows, images, facts, outcomes, *, full=True):
    validate_inventory(rows, images, facts, full=full)
    by_image = {r['cxr_candidate_id']: r for r in images}
    scored, shared_scores, image_rows = {}, {}, []
    for outcome in outcomes:
        image = by_image.get(outcome['cxr_candidate_id'])
        if (image is None or outcome['cxr_candidate_id'] in scored
                or set(outcome) != set(IMAGE_FIELDS) | {'status', 'score_pairs'}
                or any(outcome[k] != image[k] for k in IMAGE_FIELDS)
                or outcome['status'] not in ('scored', 'failed_without_replacement')
                or (outcome['status'] == 'scored') != (outcome['score_pairs'] is not None)):
            raise ValueError('unique_exact_image_scoring_outcome_required')
        pairs = outcome['score_pairs']
        if pairs is not None and set(pairs) != set(FINDINGS):
            raise ValueError('all_six_scored_findings_required')
        signature = (outcome['status'], pairs)
        if shared_scores.setdefault(image['cxr_sha256'], signature) != signature:
            raise ValueError('same_image_hash_requires_same_scores')
        scored[outcome['cxr_candidate_id']] = {}
        for finding in FINDINGS:
            readout = score_readout(pairs[finding] if pairs is not None else None)
            scored[outcome['cxr_candidate_id']][finding] = readout
            image_rows.append({**image, 'finding': finding, 'scoring_status': outcome['status'],
                               **readout, 'clinical_accuracy': None})
    if set(scored) != set(by_image):
        raise ValueError('failed_or_scored_outcome_for_every_image_required')
    fact_rows = []
    for fact in facts:
        readout = scored[fact['cxr_candidate_id']][fact['finding']]
        state = fact['states']
        bv = readout['preference_state']
        image_agreement = relation(state['xrv'], bv)
        stable = readout['template_pattern'] in ('all_positive_preference', 'all_negative_preference')
        fact_rows.append({**{k: fact[k] for k in FACT_FIELDS}, 'finding': fact['finding'],
            'ehr_state': state['ehr'], 'ehr_source_categories': ','.join(fact['ehr_source_categories']),
            'cached_xrv_state': state['xrv'], 'cached_report_state': state['chexbert'],
            'xrv_head_available': fact['finding'] != 'support_devices',
            **readout, 'image_reader_relation': image_agreement,
            'mean_agreement_image_state': bv if image_agreement == 'proxy_support' else 'unknown',
            'stable_agreement_image_state': bv if image_agreement == 'proxy_support' and stable else 'unknown',
            'ehr_biovil_relation': relation(state['ehr'], bv),
            'report_biovil_relation': relation(state['chexbert'], bv),
            'ehr_cached_xrv_relation': relation(state['ehr'], state['xrv']),
            'ehr_cached_report_relation': relation(state['ehr'], state['chexbert']),
            'report_cached_xrv_relation': relation(state['chexbert'], state['xrv']),
            'report_assertion_scope_qualified': False, 'clinical_conflict_verified': None,
            'confirmed_faulty_modality': None, 'regeneration_authorized': False})
    per_finding = []
    for finding in FINDINGS:
        subset = [r for r in fact_rows if r['finding'] == finding]
        ehr_states = {r['case_id']: r['ehr_state'] for r in subset}
        image_states = {r['cxr_candidate_id']: r for r in subset}
        per_finding.append({'finding': finding, 'fixed_ehr_cases': len(ehr_states),
            'ehr_states_one_per_case': dict(Counter(ehr_states.values())),
            'image_slots': len(image_states), 'candidate_triples': len(subset),
            'image_reader_relations_one_per_image': dict(Counter(r['image_reader_relation'] for r in image_states.values())),
            'report_biovil_relations_candidate_rows': dict(Counter(r['report_biovil_relation'] for r in subset)),
            'ehr_biovil_relations_one_per_image': dict(Counter(r['ehr_biovil_relation'] for r in image_states.values())),
            'report_explicit_proposals_candidate_rows': sum(r['cached_report_state'] in EXPLICIT for r in subset),
            'joint_three_modality_comparable_candidate_rows': sum(r['ehr_state'] in EXPLICIT
                and r['cached_report_state'] in EXPLICIT and r['mean_agreement_image_state'] in EXPLICIT for r in subset),
            'independent_clinical_reference_available': False, 'clinical_accuracy': None})
    summary = {'schema_version': SCHEMA, 'fixed_ehr_cases': len({r['case_id'] for r in rows}),
        'image_slots': len(images), 'candidate_triples': len(rows),
        'scored_image_slots': sum(r['status'] == 'scored' for r in outcomes),
        'failed_image_slots': sum(r['status'] != 'scored' for r in outcomes),
        'image_finding_rows': len(image_rows), 'candidate_finding_rows': len(fact_rows),
        'by_finding': per_finding, 'new_training_calls': 0, 'new_generation_calls': 0,
        'thresholds_fitted': False, 'templates_fitted': False,
        'unknown_states_changed': 0, 'original_selection_changed': False,
        'clinical_qualified': False, 'clinical_fault_localization_accuracy': None,
        'clinical_repair_success': None, 'regeneration_authorized': False,
        'same_image_readers_and_templates_are_independent_votes': False,
        'interpretation': 'Named-finding numeric alignment only. Cached report scope and independent image truth are unqualified. Missing device XRV head and unknown EHR remain unavailable, not negatives. No proxy opposition confirms a faulty modality.'}
    return image_rows, fact_rows, summary


def cpu_plan_summary(rows, images, facts):
    """Useful readiness counts, not synthetic body summaries or new scores."""
    validate_inventory(rows, images, facts)
    per_finding = []
    for finding in FINDINGS:
        subset = [r for r in facts if r['finding'] == finding]
        cases = {r['case_id']: r['states']['ehr'] for r in subset}
        image_states = {r['cxr_candidate_id']: r['states']['xrv'] for r in subset}
        per_finding.append({'finding': finding,
            'ehr_states_one_per_case': dict(Counter(cases.values())),
            'cached_image_states_one_per_image': dict(Counter(image_states.values())),
            'cached_report_states_candidate_rows': dict(Counter(r['states']['chexbert'] for r in subset)),
            'xrv_head_available': finding != 'support_devices'})
    return {'schema_version': SCHEMA + '-readiness',
        'fixed_ehr_cases': len({r['case_id'] for r in rows}),
        'image_slots': len(images), 'candidate_triples': len(rows),
        'unique_images': len({r['cxr_sha256'] for r in images}),
        'planned_image_finding_rows': len(images)*len(FINDINGS),
        'planned_candidate_finding_rows': len(facts), 'by_finding': per_finding,
        'new_model_calls': 0, 'scoring_status': 'pending_separate_gpu_approval',
        'bodies_pixels_or_weights_read': False, 'selection_changed': False,
        'clinical_qualified': False, 'regeneration_authorized': False}


def source_pins():
    files = [Path(__file__), PROTOCOL, TESTS, Path(cached.__file__),
             Path(cached.policy_module.__file__)]
    # Reused adapters/runtime imports are authenticated without opening weights.
    files.extend(sorted((ROOT / 'src/tricompose_v12').glob('*.py')))
    return {str(p): sha256_file(p) for p in files}


def prepare(args):
    cached.cpu_guard()
    rules = protocol()
    rows, original_images, runtime, pins = lineage.source_inputs()
    pins.update(source_pins())
    bank, _ = cached.load_primary(pins)
    facts = project_facts(rows, bank)
    images = [{k: r[k] for k in IMAGE_FIELDS} for r in original_images]
    validate_inventory(rows, images, facts)
    unique = len({r['cxr_sha256'] for r in images})
    plan = {'schema_version': SCHEMA + '-plan', 'protocol': rules,
        'rows': rows, 'images': images, 'facts': facts, 'runtime': runtime,
        'authored_probes': catalog(), 'modality_source': 'fully_synthetic',
        'expected_image_encoder_calls': unique + rules['image_replays'],
        'expected_text_encoder_inputs': len(catalog()) * 2 + rules['text_replays'],
        'bodies_pixels_or_weights_read': False, 'new_model_calls': 0}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'plan.json', plan)
        write_private_json(temporary / 'readiness.json', cpu_plan_summary(rows, images, facts))
        lineage.reuse.verify_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-plan-manifest',
            'sources': pins, 'artifacts': {name: sha256_file(temporary / name)
                for name in ('plan.json', 'readiness.json')},
            'new_model_calls': 0, 'clinical_qualified': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def evaluate(args):
    lineage.reuse.guard(gpu=True, approved=args.allow_synthetic_finding_images)
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise RuntimeError('offline_frozen_runtime_required')
    target = require_inside(Path(args.output_root) / args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('output_run_exists')
    plan_root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    pins = {}
    manifest = lineage.reuse.metadata(plan_root / 'manifest.json', pins, args.plan_manifest_sha256)
    if manifest['schema_version'] != SCHEMA + '-plan-manifest':
        raise ValueError('sealed_finding_plan_required')
    lineage.reuse.verify_pins(manifest['sources']); pins.update(manifest['sources'])
    plan = lineage.reuse.metadata(plan_root / 'plan.json', pins, manifest['artifacts']['plan.json'], cap=16*1024**2)
    if (plan['schema_version'] != SCHEMA + '-plan' or plan['protocol'] != protocol()
            or plan['authored_probes'] != catalog() or plan['modality_source'] != 'fully_synthetic'
            or plan['bodies_pixels_or_weights_read'] is not False or plan['new_model_calls'] != 0
            or plan['expected_image_encoder_calls'] != len({r['cxr_sha256'] for r in plan['images']}) + 2
            or plan['expected_text_encoder_inputs'] != 38):
        raise ValueError('fixed_synthetic_finding_plan_required')
    validate_inventory(plan['rows'], plan['images'], plan['facts'])
    import ricord_biovil50 as frozen
    if frozen.runtime_metadata() != plan['runtime']:
        raise ValueError('unchanged_deployed_frozen_runtime_required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('allocated_cuda_required')
    paths = {}
    for image in plan['images']:
        path = require_inside(image['cxr_path'], PROTECTED_ROOT, must_exist=True)
        if (path.suffix != '.png' or not path.is_file() or not 0 < path.stat().st_size <= 16*1024**2
                or sha256_file(path) != image['cxr_sha256']):
            raise ValueError('bound_synthetic_png_changed')
        paths.setdefault(image['cxr_sha256'], path)
        pins[str(path)] = image['cxr_sha256']
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    image_calls = text_batches = text_inputs = 0
    vectors, failed, replays = {}, set(), []
    try:
        torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
        prompts = [p[k] for p in catalog() for k in ('positive_text', 'negative_text')]
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            image_engine, text_engine = frozen.reuse._load_runtime(frozen.reuse.MODEL, torch.device('cuda:0'))
            if any(e.model.training or any(p.requires_grad for p in e.model.parameters()) for e in (image_engine, text_engine)):
                raise RuntimeError('frozen_eval_mode_required')
            with torch.inference_mode():
                text_vectors = []
                inputs = prompts + [prompts[0]] * 2
                for start in range(0, len(inputs), plan['protocol']['text_batch_size']):
                    batch = inputs[start:start + plan['protocol']['text_batch_size']]
                    text_batches += 1; text_inputs += len(batch)
                    value = text_engine.get_embeddings_from_prompt(batch, normalize=True, verbose=False)
                    if (tuple(value.shape) != (len(batch), 128) or not torch.isfinite(value).all()
                            or not torch.allclose(value.norm(dim=1), torch.ones(len(batch), device=value.device), atol=1e-4, rtol=1e-4)):
                        raise ValueError('normalized_128d_text_embeddings_required')
                    text_vectors.append(value)
                embeddings = torch.cat(text_vectors)
                text_replay = [float((embeddings[0] - embeddings[i]).abs().max().item()) for i in (36, 37)]
                geometry = {p['finding'] + ':' + p['family']: float((embeddings[2*i] @ embeddings[2*i+1]).item())
                            for i, p in enumerate(catalog())}
                for digest, path in sorted(paths.items()):
                    try:
                        image_calls += 1
                        vector = image_engine.get_projected_global_embedding(path)
                        if (tuple(vector.shape) != (128,) or not torch.isfinite(vector).all()
                                or not torch.allclose(vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4)):
                            raise ValueError('normalized_128d_image_embedding_required')
                        vectors[digest] = vector
                    except Exception:
                        failed.add(digest)  # Keep every slot; never substitute an image.
                for digest in sorted(paths)[:2]:
                    replay = {'cxr_sha256': digest, 'status': 'failed_without_replacement', 'max_abs_delta': None}
                    if digest in vectors:
                        try:
                            image_calls += 1
                            vector = image_engine.get_projected_global_embedding(paths[digest])
                            if (tuple(vector.shape) != (128,) or not torch.isfinite(vector).all()
                                    or not torch.allclose(vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4)):
                                raise ValueError('normalized_replay_embedding_required')
                            replay.update(status='replayed', max_abs_delta=float((vector - vectors[digest]).abs().max().item()))
                        except Exception:
                            pass
                    replays.append(replay)
                # Image evidence is sealed in memory before any label comparison.
                outcomes = []
                for image in plan['images']:
                    digest = image['cxr_sha256']
                    pairs = None
                    if digest in vectors:
                        values = (embeddings[:36] @ vectors[digest]).detach().float().cpu().tolist()
                        pairs = {finding: {} for finding in FINDINGS}
                        for i, p in enumerate(catalog()):
                            pairs[p['finding']][p['family']] = {'positive_cosine': values[2*i], 'negative_cosine': values[2*i+1]}
                    outcomes.append({**image, 'status': 'scored' if pairs is not None else 'failed_without_replacement', 'score_pairs': pairs})
        write_private_json(temporary / 'image_scores.json', {'schema_version': SCHEMA + '-image-scores', 'records': outcomes})
        image_rows, fact_rows, summary = build_tables(plan['rows'], plan['images'], plan['facts'], outcomes)
        summary.update({'elapsed_seconds': time.monotonic() - started,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
            'image_encoder_calls': image_calls, 'text_encoder_batches': text_batches,
            'text_encoder_inputs': text_inputs, 'unique_images': len(paths),
            'failed_unique_images': len(failed), 'text_replay_max_abs_deltas': text_replay,
            'image_replays': replays, 'positive_negative_text_cosines': geometry})
        write_private_text(temporary / 'image_finding_table.csv', csv_text(image_rows))
        write_private_text(temporary / 'candidate_finding_table.csv', csv_text(fact_rows))
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'frozen_protocol.json', plan['protocol'])
        lineage.reuse.verify_pins(pins)
        artifacts = {p.name: sha256_file(p) for p in temporary.iterdir() if p.is_file()}
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'plan_manifest_sha256': args.plan_manifest_sha256, 'sources': pins,
            'runtime': plan['runtime'], 'artifacts': artifacts, 'clinical_qualified': False,
            'original_selection_changed': False, 'regeneration_authorized': False,
            'new_training_calls': 0, 'new_generation_calls': 0})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output-root', default=str(BASE / 'finding_matched_biovil_plans'))
    prep.add_argument('--run-id', required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--plan-root', required=True)
    ev.add_argument('--plan-manifest-sha256', required=True)
    ev.add_argument('--allow-synthetic-finding-images', action='store_true')
    ev.add_argument('--output-root', default=str(BASE / 'finding_matched_biovil_runs'))
    ev.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        result = prepare(args) if args.command == 'prepare' else evaluate(args)
        print(json.dumps({'status': 'sealed_metadata_plan' if args.command == 'prepare' else 'completed_numeric_finding_diagnostic',
                          'manifest_sha256': sha256_file(result / 'manifest.json')}))
    except Exception:
        print('status=failed_closed private_diagnostics_retained', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

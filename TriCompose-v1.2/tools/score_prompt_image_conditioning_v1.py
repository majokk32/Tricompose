#!/usr/bin/env python3
"""Metadata-only prepare; separately approved frozen GPU prompt/CXR diagnostic.

Final prompts, not generated reports, are the text inputs. Retrieval of an
encoder-equivalence prompt group is not clinical EHR/CXR truth or attribution.
CPU preparation never hashes/opens pixels, prompts, EHRs, reports or weights.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'real_validation'),
               str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
import score_cached_opacity_candidates as reuse
from contracts import (PROTECTED_ROOT, WORKSPACE, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

SCHEMA = 'tricompose-prompt-image-conditioning-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
DELIVERY = BASE / 'deliverables/first_version_12714150_001'
DELIVERY_SHA = 'a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc'
RUNTIME_RECEIPT = BASE / 'candidate_opacity_biovil_runs/biovil_opacity240_12670345/manifest.json'
RUNTIME_SHA = '6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828'
PROTOCOL = ROOT / 'configs/prompt_image_conditioning_v1.json'
TESTS = ROOT / 'tests/test_prompt_image_conditioning_v1.py'
MODELS = ('chexgenbench_pixart', 'chexgenbench_sana', 'roentgen_v2')
EXPERTS = frozenset(('maira2', 'cxrmate_single', 'llavarad', 'chexagent2'))
HASH = re.compile(r'[0-9a-f]{64}\Z')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z')
IMAGE_FIELDS = ('case_id', 'cxr_candidate_id', 'cxr_model_id', 'cxr_seed',
    'ehr_sha256', 'ehr_facts_sha256', 'prompt_path', 'prompt_sha256',
    'cxr_path', 'cxr_sha256')
DEPENDENCIES = (ROOT / 'tools/score_cached_opacity_candidates.py',
    WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
    WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py',
    ROOT / 'real_validation/ricord_biovil50.py', ROOT / 'real_validation/rsua_biovil.py',
    ROOT / 'real_validation/biovil_matched_pairs.py',
    ROOT / 'real_validation/biovil_fact_polarity.py')


def protocol():
    value = json.loads(PROTOCOL.read_text())
    if (value['schema_version'] != SCHEMA + '-protocol'
            or value['image_models'] != list(MODELS)
            or (value['cases'], value['image_slots'], value['report_slots_retained']) != (80, 240, 960)
            or value['text_batch_size'] != 16 or value['seed'] != 0
            or value['image_replays'] != 2 or value['text_replays'] != 2
            or value['post_hoc_development'] is not True
            or any(value[k] is not False for k in (
                'different_prompt_is_clinical_negative', 'native_structured_ehr_to_cxr',
                'clinical_qualified', 'clinical_fault_attribution', 'regeneration_authorized',
                'selection_changed', 'threshold_fitting', 'training_allowed',
                'external_api_allowed', 'untouched_final_test'))):
        raise ValueError('frozen_diagnostic_only_protocol_required')
    return value


def images_from_index(rows, *, full=True):
    """Allowlisted projection; deliberately do not open any referenced artifact."""
    if not rows or len({r['triple_candidate_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty_unique_triple_inventory_required')
    images, anchors, grid, reports = {}, {}, defaultdict(set), set()
    for row in rows:
        image = {k: row[k] for k in IMAGE_FIELDS}
        for key in ('case_id', 'cxr_candidate_id', 'triple_candidate_id', 'report_candidate_id'):
            if not isinstance(row[key], str) or not ID.fullmatch(row[key]):
                raise ValueError('opaque_ids_required')
        if image['cxr_model_id'] not in MODELS or str(image['cxr_seed']) != '0' or row['report_model_id'] not in EXPERTS:
            raise ValueError('unchanged_three_generator_four_report_seed0_grid_required')
        for key in ('ehr_sha256', 'ehr_facts_sha256', 'prompt_sha256', 'cxr_sha256', 'report_sha256'):
            if not isinstance(row[key], str) or not HASH.fullmatch(row[key]):
                raise ValueError('sha256_lineage_required')
        for key, suffix in (('prompt_path', '.txt'), ('cxr_path', '.png')):
            path = require_inside(image[key], PROTECTED_ROOT, must_exist=False)
            if path.suffix != suffix:
                raise ValueError('bound_final_text_and_png_paths_required')
        anchor = (image['ehr_sha256'], image['ehr_facts_sha256'])
        if anchors.setdefault(image['case_id'], anchor) != anchor:
            raise ValueError('fixed_ehr_required')
        image_id = image['cxr_candidate_id']
        if images.setdefault(image_id, image) != image:
            raise ValueError('one_image_one_final_prompt_binding_required')
        slot = image['cxr_model_id'], row['report_model_id']
        if slot in grid[image['case_id']] or row['report_candidate_id'] in reports:
            raise ValueError('duplicate_generator_expert_or_report_slot')
        grid[image['case_id']].add(slot)
        reports.add(row['report_candidate_id'])
    expected = {(model, expert) for model in MODELS for expert in EXPERTS}
    if full and (len(rows) != 960 or len(images) != 240 or len(anchors) != 80
                 or any(slots != expected for slots in grid.values())):
        raise ValueError('complete_fixed80_3_4_bank_required')
    return [images[k] for k in sorted(images)]


def inventory(images):
    result = {'image_slots': len(images), 'cases': len({r['case_id'] for r in images}),
        'unique_images': len({r['cxr_sha256'] for r in images}),
        'unique_final_prompt_bytes': len({r['prompt_sha256'] for r in images}), 'by_generator': {}}
    for model in sorted({r['cxr_model_id'] for r in images}):
        subset = [r for r in images if r['cxr_model_id'] == model]
        result['by_generator'][model] = {'slots': len(subset),
            'unique_images': len({r['cxr_sha256'] for r in subset}),
            'unique_final_prompt_bytes': len({r['prompt_sha256'] for r in subset})}
    return result


def source_inputs():
    pins = {str(path): sha256_file(path) for path in (Path(__file__), PROTOCOL, TESTS, *DEPENDENCIES)}
    manifest = reuse.metadata(DELIVERY / 'manifest.json', pins, DELIVERY_SHA, cap=4 * 1024**2)
    if (manifest['schema_version'] != 'tricompose-first-version-protected-delivery-v1'
            or manifest['clinical_qualified'] is not False or manifest['new_model_calls'] != 0
            or manifest['original_selection_changed'] is not False):
        raise ValueError('authenticated_unchanged_delivery_required')
    rows = reuse.metadata(DELIVERY / 'candidate_index.csv', pins,
        manifest['artifacts']['candidate_index.csv']['sha256'])
    images = images_from_index(rows)
    previous = reuse.metadata(RUNTIME_RECEIPT, pins, RUNTIME_SHA)
    runtime = previous['runtime']
    for name, digest in runtime['dependencies'].items():
        path = require_inside(WORKSPACE / name, WORKSPACE, must_exist=True)
        if sha256_file(path) != digest:
            raise ValueError('deployed_loader_changed')
        pins[str(path)] = digest
    reuse.verify_pins(pins)
    return rows, images, runtime, pins


def prepare(args):
    reuse.guard()
    rules = protocol()
    rows, images, runtime, pins = source_inputs()
    plan = {'schema_version': SCHEMA + '-plan', 'protocol': rules, 'rows': rows,
        'images': images, 'runtime': runtime, 'inventory': inventory(images),
        'modality_source': 'fully_synthetic', 'clinical_bodies_or_pixels_read': False,
        'model_weights_read': False, 'new_model_calls': 0,
        'expected_image_encoder_calls': len({r['cxr_sha256'] for r in images}) + rules['image_replays'],
        'expected_text_encoder_inputs': len({r['prompt_sha256'] for r in images}) + rules['text_replays'],
        'output_prefix': 'conditioning_diagnostic_', 'selection_changed': False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'plan.json', plan)
        reuse.verify_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-plan-manifest',
            'sources': pins, 'artifacts': {'plan.json': sha256_file(temporary / 'plan.json')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def token_sequence_hash(ids):
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError('nonempty_unpadded_token_sequence_required')
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def make_groups(images, text_records):
    """Merge byte-distinct prompts when the actual frozen critic tokens collide."""
    if set(text_records) != {r['prompt_sha256'] for r in images}:
        raise ValueError('every_bound_prompt_must_have_explicit_outcome')
    groups = {}
    for image in images:
        record = text_records[image['prompt_sha256']]
        if record['status'] == 'encoded':
            if not HASH.fullmatch(record['token_sequence_sha256']):
                raise ValueError('encoded_token_hash_required')
            token = record['token_sequence_sha256']
            group_id = 'tokens_' + token
        elif record['status'] == 'failed_without_replacement' and record['token_sequence_sha256'] is None:
            group_id = 'unavailable_' + image['prompt_sha256']
        else:
            raise ValueError('explicit_text_success_or_failure_required')
        key = image['cxr_model_id'], group_id
        group = groups.setdefault(key, {'cxr_model_id': key[0], 'prompt_group_id': key[1],
            'status': record['status'], 'prompt_sha256s': set(), 'case_ids': set()})
        group['prompt_sha256s'].add(image['prompt_sha256'])
        group['case_ids'].add(image['case_id'])
    return [{**r, 'prompt_sha256s': sorted(r['prompt_sha256s']), 'case_ids': sorted(r['case_ids'])}
            for _, r in sorted(groups.items())]


def retrieval(scores, own_group):
    """No clinical threshold; nullable complete-inventory ranks and exact ties."""
    if own_group not in scores or not scores:
        raise ValueError('matched_prompt_group_required')
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                   or abs(value) > 1.00001) for value in scores.values()):
        raise ValueError('finite_cosines_or_explicit_null_required')
    own = scores[own_group]
    out = {'status': 'incomplete_prompt_inventory_no_rank', 'matched_cosine': own,
        'other_prompt_mean_cosine': None, 'matched_minus_other_mean': None,
        'best_rank': None, 'worst_rank': None, 'expected_recall_at_1': None,
        'expected_recall_at_5': None, 'expected_reciprocal_rank': None,
        'prompt_groups': len(scores), 'available_prompt_groups': sum(v is not None for v in scores.values())}
    if len(scores) == 1:
        return {**out, 'status': 'single_prompt_group_not_discriminating'}
    if any(v is None for v in scores.values()):
        return out
    others = [v for key, v in scores.items() if key != own_group]
    greater = sum(v > own for v in others)
    ties = 1 + sum(v == own for v in others)
    mean = sum(others) / len(others)
    return {**out, 'status': 'scored_complete_inventory', 'other_prompt_mean_cosine': mean,
        'matched_minus_other_mean': own - mean, 'best_rank': greater + 1,
        'worst_rank': greater + ties, 'expected_recall_at_1': min(1.0, max(0.0, (1 - greater) / ties)),
        'expected_recall_at_5': min(1.0, max(0.0, (5 - greater) / ties)),
        'expected_reciprocal_rank': sum(1 / rank for rank in range(greater + 1, greater + ties + 1)) / ties}


def reduce_groups(records):
    """Equal prompt-group weighting; exact duplicate images do not add votes."""
    grouped = defaultdict(dict)
    fields = ('matched_cosine', 'matched_minus_other_mean', 'expected_recall_at_1',
              'expected_recall_at_5', 'expected_reciprocal_rank')
    for row in records:
        key = row['cxr_model_id'], row['prompt_group_id']
        old = grouped[key].setdefault(row['cxr_sha256'], row)
        if any(old[field] != row[field] for field in (*fields, 'status', 'prompt_groups')):
            raise ValueError('same_image_and_encoder_group_must_share_scores')
    groups = []
    for (model, group), unique in sorted(grouped.items()):
        rows = list(unique.values())
        valid = [r for r in rows if r['status'] == 'scored_complete_inventory']
        groups.append({'cxr_model_id': model, 'prompt_group_id': group,
            'unique_images': len(rows), 'complete_images': len(valid),
            'prompt_groups': rows[0]['prompt_groups'],
            **{k: sum(r[k] for r in valid) / len(valid) if valid else None for k in fields}})
    result = {}
    for model in sorted({r['cxr_model_id'] for r in records}):
        all_groups = [r for r in groups if r['cxr_model_id'] == model]
        valid = [r for r in all_groups if r['complete_images'] > 0]
        result[model] = {'planned_prompt_groups': len(all_groups), 'available_prompt_groups': len(valid),
            'conditional_on_available_groups': len(valid) != len(all_groups),
            'planned_unique_image_group_pairs': sum(r['unique_images'] for r in all_groups),
            'complete_unique_image_group_pairs': sum(r['complete_images'] for r in all_groups),
            'conditional_on_available_images': any(r['complete_images'] != r['unique_images'] for r in all_groups),
            'uniform_random_group_recall_at_1': 1 / len(all_groups),
            **{k: sum(r[k] for r in valid) / len(valid) if valid else None for k in fields}}
    return groups, result


def candidate_overlay(rows, records):
    index = {r['cxr_candidate_id']: r for r in records}
    if len(index) != len(records) or set(index) != {r['cxr_candidate_id'] for r in rows}:
        raise ValueError('all_original_image_slots_required')
    fields = ('prompt_group_id', 'status', 'matched_cosine', 'matched_minus_other_mean',
              'best_rank', 'worst_rank', 'expected_recall_at_1', 'expected_recall_at_5')
    result = []
    for row in rows:
        record = index[row['cxr_candidate_id']]
        if any(row[k] != record[k] for k in IMAGE_FIELDS):
            raise ValueError('exact_fixed_ehr_final_prompt_image_lineage_required')
        if any(k.startswith('conditioning_diagnostic_') for k in row):
            raise ValueError('existing_overlay_cannot_be_overwritten')
        result.append({**row, **{'conditioning_diagnostic_' + k: record[k] for k in fields},
            'conditioning_diagnostic_clinical_accuracy': None,
            'conditioning_diagnostic_faulty_modality': None,
            'conditioning_diagnostic_selector_used': False})
    return result


def csv_text(rows):
    if not rows:
        raise ValueError('nonempty_output_required')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def evaluate(args):
    # This guard precedes even the plan read, imports, tokenization and bodies.
    reuse.guard(gpu=True, approved=args.allow_synthetic_prompt_image)
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise RuntimeError('offline_frozen_encoders_required')
    target = require_inside(Path(args.output_root) / args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('output_run_exists')
    pins = {}
    plan_root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    manifest = reuse.metadata(plan_root / 'manifest.json', pins, args.plan_manifest_sha256)
    reuse.verify_pins(manifest['sources']); pins.update(manifest['sources'])
    plan = reuse.metadata(plan_root / 'plan.json', pins, manifest['artifacts']['plan.json'])
    if (manifest['schema_version'] != SCHEMA + '-plan-manifest' or plan['schema_version'] != SCHEMA + '-plan'
            or plan['protocol'] != protocol() or plan['modality_source'] != 'fully_synthetic'
            or images_from_index(plan['rows']) != plan['images'] or plan['inventory'] != inventory(plan['images'])):
        raise ValueError('sealed_complete_synthetic_plan_required')
    import ricord_biovil50 as frozen
    if frozen.runtime_metadata() != plan['runtime']:
        raise ValueError('unchanged_deployed_frozen_encoder_required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('allocated_cuda_required')
    started = time.monotonic()
    rules = plan['protocol']
    images = plan['images']
    text_paths, image_paths = {}, {}
    # All bindings are checked privately after approval, including alias paths.
    for image in images:
        for field, digest_field, catalog in (('prompt_path', 'prompt_sha256', text_paths),
                                           ('cxr_path', 'cxr_sha256', image_paths)):
            path = require_inside(image[field], PROTECTED_ROOT, must_exist=True)
            if sha256_file(path) != image[digest_field]:
                raise ValueError('bound_final_prompt_or_image_changed')
            catalog.setdefault(image[digest_field], path)
            pins[str(path)] = image[digest_field]
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    text_records, text_vectors, image_vectors, image_errors = {}, {}, {}, {}
    text_batches = text_inputs = image_calls = 0
    image_replay = []
    text_replay_delta = None
    try:
        torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            image_engine, text_engine = frozen.reuse._load_runtime(frozen.reuse.MODEL, torch.device('cuda:0'))
            if any(e.model.training or any(p.requires_grad for p in e.model.parameters()) for e in (image_engine, text_engine)):
                raise RuntimeError('frozen_eval_mode_required')
            with torch.inference_mode():
                prepared = []
                for digest, path in sorted(text_paths.items()):
                    record = {'prompt_sha256': digest, 'status': 'failed_without_replacement',
                        'token_sequence_sha256': None, 'token_count': None, 'text_encoder_inputs': 0}
                    text_records[digest] = record
                    try:
                        if not 0 < path.stat().st_size <= 65536:
                            raise ValueError('bounded_final_prompt_required')
                        text = path.read_text(encoding='utf-8')
                        if not text.strip():
                            raise ValueError('empty_final_prompt_no_substitution')
                        tokens = text_engine.tokenize_input_prompts([text], verbose=False)
                        ids = tokens.input_ids[0][tokens.attention_mask[0].bool()].cpu().tolist()
                        record['token_count'] = len(ids)
                        prepared.append((digest, text, token_sequence_hash(ids)))
                    except Exception as error:
                        record['error_type'] = type(error).__name__
                payloads = prepared + ([prepared[0]] * rules['text_replays'] if prepared else [])
                vectors = []
                for begin in range(0, len(payloads), rules['text_batch_size']):
                    batch = payloads[begin:begin + rules['text_batch_size']]
                    try:
                        text_batches += 1; text_inputs += len(batch)
                        for digest, _, _ in batch:
                            text_records[digest]['text_encoder_inputs'] += 1
                        output = text_engine.get_embeddings_from_prompt([r[1] for r in batch], normalize=True, verbose=False)
                        if tuple(output.shape) != (len(batch), 128) or not torch.isfinite(output).all() or not torch.allclose(
                                output.norm(dim=1), torch.ones(len(batch), device=output.device), atol=1e-4, rtol=1e-4):
                            raise ValueError('normalized_128_text_vectors_required')
                        vectors.extend(list(output.detach().float().cpu()))
                        for offset, ((digest, _, token), vector) in enumerate(zip(batch, output, strict=True)):
                            if begin + offset < len(prepared):
                                text_vectors[digest] = vector
                                text_records[digest].update(status='encoded', token_sequence_sha256=token)
                    except Exception as error:
                        vectors.extend([None] * len(batch))
                        for offset, (digest, _, _) in enumerate(batch):
                            if begin + offset < len(prepared) and digest not in text_vectors:
                                text_records[digest]['error_type'] = type(error).__name__
                if prepared and len(vectors) == len(payloads):
                    a, b = vectors[-2:]
                    if a is not None and b is not None and vectors[0] is not None:
                        text_replay_delta = float(max((a - b).abs().max().item(),
                            (a - vectors[0]).abs().max().item(), (b - vectors[0]).abs().max().item()))
                for index, (digest, path) in enumerate(sorted(image_paths.items())):
                    try:
                        image_calls += 1
                        vector = image_engine.get_projected_global_embedding(path)
                        if tuple(vector.shape) != (128,) or not torch.isfinite(vector).all() or not torch.allclose(
                                vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4):
                            raise ValueError('normalized_128_image_vector_required')
                        image_vectors[digest] = vector
                        if index < rules['image_replays']:
                            check = {'cxr_sha256': digest, 'status': 'failed_replay_without_retry', 'max_abs_vector_delta': None}
                            try:
                                image_calls += 1
                                repeated = image_engine.get_projected_global_embedding(path)
                                delta = float((vector - repeated).abs().max().item())
                                if not math.isfinite(delta):
                                    raise ValueError('finite_replay_required')
                                check.update(status='replayed', max_abs_vector_delta=delta)
                            except Exception as error:
                                check['error_type'] = type(error).__name__
                            image_replay.append(check)
                    except Exception as error:
                        image_errors[digest] = type(error).__name__
                groups = make_groups(images, text_records)
                group_index = {key: r['prompt_group_id'] for r in groups
                    for key in ((r['cxr_model_id'], digest) for digest in r['prompt_sha256s'])}
                group_vectors = {}
                for group in groups:
                    available = [text_vectors[d] for d in group['prompt_sha256s'] if d in text_vectors]
                    if available:
                        if any(not torch.allclose(available[0], v, atol=1e-5, rtol=1e-5) for v in available[1:]):
                            raise ValueError('same_critic_tokens_embedding_collision_changed')
                        group_vectors[group['cxr_model_id'], group['prompt_group_id']] = available[0]
                pair_rows, records = [], []
                for image in images:
                    vector = image_vectors.get(image['cxr_sha256'])
                    own_group = group_index[image['cxr_model_id'], image['prompt_sha256']]
                    scores = {}
                    for group in [g for g in groups if g['cxr_model_id'] == image['cxr_model_id']]:
                        text_vector = group_vectors.get((group['cxr_model_id'], group['prompt_group_id']))
                        score = float((text_vector @ vector).item()) if vector is not None and text_vector is not None else None
                        scores[group['prompt_group_id']] = score
                        pair_rows.append({**image, 'prompt_group_id': group['prompt_group_id'],
                            'is_intended_encoder_prompt_group': group['prompt_group_id'] == own_group,
                            'raw_cosine': score, 'clinical_negative_validated': False})
                    stats = retrieval(scores, own_group)
                    if vector is None:
                        stats['status'] = 'failed_image_without_replacement'
                    records.append({**image, 'prompt_group_id': own_group, **stats,
                        'image_error_type': image_errors.get(image['cxr_sha256']),
                        'clinical_accuracy': None, 'confirmed_faulty_modality': None,
                        'selector_used': False, 'regeneration_authorized': False})
            torch.cuda.synchronize()
        grouped, model_summary = reduce_groups(records)
        candidates = candidate_overlay(plan['rows'], records)
        summary = {'schema_version': SCHEMA, 'protocol': rules, 'inventory': plan['inventory'],
            'by_generator_equal_weight_prompt_groups': model_summary,
            'image_slot_status_counts': dict(Counter(r['status'] for r in records)),
            'text_prompt_status_counts': dict(Counter(r['status'] for r in text_records.values())),
            'byte_prompt_groups': sum(plan['inventory']['by_generator'][m]['unique_final_prompt_bytes'] for m in MODELS),
            'critic_token_prompt_groups': len(groups),
            'image_encoder_calls': image_calls, 'text_encoder_batches': text_batches, 'text_encoder_inputs': text_inputs,
            'expected_image_encoder_calls': plan['expected_image_encoder_calls'],
            'expected_text_encoder_inputs': plan['expected_text_encoder_inputs'],
            'text_replay_max_abs_vector_delta': text_replay_delta, 'image_replays': image_replay,
            'old_columns_retained': len(plan['rows'][0]), 'candidate_rows_retained': len(candidates),
            'generation_calls': 0, 'training_calls': 0, 'selection_changed': False,
            'clinical_qualified': False, 'regeneration_authorized': False,
            'clinical_bodies_read': 'only_bound_synthetic_final_prompts_no_ehr_or_reports',
            'real_data_read': False, 'new_primary_score': False,
            'elapsed_seconds_before_serialization': round(time.monotonic() - started, 6),
            'peak_allocated_vram_gib_including_load': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
        reuse.verify_pins(pins)
        if frozen.runtime_metadata() != plan['runtime']:
            raise ValueError('frozen_runtime_changed_during_scoring')
        files = {'image_conditioning_table.csv': records, 'prompt_group_summary.csv': grouped,
                 'within_generator_pair_cosines.csv': pair_rows, 'candidate_conditioning_table.csv': candidates}
        for name, rows in files.items():
            write_private_text(temporary / name, csv_text(rows))
        write_private_json(temporary / 'text_encoding_receipts.json', {'records': list(text_records.values()), 'groups': groups})
        write_private_json(temporary / 'summary.json', summary)
        names = (*files, 'text_encoding_receipts.json', 'summary.json')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'plan_manifest_sha256': args.plan_manifest_sha256, 'sources': pins, 'runtime': plan['runtime'],
            'artifacts': {name: sha256_file(temporary / name) for name in names},
            'selection_changed': False, 'clinical_qualified': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('--plan-root')
    parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--allow-synthetic-prompt-image', action='store_true')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        root = prepare(args) if args.action == 'prepare' else evaluate(args)
        print(json.dumps({'status': 'sealed_metadata_plan_no_inference' if args.action == 'prepare'
                          else 'completed_conditioning_retrieval_diagnostic_not_clinical_score',
                          'manifest_sha256': sha256_file(root / 'manifest.json')}, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

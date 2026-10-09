#!/usr/bin/env python3
"""Frozen RICORD opacity probes; metadata-only prepare, separately approved GPU run."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
sys.path[:0] = [str(WORKSPACE / 'TriCompose-v1.2/real_validation'),
               str(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1')]
import rsua_biovil as reuse
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, read_json, require_inside, sha256_file, write_private_json,
    write_private_text)

SCHEMA = 'tricompose-ricord-biovil-opacity50-v1'
PLAN_SCHEMA = SCHEMA + '-plan'
SOURCE = PROTECTED_ROOT / 'tricompose_v1_2/real_validation/ricord_xrv_pilots/ricord_xrv50_12667524'
SOURCE_SHA = 'dc14a2ba1aed3823f0752a73dda297e874f638b31644d2992227eb3f1cf697a9'
PROTOCOL = WORKSPACE / 'docs/ricord_biovil50_protocol.md'
TESTS = WORKSPACE / 'TriCompose-v1.2/tests/test_ricord_biovil50.py'
PACKAGES = ('torch', 'torchvision', 'numpy', 'Pillow', 'transformers')
PREPROCESSING = {'source': 'unchanged_previously_rendered_PNG',
    'official_loader_remap': 'image_minmax_to_uint8', 'resize': 512,
    'center_crop': 448, 'embedding_dimension': 128, 'normalization': 'L2',
    'display_transform_clinically_verified': False}


def catalog():
    return [p for p in reuse.probes() if p['finding'] == 'Lung Opacity']


def require_slurm(approved=None):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or (approved is not None and approved is not True):
        raise RuntimeError('explicit_slurm_approval_required')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('actual_slurm_cgroup_required')
    if approved is True and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('allocated_cuda_environment_required')


def sources():
    names = (Path(__file__), TESTS, PROTOCOL, Path(reuse.__file__),
        WORKSPACE / 'TriCompose-v1.2/real_validation/biovil_fact_polarity.py',
        WORKSPACE / 'TriCompose-v1.2/real_validation/biovil_matched_pairs.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py')
    return {str(p.relative_to(WORKSPACE)): sha256_file(p) for p in names}


def runtime_metadata():
    result = reuse.runtime_metadata()
    result['package_versions'] = {name: metadata.version(name) for name in PACKAGES}
    result['checkpoint_sha256s'] = {name: sha256_file(reuse.MODEL / name)
        for name in (reuse.IMAGE_WEIGHT, 'pytorch_model.bin')}
    if result['checkpoint_sha256s'] != {reuse.IMAGE_WEIGHT: reuse.IMAGE_SHA,
                                     'pytorch_model.bin': reuse.TEXT_SHA}:
        raise ValueError('frozen_checkpoint_changed')
    return result


def source_receipt():
    root = require_inside(SOURCE, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / 'manifest.json') != SOURCE_SHA:
        raise ValueError('fixed_source_manifest_changed')
    manifest = read_json(root / 'manifest.json')
    if manifest['schema_version'] != 'tricompose-ricord-xrv50-diagnostic-v1-manifest':
        raise ValueError('source_schema_changed')
    for name, digest in manifest['artifacts'].items():
        if sha256_file(require_inside(root / name, root, must_exist=True)) != digest:
            raise ValueError('source_artifact_changed')
    for collection in ('sources', 'dependency_sources'):
        for name, digest in manifest[collection].items():
            if sha256_file(WORKSPACE / name) != digest:
                raise ValueError('consumed_source_changed')
    summary = read_json(root / 'summary.json')
    if (summary['scored_cases'] != 50 or summary['failed_cases'] != 0 or
            summary['frozen'] is not True or summary['patient_grouping_verified'] is not True or
            summary['primary_metric_eligible'] is not False):
        raise ValueError('complete_fixed50_reference_required')
    images = [{'case_id': Path(name).stem, 'artifact': name, 'sha256': digest}
        for name, digest in sorted(manifest['artifacts'].items()) if name.startswith('displays/')]
    if len(images) != 50 or [r['case_id'] for r in images] != [f'case_{i:03d}' for i in range(50)] or \
            any(not re.fullmatch(r'displays/case_[0-9]{3}\.png', r['artifact']) for r in images):
        raise ValueError('all_original_opaque_displays_required')
    return root, summary, images


def prepare(args):
    require_slurm()
    _, original, images = source_receipt()  # Hashes only; no pixels or score rows.
    plan = {'schema_version': PLAN_SCHEMA, 'source_manifest_sha256': SOURCE_SHA,
        'source_root': str(SOURCE.relative_to(WORKSPACE)), 'images': images,
        'authored_probes': catalog(), 'primary_reduction': reuse.MEAN,
        'decision_rule': 'margin_positive_positive_negative_negative_exact_zero_unknown',
        'preprocessing': PREPROCESSING, 'runtime': runtime_metadata(),
        'reference_kind': original['reference_kind'], 'case_count': 50,
        'image_replay_indices': [0, 1], 'text_inputs_including_duplicate': 8,
        'expected_image_encoder_calls': 52, 'expected_text_encoder_batches': 1,
        'source_reference_result_already_seen': True, 'post_hoc_development': True,
        'primary_metric_eligible': False, 'training_calls': 0, 'generation_calls': 0,
        'model_calls': 0, 'pixels_opened': False, 'selection_changed': False,
        'thresholds_fitted': False, 'templates_fitted': False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'plan.json', plan)
        write_private_json(temporary / 'manifest.json', {'schema_version': PLAN_SCHEMA + '-manifest',
            'sources': sources(), 'artifacts': {'plan.json': sha256_file(temporary / 'plan.json')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'sealed_metadata_only_no_inference', 'model_calls': 0}


def reference_join(original, images):
    rows = original['records']
    index = {r['case_id']: r for r in rows}
    if len(index) != 50 or len(rows) != 50 or set(index) != {r['case_id'] for r in images}:
        raise ValueError('complete_reference_lineage_required')
    if Counter(r['reference_state'] for r in rows) != {'positive': 25, 'negative': 25}:
        raise ValueError('balanced_positive_negative_reference_required')
    for image in images:
        row = index[image['case_id']]
        score = row['direct_lung_opacity_score']
        if row['display_sha256'] != image['sha256'] or type(score) not in (int, float) or \
                not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('same_display_exact_xrv_required')
    return index


def summarize(records, references):
    ids = [r['case_id'] for r in records]
    if not references or len(set(ids)) != len(ids) or not set(ids).issubset(references):
        raise ValueError('unique_scored_subset_required')
    states = {cid: r['reference_state'] for cid, r in references.items()}
    if any(s not in ('positive', 'negative') for s in states.values()):
        raise ValueError('unknown_reference_not_negative')
    metrics = reuse.summarize(records, {cid: states[cid] for cid in ids}) if records else None
    successes = Counter(states[cid] for cid in ids)
    total = Counter(states.values())
    for value in (metrics or {}).values():
        value['positive_reference_recovery_all_planned'] = value['positive_polarity_wins'] / total['positive'] if total['positive'] else None
        value['negative_reference_recovery_all_planned'] = value['negative_polarity_wins'] / total['negative'] if total['negative'] else None
    counts = Counter()
    for row in records:
        margin = reuse.margins(row)[reuse.MEAN]
        image_positive = references[row['case_id']]['direct_lung_opacity_score'] >= .5
        key = 'biovil_tie_unknown' if margin == 0 else 'both_positive' if image_positive and margin > 0 else \
            'both_negative' if not image_positive and margin < 0 else \
            'xrv_positive_biovil_negative' if image_positive else 'xrv_negative_biovil_positive'
        counts[key] += 1
    return {'planned_cases': len(references), 'scored_cases': len(records),
        'failed_cases': len(references) - len(records), 'coverage': len(records) / len(references),
        'planned_positive': total['positive'], 'planned_negative': total['negative'],
        'scored_positive': successes['positive'], 'scored_negative': successes['negative'],
        'metrics_conditional_on_scored_subset': len(records) != len(references),
        'templates': metrics, 'same_display_xrv_exact_0_5_comparison': dict(counts),
        'cross_model_agreement_is_clinical_truth': False}


def result_markdown(summary):
    lines = ['# Frozen RICORD opacity probes / 图文否定能力诊断', '',
        'Post-hoc DEVELOPMENT, not a held-out clinical evaluator or repair success.', '',
        '| Profile | AUROC | AP | Positive wins / scored positive | Negative wins / scored negative | Ties |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in (summary['templates'] or {}).items():
        show = lambda x: 'NA' if x is None else f'{x:.4f}'
        lines.append(f"| {name} | {show(row['auroc'])} | {show(row['average_precision'])} | "
            f"{row['positive_polarity_wins']}/{row['positive']} | {row['negative_polarity_wins']}/{row['negative']} | {row['ties']} |")
    lines.extend(['', 'Unknown/ties/failures are not negatives or successful repair. All 50 cases retained.',
        'A high AUROC can coexist with poor negative-assertion direction. No template/threshold fitting.',
        'The reused PNGs retain their bytes; the official BioViL loader internally applies min/max remapping.',
        'This is not identical preprocessing to XRV, not an independent synthetic clinical endpoint.',
        '', '```json', json.dumps(summary, indent=2, sort_keys=True), '```', ''])
    return '\n'.join(lines)


def run(args):
    require_slurm(getattr(args, 'allow_ricord_biovil_pixels', False))
    plan_root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    if sha256_file(plan_root / 'manifest.json') != args.plan_manifest_sha256:
        raise ValueError('plan_manifest_changed')
    manifest = read_json(plan_root / 'manifest.json')
    for name, digest in manifest['sources'].items():
        if sha256_file(WORKSPACE / name) != digest:
            raise ValueError('frozen_program_protocol_changed')
    if sha256_file(plan_root / 'plan.json') != manifest['artifacts']['plan.json']:
        raise ValueError('plan_changed')
    plan = read_json(plan_root / 'plan.json')
    source, _, images = source_receipt()
    if (plan['schema_version'] != PLAN_SCHEMA or plan['images'] != images or
            plan['authored_probes'] != catalog() or plan['primary_reduction'] != reuse.MEAN or
            plan['preprocessing'] != PREPROCESSING or plan['runtime'] != runtime_metadata() or
            plan['source_manifest_sha256'] != SOURCE_SHA or plan['image_replay_indices'] != [0, 1]):
        raise ValueError('fixed_probe_runtime_contract_changed')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('allocated_gpu_required')
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started, records, outcomes, replay = time.monotonic(), [], [], []
    image_calls = text_batches = 0
    try:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        prompts = [p[k] for p in catalog() for k in ('positive_text', 'negative_text')]
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            image_engine, text_engine = reuse._load_runtime(reuse.MODEL, torch.device('cuda:0'))
            if any(e.model.training or any(p.requires_grad for p in e.model.parameters()) for e in (image_engine, text_engine)):
                raise RuntimeError('model_not_frozen')
            with torch.inference_mode():
                text_batches += 1
                embeddings = text_engine.get_embeddings_from_prompt(prompts + [prompts[0]] * 2, normalize=True, verbose=False)
                if tuple(embeddings.shape) != (8, 128) or not torch.isfinite(embeddings).all() or not torch.allclose(
                        embeddings.norm(dim=1), torch.ones(8, device=embeddings.device), atol=1e-4, rtol=1e-4):
                    raise ValueError('normalized_text_embeddings_required')
                duplicate_delta = float((embeddings[-2] - embeddings[-1]).abs().max().item())
                geometry = {family: float((embeddings[2*i] @ embeddings[2*i+1]).item()) for i, family in enumerate(reuse.FAMILIES)}
                for position, image in enumerate(images):
                    outcome = {'case_id': image['case_id'], 'status': 'failed_without_replacement', 'image_calls': 0}
                    try:
                        path = require_inside(source / image['artifact'], source, must_exist=True)
                        if sha256_file(path) != image['sha256']:
                            raise ValueError('fixed_png_changed')
                        image_calls += 1; outcome['image_calls'] += 1
                        vector = image_engine.get_projected_global_embedding(path)
                        if tuple(vector.shape) != (128,) or not torch.isfinite(vector).all() or not torch.allclose(
                                vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4):
                            raise ValueError('normalized_image_embedding_required')
                        values = (embeddings[:6] @ vector).detach().float().cpu().tolist()
                        row = {'case_id': image['case_id'], 'source_image_sha256': image['sha256'],
                            'score_pairs': {family: {'positive_cosine': values[2*i], 'negative_cosine': values[2*i+1]}
                                for i, family in enumerate(reuse.FAMILIES)}}
                        reuse.margins(row)
                        records.append(row); outcome['status'] = 'scored'
                        if position in plan['image_replay_indices']:
                            check = {'case_id': image['case_id'], 'status': 'failed_replay_without_retry'}
                            try:
                                image_calls += 1; outcome['image_calls'] += 1
                                repeated = image_engine.get_projected_global_embedding(path)
                                delta = float(((embeddings[:6] @ repeated) - (embeddings[:6] @ vector)).abs().max().item())
                                if not math.isfinite(delta):
                                    raise ValueError('nonfinite_replay_delta')
                                check.update(status='replayed', max_abs_score_delta=delta)
                            except Exception as error:
                                check['error_type'] = type(error).__name__
                            replay.append(check)
                    except Exception as error:
                        outcome['error_type'] = type(error).__name__
                    outcomes.append(outcome)
            torch.cuda.synchronize()
        # Only now join the reference and prior XRV predictions; neither enters encoding.
        references = reference_join(read_json(source / 'scores.json'), images)
        summary = {'schema_version': SCHEMA, **summarize(records, references),
            'status': 'completed_same_cohort_diagnostic_unqualified' if len(records) == 50 else 'incomplete_all_denominators_retained',
            'reference_kind': plan['reference_kind'], 'patient_grouping_verified': True,
            'official_adjudication_reproduced': False, 'checkpoint_training_overlap_verified': False,
            'post_hoc_development': True, 'primary_metric_eligible': False, 'independent_final_endpoint': False,
            'frozen': True, 'probability_semantics': False, 'primary_reduction': reuse.MEAN,
            'thresholds_fitted': False, 'templates_fitted': False, 'selection_changed': False,
            'regeneration_authorized': False, 'preprocessing': PREPROCESSING,
            'model_received_reference_labels': False, 'raw_ehr_or_report_text_read': False,
            'training_calls': 0, 'generation_calls': 0,
            'image_encoder_calls': image_calls, 'text_encoder_batches': text_batches,
            'text_inputs_including_duplicate': 8, 'duplicate_text_embedding_delta': duplicate_delta,
            'text_geometry_positive_negative_cosine': geometry, 'first_two_image_replay': replay,
            'elapsed_seconds_before_serialization': round(time.monotonic() - started, 6),
            'peak_allocated_vram_including_load_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
        _, _, final_images = source_receipt()
        if final_images != images or runtime_metadata() != plan['runtime'] or sources() != manifest['sources']:
            raise ValueError('source_changed_during_run')
        write_private_json(temporary / 'scores.json', {'schema_version': SCHEMA, 'records': records, 'outcomes': outcomes})
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md', result_markdown(summary))
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'sources': manifest['sources'], 'runtime': plan['runtime'], 'source_manifest_sha256': SOURCE_SHA,
            'plan_manifest_sha256': args.plan_manifest_sha256,
            'artifacts': {name: sha256_file(temporary / name) for name in ('scores.json', 'summary.json', 'RESULTS_CN_EN.md')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': summary['status'], 'elapsed_seconds': summary['elapsed_seconds_before_serialization'],
        'peak_allocated_vram_gib': summary['peak_allocated_vram_including_load_gib']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--plan-root')
    parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--allow-ricord-biovil-pixels', action='store_true')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        target, status = prepare(args) if args.action == 'prepare' else run(args)
        status['manifest_sha256'] = sha256_file(target / 'manifest.json')
        print(json.dumps(status, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

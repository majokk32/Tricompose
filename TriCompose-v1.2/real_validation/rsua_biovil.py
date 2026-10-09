#!/usr/bin/env python3
"""Predeclare a same-cohort BioViL-T diagnostic; pixels require approved Slurm.

Distinct frozen architecture is not independent clinical adjudication. Keep
all three existing pneumonia polarity templates, old choices and thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
sys.path.insert(0, str(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, read_json, require_inside, sha256_file,
                       write_private_json, write_private_text)
from biovil_matched_pairs import IMAGE_WEIGHT, _load_runtime, auc_ap
from biovil_fact_polarity import MODEL_METADATA_SHAS, VENDOR_TREE_SHA, probes

SCHEMA = 'tricompose-rsua-biovil-polarity-v1'
PLAN_SCHEMA = SCHEMA + '-plan'
XRV_MANIFEST_SHA = '92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0'
COHORT_SHA = 'ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1'
IMAGE_SHA = 'b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77'
TEXT_SHA = '6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c'
MODEL = WORKSPACE / 'CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5'
VENDOR = WORKSPACE / 'runtime/vendor/hi_ml_multimodal_0_2_2/health_multimodal'
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
MEAN = 'mean_all_three_predeclared'
PROFILES = {'default_0_5': 0.5, 'unchanged_weak_reference_transport': 0.55457607}
DEPENDENCIES = ('TriCompose-v1.2/real_validation/biovil_matched_pairs.py',
                'TriCompose-v1.2/real_validation/biovil_fact_polarity.py',
                'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py',
                'TriCompose-v1.0/eval/report_v1_1/contracts.py')


def catalog():
    return [row for row in probes() if row['finding'] == 'Pneumonia']


def require_gpu_approval(args):
    job = os.environ.get('SLURM_JOB_ID', '')
    if getattr(args, 'allow_rsua_pixels', False) is not True or not job.isdigit():
        raise RuntimeError('approved_rsua_biovil_slurm_required')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('slurm_cgroup_required')


def runtime_metadata():
    dependencies = {name: sha256_file(WORKSPACE / name) for name in DEPENDENCIES}
    metadata = {name: sha256_file(MODEL / name) for name in MODEL_METADATA_SHAS}
    vendor = {str(path.relative_to(VENDOR)): sha256_file(path)
              for path in sorted(VENDOR.rglob('*.py'))}
    digest = hashlib.sha256(json.dumps(vendor, sort_keys=True).encode()).hexdigest()
    if metadata != MODEL_METADATA_SHAS or digest != VENDOR_TREE_SHA:
        raise ValueError('frozen_runtime_metadata_changed')
    return {'dependencies': dependencies, 'model_metadata': metadata, 'vendor_tree_sha256': digest}


def source_receipt(source_run):
    root = require_inside(source_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / 'manifest.json') != XRV_MANIFEST_SHA:
        raise ValueError('fixed_xrv_manifest_changed')
    manifest = read_json(root / 'manifest.json')
    if manifest.get('schema_version') != 'tricompose-rsua-xrv-pilot-v1-manifest':
        raise ValueError('fixed_xrv_schema_changed')
    for name in ('summary.json', 'scores.json', 'score_table.csv'):
        if sha256_file(root / name) != manifest['artifacts'][name]:
            raise ValueError('fixed_xrv_artifact_changed')
    summary = read_json(root / 'summary.json')
    if (summary['cohort_sha256'] != COHORT_SHA or summary['model_calls'] != 50
            or summary['frozen'] is not True or summary['selection_changed'] is not False
            or summary['thresholds_fitted_on_this_cohort'] is not False
            or summary['reference_kind'] != 'published_pneumonia_vs_paper_described_normal_cohort_classes'):
        raise ValueError('fixed_source_protocol_changed')
    return root


def prepare(args):
    # Aggregate receipts, file hashes and runtime source only; no image decode.
    source = source_receipt(args.source_run)
    cohort = require_inside(args.cohort, PROTECTED_ROOT, must_exist=True)
    if sha256_file(cohort) != COHORT_SHA:
        raise ValueError('fixed_cohort_changed')
    metadata = runtime_metadata()
    for name in (IMAGE_WEIGHT, 'pytorch_model.bin'):
        if not (MODEL / name).is_file():
            raise ValueError('local_checkpoint_missing')
    plan = {'schema_version': PLAN_SCHEMA, 'source_run': str(source),
            'source_manifest_sha256': XRV_MANIFEST_SHA, 'cohort': str(cohort),
            'cohort_sha256': COHORT_SHA, 'program_sha256': sha256_file(__file__),
            'runtime_metadata': metadata, 'model_path': str(MODEL),
            'image_checkpoint_sha256': IMAGE_SHA, 'text_checkpoint_sha256': TEXT_SHA,
            'authored_probes': catalog(), 'xrv_profiles': PROFILES,
            'case_count': 50, 'replay_count': 2, 'text_inputs_including_duplicate': 8,
            'expected_image_encoder_calls': 52, 'expected_text_encoder_batches': 1,
            'primary_reduction': MEAN, 'thresholds_fitted': False,
            'template_choice_fitted': False, 'selection_changed': False,
            'primary_metric_eligible': False, 'pixel_values_decoded': False,
            'clinical_reference_is_published_cohort_proxy': True,
            'bmp_conversion': 'lossless_L_png_checked_by_grayscale_bytes',
            'preprocessing': {'resize': 512, 'center_crop': 448, 'embedding_dimension': 128,
                              'normalize': 'L2', 'official_loader_remap': 'minmax_to_uint8'}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'plan.json', plan)
        write_private_json(temporary / 'manifest.json', {'schema_version': PLAN_SCHEMA + '-manifest',
            'run_id': args.run_id, 'program_sha256': plan['program_sha256'],
            'plan_sha256': sha256_file(temporary / 'plan.json'), 'model_calls': 0})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def margins(row):
    pairs = row['score_pairs']
    if set(pairs) != set(FAMILIES):
        raise ValueError('invalid_probe_inventory')
    result = {}
    for family in FAMILIES:
        pair = pairs[family]
        if set(pair) != {'positive_cosine', 'negative_cosine'}:
            raise ValueError('invalid_pair_inventory')
        if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1.00001
               for value in pair.values()):
            raise ValueError('invalid_cosine')
        result[family] = pair['positive_cosine'] - pair['negative_cosine']
    result[MEAN] = sum(result.values()) / 3
    return result


def summarize(records, references):
    ids = [row['case_id'] for row in records]
    if not records or len(set(ids)) != len(ids) or set(ids) != set(references):
        raise ValueError('reference_lineage_mismatch')
    if any(value not in ('positive', 'negative') for value in references.values()):
        raise ValueError('unknown_is_not_negative')
    values = [margins(row) for row in records]
    labels = [int(references[row['case_id']] == 'positive') for row in records]
    result = {}
    for family in (*FAMILIES, MEAN):
        scores = [value[family] for value in values]
        positive, negative = sum(labels), len(labels) - sum(labels)
        pw = sum(label == 1 and score > 0 for label, score in zip(labels, scores))
        nw = sum(label == 0 and score < 0 for label, score in zip(labels, scores))
        result[family] = {**auc_ap(labels, scores), 'positive': positive, 'negative': negative,
            'positive_polarity_wins': pw, 'negative_polarity_wins': nw,
            'positive_reference_win_rate': pw / positive if positive else None,
            'negative_reference_win_rate': nw / negative if negative else None,
            'balanced_polarity_win_rate': (pw / positive + nw / negative) / 2 if positive and negative else None,
            'ties': sum(score == 0 for score in scores), 'mean_margin': sum(scores) / len(scores)}
    return result


def scorer_disagreement(records, references, xrv):
    ids = {row['case_id'] for row in records}
    if len(ids) != len(records) or ids != set(references) or ids != set(xrv):
        raise ValueError('scorer_lineage_mismatch')
    if any(value not in ('positive', 'negative') for value in references.values()):
        raise ValueError('unknown_is_not_negative')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in xrv.values()):
        raise ValueError('invalid_xrv_score')
    result = {}
    categories = ('both_positive', 'both_negative', 'xrv_positive_biovil_negative',
                  'xrv_negative_biovil_positive', 'biovil_tie_unknown')
    for profile, threshold in PROFILES.items():
        counts = {state: Counter() for state in ('positive', 'negative')}
        for row in records:
            margin = margins(row)[MEAN]
            positive = xrv[row['case_id']] >= threshold
            category = ('biovil_tie_unknown' if margin == 0 else
                        'both_positive' if positive and margin > 0 else
                        'both_negative' if not positive and margin < 0 else
                        'xrv_positive_biovil_negative' if positive else 'xrv_negative_biovil_positive')
            counts[references[row['case_id']]][category] += 1
        total = counts['positive'] + counts['negative']
        comparable = len(records) - total['biovil_tie_unknown']
        result[profile] = {'xrv_threshold': threshold, 'case_count': len(records),
            'comparable': comparable, 'counts': {key: total[key] for key in categories},
            'by_published_reference': {state: {key: count[key] for key in categories} for state, count in counts.items()},
            'agreement_rate_on_non_ties': (total['both_positive'] + total['both_negative']) / comparable if comparable else None,
            'interpretation': 'model_agreement_is_not_independent_clinical_adjudication'}
    return result


def lossless_png(source, destination, *, image_api=None):
    # Called only after GPU/approval guard. BMP is unsupported by official loader.
    if image_api is None:
        from PIL import Image as image_api
    with image_api.open(source) as original:
        grayscale = original.convert('L')
        if grayscale.size != (256, 256) or grayscale.getextrema()[0] == grayscale.getextrema()[1]:
            raise ValueError('invalid_or_constant_image_no_substitution')
        before = grayscale.tobytes()
        grayscale.save(destination, format='PNG')
    os.chmod(destination, 0o660)
    with image_api.open(destination) as reopened:
        if reopened.size != (256, 256) or reopened.mode != 'L' or reopened.tobytes() != before:
            raise ValueError('lossless_png_pixels_changed')


def run(args):
    require_gpu_approval(args)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('gpu_required')
    plan_root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(plan_root / 'manifest.json') != args.plan_manifest_sha256:
        raise ValueError('plan_manifest_changed')
    manifest = read_json(plan_root / 'manifest.json')
    if sha256_file(plan_root / 'plan.json') != manifest['plan_sha256']:
        raise ValueError('plan_changed')
    plan = read_json(plan_root / 'plan.json')
    if (plan['schema_version'] != PLAN_SCHEMA or plan['program_sha256'] != sha256_file(__file__)
            or plan['runtime_metadata'] != runtime_metadata() or plan['authored_probes'] != catalog()
            or plan['xrv_profiles'] != PROFILES or plan['model_path'] != str(MODEL)
            or plan['replay_count'] != 2 or plan['primary_reduction'] != MEAN):
        raise ValueError('frozen_plan_protocol_changed')
    source = source_receipt(plan['source_run'])
    cohort_path = require_inside(plan['cohort'], PROTECTED_ROOT, must_exist=True)
    if sha256_file(cohort_path) != COHORT_SHA:
        raise ValueError('fixed_cohort_changed')
    rows = read_json(source / 'scores.json')['records']
    ids = [row['case_id'] for row in rows]
    if ids != [f'case_{i:04d}' for i in range(50)]:
        raise ValueError('fixed_50_cases_required')
    assets = {MODEL / IMAGE_WEIGHT: IMAGE_SHA, MODEL / 'pytorch_model.bin': TEXT_SHA}
    started = time.monotonic()
    for path, digest in assets.items():
        if sha256_file(path) != digest:
            raise ValueError('frozen_checkpoint_changed')
    asset_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in assets}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        image_root = temporary / 'inputs'
        image_root.mkdir(mode=0o2770)
        os.chmod(image_root, 0o2770)
        records, replay = [], []
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        prompts = [row[key] for row in catalog() for key in ('positive_text', 'negative_text')]
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            image_engine, text_engine = _load_runtime(MODEL, torch.device('cuda:0'))
            if any(engine.model.training or any(p.requires_grad for p in engine.model.parameters())
                   for engine in (image_engine, text_engine)):
                raise RuntimeError('model_not_frozen')
            with torch.inference_mode():
                embeddings = text_engine.get_embeddings_from_prompt(prompts + [prompts[0], prompts[0]], normalize=True, verbose=False)
                if tuple(embeddings.shape) != (8, 128) or not torch.isfinite(embeddings).all() or not torch.allclose(
                        embeddings.norm(dim=1), torch.ones(8, device=embeddings.device), atol=1e-4, rtol=1e-4):
                    raise ValueError('invalid_normalized_text_embedding')
                duplicate_delta = float((embeddings[-2] - embeddings[-1]).abs().max().item())
                text_geometry = {family: float((embeddings[2*i] @ embeddings[2*i+1]).item()) for i, family in enumerate(FAMILIES)}
                for row in rows:
                    original = require_inside(source / 'inputs' / (row['case_id'] + '.bmp'), source, must_exist=True)
                    if sha256_file(original) != row['image_sha256']:
                        raise ValueError('fixed_image_changed')
                    png = image_root / (row['case_id'] + '.png')
                    lossless_png(original, png)
                    vector = image_engine.get_projected_global_embedding(png)
                    if tuple(vector.shape) != (128,) or not torch.isfinite(vector).all() or not torch.allclose(
                            vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4):
                        raise ValueError('invalid_normalized_image_embedding')
                    values = (embeddings[:6] @ vector).detach().float().cpu().tolist()
                    record = {'case_id': row['case_id'], 'source_image_sha256': row['image_sha256'],
                              'png_sha256': sha256_file(png),
                              'score_pairs': {family: {'positive_cosine': round(values[2*i], 8),
                                                      'negative_cosine': round(values[2*i+1], 8)} for i, family in enumerate(FAMILIES)}}
                    margins(record)
                    records.append(record)
                    if len(replay) < 2:
                        repeated = image_engine.get_projected_global_embedding(png)
                        replay.append({'case_id': row['case_id'], 'max_abs_score_delta': float(((embeddings[:6] @ repeated) - (embeddings[:6] @ vector)).abs().max().item())})
                    if sha256_file(original) != row['image_sha256'] or sha256_file(png) != record['png_sha256']:
                        raise ValueError('image_changed_during_scoring')
            torch.cuda.synchronize()
        # References and XRV predictions are joined only after label-blind encodings.
        cohort = read_json(cohort_path)
        references = {row['case_id']: row['reference_state'] for row in cohort['records']}
        if len(references) != 50 or Counter(references.values()) != {'positive': 25, 'negative': 25}:
            raise ValueError('fixed_reference_denominator_changed')
        if any(row['reference_state'] != references[row['case_id']] for row in rows):
            raise ValueError('reference_lineage_mismatch')
        if (runtime_metadata() != plan['runtime_metadata'] or source_receipt(source) != source
                or sha256_file(cohort_path) != COHORT_SHA
                or any((path.stat().st_size, path.stat().st_mtime_ns) != asset_stats[path] for path in assets)):
            raise ValueError('source_or_runtime_changed_during_run')
        summary = {'schema_version': SCHEMA, 'status': 'completed_same_cohort_polarity_diagnostic',
            'templates': summarize(records, references),
            'cross_scorer': scorer_disagreement(records, references, {row['case_id']: row['pneumonia_score'] for row in rows}),
            'primary_reduction': MEAN, 'reference_kind': 'published_pneumonia_vs_paper_described_normal_cohort_classes',
            'frozen': True, 'independent_clinical_adjudication': False, 'training_overlap_verified': False,
            'age_population_verified': False, 'patient_grouping_verified': False,
            'primary_metric_eligible': False, 'probability_semantics': False, 'thresholds_fitted': False,
            'template_choice_fitted': False, 'selection_changed': False, 'regeneration_authorized': False,
            'raw_ehr_or_report_text_read': False, 'model_received_reference_labels': False,
            'execution': {'image_encoder_calls': 52, 'text_inputs_including_duplicate': 8,
                          'text_encoder_batches': 1, 'source_cases': 50, 'lossless_png_checked': 50, 'generation_calls': 0},
            'duplicate_prompt_max_abs_embedding_delta': duplicate_delta,
            'text_geometry_positive_negative_cosine': text_geometry, 'first_two_image_replay': replay,
            'elapsed_seconds': round(time.monotonic() - started, 6),
            'peak_allocated_vram_including_load_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            'producer': {'model_id': 'biovil_t', 'image_checkpoint_sha256': IMAGE_SHA, 'text_checkpoint_sha256': TEXT_SHA}}
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'scores.json', {'schema_version': SCHEMA, 'records': records})
        write_private_json(temporary / 'prompts.json', {'authored_probes': catalog()})
        stream = io.StringIO()
        fields = ('case_id', 'source_image_sha256', 'png_sha256', *FAMILIES, MEAN)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**{key: row[key] for key in fields[:3]}, **margins(row)} for row in records)
        write_private_text(temporary / 'score_table.csv', stream.getvalue())
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'run_id': args.run_id, 'program_sha256': sha256_file(__file__),
            'plan_manifest_sha256': args.plan_manifest_sha256, 'source_manifest_sha256': XRV_MANIFEST_SHA,
            'cohort_sha256': COHORT_SHA, 'runtime_metadata': plan['runtime_metadata'],
            'artifacts': {name: sha256_file(temporary / name) for name in ('summary.json', 'scores.json', 'prompts.json', 'score_table.csv')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = modes.add_parser('prepare')
    prepare_parser.add_argument('--source-run', required=True)
    prepare_parser.add_argument('--cohort', required=True)
    run_parser = modes.add_parser('run')
    run_parser.add_argument('--plan-run', required=True)
    run_parser.add_argument('--plan-manifest-sha256', required=True)
    run_parser.add_argument('--allow-rsua-pixels', action='store_true')
    for mode in (prepare_parser, run_parser):
        mode.add_argument('--output-root', required=True)
        mode.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args) if args.mode == 'prepare' else run(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'plan_sealed' if args.mode == 'prepare' else 'completed',
                      'manifest_sha256': sha256_file(result / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

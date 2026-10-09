#!/usr/bin/env python3
"""Header-only preparation; real display + frozen XRV ONLY after Slurm approval."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
import warnings

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
sys.path[:0] = [str(WORKSPACE / 'TriCompose-v1.2/real_validation'),
               str(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1')]
import acquire_ricord_images as inputs
import ricord_display as display
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       private_directory, require_inside, sha256_file,
                       write_private_json, write_private_text)
from extract_cxr_labels_xrv import FrozenXRVRuntime, XRV_LABELS
from xrv_calibration import SCORE_SPACE, validate_bundle
from biovil_matched_pairs import auc_ap

ROOT = inputs.annotation.PROTECTED / 'tricompose_v1_2'
INPUT_ROOT = inputs.DEFAULT_ROOT / 'images_12666569_001'
INPUT_SHA = 'de8e303594a3dba71da36185b4272d516b577a0bb5ed567720fdf517b6b95284'
PLAN_INPUT_SHA = 'b88b46f9e1b9c8620432f868325d6e6f48a0435ab7ef2356cd0e3bd0f6012142'
WEIGHT_DIR = WORKSPACE / 'artifacts/protected/tricompose_v1_1/evaluation/smoke48/scoring_12257177/cache/xrv'
WEIGHT_NAME = 'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
WEIGHT_SHA = '56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899'
THRESHOLDS = ROOT / 'real_validation/real_xrv_thresholds_12549079/thresholds.json'
THRESHOLDS_SHA = '630c25c4f3dc0446e3ed4fe02806e0a761f498a518f70af19546bc71c62d6be4'
EXTERNAL_SITE = Path('/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/lib/python3.11/site-packages')
PLAN_SCHEMA = 'tricompose-ricord-xrv50-plan-v1'
SCHEMA = 'tricompose-ricord-xrv50-diagnostic-v1'
PROTOCOL = WORKSPACE / 'docs/ricord_xrv50_protocol.md'
TESTS = WORKSPACE / 'TriCompose-v1.2/tests/test_ricord_xrv50.py'
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2000, 20261005


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_pins():
    files = (Path(__file__), Path(display.__file__), Path(inputs.__file__), TESTS, PROTOCOL,
             WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
             WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/extract_cxr_labels_xrv.py',
             WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/xrv_calibration.py',
             WORKSPACE / 'TriCompose-v1.2/real_validation/biovil_matched_pairs.py',
             WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py')
    return {str(p.relative_to(WORKSPACE)): sha256_file(p) for p in files}


def dependency_pins():
    files = []
    for root in (display.HEADER_DEPENDENCIES, display.PIXEL_DEPENDENCIES):
        files.extend(p for p in root.rglob('*') if p.is_file() and
                     (p.suffix == '.py' or '.so' in p.name or p.name in ('METADATA', 'entry_points.txt')))
    return {str(p.relative_to(WORKSPACE)): sha256_file(p) for p in sorted(files)}


def environment_versions():
    return {name: metadata.version(name) for name in (
        'numpy', 'pydicom', 'pylibjpeg', 'pylibjpeg-libjpeg', 'torch',
        'torchxrayvision', 'scikit-image', 'Pillow')}


def input_contract():
    root = require_inside(INPUT_ROOT, inputs.annotation.PROTECTED, must_exist=True)
    if sha256_file(root / 'manifest.json') != INPUT_SHA:
        raise ValueError('fixed_image_manifest_changed')
    _, manifest = inputs.verified_manifest(root, inputs.SCHEMA)
    for path, digest in manifest['header_dependency_sources'].items():
        if sha256_file(WORKSPACE / path) != digest:
            raise ValueError('original_header_parser_changed')
    report = json.loads((root / 'acquisition.json').read_bytes())
    plan_root = require_inside(WORKSPACE / report['plan_root'], inputs.annotation.PROTECTED, must_exist=True)
    if sha256_file(plan_root / 'manifest.json') != PLAN_INPUT_SHA or report['plan_manifest_sha256'] != PLAN_INPUT_SHA:
        raise ValueError('fixed_cohort_manifest_changed')
    inputs.verified_manifest(plan_root, inputs.PLAN_SCHEMA)
    cohort = json.loads((plan_root / 'plan.json').read_bytes())
    cases = report['cases']
    if report['acquired_images'] != 50 or report['failed_images'] != 0 or len(cases) != 50 or \
            len({c['case_id'] for c in cases}) != 50 or \
            Counter(c['lung_opacity_reference'] for c in cases) != Counter({'positive': 25, 'negative': 25}):
        raise ValueError('complete_fixed_balanced_cohort_required')
    if [c['case_id'] for c in cases] != [c['case_id'] for c in cohort['cases']]:
        raise ValueError('cohort_order_changed')
    return root, report, plan_root, cohort


def scorer_metadata():
    if sha256_file(WEIGHT_DIR / WEIGHT_NAME) != WEIGHT_SHA or sha256_file(THRESHOLDS) != THRESHOLDS_SHA:
        raise ValueError('unchanged_model_or_threshold_hash_mismatch')
    bundle = json.loads(THRESHOLDS.read_bytes())
    heads = validate_bundle(bundle, checkpoint_sha256=WEIGHT_SHA,
        expected_provenance={'score_space': SCORE_SPACE, 'finding_mapping_sha256': fingerprint(XRV_LABELS)})
    legacy = heads['lung_opacity']
    if legacy['enabled'] is not True or legacy['positive_min'] != legacy['negative_max'] or \
            legacy['positive_min'] != 0.642907085:
        raise ValueError('legacy_threshold_transport_contract_changed')
    sources = {str(EXTERNAL_SITE / 'torchxrayvision' / name): sha256_file(EXTERNAL_SITE / 'torchxrayvision' / name)
               for name in ('models.py', 'datasets.py')}
    return {'checkpoint_sha256': WEIGHT_SHA, 'threshold_bundle_sha256': THRESHOLDS_SHA,
            'legacy_max_proxy_threshold': legacy['positive_min'], 'exact_head_threshold': 0.5,
            'legacy_threshold_preprocessing_sha256': bundle['provenance']['preprocessing_sha256'],
            'external_readonly_sources': sources, 'score_space': SCORE_SPACE,
            'probability_semantics': False, 'legacy_threshold_transport_validated': False}


def prepare(args):
    _, pd = display.dependencies()
    source, report, plan_root, cohort = input_contract()
    scorer = scorer_metadata()
    temporary, target = new_atomic_run(args.output_root or ROOT / 'real_validation/ricord_xrv_plans', args.run_id)
    try:
        inventory = json.loads((plan_root / 'raw_series_inventory.json').read_bytes())['rows']
        sops = json.loads((plan_root / 'raw_sop_inventory.json').read_bytes())['responses']
        header_counts, cases = Counter(), []
        logging.getLogger('pydicom').disabled = True
        for case, binding in zip(report['cases'], cohort['cases'], strict=True):
            path = require_inside(source / case['artifact'], source, must_exist=True)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                ds = pd.dcmread(path, stop_before_pixels=True, specific_tags=list(inputs.HEADER_TAGS) +
                    ['VOILUTFunction', 'PresentationLUTSequence', 'PixelPaddingValue', 'PixelPaddingRangeLimit'])
                inputs.header_contract(ds, inventory[binding['series_inventory_index']],
                                       sops[binding['sop_inventory_index']][0]['SOPInstanceUID'])
                presentation = display.presentation_contract(ds)
            header_counts[presentation['voi_transform']] += 1
            header_counts[presentation['polarity_source']] += 1
            cases.append({'case_id': case['case_id'], 'artifact': case['artifact'],
                          'sha256': case['sha256'], 'lung_opacity_reference': case['lung_opacity_reference'],
                          'header_presentation': presentation})
        plan = {'schema_version': PLAN_SCHEMA, 'image_root': str(source.relative_to(WORKSPACE)),
                'image_manifest_sha256': INPUT_SHA, 'cohort_manifest_sha256': PLAN_INPUT_SHA,
                'case_count': 50, 'cases': cases, 'display_policy': display.POLICY,
                'display_policy_sha256': fingerprint(display.POLICY), 'scorer': scorer,
                'environment_versions': environment_versions(), 'header_counts': dict(header_counts),
                'bootstrap_count': BOOTSTRAP_COUNT, 'bootstrap_seed': BOOTSTRAP_SEED,
                'clinical_pneumonia_reference': False, 'primary_metric_eligible': False,
                'thresholds_fitted_on_this_cohort': False, 'pixels_decoded': 0, 'model_calls': 0}
        write_private_json(temporary / 'plan.json', plan)
        write_private_json(temporary / 'manifest.json', {'schema_version': PLAN_SCHEMA + '-manifest',
            'run_id': args.run_id, 'sources': source_pins(), 'dependency_sources': dependency_pins(),
            'artifacts': {'plan.json': sha256_file(temporary / 'plan.json')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'display_and_scorer_plan_frozen_no_inference', 'cases': 50,
                    'header_counts': dict(header_counts), 'model_calls': 0, 'pixels_decoded': 0}


def wilson(successes, denominator):
    if denominator == 0:
        return None
    z = 1.959963984540054
    p = successes / denominator
    factor = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / factor
    margin = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator * denominator)) / factor
    return [round(max(0, center - margin), 8), round(min(1, center + margin), 8)]


def quantile(values, q):
    ordered = sorted(values)
    location = q * (len(ordered) - 1)
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def binary_metrics(labels, scores, threshold):
    if not labels or len(labels) != len(scores) or any(type(y) is not int or y not in (0, 1) for y in labels) or \
            any(isinstance(s, bool) or not isinstance(s, (float, int)) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores) or \
            isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid_diagnostic_labels_scores_threshold')
    positive, negative = sum(labels), len(labels) - sum(labels)
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= threshold for y, s in zip(labels, scores))
    tn, fn = negative - fp, positive - tp
    sensitivity = tp / positive if positive else None
    specificity = tn / negative if negative else None
    return {**auc_ap(labels, scores), 'threshold': threshold, 'scored_cases': len(labels),
            'positive': positive, 'negative': negative, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': sensitivity, 'specificity': specificity,
            'sensitivity_wilson95': wilson(tp, positive), 'specificity_wilson95': wilson(tn, negative),
            'balanced_accuracy': (sensitivity + specificity) / 2 if positive and negative else None}


def bootstrap(labels, scores, count=BOOTSTRAP_COUNT, seed=BOOTSTRAP_SEED):
    binary_metrics(labels, scores, 0.5)
    positive = [s for y, s in zip(labels, scores) if y == 1]
    negative = [s for y, s in zip(labels, scores) if y == 0]
    if not positive or not negative:
        return None
    rng, aucs, aps = random.Random(seed), [], []
    for _ in range(count):
        sample = [rng.choice(positive) for _ in positive] + [rng.choice(negative) for _ in negative]
        result = auc_ap([1] * len(positive) + [0] * len(negative), sample)
        aucs.append(result['auroc']); aps.append(result['average_precision'])
    return {'replicates': count, 'seed': seed, 'unit': 'patient_one_image_class_stratified',
            'auroc_percentile95': [round(quantile(aucs, q), 8) for q in (0.025, 0.975)],
            'average_precision_percentile95': [round(quantile(aps, q), 8) for q in (0.025, 0.975)]}


def guard(approved):
    if approved is not True or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('explicit_ricord_pixel_gpu_slurm_approval_required')


def evaluate(args):
    # Before input lookup, model imports or real pixel decoding.
    guard(args.allow_ricord_pixels_and_xrv)
    plan_root = require_inside(args.plan_root, inputs.annotation.PROTECTED, must_exist=True)
    if sha256_file(plan_root / 'manifest.json') != args.plan_manifest_sha256:
        raise ValueError('display_scorer_plan_manifest_changed')
    _, manifest = inputs.verified_manifest(plan_root, PLAN_SCHEMA)
    for path, digest in manifest['dependency_sources'].items():
        if sha256_file(WORKSPACE / path) != digest:
            raise ValueError('frozen_decoder_dependency_changed')
    plan = json.loads((plan_root / 'plan.json').read_bytes())
    np, pd = display.dependencies()
    if plan['display_policy'] != display.POLICY or plan['environment_versions'] != environment_versions() or \
            plan['scorer'] != scorer_metadata() or plan['bootstrap_count'] != BOOTSTRAP_COUNT or \
            plan['bootstrap_seed'] != BOOTSTRAP_SEED:
        raise ValueError('frozen_evaluation_contract_changed')
    for path, digest in plan['scorer']['external_readonly_sources'].items():
        if sha256_file(path) != digest:
            raise ValueError('external_frozen_xrv_source_changed')
    source, acquisition, source_plan_root, cohort = input_contract()
    if [(c['case_id'], c['sha256'], c['lung_opacity_reference']) for c in acquisition['cases']] != \
            [(c['case_id'], c['sha256'], c['lung_opacity_reference']) for c in plan['cases']]:
        raise ValueError('case_binding_changed')
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('allocated_cuda_gpu_unavailable')
    temporary, target = new_atomic_run(args.output_root or ROOT / 'real_validation/ricord_xrv_pilots', args.run_id)
    started, outcomes = time.monotonic(), []
    inference_attempts, decoded_images = 0, 0
    records, artifacts = [], {}
    try:
        private_directory(temporary / 'displays')
        inventory = json.loads((source_plan_root / 'raw_series_inventory.json').read_bytes())['rows']
        sops = json.loads((source_plan_root / 'raw_sop_inventory.json').read_bytes())['responses']
        logging.getLogger('pydicom').disabled = True
        torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            runtime = FrozenXRVRuntime(cache_dir=WEIGHT_DIR, weight_filename=WEIGHT_NAME,
                                       model_name='densenet121-res224-all')
            if runtime.model.training or any(p.requires_grad for p in runtime.model.parameters()):
                raise RuntimeError('scorer_not_frozen')
            for case, binding in zip(plan['cases'], cohort['cases'], strict=True):
                outcome = {'case_id': case['case_id'], 'status': 'failed_without_replacement', 'model_calls': 0}
                try:
                    path = require_inside(source / case['artifact'], source, must_exist=True)
                    if sha256_file(path) != case['sha256']:
                        raise ValueError('case_image_hash_changed')
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        ds = pd.dcmread(path, force=False)
                        inputs.header_contract(ds, inventory[binding['series_inventory_index']],
                            sops[binding['sop_inventory_index']][0]['SOPInstanceUID'])
                        contract = display.presentation_contract(ds)
                        if contract != case['header_presentation']:
                            raise ValueError('frozen_display_metadata_changed')
                        raw = pd.pixels.pixel_array(ds, decoding_plugin='pylibjpeg' if contract['decoder'] == 'pylibjpeg' else '')
                        decoded_images += 1
                        rendered, rendering = display.render_array(raw, ds)
                    output = temporary / 'displays' / (case['case_id'] + '.png')
                    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
                    with os.fdopen(fd, 'wb') as handle:
                        Image.fromarray(rendered).save(handle, format='PNG')
                    os.chmod(output, 0o660)
                    image_hash = sha256_file(output)
                    artifacts[str(output.relative_to(temporary))] = image_hash
                    inference_attempts += 1
                    outcome['model_calls'] = 1
                    scores = runtime.predict(output)
                    opacity, infiltration = scores.get('lung_opacity'), scores.get('infiltration')
                    if opacity is None or infiltration is None or any(not math.isfinite(s) or not 0 <= s <= 1 for s in (opacity, infiltration)):
                        raise ValueError('exact_opacity_or_legacy_proxy_head_unavailable')
                    records.append({'case_id': case['case_id'], 'reference_state': case['lung_opacity_reference'],
                        'dicom_sha256': case['sha256'], 'display_sha256': image_hash,
                        'direct_lung_opacity_score': opacity, 'infiltration_score': infiltration,
                        'legacy_max_proxy_score': max(opacity, infiltration), 'rendering': rendering})
                    outcome.update(status='scored', model_calls=1)
                except Exception as error:
                    outcome.update(error_type=type(error).__name__)
                outcomes.append(outcome)
            torch.cuda.synchronize()
        labels = [int(r['reference_state'] == 'positive') for r in records]
        direct, legacy = [r['direct_lung_opacity_score'] for r in records], [r['legacy_max_proxy_score'] for r in records]
        threshold = plan['scorer']['legacy_max_proxy_threshold']
        metrics = {'direct_lung_opacity_default': binary_metrics(labels, direct, 0.5),
                   'direct_lung_opacity_bootstrap': bootstrap(labels, direct),
                   'secondary_legacy_max_proxy_default': binary_metrics(labels, legacy, 0.5),
                   'secondary_legacy_max_proxy_weak_threshold_transport': binary_metrics(labels, legacy, threshold)} if records else None
        summary = {'schema_version': SCHEMA,
            'status': 'completed_50_case_opacity_diagnostic_unvalidated' if len(records) == 50 else 'incomplete_original_denominator_retained',
            'plan_manifest_sha256': args.plan_manifest_sha256, 'image_manifest_sha256': INPUT_SHA,
            'planned_cases': 50, 'scored_cases': len(records), 'failed_cases': 50 - len(records),
            'coverage': len(records) / 50, 'metrics_conditional_on_scored_cases': len(records) != 50,
            'scorer': plan['scorer'], 'display_policy': plan['display_policy'],
            'display_policy_sha256': plan['display_policy_sha256'], 'frozen': True,
            'reference_kind': acquisition['reference_kind'], 'patient_grouping_verified': True,
            'clinical_pneumonia_reference': False, 'official_adjudication_reproduced': False,
            'checkpoint_training_overlap_verified': False, 'primary_metric_eligible': False,
            'display_transform_clinically_verified': False, 'thresholds_fitted_on_this_cohort': False,
            'training_calls': 0, 'generation_calls': 0, 'model_calls': inference_attempts,
            'successfully_decoded_images': decoded_images,
            'probability_semantics': False, 'legacy_threshold_transport_validated': False,
            'metrics': metrics, 'elapsed_seconds': round(time.monotonic() - started, 6),
            'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)}
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'scores.json', {'schema_version': SCHEMA + '-scores', 'records': records, 'outcomes': outcomes})
        write_private_text(temporary / 'RESULTS_CN_EN.md', '# RICORD-1C opacity diagnostic / 肺部阴影诊断\n\n'
            'Not pneumonia diagnosis, clinical repair validation, probability calibration or paper-final evaluation.\n\n'
            + json.dumps(summary, indent=2, sort_keys=True) + '\n')
        artifacts.update({name: sha256_file(temporary / name) for name in ('summary.json', 'scores.json', 'RESULTS_CN_EN.md')})
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'run_id': args.run_id, 'input_plan_manifest_sha256': args.plan_manifest_sha256,
            'sources': source_pins(), 'dependency_sources': dependency_pins(), 'artifacts': artifacts})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {k: summary[k] for k in ('status', 'planned_cases', 'scored_cases', 'failed_cases',
                                          'model_calls', 'elapsed_seconds', 'peak_vram_gib')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('--plan-root')
    parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--output-root')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-ricord-pixels-and-xrv', action='store_true')
    args = parser.parse_args(argv)
    try:
        target, result = prepare(args) if args.action == 'prepare' else evaluate(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    result.update(output_root=str(target.relative_to(WORKSPACE)), manifest_sha256=sha256_file(target / 'manifest.json'))
    print(json.dumps(result))
    return 0 if args.action == 'prepare' or result['scored_cases'] == 50 else 2


if __name__ == '__main__':
    raise SystemExit(main())

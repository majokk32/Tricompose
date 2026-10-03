#!/usr/bin/env python3
"""Frozen, label-blind single-fact polarity probe on the existing real cohort.

Only an explicitly approved GPU Slurm entry point may read real image inputs.
Report text, EHR fields, patient keys, source paths and reference rows are never
exported. References are cached report-derived weak labels, NOT image truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
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
from biovil_matched_pairs import (IMAGE_WEIGHT, PROJECT, _load_runtime,
                                  _resolve_source, auc_ap, select_cases)

SCHEMA = 'tricompose-real-biovil-fact-polarity-v1'
METHOD = 'fixed-eight-findings-three-polarity-templates-v1'
FINDINGS = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Lung Opacity', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax')
STATES = ('positive', 'negative', 'uncertain', 'unknown')
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
PREVIOUS_SHA = '762009d2e41408b0ae43de6d49487162bd249f678981120bcaf77784d4a41c36'
REFERENCE_SHA = 'cbcaa11fa41e6ede45dd3ece70f07d7fcaad1bd0af6bdfe5c45a762b72210219'
MANIFEST_SHA = '5e3a07b441faa6c4c057fa3d217100deba492aa494c16133e541b9ed25c711cb'
VENDOR_TREE_SHA = 'dceafc8b318bc478ee67435d9206cd98d7d8abff801fa0ce8c2685ae840a8992'
MODEL_METADATA_SHAS = {
    'config.json': 'e18c4d0656ad72395dd6f5e3827aa8b5c7bd9c9045318d10c3d130a84aadab8b',
    'configuration_cxrbert.py': '857366a973291d09ea8e19cfe749fc801e81fae9ae9ce0c9b462c7c26cb67e1f',
    'modeling_cxrbert.py': 'f454c3afbd457ba0cc8cfff08944f0d205c9ca01793535b551360a069477e52e',
    'special_tokens_map.json': '303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3',
    'tokenizer_config.json': '971d7e13339bf707b96ed1b2196371a84bf66f6cafcc9ac55feadd44d3031fd9',
    'vocab.txt': 'e025f479c3a7ce6e248971bfe1b409708db0439e9488d30e254f81f27f75d678',
}
SAFE_CODES = frozenset({
    'explicit_image_polarity_slurm_required', 'slurm_cgroup_required',
    'cuda_required', 'invalid_probe_options', 'cached_source_binding_changed',
    'cached_source_schema_changed', 'cached_cohort_binding_changed',
    'fixed_source_manifest_changed', 'fixed_cohort_membership_changed',
    'fixed_cohort_requires_128_per_split', 'frozen_checkpoint_changed',
    'source_image_changed', 'source_changed_during_run', 'model_not_frozen',
    'invalid_embedding_dimensions', 'invalid_embedding_norm',
    'invalid_score_pairs', 'invalid_reference_inventory', 'invalid_score_inventory',
    'invalid_cosine_value', 'invalid_reference_state', 'bounded_linkage_required',
    'linkage_schema_changed', 'output_record_contains_private_fields'})


def probes():
    """Authored generic text; same concept, no severity/view/history additions."""
    result = []
    for index, finding in enumerate(FINDINGS):
        name = finding.lower()
        pairs = (
            (f'The chest X-ray shows {name}.', f'The chest X-ray shows no {name}.'),
            (f'There is evidence of {name}.', f'There is no evidence of {name}.'),
            (f'{name.capitalize()} is present.', f'{name.capitalize()} is absent.'),
        )
        for family, (positive, negative) in zip(FAMILIES, pairs, strict=True):
            result.append({'probe_id': f'probe_{index:02d}_{family}',
                           'finding': finding, 'family': family,
                           'positive_text': positive, 'negative_text': negative})
    return result


def require_approved_slurm(args):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not getattr(args, 'allow_real_image_polarity', False) or not job.isdigit():
        raise RuntimeError('explicit_image_polarity_slurm_required')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('slurm_cgroup_required')


def linkage_rows(path):
    """Keep linkage columns only; do not interpret EHR/report contents."""
    columns = {'subject_id', 'study_id', 'split', 'ViewPosition', 'cxr_path', 'report_path'}
    rows = []
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not columns.issubset(reader.fieldnames):
            raise ValueError('linkage_schema_changed')
        for row in reader:
            if len(rows) >= 1000000:
                raise ValueError('bounded_linkage_required')
            if None in row or any(row.get(key) is None for key in columns):
                raise ValueError('linkage_schema_changed')
            rows.append({key: row[key] for key in columns})
    return rows


def validate_cached_cohort(previous, reference):
    if (previous.get('schema_version') != 'tricompose-real-biovil-pair-check-v1'
            or reference.get('schema_version') != 'tricompose-real-xrv-weak-finding-check-v1'):
        raise ValueError('cached_source_schema_changed')
    if (previous.get('source_manifest_sha256') != MANIFEST_SHA
            or reference.get('source_manifest_sha256') != MANIFEST_SHA
            or reference.get('previous_scores_sha256') != PREVIOUS_SHA):
        raise ValueError('cached_cohort_binding_changed')
    if previous.get('producer', {}).get('frozen') is not True:
        raise ValueError('model_not_frozen')
    first = {row['case_id']: row for row in previous['records']}
    second = {row['case_id']: row for row in reference['records']}
    if (len(first) != len(previous['records']) or len(second) != len(reference['records'])
            or set(first) != set(second)):
        raise ValueError('fixed_cohort_membership_changed')
    for case_id, row in first.items():
        other = second[case_id]
        if any(row[key] != other[key] for key in ('split', 'source_row_index', 'image_sha256')):
            raise ValueError('fixed_cohort_membership_changed')
        states = other['reference_states']
        if set(states) != set(FINDINGS) or any(value not in STATES for value in states.values()):
            raise ValueError('invalid_reference_state')
    return first, second


def _cosine(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1.00001:
        raise ValueError('invalid_cosine_value')
    return float(value)


def summarize(records, references, *, min_per_class=10):
    """Reference is used AFTER label-blind scoring; no fitting or probability."""
    if type(min_per_class) is not int or min_per_class < 1:
        raise ValueError('invalid_probe_options')
    ids = [row['case_id'] for row in records]
    if len(set(ids)) != len(ids) or set(ids) != set(references):
        raise ValueError('invalid_reference_inventory')
    catalog = probes()
    probe_ids = {row['probe_id'] for row in catalog}
    for row in records:
        if set(row) != {'case_id', 'split', 'image_sha256', 'score_pairs'}:
            raise ValueError('output_record_contains_private_fields')
        if row['split'] not in {'val', 'test'} or set(row['score_pairs']) != probe_ids:
            raise ValueError('invalid_score_inventory')
        states = references[row['case_id']]
        if set(states) != set(FINDINGS) or any(value not in STATES for value in states.values()):
            raise ValueError('invalid_reference_state')
        for pair in row['score_pairs'].values():
            if set(pair) != {'positive_cosine', 'negative_cosine'}:
                raise ValueError('invalid_score_pairs')
            _cosine(pair['positive_cosine']); _cosine(pair['negative_cosine'])
    result = {}
    for split in ('val', 'test'):
        subset = [row for row in records if row['split'] == split]
        finding_results = {}
        for finding in FINDINGS:
            counts = Counter(references[row['case_id']][finding] for row in subset)
            known = [row for row in subset if references[row['case_id']][finding] in {'positive', 'negative'}]
            templates = {}
            for family in (*FAMILIES, 'mean_all_three_predeclared'):
                selected = [p for p in catalog if p['finding'] == finding
                            and (family == 'mean_all_three_predeclared' or p['family'] == family)]
                margins = []
                labels = []
                for row in known:
                    margins.append(sum(_cosine(row['score_pairs'][p['probe_id']]['positive_cosine'])
                                       - _cosine(row['score_pairs'][p['probe_id']]['negative_cosine'])
                                       for p in selected) / len(selected))
                    labels.append(int(references[row['case_id']][finding] == 'positive'))
                pos = [margin for margin, label in zip(margins, labels) if label == 1]
                neg = [margin for margin, label in zip(margins, labels) if label == 0]
                eligible = min(len(pos), len(neg)) >= min_per_class
                metrics = auc_ap(labels, margins) if eligible else {'auroc': None, 'average_precision': None}
                win = sum((margin > 0 if label else margin < 0) for margin, label in zip(margins, labels))
                pos_win = sum(m > 0 for m in pos) / len(pos) if pos else None
                neg_win = sum(m < 0 for m in neg) / len(neg) if neg else None
                templates[family] = {
                    'known_reference_checks': len(known), 'positive': len(pos), 'negative': len(neg),
                    'status': 'weak_reference_margin_diagnostic' if eligible else 'insufficient_binary_support',
                    'reference_polarity_win_rate': win / len(known) if known else None,
                    'positive_reference_win_rate': pos_win, 'negative_reference_win_rate': neg_win,
                    'balanced_polarity_win_rate': (pos_win + neg_win) / 2 if pos and neg else None,
                    'tie_rate': sum(m == 0 for m in margins) / len(known) if known else None,
                    'mean_positive_minus_negative_cosine': sum(margins) / len(margins) if margins else None,
                    **metrics}
            finding_results[finding] = {
                'source_state_counts': {state: counts[state] for state in STATES},
                'full_case_denominator': len(subset), 'known_reference_checks': len(known),
                'excluded_unknown': counts['unknown'], 'excluded_uncertain': counts['uncertain'],
                'known_reference_coverage': len(known) / len(subset) if subset else None,
                'templates': templates}
        result[split] = {'cases': len(subset), 'min_per_class_for_auc': min_per_class,
                         'findings': finding_results}
    return result


def markdown(summary):
    lines = ['# BioViL-T 单事实正负极性探针 / Single-fact polarity probe', '',
             'Report-derived weak references; NOT independently adjudicated image truth.',
             'All templates and the mean-of-three reduction were frozen before scores.',
             'No training, threshold fitting, template selection or winner change.', '']
    show = lambda value: 'NA' if value is None else f'{value:.4f}'
    for split, data in summary['summary'].items():
        lines += [f'## {split}', '',
                  '| Finding | Known / cases | Positive / negative | Polarity win | Balanced win | Margin AUROC |',
                  '|---|---:|---:|---:|---:|---:|']
        for finding, row in data['findings'].items():
            metrics = row['templates']['mean_all_three_predeclared']
            lines.append(f"| {finding} | {row['known_reference_checks']} / {row['full_case_denominator']} | "
                         f"{metrics['positive']} / {metrics['negative']} | {show(metrics['reference_polarity_win_rate'])} | "
                         f"{show(metrics['balanced_polarity_win_rate'])} | {show(metrics['auroc'])} |")
        lines.append('')
    lines += ['Each template, class-conditional win rate, ties and unknown/uncertain denominators are in summary.json.',
              'Retrieval performance alone does not validate sensitivity to clinical negation.',
              'No image-side clinical correctness, EHR consistency or localization gate is cleared.', '']
    return '\n'.join(lines)


def run(args):
    require_approved_slurm(args)
    if (not 1 <= args.text_batch_size <= 16 or not 0 <= args.replay_count <= 8
            or args.min_per_class < 1):
        raise ValueError('invalid_probe_options')
    with open(os.devnull, 'w') as silent, contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        import torch
    if not torch.cuda.is_available():
        raise RuntimeError('cuda_required')
    dataset = Path(args.dataset_root).resolve(strict=True)
    if not dataset.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError('fixed_source_manifest_changed')
    previous_path = require_inside(args.previous_scores, PROTECTED_ROOT, must_exist=True)
    reference_path = require_inside(args.reference_scores, PROTECTED_ROOT, must_exist=True)
    if sha256_file(previous_path) != PREVIOUS_SHA or sha256_file(reference_path) != REFERENCE_SHA:
        raise ValueError('cached_source_binding_changed')
    previous, reference = read_json(previous_path), read_json(reference_path)
    fixed, cached = validate_cached_cohort(previous, reference)
    manifest = dataset / 'manifest.csv'
    if sha256_file(manifest) != MANIFEST_SHA:
        raise ValueError('fixed_source_manifest_changed')
    params = previous['selection']
    if params['count_per_split'] != 128:
        raise ValueError('fixed_cohort_requires_128_per_split')
    selected = select_cases(linkage_rows(manifest), count_per_split=128, seed=params['seed'])
    if len(selected) != len(fixed) or any(
            row['case_id'] not in fixed or fixed[row['case_id']]['source_row_index'] != row['row_index']
            or fixed[row['case_id']]['split'] != row['split'] for row in selected):
        raise ValueError('fixed_cohort_membership_changed')
    model = require_inside(args.model_path, WORKSPACE, must_exist=True)
    assets = {model / IMAGE_WEIGHT: previous['producer']['image_checkpoint_sha256'],
              model / 'pytorch_model.bin': previous['producer']['text_checkpoint_sha256']}
    if any(sha256_file(path) != expected for path, expected in assets.items()):
        raise ValueError('frozen_checkpoint_changed')
    metadata = {path: sha256_file(path) for path in model.iterdir()
                if path.suffix in {'.json', '.txt', '.py'}}
    vendor = WORKSPACE / 'runtime/vendor/hi_ml_multimodal_0_2_2/health_multimodal'
    vendor_files = {path: sha256_file(path) for path in vendor.rglob('*.py')}
    vendor_digest = hashlib.sha256(json.dumps(
        {str(path.relative_to(vendor)): value for path, value in vendor_files.items()},
        sort_keys=True).encode()).hexdigest()
    if (vendor_digest != VENDOR_TREE_SHA
            or any(metadata.get(model / name) != value for name, value in MODEL_METADATA_SHAS.items())):
        raise ValueError('frozen_checkpoint_changed')
    metadata.update(vendor_files)
    metadata.update({WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py':
                     sha256_file(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py')})
    asset_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in assets}
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    catalog = probes()
    prompts = [p[key] for p in catalog for key in ('positive_text', 'negative_text')]
    # Two identical authored prompts, in one batch, diagnose duplicate-input invariance.
    text_inputs = prompts + [prompts[0], prompts[0]]
    records, replay_records = [], []
    with open(os.devnull, 'w') as silent, contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        image_engine, text_engine = _load_runtime(model, torch.device('cuda:0'))
        if any(engine.model.training or any(p.requires_grad for p in engine.model.parameters())
               for engine in (image_engine, text_engine)):
            raise RuntimeError('model_not_frozen')
        with torch.inference_mode():
            embeddings = torch.cat([text_engine.get_embeddings_from_prompt(
                text_inputs[start:start + args.text_batch_size], normalize=True, verbose=False)
                for start in range(0, len(text_inputs), args.text_batch_size)])
            if tuple(embeddings.shape) != (50, 128):
                raise ValueError('invalid_embedding_dimensions')
            if not torch.isfinite(embeddings).all() or not torch.allclose(
                    embeddings.norm(dim=1), torch.ones(50, device=embeddings.device), atol=1e-4, rtol=1e-4):
                raise ValueError('invalid_embedding_norm')
            duplicate_delta = float((embeddings[-2] - embeddings[-1]).abs().max().item())
            text_geometry = {p['probe_id']: float((embeddings[2*i] @ embeddings[2*i+1]).item())
                             for i, p in enumerate(catalog)}
            for row in selected:
                path = _resolve_source(row['cxr_path'], dataset)
                image_sha = sha256_file(path)
                if image_sha != fixed[row['case_id']]['image_sha256']:
                    raise ValueError('source_image_changed')
                image_embedding = image_engine.get_projected_global_embedding(path)
                if tuple(image_embedding.shape) != (128,) or not torch.isfinite(image_embedding).all():
                    raise ValueError('invalid_embedding_dimensions')
                if not torch.allclose(image_embedding.norm(), torch.ones((), device=image_embedding.device), atol=1e-4, rtol=1e-4):
                    raise ValueError('invalid_embedding_norm')
                values = (embeddings[:48] @ image_embedding).detach().float().cpu().tolist()
                if sha256_file(path) != image_sha:
                    raise ValueError('source_image_changed')
                pairs = {p['probe_id']: {'positive_cosine': _cosine(values[2*i]),
                                       'negative_cosine': _cosine(values[2*i+1])}
                         for i, p in enumerate(catalog)}
                records.append({'case_id': row['case_id'], 'split': row['split'],
                                'image_sha256': image_sha, 'score_pairs': pairs})
                if len(records) <= args.replay_count:
                    repeated = image_engine.get_projected_global_embedding(path)
                    delta = float(((embeddings[:48] @ repeated) - (embeddings[:48] @ image_embedding)).abs().max().item())
                    if sha256_file(path) != image_sha:
                        raise ValueError('source_image_changed')
                    replay_records.append({'case_id': row['case_id'], 'max_abs_score_delta': delta})
    torch.cuda.synchronize()
    if (sha256_file(previous_path) != PREVIOUS_SHA or sha256_file(reference_path) != REFERENCE_SHA
            or sha256_file(manifest) != MANIFEST_SHA
            or any(sha256_file(path) != value for path, value in metadata.items())
            or any((path.stat().st_size, path.stat().st_mtime_ns) != asset_stats[path] for path in assets)):
        raise ValueError('source_changed_during_run')
    references = {case_id: row['reference_states'] for case_id, row in cached.items()}
    result = {
        'schema_version': SCHEMA, 'method_version': METHOD,
        'status': 'completed_weak_reference_single_fact_polarity_diagnostic',
        'summary': summarize(records, references, min_per_class=args.min_per_class),
        'selection': {'fixed_previous_cohort': True, 'one_study_per_patient': True,
                      'patient_disjoint_splits_verified': True, 'label_based_case_selection': False,
                      'all_cases_retained': True, 'all_three_templates_predeclared': True,
                      'template_choice_fitted': False, 'zero_margin_is_pair_ranking_not_fitted_threshold': True},
        'source_sha256': {'previous_scores': PREVIOUS_SHA, 'reference_scores': REFERENCE_SHA, 'linkage_manifest': MANIFEST_SHA},
        'producer': {**previous['producer'], 'adapter_sha256': metadata[WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py'],
                     'model_and_vendor_metadata_sha256': hashlib.sha256(json.dumps(
                         {str(path.relative_to(WORKSPACE)): value for path, value in metadata.items()}, sort_keys=True).encode()).hexdigest()},
        'text_geometry_positive_negative_cosine': text_geometry,
        'duplicate_prompt_max_abs_embedding_delta': duplicate_delta,
        'image_replay': {'policy': 'first_source_order_inputs_not_label_selected', 'records': replay_records},
        'execution': {'gpu_name': torch.cuda.get_device_name(0), 'torch_version': str(torch.__version__),
                      'text_batch_size': args.text_batch_size, 'authored_text_inputs': 50,
                      'image_encoder_calls': len(records) + len(replay_records)},
        'peak_allocated_vram_including_load_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        'reference_kind': 'cached_report_extracted_weak_labels_not_image_ground_truth',
        'raw_source_reports_read': False, 'raw_ehr_fields_inspected': False,
        'patient_keys_written': False, 'source_paths_written': False,
        'per_row_reference_states_written': False, 'model_received_reference_labels': False,
        'image_pixels_written': False, 'thresholds_fitted': False,
        'probability_calibration_performed': False, 'primary_metric_eligible': False,
        'independent_image_ground_truth': False, 'selection_changed': False, 'regeneration_authorized': False,
        'interpretation': 'Controlled authored-text negation probe on real images. Report-derived references and possible MIMIC training overlap prevent an independent clinical claim. Unknown/uncertain references are excluded with denominators. Never select the best template or fit weights on test results.'}
    return result, records, catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'previous-scores', 'reference-scores', 'model-path', 'output-root', 'run-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--text-batch-size', type=int, default=16)
    parser.add_argument('--min-per-class', type=int, default=10)
    parser.add_argument('--replay-count', type=int, default=4)
    parser.add_argument('--allow-real-image-polarity', action='store_true')
    args = parser.parse_args(); temporary = None; started = time.monotonic(); os.umask(0o007)
    try:
        require_approved_slurm(args)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        summary, records, catalog = run(args)
        summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
        files = [write_private_json(temporary / 'summary.json', summary),
                 write_private_text(temporary / 'summary.md', markdown(summary)),
                 write_private_json(temporary / 'scores.json', {'schema_version': SCHEMA, 'records': records}),
                 write_private_json(temporary / 'prompts.json', {'schema_version': SCHEMA, 'authored_probes': catalog})]
        write_private_json(temporary / 'manifest.json', {
            'schema_version': SCHEMA, 'run_id': args.run_id, 'program_sha256': sha256_file(__file__),
            'source_sha256': summary['source_sha256'], 'producer': summary['producer'],
            'template_catalog_sha256': hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
            'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
            'patient_keys_or_source_report_text_written': False,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({'status': 'failed_single_fact_polarity_diagnostic',
                          'error_type': type(exc).__name__,
                          'error_code': str(exc) if str(exc) in SAFE_CODES else 'unclassified_failure'}))
        return 1
    print(json.dumps({'status': 'completed_single_fact_polarity_diagnostic',
                      'elapsed_seconds': summary['elapsed_seconds'],
                      'peak_allocated_vram_gib': summary['peak_allocated_vram_including_load_gib'],
                      'manifest_sha256': sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

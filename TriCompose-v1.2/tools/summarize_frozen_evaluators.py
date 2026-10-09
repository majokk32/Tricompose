#!/usr/bin/env python3
"""Cached public-reference scorecard; never a clinical repair policy."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'real_validation'))
sys.path.insert(0, str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'))
from biovil_matched_pairs import auc_ap
from contracts import (PROTECTED_ROOT, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

SCHEMA = 'tricompose-frozen-evaluator-scorecard-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2/real_validation'
COHORT = BASE / 'rsua_pilot_cohorts/cohort50_12625457_001/cohort.json'
COHORT_SHA = 'ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1'
PARENTS = {
    'xrv': (BASE / 'rsua_xrv_pilots/rsua_xrv50_12636566',
        '92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0', 'scores.json'),
    'biovil': (BASE / 'rsua_biovil_pilots/rsua_biovil50_12637081',
        'a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a', 'scores.json'),
    'siglip': (BASE / 'rsua_siglip_pilots/rsua_siglip50_12654662',
        '3c7cea5b9da2f2e05963af44f40b522c7faccbda55f06b16ba9783c0dbae6d6a', 'predictions.json'),
}
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
MEAN = 'mean_all_three_predeclared'
RANK_MODELS = ('xrv', 'biovil', 'siglip')
COMPARISONS = (('siglip', 'xrv'), ('biovil', 'xrv'), ('siglip', 'biovil'))
HASH = re.compile(r'[a-f0-9]{64}\Z')
REPLICATES, SEED = 2000, 0


def require_cpu_slurm():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('existing_cpu_slurm_required')
    if os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'):
        raise RuntimeError('cpu_metadata_analysis_only')


def finite(value, *, lower=None, upper=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (lower is not None and value < lower) or (upper is not None and value > upper)):
        raise ValueError('bounded_finite_numeric_readout_required')
    return value


def load_inputs():
    values, sources = {}, {
        'worker': Path(__file__),
        'tests': ROOT / 'tests/test_frozen_evaluator_scorecard.py',
        'protocol': ROOT.parent / 'docs/frozen_evaluator_scorecard_protocol.md',
        'auc_helper': Path(sys.modules['biovil_matched_pairs'].__file__),
        'imported_biovil_adapter': Path(sys.modules['score_report_cxr_biovil'].__file__),
        'atomic_contracts': Path(sys.modules['contracts'].__file__),
    }
    for label, (root, expected, filename) in PARENTS.items():
        mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
        if mp.stat().st_size > 1024 ** 2 or sha256_file(mp) != expected:
            raise ValueError('pinned_source_manifest_required')
        manifest = json.loads(mp.read_text())
        sources[label + '_manifest'] = mp
        for name in ('summary.json', filename):
            path = require_inside(root / name, root, must_exist=True)
            entry = manifest['artifacts'][name]
            digest = entry['sha256'] if isinstance(entry, dict) else entry
            if path.stat().st_size > 2 * 1024 ** 2 or sha256_file(path) != digest:
                raise ValueError('bounded_hash_bound_prediction_metadata_required')
            values[label, name] = json.loads(path.read_text())
            if sha256_file(path) != digest:
                raise ValueError('prediction_metadata_changed_during_read')
            sources[label + '_' + name] = path
        if sha256_file(mp) != expected:
            raise ValueError('source_changed_during_read')
        summary = values[label, 'summary.json']
        if (summary['frozen'] is not True or summary['selection_changed'] is not False
                or summary['primary_metric_eligible'] is not False
                or summary['probability_semantics'] is not False):
            raise ValueError('unchanged_unqualified_frozen_readouts_required')
    path = require_inside(COHORT, PROTECTED_ROOT, must_exist=True)
    if path.stat().st_size > 1024 ** 2 or sha256_file(path) != COHORT_SHA:
        raise ValueError('fixed_public_cohort_required')
    values['cohort'] = json.loads(path.read_text())
    if sha256_file(path) != COHORT_SHA:
        raise ValueError('cohort_changed_during_read')
    sources['cohort'] = path
    return values, sources


def family_margins(pairs):
    if set(pairs) != set(FAMILIES):
        raise ValueError('all_three_original_templates_required')
    result = {}
    for family in FAMILIES:
        pair = pairs[family]
        if set(pair) != {'positive_cosine', 'negative_cosine'}:
            raise ValueError('exact_cosine_endpoints_required')
        result[family] = (finite(pair['positive_cosine'], lower=-1.00001, upper=1.00001)
            - finite(pair['negative_cosine'], lower=-1.00001, upper=1.00001))
    result[MEAN] = sum(result[f] for f in FAMILIES) / len(FAMILIES)
    return result


def align_inputs(values):
    cohort = values['cohort']
    rows = cohort['records']
    ids = [f'case_{i:04d}' for i in range(50)]
    if (cohort['patient_grouping_verified'] is not False
            or cohort['independent_image_disease_adjudication'] is not False
            or [r['case_id'] for r in rows] != ids):
        raise ValueError('exact_fixed_image_level_proxy_inventory_required')
    references = [r['reference_state'] for r in rows]
    if Counter(references) != {'positive': 25, 'negative': 25}:
        raise ValueError('known_balanced_public_proxy_references_required')
    labels = [int(state == 'positive') for state in references]
    records = {m: values[m, PARENTS[m][2]]['records'] for m in RANK_MODELS}
    if any([r['case_id'] for r in records[m]] != ids for m in RANK_MODELS):
        raise ValueError('no_missing_reordered_or_duplicate_readouts')
    xrv, biovil, siglip = [], [], []
    for i in range(len(ids)):
        x, b, s = (records[m][i] for m in RANK_MODELS)
        if (x['reference_state'] != references[i]
                or not HASH.fullmatch(x['image_sha256'])
                or b['source_image_sha256'] != x['image_sha256']
                or s['original_bmp_sha256'] != x['image_sha256']
                or not HASH.fullmatch(b['png_sha256'])
                or s['artifact_sha256'] != b['png_sha256']):
            raise ValueError('same_original_and_lossless_image_hashes_required')
        if (s['clinical_state'] is not None or s['calibrated_probability'] is not None
                or s['independent_clinical_validation'] is not False
                or s['pairs'] is None or s['failure_reason'] is not None):
            raise ValueError('complete_unqualified_siglip_readout_required')
        pneumonia = [p for p in s['pairs'] if p['finding'] == 'pneumonia']
        if len(pneumonia) != 3 or len({p['family'] for p in pneumonia}) != 3:
            raise ValueError('complete_original_pneumonia_template_inventory_required')
        xrv.append(finite(x['pneumonia_score'], lower=0, upper=1))
        biovil.append(family_margins(b['score_pairs']))
        siglip.append(family_margins({p['family']: {k: p[k] for k in
            ('positive_cosine', 'negative_cosine')} for p in pneumonia}))
    if len({r['image_sha256'] for r in records['xrv']}) != 50:
        raise ValueError('fifty_unique_original_images_required')
    return labels, xrv, {'biovil': biovil, 'siglip': siglip}


def decision_metrics(labels, scores, *, threshold, equality_is_positive):
    if (not labels or len(labels) != len(scores) or any(type(v) is not int or v not in (0, 1) for v in labels)
            or set(labels) != {0, 1} or type(equality_is_positive) is not bool):
        raise ValueError('both_binary_classes_and_fixed_decision_semantics_required')
    threshold = finite(threshold)
    values = [finite(s) for s in scores]
    p, n = sum(labels), len(labels) - sum(labels)
    positive_wins = sum(y == 1 and (s >= threshold if equality_is_positive else s > threshold)
        for y, s in zip(labels, values))
    negative_wins = sum(y == 0 and s < threshold for y, s in zip(labels, values))
    return {**auc_ap(labels, values), 'positive_references': p, 'negative_references': n,
        'positive_wins': positive_wins, 'negative_wins': negative_wins,
        'positive_win_rate': positive_wins / p, 'negative_win_rate': negative_wins / n,
        'balanced_win_rate': (positive_wins / p + negative_wins / n) / 2,
        'boundary_ties': sum(s == threshold for s in values), 'threshold': threshold,
        'equality_is_positive': equality_is_positive, 'clinical_accuracy': None,
        'calibrated_probability': False}


def template_stability(labels, margins):
    if len(labels) != len(margins) or not labels or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError('matching_binary_reference_inventory_required')
    counts, wins = Counter(), 0
    for label, row in zip(labels, margins):
        if set(row) != {*FAMILIES, MEAN}:
            raise ValueError('all_original_template_margins_required')
        family_values = [finite(row[f]) for f in FAMILIES]
        if finite(row[MEAN]) != sum(family_values) / len(FAMILIES):
            raise ValueError('unchanged_arithmetic_template_mean_required')
        if any(s == 0 for s in family_values):
            state = 'tied_templates'
        elif all(s > 0 for s in family_values):
            state = 'present_prompt_higher'
            wins += label == 1
        elif all(s < 0 for s in family_values):
            state = 'absent_prompt_higher'
            wins += label == 0
        else:
            state = 'template_sensitive'
        counts[state] += 1
    stable = counts['present_prompt_higher'] + counts['absent_prompt_higher']
    return {'case_count': len(labels), 'counts': dict(sorted(counts.items())),
        'stable_cases': stable, 'stable_coverage': stable / len(labels),
        'stable_proxy_wins': wins, 'conditional_proxy_win_rate': wins / stable if stable else None,
        'clinical_accuracy': None}


def percentile(values, quantile):
    if not values or type(quantile) not in (int, float) or not 0 <= quantile <= 1:
        raise ValueError('nonempty_percentile_input_required')
    ordered = sorted(finite(v) for v in values)
    point = (len(ordered) - 1) * quantile
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def paired_bootstrap(labels, scores, *, replicates=REPLICATES, seed=SEED):
    if (set(scores) != set(RANK_MODELS) or not labels or set(labels) != {0, 1}
            or any(type(y) is not int or y not in (0, 1) for y in labels)
            or type(replicates) is not int or replicates < 2 or type(seed) is not int):
        raise ValueError('fixed_paired_binary_bootstrap_contract_required')
    for values in scores.values():
        if len(values) != len(labels):
            raise ValueError('exact_shared_case_denominators_required')
        for value in values:
            finite(value)
    pos = [i for i, y in enumerate(labels) if y]
    neg = [i for i, y in enumerate(labels) if not y]
    samples = {m: {metric: [] for metric in ('auroc', 'average_precision')} for m in RANK_MODELS}
    deltas = {a + '_minus_' + b: {metric: [] for metric in ('auroc', 'average_precision')}
        for a, b in COMPARISONS}
    rng = random.Random(seed)
    resampled_labels = [1] * len(pos) + [0] * len(neg)
    for _ in range(replicates):
        indices = [rng.choice(pos) for _ in pos] + [rng.choice(neg) for _ in neg]
        measured = {m: auc_ap(resampled_labels, [scores[m][i] for i in indices]) for m in RANK_MODELS}
        for m in RANK_MODELS:
            for metric, value in measured[m].items():
                samples[m][metric].append(value)
        for a, b in COMPARISONS:
            for metric in measured[a]:
                deltas[a + '_minus_' + b][metric].append(measured[a][metric] - measured[b][metric])
    points = {m: auc_ap(labels, scores[m]) for m in RANK_MODELS}
    def interval(point, values):
        low, high = percentile(values, .025), percentile(values, .975)
        return {'estimate': point, 'lower_95': low, 'upper_95': high,
            'interval_contains_zero': low <= 0 <= high}
    return {'replicates': replicates, 'seed': seed, 'resampling_unit': 'public_reference_image',
        'class_counts_preserved': {'positive': len(pos), 'negative': len(neg)},
        'paired_indices_shared_across_models': True, 'interval_method': 'percentile_linear_interpolation',
        'developmental_posthoc': True, 'patient_independence_verified': False,
        'clinical_significance_established': False, 'p_values': None,
        'models': {m: {k: interval(points[m][k], v) for k, v in samples[m].items()} for m in RANK_MODELS},
        'paired_differences': {a + '_minus_' + b: {k: interval(points[a][k] - points[b][k], v)
            for k, v in deltas[a + '_minus_' + b].items()} for a, b in COMPARISONS}}


def score_usage_contract():
    return {'schema_version': SCHEMA + '-usage', 'documentation_only_not_deployment_policy': True,
        'universal_weighted_score': None, 'old_lexicographic_selection_changed': False,
        'new_action_or_repair_authorization': False,
        'score_vector': ['artifact_validity', 'fixed_ehr_evidence_and_coverage',
            'raw_edge_proxy_support_opposition', 'alternate_retrieval_or_text_margin',
            'uncertainty_and_dependency', 'actual_execution_cost'],
        'unknown_policy': 'unknown_or_uncertain_is_not_negative',
        'ehr_context_policy': 'risk_medication_lab_context_is_not_a_direct_image_finding',
        'dependency_policy': 'same_image_reports_are_correlated_not_independent_votes',
        'evaluators': {
            'xrv': {'scope': 'frozen_image_finding_proxy', 'qualified_calibrated_probability': False,
                'clinical_truth_or_fault_authority': False, 'absent_head_policy': 'not_comparable'},
            'biovil': {'scope': 'secondary_image_text_retrieval_or_fixed_text_margin',
                'qualified_calibrated_probability': False, 'clinical_truth_or_fault_authority': False,
                'global_retrieval_implies_finding_negation': False},
            'siglip': {'scope': 'alternate_fixed_template_text_preference',
                'qualified_calibrated_probability': False, 'clinical_truth_or_fault_authority': False,
                'shares_vision_encoder_with_chexagent2': True, 'template_stability_implies_clinical_truth': False}},
        'clinical_fault_localization_accuracy': None, 'clinical_repair_success_rate': None,
        'unavailable_edges_policy': 'preserve_null_and_denominator_not_zero_or_perfect',
        'future_automatic_experiment_scope': 'exploratory_proxy_selection_or_repair_not_clinical_truth'}


def analyze(values):
    labels, xrv, vlm = align_inputs(values)
    rows = []
    for profile, threshold in (('default_0_5', .5), ('unchanged_weak_reference_transport', .55457607)):
        metric = decision_metrics(labels, xrv, threshold=threshold, equality_is_positive=True)
        old = values['xrv', 'summary.json'][
            'default_0_5' if profile == 'default_0_5' else 'unchanged_weak_reference_threshold_transport']
        if (metric['auroc'] != old['auroc'] or metric['average_precision'] != old['average_precision']
                or metric['positive_wins'] != old['tp'] or metric['negative_wins'] != old['tn']
                or old['threshold'] != threshold):
            raise ValueError('original_xrv_statistics_must_replay_exactly')
        rows.append({'model': 'xrv', 'profile': profile, 'decision_semantics': 'operating_point_normalized_proxy', **metric})
    for m in ('biovil', 'siglip'):
        for family in (*FAMILIES, MEAN):
            metric = decision_metrics(labels, [r[family] for r in vlm[m]], threshold=0., equality_is_positive=False)
            old = (values[m, 'summary.json']['templates'] if m == 'biovil'
                else values[m, 'summary.json']['analysis']['full_cohort_metrics'])[family]
            if (metric['auroc'] != old['auroc'] or metric['average_precision'] != old['average_precision']
                    or metric['positive_wins'] != old['positive_polarity_wins']
                    or metric['negative_wins'] != old['negative_polarity_wins']
                    or metric['boundary_ties'] != old['ties']):
                raise ValueError('all_original_template_statistics_must_replay_exactly')
            rows.append({'model': m, 'profile': family, 'decision_semantics': 'raw_authored_text_margin_sign', **metric})
    stability = {m: template_stability(labels, vlm[m]) for m in ('biovil', 'siglip')}
    old = values['siglip', 'summary.json']['analysis']
    if (stability['siglip']['counts'] != old['stable_text_direction_counts']
            or stability['siglip']['stable_coverage'] != old['stable_coverage']
            or stability['siglip']['stable_proxy_wins'] != old['stable_proxy_wins']):
        raise ValueError('original_siglip_stability_must_replay_exactly')
    bootstrap = paired_bootstrap(labels, {'xrv': xrv, **{m: [r[MEAN] for r in vlm[m]] for m in vlm}})
    return rows, stability, bootstrap


def render(rows, stability, bootstrap):
    lines = ['# Frozen scorecard / 冻结评分器汇总', '',
        'Post-hoc developmental comparison of 50 public RSUA proxy-reference images.',
        'No new model calls, source image/report/EHR reads, threshold fitting or old selection changes.', '',
        '| Frozen readout | AUROC | AP | Positive wins / 25 | Negative wins / 25 | Balanced wins |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} / {row['profile']} | {row['auroc']:.4f} | "
            f"{row['average_precision']:.4f} | {row['positive_wins']} | {row['negative_wins']} | {row['balanced_win_rate']:.2f} |")
    lines += ['', '## Paired uncertainty / 同图配对不确定性', '',
        '2,000 class-stratified paired image resamples, seed 0; percentile intervals.',
        'The sample was already inspected, patient grouping is unverified, and these are not clinical significance tests.', '',
        '| Fixed comparison | AUROC difference | Descriptive 95% interval | AP difference | Descriptive 95% interval |',
        '|---|---:|---|---:|---|']
    for name, metrics in bootstrap['paired_differences'].items():
        a, p = metrics['auroc'], metrics['average_precision']
        lines.append(f"| {name} | {a['estimate']:+.4f} | [{a['lower_95']:+.4f}, {a['upper_95']:+.4f}] | "
            f"{p['estimate']:+.4f} | [{p['lower_95']:+.4f}, {p['upper_95']:+.4f}] |")
    lines += ['', '## Template stability / 模板稳定性', '']
    for name, s in stability.items():
        lines.append(f"- {name}: stable {s['stable_cases']}/{s['case_count']}; "
            f"conditional proxy wins {s['stable_proxy_wins']}/{s['stable_cases']}. "
            'This conditional denominator does not become full-cohort clinical accuracy.')
    lines += ['', '## Pipeline use / 管线里的用途', '',
        '保留分项向量：基本有效性、固定 EHR 证据与覆盖、各边原始 proxy support/opposition、',
        '辅助图文分、模板／评分器分歧、实际成本。不把不同量纲强行加成一个临床总分。',
        'XRV 是 finding proxy；BioViL-T 是辅助检索分；SigLIP 是模板文本偏好。',
        '三者都不能单独裁决哪种模态错了，也不能通过多数票制造临床真值。',
        '本 usage contract 仅解释用途，不是已接入的新 routing/repair policy；旧获选结果完全不变。', '',
        'The references are published pneumonia/normal-proxy cohort classes, not adjudicated disease truth.',
        'AUROC ranking, polarity, coverage and clinical correctness are separate quantities.',
        'No seven-head reference extension, calibration, best-template choice or prospective superiority is claimed.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    require_cpu_slurm()
    started = time.monotonic()
    values, sources = load_inputs()
    before = {k: sha256_file(p) for k, p in sources.items()}
    rows, stability, bootstrap = analyze(values)
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        write_private_text(temporary / 'scorecard.csv', stream.getvalue())
        write_private_json(temporary / 'bootstrap_comparison.json', bootstrap)
        write_private_json(temporary / 'score_usage_contract.json', score_usage_contract())
        write_private_json(temporary / 'summary.json', {'schema_version': SCHEMA,
            'source_cases': 50, 'positive_proxy_references': 25, 'negative_proxy_references': 25,
            'scorecard_rows': len(rows), 'template_stability': stability,
            'new_model_calls': 0, 'training_or_fitting': False, 'selection_changed': False,
            'reference_images_or_weights_opened': False, 'mimic_inputs_or_targets_opened': False,
            'synthetic_output_bodies_opened': False, 'primary_metric_eligible': False,
            'regeneration_authorized': False, 'independent_clinical_validation': False,
            'clinical_accuracy': None, 'documentation_only_not_deployment_policy': True,
            'elapsed_cpu_analysis_seconds_before_commit': round(time.monotonic() - started, 6)})
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(rows, stability, bootstrap))
        if any(sha256_file(p) != before[k] for k, p in sources.items()):
            raise ValueError('consumed_source_changed_during_analysis')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'run_id': run_id, 'source_paths': {k: str(p.resolve()) for k, p in sources.items()},
            'source_sha256': before, 'artifacts': {p.name: {'sha256': sha256_file(p)}
                for p in sorted(temporary.iterdir())}, 'new_model_calls': 0,
            'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'independent_clinical_validation': False})
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE / 'evaluator_scorecards')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        path = execute(args.output_root, args.run_id)
    except Exception:
        print(json.dumps({'status': 'failed_metadata_scorecard', 'new_model_calls': 0}))
        raise SystemExit(1) from None
    print(json.dumps({'status': 'completed_cached_scorecard', 'source_cases': 50,
        'new_model_calls': 0, 'manifest_sha256': sha256_file(path / 'manifest.json')}))


if __name__ == '__main__':
    main()

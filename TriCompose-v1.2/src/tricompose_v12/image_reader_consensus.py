"""Fixed single-finding image-reference consensus diagnostic, not a policy.

References are pre-existing derived image annotations, never reader votes.
Both readers share the same image and their agreement is dependent evidence.
Template stability is one reader's sensitivity, not three independent votes.
"""
from collections import Counter
import hashlib
import math
import re

import numpy as np

from .manual_reader_risk_coverage import interval, ratio
from .radgraph_reference_contract import require

POLICIES = ('xrv_exact_0_5', 'biovil_fixed_mean', 'agree_fixed_mean', 'agree_all_templates')
CONTRASTS = (('xrv_exact_0_5', 'agree_fixed_mean'),
             ('biovil_fixed_mean', 'agree_fixed_mean'),
             ('agree_fixed_mean', 'agree_all_templates'))
FIELDS = ('attempted', 'reference_positive', 'reference_negative', 'true_positive',
          'false_negative', 'true_negative', 'false_positive', 'accepted', 'abstained', 'unavailable')


def sign(value):
    return 'positive' if value > 0 else 'negative' if value < 0 else 'unknown'


def decision(row, policy):
    require(policy in POLICIES, 'fixed_image_readout_required')
    x, values = row['xrv_score'], row['margins']
    if x is not None:
        require(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1,
                'finite_exact_operating_point_score_required')
    if values is not None:
        require(isinstance(values, list) and len(values) == 3
                and all(type(v) in (int, float) and math.isfinite(v) and -2 <= v <= 2 for v in values),
                'three_unchanged_finite_template_margins_required')
    xs = 'positive' if x is not None and x >= .5 else 'negative' if x is not None else None
    mean = sign(sum(values) / 3) if values is not None else None
    if policy == 'xrv_exact_0_5':
        state = xs
    elif policy == 'biovil_fixed_mean':
        state = mean
    else:
        if xs is None or mean is None:
            return {'state': None, 'status': 'unavailable_reader'}
        if policy == 'agree_all_templates':
            signs = [sign(v) for v in values]
            if 'unknown' in signs:
                return {'state': None, 'status': 'abstain_template_tie'}
            if len(set(signs)) > 1:
                return {'state': None, 'status': 'abstain_template_sensitive'}
            mean = signs[0]
        if mean == 'unknown':
            return {'state': None, 'status': 'abstain_mean_tie'}
        if xs != mean:
            return {'state': None, 'status': 'abstain_reader_disagreement'}
        state = xs
    if state is None:
        return {'state': None, 'status': 'unavailable_reader'}
    if state == 'unknown':
        return {'state': None, 'status': 'abstain_mean_tie'}
    return {'state': state, 'status': 'accepted_determinate'}


def fractions(a):
    c = {f: a[..., i] for i, f in enumerate(FIELDS)}
    require(np.all(c['true_positive'] + c['false_negative'] + c['true_negative'] + c['false_positive'] == c['accepted'])
            and np.all(c['accepted'] + c['abstained'] + c['unavailable'] == c['attempted']),
            'retained_full_denominator_required')
    return {
        'accepted_error_risk': (c['false_positive'] + c['false_negative'], c['accepted']),
        'proposal_coverage': (c['accepted'], c['attempted']),
        'positive_reference_recovery': (c['true_positive'], c['reference_positive']),
        'negative_reference_recovery': (c['true_negative'], c['reference_negative']),
        'conditional_positive_recovery': (c['true_positive'], c['true_positive'] + c['false_negative']),
        'conditional_negative_recovery': (c['true_negative'], c['true_negative'] + c['false_positive']),
        'unavailable_rate': (c['unavailable'], c['attempted']),
    }


def analyze(records, *, seed=0, repetitions=2000):
    require(records and type(seed) is int and seed >= 0 and type(repetitions) is int
            and 100 <= repetitions <= 10000, 'bounded_cached_bootstrap_required')
    records = sorted(records, key=lambda r: r['case_id'])
    require(len({r['case_id'] for r in records}) == len(records)
            and len({r['image_sha256'] for r in records}) == len(records)
            and all(re.fullmatch(r'case_[0-9]{3}', r['case_id'])
                    and re.fullmatch(r'[a-f0-9]{64}', r['image_sha256'])
                    and r['reference_state'] in ('positive', 'negative') for r in records),
            'unique_bound_image_reference_inventory_required')
    classes = [r['reference_state'] for r in records]
    require(set(classes) == {'positive', 'negative'}, 'both_reference_classes_required')
    counts = np.zeros((len(POLICIES), len(records), len(FIELDS)), dtype=np.int64)
    outcomes = []
    statuses = {p: Counter() for p in POLICIES}
    for p, policy in enumerate(POLICIES):
        for i, row in enumerate(records):
            value = decision(row, policy)
            statuses[policy][value['status']] += 1
            c = dict.fromkeys(FIELDS, 0)
            c['attempted'] = 1
            c['reference_' + row['reference_state']] = 1
            c['unavailable'] = value['status'] == 'unavailable_reader'
            c['abstained'] = value['status'].startswith('abstain_')
            if value['state'] is not None:
                c['accepted'] = 1
                label = ('true_' if value['state'] == row['reference_state'] else 'false_') + value['state']
                c[label] = 1
            counts[p, i] = [c[f] for f in FIELDS]
            outcomes.append({'policy': policy, 'case_id': row['case_id'], 'image_sha256': row['image_sha256'],
                             'reference_state': row['reference_state'], **value})
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = np.zeros((repetitions, len(records)), dtype=np.int64)
    for state in ('positive', 'negative'):
        indices = np.array([i for i, v in enumerate(classes) if v == state])
        selected = rng.integers(0, len(indices), size=(repetitions, len(indices)))
        for i, values in enumerate(selected):
            weights[i, indices] = np.bincount(values, minlength=len(indices))
    base, sampled = counts.sum(axis=1), np.einsum('rn,pnc->prc', weights, counts, optimize=True)
    point, boot = fractions(base), fractions(sampled)
    rows, contrasts = [], []
    for p, policy in enumerate(POLICIES):
        metrics = {}
        for name, (n, d) in point.items():
            numerator, denominator = int(n[p]), int(d[p])
            metrics[name] = {'numerator': numerator, 'denominator': denominator,
                'estimate': numerator / denominator if denominator else None,
                **interval(ratio(*boot[name])[p])}
        rows.append({'policy': policy, 'counts': {f: int(base[p, i]) for i, f in enumerate(FIELDS)},
                     'status_counts': dict(statuses[policy]), 'metrics': metrics,
                     'clinical_qualified': False, 'independent_truth_votes': False})
    for left, right in CONTRASTS:
        a, b = POLICIES.index(left), POLICIES.index(right)
        metrics = {}
        for name, (n, d) in point.items():
            estimates, samples = ratio(n, d), ratio(*boot[name])
            value = estimates[b] - estimates[a]
            metrics[name] = {'right_minus_left': float(value) if np.isfinite(value) else None,
                             **interval(samples[b] - samples[a])}
        contrasts.append({'left': left, 'right': right, 'shared_paired_draws': True, 'metrics': metrics})
    return {'schema_version': 'fixed-image-reference-consensus-risk-coverage-v1', 'rows': rows,
        'outcomes': outcomes, 'paired_contrasts': contrasts,
        'sampling': {'seed': seed, 'repetitions': repetitions, 'confidence': .95, 'numpy_version': np.__version__,
            'unit': 'one_bound_case_per_previously_verified_patient', 'class_stratified': True,
            'class_sizes': dict(Counter(classes)), 'same_cases_and_draws_for_all_policies': True,
            'patient_source_binding_rechecked_by_this_module': False,
            'weights_sha256': hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest()},
        'post_hoc_development': True, 'all_templates_are_independent_models': False,
        'clinical_qualified': False, 'thresholds_fitted': False, 'best_policy_selected': False,
        'selection_changed': False, 'regeneration_authorized': False}

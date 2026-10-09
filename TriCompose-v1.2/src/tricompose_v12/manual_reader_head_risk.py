"""Per-literal-head decomposition, never a clinically qualified score.

Keep the same report clusters, four-state references, readers, masks and draws
as the sealed pooled benchmark. This is a cached development breakdown, not a
new test set, threshold fit, policy selection or synthetic-domain validation.
"""
import hashlib

import numpy as np

from .manual_reader_risk_coverage import (
    CONTRASTS, FIELDS, FINDINGS, IX, VIEWS, aggregate_reports, draws,
    fractions, interval, ratio,
)
from .radgraph_reference_contract import require

DOMAINS = ('all', 'mimic', 'chexpert')


def aggregate_heads(references, details):
    refs, policies, pooled = aggregate_reports(references, details)
    indices = {r['report_id']: i for i, r in enumerate(refs)}
    data = np.zeros((len(VIEWS), len(policies), len(FINDINGS), len(refs), len(FIELDS)), dtype=np.int64)
    for row in details:
        i = indices[row['report_id']]
        ref = refs[i]['finding_states'][row['finding']]
        a = data[VIEWS.index(row['readout']), policies.index(row['policy']), FINDINGS.index(row['finding']), i]
        a[IX['attempted']] = 1
        a[IX[ref]] = 1
        a[IX['unavailable']] = row['status'] == 'unavailable_reader'
        if row['state'] is not None:
            a[IX['accepted']] = 1
            if ref in ('positive', 'negative'):
                a[IX['accepted_known']] = 1
                a[IX['flips']] = row['state'] != ref
                a[IX['correct_' + ref]] = row['state'] == ref
            else:
                a[IX['commit_' + ref]] = 1
    require(np.array_equal(data.sum(axis=2), pooled) and np.all(data[..., IX['attempted']] == 1),
            'head_decomposition_must_conserve_every_report_counter')
    return refs, policies, data


def support_status(positive, negative):
    if positive and negative:
        return 'both_reference_polarities_present_not_sufficient_validation'
    if positive:
        return 'no_negative_reference_support'
    if negative:
        return 'no_positive_reference_support'
    return 'no_positive_or_negative_reference_support'


def analyze_heads(references, details, *, repetitions=2000, seed=0):
    require(type(repetitions) is int and 100 <= repetitions <= 10000
            and type(seed) is int and seed >= 0, 'bounded_fixed_head_diagnostic_draws_required')
    refs, policies, data = aggregate_heads(references, details)
    domains = [r['source_domain'] for r in refs]
    weights = draws(domains, repetitions=repetitions, seed=seed)
    rows, contrasts, inventory = [], [], []
    for domain in DOMAINS:
        selected = [i for i, d in enumerate(domains) if domain == 'all' or d == domain]
        point = data[:, :, :, selected, :].sum(axis=3)
        sampled = np.einsum('rn,vphnc->vphrc', weights[:, selected], data[:, :, :, selected, :], optimize=True)
        base, boot = fractions(point), fractions(sampled)
        for h, finding in enumerate(FINDINGS):
            ref_counts = {state: sum(refs[i]['finding_states'][finding] == state for i in selected)
                          for state in ('positive', 'negative', 'uncertain', 'unknown')}
            inventory.append({'domain': domain, 'finding': finding, 'reports': len(selected), **ref_counts,
                'known_reference_coverage': (ref_counts['positive'] + ref_counts['negative']) / len(selected),
                'reference_support_status': support_status(ref_counts['positive'], ref_counts['negative']),
                'human_reference_is_literal_projection_not_image_truth': True,
                'clinical_qualified': False})
            for v, view in enumerate(VIEWS):
                for p, policy in enumerate(policies):
                    c = {name: int(point[v, p, h, i]) for name, i in IX.items()}
                    row = {'readout': view, 'domain': domain, 'policy': policy, 'finding': finding,
                        'reports': len(selected), **c,
                        'reference_support_status': support_status(c['positive'], c['negative']),
                        'uncertain_reference_support_present': bool(c['uncertain']),
                        'clinical_qualified': False, 'patient_cluster_verified': False}
                    row['metrics'] = {}
                    for name, (ns, ds) in base.items():
                        n, d = int(ns[v, p, h]), int(ds[v, p, h])
                        row['metrics'][name] = {'numerator': n, 'denominator': d,
                            'estimate': n / d if d else None,
                            **interval(ratio(*boot[name])[v, p, h])}
                    rows.append(row)
                for left, right in CONTRASTS:
                    a, b = policies.index(left), policies.index(right)
                    differences = {}
                    for name, (ns, ds) in base.items():
                        estimates, samples = ratio(ns, ds), ratio(*boot[name])
                        value = estimates[v, b, h] - estimates[v, a, h]
                        differences[name] = {'right_minus_left': float(value) if np.isfinite(value) else None,
                            **interval(samples[v, b, h] - samples[v, a, h])}
                    contrasts.append({'readout': view, 'domain': domain, 'finding': finding,
                        'left': left, 'right': right, 'shared_paired_draws': True, 'differences': differences})
    return {'schema_version': 'manual-reader-literal-head-risk-coverage-v1',
        'rows': rows, 'paired_contrasts': contrasts, 'reference_inventory': inventory,
        'sampling': {'bootstrap_unit': 'distinct_report_artifact', 'patient_groups_verified': False,
            'strata': {d: domains.count(d) for d in ('mimic', 'chexpert')},
            'all_heads_views_and_policies_share_draws': True, 'seed': seed, 'repetitions': repetitions,
            'numpy_version': np.__version__, 'confidence': .95,
            'weights_sha256': hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest()},
        'head_counters_conserve_original_pooled_report_counters': True,
        'clinical_qualified': False, 'post_hoc_development_diagnostic': True,
        'new_untouched_test_set': False, 'unknown_is_negative': False,
        'literal_unknown_promotions_are_clinical_errors': False,
        'best_policy_selected': False, 'thresholds_fitted': False,
        'synthetic_domain_transport_validated': False, 'selection_changed': False,
        'regeneration_authorized': False}

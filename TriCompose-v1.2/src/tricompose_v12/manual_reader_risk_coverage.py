"""Post-hoc report-clustered language-state risk, never clinical calibration.

Uncertain-reference commitments join polarity flips in the error numerator.
Literal-unknown references are NOT clinical negatives/errors. Shared bootstrap
draws preserve all four heads and all seven dependent masks within a report.
Report clusters are not verified patient clusters or an independent test set.
"""
import hashlib
import re

import numpy as np

from .radgraph_reference_contract import require
from .manual_three_reader_agreement import FINDINGS, POLICIES, STATES

VIEWS = ('native_labels', 'mention_conflict_view')
FIELDS = ('attempted', 'positive', 'negative', 'uncertain', 'unknown',
          'correct_positive', 'correct_negative', 'flips', 'accepted_known',
          'commit_uncertain', 'commit_unknown', 'accepted', 'unavailable')
IX = {name: i for i, name in enumerate(FIELDS)}
CONTRASTS = (('radgraph', 'agree_radgraph_chexbert'),
             ('chexbert', 'agree_radgraph_chexbert'),
             ('agree_radgraph_chexbert', 'agree_all_three'))
DECISIONS = {'accepted_determinate_proposal', 'unavailable_reader', 'abstain_unknown',
             'abstain_uncertain', 'abstain_disagreement'}


def aggregate_reports(references, details):
    require(isinstance(references, list) and references, 'nonempty_fixed_reference_inventory_required')
    refs = sorted(references, key=lambda r: r['report_id'])
    ids = {r['report_id']: i for i, r in enumerate(refs)}
    hashes = set()
    require(len(ids) == len(refs), 'unique_opaque_report_indices_required')
    for r in refs:
        require(re.fullmatch(r'report_[0-9]{4}', r['report_id'])
                and r['source_domain'] in ('mimic', 'chexpert')
                and re.fullmatch(r'[a-f0-9]{64}', r['source_sha256'])
                and r['source_sha256'] not in hashes, 'distinct_report_artifact_clusters_required')
        hashes.add(r['source_sha256'])
        require(set(r['finding_states']) == set(FINDINGS)
                and all(v in STATES for v in r['finding_states'].values()), 'complete_literal_four_states_required')
    policies = tuple(POLICIES)
    seen = set()
    data = np.zeros((len(VIEWS), len(policies), len(refs), len(FIELDS)), dtype=np.int64)
    for row in details:
        view, policy, rid, finding = (row[k] for k in ('readout', 'policy', 'report_id', 'finding'))
        key = view, policy, rid, finding
        require(view in VIEWS and policy in POLICIES and rid in ids and finding in FINDINGS and key not in seen,
                'complete_unique_fixed_mask_details_required')
        seen.add(key)
        ref = refs[ids[rid]]['finding_states'][finding]
        require(row['human_literal_reference_state'] == ref, 'unchanged_projected_reference_required')
        state, status = row['state'], row['status']
        require(status in DECISIONS and state in (None, 'positive', 'negative')
                and (state is not None) == (status == 'accepted_determinate_proposal'),
                'determinate_or_explicit_abstention_required')
        a = data[VIEWS.index(view), policies.index(policy), ids[rid]]
        a[IX['attempted']] += 1
        a[IX[ref]] += 1
        a[IX['unavailable']] += status == 'unavailable_reader'
        if state is not None:
            a[IX['accepted']] += 1
            if ref in ('positive', 'negative'):
                a[IX['accepted_known']] += 1
                a[IX['flips']] += state != ref
                if state == ref:
                    a[IX['correct_' + ref]] += 1
            elif ref == 'uncertain':
                a[IX['commit_uncertain']] += 1
            else:
                a[IX['commit_unknown']] += 1
    require(len(seen) == len(VIEWS) * len(policies) * len(refs) * len(FINDINGS)
            and np.all(data[..., IX['attempted']] == len(FINDINGS)), 'no_missing_masks_or_heads_allowed')
    return refs, policies, data


def fractions(counts):
    a = np.asarray(counts)
    require(a.ndim > 0 and a.shape[-1] == len(FIELDS) and np.all(np.isfinite(a))
            and np.all(a >= 0) and np.all(a == np.floor(a)),
            'finite_nonnegative_counter_vector_required')
    c = {name: a[..., i] for name, i in IX.items()}
    require(np.all(c['positive'] + c['negative'] + c['uncertain'] + c['unknown'] == c['attempted'])
            and np.all(c['correct_positive'] + c['correct_negative'] + c['flips'] == c['accepted_known'])
            and np.all(c['accepted_known'] + c['commit_uncertain'] + c['commit_unknown'] == c['accepted'])
            and np.all(c['unavailable'] + c['accepted'] <= c['attempted'])
            and np.all(c['correct_positive'] <= c['positive'])
            and np.all(c['correct_negative'] <= c['negative'])
            and np.all(c['accepted_known'] <= c['positive'] + c['negative'])
            and np.all(c['commit_uncertain'] <= c['uncertain'])
            and np.all(c['commit_unknown'] <= c['unknown']), 'counter_conservation_required')
    known = c['positive'] + c['negative']
    adjudicable_proposals = c['accepted_known'] + c['commit_uncertain']
    return {
        'known_recovery': (c['correct_positive'] + c['correct_negative'], known),
        'positive_recovery': (c['correct_positive'], c['positive']),
        'negative_recovery': (c['correct_negative'], c['negative']),
        'conditional_literal_assertion_error': (c['flips'] + c['commit_uncertain'], adjudicable_proposals),
        'adjudicable_proposal_coverage': (adjudicable_proposals, known + c['uncertain']),
        'proposal_coverage': (c['accepted'], c['attempted']),
        'uncertain_commitment_rate': (c['commit_uncertain'], c['uncertain']),
        'literal_unknown_commitment_rate': (c['commit_unknown'], c['unknown']),
        'unavailable_check_rate': (c['unavailable'], c['attempted']),
    }


def ratio(numerator, denominator):
    n, d = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d != 0)


def interval(samples, confidence=0.95):
    require(0 < confidence < 1, 'valid_confidence_required')
    values = np.asarray(samples, dtype=float)
    require(values.ndim == 1 and not np.any(np.isinf(values)), 'finite_or_missing_scalar_draws_required')
    valid = values[np.isfinite(values)]
    tail = (1 - confidence) / 2
    return {'percentile_interval': np.quantile(valid, [tail, 1 - tail], method='linear').tolist() if len(valid) else None,
            'valid_draws': len(valid), 'zero_denominator_draws': len(values) - len(valid),
            'zero_observed_error_does_not_bound_unseen_error': True}


def draws(domains, *, repetitions=2000, seed=0):
    require(type(repetitions) is int and repetitions >= 100 and type(seed) is int and seed >= 0,
            'fixed_bounded_bootstrap_protocol_required')
    require(repetitions <= 10000 and domains and set(domains) == {'mimic', 'chexpert'},
            'two_nonempty_fixed_domain_strata_required')
    weights = np.zeros((repetitions, len(domains)), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for domain in ('mimic', 'chexpert'):
        indices = np.array([i for i, d in enumerate(domains) if d == domain])
        chosen = rng.integers(0, len(indices), size=(repetitions, len(indices)))
        for i, sample in enumerate(chosen):
            weights[i, indices] = np.bincount(sample, minlength=len(indices))
    return weights


def analyze(references, details, *, repetitions=2000, seed=0):
    refs, policies, data = aggregate_reports(references, details)
    domains = [r['source_domain'] for r in refs]
    weights = draws(domains, repetitions=repetitions, seed=seed)
    rows, contrasts = [], []
    for domain in ('all', 'mimic', 'chexpert'):
        indices = [i for i, d in enumerate(domains) if domain == 'all' or d == domain]
        point = data[:, :, indices, :].sum(axis=2)
        sampled = np.einsum('rn,vpnc->vprc', weights[:, indices], data[:, :, indices, :], optimize=True)
        base, boot = fractions(point), fractions(sampled)
        for v, view in enumerate(VIEWS):
            for p, policy in enumerate(policies):
                row = {'readout': view, 'domain': domain, 'policy': policy, 'reports': len(indices),
                       **{name: int(point[v, p, i]) for name, i in IX.items()}}
                row['metrics'] = {}
                for metric, (ns, ds) in base.items():
                    n, d = int(ns[v, p]), int(ds[v, p])
                    row['metrics'][metric] = {'numerator': n, 'denominator': d,
                        'estimate': n / d if d else None,
                        **interval(ratio(*boot[metric])[v, p])}
                rows.append(row)
            for left, right in CONTRASTS:
                a, b = policies.index(left), policies.index(right)
                differences = {}
                for metric, (ns, ds) in base.items():
                    estimates, samples = ratio(ns, ds), ratio(*boot[metric])
                    value = estimates[v, b] - estimates[v, a]
                    differences[metric] = {'right_minus_left': float(value) if np.isfinite(value) else None,
                        **interval(samples[v, b] - samples[v, a])}
                contrasts.append({'readout': view, 'domain': domain, 'left': left, 'right': right,
                                  'shared_paired_draws': True, 'differences': differences})
    return {'schema_version': 'manual-reader-report-cluster-risk-coverage-v1',
        'rows': rows, 'paired_contrasts': contrasts,
        'sampling': {'bootstrap_unit': 'distinct_report_artifact', 'patient_groups_verified': False,
            'domains_stratified': True, 'stratum_sizes': {d: domains.count(d) for d in ('mimic', 'chexpert')},
            'all_heads_views_and_policies_paired': True, 'seed': seed, 'repetitions': repetitions,
            'generator': 'numpy.PCG64', 'numpy_version': np.__version__, 'confidence': 0.95,
            'weights_sha256': hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest()},
        'clinical_qualified': False, 'post_hoc_development_diagnostic': True,
        'literal_unknown_promotions_are_clinical_errors': False, 'best_policy_selected': False,
        'thresholds_fitted': False, 'selection_changed': False, 'regeneration_authorized': False}

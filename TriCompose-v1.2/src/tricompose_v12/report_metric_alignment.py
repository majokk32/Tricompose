"""Pure, reference-based metric/expert-error alignment; no model or IO.

Consumes separate, opaque derived score/reference contracts, never report
bodies. Published cached metrics do not qualify local frozen implementations.
Correlation with text-reference error counts is not image/EHR correctness.
"""
from collections import Counter
import math
import random
import re

VERSION = 'tricompose-report-metric-alignment-v1'
BENCHMARKS = {'radevalx-1.0.0': 8, 'rexval-1.0.0': 6}
ORIENTATIONS = ('higher_is_better', 'lower_is_better')
PROVENANCE = ('published_cached', 'local_frozen')
LIMIT = 1024
POLICY = {'version': VERSION, 'reference_free': False, 'new_training': False,
    'weight_or_threshold_fitting': False, 'clinical_qualified': False,
    'image_factuality_verified': False, 'ehr_consistency_verified': False,
    'checkpoint_training_overlap_verified': False, 'primary_metric_eligible': False,
    'selection_changed': False, 'regeneration_authorized': False,
    'missing_score_or_annotation_is_zero': False,
    'published_score_qualifies_local_implementation': False}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(obj, keys):
    require(isinstance(obj, dict) and set(obj) == set(keys), 'exact_derived_schema_required')


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def identity(row):
    require(isinstance(row.get('item_id'), str) and re.fullmatch(r'pair_[0-9]{4}', row['item_id']) and
            isinstance(row.get('source_group_id'), str) and re.fullmatch(r'group_[0-9]{4}', row['source_group_id']),
            'opaque_pair_and_group_ids_required')


def validate(predictions, references):
    exact(predictions, ('schema_version', 'benchmark', 'metric_definitions', 'records'))
    exact(references, ('schema_version', 'benchmark', 'reference_policy', 'records'))
    benchmark = predictions['benchmark']
    require(predictions['schema_version'] == VERSION + '-predictions' and
            references['schema_version'] == VERSION + '-references' and
            benchmark in BENCHMARKS and references['benchmark'] == benchmark,
            'same_supported_versioned_benchmark_required')
    require(references['reference_policy'] == ('two_reader_consensus' if benchmark == 'radevalx-1.0.0'
            else 'mean_of_all_released_readers'), 'declared_reader_aggregation_required')
    definitions = predictions['metric_definitions']
    require(isinstance(definitions, dict) and 1 <= len(definitions) <= 32, 'bounded_metric_inventory_required')
    for name, definition in definitions.items():
        require(isinstance(name, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,47}', name), 'safe_metric_name_required')
        exact(definition, ('orientation', 'provenance'))
        require(definition['orientation'] in ORIENTATIONS and definition['provenance'] in PROVENANCE,
                'explicit_metric_orientation_and_provenance_required')
    p_rows, r_rows = predictions['records'], references['records']
    require(isinstance(p_rows, list) and isinstance(r_rows, list) and
            1 <= len(p_rows) == len(r_rows) <= LIMIT, 'bounded_all_attempted_inventory_required')
    seen, combined = set(), []
    for pred, ref in zip(p_rows, r_rows):
        exact(pred, ('item_id', 'source_group_id', 'scores'))
        exact(ref, ('item_id', 'source_group_id', 'errors'))
        identity(pred)
        identity(ref)
        require(pred['item_id'] == ref['item_id'] and pred['source_group_id'] == ref['source_group_id'] and
                pred['item_id'] not in seen, 'unique_ordered_source_group_join_required')
        seen.add(pred['item_id'])
        require(isinstance(pred['scores'], dict) and set(pred['scores']) == set(definitions),
                'all_declared_metrics_per_attempt_required')
        for score in pred['scores'].values():
            exact(score, ('status', 'value'))
            require(score['status'] in ('complete', 'failed_unavailable', 'not_available_in_source'),
                    'explicit_score_availability_required')
            require(numeric(score['value']) if score['status'] == 'complete' else score['value'] is None,
                    'finite_complete_score_or_unavailable_null_required')
        exact(ref['errors'], ('clinically_significant', 'clinically_insignificant'))
        for values in ref['errors'].values():
            require(isinstance(values, list) and len(values) == BENCHMARKS[benchmark], 'exact_error_category_width_required')
            # ReXVal means can be fractional. No annotation missingness -> zero.
            require(all(v is None or (numeric(v) and v >= 0 and
                (benchmark != 'radevalx-1.0.0' or type(v) is int)) for v in values),
                'nonnegative_reference_counts_or_explicit_null_required')
        combined.append({'item_id': pred['item_id'], 'source_group_id': pred['source_group_id'],
                         'scores': pred['scores'], 'errors': ref['errors']})
    if benchmark == 'radevalx-1.0.0':
        require(len({r['source_group_id'] for r in combined}) == len(combined),
                'radevalx_one_candidate_per_study_required')
    return definitions, combined


def ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(left, right):
    a, b = sum(left) / len(left), sum(right) / len(right)
    x, y = [v - a for v in left], [v - b for v in right]
    denominator = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    return sum(u*v for u, v in zip(x, y)) / denominator if denominator else None


def kendall(left, right):
    concordant = discordant = tied_left = tied_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            a, b = left[i] - left[j], right[i] - right[j]
            if not a and not b:
                continue
            if not a:
                tied_left += 1
            elif not b:
                tied_right += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    signed = concordant + discordant
    denominator = math.sqrt((signed + tied_left) * (signed + tied_right))
    return {'value': (concordant-discordant) / denominator if denominator else None,
            'concordant_pairs': concordant, 'discordant_pairs': discordant,
            'left_only_ties': tied_left, 'right_only_ties': tied_right}


def correlation(values):
    require(isinstance(values, list) and all(isinstance(v, tuple) and len(v) == 2 and
            all(numeric(x) for x in v) for v in values), 'finite_paired_numbers_required')
    if len(values) < 3:
        return {'status': 'insufficient_pairs', 'paired_rows': len(values),
                'spearman': None, 'kendall_tau_b': None, 'kendall_counts': None}
    left, right = [r[0] for r in values], [r[1] for r in values]
    rho = pearson(ranks(left), ranks(right))
    tau = kendall(left, right)
    return {'status': 'complete' if rho is not None else 'constant_score_or_reference',
            'paired_rows': len(values), 'spearman': rho, 'kendall_tau_b': tau['value'],
            'kendall_counts': {k: v for k, v in tau.items() if k != 'value'}}


def outcomes(row):
    result = {}
    sums = {}
    for severity, values in row['errors'].items():
        sums[severity] = sum(values) if all(v is not None for v in values) else None
        result[severity + '_total'] = sums[severity]
        result.update({severity + '_category_' + str(i+1): v for i, v in enumerate(values)})
    result['all_errors_total'] = sum(sums.values()) if all(v is not None for v in sums.values()) else None
    return result


def quantile(values, proportion):
    ordered = sorted(values)
    position = (len(ordered)-1) * proportion
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def clustered_spearman_interval(pairs, resamples, seed):
    """Resample source groups, not dependent candidate rows or reader cells."""
    require(type(resamples) is int and 0 <= resamples <= 2000 and type(seed) is int,
            'bounded_fixed_bootstrap_required')
    groups = {}
    for group, quality, negative_errors in pairs:
        groups.setdefault(group, []).append((quality, negative_errors))
    keys = sorted(groups)
    if not resamples or len(keys) < 3:
        return {'status': 'disabled' if not resamples else 'insufficient_source_groups',
            'source_groups': len(keys), 'requested_resamples': resamples, 'usable_resamples': 0,
            'seed': seed, 'spearman_interval_95': None}
    rng, values = random.Random(seed), []
    for _ in range(resamples):
        selected = [pair for _ in keys for pair in groups[keys[rng.randrange(len(keys))]]]
        left, right = [v[0] for v in selected], [v[1] for v in selected]
        rho = pearson(ranks(left), ranks(right))
        if rho is not None:
            values.append(rho)
    # Do not hide constant resamples or claim an interval from a handful.
    usable = len(values) >= max(20, math.ceil(resamples * 0.9))
    return {'status': 'complete' if usable else 'insufficient_nonconstant_resamples',
        'source_groups': len(keys), 'requested_resamples': resamples, 'usable_resamples': len(values),
        'seed': seed, 'spearman_interval_95': [quantile(values, .025), quantile(values, .975)] if usable else None}


def evaluate(predictions, references, *, bootstrap_resamples=1000, seed=0):
    definitions, rows = validate(predictions, references)
    require(type(bootstrap_resamples) is int and 0 <= bootstrap_resamples <= 2000 and type(seed) is int,
            'bounded_fixed_bootstrap_required')
    results = {}
    outcome_rows = {r['item_id']: outcomes(r) for r in rows}
    outcome_names = sorted(next(iter(outcome_rows.values())))
    for metric, definition in definitions.items():
        direction = 1 if definition['orientation'] == 'higher_is_better' else -1
        complete = sum(r['scores'][metric]['status'] == 'complete' for r in rows)
        per_outcome = {}
        for outcome in outcome_names:
            eligible = [(r['source_group_id'], direction * r['scores'][metric]['value'],
                -outcome_rows[r['item_id']][outcome]) for r in rows if r['scores'][metric]['status'] == 'complete'
                and outcome_rows[r['item_id']][outcome] is not None]
            result = correlation([(v[1], v[2]) for v in eligible])
            result.update(all_attempted_rows=len(rows), paired_coverage=len(eligible)/len(rows),
                reference_available=sum(outcome_rows[r['item_id']][outcome] is not None for r in rows))
            # Predeclared primary CI target; per-category results remain descriptive.
            if outcome == 'clinically_significant_total':
                result['cluster_bootstrap'] = clustered_spearman_interval(eligible, bootstrap_resamples, seed)
            per_outcome[outcome] = result
        results[metric] = {'definition': definition, 'complete_scores': complete,
            'score_coverage': complete/len(rows),
            'score_status_counts': dict(sorted(Counter(r['scores'][metric]['status'] for r in rows).items())),
            'outcomes': per_outcome}
    return {'schema_version': VERSION, 'benchmark': predictions['benchmark'], 'policy': POLICY,
        'all_attempted_rows': len(rows), 'source_groups': len({r['source_group_id'] for r in rows}),
        'reference_policy': references['reference_policy'], 'metrics': results,
        'correlation_direction': 'quality_score_vs_negative_expert_error_burden',
        'new_model_calls': 0, 'external_api': False, 'local_implementation_qualified': False,
        'reference_text_or_image_factuality_verified': False}

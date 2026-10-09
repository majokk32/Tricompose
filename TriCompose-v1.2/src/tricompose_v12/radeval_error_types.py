"""Post-hoc numerical error-type diagnostic, not a verifier or repair policy.

Consumes the sealed derived image benchmark table only. No patient text,
images, embeddings, model calls, label creation or score/threshold fitting.
Released report-reference error counts are not image finding ground truth.
"""
from collections import defaultdict
import math
import re

from .radeval_expert import CATEGORIES, SEVERITIES, outcomes, require
from .radeval_image_benchmark import METRICS, STATUSES, delta_interval
from .report_metric_alignment import correlation

VERSION = 'tricompose-radeval-error-type-diagnostic-v1'
ROW_KEYS = {'item_id', 'source_id', 'source_group_id', 'section_id',
    'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256',
    'image_score_status', 'paired_cohort_available', 'scores', 'expert_outcomes'}


def validate(records):
    require(isinstance(records, list) and 1 <= len(records) <= 1024,
            'bounded_complete_derived_table_required')
    seen, anchors = set(), defaultdict(list)
    for row in records:
        require(isinstance(row, dict) and set(row) == ROW_KEYS,
                'derived_only_exact_schema_required')
        for name, pattern in (('item_id', r'pair_[0-9]{4}'),
                              ('source_id', r'source_[0-9]{4}'),
                              ('source_group_id', r'group_[0-9]{4}'),
                              ('section_id', r'section_[0-9]{2}'),
                              ('reference_sha256', r'[0-9a-f]{64}'),
                              ('hypothesis_sha256', r'[0-9a-f]{64}')):
            require(isinstance(row[name], str) and re.fullmatch(pattern, row[name]),
                    'opaque_ids_and_hashes_required')
        require(row['item_id'] not in seen, 'unique_pair_inventory_required')
        seen.add(row['item_id'])
        require(row['section_name'] in ('findings', 'impression', 'other_author_section')
                and type(row['candidate_slot']) is int and 1 <= row['candidate_slot'] <= 3,
                'declared_section_and_candidate_slot_required')
        require(row['image_score_status'] in STATUSES and
                type(row['paired_cohort_available']) is bool,
                'explicit_score_availability_required')
        require(isinstance(row['scores'], dict) and set(row['scores']) == set(METRICS),
                'all_four_metrics_on_same_cohort_required')
        if row['paired_cohort_available']:
            require(row['image_score_status'] == 'complete', 'available_status_mismatch')
            for metric, value in row['scores'].items():
                require(type(value) in (int, float) and math.isfinite(value) and
                        (-1.00001 <= value <= 1.00001 if metric == METRICS[0]
                         else 0 <= value <= 1), 'finite_native_scores_required')
        else:
            require(all(v is None for v in row['scores'].values()),
                    'missing_common_cohort_scores_must_remain_null')
        expert = row['expert_outcomes']
        require(isinstance(expert, dict), 'derived_error_counts_required')
        errors = {s: [expert.get(s + '_' + c) for c in CATEGORIES] for s in SEVERITIES}
        require(outcomes(errors) == expert, 'exact_complete_category_count_contract_required')
        anchors[(row['source_id'], row['section_id'], row['reference_sha256'])].append(row)
    require(all(sorted(r['candidate_slot'] for r in rows) == [1, 2, 3]
                and len({r['source_group_id'] for r in rows}) == 1
                for rows in anchors.values()), 'three_same_group_candidates_per_anchor_required')
    return {key: sorted(rows, key=lambda r: r['candidate_slot'])
            for key, rows in sorted(anchors.items())}


def diagnose(records, *, resamples=1000, seed=0):
    require(type(resamples) is int and 20 <= resamples <= 2000 and type(seed) is int,
            'bounded_fixed_bootstrap_required')
    anchors = validate(records)
    records = sorted(records, key=lambda r: r['item_id'])
    results = []
    for category in CATEGORIES:
        target = 'clinically_significant_' + category
        available = [r for r in records if r['paired_cohort_available'] and
                     r['expert_outcomes'][target] is not None]
        for metric in METRICS:
            paired = [(r['scores'][metric], -r['expert_outcomes'][target]) for r in available]
            choices, strict, correct, ties = [], 0, 0.0, 0
            for rows in anchors.values():
                if not all(r['paired_cohort_available'] and
                           r['expert_outcomes'][target] is not None for r in rows):
                    continue
                values = [r['scores'][metric] for r in rows]
                errors = [r['expert_outcomes'][target] for r in rows]
                top = [i for i, v in enumerate(values) if v == max(values)]
                selected = sum(errors[i] for i in top) / len(top)
                random_mean = sum(errors) / 3
                choices.append({'source_group_id': rows[0]['source_group_id'],
                    'selected_errors': selected, 'random_errors': random_mean,
                    'metric_minus_random_errors': selected - random_mean})
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    if errors[i] == errors[j]:
                        continue
                    strict += 1
                    if values[i] == values[j]:
                        ties += 1
                        correct += .5
                    else:
                        correct += int((values[i] > values[j]) == (errors[i] < errors[j]))
            means = {key: sum(c[key] for c in choices) / len(choices) if choices else None
                     for key in ('selected_errors', 'random_errors', 'metric_minus_random_errors')}
            results.append({'category': category, 'metric': metric,
                'all_attempted_pairs': len(records), 'score_and_category_available_pairs': len(available),
                'error_positive_pairs': sum(r['expert_outcomes'][target] > 0 for r in available),
                'category_error_sum': sum(r['expert_outcomes'][target] for r in available),
                'pooled_correlation': correlation(paired),
                'attempted_anchors': len(anchors), 'complete_anchors': len(choices),
                'strict_candidate_pairs': strict, 'score_tied_strict_pairs': ties,
                'pairwise_accuracy': correct / strict if strict else None,
                'has_error_discriminating_pairs': bool(strict), 'means': means,
                'error_delta_cluster_ci': delta_interval(choices, resamples, seed)})
    return {'schema_version': VERSION, 'post_hoc_descriptive_diagnostic': True,
        'source': 'sealed_derived_image_benchmark_table', 'all_attempted_pairs': len(records),
        'common_score_available_pairs': sum(r['paired_cohort_available'] for r in records),
        'categories': list(CATEGORIES), 'metrics': list(METRICS), 'results': results,
        'missing_scores_or_annotations_are_zero': False, 'multiplicity_adjusted': False,
        'expert_counts_are_image_finding_labels': False, 'clinical_qualified': False,
        'new_model_calls': 0, 'selection_changed': False, 'weight_or_threshold_fitting': False,
        'raw_patient_input_read': False, 'regeneration_authorized': False}

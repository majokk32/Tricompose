"""Frozen image-score versus expert-error benchmark, no clinical policy.

All attempted pairs retained. Reference RadGraph baselines are compared only
on the same image-score-available cohort. Cosine is not a probability.
"""
from collections import defaultdict
import math
import random

from .radeval_expert import outcomes, require
from .radgraph_reference_contract import METRICS as RADGRAPH_METRICS
from .report_metric_alignment import correlation, clustered_spearman_interval, quantile

VERSION = 'tricompose-radeval-image-benchmark-v1'
METRIC = 'biovil_t_raw_cosine'
METRICS = (METRIC, *RADGRAPH_METRICS)
TARGETS = ('clinically_significant_total', 'all_errors_total')
STATUSES = ('complete', 'unavailable_source_image', 'failed_image_or_text')


def join(plan, image_scores, radgraph_scores):
    require(len(plan['records']) == len(image_scores) == len(radgraph_scores),
            'complete_attempted_pair_inventory_required')
    records = []
    for pair, image, reference in zip(plan['records'], image_scores, radgraph_scores):
        require(pair['item_id'] == image['item_id'] == reference['item_id'], 'exact_ordered_pair_join_required')
        require(image['status'] in STATUSES, 'explicit_image_availability_required')
        value = image['value']
        require((type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1.00001)
            if image['status'] == 'complete' else value is None, 'finite_cosine_or_unavailable_null_required')
        available = image['status'] == 'complete' and reference['status'] == 'complete'
        scores = {METRIC: value if available else None,
                  **{m: reference['scores'][m] if available else None for m in RADGRAPH_METRICS}}
        if available:
            require(all(type(scores[m]) in (int, float) and math.isfinite(scores[m]) and
                        0 <= scores[m] <= 1 for m in RADGRAPH_METRICS), 'official_reference_scores_required')
        records.append({**{k: pair[k] for k in ('item_id', 'source_id', 'source_group_id',
            'section_id', 'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256')},
            'image_score_status': image['status'], 'paired_cohort_available': available,
            'scores': scores, 'expert_outcomes': outcomes(pair['errors'])})
    return records


def delta_interval(rows, resamples, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_group_id']].append(row['metric_minus_random_errors'])
    keys = sorted(groups)
    if len(keys) < 3:
        return {'status': 'insufficient_groups', 'interval95': None, 'source_groups': len(keys)}
    rng, boot = random.Random(seed), []
    for _ in range(resamples):
        sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
        boot.append(sum(sample)/len(sample))
    return {'status': 'complete', 'interval95': [quantile(boot,.025), quantile(boot,.975)],
            'source_groups': len(keys), 'resamples': resamples, 'seed': seed}


def evaluate(records, *, resamples=1000, seed=0):
    require(isinstance(records, list) and records and type(resamples) is int and
            20 <= resamples <= 2000 and type(seed) is int, 'bounded_fixed_benchmark_options_required')
    anchors = defaultdict(list)
    for row in records:
        anchors[(row['source_id'], row['section_id'], row['reference_sha256'])].append(row)
    require(all(sorted(r['candidate_slot'] for r in rows) == [1,2,3] for rows in anchors.values()),
            'fixed_three_candidates_per_anchor_required')
    correlations, choices, selection = {}, [], []
    for metric in METRICS:
        per_target = {}
        for target in TARGETS:
            values = [(r['source_group_id'], r['scores'][metric], -r['expert_outcomes'][target])
                for r in records if r['scores'][metric] is not None and r['expert_outcomes'][target] is not None]
            corr = correlation([(v[1],v[2]) for v in values])
            corr.update(all_attempted_pairs=len(records), paired_coverage=len(values)/len(records),
                cluster_bootstrap=clustered_spearman_interval(values,resamples,seed))
            per_target[target] = corr
            comparable = []
            for index, rows in enumerate(anchors.values()):
                rows = sorted(rows, key=lambda r: r['candidate_slot'])
                scores = [r['scores'][metric] for r in rows]
                errors = [r['expert_outcomes'][target] for r in rows]
                if None in scores or None in errors:
                    continue
                top = [i for i,v in enumerate(scores) if v == max(scores)]
                selected = sum(errors[i] for i in top)/len(top)
                random_mean = sum(errors)/3
                strict = correct = ties = 0
                for i,j in ((0,1),(0,2),(1,2)):
                    if errors[i] == errors[j]:
                        continue
                    strict += 1
                    if scores[i] == scores[j]:
                        correct += .5
                        ties += 1
                    else:
                        correct += int((scores[i] > scores[j]) == (errors[i] < errors[j]))
                row = {'anchor_id': f'anchor_{index:04d}', 'source_group_id': rows[0]['source_group_id'],
                    'metric': metric, 'target': target, 'top_tie_size': len(top),
                    'selected_expected_errors': selected, 'random_expected_errors': random_mean,
                    'oracle_errors': min(errors), 'metric_minus_random_errors': selected - random_mean,
                    'strict_pairs': strict, 'expected_correct_pairs': correct, 'tied_score_pairs': ties}
                choices.append(row)
                comparable.append(row)
            strict = sum(r['strict_pairs'] for r in comparable)
            mean_fields = ('selected_expected_errors', 'random_expected_errors', 'oracle_errors', 'metric_minus_random_errors')
            selection.append({'metric': metric, 'target': target, 'attempted_anchors': len(anchors),
                'complete_anchors': len(comparable), 'all_attempted_anchor_coverage': len(comparable)/len(anchors),
                'means': {key: sum(r[key] for r in comparable)/len(comparable) if comparable else None for key in mean_fields},
                'error_delta_cluster_ci': delta_interval(comparable,resamples,seed),
                'strict_candidate_pairs': strict,
                'pairwise_accuracy': sum(r['expected_correct_pairs'] for r in comparable)/strict if strict else None,
                'score_tied_strict_pairs': sum(r['tied_score_pairs'] for r in comparable)})
        correlations[metric] = per_target
    return {'schema_version': VERSION, 'reference_free_biovil_inputs': True,
        'reference_free_radgraph_baselines': False, 'expert_evaluation_reference_based': True,
        'clinical_qualified': False, 'selection_changed': False, 'scorer_fitting': False,
        'common_cohort_for_all_four_metrics': True, 'all_attempted_pairs': len(records),
        'common_score_available_pairs': sum(r['paired_cohort_available'] for r in records),
        'correlations': correlations, 'selection_diagnostic': selection, 'analytical_choices': choices,
        'primary_endpoint': 'clinically_significant_total_correlation',
        'secondary_endpoint': 'within_anchor_expected_error_difference_vs_uniform_choice',
        'score_tie_policy': 'uniform_expected_choice_over_exact_maximum_ties',
        'mimic_training_overlap_verified': False, 'synthetic_domain_transfer_verified': False,
        'independent_image_radiologist_adjudication': False,
        'error_count_source': 'released_expert_report_reference_comparison_not_new_image_adjudication'}

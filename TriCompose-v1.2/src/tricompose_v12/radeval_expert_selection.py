"""Post-hoc within-anchor ranking diagnostic, never a clinical selector.

Reference-based official metrics and released expert counts only. No models,
score fitting, source text, actual candidate changes, or repair claims.
"""
from collections import defaultdict
import math
import random

from .radeval_expert import outcomes, require
from .radgraph_reference_contract import METRICS
from .report_metric_alignment import quantile

TARGETS = ('clinically_significant_total', 'all_errors_total')


def replay(plan, scores):
    require(len(scores) == len(plan['records']) and all(p['item_id'] == s['item_id']
        for p, s in zip(plan['records'], scores)), 'exact_pair_score_join_required')
    anchors = {}
    for pair, score in zip(plan['records'], scores):
        key = pair['source_id'], pair['section_id'], pair['reference_sha256']
        anchor = anchors.setdefault(key, {'anchor_id': f'anchor_{len(anchors):04d}',
            'source_group_id': pair['source_group_id'], 'pairs': {}})
        require(anchor['source_group_id'] == pair['source_group_id'] and
                pair['candidate_slot'] not in anchor['pairs'], 'unambiguous_fixed_anchor_required')
        anchor['pairs'][pair['candidate_slot']] = pair, score
    records = []
    for anchor in anchors.values():
        require(set(anchor['pairs']) == {1, 2, 3}, 'three_released_slots_per_anchor_required')
        ordered = [anchor['pairs'][i] for i in (1, 2, 3)]
        for target in TARGETS:
            errors = [outcomes(p['errors'])[target] for p, _ in ordered]
            for metric in METRICS:
                values = [s['scores'][metric] for _, s in ordered]
                available = all(s['status'] == 'complete' for _, s in ordered) and all(e is not None for e in errors)
                record = {'anchor_id': anchor['anchor_id'], 'source_group_id': anchor['source_group_id'],
                    'target': target, 'metric': metric, 'status': 'complete' if available else 'incomplete_anchor',
                    'top_tie_size': None, 'uniform_random_expected_errors': None,
                    'fixed_slot_1_errors': None, 'fixed_slot_2_errors': None, 'fixed_slot_3_errors': None,
                    'metric_top_expected_errors': None, 'oracle_min_errors': None,
                    'metric_minus_random_errors': None, 'metric_oracle_hit_probability': None,
                    'random_oracle_hit_probability': None, 'strict_expert_pair_comparisons': None,
                    'expected_metric_pairwise_correct': None, 'metric_score_tied_strict_pairs': None}
                if available:
                    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
                            'finite_official_scores_required')
                    best = [i for i, v in enumerate(values) if v == max(values)]
                    oracle = [i for i, e in enumerate(errors) if e == min(errors)]
                    selected = sum(errors[i] for i in best) / len(best)
                    random_mean = sum(errors) / 3
                    strict = correct = tied = 0
                    for i, j in ((0, 1), (0, 2), (1, 2)):
                        if errors[i] == errors[j]:
                            continue
                        strict += 1
                        if values[i] == values[j]:
                            correct += .5
                            tied += 1
                        elif (values[i] > values[j]) == (errors[i] < errors[j]):
                            correct += 1
                    record.update(top_tie_size=len(best),
                        uniform_random_expected_errors=random_mean,
                        **{f'fixed_slot_{i+1}_errors': value for i, value in enumerate(errors)},
                        metric_top_expected_errors=selected, oracle_min_errors=min(errors),
                        metric_minus_random_errors=selected - random_mean,
                        metric_oracle_hit_probability=len(set(best) & set(oracle)) / len(best),
                        random_oracle_hit_probability=len(oracle) / 3,
                        strict_expert_pair_comparisons=strict,
                        expected_metric_pairwise_correct=correct,
                        metric_score_tied_strict_pairs=tied)
                records.append(record)
    return {'schema_version': 'radeval-expert-within-anchor-replay-v1',
        'post_hoc_diagnostic': True, 'reference_free': False,
        'clinical_qualified': False, 'selection_changed': False,
        'anchor_inventory': len(anchors), 'records': records,
        'score_tie_policy': 'uniform_expected_choice_over_exact_maximum_ties',
        'missing_policy': 'all_three_expert_counts_and_scores_required_per_anchor_target'}


def summarize(replayed, *, resamples=1000, seed=0):
    require(type(resamples) is int and 20 <= resamples <= 2000 and type(seed) is int,
            'fixed_bounded_cluster_bootstrap_required')
    summaries = []
    for target in TARGETS:
        for metric in METRICS:
            all_rows = [r for r in replayed['records'] if r['target'] == target and r['metric'] == metric]
            rows = [r for r in all_rows if r['status'] == 'complete']
            groups = defaultdict(list)
            for row in rows:
                groups[row['source_group_id']].append(row['metric_minus_random_errors'])
            keys = sorted(groups)
            rng, boot = random.Random(seed), []
            if len(keys) >= 3:
                for _ in range(resamples):
                    values = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
                    boot.append(sum(values)/len(values))
            mean_fields = ('uniform_random_expected_errors', 'fixed_slot_1_errors', 'fixed_slot_2_errors',
                'fixed_slot_3_errors', 'metric_top_expected_errors', 'oracle_min_errors',
                'metric_minus_random_errors', 'metric_oracle_hit_probability', 'random_oracle_hit_probability')
            strict = sum(r['strict_expert_pair_comparisons'] for r in rows)
            summaries.append({'target': target, 'metric': metric, 'attempted_anchors': len(all_rows),
                'eligible_anchors': len(rows), 'eligible_source_groups': len(keys),
                'anchor_coverage': len(rows)/len(all_rows),
                'means': {field: sum(r[field] for r in rows)/len(rows) if rows else None for field in mean_fields},
                'metric_minus_random_error_delta_ci95': [quantile(boot, .025), quantile(boot, .975)] if boot else None,
                'metric_tied_top_anchors': sum(r['top_tie_size'] > 1 for r in rows),
                'strict_expert_pair_comparisons': strict,
                'metric_score_tied_strict_pairs': sum(r['metric_score_tied_strict_pairs'] for r in rows),
                'expected_pairwise_accuracy': sum(r['expected_metric_pairwise_correct'] for r in rows)/strict if strict else None,
                'bootstrap_resamples': resamples, 'seed': seed})
    return {'schema_version': 'radeval-expert-within-anchor-summary-v1',
        'post_hoc_diagnostic': True, 'clinical_qualified': False,
        'selection_changed': False, 'summaries': summaries,
        'error_delta_direction': 'metric_selected_errors_minus_uniform_random_errors_lower_is_better',
        'reference_based_replay_not_actual_synthetic_bank_selection': True}

"""Frozen report-reference MedCPT diagnostic, not a clinical selector.

Pure opaque-score/count joins; no models, IO, fitting, source text or repair.
Use identical available cohorts for comparators, preserving missing rows.
"""
from collections import defaultdict
import math

from .radeval_expert import POLICY as EXPERT_POLICY, VERSION as EXPERT_VERSION, outcomes, require
from .radgraph_reference_contract import METRICS as RADGRAPH
from .report_metric_alignment import correlation, clustered_spearman_interval
from .radeval_image_benchmark import delta_interval

VERSION = 'tricompose-radeval-medcpt-benchmark-v1'
METRIC = 'medcpt_query_fulltext_cosine'
BIOVIL = 'biovil_t_raw_cosine'
FULL_METRICS = (METRIC, *RADGRAPH)
IMAGE_METRICS = (METRIC, BIOVIL, *RADGRAPH)
TARGETS = ('clinically_significant_total', 'all_errors_total')
STATUSES = ('complete', 'empty_input', 'over_capacity', 'failed_embedding')
POLICY = {**EXPERT_POLICY, 'primary_metric_eligible': False,
          'semantic_equivalence_verified': False, 'current_patient_scope_verified': False}


def finite(value, lower, upper):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def join(plan, medcpt, radgraph, biovil):
    require(plan['schema_version'] == EXPERT_VERSION + '-inventory' and plan['policy'] == EXPERT_POLICY,
            'frozen_expert_inventory_required')
    require(1 <= len(plan['records']) == len(medcpt) == len(radgraph) == len(biovil) <= 1024,
            'all_attempted_ordered_pair_inventory_required')
    records, seen = [], set()
    for pair, semantic, graph, image in zip(plan['records'], medcpt, radgraph, biovil):
        require(pair['item_id'] == semantic['item_id'] == graph['item_id'] == image['item_id']
                and pair['item_id'] not in seen, 'unique_exact_ordered_pair_join_required')
        seen.add(pair['item_id'])
        require(semantic['status'] in STATUSES and
                (finite(semantic['value'], -1, 1) if semantic['status'] == 'complete' else semantic['value'] is None),
                'finite_relatedness_or_unavailable_null_required')
        require(pair['input_nonempty'] if semantic['status'] == 'complete' else True,
                'empty_pair_cannot_be_complete')
        require(graph['status'] in ('complete', 'empty_input', 'unavailable_graph') and
                set(graph['scores']) == set(RADGRAPH) and all(
                    finite(graph['scores'][m], 0, 1) if graph['status'] == 'complete' else graph['scores'][m] is None
                    for m in RADGRAPH), 'native_graph_score_or_explicit_missing_required')
        require(image['status'] in ('complete', 'unavailable_source_image', 'failed_image_or_text') and
                (finite(image['value'], -1.00001, 1.00001) if image['status'] == 'complete' else image['value'] is None),
                'cached_image_score_or_explicit_missing_required')
        full = semantic['status'] == graph['status'] == 'complete'
        common_image = full and image['status'] == 'complete'
        records.append({**{k: pair[k] for k in ('item_id', 'source_id', 'source_group_id',
            'section_id', 'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256')},
            'medcpt_status': semantic['status'], 'radgraph_status': graph['status'],
            'biovil_status': image['status'], 'full_common_available': full,
            'image_common_available': common_image,
            'scores': {METRIC: semantic['value'], BIOVIL: image['value'], **graph['scores']},
            'expert_outcomes': outcomes(pair['errors'])})
    return records


def choices(records, metric, target, mask):
    anchors = defaultdict(list)
    for row in records:
        anchors[(row['source_id'], row['section_id'], row['reference_sha256'])].append(row)
    rows = []
    for index, candidates in enumerate(anchors.values()):
        require(sorted(r['candidate_slot'] for r in candidates) == [1, 2, 3] and
                len({r['source_group_id'] for r in candidates}) == 1, 'fixed_unambiguous_three_slot_anchor_required')
        candidates = sorted(candidates, key=lambda r: r['candidate_slot'])
        complete = all(r[mask] and r['expert_outcomes'][target] is not None for r in candidates)
        row = {'anchor_id': f'anchor_{index:04d}', 'source_group_id': candidates[0]['source_group_id'],
            'status': 'complete' if complete else 'incomplete_anchor', 'top_tie_size': None,
            'selected_expected_errors': None, 'random_expected_errors': None,
            'oracle_errors': None, 'metric_minus_random_errors': None,
            'strict_pairs': None, 'expected_correct_pairs': None, 'score_tied_strict_pairs': None}
        if complete:
            scores = [r['scores'][metric] for r in candidates]
            errors = [r['expert_outcomes'][target] for r in candidates]
            top = [i for i, v in enumerate(scores) if v == max(scores)]
            selected, uniform = sum(errors[i] for i in top) / len(top), sum(errors) / 3
            strict = correct = ties = 0
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if errors[i] == errors[j]:
                    continue
                strict += 1
                if scores[i] == scores[j]:
                    correct += .5
                    ties += 1
                else:
                    correct += int((scores[i] > scores[j]) == (errors[i] < errors[j]))
            row.update(top_tie_size=len(top), selected_expected_errors=selected,
                random_expected_errors=uniform, oracle_errors=min(errors),
                metric_minus_random_errors=selected - uniform, strict_pairs=strict,
                expected_correct_pairs=correct, score_tied_strict_pairs=ties)
        rows.append(row)
    return rows


def evaluate(records, *, resamples=1000, seed=0):
    require(isinstance(records, list) and records and type(resamples) is int and 20 <= resamples <= 2000
            and type(seed) is int, 'bounded_fixed_diagnostic_options_required')
    cohorts = {}
    for name, mask, metrics in (('full_text_common', 'full_common_available', FULL_METRICS),
                                ('cached_image_common', 'image_common_available', IMAGE_METRICS)):
        eligible = [r for r in records if r[mask]]
        metric_results = {}
        for metric in metrics:
            per_outcome = {}
            for target in records[0]['expert_outcomes']:
                values = [(r['source_group_id'], r['scores'][metric], -r['expert_outcomes'][target])
                    for r in eligible if r['expert_outcomes'][target] is not None]
                result = correlation([(v[1], v[2]) for v in values])
                result.update(all_attempted_pairs=len(records), common_available_pairs=len(eligible),
                    paired_coverage=len(values) / len(records),
                    reference_available=sum(r['expert_outcomes'][target] is not None for r in records))
                if target in TARGETS:
                    result['cluster_bootstrap'] = clustered_spearman_interval(values, resamples, seed)
                per_outcome[target] = result
            selection = {}
            for target in TARGETS:
                all_rows = choices(records, metric, target, mask)
                usable = [r for r in all_rows if r['status'] == 'complete']
                strict = sum(r['strict_pairs'] for r in usable)
                fields = ('selected_expected_errors', 'random_expected_errors', 'oracle_errors', 'metric_minus_random_errors')
                selection[target] = {'attempted_anchors': len(all_rows), 'complete_anchors': len(usable),
                    'means': {f: sum(r[f] for r in usable) / len(usable) if usable else None for f in fields},
                    'error_delta_cluster_ci': delta_interval(usable, resamples, seed),
                    'strict_pairs': strict,
                    'expected_pairwise_accuracy': sum(r['expected_correct_pairs'] for r in usable) / strict if strict else None,
                    'score_tied_strict_pairs': sum(r['score_tied_strict_pairs'] for r in usable),
                    'anchor_records': all_rows}
            metric_results[metric] = {'outcomes': per_outcome, 'selection_diagnostic': selection}
        cohorts[name] = {'all_attempted_pairs': len(records), 'common_score_available_pairs': len(eligible),
            'common_source_groups': len({r['source_group_id'] for r in eligible}),
            'same_available_mask_for_all_comparators': True, 'metrics': metric_results}
    return {'schema_version': VERSION, 'policy': dict(POLICY), 'cohorts': cohorts,
        'bootstrap_resamples': resamples, 'seed': seed,
        'primary_outcome': 'clinically_significant_total', 'secondary_outcome': 'all_errors_total',
        'correlation_direction': 'higher_relatedness_vs_negative_expert_error_count',
        'selection_diagnostic_not_actual_bank_selection': True,
        'score_tie_policy': 'uniform_expected_choice_over_exact_maximum_ties',
        'partial_anchor_candidate_dropping_allowed': False,
        'medcpt_and_radgraph_reference_based': True, 'biovil_reference_report_input': False,
        'expert_evaluation_reference_based': True, 'untouched_final_test': False}

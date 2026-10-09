"""Expert-count diagnostic for native literal proposals, never clinical truth.

No models, IO, aliases, threshold fitting, entity gold, image/EHR truth or repair.
All author error categories are retained. Exact unmatched spans are not omissions.
"""
from collections import defaultdict
import re

from .fact_comparison_contract import VERSION as FACT_VERSION, POLICY as FACT_POLICY, digest
from .radeval_expert import VERSION as EXPERT_VERSION, POLICY as EXPERT_POLICY, outcomes, require
from .report_metric_alignment import correlation, clustered_spearman_interval
from .radeval_medcpt_benchmark import choices
from .radeval_image_benchmark import delta_interval

VERSION = 'tricompose-radeval-literal-fact-diagnostic-v1'
FEATURES = (
    'polarity_opposition_proposals', 'different_anatomy_context_concepts',
    'modifier_token_set_differences', 'hypothesis_only_literal_atoms',
    'reference_only_literal_atoms', 'uncertain_or_mixed_literal_atoms',
)
TARGETED = {
    'polarity_opposition_proposals': 'clinically_significant_false_prediction',
    'different_anatomy_context_concepts': 'clinically_significant_incorrect_location',
}
POLICY = {'clinical_qualified': False, 'entity_extraction_accuracy_verified': False,
    'negation_specific_accuracy_verified': False, 'current_patient_scope_verified': False,
    'image_truth_verified': False, 'ehr_truth_verified': False,
    'unmatched_span_is_omission': False, 'modifier_difference_is_severity_error': False,
    'missing_is_zero': False, 'selection_changed': False, 'regeneration_authorized': False,
    'independent_truth_votes': False, 'weight_or_threshold_fitting': False}


def feature_row(comparison):
    fields = {'schema_version', 'case_id', 'left_adapter_id', 'right_adapter_id',
        'left_source_artifact_sha256', 'right_source_artifact_sha256', 'presence',
        'anatomy', 'severity', 'temporality', 'experiencer', 'modifier_difference_not_severity',
        'clinical_score', 'confirmed_faulty_modality', 'policy', 'comparison_id'}
    require(isinstance(comparison, dict) and
        set(comparison) == fields and comparison['comparison_id'] == 'comparison_' +
        digest({k: v for k, v in comparison.items() if k != 'comparison_id'}) and
        comparison.get('schema_version') == FACT_VERSION + '-native-report-comparison' and
        comparison.get('policy') == FACT_POLICY and comparison.get('clinical_score') is None and
        comparison.get('confirmed_faulty_modality') is None, 'unqualified_native_comparison_required')
    require(all(comparison.get(k) == {'status': 'unavailable_not_extracted', 'comparison': None}
                for k in ('severity', 'temporality', 'experiencer')), 'unavailable_dimensions_must_remain_unavailable')
    presence = comparison['presence']
    require(presence['status'] == 'native_proposals_only' and presence['current_scope_verified'] is False,
            'native_scope_not_qualified')
    counts = presence['counts']
    names = {'native_state_agreement', 'explicit_polarity_opposition_proposal',
        'uncertain_or_mixed_state', 'anatomy_context_not_definite', 'unmentioned_in_left', 'unmentioned_in_right'}
    require(set(counts) == names and all(type(v) is int and v >= 0 for v in counts.values()),
            'complete_nonnegative_native_count_inventory_required')
    require(len(presence['details']) == sum(counts.values()) and all(
        sum(d['relation'] == name for d in presence['details']) == count for name, count in counts.items()),
        'native_details_count_binding_required')
    anatomy = comparison['anatomy']['different_context_concepts']
    modifiers = comparison['modifier_difference_not_severity']
    require(comparison['anatomy']['status'] == 'context_differences_not_exclusive' and
            all(type(v) is int and v >= 0 for v in (anatomy, modifiers)), 'nonnegative_context_counts_required')
    # Left is hypothesis, right reference. No normalization or sign flipping.
    return dict(zip(FEATURES, (
        counts['explicit_polarity_opposition_proposal'], anatomy, modifiers,
        counts['unmentioned_in_right'], counts['unmentioned_in_left'],
        counts['uncertain_or_mixed_state'] + counts['anatomy_context_not_definite'],
    )))


def join(plan, predictions):
    require(plan['schema_version'] == EXPERT_VERSION + '-inventory' and plan['policy'] == EXPERT_POLICY,
            'frozen_expert_inventory_required')
    require(1 <= len(plan['records']) == len(predictions) <= 1024,
            'all_attempted_ordered_inventory_required')
    result, seen, anchors = [], set(), defaultdict(list)
    for pair, pred in zip(plan['records'], predictions):
        require(set(pred) == {'item_id', 'status', 'comparison'} and
                pair['item_id'] == pred['item_id'] and pair['item_id'] not in seen,
                'unique_ordered_pair_join_required')
        seen.add(pair['item_id'])
        for key, pattern in (('item_id', r'pair_[0-9]{4}'), ('source_id', r'source_[0-9]{4}'),
            ('source_group_id', r'group_[0-9]{4}'), ('section_id', r'section_[0-9]{2}'),
            ('reference_sha256', r'[0-9a-f]{64}'), ('hypothesis_sha256', r'[0-9a-f]{64}')):
            require(isinstance(pair[key], str) and re.fullmatch(pattern, pair[key]), 'opaque_ids_hashes_required')
        require(type(pair['candidate_slot']) is int and pair['candidate_slot'] in (1, 2, 3), 'fixed_slot_required')
        require(pred['status'] in ('complete', 'empty_input', 'unavailable_graph'), 'declared_availability_required')
        if pred['status'] == 'complete':
            require(pair['input_nonempty'], 'empty_cannot_be_complete')
            c = pred['comparison']
            require(c['left_source_artifact_sha256'] == pair['hypothesis_sha256'] and
                c['right_source_artifact_sha256'] == pair['reference_sha256'], 'exact_directional_text_lineage_required')
            features = feature_row(c)
        else:
            require(pred['comparison'] is None, 'unavailable_must_not_be_zero_or_comparison')
            features = dict.fromkeys(FEATURES)
        row = {**{k: pair[k] for k in ('item_id', 'source_id', 'source_group_id', 'section_id',
            'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256')},
            'status': pred['status'], 'features': features, 'expert_outcomes': outcomes(pair['errors'])}
        result.append(row)
        anchors[(pair['source_id'], pair['section_id'], pair['reference_sha256'])].append(row)
    require(all(sorted(r['candidate_slot'] for r in group) == [1, 2, 3] and
                len({r['source_group_id'] for r in group}) == 1 for group in anchors.values()),
            'three_same_group_slots_per_anchor_required')
    return result


def binary_diagnostic(values):
    """Fixed count>0 proposal vs any expert category error, NOT entity accuracy."""
    require(all(type(count) is int and count >= 0 and type(error) in (int, float) and error >= 0
                for count, error in values), 'nonnegative_counts_required')
    tp = sum(c > 0 and e > 0 for c, e in values)
    fp = sum(c > 0 and e == 0 for c, e in values)
    fn = sum(c == 0 and e > 0 for c, e in values)
    tn = sum(c == 0 and e == 0 for c, e in values)
    positives = [c for c, e in values if e > 0]
    negatives = [c for c, e in values if e == 0]
    auc = (sum(int(p > n) + .5 * int(p == n) for p in positives for n in negatives) /
           (len(positives) * len(negatives))) if positives and negatives else None
    return {'paired_rows': len(values), 'expert_error_positive_pairs': len(positives),
        'proposal_positive_pairs': tp + fp, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
        'specificity': tn / (tn + fp) if tn + fp else None, 'count_auroc': auc,
        'target_is_any_expert_category_error_not_entity_gold': True,
        'fixed_alarm_definition': 'literal_count_greater_than_zero_not_fitted'}


def evaluate(records, *, resamples=1000, seed=0):
    require(records and type(resamples) is int and 20 <= resamples <= 2000 and type(seed) is int,
            'bounded_fixed_bootstrap_required')
    result = []
    ranking_rows = [{**r, 'full_common_available': r['status'] == 'complete',
        'scores': {feature: -r['features'][feature] if r['status'] == 'complete' else None
                   for feature in FEATURES}} for r in records]
    for feature in FEATURES:
        for target in sorted(records[0]['expert_outcomes']):
            eligible = [r for r in records if r['status'] == 'complete' and r['expert_outcomes'][target] is not None]
            values = [(r['features'][feature], r['expert_outcomes'][target]) for r in eligible]
            item = {'feature': feature, 'target': target, 'all_attempted_pairs': len(records),
                'paired_coverage': len(values) / len(records),
                'correlation': correlation(values), 'binary_diagnostic': binary_diagnostic(values)}
            anchor_rows = choices(ranking_rows, feature, target, 'full_common_available')
            usable = [a for a in anchor_rows if a['status'] == 'complete']
            strict = sum(a['strict_pairs'] for a in usable)
            fields = ('selected_expected_errors', 'random_expected_errors', 'oracle_errors', 'metric_minus_random_errors')
            item['within_anchor_diagnostic'] = {'attempted_anchors': len(anchor_rows),
                'complete_anchors': len(usable), 'strict_candidate_pairs': strict,
                'expected_pairwise_accuracy': sum(a['expected_correct_pairs'] for a in usable) / strict if strict else None,
                'means': {f: sum(a[f] for a in usable) / len(usable) if usable else None for f in fields},
                'anchor_rows': anchor_rows}
            if target in ('clinically_significant_total', 'all_errors_total', TARGETED.get(feature)):
                item['cluster_bootstrap'] = clustered_spearman_interval(
                    [(r['source_group_id'], r['features'][feature], r['expert_outcomes'][target]) for r in eligible],
                    resamples, seed)
                item['within_anchor_diagnostic']['error_delta_cluster_ci'] = delta_interval(usable, resamples, seed)
            result.append(item)
    return {'schema_version': VERSION, 'policy': dict(POLICY), 'all_attempted_pairs': len(records),
        'complete_pairs': sum(r['status'] == 'complete' for r in records),
        'features': list(FEATURES), 'results': result, 'bootstrap_resamples': resamples, 'seed': seed,
        'correlation_direction': 'more_literal_differences_vs_more_expert_errors',
        'targeted_hypotheses': TARGETED, 'post_hoc_development_diagnostic': True,
        'independently_held_out_or_blinded': False, 'multiplicity_adjusted': False,
        'reference_based': True, 'same_generated_report_consensus_is_reference_truth': False,
        'ranking_direction': 'minimize_each_separate_literal_count_not_fitted',
        'exact_minimum_count_ties': 'uniform_expected_choice_not_first_slot',
        'actual_candidate_selection_performed': False,
        'native_model_calls': 0, 'severity_scope_accuracy': None,
        'clinical_score': None, 'confirmed_faulty_modality': None}

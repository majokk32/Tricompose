"""Cached image-subset ranking versus metadata-only negative controls.

No clinical text, image, model, fitted score, threshold or selection mutation.
The already inspected expert cohort is a post-hoc development diagnostic.
"""
import math
import re

from .radeval_expert import CATEGORIES, SEVERITIES, outcomes, require
from .radeval_image_benchmark import METRICS, TARGETS, STATUSES, delta_interval
from .radeval_length_controls import CONTROLS, IDENTITY, SELECTORS as OLD_SELECTORS
from .radeval_medcpt_benchmark import choices
from .report_metric_alignment import correlation

VERSION = 'tricompose-radeval-image-length-controls-v1'
SELECTORS = (*METRICS, *CONTROLS)
POLICY = {'clinical_qualified': False, 'selection_changed': False,
    'regeneration_authorized': False, 'weight_or_threshold_fitting': False,
    'image_truth_verified': False, 'ehr_truth_verified': False,
    'checkpoint_training_overlap_verified': False, 'synthetic_transfer_verified': False,
    'length_confounding_conclusively_excluded': False, 'untouched_final_test': False,
    'missing_is_zero': False, 'entity_count_is_clinical_quality': False}


def number(value, lo, hi):
    return type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi


def validate_outcomes(value):
    require(isinstance(value, dict), 'explicit_expert_count_dictionary_required')
    keys = {'all_errors_total', *(s + '_total' for s in SEVERITIES),
            *(s + '_' + c for s in SEVERITIES for c in CATEGORIES)}
    require(set(value) == keys and all(v is None or number(v, 0, 14000) for v in value.values()),
            'fixed_nonnegative_or_missing_expert_counts_required')
    errors = {s: [value[s + '_' + c] for c in CATEGORIES] for s in SEVERITIES}
    require(outcomes(errors) == value, 'expert_totals_must_match_released_category_counts')


def join(image_rows, metadata_rows):
    require(isinstance(image_rows, list) and isinstance(metadata_rows, list) and
            1 <= len(image_rows) == len(metadata_rows) <= 1024,
            'all_attempted_ordered_pair_inventory_required')
    rows, seen = [], set()
    for image, metadata in zip(image_rows, metadata_rows):
        require(isinstance(image, dict) and set(image) == {*IDENTITY, 'image_score_status',
            'paired_cohort_available', 'scores', 'expert_outcomes'} and
            isinstance(metadata, dict) and set(metadata) == {*IDENTITY, 'common_available',
            'costs', 'expert_outcomes', 'lengths', 'native_entity_count'},
            'exact_metadata_only_source_schemas_required')
        require(all(image[k] == metadata[k] for k in IDENTITY) and image['item_id'] not in seen,
                'unique_exact_ordered_hash_bound_join_required')
        seen.add(image['item_id'])
        for key, pattern in (('item_id', r'pair_[0-9]{4}'), ('source_id', r'source_[0-9]{4}'),
            ('source_group_id', r'group_[0-9]{4}'), ('section_id', r'section_[0-9]{2}'),
            ('reference_sha256', r'[a-f0-9]{64}'), ('hypothesis_sha256', r'[a-f0-9]{64}')):
            require(isinstance(image[key], str) and re.fullmatch(pattern, image[key]),
                    'opaque_identity_and_hash_required')
        require(type(image['candidate_slot']) is int and image['candidate_slot'] in (1, 2, 3) and
                image['section_name'] in ('findings', 'impression', 'other_author_section'),
                'fixed_section_and_three_slots_required')
        validate_outcomes(image['expert_outcomes'])
        require(image['expert_outcomes'] == metadata['expert_outcomes'], 'unchanged_expert_counts_required')
        require(type(image['paired_cohort_available']) is bool and
                type(metadata['common_available']) is bool and image['image_score_status'] in STATUSES and
                (not image['paired_cohort_available'] or image['image_score_status'] == 'complete'),
                'explicit_image_and_metadata_availability_required')
        require(isinstance(image['scores'], dict) and set(image['scores']) == set(METRICS) and
                all(number(v, -1.00001 if k == METRICS[0] else 0, 1.00001 if k == METRICS[0] else 1)
                    if image['paired_cohort_available'] else v is None
                    for k, v in image['scores'].items()), 'finite_cached_scores_or_explicit_null_required')
        require(isinstance(metadata['costs'], dict) and set(metadata['costs']) == set(OLD_SELECTORS) and
                all(type(v) is int and 0 <= v <= 100000 if metadata['common_available'] else v is None
                    for v in metadata['costs'].values()), 'fixed_native_metadata_costs_required')
        require(isinstance(metadata['lengths'], dict) and set(metadata['lengths']) == {'hypothesis', 'reference'} and
                all(v is None or type(v) is int and 1 <= v <= 100000 for v in metadata['lengths'].values()) and
                (metadata['native_entity_count'] is None or type(metadata['native_entity_count']) is int and
                 0 <= metadata['native_entity_count'] <= 4096), 'full_native_length_and_entity_metadata_required')
        if metadata['common_available']:
            lengths, costs = metadata['lengths'], metadata['costs']
            require(None not in lengths.values() and metadata['native_entity_count'] is not None and
                costs[CONTROLS[0]] == lengths['hypothesis'] and
                costs[CONTROLS[1]] == abs(lengths['hypothesis'] - lengths['reference']) and
                costs[CONTROLS[2]] == metadata['native_entity_count'], 'native_controls_must_match_metadata')
        available = image['paired_cohort_available'] and metadata['common_available']
        scores = {**image['scores'], **{c: -metadata['costs'][c] if available else None for c in CONTROLS}}
        if not available:
            scores = dict.fromkeys(SELECTORS)
        rows.append({**{k: image[k] for k in IDENTITY},
            'image_score_status': image['image_score_status'],
            'original_image_cohort_available': image['paired_cohort_available'],
            'common_available': available, 'scores': scores,
            'expert_outcomes': dict(image['expert_outcomes'])})
    return rows


def evaluate(records, *, resamples=1000, seed=0):
    require(isinstance(records, list) and 1 <= len(records) <= 1024 and type(resamples) is int and
            20 <= resamples <= 2000 and type(seed) is int, 'bounded_fixed_statistics_required')
    results, indexed = [], {}
    for selector in SELECTORS:
        for target in TARGETS:
            anchor_rows = choices(records, selector, target, 'common_available')
            usable = [a for a in anchor_rows if a['status'] == 'complete']
            strict = sum(a['strict_pairs'] for a in usable)
            values = [(r['scores'][selector], -r['expert_outcomes'][target]) for r in records
                      if r['common_available'] and r['expert_outcomes'][target] is not None]
            fields = ('selected_expected_errors', 'random_expected_errors', 'oracle_errors', 'metric_minus_random_errors')
            item = {'selector': selector, 'target': target, 'paired_rows': len(values),
                'correlation': correlation(values), 'attempted_anchors': len(anchor_rows),
                'complete_anchors': len(usable), 'anchor_rows': anchor_rows,
                'means': {f: sum(a[f] for a in usable) / len(usable) if usable else None for f in fields},
                'strict_candidate_pairs': strict,
                'pairwise_accuracy': sum(a['expected_correct_pairs'] for a in usable) / strict if strict else None,
                'score_tied_strict_pairs': sum(a['score_tied_strict_pairs'] for a in usable),
                'error_delta_cluster_ci': delta_interval(usable, resamples, seed)}
            results.append(item)
            indexed[(selector, target)] = item
    paired = []
    for metric in METRICS:
        for control in CONTROLS:
            for target in TARGETS:
                differences = []
                for a, b in zip(indexed[(metric, target)]['anchor_rows'], indexed[(control, target)]['anchor_rows']):
                    require(all(a[k] == b[k] for k in ('anchor_id', 'source_group_id', 'status')),
                            'identical_complete_anchor_masks_required')
                    if a['status'] == 'complete':
                        differences.append({'anchor_id': a['anchor_id'], 'source_group_id': a['source_group_id'],
                            'metric_minus_random_errors': a['selected_expected_errors'] - b['selected_expected_errors']})
                paired.append({'metric': metric, 'control': control, 'target': target,
                    'complete_anchors': len(differences), 'paired_anchor_rows': differences,
                    'metric_minus_control_mean_errors': sum(r['metric_minus_random_errors'] for r in differences) / len(differences)
                        if differences else None, 'paired_cluster_ci': delta_interval(differences, resamples, seed)})
    return {'schema_version': VERSION, 'policy': dict(POLICY), 'selectors': list(SELECTORS),
        'all_attempted_pairs': len(records), 'original_image_available_pairs': sum(r['original_image_cohort_available'] for r in records),
        'common_available_pairs': sum(r['common_available'] for r in records), 'results': results,
        'paired_metric_control_comparisons': paired, 'bootstrap_resamples': resamples, 'seed': seed,
        'tie_policy': 'uniform_expected_choice_over_exact_maximum_ties',
        'post_hoc_development_diagnostic': True, 'multiplicity_adjusted': False,
        'same_available_mask_for_all_selectors': True, 'reference_based_selectors': [*METRICS[1:], CONTROLS[1]],
        'reference_free_selector_inputs': [METRICS[0], CONTROLS[0], CONTROLS[2]],
        'expert_evaluation_reference_based': True, 'length_unit': 'cached_medcpt_native_tokens_including_special_tokens',
        'entity_unit': 'all_cached_native_radgraph_entities_not_only_findings',
        'clinical_score': None, 'new_model_calls': 0}

"""Metadata-only controls for literal report-reference ranking diagnostics.

No text, models, learned adjustment, weights or clinical/repair policy. Keep all
six earlier signals rather than retaining only their observed best feature.
"""
from collections import defaultdict
import re

from .radeval_literal_facts import FEATURES
from .radeval_expert import VERSION as EXPERT_VERSION, POLICY as EXPERT_POLICY, outcomes, require
from .radeval_medcpt_benchmark import choices
from .radeval_image_benchmark import delta_interval
from .report_metric_alignment import correlation

VERSION = 'tricompose-radeval-length-controls-v1'
CONTROLS = ('shortest_native_token_length', 'closest_reference_token_length', 'fewest_native_entities')
SELECTORS = (*FEATURES, *CONTROLS)
TARGETS_WITH_INTERVAL = ('clinically_significant_total', 'all_errors_total')
POLICY = {'clinical_qualified': False, 'reference_free': False,
    'entity_extraction_accuracy_verified': False, 'scope_or_severity_accuracy_verified': False,
    'image_truth_verified': False, 'ehr_truth_verified': False,
    'fitted_statistical_adjustment': False, 'weight_or_threshold_fitting': False,
    'length_confounding_conclusively_excluded': False, 'missing_is_zero': False,
    'selection_changed': False, 'regeneration_authorized': False,
    'entity_count_is_clinical_quality': False, 'unmatched_atom_is_clinical_error': False}
IDENTITY = ('item_id', 'source_id', 'source_group_id', 'section_id', 'section_name',
            'candidate_slot', 'reference_sha256', 'hypothesis_sha256')


def token_length(receipt, expected_sha):
    fields = {'text_sha256', 'status', 'failure_type', 'native_token_count',
              'token_ids_sha256', 'truncated', 'adapter_added_prefix'}
    require(isinstance(receipt, dict) and set(receipt) == fields and
        receipt['text_sha256'] == expected_sha and receipt['status'] in
        ('complete', 'empty_input', 'over_capacity', 'failed_embedding') and
        receipt['truncated'] is False and receipt['adapter_added_prefix'] is False,
        'unchanged_full_native_token_receipt_required')
    n = receipt['native_token_count']
    if n is None:
        require(receipt['status'] in ('empty_input', 'failed_embedding') and receipt['token_ids_sha256'] is None,
                'missing_tokenization_explicit_required')
        return None
    require(type(n) is int and 1 <= n <= 100000 and isinstance(receipt['token_ids_sha256'], str)
        and re.fullmatch(r'[a-f0-9]{64}', receipt['token_ids_sha256']), 'full_token_count_and_hash_required')
    require(receipt['status'] != 'empty_input' and
            (n > 512 if receipt['status'] == 'over_capacity' else True), 'declared_token_status_mismatch')
    # A failed embedding does not invalidate successful full tokenization.
    return n


def build_table(plan, literals, tokens, graph_receipts):
    require(plan['schema_version'] == EXPERT_VERSION + '-inventory' and plan['policy'] == EXPERT_POLICY
        and 1 <= len(plan['records']) == len(literals) <= 1024, 'frozen_complete_ordered_inventory_required')
    expected = {g['graph_id']: g['text_sha256'] for g in plan['graphs']}
    require(set(tokens) == set(expected.values()), 'exact_tokenized_text_inventory_required')
    lengths = {sha: token_length(receipt, sha) for sha, receipt in tokens.items()}
    graphs = {g['graph_id']: g for g in graph_receipts}
    require(set(graphs) == set(expected) and len(graphs) == len(graph_receipts), 'exact_cached_graph_receipts_required')
    for key, receipt in graphs.items():
        require(receipt['text_sha256'] == expected[key] and receipt['status'] in
            ('complete', 'empty_input', 'failed_unavailable'), 'same_report_graph_metadata_required')
        if receipt['status'] == 'complete':
            require(isinstance(receipt['metadata'], dict) and type(receipt['metadata']['entity_count']) is int
                and 0 <= receipt['metadata']['entity_count'] <= 4096, 'nonnegative_native_entity_count_required')
        else:
            require(receipt['metadata'] is None, 'missing_native_graph_metadata_must_be_null')
    records, seen = [], set()
    for pair, literal in zip(plan['records'], literals):
        require(set(literal) == {*IDENTITY, 'status', 'features', 'expert_outcomes'} and
            all(pair[k] == literal[k] for k in IDENTITY) and pair['item_id'] not in seen,
            'unique_exact_ordered_literal_join_required')
        seen.add(pair['item_id'])
        for key, pattern in (('item_id', r'pair_[0-9]{4}'), ('source_id', r'source_[0-9]{4}'),
            ('source_group_id', r'group_[0-9]{4}'), ('section_id', r'section_[0-9]{2}'),
            ('reference_sha256', r'[a-f0-9]{64}'), ('hypothesis_sha256', r'[a-f0-9]{64}')):
            require(isinstance(pair[key], str) and re.fullmatch(pattern, pair[key]), 'opaque_id_or_hash_required')
        require(type(pair['candidate_slot']) is int and pair['candidate_slot'] in (1, 2, 3)
            and pair['section_name'] in ('findings', 'impression', 'other_author_section'), 'fixed_section_slot_required')
        require(literal['status'] in ('complete', 'empty_input', 'unavailable_graph') and
            set(literal['features']) == set(FEATURES) and all(
                type(v) is int and v >= 0 if literal['status'] == 'complete' else v is None
                for v in literal['features'].values()), 'literal_counts_or_unavailable_null_required')
        require(literal['expert_outcomes'] == outcomes(pair['errors']), 'unchanged_all_expert_categories_required')
        hyp_length, ref_length = lengths[pair['hypothesis_sha256']], lengths[pair['reference_sha256']]
        hyp_graph = graphs[pair['hypothesis_graph_id']]
        ref_graph = graphs[pair['reference_graph_id']]
        require(expected[pair['hypothesis_graph_id']] == pair['hypothesis_sha256'] and
                expected[pair['reference_graph_id']] == pair['reference_sha256'], 'exact_graph_id_text_binding_required')
        complete = (literal['status'] == 'complete' and hyp_length is not None and ref_length is not None and
            hyp_graph['status'] == ref_graph['status'] == 'complete')
        require(not complete or pair['input_nonempty'], 'empty_pair_cannot_be_complete')
        costs = {**literal['features'], CONTROLS[0]: hyp_length,
                 CONTROLS[1]: abs(hyp_length - ref_length) if hyp_length is not None and ref_length is not None else None,
                 CONTROLS[2]: hyp_graph['metadata']['entity_count'] if hyp_graph['status'] == 'complete' else None}
        if not complete:
            costs = dict.fromkeys(SELECTORS)
        records.append({**{k: pair[k] for k in IDENTITY},
            'common_available': complete, 'costs': costs, 'expert_outcomes': literal['expert_outcomes'],
            'lengths': {'hypothesis': hyp_length, 'reference': ref_length},
            'native_entity_count': hyp_graph['metadata']['entity_count'] if hyp_graph['status'] == 'complete' else None})
    anchors = defaultdict(list)
    for r in records:
        anchors[(r['source_id'], r['section_id'], r['reference_sha256'])].append(r)
    require(all(sorted(r['candidate_slot'] for r in group) == [1, 2, 3] and
        len({r['source_group_id'] for r in group}) == 1 for group in anchors.values()), 'fixed_three_slot_anchor_required')
    return records


def evaluate(records, *, resamples=1000, seed=0):
    require(records and type(resamples) is int and 20 <= resamples <= 2000 and type(seed) is int,
            'fixed_bounded_statistics_required')
    rankings = [{**r, 'scores': {name: -r['costs'][name] if r['common_available'] else None for name in SELECTORS}}
                for r in records]
    results, indexed = [], {}
    for selector in SELECTORS:
        for target in sorted(records[0]['expert_outcomes']):
            eligible = [r for r in records if r['common_available'] and r['expert_outcomes'][target] is not None]
            all_anchors = choices(rankings, selector, target, 'common_available')
            usable = [a for a in all_anchors if a['status'] == 'complete']
            strict = sum(a['strict_pairs'] for a in usable)
            fields = ('selected_expected_errors', 'random_expected_errors', 'oracle_errors', 'metric_minus_random_errors')
            item = {'selector': selector, 'target': target, 'paired_rows': len(eligible),
                'correlation': correlation([(r['costs'][selector], r['expert_outcomes'][target]) for r in eligible]),
                'attempted_anchors': len(all_anchors), 'complete_anchors': len(usable),
                'means': {f: sum(a[f] for a in usable) / len(usable) if usable else None for f in fields},
                'expected_pairwise_accuracy': sum(a['expected_correct_pairs'] for a in usable) / strict if strict else None,
                'anchor_rows': all_anchors}
            if target in TARGETS_WITH_INTERVAL:
                item['error_delta_cluster_ci'] = delta_interval(usable, resamples, seed)
            results.append(item)
            indexed[(selector, target)] = item
    paired = []
    for feature in FEATURES:
        for control in CONTROLS:
            for target in sorted(records[0]['expert_outcomes']):
                a, b = indexed[(feature, target)], indexed[(control, target)]
                rows = []
                for literal, baseline in zip(a['anchor_rows'], b['anchor_rows']):
                    require(literal['anchor_id'] == baseline['anchor_id'] and
                        literal['source_group_id'] == baseline['source_group_id'] and
                        literal['status'] == baseline['status'], 'identical_available_anchor_mask_required')
                    if literal['status'] == 'complete':
                        rows.append({'anchor_id': literal['anchor_id'], 'source_group_id': literal['source_group_id'],
                            'metric_minus_random_errors': literal['selected_expected_errors'] - baseline['selected_expected_errors']})
                item = {'feature': feature, 'control': control, 'target': target, 'complete_anchors': len(rows),
                    'literal_minus_control_mean_errors': sum(r['metric_minus_random_errors'] for r in rows) / len(rows) if rows else None}
                if target in TARGETS_WITH_INTERVAL:
                    item['paired_cluster_ci'] = delta_interval(rows, resamples, seed)
                    item['paired_anchor_rows'] = rows
                paired.append(item)
    return {'schema_version': VERSION, 'policy': dict(POLICY), 'all_attempted_pairs': len(records),
        'common_available_pairs': sum(r['common_available'] for r in records), 'selectors': list(SELECTORS),
        'results': results, 'paired_literal_control_comparisons': paired,
        'bootstrap_resamples': resamples, 'seed': seed, 'minimum_cost_tie_policy': 'uniform_expected_choice',
        'previous_best_feature_known_before_controls': True, 'post_hoc_development_diagnostic': True,
        'untouched_or_blinded_test': False, 'multiplicity_adjusted': False, 'actual_bank_selection_performed': False,
        'length_unit': 'cached_medcpt_native_token_count_including_special_tokens',
        'entity_unit': 'all_cached_native_radgraph_entities_not_only_findings',
        'shared_available_mask_for_all_selectors': True, 'clinical_score': None, 'new_model_calls': 0}

"""Describe existing choices under fixed masks; never choose or regenerate.

All seed replicates are averaged within their fixed synthetic EHR first. Missing
choices and abstentions retain the full denominator. Dependent reused artifacts
and post-hoc development choices do not support clinical superiority claims.
"""
from collections import defaultdict
import math

from .candidate_image_masks import POLICIES, CONTRASTS, PREFIX, relation
from .radgraph_reference_contract import require

METHODS = ('fixed', 'random', 'static_rerank', 'targeted_heuristic',
           'random_acquisition_score_free_final')
CAPS = (4, 8, 12, 20, 30)
REPLICATES = dict(zip(METHODS, (1, 5, 1, 1, 25)))
COUNTERS = ('selected', 'accepted_image', 'comparable', 'proxy_support',
            'proxy_opposition', 'not_comparable', 'simulated_calls')
METHOD_CONTRASTS = tuple((m, 'fixed') for m in METHODS if m != 'fixed') + (
    ('random', 'random_acquisition_score_free_final'),)


def counters(state, report, selected, calls):
    value = relation(state, report)
    return dict(selected=int(selected), accepted_image=int(state is not None),
        comparable=int(value != 'not_comparable'), proxy_support=int(value == 'proxy_support'),
        proxy_opposition=int(value == 'proxy_opposition'),
        not_comparable=int(value == 'not_comparable'), simulated_calls=calls)


def attach(trials, rows, *, cases, caps=CAPS):
    require(cases and len(set(cases)) == len(cases) and caps and set(caps) <= set(CAPS),
            'explicit_fixed_case_and_budget_inventory_required')
    candidates = {r['triple_candidate_id']: r for r in rows}
    require(len(candidates) == len(rows) and rows, 'unique_unchanged_candidate_table_required')
    fixed = {}
    for r in rows:
        require(r['case_id'] in cases and r['opacity_cached_ehr_state'] == 'unknown',
                'same_fixed_opacity_unknown_ehr_scope_required')
        require(fixed.setdefault(r['case_id'], r['ehr_sha256']) == r['ehr_sha256'],
                'fixed_ehr_hash_required')
        require(all(r[PREFIX + f] == 'False' for f in ('clinical_primary_eligible',
            'reference_metric_transferred', 'synthetic_domain_transport_validated',
            'selector_used', 'regeneration_authorized')), 'diagnostic_mask_not_action_authority_required')
        for p in POLICIES:
            state, status = r[PREFIX + p + '_state'] or None, r[PREFIX + p + '_status']
            require(state in (None, 'positive', 'negative') and
                ((status == 'accepted_determinate') if state is not None else
                 (status == 'unavailable_reader' or status.startswith('abstain_'))),
                'explicit_mask_abstention_or_state_required')
            require(r[PREFIX + p + '_report_relation'] == relation(state, r['opacity_cached_report_state']),
                    'bound_cached_report_relation_required')
    require(set(fixed) == set(cases), 'all_ehrs_retained_required')
    groups, seen, output = defaultdict(list), set(), []
    for t in trials:
        require(t['head_definition'] == 'exact_opacity_0_5', 'one_existing_trial_not_both_head_replicas_required')
        key = t['case_id'], t['method'], t['model_call_budget']
        require(key[0] in cases and key[1] in METHODS and key[2] in caps, 'fixed_trial_grid_required')
        seeds = t['acquisition_seed'], t['final_choice_seed']
        if t['method'] == 'random':
            require(seeds[0] in range(5) and seeds[1] is None, 'all_five_acquisition_seeds_required')
        elif t['method'] == METHODS[-1]:
            require(seeds[0] in range(5) and seeds[1] in range(5), 'all_25_fixed_seed_pairs_required')
        else:
            require(seeds == (None, None), 'deterministic_method_no_seed_replicates_required')
        require((*key, *seeds) not in seen, 'unique_frozen_trial_required')
        seen.add((*key, *seeds))
        require(t['selected_ehr_sha256'] == fixed[key[0]] and type(t['simulated_calls']) in (int, float)
                and math.isfinite(t['simulated_calls']) and 0 <= t['simulated_calls'] <= key[2],
                'unchanged_fixed_ehr_and_historical_call_count_required')
        cid = t['selected_candidate_id']
        row = candidates.get(cid) if cid is not None else None
        require(cid is None or row is not None, 'frozen_choice_cannot_be_replaced_required')
        require((t['selected'] == 1) == (row is not None), 'missing_choice_retained_required')
        if row:
            require(all(t[a] == row[b] for a, b in (
                ('case_id', 'case_id'), ('selected_ehr_sha256', 'ehr_sha256'),
                ('selected_cxr_sha256', 'cxr_sha256'), ('selected_report_sha256', 'report_sha256'))),
                'same_frozen_choice_artifact_lineage_required')
            require(t['image_state'] == row['opacity_exact_state_0_5'] and
                    t['report_state'] == row['opacity_cached_report_state'], 'unchanged_old_exact_opacity_states_required')
        require(t['clinical_accuracy'] is None and t['ehr_cxr_clinical_score'] is None
                and t['ehr_report_clinical_score'] is None, 'historical_unqualified_scope_required')
        report = row['opacity_cached_report_state'] if row else 'unknown'
        masks = {}
        for p in POLICIES:
            state = row[PREFIX + p + '_state'] or None if row else None
            masks[p] = counters(state, report, row is not None, t['simulated_calls'])
        # Persist exact source trial bindings, not source bodies or a new choice.
        out = {k: t[k] for k in ('case_id', 'method', 'model_call_budget', 'acquisition_seed',
            'final_choice_seed', 'frozen_trial_sha256', 'selected_candidate_id', 'selected_ehr_sha256',
            'selected_cxr_sha256', 'selected_report_sha256', 'simulated_calls')}
        out['masks'] = masks
        output.append(out)
        groups[key].append(out)
    expected = {(c, m, cap) for c in cases for m in METHODS for cap in caps}
    require(set(groups) == expected and all(len(v) == REPLICATES[k[1]] for k, v in groups.items()),
            'all_cases_methods_budgets_seed_replicates_required')
    means = []
    for (case, method, cap), group in sorted(groups.items()):
        for p in POLICIES:
            means.append({'case_id': case, 'method': method, 'model_call_budget': cap, 'mask': p,
                'selected_ehr_sha256': fixed[case], 'seed_replicates': len(group),
                **{f: sum(r['masks'][p][f] for r in group) / len(group) for f in COUNTERS}})
    return output, means


def summarize(means, *, cases, caps=CAPS):
    index = {(r['case_id'], r['method'], r['model_call_budget'], r['mask']): r for r in means}
    expected = {(c, m, cap, p) for c in cases for m in METHODS for cap in caps for p in POLICIES}
    require(len(index) == len(means) and set(index) == expected, 'full_case_mean_inventory_required')
    table = []
    for method in METHODS:
        for cap in caps:
            for p in POLICIES:
                group = [index[c, method, cap, p] for c in sorted(cases)]
                values = {f: sum(r[f] for r in group) / len(cases) for f in COUNTERS}
                require(math.isclose(values['proxy_support'] + values['proxy_opposition'], values['comparable'])
                    and math.isclose(values['comparable'] + values['not_comparable'], 1), 'full_denominator_conservation_required')
                table.append({'method': method, 'model_call_budget': cap, 'mask': p,
                    'fixed_ehr_cases': len(cases), **{'mean_' + f: v for f, v in values.items()},
                    'conditional_proxy_agreement': values['proxy_support'] / values['comparable'] if values['comparable'] else None,
                    'ehr_cases_with_any_comparison': sum(r['comparable'] > 0 for r in group),
                    'clinical_accuracy': None, 'clinical_repair_success': None})
    paired, withheld = [], []
    for cap in caps:
        for p in POLICIES:
            for method, reference in METHOD_CONTRASTS:
                a, b = [index[c, method, cap, p] for c in sorted(cases)], [index[c, reference, cap, p] for c in sorted(cases)]
                same = (method, reference) == METHOD_CONTRASTS[-1]
                require(not same or all(x['simulated_calls'] == y['simulated_calls'] for x, y in zip(a, b)),
                        'scored_vs_uniform_final_same_historical_expenditure_required')
                row = {'method': method, 'reference': reference, 'model_call_budget': cap, 'mask': p,
                    'fixed_ehr_cases': len(cases), 'same_acquisition_and_expenditure': same}
                for f in COUNTERS:
                    values = [x[f] - y[f] for x, y in zip(a, b)]
                    row['mean_delta_' + f] = sum(values) / len(cases)
                row['opposition_reduction_with_lost_all_comparison_ehr_cases'] = sum(
                    x['proxy_opposition'] < y['proxy_opposition'] and x['comparable'] == 0 and y['comparable'] > 0 for x, y in zip(a, b))
                paired.append(row)
        for method in METHODS:
            for left, right in CONTRASTS:
                loss = {'method': method, 'model_call_budget': cap, 'left_mask': left, 'right_mask': right,
                        'fixed_ehr_cases': len(cases)}
                for f in ('accepted_image', 'proxy_support', 'proxy_opposition', 'comparable'):
                    values = [index[c, method, cap, left][f] - index[c, method, cap, right][f] for c in sorted(cases)]
                    require(all(v >= -1e-12 for v in values), 'nested_mask_cannot_add_support_or_opposition_required')
                    loss['mean_withheld_' + f] = sum(values) / len(cases)
                loss['clinical_errors_repaired'] = None
                withheld.append(loss)
    return table, paired, withheld

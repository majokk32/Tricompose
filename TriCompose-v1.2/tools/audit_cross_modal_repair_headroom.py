#!/usr/bin/env python3
"""Finite-cache repair feasibility, not clinical truth or a new selector."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import scope_guarded_stopping as guarded
from tricompose_v12 import report_repair_headroom as reports
from tricompose_v12.invariant_verification import _candidate_facts, anchor_from_cached_candidate
from contracts import (PROTECTED_ROOT, sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

SCHEMA = 'tricompose-cross-modal-cache-headroom-v1'
EXPLICIT = frozenset(('positive', 'negative'))
EDGES = ('ehr_cxr', 'ehr_report', 'cxr_report')
GUARD_REFERENCE = (guarded.baseline.BASE / 'automatic_replays/scope_guarded_stop_12654973_001',
    '3d4553641af5408d639e857b3f20d830761458c1c08ceff9064dc6cb3d549360')


def raw_edges(states):
    if (set(states) != {'ehr', 'xrv', 'chexbert'} or any(set(v) != set(guarded.legacy.FINDINGS)
            or any(s not in guarded.legacy.STATES for s in v.values()) for v in states.values())):
        raise ValueError('complete_unchanged_four_state_inventory_required')
    out = {}
    for edge, left, right in (('ehr_cxr', 'ehr', 'xrv'), ('ehr_report', 'ehr', 'chexbert'),
            ('cxr_report', 'xrv', 'chexbert')):
        a, b = states[left], states[right]
        known = {f for f in a if a[f] in EXPLICIT}
        comparable = {f for f in known if b[f] in EXPLICIT}
        support = {f for f in comparable if a[f] == b[f]}
        out[edge] = {k: sorted(v) for k, v in {'known': known, 'comparable': comparable,
            'support': support, 'positive_support': {f for f in support if a[f] == 'positive'},
            'opposition': comparable - support}.items()}
    return out


def signature(candidate):
    anchor_from_cached_candidate(candidate)  # Complete, hash-bound cached provenance.
    facts = _candidate_facts(candidate)
    states = {key: {f: facts[f]['states'][key] for f in guarded.legacy.FINDINGS}
        for key in ('ehr', 'xrv', 'chexbert')}
    return {**reports.signature(candidate), 'report_reference_states': states['chexbert'],
        'raw_edges': raw_edges(states), 'image_branch_basis': guarded.image_branch_basis(candidate)}


def validate_signature(value):
    expected = raw_edges({'ehr': value['ehr_reference_states'], 'xrv': value['image_reference_states'],
        'chexbert': value['report_reference_states']})
    if value['raw_edges'] != expected:
        raise ValueError('raw_fact_sets_cannot_be_forged_or_globally_expanded')
    quality = value['report_structure_quality']
    if quality is not None and (type(quality) not in (int, float) or not math.isfinite(quality) or not 0 <= quality <= 1):
        raise ValueError('finite_existing_structure_quality_or_null_required')
    if (type(value['source_image_validity']) is not bool or type(value['source_gate_failure_count']) is not int
            or value['source_gate_failure_count'] < 0):
        raise ValueError('typed_cached_artifact_metadata_required')
    for f, s in value['ehr_reference_states'].items():
        if s != 'unknown' and not value['ehr_source_categories'][f]:
            raise ValueError('asserted_ehr_fact_requires_cached_direct_provenance')
    if value['direct_ehr_known_facts'] != len(expected['ehr_cxr']['known']):
        raise ValueError('unchanged_direct_fact_denominator_required')
    for key, edge, field in (('image_positive_support', 'cxr_report', 'positive_support'),
            ('ehr_direct_support', 'ehr_report', 'support'), ('image_comparable', 'cxr_report', 'comparable'),
            ('ehr_comparable', 'ehr_report', 'comparable'), ('image_opposition', 'cxr_report', 'opposition'),
            ('ehr_opposition', 'ehr_report', 'opposition')):
        if value[key] != expected[edge][field]:
            raise ValueError('report_signature_sets_must_match_raw_states')
    basis = value['image_branch_basis']
    opposition = expected['ehr_cxr']['opposition']
    allowed = not value['source_image_validity'] or bool(opposition)
    if (basis['cached_image_branch_allowed'] is not allowed
            or basis['known_direct_ehr_facts'] != len(expected['ehr_cxr']['known'])
            or basis['comparable_direct_ehr_image_facts'] != len(expected['ehr_cxr']['comparable'])
            or len(basis['explicit_proxy_opposition_evidence_ids']) != len(opposition)
            or any(basis[k] is not False for k in ('report_votes_used', 'confirmed_image_fault', 'clinical_regeneration_authorized'))):
        raise ValueError('image_branch_basis_must_follow_direct_raw_evidence')


def compare_image(base, alternative):
    """Same EHR; strict anchored gain and no per-finding regression on any edge."""
    for value in (base, alternative):
        validate_signature(value)
    if (base['case_id'] != alternative['case_id'] or any(base['lineage'][k] != alternative['lineage'][k]
            for k in ('ehr_sha256', 'ehr_facts_sha256')) or any(base[k] != alternative[k]
            for k in ('ehr_reference_states', 'ehr_source_categories'))):
        raise ValueError('fixed_ehr_hash_state_and_source_scope_required')
    duplicate = any(base['lineage'][k] == alternative['lineage'][k] for k in ('cxr_candidate_id', 'cxr_sha256'))
    loss, new, gain, removed, silenced = {}, {}, {}, {}, {}
    for edge in EDGES:
        a, b = base['raw_edges'][edge], alternative['raw_edges'][edge]
        support_key = 'positive_support' if edge == 'cxr_report' else 'support'
        loss[edge] = {key: sorted(set(a[key]) - set(b[key])) for key in ('comparable', support_key)}
        new[edge] = sorted(set(b['opposition']) - set(a['opposition']))
        gain[edge] = sorted(set(b['support']) - set(a['support']))
        removed[edge] = sorted(set(a['opposition']) - set(b['opposition']))
        silenced[edge] = sorted(set(removed[edge]) & set(loss[edge]['comparable']))
    available = all(v['report_structure_quality'] is not None for v in (base, alternative))
    quality_ok = available and alternative['report_structure_quality'] >= base['report_structure_quality']
    artifacts = all(v['source_gate_failure_count'] == 0 and v['source_image_validity'] for v in (base, alternative))
    fixed_gain = bool(gain['ehr_cxr'] or removed['ehr_cxr'])
    lost = any(v for edge in loss.values() for v in edge.values())
    reasons = []
    if duplicate: reasons.append('same_image_identity_or_hash_not_new_image')
    if not artifacts: reasons.append('cached_artifact_metadata_failed')
    if not available: reasons.append('cached_report_structure_unavailable')
    elif not quality_ok: reasons.append('cached_report_structure_lower')
    if lost: reasons.append('lost_supported_or_comparable_finding_ids')
    if any(new.values()): reasons.append('new_explicit_proxy_opposition')
    if any(silenced.values()): reasons.append('opposition_removed_by_missing_evidence_not_repair')
    if not fixed_gain: reasons.append('no_strict_fixed_ehr_image_evidence_gain')
    return {'baseline_candidate_id': base['triple_candidate_id'],
        'alternative_candidate_id': alternative['triple_candidate_id'],
        'cross_image_preserving_proxy_gain': bool(fixed_gain and not lost and not any(new.values())
            and quality_ok and artifacts and not duplicate),
        'headroom_assessable': bool(available and artifacts and not duplicate),
        'gained_support_fact_ids': gain, 'lost_fact_ids': loss, 'new_opposition_fact_ids': new,
        'removed_opposition_fact_ids': removed, 'silenced_opposition_fact_ids': silenced,
        'block_reasons': reasons, 'clinical_repair_success': False, 'clinical_accuracy': None}


def image_pair(base, reportonly, alternative):
    fixed = ('cxr_candidate_id', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')
    if (base['case_id'] != reportonly['case_id'] or any(base['lineage'][k] != reportonly['lineage'][k]
            for k in fixed) or any(base[k] != reportonly[k] for k in
            ('ehr_reference_states', 'ehr_source_categories', 'image_reference_states'))):
        raise ValueError('report_only_reference_must_keep_initial_image_and_ehr')
    comparisons = [compare_image(base, alternative), compare_image(reportonly, alternative)]
    passing = all(r['cross_image_preserving_proxy_gain'] for r in comparisons)
    basis = base['image_branch_basis']
    return {'pair_kind': 'cross_image', 'baseline_candidate_id': base['triple_candidate_id'],
        'report_only_reference_candidate_id': reportonly['triple_candidate_id'],
        'alternative_candidate_id': alternative['triple_candidate_id'],
        'comparisons_against_both_references': comparisons,
        'proxy_preserving_opportunity': passing,
        'scope_allows_cached_image_branch': basis['cached_image_branch_allowed'],
        'opportunity_with_current_scope_basis': bool(passing and basis['cached_image_branch_allowed']),
        'image_branch_basis': copy.deepcopy(basis), 'clinical_repair_success': False}


def prepare(bank, policy, reportonly_ids, guarded_ids):
    """No endpoint argument or best-alternative selection; all cases/pairs stay."""
    if set(bank) != set(reportonly_ids) or set(bank) != set(guarded_ids):
        raise ValueError('every_fixed_ehr_reference_required')
    initial_slot = (*policy['image_order'][0], policy['report_order'][0])
    evidence = {c['score_record']['triple_candidate_id']: signature(c) for grid in bank.values() for c in grid.values()}
    pairs, cases, selected = [], [], []
    for case, grid in sorted(bank.items()):
        base = evidence[grid[initial_slot]['score_record']['triple_candidate_id']]
        for candidate in grid.values():
            validate_signature(evidence[candidate['score_record']['triple_candidate_id']])
        same_ids = {c['score_record']['triple_candidate_id'] for c in grid.values()}
        if reportonly_ids[case] not in same_ids or guarded_ids[case] not in same_ids:
            raise ValueError('references_cannot_cross_case_or_inventory')
        reportonly = evidence[reportonly_ids[case]]
        same, other = [], []
        for slot, candidate in sorted(grid.items()):
            alternative = evidence[candidate['score_record']['triple_candidate_id']]
            if alternative['triple_candidate_id'] == base['triple_candidate_id']:
                continue
            if slot[:2] == initial_slot[:2]:
                detail = reports.compare(base, alternative)
                row = {'pair_kind': 'same_image_report', 'baseline_candidate_id': base['triple_candidate_id'],
                    'alternative_candidate_id': alternative['triple_candidate_id'], 'report_comparison': detail,
                    'proxy_preserving_opportunity': detail['strict_label_preserving_headroom'],
                    'clinical_repair_success': False}
                same.append(row)
            else:
                row = image_pair(base, reportonly, alternative)
                other.append(row)
            pairs.append({'case_id': case, **row})
        actual = evidence[guarded_ids[case]]
        if actual['triple_candidate_id'] == base['triple_candidate_id']:
            audit = {'change': 'unchanged', 'proxy_preserving_opportunity': False, 'comparison': None}
        elif actual['lineage']['cxr_candidate_id'] == base['lineage']['cxr_candidate_id']:
            detail = reports.compare(base, actual)
            audit = {'change': 'same_image_report', 'proxy_preserving_opportunity': detail['strict_label_preserving_headroom'],
                'comparison': detail}
        else:
            detail = image_pair(base, reportonly, actual)
            audit = {'change': 'cross_image', 'proxy_preserving_opportunity': detail['proxy_preserving_opportunity'],
                'comparison': detail}
        selected.append({'case_id': case, 'baseline_candidate_id': base['triple_candidate_id'],
            'guarded_selected_candidate_id': guarded_ids[case], **audit, 'clinical_repair_success': False})
        report_pass = [p['alternative_candidate_id'] for p in same if p['proxy_preserving_opportunity']]
        image_pass = [p['alternative_candidate_id'] for p in other if p['proxy_preserving_opportunity']]
        scoped_pass = [p['alternative_candidate_id'] for p in other if p['opportunity_with_current_scope_basis']]
        category = 'both' if report_pass and image_pass else 'report_only' if report_pass else 'cross_image_only' if image_pass else 'none_in_finite_cache'
        cases.append({'case_id': case, 'ehr_sha256': base['lineage']['ehr_sha256'],
            'ehr_facts_sha256': base['lineage']['ehr_facts_sha256'],
            'direct_ehr_known_facts': base['direct_ehr_known_facts'], 'image_branch_basis': base['image_branch_basis'],
            'report_alternative_slots': len(same), 'image_report_alternative_slots': len(other),
            'report_preserving_alternative_ids': report_pass, 'cross_image_preserving_alternative_ids': image_pass,
            'cross_image_alternatives_with_current_scope_basis': scoped_pass, 'finite_cache_category': category,
            'new_selected_candidate_id': None, 'clinical_repair_success': False})
    return {'schema_version': SCHEMA, 'cases': cases, 'directed_pairs': pairs,
        'guarded_selection_audit': selected, 'secondary_endpoint_used_for_opportunity': False,
        'new_selected_candidate_id': None, 'new_model_calls': 0, 'clinical_acceptance': False}


def attach(plan, candidates):
    measured = []
    for row in plan['directed_pairs']:
        a, b = (guarded.baseline.candidate_readout(candidates[row[k]])['biovil_raw_cosine']
            for k in ('baseline_candidate_id', 'alternative_candidate_id'))
        measured.append({**row, 'baseline_biovil_raw_cosine': a, 'alternative_biovil_raw_cosine': b,
            'biovil_delta_alternative_minus_baseline': b - a if a is not None and b is not None else None})
    groups = defaultdict(list)
    for row in measured:
        if row['proxy_preserving_opportunity']:
            groups[row['pair_kind'], row['case_id']].append(row['biovil_delta_alternative_minus_baseline'])
    conditional = []
    for kind in ('same_image_report', 'cross_image'):
        values = [sum(ds) / len(ds) if all(d is not None for d in ds) else None
            for (k, _), ds in groups.items() if k == kind]
        conditional.append({'pair_kind': kind, 'fixed_ehr_denominator': len(plan['cases']),
            'opportunity_ehr_cases': len(values), 'complete_endpoint_opportunity_ehr_cases': sum(v is not None for v in values),
            'conditional_case_balanced_all_alternative_mean_delta': sum(values) / len(values)
                if values and all(v is not None for v in values) else None,
            'selected_output_effect': False, 'clinical_accuracy': None})
    return measured, conditional


def summarize(plan):
    pairs, cases, audits = plan['directed_pairs'], plan['cases'], plan['guarded_selection_audit']
    return {'schema_version': SCHEMA, 'cohort_role': 'already_inspected_development',
        'fixed_ehr_cases': len(cases), 'direct_ehr_cases': sum(c['direct_ehr_known_facts'] > 0 for c in cases),
        'directed_candidate_pairs': len(pairs), 'report_alternative_pairs': sum(p['pair_kind'] == 'same_image_report' for p in pairs),
        'cross_image_alternative_pairs': sum(p['pair_kind'] == 'cross_image' for p in pairs),
        'strict_report_preserving_pairs': sum(p['pair_kind'] == 'same_image_report' and p['proxy_preserving_opportunity'] for p in pairs),
        'cross_image_preserving_pairs_beating_both_references': sum(p['pair_kind'] == 'cross_image' and p['proxy_preserving_opportunity'] for p in pairs),
        'cross_image_preserving_pairs_with_current_scope_basis': sum(p.get('opportunity_with_current_scope_basis', False) for p in pairs),
        'report_opportunity_ehr_cases': sum(bool(c['report_preserving_alternative_ids']) for c in cases),
        'cross_image_opportunity_ehr_cases': sum(bool(c['cross_image_preserving_alternative_ids']) for c in cases),
        'cross_image_scope_eligible_opportunity_ehr_cases': sum(bool(c['cross_image_alternatives_with_current_scope_basis']) for c in cases),
        'finite_cache_case_categories': dict(Counter(c['finite_cache_category'] for c in cases)),
        'initial_image_branch_basis_counts': dict(Counter(c['image_branch_basis']['reason'] for c in cases)),
        'guarded_change_counts': dict(Counter(a['change'] for a in audits)),
        'guarded_changed_outputs_passing_preservation': sum(a['proxy_preserving_opportunity'] for a in audits),
        'guarded_changed_outputs_not_passing_preservation': sum(a['change'] != 'unchanged' and not a['proxy_preserving_opportunity'] for a in audits),
        'clinical_accuracy': None, 'clinical_repair_success': False, 'clinical_acceptance': False,
        'new_model_calls': 0, 'original_selection_changed': False, 'regeneration_authorized': False,
        'primary_metric_eligible': False, 'source_bodies_pixels_weights_read': False}


def case_table(plan):
    return [{'case_id': c['case_id'], 'direct_ehr_known_facts': c['direct_ehr_known_facts'],
        'image_branch_basis': c['image_branch_basis']['reason'],
        'report_preserving_alternatives': len(c['report_preserving_alternative_ids']),
        'cross_image_preserving_alternatives': len(c['cross_image_preserving_alternative_ids']),
        'cross_image_alternatives_with_current_scope_basis': len(c['cross_image_alternatives_with_current_scope_basis']),
        'finite_cache_category': c['finite_cache_category'], 'clinical_repair_success': False} for c in plan['cases']]


def render(summary, conditional):
    lines = ['# Cross-modal repair feasibility / 换报告与换图的修复空间', '',
        'Fixed 80-EHR/960-triple DEVELOPMENT cache. No new generation, winner or clinical truth.', '',
        '| Diagnostic | Count |', '|---|---:|']
    for key in ('fixed_ehr_cases', 'direct_ehr_cases', 'directed_candidate_pairs', 'report_opportunity_ehr_cases',
            'cross_image_opportunity_ehr_cases', 'cross_image_scope_eligible_opportunity_ehr_cases',
            'guarded_changed_outputs_passing_preservation', 'guarded_changed_outputs_not_passing_preservation'):
        lines.append(f'| {key} | {summary[key]} |')
    lines += ['', '## Conditional secondary readout / 仅机会病例的辅助读数', '',
        '| Kind | Opportunity EHRs / all | Mean all-alternative BioViL delta |', '|---|---:|---:|']
    for row in conditional:
        v = row['conditional_case_balanced_all_alternative_mean_delta']
        lines.append(f"| {row['pair_kind']} | {row['opportunity_ehr_cases']}/{row['fixed_ehr_denominator']} | "
            + ('NA' if v is None else f'{v:.4f}') + ' |')
    lines += ['', 'Unknown/uncertain cannot silence an opposition and earn repair credit.',
        'Cross-image alternatives must improve fixed-EHR image evidence and preserve finding IDs against BOTH references.',
        'Opportunity means this finite cached proxy predicate, not clinical repair or a prospective action success rate.',
        'Conditional endpoint gaps are not selected-output effects; no endpoint chooses an alternative.',
        'No opportunity is not global optimality. Clinical localization/repair remain unvalidated.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    guarded.baseline.cpu_guard(); started = time.monotonic()
    old, candidates, bank, policy, scorefree, reportonly, sources = guarded.load()
    outcomes = guarded.checked_metadata(*GUARD_REFERENCE, 'guarded_outcomes.jsonl', sources, 'guarded_reference')
    if len(outcomes) != 400:
        raise ValueError('complete_guarded_trials_required')
    for row in outcomes:
        if row != guarded.replay(bank[row['case_id']], policy, row['model_call_budget']):
            raise ValueError('guarded_reference_must_replay_exactly')
    report_ids = {r['case_id']: r['selected_candidate_id'] for r in reportonly if r['model_call_budget'] == 30}
    guarded_ids = {r['case_id']: r['selected_candidate_id'] for r in outcomes if r['model_call_budget'] == 30}
    sources.update(headroom_worker=Path(__file__), headroom_tests=ROOT / 'tests/test_cross_modal_repair_headroom.py',
        headroom_protocol=ROOT.parent / 'docs/cross_modal_repair_headroom_protocol.md')
    for name, module in sorted(sys.modules.copy().items()):
        if name.startswith('tricompose_v12') and getattr(module, '__file__', None):
            sources[name] = Path(module.__file__)
    before = {k: sha256_file(p) for k, p in sources.items()}
    plan = prepare(bank, policy, report_ids, guarded_ids)
    if len(plan['cases']) != 80 or len(plan['directed_pairs']) != 880:
        raise ValueError('all_eighty_cases_and_eleven_alternatives_required')
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        write_private_json(temporary / 'label_plan.json', plan)
        plan_hash = sha256_file(temporary / 'label_plan.json')
        measured, conditional = attach(plan, candidates)
        summary = summarize(plan)
        summary['elapsed_cpu_seconds_before_serialization'] = round(time.monotonic() - started, 6)
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'conditional_readouts.json', conditional)
        write_private_text(temporary / 'pair_opportunities.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in measured))
        write_private_text(temporary / 'case_opportunities.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in plan['cases']))
        write_private_text(temporary / 'case_table.csv', guarded.baseline.csv_text(case_table(plan)))
        write_private_text(temporary / 'guarded_selection_audit.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in plan['guarded_selection_audit']))
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(summary, conditional))
        if before != {k: sha256_file(p) for k, p in sources.items()} or plan_hash != sha256_file(temporary / 'label_plan.json'):
            raise ValueError('sealed_sources_or_label_plan_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir()},
            'label_plan_sha256_before_endpoint_attachment': plan_hash,
            'secondary_endpoint_used_for_opportunity': False, 'original_selection_changed': False,
            'new_model_calls': 0, 'clinical_repair_success': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=guarded.baseline.BASE / 'repair_headroom')
    args = parser.parse_args()
    try:
        path = execute(args.output_root, args.run_id)
    except Exception:
        print(json.dumps({'status': 'failed_cached_cross_modal_headroom', 'new_model_calls': 0}))
        raise SystemExit(1) from None
    print(json.dumps({'status': 'completed_cached_cross_modal_headroom', 'new_model_calls': 0,
        'manifest_sha256': sha256_file(path / 'manifest.json')}))


if __name__ == '__main__':
    main()

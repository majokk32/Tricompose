#!/usr/bin/env python3
"""Observed-only proxy-preservation veto; never clinical acceptance."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import audit_cross_modal_repair_headroom as headroom
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

guarded = headroom.guarded
SCHEMA = 'tricompose-observed-output-acceptance-v1'
METHODS = {guarded.METHOD: 'preservation_gate_scope_guarded',
    'report_only_static': 'preservation_gate_report_only_static'}
HEADROOM_ROOT = guarded.baseline.BASE / 'repair_headroom/cross_modal_headroom_12654973_001'
HEADROOM_HASH = '9f179fa5597a88acd9aa66f5a2d77078ed68bce781468f9ef812cd98837382a3'


def controller_view(candidate):
    view = guarded.controller_view(candidate)
    view['score_record']['schema_version'] = candidate['score_record']['schema_version']
    return view


def eligible(signature):
    return signature['source_gate_failure_count'] == 0 and signature['source_image_validity']


def assess_output(baseline, proposed, observed_candidates):
    """Accept only a strictly proxy-preserving observed change; else initial fallback."""
    views = [controller_view(c) for c in observed_candidates]
    observed = {c['score_record']['triple_candidate_id']: c for c in views}
    if not observed or len(observed) != len(views):
        raise ValueError('nonempty_unique_observed_candidates_required')
    base_view = controller_view(baseline)
    base_id = base_view['score_record']['triple_candidate_id']
    proposed_view = None if proposed is None else controller_view(proposed)
    prop_id = None if proposed_view is None else proposed_view['score_record']['triple_candidate_id']
    if (observed.get(base_id) != base_view or views[0] != base_view
            or proposed_view is not None and observed.get(prop_id) != proposed_view):
        raise ValueError('baseline_and_proposal_must_match_already_observed_records')
    signatures = {cid: headroom.signature(c) for cid, c in observed.items()}
    base = signatures[base_id]
    images, image_hashes, report_hashes = {}, {}, {}
    for s in signatures.values():
        headroom.validate_signature(s)
        if (s['case_id'] != base['case_id'] or any(s['lineage'][k] != base['lineage'][k]
                for k in ('ehr_sha256', 'ehr_facts_sha256')) or any(s[k] != base[k]
                for k in ('ehr_reference_states', 'ehr_source_categories'))):
            raise ValueError('fixed_observed_ehr_hashes_states_and_provenance_required')
        image = s['lineage']['cxr_candidate_id']
        image_value = (s['lineage']['cxr_sha256'], s['image_reference_states'])
        if images.setdefault(image, image_value) != image_value:
            raise ValueError('same_image_report_change_cannot_change_classifier_reference')
        if image_hashes.setdefault(s['lineage']['cxr_sha256'], s['image_reference_states']) != s['image_reference_states']:
            raise ValueError('shared_image_hash_requires_same_classifier_reference')
        if report_hashes.setdefault(s['lineage']['report_sha256'], s['report_reference_states']) != s['report_reference_states']:
            raise ValueError('shared_report_hash_requires_same_frozen_report_labels')
    initial = [c for cid, c in observed.items() if eligible(signatures[cid]) and
        signatures[cid]['lineage']['cxr_candidate_id'] == base['lineage']['cxr_candidate_id']]
    reference = min(initial, key=guarded.original.candidate_key) if initial else None
    ref_id = None if reference is None else reference['score_record']['triple_candidate_id']
    detail, reasons, passes = None, [], False
    if prop_id == base_id:
        status = 'unchanged_unverified' if eligible(base) else 'unresolved_no_metadata_eligible_output'
    elif prop_id is None:
        status = 'unresolved_missing_proposal_fixed_retained'
        reasons = ['parent_proposal_unavailable']
    else:
        proposal = signatures[prop_id]
        if proposal['lineage']['cxr_candidate_id'] == base['lineage']['cxr_candidate_id']:
            detail = headroom.reports.compare(base, proposal)
            passes = detail['strict_label_preserving_headroom']
            reasons = detail['block_or_equivalence_reasons']
            status = 'proxy_preserving_report_change_unverified'
        elif reference is not None:
            detail = headroom.image_pair(base, signatures[ref_id], proposal)
            passes = detail['opportunity_with_current_scope_basis']
            reasons = [reason for c in detail['comparisons_against_both_references'] for reason in c['block_reasons']]
            if not detail['scope_allows_cached_image_branch']:
                reasons.append('no_direct_observed_image_branch_basis')
            status = 'proxy_preserving_image_change_unverified'
        else:
            reasons = ['initial_image_report_reference_unavailable']
            status = 'unresolved_no_metadata_eligible_output'
        if not passes:
            status = 'unresolved_proposal_veto_fixed_retained'
    selected = prop_id if passes else base_id if eligible(base) else None
    if selected is None:
        status = 'unresolved_no_metadata_eligible_output'
    return {'schema_version': SCHEMA, 'case_id': base['case_id'],
        'baseline_candidate_id': base_id, 'proposed_candidate_id': prop_id,
        'selected_candidate_id': selected, 'observed_initial_report_reference_candidate_id': ref_id,
        'observed_initial_report_reference_inventory': sorted(c['score_record']['triple_candidate_id'] for c in initial),
        'status': status, 'proposal_passes_proxy_preservation': bool(passes),
        'proposal_vetoed': prop_id is not None and prop_id != base_id and not passes,
        'output_changed_from_fixed': selected is not None and selected != base_id,
        'rejection_reason_codes': sorted(set(reasons)), 'comparison': detail,
        'ehr_sha256': base['lineage']['ehr_sha256'], 'ehr_facts_sha256': base['lineage']['ehr_facts_sha256'],
        'clinical_acceptance': False, 'clinical_repair_success': False, 'clinical_fault_confirmed': False,
        'regeneration_authorized': False, 'alternate_endpoint_used_for_gate': False}


def apply_trial(parent, grid, candidates):
    if parent['method'] not in METHODS:
        raise ValueError('declared_parent_method_required')
    guarded.baseline.validate_trace(parent, candidates)
    observed = [grid[tuple(a['request_slot'])] for a in parent['action_trace']]
    if not observed:
        raise ValueError('initial_reference_requires_completed_observation')
    index = {c['score_record']['triple_candidate_id']: c for c in observed}
    proposed_id = parent['selected_candidate_id']
    proposed = None if proposed_id is None else index[proposed_id]
    decision = assess_output(observed[0], proposed, observed)
    selected = decision['selected_candidate_id']
    outcome = {'case_id': parent['case_id'], 'method': METHODS[parent['method']],
        'parent_method': parent['method'], 'parent_trial_sha256': guarded.baseline.digest(parent),
        'parent_selected_candidate_id': proposed_id, 'model_call_budget': parent['model_call_budget'], 'random_seed': None,
        'selected_candidate_id': selected, 'selected_ehr_sha256': parent['selected_ehr_sha256'],
        'input_ehr_assessment_scope': parent['input_ehr_assessment_scope'],
        'simulated_calls': copy.deepcopy(parent['simulated_calls']), 'simulated_model_calls': parent['simulated_model_calls'],
        'observed_candidates': parent['observed_candidates'], 'observed_images': parent['observed_images'],
        'terminal_reason': parent['terminal_reason'], 'action_trace': copy.deepcopy(parent['action_trace']),
        'output_status': decision['status'], 'output_decision': decision,
        'selected_proxy_stop_conditions_met': selected is not None and guarded.original.action_signal(index[selected])[0] == 'stop_proxy_satisfied',
        'actual_regeneration_executed': False, 'clinical_fault_confirmed': False, 'clinical_acceptance': False}
    guarded.baseline.validate_trace(outcome, candidates)
    return outcome


def load():
    old, candidates, bank, policy, scorefree, reportonly, sources = guarded.load()
    trials = guarded.checked_metadata(*headroom.GUARD_REFERENCE, 'guarded_outcomes.jsonl', sources, 'guarded_reference')
    expected = {(case, cap) for case in bank for cap in guarded.baseline.CAPS}
    if len(trials) != 400 or {(r['case_id'], r['model_call_budget']) for r in trials} != expected:
        raise ValueError('every_guarded_case_and_budget_required')
    for row in trials:
        if row != guarded.replay(bank[row['case_id']], policy, row['model_call_budget']):
            raise ValueError('unchanged_guarded_trials_must_replay_exactly')
    mp = require_inside(HEADROOM_ROOT / 'manifest.json', PROTECTED_ROOT, must_exist=True)
    if mp.stat().st_size > 1024 ** 2 or sha256_file(mp) != HEADROOM_HASH:
        raise ValueError('sealed_preservation_helper_reference_required')
    manifest = json.loads(mp.read_text())
    for label in manifest['source_paths']:
        if label not in ('headroom_worker', 'headroom_tests', 'headroom_protocol') and not label.startswith('tricompose_v12'):
            continue
        path = Path(manifest['source_paths'][label])
        if sha256_file(path) != manifest['source_sha256'][label]:
            raise ValueError('unchanged_fact_preservation_rule_required')
        sources[label] = path
    sources['headroom_reference_manifest'] = mp
    sources.update(acceptance_worker=Path(__file__), acceptance_tests=ROOT / 'tests/test_output_acceptance.py',
        acceptance_protocol=ROOT.parent / 'docs/output_acceptance_protocol.md')
    for name, module in sorted(sys.modules.copy().items()):
        if name.startswith('tricompose_v12') and getattr(module, '__file__', None):
            sources[name] = Path(module.__file__)
    return old, candidates, bank, policy, scorefree, reportonly, trials, sources


def policy_description():
    return {'schema_version': SCHEMA, 'cohort_role': 'already_inspected_development',
        'change': 'observed_only_post_selection_preservation_veto_no_reranking',
        'parents': list(METHODS), 'model_call_caps': list(guarded.baseline.CAPS),
        'cross_image_reference': 'best_eligible_observed_report_on_initial_image_not_full_bank_winner',
        'failed_proposal_fallback': 'initial_metadata_eligible_fixed_candidate_else_null',
        'already_incurred_parent_costs_retained': True, 'alternate_endpoints_used_for_gate': False,
        'new_thresholds_weights_or_training': False, 'clinical_acceptance': False,
        'regeneration_authorized': False, 'installed_into_live_gpu_controller': False}


def contrasts(means):
    methods = [(METHODS[guarded.METHOD], guarded.METHOD),
        (METHODS['report_only_static'], 'report_only_static'),
        (METHODS[guarded.METHOD], METHODS['report_only_static']),
        (METHODS[guarded.METHOD], 'fixed'), (METHODS[guarded.METHOD], 'static_rerank')]
    index = {(r['case_id'], r['method'], r['model_call_budget']): r for r in means}
    cases = sorted({r['case_id'] for r in means}); rows = []
    for cap in guarded.baseline.CAPS:
        for left, right in methods:
            for scope in ('all', 'explicit_fact_proxy', 'no_direct_comparable_ehr_facts'):
                pairs = [(index[case, left, cap], index[case, right, cap]) for case in cases
                    if scope == 'all' or index[case, left, cap]['ehr_scope'] == scope]
                ds = [a['biovil_raw_cosine'] - b['biovil_raw_cosine'] for a, b in pairs
                    if a['biovil_raw_cosine'] is not None and b['biovil_raw_cosine'] is not None]
                rows.append({'ehr_scope': scope, 'model_call_budget': cap, 'left_method': left, 'right_method': right,
                    'fixed_ehr_cases': len(pairs), 'paired_biovil_cases': len(ds),
                    'mean_biovil_delta_left_minus_right': sum(ds) / len(ds) if ds else None,
                    'left_higher': sum(d > 0 for d in ds), 'tied': sum(d == 0 for d in ds), 'right_higher': sum(d < 0 for d in ds),
                    'mean_simulated_call_delta_left_minus_right': sum(a['mean_simulated_calls'] - b['mean_simulated_calls'] for a, b in pairs) / len(pairs),
                    'selector_only_equal_charged_acquisition': (left, right) in methods[:2], 'clinical_accuracy': None})
    return rows


def summarize(gated):
    rows = []
    for method in METHODS.values():
        for cap in guarded.baseline.CAPS:
            trials = [r for r in gated if r['method'] == method and r['model_call_budget'] == cap]
            rows.append({'method': method, 'model_call_budget': cap, 'fixed_ehr_cases': len(trials),
                'status_counts': dict(Counter(r['output_status'] for r in trials)),
                'vetoed_proposals': sum(r['output_decision']['proposal_vetoed'] for r in trials),
                'proxy_preserving_changed_outputs': sum(r['output_decision']['proposal_passes_proxy_preservation'] for r in trials),
                'null_outputs': sum(r['selected_candidate_id'] is None for r in trials)})
    full = {(r['case_id'], r['method']): r for r in gated if r['model_call_budget'] == 30}
    methods = list(METHODS.values())
    return {'schema_version': SCHEMA, 'fixed_ehr_cases': 80, 'gated_trials': len(gated),
        'case_budget_status': rows, 'full_cap_gated_methods_same_selected_output':
            sum(full[case, methods[0]]['selected_candidate_id'] == full[case, methods[1]]['selected_candidate_id'] for case in {c for c, _ in full}),
        'new_model_calls': 0, 'original_selection_changed': False, 'clinical_acceptance': False,
        'clinical_accuracy': None, 'clinical_repair_success': False, 'regeneration_authorized': False,
        'primary_metric_eligible': False, 'actual_gpu_savings': None, 'source_bodies_pixels_weights_read': False}


def render(summary, comparison):
    def show(v): return 'NA' if v is None else f'{v:.4f}'
    lines = ['# Observed-only output acceptance / 输出验收对照', '',
        'Same fixed DEVELOPMENT EHRs, observed candidates and already incurred charges; no new generation.', '',
        '| Method | Cap | Mean simulated calls | BioViL | CXR–Report support / known | Opposition / known | Comparable coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        if r['ehr_scope'] == 'all' and r['model_call_budget'] == 30:
            lines.append(f"| {r['method']} | 30 | {show(r['mean_simulated_calls_mean'])} | {show(r['biovil_raw_cosine_mean'])} | "
                f"{show(r['cxr_report_support_over_known_mean'])} | {show(r['cxr_report_opposition_over_known_mean'])} | {show(r['cxr_report_coverage_over_known_mean'])} |")
    lines += ['', 'Proxy-preserving change, unresolved fallback and unchanged output are distinct; all remain clinically unverified.',
        'No unseen report-only winner enters the gate; failed proposals retain costs, not refunds.',
        'Fallback is not a correctness certificate. Missing EHR edges stay NA; no EHR is replaced.',
        f"Full-cap gated methods choose the same output on {summary['full_cap_gated_methods_same_selected_output']}/80 EHRs.",
        'Do not claim a dynamic advantage if the gated report-only control is equivalent.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    guarded.baseline.cpu_guard(); started = time.monotonic()
    old, candidates, bank, policy, scorefree, reportonly, trials, sources = load()
    before = {k: sha256_file(p) for k, p in sources.items()}
    gated = [apply_trial(row, bank[row['case_id']], candidates) for row in [*trials, *reportonly]]
    if len(gated) != 800:
        raise ValueError('all_eighty_cases_two_parents_five_caps_required')
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        write_private_json(temporary / 'frozen_policy.json', policy_description())
        write_private_json(temporary / 'selection_plan.json', gated)
        plan_hash = sha256_file(temporary / 'selection_plan.json')
        means = guarded.baseline.case_means([*old, *scorefree, *reportonly, *trials, *gated], candidates)
        comparison, _ = guarded.baseline.comparisons(means); paired = contrasts(means)
        summary = summarize(gated)
        summary['elapsed_cpu_seconds_before_final_serialization'] = round(time.monotonic() - started, 6)
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'gated_outcomes.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in gated))
        write_private_text(temporary / 'case_means.csv', guarded.baseline.csv_text(means))
        write_private_text(temporary / 'method_comparison.csv', guarded.baseline.csv_text(comparison))
        write_private_text(temporary / 'paired_case_comparison.csv', guarded.baseline.csv_text(paired))
        write_private_text(temporary / 'selected_references.jsonl', ''.join(json.dumps({'case_id': r['case_id'], 'method': r['method'],
            'model_call_budget': r['model_call_budget'], 'selected_candidate_id': r['selected_candidate_id'],
            'artifact_hashes': None if r['selected_candidate_id'] is None else {k: candidates[r['selected_candidate_id']][k]
                for k in ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')},
            'output_status': r['output_status'], 'clinical_acceptance': False}, sort_keys=True) + '\n' for r in gated))
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(summary, comparison))
        if before != {k: sha256_file(p) for k, p in sources.items()} or plan_hash != sha256_file(temporary / 'selection_plan.json'):
            raise ValueError('sealed_sources_or_pre_endpoint_selection_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir()},
            'selection_sha256_before_endpoint_attachment': plan_hash,
            'new_model_calls': 0, 'original_selection_changed': False, 'clinical_accuracy': None,
            'clinical_repair_success': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=guarded.baseline.BASE / 'output_acceptance')
    args = parser.parse_args()
    try:
        path = execute(args.output_root, args.run_id)
    except Exception:
        print(json.dumps({'status': 'failed_cached_output_acceptance', 'new_model_calls': 0}))
        raise SystemExit(1) from None
    print(json.dumps({'status': 'completed_cached_output_acceptance', 'new_model_calls': 0,
        'manifest_sha256': sha256_file(path / 'manifest.json')}))


if __name__ == '__main__':
    main()

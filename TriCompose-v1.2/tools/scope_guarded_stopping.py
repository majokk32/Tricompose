#!/usr/bin/env python3
"""Scope-based veto of cached image exploration; never clinical repair truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
import score_free_random_control as baseline
from tricompose_v12 import automatic_replay as original
from tricompose_v12 import legacy_replay_adapter as legacy
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

SCHEMA = 'tricompose-scope-guarded-stopping-v1'
METHOD = 'scope_guarded_targeted'
REFERENCES = {
    'scorefree': (baseline.BASE / 'automatic_replays/score_free_random_12654973_001',
        '771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124', 'control_outcomes.jsonl'),
    'reportonly': (baseline.BASE / 'automatic_replays/fixed_image_control_pool80_12605930_001',
        '35ec5d369e033c6b81877247e13224921115403e0c2461c28109737551c12e60', 'control_outcomes.jsonl'),
}


def checked_metadata(root, expected, name, sources, label):
    mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
    if mp.stat().st_size > 1024 ** 2 or sha256_file(mp) != expected:
        raise ValueError('unchanged_reference_manifest_required')
    manifest = json.loads(mp.read_text())
    if manifest['original_selection_changed'] is not False:
        raise ValueError('immutable_original_choices_required')
    path = require_inside(root / name, root, must_exist=True)
    if path.stat().st_size > 32 * 1024 ** 2 or sha256_file(path) != manifest['artifacts'][name]['sha256']:
        raise ValueError('bounded_hash_bound_control_metadata_required')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if sha256_file(path) != manifest['artifacts'][name]['sha256'] or sha256_file(mp) != expected:
        raise ValueError('control_reference_changed_during_read')
    sources[label + '_manifest'], sources[label + '_outcomes'] = mp, path
    return rows


def load():
    inputs, sources = baseline.load()
    old, candidates = baseline.prepare(inputs)
    replay_manifest = json.loads(sources['replay_manifest'].read_text())
    cache = {}
    for name, limit in (('source_scores', 4 * 1024 ** 2), ('source_edges', 8 * 1024 ** 2)):
        path = require_inside(replay_manifest['source_paths'][name], PROTECTED_ROOT, must_exist=True)
        expected = replay_manifest['source_sha256'][name]
        if path.stat().st_size > limit or sha256_file(path) != expected:
            raise ValueError('unchanged_synthetic_state_source_required')
        text = path.read_text()
        cache[name] = ([json.loads(line) for line in text.splitlines() if line]
            if name == 'source_scores' else json.loads(text))
        if sha256_file(path) != expected:
            raise ValueError('cached_state_source_changed_during_read')
        sources['legacy_' + name] = path
    policy = inputs['replay', 'frozen_policy.json']
    bank = legacy.make_legacy_bank(cache['source_scores'], cache['source_edges'], policy)
    if len(bank) != 80 or sum(map(len, bank.values())) != 960:
        raise ValueError('all_fixed_legacy_cases_required')
    for grid in bank.values():
        for candidate in grid.values():
            row = candidate['score_record']; saved = candidates[row['triple_candidate_id']]
            if row['case_id'] != saved['case_id'] or any(row['lineage'][k] != saved[k] for k in
                    ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256',
                        'cxr_candidate_id', 'report_candidate_id', 'report_model_id')):
                raise ValueError('full_bank_cached_fact_lineage_required')
    for row in old:
        replayed = original.replay_case(bank[row['case_id']], policy, row['method'], row['model_call_budget'],
            random_seed=row['random_seed'] if row['method'] == 'random' else 0)
        if replayed != {k: v for k, v in row.items() if k != 'input_ehr_assessment_scope'}:
            raise ValueError('all_original_trials_must_replay_exactly')
    references = {label: checked_metadata(root, expected, name, sources, label)
        for label, (root, expected, name) in REFERENCES.items()}
    scorefree = references['scorefree']
    if scorefree != baseline.make_controls(old, candidates):
        raise ValueError('unchanged_all_score_free_seed_trials_required')
    reportonly = [r for r in references['reportonly'] if r['method'] == 'report_only_static']
    if (len(reportonly) != 400 or {(r['case_id'], r['model_call_budget']) for r in reportonly}
            != {(case, cap) for case in bank for cap in baseline.CAPS}):
        raise ValueError('all_fixed_image_static_reference_trials_required')
    for row in reportonly:
        baseline.validate_trace(row, candidates)
    sources.update(guard_worker=Path(__file__), guard_tests=ROOT / 'tests/test_scope_guarded_stopping.py',
        guard_protocol=ROOT.parent / 'docs/scope_guarded_stopping_protocol.md')
    for name, module in sorted(sys.modules.copy().items()):
        if name.startswith('tricompose_v12') and getattr(module, '__file__', None):
            sources[name] = Path(module.__file__)
    return old, candidates, bank, policy, scorefree, reportonly, sources


def controller_view(candidate):
    """No alternate endpoint, full-bank rank or old selected flag is exposed."""
    row = candidate['score_record']; scoring = row['scoring']
    return {'score_record': {'case_id': row['case_id'], 'triple_candidate_id': row['triple_candidate_id'],
        'lineage': copy.deepcopy(row['lineage']), 'scoring': {
            'selection': {'hard_gate_failure_count': scoring['selection']['hard_gate_failure_count']},
            **{k: copy.deepcopy(scoring[k]) for k in
                ('clinical_totals', 'edge_metrics', 'modality_quality', 'cost')}}},
        'facts': copy.deepcopy(candidate['facts'])}


def image_branch_basis(candidate):
    facts, row = candidate['facts'], candidate['score_record']
    known, comparable, oppositions = [], [], []
    for fact in facts:
        ehr, xrv = fact['states']['ehr'], fact['states']['xrv']
        if ehr not in legacy.STATES or xrv not in legacy.STATES or fact['weak_context_promoted'] is not False:
            raise ValueError('unchanged_four_state_direct_evidence_required')
        expected_opposition = ehr in original.EXPLICIT and xrv in original.EXPLICIT and ehr != xrv
        if (fact['relations']['raw']['ehr_cxr'] == 'opposition') != expected_opposition:
            raise ValueError('raw_direct_opposition_must_match_explicit_states')
        if ehr in original.EXPLICIT:
            if not fact['source_categories']:
                raise ValueError('known_direct_ehr_fact_requires_source_category')
            known.append(fact['evidence_id'])
            if xrv in original.EXPLICIT:
                comparable.append(fact['evidence_id'])
            if expected_opposition:
                oppositions.append(fact['evidence_id'])
    invalid = row['scoring']['modality_quality']['cxr_basic_validity_pass'] is False
    allowed = invalid or bool(oppositions)
    reason = ('basic_invalid_image_metadata' if invalid else 'direct_ehr_xrv_opposition_proxy' if oppositions
        else 'no_direct_ehr_image_reference' if not known else 'no_comparable_ehr_image_evidence' if not comparable
        else 'no_direct_ehr_image_opposition')
    return {'cached_image_branch_allowed': allowed, 'reason': reason,
        'known_direct_ehr_facts': len(known), 'comparable_direct_ehr_image_facts': len(comparable),
        'explicit_proxy_opposition_evidence_ids': sorted(oppositions),
        'report_votes_used': False, 'confirmed_image_fault': False, 'clinical_regeneration_authorized': False}


def replay(case_bank, policy, budget):
    if type(budget) is not int or budget < 0:
        raise ValueError('nonnegative_integer_budget_required')
    images = [tuple(i) for i in policy['image_order']]; reports = list(policy['report_order'])
    slot = (*images[0], reports[0])
    initial = case_bank[slot]['score_record']; case = initial['case_id']
    observed, used_images, calls, trace, decisions = {}, set(), Counter(), [], []
    fixed_ehr, shared_images = None, {}
    terminal = 'model_call_budget_exhausted'
    action, reasons, evidence = 'generate_candidate', ['predeclared_initial_path'], []
    while True:
        cost = 2 + (0 if slot[:2] in used_images else 2)
        if sum(calls.values()) + cost > budget:
            terminal = 'model_call_budget_exhausted'; break
        if slot in observed:
            raise ValueError('observed_slot_cannot_be_recharged')
        candidate = controller_view(case_bank[slot])
        row, lineage = candidate['score_record'], candidate['score_record']['lineage']
        ehr_reference = (lineage['ehr_sha256'], lineage['ehr_facts_sha256'],
            tuple((f['finding'], f['states']['ehr'], tuple(f['source_categories'])) for f in candidate['facts']))
        if fixed_ehr is None:
            fixed_ehr = ehr_reference
        if row['case_id'] != case or lineage['ehr_sha256'] != initial['lineage']['ehr_sha256'] or fixed_ehr != ehr_reference:
            raise ValueError('fixed_ehr_hashes_states_and_sources_required')
        image_reference = (lineage['cxr_sha256'], tuple((f['finding'], f['states']['xrv']) for f in candidate['facts']))
        if shared_images.setdefault(slot[:2], image_reference) != image_reference:
            raise ValueError('report_change_cannot_change_classifier_reference')
        if slot[:2] not in used_images:
            calls.update(cxr_generator=1, xrv=1); used_images.add(slot[:2])
        calls.update(report_generator=1, chexbert=1)
        observed[slot] = candidate
        trace.append({'step': len(trace), 'action': action, 'request_slot': list(slot),
            'observed_candidate_id': row['triple_candidate_id'], 'reason_codes': reasons,
            'trigger_evidence_ids': evidence, 'charged_model_calls': cost,
            'cumulative_model_calls': sum(calls.values())})
        proposed, proposed_reasons, proposed_ids = original.action_signal(candidate)
        remaining_reports = [(*slot[:2], report) for report in reports if (*slot[:2], report) not in observed]
        decision = {'after_step': len(trace) - 1, 'proposed_original_action': proposed,
            'vetoed_image_branch': False, 'basis': None, 'next_slot': None, 'decision': None}
        if proposed == 'stop_proxy_satisfied':
            terminal = 'stop_proxy_satisfied_not_clinical_acceptance'
            decision['decision'] = terminal; decisions.append(decision); break
        if proposed == 'switch_report_model' and remaining_reports:
            slot, action, reasons, evidence = remaining_reports[0], proposed, proposed_reasons, proposed_ids
            decision.update(next_slot=list(slot), decision='continue_original_report_branch')
            decisions.append(decision); continue
        basis = image_branch_basis(candidate)
        decision['basis'] = basis
        if not basis['cached_image_branch_allowed']:
            decision['vetoed_image_branch'] = True
            if remaining_reports:
                slot, action = remaining_reports[0], 'switch_report_model'
                reasons, evidence = ['unsupported_image_branch_try_remaining_report'], proposed_ids
                decision.update(next_slot=list(slot), decision='veto_image_try_report')
                decisions.append(decision); continue
            terminal = 'stop_unresolved_' + basis['reason']
            decision['decision'] = terminal; decisions.append(decision); break
        unseen_images = [image for image in images if image not in used_images]
        if not unseen_images:
            terminal = 'candidate_inventory_exhausted_unresolved'
            decision['decision'] = terminal; decisions.append(decision); break
        slot, action = (*unseen_images[0], reports[0]), 'regenerate_cxr'
        reasons = ['scope_allows_cached_image_exploration_not_confirmed_repair', basis['reason']]
        evidence = basis['explicit_proxy_opposition_evidence_ids']
        decision.update(next_slot=list(slot), decision='allow_cached_image_branch_unqualified')
        decisions.append(decision)
    eligible = [c for c in observed.values() if c['score_record']['scoring']['selection']['hard_gate_failure_count'] == 0]
    selected = (candidate if terminal == 'stop_proxy_satisfied_not_clinical_acceptance'
        else min(eligible, key=original.candidate_key) if eligible else None)
    first_facts = next(iter(observed.values()))['facts'] if observed else []
    scope = ('explicit_fact_proxy' if any(f['states']['ehr'] in original.EXPLICIT for f in first_facts)
        else 'no_direct_comparable_ehr_facts' if observed else 'not_observed')
    return {'case_id': case, 'method': METHOD, 'model_call_budget': budget, 'random_seed': None,
        'selected_candidate_id': None if selected is None else selected['score_record']['triple_candidate_id'],
        'selected_ehr_sha256': initial['lineage']['ehr_sha256'], 'input_ehr_assessment_scope': scope,
        'simulated_calls': dict(calls), 'simulated_model_calls': sum(calls.values()),
        'observed_candidates': len(observed), 'observed_images': len(used_images),
        'terminal_reason': terminal, 'action_trace': trace, 'decision_trace': decisions,
        'selected_proxy_stop_conditions_met': selected is not None and original.action_signal(selected)[0] == 'stop_proxy_satisfied',
        'actual_regeneration_executed': False, 'clinical_fault_confirmed': False, 'clinical_acceptance': False,
        'alternate_endpoint_used_for_actions_or_selection': False}


def paired_comparisons(means):
    index = {(r['case_id'], r['method'], r['model_call_budget']): r for r in means}
    cases = sorted({r['case_id'] for r in means})
    rows = []
    for cap in baseline.CAPS:
        for method in (*baseline.OLD_METHODS, baseline.NEW_METHOD, 'report_only_static'):
            for scope in ('all', 'explicit_fact_proxy', 'no_direct_comparable_ehr_facts'):
                pairs = [(index[case, METHOD, cap], index[case, method, cap]) for case in cases
                    if scope == 'all' or index[case, METHOD, cap]['ehr_scope'] == scope]
                available = [(a, b) for a, b in pairs if a['biovil_raw_cosine'] is not None and b['biovil_raw_cosine'] is not None]
                ds = [a['biovil_raw_cosine'] - b['biovil_raw_cosine'] for a, b in available]
                rows.append({'ehr_scope': scope, 'model_call_budget': cap, 'baseline': method,
                    'fixed_ehr_cases': len(pairs), 'paired_biovil_cases': len(available),
                    'mean_biovil_delta_guarded_minus_baseline': sum(ds) / len(ds) if ds else None,
                    'guarded_higher': sum(d > 0 for d in ds), 'tied': sum(d == 0 for d in ds),
                    'baseline_higher': sum(d < 0 for d in ds),
                    'mean_simulated_call_delta_guarded_minus_baseline':
                        sum(a['mean_simulated_calls'] - b['mean_simulated_calls'] for a, b in pairs) / len(pairs) if pairs else None,
                    'clinical_accuracy': None})
    return rows


def policy_description():
    return {'schema_version': SCHEMA + '-policy', 'cohort_role': 'already_inspected_development',
        'change': 'veto_unsupported_image_exploration_keep_original_report_rank_and_proxy_stop',
        'allowed_image_branch_basis': ['basic_invalid_image_flag', 'explicit_direct_ehr_xrv_opposition'],
        'no_new_numerical_threshold_or_weight': True, 'report_votes_are_independent_truth': False,
        'alternate_endpoints_used_for_routing': False, 'clinical_acceptance': False,
        'regeneration_authorized': False, 'replace_or_enrich_ehr': False,
        'actual_gpu_savings': None, 'model_call_caps': list(baseline.CAPS)}


def render(summary, comparisons, paired):
    def show(v): return 'NA' if v is None else f'{v:.4f}'
    lines = ['# Scope-guarded stopping / 按证据范围停止', '',
        'Same 80 fixed EHRs/960 cached triples. New deterministic cache policy, not new generation or clinical repair.',
        'No BioViL threshold, endpoint-driven tuning, changed model priority or EHR replacement.', '',
        '| Method | Cap | Mean simulated calls | BioViL mean | CXR–Report support / known | Opposition / known | Comparable coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['ehr_scope'] == 'all':
            lines.append(f"| {r['method']} | {r['model_call_budget']} | {show(r['mean_simulated_calls_mean'])} | "
                f"{show(r['biovil_raw_cosine_mean'])} | {show(r['cxr_report_support_over_known_mean'])} | "
                f"{show(r['cxr_report_opposition_over_known_mean'])} | {show(r['cxr_report_coverage_over_known_mean'])} |")
    lines += ['', '## Paired full-cap comparison / 满预算同 EHR 对照', '',
        '| Baseline | Paired EHRs | BioViL delta guarded − baseline | Guarded higher / tied / lower | Call delta |',
        '|---|---:|---:|---|---:|']
    for r in paired:
        if r['ehr_scope'] == 'all' and r['model_call_budget'] == 30:
            lines.append(f"| {r['baseline']} | {r['paired_biovil_cases']}/{r['fixed_ehr_cases']} | "
                f"{show(r['mean_biovil_delta_guarded_minus_baseline'])} | {r['guarded_higher']} / {r['tied']} / "
                f"{r['baseline_higher']} | {show(r['mean_simulated_call_delta_guarded_minus_baseline'])} |")
    lines += ['', '## Stop status / 停止不是临床成功', '', json.dumps(summary['full_cap_terminal_counts'], sort_keys=True), '',
        'Report-only static is included to expose equivalence with a cheaper fixed-image control.',
        'Raw fourteen-head proxies/global retrieval are not clinical truth; unknown and missing EHR edges remain unavailable.',
        'Scope permits cached exploration, not a confirmed image fault or permission to launch model inference.',
        'Every cap/subgroup is retained; equal caps need not mean equal simulated work. No measured GPU savings.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    baseline.cpu_guard(); started = time.monotonic()
    old, candidates, bank, policy, scorefree, reportonly, sources = load()
    before = {k: sha256_file(p) for k, p in sources.items()}
    guarded = [replay(grid, policy, cap) for case, grid in sorted(bank.items()) for cap in baseline.CAPS]
    for row in guarded:
        baseline.validate_trace(row, candidates)
    means = baseline.case_means([*old, *scorefree, *reportonly, *guarded], candidates)
    comparisons, _ = baseline.comparisons(means)
    paired = paired_comparisons(means)
    full = [r for r in guarded if r['model_call_budget'] == 30]
    summary = {'schema_version': SCHEMA, 'fixed_ehr_cases': 80, 'source_triples': 960,
        'guarded_trials': len(guarded), 'original_trial_exact_replays': len(old),
        'reused_score_free_trials': len(scorefree), 'reused_report_only_static_trials': len(reportonly),
        'full_cap_terminal_counts': dict(Counter(r['terminal_reason'] for r in full)),
        'full_cap_image_count_distribution': dict(Counter(str(r['observed_images']) for r in full)),
        'full_cap_veto_decision_count': sum(d['vetoed_image_branch'] for r in full for d in r['decision_trace']),
        'full_cap_proxy_stops_not_clinical_acceptance': sum(r['selected_proxy_stop_conditions_met'] for r in full),
        'new_model_calls': 0, 'original_selection_changed': False, 'clinical_accuracy': None,
        'primary_metric_eligible': False, 'regeneration_authorized': False,
        'source_ehr_report_image_weight_bodies_opened': False, 'actual_gpu_savings': None,
        'elapsed_cpu_seconds_before_serialization': round(time.monotonic() - started, 6)}
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'frozen_policy.json', policy_description())
        write_private_text(temporary / 'guarded_outcomes.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in guarded))
        write_private_text(temporary / 'case_means.csv', baseline.csv_text(means))
        write_private_text(temporary / 'method_comparison.csv', baseline.csv_text(comparisons))
        write_private_text(temporary / 'paired_case_comparison.csv', baseline.csv_text(paired))
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(summary, comparisons, paired))
        if before != {k: sha256_file(p) for k, p in sources.items()}:
            raise ValueError('immutable_policy_sources_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir()},
            'new_model_calls': 0, 'original_selection_changed': False, 'clinical_accuracy': None,
            'primary_metric_eligible': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=baseline.BASE / 'automatic_replays')
    args = parser.parse_args()
    try:
        path = execute(args.output_root, args.run_id)
    except Exception:
        print(json.dumps({'status': 'failed_cached_scope_guard', 'new_model_calls': 0}))
        raise SystemExit(1) from None
    print(json.dumps({'status': 'completed_cached_scope_guard', 'guarded_trials': 400,
        'new_model_calls': 0, 'manifest_sha256': sha256_file(path / 'manifest.json')}))


if __name__ == '__main__':
    main()

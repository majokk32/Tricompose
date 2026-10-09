#!/usr/bin/env python3
"""Plan exact-artifact evidence acquisition; never execute or rank clinical quality."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import attach_report_gate_availability as tables
import attach_image_verification_availability as image_tables
import image_validity_guard as guard
import verify_cached_image_findings as cached
from contracts import (PROTECTED_ROOT, CHEXPERT_FINDINGS, require_inside,
    sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SCHEMA = 'tricompose-independent-evidence-verification-frontier-v1'
LIVE_SHA = '0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb'
PARENTS = {
    'live': (BASE / 'guarded_image_availability/liveimage_pool960_12645021_001',
        LIVE_SHA, ('candidate_score_table.csv', 'fact_verification_availability.jsonl',
            'unique_image_finding_table.jsonl', 'summary.json')),
    'requests': (BASE / 'image_verification_availability/imagecheck_pool960_12645021_001',
        'd933d9ba37181e2eed31f3f92fd8831d514a313fc225c6a6bb6f4260ad08f1bf',
        ('evidence_request_availability.jsonl',)),
}
EXPLICIT = frozenset(('positive', 'negative'))
HASH_FIELDS = ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')
LIVE_SHARED = ('liveimage_status', 'liveimage_guard_status',
    'liveimage_guard_receipt_sha256', 'liveimage_run_manifest_sha256',
    'liveimage_response_sha256', 'liveimage_baseline_state', 'liveimage_qwen_state',
    'liveimage_readout_repeatability', 'liveimage_readout_changed', 'liveimage_xrv_relation')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_inputs():
    sources = {'worker': Path(__file__), 'tests': ROOT / 'tests/test_verification_frontier.py',
        'protocol': ROOT.parent / 'docs/verification_frontier_protocol.md'}
    for module in (tables, image_tables, guard, cached, cached.image_interface,
            cached.frozen, sys.modules['contracts']):
        sources[module.__name__] = Path(module.__file__)
    for name in ('reliability_preview', 'scorer_reliability', 'legacy_replay_adapter',
            'invariant_verification', 'decision_preview'):
        sources[name] = ROOT / 'src/tricompose_v12' / (name + '.py')
    values = {}
    for label, (root, expected, names) in PARENTS.items():
        mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
        if sha256_file(mp) != expected:
            raise ValueError('fixed_parent_manifest_required')
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        if (manifest['new_model_calls'] != 0 or any(manifest[k] is not False for k in
                ('primary_metric_eligible', 'selection_changed', 'regeneration_authorized'))):
            raise ValueError('unchanged_unqualified_parent_required')
        sources[label + '_manifest'] = mp
        for name in names:
            path = require_inside(root / name, root, must_exist=True)
            if (not path.is_file() or path.stat().st_size > 64 * 1024 * 1024
                    or sha256_file(path) != manifest['artifacts'][name]['sha256']):
                raise ValueError('bounded_fixed_quote_free_artifact_required')
            if name.endswith('.csv'):
                with path.open(encoding='utf-8', newline='') as handle:
                    value = list(csv.DictReader(handle))
                if any(None in r or None in r.values() for r in value):
                    raise ValueError('complete_csv_cells_required')
            elif name.endswith('.jsonl'):
                value = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
            else:
                value = json.loads(path.read_text(encoding='utf-8'))
            values[label, name] = value
            sources[label + '_' + name] = path
        if sha256_file(mp) != expected:
            raise ValueError('parent_changed_during_read')
    return values, sources


def hash_required(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('named_sha256_required')


def validate_fact(fact, candidate):
    if (any(fact[k] != candidate[k] for k in
            ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'report_candidate_id'))
            or set(fact['artifact_hashes']) != set(HASH_FIELDS)
            or any(fact['artifact_hashes'][k] != candidate[k] for k in HASH_FIELDS)
            or fact['finding'] not in CHEXPERT_FINDINGS
            or set(fact['states']) != {'ehr', 'xrv', 'chexbert'}
            or not set(fact['states'].values()) <= guard.STATES):
        raise ValueError('exact_named_finding_lineage_required')
    if any(fact[k] is not False for k in ('liveimage_independent_clinical_validation',
            'liveimage_primary_metric_eligible', 'liveimage_regeneration_authorized',
            'reportgate_independent_clinical_validation', 'reportgate_primary_metric_eligible',
            'reportgate_regeneration_authorized')):
        raise ValueError('unqualified_fact_required')
    if fact['states']['ehr'] in EXPLICIT and not fact['cached_source_categories']:
        raise ValueError('explicit_ehr_evidence_required')
    status = fact['liveimage_status']
    fresh, old = fact['liveimage_qwen_state'], fact['liveimage_baseline_state']
    changed = fact['liveimage_readout_changed']
    if status == 'checked_clinically_unqualified':
        if fact['finding'] not in guard.HEADS or fresh not in guard.STATES:
            raise ValueError('supported_four_state_readout_required')
        for key in ('liveimage_run_manifest_sha256', 'liveimage_response_sha256',
                'liveimage_guard_receipt_sha256'):
            hash_required(fact[key])
        if fact['liveimage_guard_status'] != 'basic_pass_not_clinical':
            raise ValueError('checked_readout_requires_basic_guard')
        if old is None:
            if changed is not None or fact['liveimage_readout_repeatability'] != 'old_verifier_unavailable':
                raise ValueError('missing_baseline_is_not_stable')
        elif (old not in guard.STATES or type(changed) is not bool or changed != (old != fresh)
                or fact['liveimage_readout_repeatability'] !=
                    ('changed_state_unqualified' if changed else 'same_state_unqualified')):
            raise ValueError('exact_readout_repeatability_required')
        if fact['liveimage_xrv_relation'] != cached.relation(fact['states']['xrv'], fresh):
            raise ValueError('unknown_safe_current_relation_required')
    elif status in ('not_checked', 'outside_image_verifier_scope', 'image_verifier_unavailable'):
        if any(v is not None for v in (fresh, old, changed)):
            raise ValueError('unavailable_or_outside_scope_state_must_be_null')
        if status == 'outside_image_verifier_scope' and fact['finding'] in guard.HEADS:
            raise ValueError('supported_head_cannot_be_outside_scope')
        if fact['liveimage_xrv_relation'] != status:
            raise ValueError('unavailable_relation_must_preserve_status')
    else:
        raise ValueError('known_availability_required')
    retained = fact['reportgate_retained_state']
    if retained is not None and (fact['reportgate_status'] != 'scope_commit'
            or retained not in guard.STATES - {'unknown'}):
        raise ValueError('exact_scope_retained_report_state_required')
    if type(fact['liveimage_exact_report_comparison_available']) is not bool:
        raise ValueError('exact_report_comparison_flag_required')
    expected = status
    if status == 'checked_clinically_unqualified':
        expected = ('no_exact_report_check' if not fact['liveimage_exact_report_comparison_available']
            else 'report_assertion_not_retained' if retained is None
            else cached.relation(fresh, retained))
    if fact['liveimage_retained_report_relation'] != expected:
        raise ValueError('exact_unknown_safe_report_relation_required')


def index_inputs(rows, facts, unique):
    if not rows or len(rows) > 960 or len(facts) != len(rows) * 14:
        raise ValueError('bounded_complete_candidate_inventory_required')
    candidates, case_ehr, image_anchors = {}, {}, {}
    image_consumers = defaultdict(list)
    for row in rows:
        cid = row['triple_candidate_id']
        if cid in candidates or any(k.startswith('frontier_') for k in row):
            raise ValueError('unique_unannotated_candidate_required')
        for key in HASH_FIELDS:
            hash_required(row[key])
        candidates[cid] = row
        ehr = tuple(row[k] for k in ('ehr_sha256', 'ehr_facts_sha256'))
        if case_ehr.setdefault(row['case_id'], ehr) != ehr:
            raise ValueError('fixed_case_ehr_required')
        anchor = (row['case_id'], *ehr, row['cxr_sha256'])
        if image_anchors.setdefault(row['cxr_candidate_id'], anchor) != anchor:
            raise ValueError('fixed_exact_shared_image_anchor_required')
        image_consumers[row['cxr_candidate_id']].append(cid)
    lookup, shared, case_facts, grouped = {}, {}, {}, defaultdict(set)
    for fact in facts:
        key = fact['triple_candidate_id'], fact['finding']
        if key in lookup or key[0] not in candidates:
            raise ValueError('unique_existing_candidate_finding_required')
        validate_fact(fact, candidates[key[0]])
        lookup[key] = fact
        grouped[key[0]].add(fact['finding'])
        identity = fact['cxr_candidate_id'], fact['finding']
        signature = (fact['states']['ehr'], fact['states']['xrv'],
            *(fact[k] for k in LIVE_SHARED))
        if shared.setdefault(identity, signature) != signature:
            raise ValueError('same_image_finding_cannot_vary_by_report')
        ehr_fact = (fact['states']['ehr'], fact['cached_source_categories'])
        if case_facts.setdefault((fact['case_id'], fact['finding']), ehr_fact) != ehr_fact:
            raise ValueError('fixed_case_fact_evidence_required')
    if any(heads != set(CHEXPERT_FINDINGS) for heads in grouped.values()):
        raise ValueError('complete_fourteen_finding_inventory_required')
    unique_lookup = {}
    for value in unique:
        key = value['cxr_candidate_id'], value['finding']
        consumers = image_consumers.get(key[0])
        if key in unique_lookup or not consumers or key[1] not in CHEXPERT_FINDINGS:
            raise ValueError('unique_existing_image_finding_required')
        f = lookup[consumers[0], key[1]]
        if (value['cxr_sha256'] != f['artifact_hashes']['cxr_sha256']
                or value['raw_xrv_state'] != f['states']['xrv']
                or any(value[k] != f[k] for k in LIVE_SHARED)
                or f['liveimage_status'] == 'not_checked'):
            raise ValueError('exact_unique_image_readout_required')
        if any(value[k] is not False for k in ('liveimage_independent_clinical_validation',
                'liveimage_primary_metric_eligible', 'liveimage_regeneration_authorized')):
            raise ValueError('unqualified_unique_image_readout_required')
        unique_lookup[key] = value
    expected = {(f['cxr_candidate_id'], f['finding']) for f in facts
        if f['liveimage_status'] != 'not_checked'}
    if set(unique_lookup) != expected:
        raise ValueError('all_and_only_checked_image_inventory_required')
    return candidates, lookup, image_consumers


def image_requirement(fact, pair=False):
    if fact['liveimage_readout_changed'] is True:
        return 0, 'independent_evidence_required_for_unstable_readout'
    if fact['liveimage_xrv_relation'] == 'explicit_opposition_unqualified':
        return 1, 'independent_evidence_required_for_scorer_disagreement'
    status = fact['liveimage_status']
    if status == 'outside_image_verifier_scope':
        return 2, 'alternative_verifier_required_outside_scope'
    if status == 'image_verifier_unavailable':
        return 2, 'verifier_result_unavailable'
    if status == 'not_checked':
        return 3, 'initial_verification_required'
    if pair and not fact['liveimage_exact_report_comparison_available']:
        return 3, 'initial_exact_report_check_required'
    if pair and fact['reportgate_retained_state'] is None:
        return 2, 'report_scope_not_retained'
    return 4, 'cached_readout_available_clinically_unqualified'


def requirement(kind, fact):
    if kind == 'assess_ehr_radiographic_observability':
        return 5, 'ehr_radiographic_observability_unresolved'
    if kind == 'verify_conditioning_and_ehr_fact_scope':
        return 5, 'conditioning_scope_unresolved'
    if kind in ('verify_image_finding', 'verify_image_report_relation'):
        return image_requirement(fact, pair=kind == 'verify_image_report_relation')
    gate = fact['reportgate_status']
    if gate == 'not_checked':
        return 3, 'initial_verification_required'
    if gate == 'outside_verifier_scope':
        return 2, 'alternative_verifier_required_outside_scope'
    if gate == 'verifier_unavailable':
        return 2, 'verifier_result_unavailable'
    if gate in ('abstain', 'no_model_assertion'):
        return 2, 'report_scope_not_retained'
    if gate != 'scope_commit':
        raise ValueError('known_report_gate_status_required')
    return 4, 'cached_readout_available_clinically_unqualified'


def annotate_requests(requests, candidates, facts):
    result, seen = [], set()
    for request in requests:
        rid, kind = request['request_id'], request['request_kind']
        if (rid in seen or kind not in tables.REQUEST_DEPENDENCIES
                or request['execution_status'] != 'not_executed'
                or any(request[k] is not False for k in ('model_execution_allowed',
                    'clinical_truth_established', 'reportcheck_clinically_resolved',
                    'reportgate_clinically_resolved', 'imageverify_clinically_resolved',
                    'imageverify_regeneration_authorized'))
                or any(request[k] != 0 for k in ('reportcheck_new_model_calls',
                    'reportgate_new_model_calls', 'imageverify_new_model_calls'))
                or any(k.startswith('frontier_') for k in request)):
            raise ValueError('unique_unexecuted_unqualified_request_required')
        seen.add(rid)
        consumers = request['consumer_candidate_ids']
        if not consumers or len(set(consumers)) != len(consumers):
            raise ValueError('unique_nonempty_consumers_required')
        decisions, relations, checked, unchecked = [], Counter(), [], 0
        for cid in consumers:
            row = candidates.get(cid)
            if (row is None or row['case_id'] != request['case_id']
                    or set(request['dependency_hashes']) != set(tables.REQUEST_DEPENDENCIES[kind])
                    or any(row[k] != v for k, v in request['dependency_hashes'].items())):
                raise ValueError('exact_same_case_request_dependencies_required')
            if kind == 'assess_ehr_radiographic_observability':
                if request['finding'] is not None:
                    raise ValueError('observability_request_has_no_invented_finding')
                decisions.append(requirement(kind, None))
                continue
            fact = facts.get((cid, request['finding']))
            if fact is None:
                raise ValueError('existing_request_finding_required')
            decisions.append(requirement(kind, fact))
            if kind in ('verify_image_finding', 'verify_image_report_relation'):
                checked_now = fact['liveimage_status'] != 'not_checked'
                relation = fact['liveimage_retained_report_relation'] if kind == 'verify_image_report_relation' else fact['liveimage_xrv_relation']
                relations[relation] += 1
            elif kind == 'verify_report_assertion':
                checked_now = fact['reportgate_status'] != 'not_checked'
                relations[fact['reportgate_status']] += 1
            else:
                checked_now = False
            if checked_now:
                checked.append(cid)
            else:
                unchecked += 1
        tier = min(t for t, _ in decisions)
        statuses = sorted({s for t, s in decisions if t == tier})
        result.append({**request, 'frontier_schema_version': SCHEMA,
            'frontier_priority_tier': tier, 'frontier_next_requirements': statuses,
            'frontier_consumer_requirement_counts': dict(sorted(Counter(s for _, s in decisions).items())),
            'frontier_checked_consumer_ids': sorted(checked),
            'frontier_unchecked_consumer_count': unchecked,
            'frontier_consumer_relation_counts': dict(sorted(relations.items())),
            'frontier_clinically_resolved': False, 'frontier_primary_metric_eligible': False,
            'frontier_model_execution_allowed': False, 'frontier_regeneration_authorized': False,
            'frontier_estimated_model_calls': None, 'frontier_estimated_gpu_seconds': None,
            'frontier_declared_budget': None, 'frontier_new_model_calls': 0})
    image_tables.preserve(requests, result)
    return result


def supplemental_requests(unique, candidates, lookup, consumers):
    result = []
    for value in sorted(unique, key=lambda r: (r['cxr_candidate_id'], r['finding'])):
        tier, need = image_requirement(value)
        if tier not in (0, 1):
            continue
        cids = sorted(consumers[value['cxr_candidate_id']])
        row = candidates[cids[0]]
        reasons = []
        if value['liveimage_readout_changed'] is True:
            reasons.append('changed_same_image_readout_unqualified')
        if value['liveimage_xrv_relation'] == 'explicit_opposition_unqualified':
            reasons.append('explicit_current_image_scorer_opposition_unqualified')
        identity = [SCHEMA, row['case_id'], value['cxr_candidate_id'], value['cxr_sha256'],
            value['finding'], LIVE_SHA]
        result.append({'schema_version': SCHEMA, 'request_id': digest(identity),
            'request_kind': 'obtain_independent_image_finding_evidence',
            'case_id': row['case_id'], 'cxr_candidate_id': value['cxr_candidate_id'],
            'finding': value['finding'], 'dependency_hashes': {'cxr_sha256': value['cxr_sha256']},
            'consumer_candidate_ids': cids,
            'source_evidence_ids': sorted(lookup[cid, value['finding']]['source_evidence_id'] for cid in cids),
            'source_availability_manifest_sha256': LIVE_SHA,
            'source_guarded_run_manifest_sha256': value['liveimage_run_manifest_sha256'],
            'source_guard_receipt_sha256': value['liveimage_guard_receipt_sha256'],
            'source_response_sha256': value['liveimage_response_sha256'],
            'baseline_readout': value['liveimage_baseline_state'], 'current_readout': value['liveimage_qwen_state'],
            'raw_xrv_state': value['raw_xrv_state'], 'reason_codes': reasons,
            'frontier_priority_tier': tier, 'frontier_next_requirements': [need],
            'execution_status': 'not_executed', 'model_execution_allowed': False,
            'regeneration_authorized': False, 'clinical_truth_established': False,
            'clinically_resolved': False, 'confirmed_faulty_modality': None,
            'primary_metric_eligible': False, 'same_model_retry_is_independent_evidence': False,
            'clinical_selection_score': None, 'estimated_model_calls': None,
            'estimated_gpu_seconds': None, 'declared_budget': None, 'new_model_calls': 0})
    if len({r['request_id'] for r in result}) != len(result):
        raise ValueError('unique_supplemental_request_ids_required')
    return result


def candidate_actions(rows, facts, original_requests, supplements):
    links = defaultdict(list)
    for kind, requests in (('original', original_requests), ('supplemental', supplements)):
        for request in requests:
            for cid in request['consumer_candidate_ids']:
                links[cid].append((kind, request))
    result = []
    for row in rows:
        cid = row['triple_candidate_id']
        own = links[cid]
        initial_image = any(facts[cid, head]['liveimage_status'] == 'not_checked' for head in guard.HEADS)
        initial_report = any(facts[cid, head]['reportgate_status'] == 'not_checked' for head in tables.HEADS)
        tier = min((r['frontier_priority_tier'] for _, r in own), default=4)
        if initial_image or initial_report:
            tier = min(tier, 3)
        next_ids = sorted(r['request_id'] for _, r in own if r['frontier_priority_tier'] == tier)
        next_needs = {s for _, r in own if r['frontier_priority_tier'] == tier
            for s in r['frontier_next_requirements']}
        if tier == 3:
            if initial_image:
                next_needs.add('initial_exact_image_check_required')
            if initial_report:
                next_needs.add('initial_exact_report_check_required')
        next_needs = sorted(next_needs) or ['independent_clinical_qualification_missing']
        known = sum(facts[cid, head]['states']['ehr'] in EXPLICIT for head in CHEXPERT_FINDINGS)
        result.append({**row, 'frontier_schema_version': SCHEMA,
            'frontier_action': 'verify_before_clinical_selection',
            'frontier_priority_tier': tier,
            'frontier_next_requirements': json.dumps(next_needs, separators=(',', ':')),
            'frontier_next_request_ids': json.dumps(next_ids, separators=(',', ':')),
            'frontier_initial_image_check_required': initial_image,
            'frontier_initial_report_check_required': initial_report,
            'frontier_next_requirement_has_no_request_id': not bool(next_ids),
            'frontier_original_logical_request_count': sum(k == 'original' for k, _ in own),
            'frontier_supplemental_logical_request_count': sum(k == 'supplemental' for k, _ in own),
            'frontier_pending_logical_request_count': len(own),
            'frontier_direct_ehr_fact_count': known,
            'frontier_ehr_observability': 'direct_findings_available_unqualified' if known else 'no_direct_radiographic_reference',
            'frontier_report_consumers_are_independent_image_votes': False,
            'frontier_clinical_selection_score': None, 'frontier_clinical_selection_eligible': False,
            'frontier_confirmed_faulty_modality': None, 'frontier_regeneration_authorized': False,
            'frontier_model_execution_allowed': False, 'frontier_primary_metric_eligible': False,
            'frontier_estimated_model_calls': None, 'frontier_estimated_gpu_seconds': None,
            'frontier_declared_budget': None})
    image_tables.preserve(rows, result)
    return result


def build(rows, facts, unique, requests):
    candidates, lookup, consumers = index_inputs(rows, facts, unique)
    original = annotate_requests(requests, candidates, lookup)
    supplements = supplemental_requests(unique, candidates, lookup, consumers)
    if set(r['request_id'] for r in original) & set(r['request_id'] for r in supplements):
        raise ValueError('supplemental_requests_must_not_overwrite_history')
    actions = candidate_actions(rows, lookup, original, supplements)
    ehr = {r['case_id']: r['frontier_direct_ehr_fact_count'] for r in actions}
    summary = {'schema_version': SCHEMA, 'candidate_rows': len(actions), 'fact_rows': len(facts),
        'fixed_ehr_cases': len(ehr), 'fixed_image_slots': len(consumers),
        'original_logical_requests': len(original), 'supplemental_logical_requests': len(supplements),
        'total_logical_requests': len(original) + len(supplements),
        'original_candidate_request_links': sum(len(r['consumer_candidate_ids']) for r in original),
        'supplemental_candidate_request_links': sum(len(r['consumer_candidate_ids']) for r in supplements),
        'supplemental_unique_image_slots': len({r['cxr_candidate_id'] for r in supplements}),
        'supplemental_priority_counts': dict(sorted(Counter(str(r['frontier_priority_tier']) for r in supplements).items())),
        'original_request_priority_counts': dict(sorted(Counter(str(r['frontier_priority_tier']) for r in original).items())),
        'candidate_priority_counts': dict(sorted(Counter(str(r['frontier_priority_tier']) for r in actions).items())),
        'candidate_slots_requiring_initial_image_check': sum(r['frontier_initial_image_check_required'] for r in actions),
        'candidate_slots_requiring_initial_report_check': sum(r['frontier_initial_report_check_required'] for r in actions),
        'candidate_slots_with_no_next_request_id': sum(r['frontier_next_requirement_has_no_request_id'] for r in actions),
        'cases_without_direct_radiographic_reference': sum(n == 0 for n in ehr.values()),
        'cases_with_direct_radiographic_reference': sum(n > 0 for n in ehr.values()),
        'original_candidate_and_request_cells_preserved': True,
        'old_execution_history_unchanged': True, 'selection_changed': False,
        'fixed_ehr_changed': False, 'primary_metric_eligible': False,
        'model_execution_allowed': False, 'regeneration_authorized': False,
        'clinically_resolved_requests': 0, 'clinical_accuracy': None,
        'declared_budget': None, 'cost_estimates': None,
        'new_model_calls': 0, 'new_slurm_submissions': 0,
        'report_image_ehr_bodies_opened': False, 'raw_patient_inputs_opened': False,
        'same_model_readout_agreement_is_clinical_truth': False,
        'priority_is_clinical_severity_or_quality_score': False,
        'development_planning_not_heldout_evaluation': True}
    return actions, original, supplements, summary


def execute(output_root, run_id):
    tables.require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        inputs, sources = load_inputs()
        before = {k: sha256_file(p) for k, p in sources.items()}
        rows = inputs['live', 'candidate_score_table.csv']
        facts = inputs['live', 'fact_verification_availability.jsonl']
        unique = inputs['live', 'unique_image_finding_table.jsonl']
        requests = inputs['requests', 'evidence_request_availability.jsonl']
        if (len(rows), len(facts), len(unique), len(requests)) != (960, 13440, 84, 2072):
            raise ValueError('fixed_full_bank_inventory_required')
        actions, annotated, supplements, summary = build(rows, facts, unique, requests)
        parent = inputs['live', 'summary.json']
        if (summary['fixed_ehr_cases'] != parent['fixed_ehr_cases']
                or summary['fixed_image_slots'] != parent['image_slots']
                or sum(r['frontier_priority_tier'] == 0 for r in supplements) != parent['unique_changed_readout_slots']):
            raise ValueError('fixed_parent_denominators_required')
        summary['existing_cpu_job_id'] = os.environ['SLURM_JOB_ID']
        write_private_text(temporary / 'candidate_action_table.csv', tables.csv_text(actions))
        write_private_text(temporary / 'evidence_request_frontier.jsonl', tables.jsonl_text(annotated))
        write_private_text(temporary / 'supplemental_image_evidence_requests.jsonl', tables.jsonl_text(supplements))
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md',
            '# Verification frontier / 下一步验证动作表\n\n'
            '保留旧分数、缺失值与执行历史；优先级是工程约定，不是临床分数。\n\n'
            'Supplemental requests are deduplicated image/finding plans, not actual model calls.\n\n'
            '不稳定读数及评分器分歧需要独立证据；不自动更换 EHR 或生成任何模态。\n\n'
            'No new inference, clinical acceptance, selection, localization accuracy or repair.\n\n'
            '```json\n' + json.dumps(summary, sort_keys=True, indent=2) + '\n```\n')
        if any(sha256_file(p) != before[k] for k, p in sources.items()):
            raise ValueError('consumed_source_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()},
            'source_sha256': before, 'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'report_image_ehr_bodies_opened': False})
        for p in (temporary, *temporary.iterdir()):
            st = p.stat()
            if st.st_gid not in (96293, 65534) or st.st_mode & 0o7777 != (0o2770 if p.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE / 'verification_frontiers')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = execute(args.output_root, args.run_id)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'verification_frontier_prepared',
        'candidate_rows': summary['candidate_rows'], 'original_logical_requests': summary['original_logical_requests'],
        'supplemental_logical_requests': summary['supplemental_logical_requests'],
        'new_model_calls': 0, 'manifest_sha256': sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

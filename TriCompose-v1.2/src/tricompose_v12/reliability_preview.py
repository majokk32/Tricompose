"""Evidence-request preview for the lossless historical-bank reliability sidecar.

No ranking, clinical verdict, execution, rejection or repair. Shared-artifact
requests are deduplicated; logical requests are not model invocations or costs.
The older eight-finding scoped decision interface is intentionally unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from .decision_preview import Budget, budget_status
from .scorer_reliability import (SCHEMA as SOURCE_SCHEMA, EDGES, EXPLICIT,
    FINDINGS, HASH_FIELDS, PROFILE, STATES, _edge, digest, status)

SCHEMA = 'tricompose-reliability-evidence-request-preview-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z')
REQUEST_DEPENDENCIES = {
    'assess_ehr_radiographic_observability': ('ehr_sha256', 'ehr_facts_sha256'),
    'verify_conditioning_and_ehr_fact_scope': ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256'),
    'verify_image_finding': ('cxr_sha256',),
    'verify_report_assertion': ('report_sha256',),
    'verify_image_report_relation': ('cxr_sha256', 'report_sha256'),
}


def _false(value):
    return value is False or type(value) is str and value == 'false'


def _missing(value):
    return value is None or type(value) is str and value == ''


def _matches(value, expected):
    if expected is None:
        return _missing(value)
    if type(value) is bool or _missing(value):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and math.isclose(number, expected, rel_tol=0, abs_tol=1e-8)


def validate_sidecar(rows, facts, groups):
    """Validate metadata only; do not manufacture missing scope/scorer evidence."""
    if not 1 <= len(rows) <= 960 or len(facts) != len(rows) * len(FINDINGS):
        raise ValueError('bounded_complete_reliability_inventory_required')
    index = {}
    false_columns = ('reliability_primary_metric_eligible', 'reliability_automatic_regeneration_authorized',
                     'reliability_benchmark_metric_transferred', 'reliability_shared_image_report_votes_independent')
    for row in rows:
        cid = row['triple_candidate_id']
        if cid in index or any(not isinstance(row[key], str) or not _ID.fullmatch(row[key]) for key in
                ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'report_candidate_id')):
            raise ValueError('duplicate_or_invalid_candidate_identity')
        if (row['profile'] != PROFILE or row['reliability_source_profile'] != PROFILE
                or row['reliability_profile_id'] != SOURCE_SCHEMA
                or row['reliability_clinical_verdict'] != 'unverified'
                or any(not _false(row[key]) for key in false_columns)
                or not _missing(row['reliability_confirmed_faulty_modality'])
                or not _missing(row['reliability_image_scorer_disagreement'])
                or row['reliability_image_scorer_disagreement_status'] != 'not_measured_on_this_synthetic_candidate'
                or any(not _missing(row['reliability_'+edge+'_clinical_score']) for edge in EDGES)):
            raise ValueError('unverified_unchanged_legacy_profile_required')
        if any(not isinstance(row[key], str) or not _HASH.fullmatch(row[key]) for key in HASH_FIELDS):
            raise ValueError('invalid_artifact_hash')
        index[cid] = row
    by_candidate, peers = defaultdict(list), defaultdict(dict)
    fixed, shared, evidence_ids = {}, {}, set()
    for fact in facts:
        cid = fact['triple_candidate_id']
        if cid not in index:
            raise ValueError('foreign_candidate_fact')
        row, states = index[cid], fact['states']
        if (fact['schema_version'] != SOURCE_SCHEMA or fact['profile'] != PROFILE
                or fact['finding'] not in FINDINGS or set(states) != {'ehr', 'xrv', 'chexbert'}
                or any(value not in STATES for value in states.values())
                or fact['artifact_hashes'] != {key: row[key] for key in HASH_FIELDS}
                or any(fact[key] != row[key] for key in ('case_id', 'cxr_candidate_id', 'report_candidate_id'))
                or fact['clinical_truth_available'] is not False or fact['clinical_conflict_verified'] is not None
                or fact['confirmed_faulty_modality'] is not None or fact['automatic_regeneration_authorized'] is not False
                or fact['benchmark_metric_transferred_to_candidate'] is not False
                or fact['candidate_same_fact_image_scorer_disagreement'] is not None
                or fact['image_scorer_disagreement_status'] != 'not_measured_on_this_synthetic_candidate'):
            raise ValueError('fact_scope_lineage_or_eligibility_changed')
        eid = fact['source_evidence_id']
        categories = fact['cached_source_categories']
        if (not isinstance(eid, str) or not _HASH.fullmatch(eid) or eid in evidence_ids
                or not isinstance(categories, list) or any(not isinstance(c, str) or not _ID.fullmatch(c) for c in categories)
                or states['ehr'] != 'unknown' and not categories):
            raise ValueError('cached_fact_provenance_missing_or_duplicate')
        evidence_ids.add(eid)
        finding = fact['finding']
        for key, value in (
                (('ehr', row['case_id'], finding), (row['ehr_sha256'], row['ehr_facts_sha256'], states['ehr'], tuple(categories))),
                (('image', row['cxr_sha256'], finding), states['xrv']),
                (('report', row['report_sha256'], finding), states['chexbert'])):
            if shared.setdefault(key, value) != value:
                raise ValueError('fixed_or_shared_artifact_evidence_changed')
        anchor = (row['ehr_sha256'], row['ehr_facts_sha256'])
        if fixed.setdefault(row['case_id'], anchor) != anchor:
            raise ValueError('fixed_ehr_anchor_changed')
        expected = {edge: ('not_comparable' if 'uncertain' in (states[left], states[right]) else
                          'unknown' if 'unknown' in (states[left], states[right]) else
                          'support' if states[left] == states[right] else 'proxy_opposition')
                    for edge, (left, right) in EDGES.items()}
        if fact['relations'] != expected:
            raise ValueError('raw_fact_relation_changed')
        group_id = digest((row['case_id'], row['cxr_sha256'], finding))
        if fact['report_dependency_group'] != group_id:
            raise ValueError('report_dependency_identity_changed')
        peers[group_id][row['report_sha256']] = states['chexbert']
        by_candidate[cid].append(fact)
    group_index = {}
    for group in groups:
        gid = group['dependency_group_id']
        if gid in group_index or gid not in peers:
            raise ValueError('foreign_or_duplicate_dependency_group')
        counts = Counter(peers[gid].values())
        expected_disagreement = bool(counts['positive'] and counts['negative'])
        if (gid != digest((group['case_id'], group['cxr_sha256'], group['finding']))
                or group['state_counts'] != {state: counts[state] for state in sorted(STATES)}
                or not all(type(count) is int for count in group['state_counts'].values())
                or type(group['unique_report_artifacts']) is not int
                or group['unique_report_artifacts'] != sum(counts.values())
                or group['explicit_positive_negative_disagreement'] is not expected_disagreement
                or group['independent_votes'] is not False or group['clinical_conflict_verified'] is not None
                or group['confirmed_faulty_modality'] is not None):
            raise ValueError('correlated_report_group_changed')
        group_index[gid] = group
    if set(group_index) != set(peers):
        raise ValueError('complete_dependency_inventory_required')
    for row in rows:
        candidate = by_candidate[row['triple_candidate_id']]
        if len(candidate) != len(FINDINGS) or {f['finding'] for f in candidate} != set(FINDINGS):
            raise ValueError('complete_fourteen_finding_inventory_required')
        for fact in candidate:
            group = group_index[fact['report_dependency_group']]
            if (fact['same_image_report_expert_proxy_disagreement'] is not group['explicit_positive_negative_disagreement']
                    or type(fact['correlated_report_artifact_count']) is not int
                    or fact['correlated_report_artifact_count'] != group['unique_report_artifacts']):
                raise ValueError('fact_dependency_annotation_changed')
        if not _matches(row['reliability_report_expert_disagreement_findings'],
                        sum(f['same_image_report_expert_proxy_disagreement'] for f in candidate)):
            raise ValueError('candidate_report_disagreement_count_changed')
        for edge, (left, right) in EDGES.items():
            value = _edge([f['states'] for f in candidate], left, right)
            if (any(not _matches(row[edge+'_'+key], expected) for key, expected in value.items())
                    or row['reliability_'+edge+'_evidence_status'] != status(value)):
                raise ValueError('raw_edge_or_availability_changed')
    return by_candidate


def preview_reliability(rows, facts, groups, *, budget=None, history=(), verification_gpu_seconds=None):
    """Produce auditable evidence requests only. All cases and anchors are kept."""
    by_candidate = validate_sidecar(rows, facts, groups)
    if budget is None:
        if history or verification_gpu_seconds is not None:
            raise ValueError('explicit_budget_required_for_cost_accounting')
        ledger = {'verification_affordability': 'additional_budget_not_configured',
                  'max_additional_model_calls': None, 'max_additional_gpu_seconds': None,
                  'attempted_additional_calls': 0, 'remaining_additional_calls': None,
                  'remaining_additional_gpu_seconds': None, 'verification_gpu_seconds_estimate': None,
                  'already_generated_bank_cost_excluded': True, 'actual_compute_savings': None}
    else:
        ledger = budget_status(budget, history, verification_gpu_seconds=verification_gpu_seconds)
    blocked = ledger['verification_affordability'] in {'budget_exhausted', 'verification_exceeds_remaining_time'}
    requests, decisions = {}, []

    def request(row, kind, finding, reason, evidence_id=None):
        dependencies = {key: row[key] for key in REQUEST_DEPENDENCIES[kind]}
        rid = digest([SCHEMA, row['case_id'], kind, finding, dependencies])
        if rid not in requests:
            requests[rid] = {'request_id': rid, 'case_id': row['case_id'], 'request_kind': kind,
                'finding': finding, 'dependency_hashes': dependencies, 'consumer_candidate_ids': set(),
                'source_evidence_ids': set(), 'reason_codes': set(), 'execution_status': 'not_executed',
                'model_execution_allowed': False, 'clinical_truth_established': False,
                'estimated_model_calls': None, 'estimated_gpu_seconds': None,
                'blocked_by_declared_budget': blocked}
        requests[rid]['consumer_candidate_ids'].add(row['triple_candidate_id'])
        requests[rid]['reason_codes'].add(reason)
        if evidence_id is not None:
            requests[rid]['source_evidence_ids'].add(evidence_id)
        return rid

    for row in rows:
        candidate = sorted(by_candidate[row['triple_candidate_id']], key=lambda f: FINDINGS.index(f['finding']))
        patterns, reasons, request_ids, trace = Counter(), set(), set(), []
        known = sum(f['states']['ehr'] in EXPLICIT for f in candidate)
        reasons.update(('independent_clinical_evidence_missing', 'cached_proxy_scores_do_not_authorize_repair',
                        'same_finding_image_scorer_check_not_measured', 'report_scope_not_computed_for_legacy_profile'))
        if not known:
            reasons.add('no_explicit_ehr_reference_facts')
            request_ids.add(request(row, 'assess_ehr_radiographic_observability', None,
                'absence_of_direct_comparison_facts_is_not_a_negative_or_invalid_ehr'))
        for fact in candidate:
            ehr, image, report = (fact['states'][key] for key in ('ehr', 'xrv', 'chexbert'))
            fid, eid = fact['finding'], fact['source_evidence_id']
            comparable = all(state in EXPLICIT for state in (ehr, image, report))
            pattern = ('not_three_way_comparable' if not comparable else
                       'all_three_proxies_agree_unverified' if ehr == image == report else
                       'report_differs_from_ehr_and_image_proxies' if ehr == image else
                       'image_differs_from_ehr_and_report_proxies' if ehr == report else
                       'both_downstream_proxies_oppose_ehr')
            patterns[pattern] += 1
            fact_requests = set()
            def add(kind, reason):
                fact_requests.add(request(row, kind, fid, reason, eid))
                reasons.add(reason)
            if ehr in EXPLICIT:
                if image not in EXPLICIT:
                    add('verify_image_finding', 'image_evidence_missing_for_explicit_ehr_fact')
                elif image != ehr:
                    add('verify_image_finding', 'ehr_image_proxy_opposition_unverified')
                if report not in EXPLICIT:
                    add('verify_report_assertion', 'report_evidence_missing_for_explicit_ehr_fact')
                elif report != ehr:
                    add('verify_report_assertion', 'ehr_report_proxy_opposition_unverified')
            if pattern == 'both_downstream_proxies_oppose_ehr':
                add('verify_conditioning_and_ehr_fact_scope', 'shared_image_reports_cannot_adjudicate_ehr_conditioning')
            if pattern == 'image_differs_from_ehr_and_report_proxies':
                reasons.add('cxr_conditioned_report_is_not_independent_image_truth')
            if fact['relations']['cxr_report'] == 'proxy_opposition':
                add('verify_image_report_relation', 'image_report_proxy_opposition_unverified')
            if fact['same_image_report_expert_proxy_disagreement']:
                add('verify_image_finding', 'correlated_same_image_reports_disagree')
                if report in EXPLICIT:
                    add('verify_report_assertion', 'correlated_same_image_reports_disagree')
            if comparable and pattern == 'all_three_proxies_agree_unverified':
                reasons.add('proxy_agreement_is_not_clinical_acceptance')
            if fact_requests or comparable:
                trace.append({'source_evidence_id': eid, 'finding': fid, 'states': dict(fact['states']),
                    'proxy_pattern': pattern, 'request_ids': sorted(fact_requests),
                    'same_image_report_expert_proxy_disagreement': fact['same_image_report_expert_proxy_disagreement']})
            request_ids.update(fact_requests)
        if blocked:
            reasons.add(ledger['verification_affordability'])
        action = ('stop_unresolved_preview' if blocked else
                  'request_verification_preview' if request_ids else 'retain_unresolved_preview')
        decisions.append({'schema_version': SCHEMA, 'profile': PROFILE, 'case_id': row['case_id'],
            'triple_candidate_id': row['triple_candidate_id'], 'artifact_hashes': {key: row[key] for key in HASH_FIELDS},
            'preview_action': action, 'reason_codes': sorted(reasons), 'request_ids': sorted(request_ids),
            'finding_inventory': len(candidate), 'explicit_ehr_reference_facts': known,
            'proxy_pattern_counts': dict(sorted(patterns.items())), 'finding_trace': trace,
            'edge_evidence_status': {edge: row['reliability_'+edge+'_evidence_status'] for edge in EDGES},
            'budget': dict(ledger), 'source_row_order_retained': True, 'case_rejected': False,
            'fixed_ehr_retained': True, 'clinical_acceptance': False, 'confirmed_faulty_modality': None,
            'clinical_selection_score': None, 'model_execution_allowed': False,
            'regeneration_authorized': False, 'selection_changed': False})
    deduplicated = []
    for rid in sorted(requests):
        payload = requests[rid]
        deduplicated.append({key: sorted(value) if isinstance(value, set) else value for key, value in payload.items()})
    unique_cases = {row['case_id'] for row in rows}
    no_fact_cases = {d['case_id'] for d in decisions if not d['explicit_ehr_reference_facts']}
    summary = {'schema_version': SCHEMA, 'profile': PROFILE, 'candidate_rows': len(decisions),
        'fixed_ehr_cases': len(unique_cases), 'image_slots': len({(r['case_id'], r['cxr_candidate_id']) for r in rows}),
        'fact_rows': len(facts), 'fixed_ehr_cases_without_explicit_facts': len(no_fact_cases),
        'action_counts': dict(sorted(Counter(d['preview_action'] for d in decisions).items())),
        'reason_counts_candidate_rows': dict(sorted(Counter(code for d in decisions for code in d['reason_codes']).items())),
        'proxy_pattern_counts_fact_rows': dict(sorted(sum((Counter(d['proxy_pattern_counts']) for d in decisions), Counter()).items())),
        'candidate_request_links': sum(len(d['request_ids']) for d in decisions),
        'deduplicated_logical_evidence_requests': len(deduplicated),
        'request_counts_by_kind': dict(sorted(Counter(r['request_kind'] for r in deduplicated).items())),
        'requests_are_model_calls': False, 'estimated_additional_model_calls': None,
        'estimated_additional_gpu_seconds': None, 'new_model_calls': 0,
        'clinical_localization_accuracy': None, 'actual_compute_savings': None,
        'cases_dropped_or_rejected': 0, 'fixed_ehr_changed': False,
        'selection_changed': False, 'regeneration_authorized': False, 'clinical_acceptance': False,
        'primary_metric_eligible': False, 'budget': dict(ledger)}
    return decisions, deduplicated, summary

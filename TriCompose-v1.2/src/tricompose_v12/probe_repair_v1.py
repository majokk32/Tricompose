"""Observed-action feedback controller, frozen proxies and no clinical claims.

The controller receives ONE completed observation at a time. It cannot inspect
unrequested candidates or the secondary endpoint. A changed score is not enough
to commit a replacement: exact fact identities and comparison coverage must be
preserved. Reports sharing an image are never counted as independent votes.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

from .invariant_verification import _candidate_facts
from .legacy_replay_adapter import FINDINGS, STATES

VERSION = 'tricompose-observed-probe-repair-v1'
EXPLICIT = frozenset(('positive', 'negative'))
EDGES = ('ehr_cxr', 'ehr_report', 'cxr_report')
COST = {'report_probe': 2, 'image_probe': 4}
HASH = re.compile(r'[a-f0-9]{64}\Z')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def snapshot(candidate):
    """Project a validated cache candidate into a text/pixel/endpoint-free view."""
    facts = _candidate_facts(candidate)
    row = candidate['score_record']
    lineage = row['lineage']
    value = {'schema_version': VERSION + '-observation', 'case_id': row['case_id'],
        'candidate_id': row['triple_candidate_id'],
        'lineage': {k: lineage[k] for k in ('ehr_sha256', 'ehr_facts_sha256',
            'cxr_sha256', 'report_sha256', 'cxr_candidate_id', 'report_candidate_id',
            'cxr_model_id', 'cxr_seed', 'report_model_id')},
        'states': {key: {f: facts[f]['states'][key] for f in FINDINGS}
            for key in ('ehr', 'xrv', 'chexbert')},
        'ehr_sources': {f: sorted(facts[f]['source_categories']) for f in FINDINGS},
        'quality': {k: row['scoring']['modality_quality'][k] for k in
            ('cxr_basic_validity_pass', 'report_structure_quality_score_0_1')},
        'artifact_gate_failures': row['scoring']['selection']['hard_gate_failure_count'],
        'clinical_qualified': False}
    validate_snapshot(value)
    return value


def validate_snapshot(value):
    fields = {'schema_version', 'case_id', 'candidate_id', 'lineage', 'states',
              'ehr_sources', 'quality', 'artifact_gate_failures', 'clinical_qualified'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema_version'] != VERSION + '-observation'
            or value['clinical_qualified'] is not False
            or any(not isinstance(value[k], str) or not ID.fullmatch(value[k])
                   for k in ('case_id', 'candidate_id'))):
        raise ValueError('exact_nonclinical_observation_required')
    lin = value['lineage']
    required = {'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256',
                'cxr_candidate_id', 'report_candidate_id', 'cxr_model_id',
                'cxr_seed', 'report_model_id'}
    if (not isinstance(lin, dict) or set(lin) != required
            or any(not isinstance(lin[k], str) or not HASH.fullmatch(lin[k])
                   for k in required if k.endswith('sha256'))
            or any(not isinstance(lin[k], str) or not ID.fullmatch(lin[k])
                   for k in required if k != 'cxr_seed' and not k.endswith('sha256'))
            or type(lin['cxr_seed']) is not int or lin['cxr_seed'] < 0):
        raise ValueError('opaque_hash_bound_lineage_required')
    states = value['states']
    if (not isinstance(states, dict) or set(states) != {'ehr', 'xrv', 'chexbert'}
            or any(not isinstance(v, dict) or set(v) != set(FINDINGS)
                   or any(s not in STATES for s in v.values()) for v in states.values())):
        raise ValueError('complete_four_state_vectors_required')
    sources = value['ehr_sources']
    if (not isinstance(sources, dict) or set(sources) != set(FINDINGS)
            or any(not isinstance(v, list) or any(not isinstance(s, str) or not ID.fullmatch(s)
                   for s in v) or v != sorted(set(v)) for v in sources.values())
            or any(states['ehr'][f] != 'unknown' and not sources[f] for f in FINDINGS)):
        raise ValueError('fixed_ehr_provenance_required')
    q = value['quality']
    if (not isinstance(q, dict) or set(q) != {'cxr_basic_validity_pass', 'report_structure_quality_score_0_1'}
            or type(q['cxr_basic_validity_pass']) is not bool
            or type(value['artifact_gate_failures']) is not int or value['artifact_gate_failures'] < 0):
        raise ValueError('typed_artifact_quality_required')
    score = q['report_structure_quality_score_0_1']
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('finite_structure_proxy_or_explicit_na_required')
    return value


def edges(value):
    validate_snapshot(value)
    states = value['states']
    result = {}
    for edge, left, right in (('ehr_cxr', 'ehr', 'xrv'), ('ehr_report', 'ehr', 'chexbert'),
                              ('cxr_report', 'xrv', 'chexbert')):
        known = {f for f in FINDINGS if states[left][f] in EXPLICIT}
        comparable = {f for f in known if states[right][f] in EXPLICIT}
        support = {f for f in comparable if states[left][f] == states[right][f]}
        result[edge] = {'known': known, 'comparable': comparable, 'support': support,
            'positive_support': {f for f in support if states[left][f] == 'positive'},
            'opposition': comparable - support}
    return result


def action_credit(before, after, action):
    """Action effect on exact proxy fact sets; not causal or clinical attribution."""
    if action not in COST:
        raise ValueError('registered_probe_action_required')
    a, b = edges(before), edges(after)
    if (before['case_id'] != after['case_id']
            or any(before['lineage'][k] != after['lineage'][k] for k in ('ehr_sha256', 'ehr_facts_sha256'))
            or before['states']['ehr'] != after['states']['ehr']
            or before['ehr_sources'] != after['ehr_sources']):
        raise ValueError('immutable_ehr_anchor_required')
    if before['candidate_id'] == after['candidate_id']:
        raise ValueError('new_completed_candidate_required')
    if action == 'report_probe' and (any(before['lineage'][k] != after['lineage'][k]
            for k in ('cxr_sha256', 'cxr_candidate_id', 'cxr_model_id', 'cxr_seed'))
            or before['states']['xrv'] != after['states']['xrv']):
        raise ValueError('report_probe_must_hold_image_and_classifier_fixed')
    if action == 'image_probe' and before['lineage']['report_model_id'] != after['lineage']['report_model_id']:
        raise ValueError('paired_image_probe_holds_report_expert_fixed')
    resolved = {e: sorted(a[e]['opposition'] & b[e]['support']) for e in EDGES}
    new = {e: sorted(b[e]['opposition'] - a[e]['opposition']) for e in EDGES}
    lost_comparable = {e: sorted(a[e]['comparable'] - b[e]['comparable']) for e in EDGES}
    lost_support = {e: sorted(a[e]['positive_support' if e == 'cxr_report' else 'support']
                             - b[e]['positive_support' if e == 'cxr_report' else 'support']) for e in EDGES}
    gained = {e: sorted(b[e]['positive_support' if e == 'cxr_report' else 'support']
                       - a[e]['positive_support' if e == 'cxr_report' else 'support']) for e in EDGES}
    silenced = {e: sorted(a[e]['opposition'] - b[e]['comparable']) for e in EDGES}
    q0, q1 = (v['quality']['report_structure_quality_score_0_1'] for v in (before, after))
    quality_available = q0 is not None and q1 is not None
    quality_ok = quality_available and q1 >= q0
    artifact_ok = after['artifact_gate_failures'] == 0 and after['quality']['cxr_basic_validity_pass']
    artifact_gain = action == 'image_probe' and not before['quality']['cxr_basic_validity_pass'] and artifact_ok
    target_edges = ('ehr_report', 'cxr_report') if action == 'report_probe' else ('ehr_cxr',)
    gain = artifact_gain or any(resolved[e] or gained[e] for e in target_edges)
    duplicate = before['lineage']['report_sha256' if action == 'report_probe' else 'cxr_sha256'] == \
        after['lineage']['report_sha256' if action == 'report_probe' else 'cxr_sha256']
    reasons = []
    if duplicate: reasons.append('duplicate_bytes_not_new_evidence')
    if not artifact_ok: reasons.append('new_artifact_invalid')
    if not quality_available: reasons.append('structure_comparison_unavailable')
    elif not quality_ok: reasons.append('structure_proxy_regressed')
    if any(new.values()): reasons.append('new_explicit_proxy_opposition')
    if any(lost_comparable.values()): reasons.append('lost_comparable_fact_ids')
    if any(lost_support.values()): reasons.append('lost_protected_supported_fact_ids')
    if any(silenced.values()): reasons.append('conflict_silenced_not_resolved')
    if not gain: reasons.append('no_strict_target_action_gain')
    return {'schema_version': VERSION + '-credit', 'action': action,
        'before_observation_sha256': digest(before), 'after_observation_sha256': digest(after),
        'resolved_proxy_conflicts': resolved, 'new_proxy_conflicts': new,
        'lost_comparable_fact_ids': lost_comparable, 'lost_protected_fact_ids': lost_support,
        'gained_supported_fact_ids': gained, 'silenced_conflict_fact_ids': silenced,
        'coverage_delta': {e: len(b[e]['comparable']) - len(a[e]['comparable']) for e in EDGES},
        'target_gain_observed': bool(gain), 'quality_nonregression': bool(quality_ok),
        'duplicate_bytes': duplicate, 'replacement_allowed_under_proxy_contract': not reasons,
        'reasons': reasons, 'clinical_fault_location': None, 'clinical_repair_success': False}


def needs(value):
    es = edges(value)
    s = value['states']
    return {'image': not value['quality']['cxr_basic_validity_pass'] or bool(es['ehr_cxr']['opposition']),
        'report': bool(es['ehr_report']['opposition'] or es['cxr_report']['opposition'])
            or value['artifact_gate_failures'] > 0
            or any(s['chexbert'][f] not in EXPLICIT and
                   (s['xrv'][f] == 'positive' or s['ehr'][f] in EXPLICIT) for f in FINDINGS)}


class ProbeController:
    """A bounded observe/propose/complete loop; no full-bank scoring interface."""

    def __init__(self, initial, image_order, report_order, budget, *, feedback_enabled=True):
        validate_snapshot(initial)
        if (type(budget) is not int or budget < 4 or type(feedback_enabled) is not bool
                or not image_order or not report_order
                or len(set(map(tuple, image_order))) != len(image_order)
                or len(set(report_order)) != len(report_order)
                or any(len(x) != 2 or not isinstance(x[0], str) or not ID.fullmatch(x[0])
                       or type(x[1]) is not int or x[1] < 0 for x in image_order)
                or any(not isinstance(x, str) or not ID.fullmatch(x) for x in report_order)):
            raise ValueError('registered_order_and_at_least_initial_four_call_budget_required')
        if self._slot(initial) != (*tuple(image_order[0]), report_order[0]):
            raise ValueError('predeclared_initial_path_required')
        self.current = deepcopy(initial)
        self.initial = deepcopy(initial)
        self.image_order = [tuple(x) for x in image_order]
        self.report_order = list(report_order)
        self.budget = budget
        self.feedback_enabled = feedback_enabled
        self.spent = 4  # CXR, XRV, report, CheXbert: simulated diagnostic units.
        self.attempted = {self._slot(initial)}
        self.observed_ids = {initial['candidate_id']}
        self.history = []
        self.pending = None

    @staticmethod
    def _slot(value):
        lin = value['lineage']
        return (lin['cxr_model_id'], lin['cxr_seed'], lin['report_model_id'])

    def propose(self):
        if self.pending is not None:
            raise RuntimeError('complete_reserved_probe_before_next_proposal')
        required = needs(self.current)
        current = self._slot(self.current)
        reports = [(*current[:2], m) for m in self.report_order if (*current[:2], m) not in self.attempted]
        images = [(*slot, current[2]) for slot in self.image_order
                  if slot != current[:2] and not any(x[:2] == slot for x in self.attempted)]
        feasible = {}
        for action, needed, slots in (('report_probe', required['report'], reports),
                                     ('image_probe', required['image'], images)):
            if needed and slots and self.spent + COST[action] <= self.budget:
                feasible[action] = slots[0]
        if not feasible:
            return {'action': 'stop', 'reason': 'no_proxy_target_detected' if not any(required.values())
                else 'unresolved_or_no_affordable_registered_probe',
                'clinical_acceptance': False}
        # Start with a cheap report probe if BOTH branches have a target. If it
        # failed to improve, test the other eligible branch rather than blindly
        # exhausting reports. Only already-observed action effects are consulted.
        action = 'report_probe' if 'report_probe' in feasible else 'image_probe'
        switched = self.feedback_enabled and self.history and self.history[-1]['action'] == 'report_probe' \
                and not self.history[-1]['accepted'] and 'image_probe' in feasible
        if switched:
            action = 'image_probe'
        request = {'schema_version': VERSION + '-request', 'step': len(self.history),
            'case_id': self.current['case_id'], 'action': action, 'slot': list(feasible[action]),
            'parent_candidate_id': self.current['candidate_id'],
            'parent_observation_sha256': digest(self.current), 'reserved_simulated_calls': COST[action],
            'reason': 'observed_report_probe_failed_try_image' if switched
                else 'cheapest_feasible_targeted_probe', 'uses_unseen_scores': False,
            'uses_report_votes_as_independent_image_truth': False}
        request['request_sha256'] = digest(request)
        self.pending = deepcopy(request)
        self.spent += COST[action]  # Reserve before execution; failure is not free.
        self.attempted.add(tuple(request['slot']))
        return deepcopy(request)

    def complete(self, request, observation=None, *, failure_type=None):
        if self.pending is None or request != self.pending:
            raise ValueError('exact_reserved_request_required')
        if digest(self.current) != request['parent_observation_sha256']:
            raise ValueError('parent_observation_changed_during_probe')
        if observation is None:
            if failure_type not in ('worker_failed', 'verification_failed', 'unavailable_cache_slot'):
                raise ValueError('typed_failure_required_and_still_charged')
            credit, accepted = None, False
        else:
            if failure_type is not None:
                raise ValueError('completed_observation_cannot_also_be_failure')
            validate_snapshot(observation)
            if self._slot(observation) != tuple(request['slot']) or observation['candidate_id'] in self.observed_ids:
                raise ValueError('new_requested_slot_observation_required')
            credit = action_credit(self.current, observation, request['action'])
            self.observed_ids.add(observation['candidate_id'])
            accepted = credit['replacement_allowed_under_proxy_contract']
            if accepted:
                self.current = deepcopy(observation)
        self.history.append({'step': request['step'], 'action': request['action'],
            'request_sha256': request['request_sha256'], 'slot': request['slot'],
            'reserved_simulated_calls': request['reserved_simulated_calls'],
            'failure_type': failure_type, 'credit': credit, 'accepted': accepted,
            'committed_candidate_id': self.current['candidate_id']})
        self.pending = None
        return deepcopy(self.history[-1])

    def result(self):
        if self.pending is not None:
            raise RuntimeError('pending_probe_cannot_be_reported_as_complete')
        return {'schema_version': VERSION, 'case_id': self.current['case_id'],
            'selected_candidate_id': self.current['candidate_id'],
            'selected_observation': deepcopy(self.current), 'model_call_budget': self.budget,
            'simulated_calls': self.spent, 'history': deepcopy(self.history),
            'feedback_enabled': self.feedback_enabled,
            'ehr_sha256': self.initial['lineage']['ehr_sha256'],
            'uses_secondary_endpoint': False, 'actual_regeneration_executed': False,
            'clinical_fault_location': None, 'clinical_repair_success': False,
            'scope': 'observed_action_feedback_development_proxy_replay'}

"""Lossless cached-candidate reliability sidecar, not a selector or repair policy.

Benchmark limitations are metadata, never new candidate labels or fitted weights.
Unknown/uncertain stay unchanged; shared-image reports are correlated evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math

from .invariant_verification import _edge
from .legacy_replay_adapter import FINDINGS, PROFILE, HASH_FIELDS, STATES

SCHEMA = 'tricompose-candidate-scorer-reliability-sidecar-v1'
EDGES = {'ehr_cxr': ('ehr', 'xrv'), 'ehr_report': ('ehr', 'chexbert'),
         'cxr_report': ('xrv', 'chexbert')}
EXPLICIT = {'positive', 'negative'}
REASONS = ('no_explicit_reference_facts', 'no_comparable_proxy_facts',
           'proxy_opposition_unverified', 'proxy_support_with_missing_evidence',
           'proxy_support_only_unverified')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def status(edge):
    # Evidence availability only. Zero contradiction with zero coverage is not good.
    if edge['known_reference_facts'] == 0:
        return REASONS[0]
    if edge['comparable_facts'] == 0:
        return REASONS[1]
    if edge['proxy_opposition_facts']:
        return REASONS[2]
    if edge['missing_comparisons']:
        return REASONS[3]
    return REASONS[4]


def _numeric(value):
    if value in ('', None):
        return None
    if isinstance(value, bool):
        raise ValueError('boolean_is_not_numeric_evidence')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite_cached_evidence')
    return number


def _same(index, key, value):
    if index.setdefault(key, value) != value:
        raise ValueError('shared_artifact_or_fixed_ehr_changed')


def build_overlay(source_rows, bank):
    """Use aligned synthetic cache only. Keep every original cell and row order."""
    if not 1 <= len(source_rows) <= 960:
        raise ValueError('bounded_complete_candidate_inventory_required')
    candidates = {candidate['score_record']['triple_candidate_id']: candidate
                  for grid in bank.values() for candidate in grid.values()}
    if len(candidates) != sum(len(grid) for grid in bank.values()):
        raise ValueError('duplicate_cached_candidate')
    ids = [row['triple_candidate_id'] for row in source_rows]
    if len(set(ids)) != len(ids) or set(ids) != set(candidates):
        raise ValueError('exact_source_candidate_inventory_required')
    if any(any(key.startswith('reliability_') for key in row) for row in source_rows):
        raise ValueError('already_annotated_source_refused')
    if any(row['profile'] != PROFILE for row in source_rows):
        raise ValueError('mixed_or_fresh_profile_refused')
    cases, images, image_hash_states, reports, report_groups = {}, {}, {}, {}, defaultdict(dict)
    fact_rows, by_candidate = [], defaultdict(list)
    for row in source_rows:
        candidate = candidates[row['triple_candidate_id']]
        original = candidate['score_record']
        lineage = original['lineage']
        if (row['case_id'] != original['case_id'] or any(row[key] != lineage[key] for key in HASH_FIELDS)
                or any(row[key] != lineage[key] for key in ('cxr_candidate_id', 'report_candidate_id', 'report_model_id'))):
            raise ValueError('candidate_lineage_or_hash_changed')
        facts = candidate['facts']
        if [fact['finding'] for fact in facts] != list(FINDINGS):
            raise ValueError('ordered_fourteen_fact_inventory_required')
        for fact in facts:
            if (set(fact['states']) != {'ehr', 'xrv', 'chexbert'}
                    or any(value not in STATES for value in fact['states'].values())
                    or fact['weak_context_promoted'] is not False
                    or fact['clinical_truth_verified'] is not False
                    or fact['case_id'] != row['case_id'] or fact['triple_candidate_id'] != row['triple_candidate_id']
                    or any(fact['artifact_hashes'][key] != row[key] for key in HASH_FIELDS)):
                raise ValueError('cached_fact_scope_or_lineage_changed')
            state = dict(fact['states'])
            if state['ehr'] != 'unknown' and not fact['source_categories']:
                raise ValueError('asserted_ehr_fact_lacks_cached_source_category')
            finding = fact['finding']
            key = (row['case_id'], row['cxr_sha256'], finding)
            _same(report_groups[key], row['report_sha256'], state['chexbert'])
            _same(cases, (row['case_id'], finding), (row['ehr_sha256'], row['ehr_facts_sha256'], state['ehr']))
            _same(images, (row['case_id'], row['cxr_candidate_id'], finding), (row['cxr_sha256'], state['xrv']))
            _same(image_hash_states, (row['cxr_sha256'], finding), state['xrv'])
            _same(reports, (row['report_sha256'], finding), state['chexbert'])
            derived = {edge: ('not_comparable' if 'uncertain' in (state[left], state[right]) else
                             'unknown' if 'unknown' in (state[left], state[right]) else
                             'support' if state[left] == state[right] else 'proxy_opposition')
                       for edge, (left, right) in EDGES.items()}
            result = {'schema_version': SCHEMA, 'profile': PROFILE,
                'case_id': row['case_id'], 'triple_candidate_id': row['triple_candidate_id'],
                'cxr_candidate_id': row['cxr_candidate_id'], 'report_candidate_id': row['report_candidate_id'],
                'artifact_hashes': dict(fact['artifact_hashes']), 'finding': finding,
                'source_evidence_id': fact['evidence_id'], 'states': state,
                'cached_source_categories': list(fact['source_categories']),
                'relations': derived, 'report_dependency_group': digest(key),
                'candidate_same_fact_image_scorer_disagreement': None,
                'image_scorer_disagreement_status': 'not_measured_on_this_synthetic_candidate',
                'benchmark_scope': 'rsua_pneumonia_cohort_proxy_diagnostic_only' if finding == 'pneumonia'
                                   else 'not_covered_by_rsua_pneumonia_diagnostic',
                'benchmark_metric_transferred_to_candidate': False,
                'clinical_conflict_verified': None, 'confirmed_faulty_modality': None,
                'clinical_truth_available': False, 'automatic_regeneration_authorized': False}
            fact_rows.append(result)
            by_candidate[row['triple_candidate_id']].append(result)
    groups = []
    group_index = {}
    for key, vector in sorted(report_groups.items()):
        counts = Counter(vector.values())
        group = {'dependency_group_id': digest(key), 'case_id': key[0], 'cxr_sha256': key[1], 'finding': key[2],
            'unique_report_artifacts': len(vector), 'state_counts': {state: counts[state] for state in sorted(STATES)},
            'explicit_positive_negative_disagreement': bool(counts['positive'] and counts['negative']),
            'independent_votes': False, 'clinical_conflict_verified': None, 'confirmed_faulty_modality': None}
        groups.append(group)
        group_index[group['dependency_group_id']] = group
    annotated = []
    edge_counts = {edge: Counter() for edge in EDGES}
    for row in source_rows:
        facts = by_candidate[row['triple_candidate_id']]
        for fact in facts:
            group = group_index[fact['report_dependency_group']]
            fact['same_image_report_expert_proxy_disagreement'] = group['explicit_positive_negative_disagreement']
            fact['correlated_report_artifact_count'] = group['unique_report_artifacts']
        states = [fact['states'] for fact in facts]
        extension = {'reliability_profile_id': SCHEMA,
            'reliability_source_profile': PROFILE,
            'reliability_clinical_verdict': 'unverified',
            'reliability_confirmed_faulty_modality': None,
            'reliability_primary_metric_eligible': False,
            'reliability_automatic_regeneration_authorized': False,
            'reliability_benchmark_metric_transferred': False,
            'reliability_image_scorer_disagreement': None,
            'reliability_image_scorer_disagreement_status': 'not_measured_on_this_synthetic_candidate',
            'reliability_shared_image_report_votes_independent': False,
            'reliability_report_expert_disagreement_findings': sum(f['same_image_report_expert_proxy_disagreement'] for f in facts)}
        for edge, (left, right) in EDGES.items():
            value = _edge(states, left, right)
            for field, expected in value.items():
                actual = _numeric(row[f'{edge}_{field}'])
                if ((expected is None) != (actual is None)
                        or expected is not None and not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)):
                    raise ValueError('raw_edge_readout_differs_from_cached_states')
            label = status(value)
            edge_counts[edge][label] += 1
            extension[f'reliability_{edge}_evidence_status'] = label
            extension[f'reliability_{edge}_clinical_score'] = None
        cosine = _numeric(row['biovil_raw_cosine'])
        if cosine is None:
            if not row['endpoint_unavailable_reason']:
                raise ValueError('missing_endpoint_reason_required')
            extension['reliability_biovil_role'] = 'unavailable_with_source_reason'
        else:
            if abs(cosine) > 1.01 or row['endpoint_unavailable_reason'] not in ('', None):
                raise ValueError('invalid_raw_endpoint')
            extension['reliability_biovil_role'] = 'whole_report_retrieval_secondary_not_fact_negation_or_ehr_fidelity'
        annotated.append({**row, **extension})
    known_cases = {case for (case, _), value in cases.items() if value[2] in EXPLICIT}
    summary = {'schema_version': SCHEMA, 'profile': PROFILE, 'candidate_rows': len(annotated),
        'fixed_ehr_cases': len({key[0] for key in cases}),
        'image_slots': len({key[:2] for key in images}),
        'unique_report_artifacts': len({key[0] for key in reports}),
        'fact_rows': len(fact_rows), 'image_finding_dependency_groups': len(groups),
        'fixed_ehr_cases_with_explicit_cached_facts': len(known_cases),
        'fixed_ehr_cases_without_explicit_cached_facts': len({key[0] for key in cases} - known_cases),
        'same_image_report_expert_proxy_disagreement_groups': sum(g['explicit_positive_negative_disagreement'] for g in groups),
        'candidate_rows_with_report_expert_proxy_disagreement': sum(bool(row['reliability_report_expert_disagreement_findings']) for row in annotated),
        'candidate_same_fact_image_scorer_disagreements_measured': 0,
        'edge_evidence_status_candidate_counts': {edge: {reason: counts[reason] for reason in REASONS} for edge, counts in edge_counts.items()},
        'original_cells_preserved': all(all(out[key] == value for key, value in original.items())
                                       for original, out in zip(source_rows, annotated)),
        'original_row_order_preserved': ids == [row['triple_candidate_id'] for row in annotated],
        'cases_dropped': 0, 'unknown_or_uncertain_states_changed': 0,
        'original_selection_changed': False, 'thresholds_or_weights_changed': False,
        'new_model_calls': 0, 'primary_metric_eligible': False, 'regeneration_authorized': False,
        'clinical_fault_localization_accuracy': None,
        'interpretation': 'Lossless reliability metadata, not rescoring. Same-image report disagreement is cached CheXbert proposal disagreement, not independent votes or confirmed clinical error. RSUA is not candidate-level evidence.'}
    return annotated, fact_rows, groups, summary

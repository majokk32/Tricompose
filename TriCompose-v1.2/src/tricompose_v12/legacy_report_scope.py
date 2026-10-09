"""Source-bound scope checks on a fixed small historical synthetic cohort.

Reuses the frozen four-finding literal gate without expanding its vocabulary.
All fourteen raw finding states remain immutable in a separate diagnostic.
"""
from __future__ import annotations

from collections import Counter

from .reliability_preview import validate_sidecar
from .scorer_reliability import FINDINGS, PROFILE, HASH_FIELDS
from .report_assertions import FINDINGS as SCOPE_FINDINGS, GATE_VERSION, digest, gate_assertion
from .report_scope_table import EDGES, edge_summary, relation

SCHEMA = 'tricompose-legacy-report-scope-smoke-v1'
SELECTION_RULE = 'first_two_sorted_opaque_case_ids_all_three_images_all_four_reports'


def select_cases(rows, facts, groups, count=2):
    validate_sidecar(rows, facts, groups)
    if type(count) is not int or not 1 <= count <= 2:
        raise ValueError('fixed_small_case_count_required')
    cases = sorted({row['case_id'] for row in rows})
    if len(cases) < count:
        raise ValueError('insufficient_fixed_cases')
    chosen = set(cases[:count])
    return ([row for row in rows if row['case_id'] in chosen],
            [fact for fact in facts if fact['case_id'] in chosen],
            [group for group in groups if group['case_id'] in chosen])


def build_scope_audit(rows, facts, groups, texts, scope_checker, *, text_unavailable=None):
    """Text is supplied only from verified synthetic artifacts, never real gold.

Unavailable text gets an explicit reason and retains the complete denominator.
Evidence offsets/hashes are saved without copying the report quote into output.
"""
    if len(rows) > 24 or len({row['case_id'] for row in rows}) > 2:
        raise ValueError('small_fixed_scope_smoke_required')
    by_candidate = validate_sidecar(rows, facts, groups)
    report_hashes = {row['report_sha256'] for row in rows}
    unavailable = {} if text_unavailable is None else dict(text_unavailable)
    if set(texts) | set(unavailable) != report_hashes or set(texts) & set(unavailable):
        raise ValueError('complete_disjoint_text_availability_inventory_required')
    for h, text in texts.items():
        if not isinstance(text, str) or not 1 <= len(text) <= 8192 or digest(text) != h:
            raise ValueError('bounded_source_text_hash_required')
    allowed = {'empty_report', 'overlength_report_bytes', 'overlength_report_characters', 'invalid_utf8'}
    if any(reason not in allowed for reason in unavailable.values()):
        raise ValueError('explicit_bounded_text_unavailability_reason_required')
    assertions, scoped_facts, candidate_rows = {}, [], []
    for row in rows:
        cid, h = row['triple_candidate_id'], row['report_sha256']
        candidate_facts = sorted(by_candidate[cid], key=lambda f: FINDINGS.index(f['finding']))
        for original in candidate_facts:
            finding, proposal = original['finding'], original['states']['chexbert']
            supported = finding in SCOPE_FINDINGS
            key = (h, finding)
            if not supported:
                checked = {'state': 'unknown', 'decision': 'outside_scope_inventory',
                    'reason': 'frozen_four_finding_guard_has_no_head', 'scope_verified': False,
                    'evidence_offsets_and_hashes': [], 'scope_check': None}
            elif h in unavailable:
                checked = {'state': 'unknown', 'decision': 'source_text_unavailable',
                    'reason': unavailable[h], 'scope_verified': False,
                    'evidence_offsets_and_hashes': [], 'scope_check': None}
            else:
                if key not in assertions:
                    result = gate_assertion(texts[h], finding, proposal, scope_checker)
                    assertions[key] = {'state': result['state'], 'decision': result['decision'],
                        'reason': result['reason'], 'scope_verified': result['scope_verified'],
                        'evidence_offsets_and_hashes': [{k: span[k] for k in
                            ('char_start', 'char_end', 'quote_sha256', 'offset_unit', 'evidence_id')}
                            for span in result['evidence']], 'scope_check': result.get('scope_check')}
                checked = assertions[key]
            if checked['state'] not in {proposal, 'unknown'}:
                raise ValueError('scope_cannot_flip_or_invent_a_label')
            available = checked['decision'] in {'scope_commit', 'no_model_assertion'}
            states = dict(original['states'])
            raw = {'ehr_cxr': relation(states['ehr'], states['xrv']),
                   'ehr_report': relation(states['ehr'], states['chexbert']),
                   'cxr_report': relation(states['xrv'], states['chexbert'])}
            scoped = {'ehr_cxr': raw['ehr_cxr'],
                'ehr_report': relation(states['ehr'], checked['state'], available=available),
                'cxr_report': relation(states['xrv'], checked['state'], available=available)}
            scoped_facts.append({'schema_version': SCHEMA, 'profile': PROFILE,
                'case_id': row['case_id'], 'triple_candidate_id': cid,
                'cxr_candidate_id': row['cxr_candidate_id'], 'report_candidate_id': row['report_candidate_id'],
                'report_model_id': row['report_model_id'], 'source_evidence_id': original['source_evidence_id'],
                'finding': finding, 'artifact_hashes': {name: row[name] for name in HASH_FIELDS},
                'states': states, 'cached_source_categories': list(original['cached_source_categories']),
                'scope_report_state': checked['state'], 'scope_supported_finding': supported,
                'scope_available': available, 'scope_decision': checked['decision'], 'scope_reason': checked['reason'],
                'scope_verified': checked['scope_verified'], 'gate_version': GATE_VERSION,
                'scope_check': checked['scope_check'], 'evidence_offsets_and_hashes': checked['evidence_offsets_and_hashes'],
                'relations': {'raw': raw, 'scoped': scoped}, 'report_quote_copied_to_output': False,
                'independent_clinical_validation': False, 'confirmed_faulty_modality': None,
                'regeneration_authorized': False, 'selection_changed': False})
        selected = scoped_facts[-len(FINDINGS):]
        candidate_rows.append({'case_id': row['case_id'], 'triple_candidate_id': cid,
            'report_model_id': row['report_model_id'], 'artifact_hashes': {name: row[name] for name in HASH_FIELDS},
            'finding_inventory': len(selected),
            'scope_decision_counts': dict(Counter(f['scope_decision'] for f in selected)),
            'edges': {stage: {edge: edge_summary(selected, edge, stage) for edge in EDGES}
                      for stage in ('raw', 'scoped')},
            'clinical_selection_score': None, 'confirmed_faulty_modality': None,
            'regeneration_authorized': False, 'selection_changed': False})
    image_rows = {}
    for fact in scoped_facts:
        image_rows.setdefault((fact['case_id'], fact['artifact_hashes']['cxr_sha256'], fact['finding']), fact)
    source_groups = {}
    for fact in scoped_facts:
        key = (fact['case_id'], fact['artifact_hashes']['cxr_sha256'], fact['finding'])
        source_groups.setdefault(key, {})[fact['artifact_hashes']['report_sha256']] = fact
    group_rows = []
    for (case, image, finding), reports in sorted(source_groups.items()):
        raw = Counter(f['states']['chexbert'] for f in reports.values())
        scoped = Counter(f['scope_report_state'] for f in reports.values())
        group_rows.append({'case_id': case, 'cxr_sha256': image, 'finding': finding,
            'unique_report_artifacts': len(reports), 'raw_state_counts': dict(raw), 'scoped_state_counts': dict(scoped),
            'raw_positive_negative_disagreement': bool(raw['positive'] and raw['negative']),
            'scoped_positive_negative_disagreement': bool(scoped['positive'] and scoped['negative']),
            'independent_votes': False, 'clinical_conflict_verified': None})
    summary = {'schema_version': SCHEMA, 'profile': PROFILE, 'candidate_rows': len(rows),
        'fixed_ehr_cases': len({r['case_id'] for r in rows}), 'image_slots': len(image_rows)//len(FINDINGS),
        'fact_rows': len(scoped_facts), 'finding_inventory': list(FINDINGS), 'scope_finding_inventory': list(SCOPE_FINDINGS),
        'scope_decision_counts_candidate_facts': dict(sorted(Counter(f['scope_decision'] for f in scoped_facts).items())),
        'per_supported_finding_decisions': {name: dict(sorted(Counter(f['scope_decision'] for f in scoped_facts
            if f['finding'] == name).items())) for name in SCOPE_FINDINGS},
        'unique_report_artifacts': len(report_hashes), 'verified_text_artifacts': len(texts),
        'unavailable_text_artifacts': len(unavailable), 'text_unavailability_reasons': dict(Counter(unavailable.values())),
        'relations': {stage: {edge: edge_summary(list(image_rows.values()) if edge == 'ehr_cxr' else scoped_facts, edge, stage)
            for edge in EDGES} for stage in ('raw', 'scoped')},
        'withdrawn_proxy_comparisons': {edge: {decision: sum(f['relations']['raw'][edge] in {'support', 'opposition'}
                and f['relations']['scoped'][edge] not in {'support', 'opposition'} and f['scope_decision'] == decision
                for f in scoped_facts) for decision in ('abstain', 'outside_scope_inventory', 'source_text_unavailable')}
                for edge in ('ehr_report', 'cxr_report')},
        'cross_report_dependency_groups': len(group_rows),
        'raw_report_disagreement_groups': sum(g['raw_positive_negative_disagreement'] for g in group_rows),
        'scoped_report_disagreement_groups': sum(g['scoped_positive_negative_disagreement'] for g in group_rows),
        'new_model_calls': 0, 'new_generation_samples': 0, 'fixed_ehr_changed': False,
        'selection_changed': False, 'regeneration_authorized': False, 'primary_metric_eligible': False,
        'clinical_accuracy': None, 'clinical_localization_accuracy': None,
        'report_scope_is_clinical_truth': False, 'report_quote_copied_to_output': False,
        'interpretation': 'Frozen limited report-text scope only. Withdrawn opposition/support is abstention or missing heads, not improved generated artifacts. All raw states and fourteen-finding denominators retained.'}
    return scoped_facts, candidate_rows, group_rows, summary

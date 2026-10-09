"""Literal opacity annotation inventory, not current-patient lung-opacity truth.

Human observation spans supply polarity; literal mentions select the vocabulary.
No synonyms, generic-normal expansion, scope inference or model-derived gold.
Unannotated mentions stay unknown. Qualifier/anatomy/temporal scope is unverified,
so even a negative literal annotation cannot authorize a global image negative.
"""
from collections import Counter
import hashlib
import re

from .entity_gold_contract import STATES
from .radgraph_reference_contract import require

VERSION = 'manual-opacity-literal-inventory-v1'
PATTERN = r'\b(?P<head>opacit(?:y|ies))\b'
ASSERTIONS = ('positive', 'negative', 'uncertain', 'unknown')


def project(text, entities, source_sha256):
    require(isinstance(text, str) and 0 < len(text) <= 100000 and text == ' '.join(text.split()),
            'bounded_canonical_literal_source_required')
    require(hashlib.sha256(text.encode('utf-8')).hexdigest() == source_sha256, 'literal_source_hash_binding_required')
    require(isinstance(entities, list) and len(entities) <= 4096, 'bounded_character_annotations_required')
    spans = set()
    for e in entities:
        require(isinstance(e, (tuple, list)) and len(e) == 3 and all(type(n) is int for n in e[:2])
                and 0 <= e[0] < e[1] <= len(text) and e[2] in (*STATES, 'Anatomy::definitely present')
                and tuple(e[:2]) not in spans, 'unique_supported_exact_character_spans_required')
        spans.add(tuple(e[:2]))
    heads, evidence = [], []
    for m in re.finditer(PATTERN, text, flags=re.IGNORECASE):
        left, right = m.span('head')
        matches = []
        for start, end, label in entities:
            if label in STATES and start <= left and right <= end:
                ev = {'head_start': left, 'head_end_exclusive': right, 'char_start': start,
                      'char_end_exclusive': end, 'human_native_state': STATES[label]}
                matches.append(ev)
                evidence.append(ev)
        heads.append({'head_start': left, 'head_end_exclusive': right,
                      'annotated_observation_count': len(matches)})
    values = {e['human_native_state'] for e in evidence}
    state = 'unknown' if not values else 'uncertain' if 'uncertain' in values or len(values) > 1 else next(iter(values))
    return {'schema_version': VERSION + '-projection', 'source_sha256': source_sha256,
        'manual_literal_state': state, 'literal_mentions': heads, 'human_span_evidence': evidence,
        'literal_mention_count': len(heads),
        'unannotated_literal_mentions': sum(h['annotated_observation_count'] == 0 for h in heads),
        'unknown_means': 'no_annotated_literal_opacity_observation_not_global_clinical_absence',
        'current_lung_opacity_reference': None, 'current_patient_scope_verified': False,
        'qualifier_scope_verified': False, 'pulmonary_anatomy_scope_verified': False,
        'generic_normality_expanded': False, 'synonyms_added': False,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False}


def summarize(records, *, previous_test_hashes):
    require(isinstance(records, list) and 0 < len(records) <= 1024 and
            len({r['report_id'] for r in records}) == len(records), 'complete_unique_fixed_release_inventory_required')
    require(isinstance(previous_test_hashes, set) and all(re.fullmatch(r'[a-f0-9]{64}', h) for h in previous_test_hashes),
            'prior_test_artifact_hash_inventory_required')
    complete, states, failures, hashes = [], Counter(), Counter(), Counter()
    for i, r in enumerate(records):
        require(r['report_id'] == f'report_{i:04d}' and r['source_index'] == i,
                'release_order_no_easiness_selection_required')
        require(r['status'] in ('complete', 'failed_unavailable'), 'explicit_attempt_outcome_required')
        h = r['source_sha256']
        require(h is None or isinstance(h, str) and re.fullmatch(r'[a-f0-9]{64}', h), 'opaque_source_hash_only_required')
        if h:
            hashes[h] += 1
        if r['status'] == 'complete':
            p = r['projection']
            require(p['source_sha256'] == h and p['manual_literal_state'] in ASSERTIONS and
                    p['current_lung_opacity_reference'] is None and p['clinical_qualified'] is False,
                    'narrow_language_reference_not_image_truth_required')
            complete.append(p)
            states[p['manual_literal_state']] += 1
        else:
            require(r['projection'] is None and r['failure_type'] is not None,
                    'failed_record_cannot_be_unknown_or_negative_reference')
            failures[r['failure_type']] += 1
    return {'schema_version': VERSION + '-summary', 'attempted_reports': len(records),
        'complete_reports': len(complete), 'failed_reports': len(records) - len(complete),
        'failure_type_counts': dict(failures),
        'manual_literal_state_counts': {s: states[s] for s in ASSERTIONS},
        'reports_with_literal_mentions': sum(p['literal_mention_count'] > 0 for p in complete),
        'reports_without_literal_mentions': sum(p['literal_mention_count'] == 0 for p in complete),
        'reports_with_unannotated_literal_mentions': sum(p['unannotated_literal_mentions'] > 0 for p in complete),
        'literal_mentions': sum(p['literal_mention_count'] for p in complete),
        'human_span_evidence_bindings': sum(len(p['human_span_evidence']) for p in complete),
        'duplicate_source_hash_slots_retained': sum(n - 1 for n in hashes.values()),
        'prior_test_source_hash_overlap_slots': sum(n for h, n in hashes.items() if h in previous_test_hashes),
        'unique_source_artifacts': len(hashes), 'reference_states_available': [s for s in ASSERTIONS if states[s]],
        'case_count_or_balanced_classes_preassumed': False,
        'no_annotated_mentions_is_clinical_negative': False, 'patient_disjointness_verified': False,
        'checkpoint_training_overlap_verified': False, 'untouched_final_test': False,
        'literal_annotation_reference_only': True, 'current_lung_opacity_reference_available': False,
        'clinical_qualified': False, 'new_model_calls': 0, 'new_training': False,
        'selection_changed': False, 'regeneration_authorized': False}

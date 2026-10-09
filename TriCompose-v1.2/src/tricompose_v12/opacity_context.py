"""Rule-parser evidence contract, not clinical truth or a new learned scorer.

Official ConText context rules are left unchanged. A disclosed two-word-form
literal target inventory is not a radiological ontology or synonym recognizer.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from .report_assertions import CONTEXT_FLAGS, context_state, aggregate_context_mentions

VERSION = 'tricompose-official-context-opacity-v1'
TARGET_PATTERN = r'\bopacit(?:y|ies)\b'
STATES = ('positive', 'negative', 'uncertain', 'unknown')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def targets(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 8192:
        raise ValueError('bounded_nonempty_authored_source_required')
    return [span(text, m.start(), m.end()) for m in re.finditer(TARGET_PATTERN, text, re.I)]


def span(text, start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError('exact_unicode_span_required')
    return {'char_start': start, 'char_end': end, 'quote_sha256': digest(text[start:end]),
            'offset_unit': 'unicode_codepoint'}


def aggregate(mentions):
    # Reuse the old, already consumed noncurrent/uncertainty mapping unchanged.
    return aggregate_context_mentions(mentions)


def serialize(doc, text):
    if doc.text != text:
        raise ValueError('unchanged_source_required')
    expected = targets(text)
    actual = [span(text, e.start_char, e.end_char) for e in doc.ents]
    if actual != expected or any(e.label_ != 'lung_opacity' for e in doc.ents):
        raise ValueError('exact_literal_entity_inventory_required')
    mentions = []
    for entity, source_span in zip(doc.ents, expected):
        flags = {name: getattr(entity._, name) for name in CONTEXT_FLAGS}
        modifiers = []
        for modifier in entity._.modifiers:
            start, end = modifier.modifier_span
            left, right = modifier.scope_span
            cue, scope = doc[start:end], doc[left:right]
            modifiers.append({'category': modifier.category, 'direction': modifier.direction,
                'cue': span(text, cue.start_char, cue.end_char),
                'scope': span(text, scope.start_char, scope.end_char)})
        mentions.append({'span': source_span, 'flags': flags, 'context_state': context_state(flags),
            'modifiers': modifiers, 'unmodified_positive_is_semantically_unverified': not modifiers})
    return {'status': 'complete', 'state': aggregate(mentions), 'failure_reason': None,
        'literal_target_mentions': len(mentions), 'mentions': mentions,
        'semantic_scope_verified': False, 'independent_clinical_validation': False,
        'hard_action_eligible': False, 'regeneration_authorized': False,
        'target_inventory_is_not_domain_disambiguation': True}


def veto(proposal, context):
    """May withhold signed proposals, never replace/flip/promote their state."""
    for row in (proposal, context):
        if row.get('status') not in ('complete', 'failed_unavailable'):
            raise ValueError('explicit_status_required')
        if (row['status'] == 'complete' and row.get('state') not in STATES) or \
                (row['status'] != 'complete' and row.get('state') is not None):
            raise ValueError('four_state_or_failed_null_required')
    state = proposal['state']
    reason = ('proposal_unavailable' if proposal['status'] != 'complete' else
              'proposal_unknown_noncomparable' if state == 'unknown' else
              'proposal_uncertain_noncomparable' if state == 'uncertain' else
              'context_unavailable' if context['status'] != 'complete' else
              'context_no_literal_target' if context.get('literal_target_mentions') == 0 else
              'context_state_disagreement' if context['state'] != state else
              'context_agreement_soft_only')
    retained = reason == 'context_agreement_soft_only'
    return {'raw_status': proposal['status'], 'raw_state': state,
        'context_status': context['status'], 'context_state': context['state'],
        'decision': reason, 'soft_retained_state': state if retained else None,
        'soft_comparable': retained, 'state_corrected': False, 'candidate_dropped': False,
        'hard_action_eligible': False, 'clinical_score': None,
        'independent_clinical_validation': False, 'parsers_are_independent_clinical_votes': False,
        'selection_changed': False, 'regeneration_authorized': False}


def metrics(checks):
    matrix = {s: dict.fromkeys((*STATES, 'unavailable'), 0) for s in STATES}
    for row in checks:
        if row['expected_state'] not in STATES:
            raise ValueError('four_state_authored_reference_required')
        if row['status'] not in ('complete', 'failed_unavailable') or \
                (row['status'] != 'complete' and row['state'] is not None):
            raise ValueError('explicit_availability_required')
        matrix[row['expected_state']][row['state'] if row['status'] == 'complete' else 'unavailable'] += 1
    f1 = {}
    for s in STATES:
        support = sum(matrix[s].values())
        predicted = sum(matrix[t][s] for t in STATES)
        denominator = support + predicted
        f1[s] = 2 * matrix[s][s] / denominator if denominator else None
    return {'rows': len(checks), 'complete': sum(r['status'] == 'complete' for r in checks),
        'unavailable': sum(r['status'] != 'complete' for r in checks),
        'exact_matches': sum(r['status'] == 'complete' and r['state'] == r['expected_state'] for r in checks),
        'confusion': matrix, 'per_class_f1': f1,
        'macro_f1_present_classes': sum(v for v in f1.values() if v is not None) / sum(v is not None for v in f1.values())
            if any(v is not None for v in f1.values()) else None,
        'determinate_on_uncertain_unknown': sum(r['status'] == 'complete' and
            r['state'] in ('positive', 'negative') and r['expected_state'] in ('uncertain', 'unknown') for r in checks),
        'hard_positive_negative_flips': sum(r['status'] == 'complete' and
            {r['state'], r['expected_state']} == {'positive', 'negative'} for r in checks),
        'reference_state_counts': dict(sorted(Counter(r['expected_state'] for r in checks).items())),
        'clinical_accuracy': None, 'primary_metric_eligible': False}

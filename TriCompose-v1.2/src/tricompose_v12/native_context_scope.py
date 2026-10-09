"""Source-bound native/ConText eligibility, veto only, not clinical truth.

All official categories are represented; No Finding is aggregate-only. No new
language/anatomy/qualifier rules are fitted. Same-report agreement is soft and
correlated. Unflagged context is not proof of current patient lung pathology.
"""
from __future__ import annotations

from collections import Counter
import re
import string

from .chexpert_negbio_contract import FINDINGS, STATES, digest, span
from .report_assertions import CONTEXT_FLAGS, context_state

VERSION = 'tricompose-native-context-scope-v1'
GROUP = 'tricompose_native_mentions'
SIGNED = frozenset(('positive', 'negative'))
POLICY = {'version': VERSION, 'official_context_rules_changed': False,
    'same_exact_cleaned_source_required': True, 'same_native_mention_inventory_required': True,
    'native_labels_overwritten': False, 'label_correction': False,
    'any_noncurrent_mention_vetoes_signed_comparison': True,
    'all_context_mentions_must_agree_with_native_sign': True,
    'anatomy_verified': False, 'qualified_absence_verified': False,
    'no_finding_is_aggregate_only': True, 'same_report_parsers_are_independent_votes': False,
    'hard_action_eligible': False, 'clinical_qualified': False, 'primary_metric_eligible': False,
    'selection_changed': False, 'regeneration_authorized': False}


def cleaned_full_report(text):
    """Literal mirror of pinned Loader.clean + text2document printable handling.

    This is source normalization only; hash equality to native output is required
    before any context annotation. Never silently accept altered source text.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 8192:
        raise ValueError('bounded_authored_source_required')
    text = re.sub('and/or', 'or', text.lower())
    text = re.sub(r'(?<=[a-zA-Z])/(?=[a-zA-Z])', ' or ', text)
    text = text.replace('..', '.').translate(str.maketrans({key: key+' ' for key in '.,'}))
    text = re.sub(r'\.\s+\.', '.', ' '.join(text.split()))
    return ''.join(c for c in text if c in string.printable).replace('\r\n', '\n')


def check_native(text, native):
    cleaned = cleaned_full_report(text)
    if native.get('status') != 'complete' or native.get('original_text_sha256') != digest(text) or \
            native.get('cleaned_text_sha256') != digest(cleaned) or set(native['native_labels']) != set(FINDINGS):
        raise ValueError('complete_same_source_native_output_required')
    seen = set()
    for mention in native['mentions']:
        location = mention['span']
        if mention['annotation_id'] in seen or mention['finding'] not in FINDINGS or \
                location != span(cleaned, location['char_start'], location['char_end']-location['char_start']):
            raise ValueError('exact_source_native_inventory_required')
        seen.add(mention['annotation_id'])
    return cleaned


def target_key(mention):
    return (mention['span']['char_start'], mention['span']['char_end'], mention['finding'])


def context_unavailable(reason):
    if reason not in ('native_unavailable', 'source_or_token_alignment_failed', 'context_parser_failed'):
        raise ValueError('sanitized_failure_reason_required')
    return {'schema_version': VERSION, 'status': 'failed_unavailable', 'failure_reason': reason,
            'mentions': None, 'clinical_qualified': False, 'hard_action_eligible': False}


def serialize_context(doc, cleaned, native):
    if doc.text != cleaned or digest(cleaned) != native['cleaned_text_sha256']:
        raise ValueError('same_cleaned_source_required')
    targets = {(entity.start_char, entity.end_char, entity.label_): entity for entity in doc.spans[GROUP]}
    expected = {target_key(mention) for mention in native['mentions']}
    if set(targets) != expected or len(targets) != len(doc.spans[GROUP]):
        raise ValueError('all_unique_native_mentions_required')
    mentions = []
    for original in native['mentions']:
        entity = targets[target_key(original)]
        flags = {key: getattr(entity._, key) for key in CONTEXT_FLAGS}
        state = context_state(flags)
        modifiers = []
        for modifier in entity._.modifiers:
            start, end = modifier.modifier_span
            left, right = modifier.scope_span
            cue, scope = doc[start:end], doc[left:right]
            modifiers.append({'category': modifier.category, 'direction': modifier.direction,
                'cue': span(cleaned, cue.start_char, cue.end_char-cue.start_char),
                'scope': span(cleaned, scope.start_char, scope.end_char-scope.start_char)})
        mentions.append({'annotation_id': original['annotation_id'], 'finding': original['finding'],
            'span': dict(original['span']), 'flags': flags, 'context_state': state, 'modifiers': modifiers,
            'temporal_scope': ('historical_flagged' if flags['is_historical'] else
                'hypothetical_flagged' if flags['is_hypothetical'] else 'not_flagged'),
            'experiencer_scope': 'family_flagged' if flags['is_family'] else 'not_flagged',
            'anatomy_scope': 'unverified', 'qualified_absence_scope': 'unverified',
            'unflagged_is_not_verified_current_patient_finding': True})
    return {'schema_version': VERSION, 'status': 'complete', 'failure_reason': None,
        'original_text_sha256': native['original_text_sha256'], 'cleaned_text_sha256': digest(cleaned),
        'source_mode': 'official_cleaned_full_report', 'mentions': mentions,
        'target_inventory_is_native_not_literal_custom_head': True,
        'clinical_qualified': False, 'hard_action_eligible': False}


def parse_context(nlp, text, native):
    if native.get('status') != 'complete':
        return context_unavailable('native_unavailable')
    try:
        cleaned = check_native(text, native)
        doc = nlp.make_doc(cleaned)
        targets = []
        for left, right, finding in sorted({target_key(mention) for mention in native['mentions']}):
            entity = doc.char_span(left, right, label=finding, alignment_mode='strict')
            if entity is None:
                raise ValueError('strict_native_span_token_alignment_required')
            targets.append(entity)
        # Official group API supports overlap; no phrase is dropped or expanded.
        doc.spans[GROUP] = targets
    except Exception:
        return context_unavailable('source_or_token_alignment_failed')
    try:
        return serialize_context(nlp(doc), cleaned, native)
    except Exception:
        return context_unavailable('context_parser_failed')


def gate(native, context):
    if native.get('status') not in ('complete', 'failed_unavailable') or \
            context.get('status') not in ('complete', 'failed_unavailable'):
        raise ValueError('explicit_native_and_context_availability_required')
    if context['status'] == 'complete':
        if native['status'] != 'complete' or any(context.get(key) != native.get(key) for key in
                ('original_text_sha256', 'cleaned_text_sha256')):
            raise ValueError('same_source_context_required')
        signature = lambda row: (row['annotation_id'], row['finding'], row['span'])
        if [signature(row) for row in native['mentions']] != [signature(row) for row in context['mentions']]:
            raise ValueError('same_exact_mention_inventory_required')
        if any(context_state(row['flags']) != row['context_state'] for row in context['mentions']):
            raise ValueError('unchanged_context_flag_mapping_required')
    results = {}
    for finding in FINDINGS:
        state = native['native_labels'][finding]['state'] if native['status'] == 'complete' else None
        if native['status'] == 'complete' and state not in STATES:
            raise ValueError('native_four_state_required')
        rows = [row for row in context['mentions'] if row['finding'] == finding] if context['status'] == 'complete' else []
        reason = ('native_unavailable' if native['status'] != 'complete' else
            'aggregate_only_no_finding' if finding == 'no_finding' else
            'native_unknown_noncomparable' if state == 'unknown' else
            'native_uncertain_noncomparable' if state == 'uncertain' else
            'context_unavailable' if context['status'] != 'complete' else
            'native_evidence_unavailable' if not rows else
            'native_mention_conflict' if native['mention_conflict_view'][finding]['state'] != state else
            'noncurrent_context_flagged' if any(row['flags'][key] for row in rows
                for key in ('is_historical', 'is_hypothetical', 'is_family')) else
            'context_uncertainty_flagged' if any(row['flags']['is_uncertain'] for row in rows) else
            'native_context_polarity_disagreement' if any(row['context_state'] != state for row in rows) else
            'same_source_agreement_soft_only')
        retained = reason == 'same_source_agreement_soft_only'
        results[finding] = {'raw_status': native['status'], 'raw_state': state,
            'context_status': context['status'], 'decision': reason, 'soft_retained_state': state if retained else None,
            'soft_comparable': retained, 'context_mentions': len(rows),
            'evidence_annotation_ids': [row['annotation_id'] for row in rows],
            'state_corrected': False, 'candidate_dropped': False,
            'semantic_scope_verified': False, 'anatomy_verified': False,
            'qualified_absence_verified': False, 'independent_clinical_validation': False,
            'same_report_parsers_are_independent_votes': False, 'clinical_score': None,
            'hard_action_eligible': False, 'selection_changed': False, 'regeneration_authorized': False}
    return {'schema_version': VERSION, 'policy': POLICY, 'findings': results,
            'decision_counts': dict(sorted(Counter(row['decision'] for row in results.values()).items())),
            'soft_retained_findings': sum(row['soft_comparable'] for row in results.values()),
            'hard_action_eligible_findings': 0, 'clinical_qualified': False,
            'selection_changed': False, 'regeneration_authorized': False}

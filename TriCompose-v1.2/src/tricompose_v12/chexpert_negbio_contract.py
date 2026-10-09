"""Output contract only: no report labeling, parser initialization or downloads.

Native CheXpert aggregation and a TriCompose mention-conflict view are separate.
Neither is clinically qualified by serialization/component tests. Offsets belong
to the official cleaned, full-report passage, never implicitly to original text.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

VERSION = 'tricompose-chexpert-negbio-output-contract-v1'
UPSTREAM = {
    'chexpert_labeler': '44ddeb363149aa657296237f18b5472a73c1756f',
    'negbio': '073199e2792824740e89844a59c13d3d40ce4d23',
}
CATEGORIES = (
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
    'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices',
)
FINDINGS = tuple(name.lower().replace(' ', '_') for name in CATEGORIES)
STATES = ('positive', 'negative', 'uncertain', 'unknown')
FAILURES = frozenset((
    'dependencies_unavailable', 'empty_cleaned_source', 'invalid_native_output',
    'parse_tree_unavailable', 'dependency_graph_unavailable', 'detector_error',
    'evidence_alignment_failed', 'unsupported_section_configuration',
))


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def bounded_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 32768:
        raise ValueError('bounded_nonempty_text_required')
    return text


def native_value(value):
    """Official NaN means unmentioned, not failed processing; reject None/bool."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('official_numeric_label_or_nan_required')
    if math.isnan(value):
        return {'value': None, 'state': 'unknown'}
    if not math.isfinite(value) or value not in (-1, 0, 1):
        raise ValueError('official_numeric_label_or_nan_required')
    return {'value': int(value), 'state': {-1: 'uncertain', 0: 'negative', 1: 'positive'}[value]}


def native_vector(values):
    values = list(values)
    if len(values) != len(CATEGORIES):
        raise ValueError('exact_official_fourteen_category_order_required')
    return {name: native_value(value) for name, value in zip(FINDINGS, values)}


def mention_state(infons):
    # Exact upstream key-presence semantics, including negation precedence.
    # A string 'False' still counts as a present key in official Aggregator.
    if 'negation' in infons:
        return 'negative'
    if 'uncertainty' in infons:
        return 'uncertain'
    return 'positive'


def conflict_view(states):
    states = list(states)
    if any(state not in ('positive', 'negative', 'uncertain') for state in states):
        raise ValueError('only_classified_native_mentions_required')
    present = set(states)
    opposed = {'positive', 'negative'} <= present
    result = ('uncertain' if opposed or 'uncertain' in present else
              next(iter(present)) if present else 'unknown')
    return {'state': result, 'opposing_native_mentions': opposed,
            'mention_count': len(states), 'independent_votes': False,
            'current_scope_or_anatomy_verified': False}


def span(text, offset, length):
    if type(offset) is not int or type(length) is not int or length <= 0 or \
            not 0 <= offset < offset + length <= len(text):
        raise ValueError('exact_cleaned_unicode_span_required')
    return {'char_start': offset, 'char_end': offset + length,
            'quote_sha256': digest(text[offset:offset + length]),
            'offset_unit': 'unicode_codepoint', 'offset_space': 'official_cleaned_full_report'}


def annotation_inventory(passage):
    rows, seen = [], set()
    for annotation in passage.annotations:
        category = annotation.infons.get('observation')
        if category not in CATEGORIES or len(annotation.locations) != 1 or \
                not isinstance(annotation.id, str) or not annotation.id.isdigit() or annotation.id in seen:
            raise ValueError('exact_native_annotation_inventory_required')
        seen.add(annotation.id)
        loc = annotation.locations[0]
        evidence = span(passage.text, loc.offset, loc.length)
        if passage.text[loc.offset:loc.offset + loc.length] != annotation.text:
            raise ValueError('evidence_alignment_failed')
        rows.append({'annotation_id': annotation.id, 'category': category, 'span': evidence})
    return digest(json.dumps(rows, sort_keys=True, separators=(',', ':')))


def syntax_receipt(document):
    """Check native intermediates BEFORE Classifier deletes its sentences.

    This validates availability, not correctness of the frozen parser's syntax.
    It does not invoke or implement NegBio detection. A separate worker must
    capture errors that official negdetect.detect can swallow, and refuse them.
    """
    if len(document.passages) != 1 or document.passages[0].offset != 0:
        raise ValueError('unsupported_section_configuration')
    passage = document.passages[0]
    bounded_text(passage.text)
    if not passage.sentences:
        raise ValueError('parse_tree_unavailable')
    sentence_rows = []
    ids_and_bounds = []
    for sentence in passage.sentences:
        if not isinstance(sentence.infons.get('parse tree'), str) or \
                not sentence.infons['parse tree'].strip():
            raise ValueError('parse_tree_unavailable')
        evidence = span(passage.text, sentence.offset, len(sentence.text))
        if passage.text[sentence.offset:sentence.offset + len(sentence.text)] != sentence.text:
            raise ValueError('evidence_alignment_failed')
        if not sentence.annotations:
            raise ValueError('dependency_graph_unavailable')
        node_ids = set()
        node_bounds = []
        for annotation in sentence.annotations:
            if annotation.id in node_ids or not isinstance(annotation.id, str) or not annotation.id:
                raise ValueError('dependency_graph_unavailable')
            node_ids.add(annotation.id)
            if any(not isinstance(annotation.infons.get(key), str) or not annotation.infons[key]
                   for key in ('tag', 'lemma')) or len(annotation.locations) != 1:
                raise ValueError('dependency_graph_unavailable')
            location = annotation.locations[0]
            bound = span(passage.text, location.offset, location.length)
            if not evidence['char_start'] <= bound['char_start'] < bound['char_end'] <= evidence['char_end'] or \
                    passage.text[bound['char_start']:bound['char_end']] != annotation.text:
                raise ValueError('evidence_alignment_failed')
            node_bounds.append((bound['char_start'], bound['char_end']))
        for relation in sentence.relations:
            if not isinstance(relation.infons.get('dependency'), str) or not relation.infons['dependency'] or \
                    len(relation.nodes) != 2 or {n.role for n in relation.nodes} != {'governor', 'dependant'} or \
                    any(n.refid not in node_ids for n in relation.nodes):
                raise ValueError('dependency_graph_unavailable')
        if len(node_ids) > 1 and not sentence.relations:
            raise ValueError('dependency_graph_unavailable')
        sentence_rows.append({'span': evidence, 'nodes': len(node_ids),
                              'relations': len(sentence.relations)})
        ids_and_bounds.append(node_bounds)
    for annotation in passage.annotations:
        if len(annotation.locations) != 1:
            raise ValueError('evidence_alignment_failed')
        loc = annotation.locations[0]
        bound = span(passage.text, loc.offset, loc.length)
        containing = [i for i, row in enumerate(sentence_rows)
                      if row['span']['char_start'] <= bound['char_start'] < bound['char_end'] <= row['span']['char_end']]
        if len(containing) != 1 or not any(left < bound['char_end'] and right > bound['char_start']
                                          for left, right in ids_and_bounds[containing[0]]):
            raise ValueError('dependency_graph_unavailable')
    return {'cleaned_text_sha256': digest(passage.text), 'sentences': sentence_rows,
            'annotation_inventory_sha256': annotation_inventory(passage),
            'syntax_correctness_verified': False}


def unavailable(reason):
    if reason not in FAILURES:
        raise ValueError('sanitized_failure_code_required')
    return {'schema_version': VERSION, 'status': 'failed_unavailable',
            'failure_reason': reason, 'native_labels': None, 'mentions': None,
            'mention_conflict_view': None, 'hard_action_eligible': False,
            'clinical_score': None, 'regeneration_authorized': False}


def serialize(document, original_text, values, receipt, *, detector_error_count):
    """Serialize already computed labels; never process raw files or infer states.

    receipt must have been captured from the same cleaned passage before native
    cleanup. An ERROR emitted by swallowed detector code prevents completion.
    This adapter is diagnostic-only; no label is promoted or old score changed.
    """
    bounded_text(original_text)
    if type(detector_error_count) is not int or detector_error_count < 0:
        raise ValueError('explicit_sanitized_detector_error_count_required')
    if detector_error_count:
        return unavailable('detector_error')
    if len(document.passages) != 1 or document.passages[0].offset != 0:
        return unavailable('unsupported_section_configuration')
    passage = document.passages[0]
    if not isinstance(passage.text, str) or not passage.text.strip():
        return unavailable('empty_cleaned_source')
    bounded_text(passage.text)
    if not isinstance(receipt, dict) or set(receipt) != {
            'cleaned_text_sha256', 'sentences', 'annotation_inventory_sha256', 'syntax_correctness_verified'} or \
            receipt['cleaned_text_sha256'] != digest(passage.text) or \
            receipt['annotation_inventory_sha256'] != annotation_inventory(passage) or \
            receipt['syntax_correctness_verified'] is not False or \
            not isinstance(receipt['sentences'], list) or not receipt['sentences']:
        raise ValueError('same_source_syntax_availability_receipt_required')
    for row in receipt['sentences']:
        if set(row) != {'span', 'nodes', 'relations'} or type(row['nodes']) is not int or row['nodes'] < 1 or \
                type(row['relations']) is not int or row['relations'] < 0 or \
                (row['nodes'] > 1 and row['relations'] == 0) or \
                row['span'] != span(passage.text, row['span']['char_start'],
                                   row['span']['char_end'] - row['span']['char_start']):
            raise ValueError('same_source_syntax_availability_receipt_required')
    native = native_vector(values)
    mentions, grouped = [], {name: [] for name in FINDINGS}
    seen = set()
    for annotation in passage.annotations:
        category = annotation.infons.get('observation')
        if category not in CATEGORIES or len(annotation.locations) != 1 or \
                not isinstance(annotation.id, str) or not annotation.id.isdigit() or annotation.id in seen:
            raise ValueError('exact_native_annotation_inventory_required')
        seen.add(annotation.id)
        location = annotation.locations[0]
        evidence = span(passage.text, location.offset, location.length)
        if passage.text[location.offset:location.offset + location.length] != annotation.text:
            raise ValueError('evidence_alignment_failed')
        finding = category.lower().replace(' ', '_')
        state = mention_state(annotation.infons)
        grouped[finding].append(state)
        mentions.append({'annotation_id': annotation.id, 'finding': finding, 'span': evidence,
                         'native_mention_state': state,
                         'negation_key_present': 'negation' in annotation.infons,
                         'uncertainty_key_present': 'uncertainty' in annotation.infons,
                         'unmarked_positive_is_not_verified_current_presence': state == 'positive'})
    view = {finding: conflict_view(states) for finding, states in grouped.items()}
    # Native No Finding is an aggregation convenience, not a current finding.
    view['no_finding'] = None
    return {'schema_version': VERSION, 'status': 'complete', 'failure_reason': None,
            'upstream_revisions': UPSTREAM, 'original_text_sha256': digest(original_text),
            'cleaned_text_sha256': digest(passage.text), 'source_mode': 'official_default_full_report',
            'original_offset_alignment_available': False, 'native_labels': native,
            'mentions': mentions, 'mention_conflict_view': view, 'syntax_receipt': receipt,
            'native_ontology_is_not_literal_opacity_head': True,
            'native_no_finding_does_not_negate_unmentioned_findings': True,
            'clinical_score': None, 'hard_action_eligible': False,
            'independent_clinical_validation': False, 'regeneration_authorized': False}

"""Character-preserving bridge for official stem-level mentions, no rule fix.

V1 and all consumed outputs remain immutable. Only blank Doc token boundaries
are split at existing native offsets; source bytes/mention intervals are not
expanded, contracted or reinterpreted. Eligibility semantics are unchanged.
"""
from __future__ import annotations

from . import native_context_scope as v1

VERSION = 'tricompose-native-context-scope-v2'
GROUP = v1.GROUP
POLICY = {**v1.POLICY, 'version': VERSION, 'native_character_boundary_bridge': True,
          'source_or_native_span_expansion': False, 'default_token_boundaries_may_change': True,
          'new_linguistic_or_anatomy_rules': False}


def split_plan(doc, mentions):
    boundaries = {value for row in mentions for value in (row['span']['char_start'], row['span']['char_end'])}
    operations = []
    for token in doc:
        left, right = token.idx, token.idx + len(token.text)
        cuts = sorted(value for value in boundaries if left < value < right)
        if not cuts:
            continue
        edges = [left, *cuts, right]
        parts = [doc.text[start:end] for start, end in zip(edges, edges[1:])]
        if any(not part for part in parts) or ''.join(parts) != token.text:
            raise ValueError('exact_character_preserving_split_required')
        operations.append((token, parts, {'original_token_span': v1.span(doc.text, left, right-left),
                                         'split_char_offsets': cuts}))
    return operations


def exact_token_bridge(doc, mentions):
    if doc.has_annotation('DEP'):
        raise ValueError('blank_unparsed_doc_required_for_boundary_bridge')
    original = doc.text
    operations = split_plan(doc, mentions)
    with doc.retokenize() as retokenizer:
        for token, parts, _ in operations:
            # API-required heads for an unparsed Doc, not inferred dependencies.
            retokenizer.split(token, orths=parts, heads=[token]*len(parts))
    if doc.text != original:
        raise ValueError('source_text_changed_during_boundary_bridge')
    return {'source_text_changed': False, 'native_spans_expanded_or_contracted': False,
            'default_token_boundaries_changed': bool(operations),
            'split_tokens': len(operations), 'operations': [row[2] for row in operations],
            'dependency_annotation_is_not_produced_or_verified': True}


def parse_context(nlp, text, native):
    if native.get('status') != 'complete':
        return {**v1.context_unavailable('native_unavailable'), 'schema_version': VERSION}
    try:
        cleaned = v1.check_native(text, native)
        doc = nlp.make_doc(cleaned)
        bridge = exact_token_bridge(doc, native['mentions'])
        targets = []
        for left, right, finding in sorted({v1.target_key(row) for row in native['mentions']}):
            entity = doc.char_span(left, right, label=finding, alignment_mode='strict')
            if entity is None:
                raise ValueError('exact_native_boundaries_still_required')
            targets.append(entity)
        doc.spans[GROUP] = targets
    except Exception:
        return {**v1.context_unavailable('source_or_token_alignment_failed'), 'schema_version': VERSION}
    try:
        output = v1.serialize_context(nlp(doc), cleaned, native)
        return {**output, 'schema_version': VERSION, 'token_boundary_bridge': bridge}
    except Exception:
        return {**v1.context_unavailable('context_parser_failed'), 'schema_version': VERSION}


def gate(native, context):
    result = v1.gate(native, context)
    return {**result, 'schema_version': VERSION, 'policy': POLICY}

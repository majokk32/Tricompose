"""CXRGraph manual JSON schema adapter, not a clinical scorer or extractor.

Public code inspected at the pinned author revision below; no source dataset
has been obtained. Patients may only be passed internally by an approved
private Slurm worker. This module has no IO/model/network operations.
"""
from .entity_gold_contract import (
    POLICY as BASE_POLICY, RELATIONS as COMMON_RELATIONS, V1_LABELS,
    compare_annotations, digest, normalize_graph, validate_record,
)
from .radgraph_reference_contract import require

VERSION = 'tricompose-cxrgraph-manual-adapter-v1'
SOURCE = {
    'repository': 'yxliao95/arrg_cxrgraph',
    'revision': '4b0edaf75d18128cbccfaf90ff92549984600056',
    'source_sha256': {
        'config.py': '84d4a170c42c5ccf48e4a693e259d5e9375b63e1ddd298380496f089ec31ff89',
        'inference/inference_ent.py': 'c4eb4597626c5710baf0ce60a4f1ec33c19e2a31d995a210f63da68c2460f3aa',
        'pipe1_ner_tokaux_attrcls_sent.py': '764e4b5a523bd6da82dd581382c6f32ead9e2102cbd11824c12bf89e839f3e0a',
    },
}
CORE_LABELS = {
    'Anatomy': 'ANAT-DP', 'Observation-Present': 'OBS-DP',
    'Observation-Absent': 'OBS-DA', 'Observation-Uncertain': 'OBS-U',
}
LABELS = (*CORE_LABELS, 'Location-Attribute')
RELATIONS = (*COMMON_RELATIONS, 'part_of')
ATTRIBUTES = {
    'normality': ('Normal', 'Abnormal'),
    'action': ('Removable', 'Essential'),
    'change': ('Positive', 'Negative', 'Unchanged'),
}
POLICY = {**BASE_POLICY, 'native_dataset_schema_validated': False,
    'full_official_cxrgraph_metric': False, 'location_path_flattened': False,
    'change_is_finding_polarity': False, 'normality_defaults_imputed': False,
    'native_attribute_clinical_eligibility_verified': False}


def _project(entities, relations):
    """Literal common-label projection only; never collapse location paths."""
    core = {tuple(e): [e[0], e[1], V1_LABELS[CORE_LABELS[e[2]]]]
            for e in entities if e[2] in CORE_LABELS}
    projected_entities = sorted(core.values())
    projected_relations = sorted([[r[0], *core[tuple(r[1:4])], *core[tuple(r[4:])]]
        for r in relations if r[0] in COMMON_RELATIONS
        and tuple(r[1:4]) in core and tuple(r[4:]) in core])
    return projected_entities, projected_relations


def _coverage(entities, relations, attributes):
    # 3*N is a storage-slot inventory, not a clinically eligible-attribute denominator.
    return {'native_entity_count': len(entities), 'native_relation_count': len(relations),
        'location_attribute_entities_unmapped': sum(e[2] == 'Location-Attribute' for e in entities),
        'part_of_relations_unmapped': sum(r[0] == 'part_of' for r in relations),
        'relations_touching_location_attribute_unmapped': sum(
            r[3] == 'Location-Attribute' or r[6] == 'Location-Attribute' for r in relations),
        'cross_sentence_relations_preserved': None,
        'assigned_attributes_by_dimension': {name: sum(a[3] == name for a in attributes)
            for name in ATTRIBUTES},
        'unassigned_attribute_slots_not_imputed': len(entities) * 3 - len(attributes)}


def normalize_manual_document(document, *, report_id, source_report_sha256,
                              origin, reader_id='reader_01'):
    """Preserve manual native spans and attributes, without report/source text.

    The source uses document-global inclusive token offsets within sentence
    lists. Gold attributes use NA; X is a classifier null class, not gold.
    Declared joint annotations do not become two independent reader votes.
    """
    require(origin in ('authored_fixture', 'declared_human_annotation'),
        'manual_gold_origin_required')
    require(isinstance(document, dict) and set(document) ==
        {'doc_key', 'sentences', 'ner', 'relations', 'entity_attributes'}
        and isinstance(document['doc_key'], str), 'exact_manual_document_schema_required')
    sentences = document['sentences']
    require(isinstance(sentences, list) and 1 <= len(sentences) <= 2048,
        'bounded_manual_sentences_required')
    words, bounds, token_characters = [], [], 0
    for sentence in sentences:
        require(isinstance(sentence, list) and 1 <= len(sentence) <= 20000
            and all(isinstance(w, str) and w and len(w) <= 10000
                    and not any(c.isspace() for c in w) for w in sentence),
            'exact_nonempty_native_tokens_required')
        start = len(words)
        words.extend(sentence)
        token_characters += sum(len(w) for w in sentence)
        require(len(words) <= 20000 and token_characters + len(words) - 1 <= 100000,
            'bounded_manual_document_required')
        bounds.append([start, len(words)])
    require(len(words) <= 20000 and len(' '.join(words)) <= 100000,
        'bounded_manual_document_required')
    for key in ('ner', 'relations', 'entity_attributes'):
        require(isinstance(document[key], list) and len(document[key]) == len(sentences)
            and all(isinstance(items, list) and len(items) <= 4096 for items in document[key]),
            'sentence_annotation_inventory_required')

    def span(start, end, sentence_index):
        left, right = bounds[sentence_index]
        require(type(start) is int and type(end) is int and left <= start <= end < right,
            'document_global_sentence_bound_span_required')
        return (start, end + 1)

    entities, by_span, sentence_for_span = [], {}, {}
    for index, items in enumerate(document['ner']):
        for item in items:
            require(isinstance(item, (list, tuple)) and len(item) == 3
                and isinstance(item[2], str) and item[2] in LABELS, 'native_manual_entity_required')
            key = span(item[0], item[1], index)
            require(key not in by_span, 'duplicate_or_ambiguous_manual_entity_span')
            entity = [*key, item[2]]
            by_span[key] = entity
            sentence_for_span[key] = index
            entities.append(entity)
    require(len(entities) <= 4096, 'bounded_manual_entities_required')
    relations, cross_sentence = [], 0
    for index, items in enumerate(document['relations']):
        for item in items:
            require(isinstance(item, (list, tuple)) and len(item) == 5
                and isinstance(item[4], str) and item[4] in RELATIONS
                and all(type(v) is int for v in item[:4]), 'native_manual_relation_required')
            head = span(item[0], item[1], index)
            tail = (item[2], item[3] + 1)
            require(head in by_span and tail in by_span, 'manual_relation_entity_binding_required')
            cross_sentence += sentence_for_span[head] != sentence_for_span[tail]
            relations.append([item[4], *by_span[head], *by_span[tail]])
    require(len(relations) <= 16384 and len({tuple(r) for r in relations}) == len(relations),
        'bounded_unique_manual_relations_required')
    attributes, attributed_spans = [], set()
    for index, items in enumerate(document['entity_attributes']):
        for item in items:
            require(isinstance(item, (list, tuple)) and len(item) == 5,
                'three_native_attribute_slots_required')
            key = span(item[0], item[1], index)
            require(key in by_span and key not in attributed_spans, 'unique_attribute_entity_binding_required')
            attributed_spans.add(key)
            for name, value in zip(ATTRIBUTES, item[2:]):
                require(isinstance(value, str) and (value == 'NA' or value in ATTRIBUTES[name]),
                    'native_attribute_value_required')
                if value != 'NA':
                    attributes.append([*by_span[key], name, value])
    entities, relations, attributes = sorted(entities), sorted(relations), sorted(attributes)
    common_entities, common_relations = _project(entities, relations)
    ids = {tuple(e): str(i + 1) for i, e in enumerate(common_entities)}
    inverse = {value: key for key, value in V1_LABELS.items()}
    graph = {'text': ' '.join(words), 'entities': {ids[tuple(e)]: {
        'start_ix': e[0], 'end_ix': e[1] - 1, 'label': inverse[e[2]],
        'tokens': ' '.join(words[e[0]:e[1]]),
        'relations': [[r[0], ids[tuple(r[4:])]] for r in common_relations if r[1:4] == e],
    } for e in common_entities}}
    record = normalize_graph(graph, native_schema='radgraph_v1', report_id=report_id,
        source_report_sha256=source_report_sha256, origin=origin, reader_id=reader_id)
    coverage = _coverage(entities, relations, attributes)
    coverage['cross_sentence_relations_preserved'] = cross_sentence
    payload = {'schema_version': VERSION, 'source_definition': SOURCE,
        'annotation_process': 'joint_manual_annotation_declared_not_two_independent_readers',
        'sentence_bounds': bounds, 'token_sequence_sha256': digest(words),
        'native_entities': entities, 'native_relations': relations, 'native_attributes': attributes,
        'native_coverage': coverage, 'common_label_record': record,
        'native_extra_dimension_scores': {name: None for name in
            ('normality', 'action', 'change', 'location_attribute', 'part_of')},
        'clinical_score': None, 'policy': dict(POLICY)}
    return {**payload, 'adapter_sha256': digest(payload)}


def validate_adapter(adapter):
    fields = {'schema_version', 'source_definition', 'annotation_process', 'sentence_bounds',
        'token_sequence_sha256', 'native_entities', 'native_relations', 'native_attributes',
        'native_coverage', 'common_label_record', 'native_extra_dimension_scores',
        'clinical_score', 'policy', 'adapter_sha256'}
    require(isinstance(adapter, dict) and set(adapter) == fields
        and adapter['schema_version'] == VERSION and adapter['source_definition'] == SOURCE
        and adapter['policy'] == POLICY and adapter['clinical_score'] is None
        and adapter['adapter_sha256'] == digest({k: v for k, v in adapter.items() if k != 'adapter_sha256'})
        and adapter['annotation_process'] == 'joint_manual_annotation_declared_not_two_independent_readers'
        and adapter['native_extra_dimension_scores'] == {name: None for name in
            ('normality', 'action', 'change', 'location_attribute', 'part_of')},
        'sealed_native_manual_adapter_required')
    record = validate_record(adapter['common_label_record'])
    require(record['native_schema'] == 'radgraph_v1' and record['reader_id'] is not None
        and record['origin'] in ('authored_fixture', 'declared_human_annotation')
        and record['token_sequence_sha256'] == adapter['token_sequence_sha256'],
        'bound_manual_common_record_required')
    bounds = adapter['sentence_bounds']
    require(isinstance(bounds, list) and 1 <= len(bounds) <= 2048, 'bounded_sentence_partition_required')
    end = 0
    for pair in bounds:
        require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair)
            and pair[0] == end and pair[0] < pair[1] <= record['token_count'],
            'contiguous_document_sentence_partition_required')
        end = pair[1]
    require(end == record['token_count'], 'complete_sentence_partition_required')
    entities, relations, attributes = (adapter[k] for k in
        ('native_entities', 'native_relations', 'native_attributes'))
    require(isinstance(entities, list) and len(entities) <= 4096, 'bounded_native_entities_required')
    by_span, sentence_for_span = {}, {}
    for e in entities:
        require(isinstance(e, list) and len(e) == 3 and all(type(v) is int for v in e[:2])
            and 0 <= e[0] < e[1] <= record['token_count'] and isinstance(e[2], str) and e[2] in LABELS,
            'typed_native_manual_entity_required')
        key = tuple(e[:2])
        require(key not in by_span, 'unique_native_manual_spans_required')
        sentence = next((i for i, (a, b) in enumerate(bounds) if a <= e[0] < e[1] <= b), None)
        require(sentence is not None, 'native_entity_within_sentence_required')
        sentence_for_span[key] = sentence
        by_span[key] = e
    require(entities == sorted(entities), 'canonical_native_entities_required')
    keys = {tuple(e) for e in entities}
    require(isinstance(relations, list) and len(relations) <= 16384, 'bounded_native_relations_required')
    for r in relations:
        require(isinstance(r, list) and len(r) == 7 and isinstance(r[0], str) and r[0] in RELATIONS
            and all(type(r[i]) is int for i in (1, 2, 4, 5))
            and all(isinstance(r[i], str) for i in (3, 6))
            and tuple(r[1:4]) in keys and tuple(r[4:]) in keys, 'exact_native_relation_binding_required')
    require(relations == sorted(relations) and len({tuple(r) for r in relations}) == len(relations),
        'canonical_native_relations_required')
    require(isinstance(attributes, list) and len(attributes) <= len(entities) * 3,
        'bounded_native_attributes_required')
    for a in attributes:
        require(isinstance(a, list) and len(a) == 5 and all(type(v) is int for v in a[:2])
            and all(isinstance(a[i], str) for i in (2, 3, 4)) and tuple(a[:3]) in keys
            and a[3] in ATTRIBUTES and a[4] in ATTRIBUTES[a[3]], 'exact_native_attribute_binding_required')
    require(attributes == sorted(attributes) and len({tuple(a[:4]) for a in attributes}) == len(attributes),
        'canonical_unique_native_attribute_slots_required')
    common_entities, common_relations = _project(entities, relations)
    require(record['entities'] == common_entities and record['relations'] == common_relations,
        'literal_common_projection_required')
    coverage = _coverage(entities, relations, attributes)
    coverage['cross_sentence_relations_preserved'] = sum(
        sentence_for_span[tuple(r[1:3])] != sentence_for_span[tuple(r[4:6])] for r in relations)
    require(adapter['native_coverage'] == coverage, 'complete_native_dimension_coverage_required')
    return adapter


def compare_to_native_prediction(adapter, prediction_record):
    """Common extraction diagnostic plus unscored native dimensions.

    Not official CXRGraph evaluation: RadGraph-XL does not predict its extra
    attributes or entity/relation types, and annotation granularity may differ.
    """
    validate_adapter(adapter)
    result = compare_annotations(adapter['common_label_record'], prediction_record)
    payload = {'schema_version': VERSION + '-common-comparison',
        'adapter_sha256': adapter['adapter_sha256'], 'common_comparison': result,
        'native_coverage': adapter['native_coverage'],
        'native_extra_dimension_scores': adapter['native_extra_dimension_scores'],
        'metric_role': 'common_label_exact_extraction_diagnostic_not_full_cxrgraph_metric',
        'clinical_score': None, 'policy': dict(POLICY)}
    return {**payload, 'comparison_sha256': digest(payload)}

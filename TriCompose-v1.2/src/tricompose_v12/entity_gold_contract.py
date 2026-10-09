"""Exact-span annotation diagnostic; no inference, IO or clinical qualification.

RadGraph-v1's four labels are mapped only to their exact native XL equivalents.
Do not interpret observations as disease concepts or location/severity/scope.
Source identity and human provenance are declarations: this module cannot
authenticate access, verify original report bytes or establish a held-out set.
"""
import hashlib
import json
import re

from .radgraph_reference_contract import require
from .radgraph_reference_contract_v2 import NATIVE_LABELS, graph_metadata

VERSION = 'tricompose-entity-gold-contract-v1'
V1_LABELS = {
    'ANAT-DP': 'Anatomy::definitely present',
    'OBS-DP': 'Observation::definitely present',
    'OBS-DA': 'Observation::definitely absent',
    'OBS-U': 'Observation::uncertain',
}
STATES = {
    'Observation::definitely present': 'positive',
    'Observation::definitely absent': 'negative',
    'Observation::uncertain': 'uncertain',
}
RELATIONS = ('located_at', 'modify', 'suggestive_of')
POLICY = {
    'clinical_qualified': False, 'access_verified_here': False,
    'human_provenance_verified_here': False, 'raw_source_bytes_verified_here': False,
    'checkpoint_heldout_verified': False, 'scope_verified': False,
    'image_truth_verified': False, 'ehr_truth_verified': False,
    'missing_is_negative': False, 'selection_changed': False,
    'regeneration_authorized': False, 'new_training': False,
}
ORIGINS = ('declared_human_annotation', 'declared_frozen_prediction', 'authored_fixture')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[a-f0-9]{64}', value))


def normalize_graph(graph, *, native_schema, report_id, source_report_sha256,
                    origin, reader_id=None):
    """Remove text and source keys; keep word offsets and exact label identity.

    Caller must use an approved private worker for any patient graph. The
    token-sequence hash must match between annotation and prediction; no
    guessed token alignment or normalization of punctuation/case is allowed.
    """
    require(native_schema in ('radgraph_v1', 'radgraph_xl'), 'unsupported_annotation_schema')
    require(origin in ORIGINS and isinstance(report_id, str)
        and re.fullmatch(r'report_[0-9]{4,6}', report_id) and valid_hash(source_report_sha256),
        'opaque_report_and_declared_source_required')
    require(reader_id is None or isinstance(reader_id, str)
        and re.fullmatch(r'reader_[0-9]{2}', reader_id), 'opaque_reader_required')
    require(origin != 'declared_human_annotation' or reader_id is not None,
        'human_reader_required')
    require(origin != 'declared_frozen_prediction' or reader_id is None,
        'prediction_is_not_a_reader')
    require(isinstance(graph, dict) and isinstance(graph.get('text'), str)
        and len(graph['text']) <= 100000 and isinstance(graph.get('entities'), dict)
        and len(graph['entities']) <= 4096, 'bounded_native_graph_required')
    if native_schema == 'radgraph_xl':
        graph_metadata(graph)  # Reuse the unchanged pinned native validator.
    tokens = graph['text'].split()
    require(len(tokens) <= 20000, 'bounded_token_sequence_required')
    supported = sorted(V1_LABELS.values() if native_schema == 'radgraph_v1' else NATIVE_LABELS)
    entities, by_id, occupied = [], {}, set()
    for key, entity in graph['entities'].items():
        require(isinstance(key, str) and re.fullmatch(r'[1-9][0-9]*', key)
            and isinstance(entity, dict), 'native_entity_required')
        label = entity.get('label')
        require(isinstance(label, str) and label in
            (V1_LABELS if native_schema == 'radgraph_v1' else NATIVE_LABELS),
            'unsupported_native_entity_label')
        label = V1_LABELS[label] if native_schema == 'radgraph_v1' else label
        start, end = entity.get('start_ix'), entity.get('end_ix')
        require(type(start) is int and type(end) is int and 0 <= start <= end < len(tokens),
            'valid_native_word_span_required')
        require(entity.get('tokens') == ' '.join(tokens[start:end + 1]),
            'native_entity_text_span_mismatch')
        require((start, end + 1) not in occupied, 'duplicate_or_ambiguous_entity_span')
        occupied.add((start, end + 1))
        value = [start, end + 1, label]
        by_id[key] = value
        entities.append(value)
    relations = []
    for key, entity in graph['entities'].items():
        outgoing = entity.get('relations')
        require(isinstance(outgoing, list) and len(outgoing) <= 4096,
            'bounded_native_relations_required')
        for relation in outgoing:
            require(isinstance(relation, (list, tuple)) and len(relation) == 2
                and isinstance(relation[0], str) and relation[0] in RELATIONS
                and isinstance(relation[1], str) and relation[1] in by_id,
                'supported_typed_relation_required')
            relations.append([relation[0], *by_id[key], *by_id[relation[1]]])
    require(len(relations) <= 16384 and len({tuple(r) for r in relations}) == len(relations),
        'bounded_unique_relations_required')
    payload = {'schema_version': VERSION, 'report_id': report_id, 'reader_id': reader_id,
        'origin': origin, 'native_schema': native_schema,
        'source_report_sha256': source_report_sha256,
        'token_sequence_sha256': digest(tokens), 'token_count': len(tokens),
        'offset_unit': 'whitespace_word_end_exclusive', 'supported_labels': supported,
        'entities': sorted(entities), 'relations': sorted(relations), 'policy': dict(POLICY)}
    return {**payload, 'record_sha256': digest(payload)}


def validate_record(record):
    fields = {'schema_version', 'report_id', 'reader_id', 'origin', 'native_schema',
        'source_report_sha256', 'token_sequence_sha256', 'token_count', 'offset_unit',
        'supported_labels', 'entities', 'relations', 'policy', 'record_sha256'}
    require(isinstance(record, dict) and set(record) == fields,
        'sealed_annotation_record_required')
    require(record['schema_version'] == VERSION and record['policy'] == POLICY
        and valid_hash(record['record_sha256'])
        and record['record_sha256'] == digest({k: v for k, v in record.items()
                                              if k != 'record_sha256'}),
        'annotation_record_integrity_required')
    require(record['native_schema'] in ('radgraph_v1', 'radgraph_xl')
        and record['origin'] in ORIGINS and isinstance(record['report_id'], str)
        and re.fullmatch(r'report_[0-9]{4,6}', record['report_id'])
        and valid_hash(record['source_report_sha256']) and valid_hash(record['token_sequence_sha256'])
        and type(record['token_count']) is int and 0 <= record['token_count'] <= 20000
        and record['offset_unit'] == 'whitespace_word_end_exclusive',
        'bounded_annotation_identity_required')
    reader = record['reader_id']
    require(reader is None or isinstance(reader, str) and re.fullmatch(r'reader_[0-9]{2}', reader),
        'opaque_reader_required')
    require(record['origin'] != 'declared_human_annotation' or reader is not None,
        'human_reader_required')
    require(record['origin'] != 'declared_frozen_prediction' or reader is None,
        'prediction_is_not_a_reader')
    expected = sorted(V1_LABELS.values() if record['native_schema'] == 'radgraph_v1' else NATIVE_LABELS)
    require(record['supported_labels'] == expected, 'annotation_label_scope_required')
    entities = record['entities']
    require(isinstance(entities, list) and len(entities) <= 4096, 'bounded_entities_required')
    for entity in entities:
        require(isinstance(entity, list) and len(entity) == 3 and type(entity[0]) is int
            and type(entity[1]) is int and 0 <= entity[0] < entity[1] <= record['token_count']
            and isinstance(entity[2], str) and entity[2] in expected,
            'typed_annotation_span_required')
    require(entities == sorted(entities) and len({tuple(e[:2]) for e in entities}) == len(entities),
        'canonical_unique_entity_spans_required')
    keys = {tuple(e) for e in entities}
    relations = record['relations']
    require(isinstance(relations, list) and len(relations) <= 16384, 'bounded_relations_required')
    for relation in relations:
        require(isinstance(relation, list) and len(relation) == 7
            and isinstance(relation[0], str) and relation[0] in RELATIONS
            and all(type(relation[i]) is int for i in (1, 2, 4, 5))
            and all(isinstance(relation[i], str) for i in (3, 6))
            and tuple(relation[1:4]) in keys and tuple(relation[4:]) in keys,
            'exact_typed_relation_endpoints_required')
    require(relations == sorted(relations)
        and len({tuple(r) for r in relations}) == len(relations), 'canonical_unique_relations_required')
    return record


def count_metrics(tp, fp, fn):
    require(all(type(n) is int and n >= 0 for n in (tp, fp, fn)), 'nonnegative_counts_required')
    return {'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def exact_metrics(gold, prediction):
    return count_metrics(len(gold & prediction), len(prediction - gold), len(gold - prediction))


def compare_annotations(gold, prediction):
    """Extraction diagnostic for one declared reader, not report factuality.

    Out-of-gold-scope XL labels and their relations are counted as unavailable,
    not silently projected to v1 observations. Missing prediction is an error;
    valid zero-entity output remains distinguishable from unavailable output.
    """
    validate_record(gold)
    validate_record(prediction)
    require(gold['origin'] in ('declared_human_annotation', 'authored_fixture')
        and prediction['origin'] in ('declared_frozen_prediction', 'authored_fixture')
        and gold['reader_id'] is not None and prediction['reader_id'] is None,
        'separate_declared_gold_reader_and_prediction_required')
    require(all(gold[key] == prediction[key] for key in
        ('report_id', 'source_report_sha256', 'token_sequence_sha256', 'token_count', 'offset_unit')),
        'same_declared_report_and_exact_token_sequence_required')
    scope = set(gold['supported_labels'])
    g = {tuple(e) for e in gold['entities']}
    all_p = {tuple(e) for e in prediction['entities']}
    p = {e for e in all_p if e[2] in scope}
    gr = {tuple(r) for r in gold['relations']}
    all_pr = {tuple(r) for r in prediction['relations']}
    pr = {r for r in all_pr if r[3] in scope and r[6] in scope}
    go = {e[:2]: STATES[e[2]] for e in g if e[2] in STATES}
    po = {e[:2]: STATES[e[2]] for e in p if e[2] in STATES}
    shared = go.keys() & po.keys()
    confusion = {a: {b: 0 for b in STATES.values()} for a in STATES.values()}
    for span in shared:
        confusion[go[span]][po[span]] += 1
    matched = len(shared)
    result = {'schema_version': VERSION + '-comparison', 'report_id': gold['report_id'],
        'reader_id': gold['reader_id'], 'gold_record_sha256': gold['record_sha256'],
        'prediction_record_sha256': prediction['record_sha256'],
        'gold_origin': gold['origin'], 'prediction_origin': prediction['origin'],
        'entity_metrics_in_gold_label_scope': exact_metrics(g, p),
        'entity_metrics_by_label': {label: exact_metrics({e for e in g if e[2] == label},
            {e for e in p if e[2] == label}) for label in sorted(scope)},
        'relation_metrics_in_gold_label_scope': exact_metrics(gr, pr),
        'relation_metrics_by_type': {kind: exact_metrics({r for r in gr if r[0] == kind},
            {r for r in pr if r[0] == kind}) for kind in RELATIONS},
        'out_of_gold_label_scope': {'prediction_entities': len(all_p - p),
            'prediction_relations': len(all_pr - pr), 'labels': sorted({e[2] for e in all_p - p})},
        'polarity': {'gold_observation_spans': len(go), 'prediction_observation_spans': len(po),
            'matched_observation_spans': matched, 'gold_spans_without_matched_observation': len(go) - matched,
            'prediction_spans_without_gold_observation': len(po) - matched,
            'gold_span_coverage': matched / len(go) if go else None,
            'conditional_accuracy': sum(confusion[a][a] for a in confusion) / matched if matched else None,
            'confusion_on_matched_observation_spans': confusion},
        'metric_role': 'exact_span_extraction_diagnostic_not_official_f1radgraph_reward',
        'clinical_score': None, 'confirmed_faulty_modality': None, 'policy': dict(POLICY)}
    return {**result, 'comparison_sha256': digest(result)}


def aggregate_reader(comparisons, *, reader_id, attempted_report_ids, supported_labels):
    """Micro counts on eligible pairs with an explicit attempted denominator.

    Never merge readers, duplicate reports or authored and declared-human gold.
    Missing/failed comparisons remain unavailable, not zero-error/zero-score.
    No patient-level CI: patient grouping has not been supplied or verified.
    """
    require(isinstance(reader_id, str) and re.fullmatch(r'reader_[0-9]{2}', reader_id),
        'opaque_reader_required')
    require(isinstance(attempted_report_ids, list) and 1 <= len(attempted_report_ids) <= 1024
        and all(isinstance(r, str) and re.fullmatch(r'report_[0-9]{4,6}', r)
                for r in attempted_report_ids)
        and attempted_report_ids == sorted(set(attempted_report_ids)),
        'fixed_unique_attempted_report_inventory_required')
    require(isinstance(supported_labels, list) and supported_labels in
        (sorted(V1_LABELS.values()), sorted(NATIVE_LABELS)), 'fixed_annotation_label_scope_required')
    require(isinstance(comparisons, list) and len(comparisons) <= len(attempted_report_ids),
        'bounded_reader_comparisons_required')
    seen, origins = set(), set()
    counts = {'entity': [0, 0, 0], 'relation': [0, 0, 0]}
    labels = {label: [0, 0, 0] for label in supported_labels}
    relations = {kind: [0, 0, 0] for kind in RELATIONS}
    confusion = {a: {b: 0 for b in STATES.values()} for a in STATES.values()}
    gold_obs = pred_obs = matched = outside_entities = outside_relations = 0
    outside_labels = set()
    fields = {'schema_version', 'report_id', 'reader_id', 'gold_record_sha256',
        'prediction_record_sha256', 'gold_origin', 'prediction_origin',
        'entity_metrics_in_gold_label_scope', 'entity_metrics_by_label',
        'relation_metrics_in_gold_label_scope', 'relation_metrics_by_type',
        'out_of_gold_label_scope', 'polarity', 'metric_role', 'clinical_score',
        'confirmed_faulty_modality', 'policy', 'comparison_sha256'}

    def triple(metric):
        require(isinstance(metric, dict) and set(metric) == {'tp', 'fp', 'fn', 'precision', 'recall', 'f1'},
            'exact_count_metric_required')
        values = [metric[k] for k in ('tp', 'fp', 'fn')]
        require(metric == count_metrics(*values), 'count_metric_integrity_required')
        return values

    def add(target, values):
        for i, n in enumerate(values):
            target[i] += n

    for r in comparisons:
        require(isinstance(r, dict) and set(r) == fields and r['policy'] == POLICY
            and r['schema_version'] == VERSION + '-comparison'
            and r['comparison_sha256'] == digest({k: v for k, v in r.items() if k != 'comparison_sha256'})
            and r['clinical_score'] is None and r['confirmed_faulty_modality'] is None
            and r['metric_role'] == 'exact_span_extraction_diagnostic_not_official_f1radgraph_reward',
            'sealed_extraction_comparison_required')
        require(r['reader_id'] == reader_id and isinstance(r['report_id'], str)
            and r['report_id'] in attempted_report_ids and r['report_id'] not in seen
            and r['gold_origin'] in ('authored_fixture', 'declared_human_annotation')
            and r['prediction_origin'] in ('authored_fixture', 'declared_frozen_prediction')
            and valid_hash(r['gold_record_sha256']) and valid_hash(r['prediction_record_sha256']),
            'same_reader_unique_inventory_bound_reports_required')
        seen.add(r['report_id'])
        origins.add((r['gold_origin'], r['prediction_origin']))
        require(len(origins) == 1, 'do_not_mix_authored_and_declared_human_cohorts')
        require(isinstance(r['entity_metrics_by_label'], dict)
            and sorted(r['entity_metrics_by_label']) == supported_labels
            and isinstance(r['relation_metrics_by_type'], dict)
            and set(r['relation_metrics_by_type']) == set(RELATIONS), 'same_reader_metric_scope_required')
        for group, name, by_name, target in (
            ('entity', 'entity_metrics_in_gold_label_scope', 'entity_metrics_by_label', labels),
            ('relation', 'relation_metrics_in_gold_label_scope', 'relation_metrics_by_type', relations)):
            total = triple(r[name])
            parts = {key: triple(value) for key, value in r[by_name].items()}
            require(total == [sum(p[i] for p in parts.values()) for i in range(3)],
                'partitioned_count_totals_required')
            add(counts[group], total)
            for key, values in parts.items():
                add(target[key], values)
        outside = r['out_of_gold_label_scope']
        require(isinstance(outside, dict) and set(outside) ==
            {'prediction_entities', 'prediction_relations', 'labels'}
            and all(type(outside[k]) is int and outside[k] >= 0
                for k in ('prediction_entities', 'prediction_relations'))
            and isinstance(outside['labels'], list)
            and all(isinstance(v, str) and v in NATIVE_LABELS and v not in supported_labels
                for v in outside['labels'])
            and outside['labels'] == sorted(set(outside['labels'])), 'explicit_out_of_scope_counts_required')
        outside_entities += outside['prediction_entities']
        outside_relations += outside['prediction_relations']
        outside_labels.update(outside['labels'])
        p = r['polarity']
        require(isinstance(p, dict) and set(p) == {'gold_observation_spans', 'prediction_observation_spans',
            'matched_observation_spans', 'gold_spans_without_matched_observation',
            'prediction_spans_without_gold_observation', 'gold_span_coverage',
            'conditional_accuracy', 'confusion_on_matched_observation_spans'}, 'polarity_coverage_required')
        for k in ('gold_observation_spans', 'prediction_observation_spans', 'matched_observation_spans',
                  'gold_spans_without_matched_observation', 'prediction_spans_without_gold_observation'):
            require(type(p[k]) is int and p[k] >= 0, 'nonnegative_polarity_counts_required')
        cm = p['confusion_on_matched_observation_spans']
        require(isinstance(cm, dict) and set(cm) == set(confusion)
            and all(isinstance(row, dict) and set(row) == set(confusion)
                    and all(type(n) is int and n >= 0 for n in row.values()) for row in cm.values()),
            'three_state_confusion_required')
        ng, np, nm = (p[k] for k in ('gold_observation_spans', 'prediction_observation_spans', 'matched_observation_spans'))
        by_label = r['entity_metrics_by_label']
        require(ng == sum(by_label[label]['tp'] + by_label[label]['fn'] for label in STATES)
            and np == sum(by_label[label]['tp'] + by_label[label]['fp'] for label in STATES)
            and nm == sum(sum(row.values()) for row in cm.values()) and nm <= min(ng, np)
            and p['gold_spans_without_matched_observation'] == ng - nm
            and p['prediction_spans_without_gold_observation'] == np - nm
            and p['gold_span_coverage'] == (nm / ng if ng else None)
            and p['conditional_accuracy'] == (sum(cm[a][a] for a in cm) / nm if nm else None),
            'polarity_count_and_coverage_integrity_required')
        for label, state in STATES.items():
            require(sum(cm[state].values()) <= by_label[label]['tp'] + by_label[label]['fn']
                and sum(cm[a][state] for a in cm) <= by_label[label]['tp'] + by_label[label]['fp'],
                'polarity_class_denominators_required')
        gold_obs += ng
        pred_obs += np
        matched += nm
        for a in cm:
            for b in cm[a]:
                confusion[a][b] += cm[a][b]
    result = {'schema_version': VERSION + '-reader-aggregate', 'reader_id': reader_id,
        'attempted_reports': len(attempted_report_ids), 'eligible_reports': len(seen),
        'unavailable_report_ids': sorted(set(attempted_report_ids) - seen),
        'comparison_availability': len(seen) / len(attempted_report_ids),
        'gold_origin': next(iter(origins))[0] if origins else None,
        'entity_micro': count_metrics(*counts['entity']) if seen else None,
        'relation_micro': count_metrics(*counts['relation']) if seen else None,
        'entity_by_label': {label: count_metrics(*n) if seen else None for label, n in labels.items()},
        'relation_by_type': {kind: count_metrics(*n) if seen else None for kind, n in relations.items()},
        'out_of_gold_label_scope': {'prediction_entities': outside_entities,
            'prediction_relations': outside_relations, 'labels': sorted(outside_labels)},
        'polarity': {'gold_observation_spans': gold_obs, 'prediction_observation_spans': pred_obs,
            'matched_observation_spans': matched, 'gold_span_coverage': matched / gold_obs if gold_obs else None,
            'conditional_accuracy': sum(confusion[a][a] for a in confusion) / matched if matched else None,
            'confusion_on_matched_observation_spans': confusion},
        'clinical_score': None, 'policy': dict(POLICY)}
    return {**result, 'aggregate_sha256': digest(result)}

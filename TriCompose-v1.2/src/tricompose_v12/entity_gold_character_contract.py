"""Lossless exact character-span extraction diagnostic across tokenizers.

Never changes source characters/case, guesses a span, snaps to a gold boundary,
or uses gold labels to align predictions. Offsets refer to the released token
sequence joined with single spaces. Not an official CXRGraph clinical metric.
"""
from collections import Counter

from .cxrgraph_gold_adapter import validate_adapter
from .entity_gold_contract import (
    STATES, RELATIONS, V1_LABELS, count_metrics, digest, exact_metrics, validate_record,
)
from .radgraph_reference_contract import require

VERSION = 'exact-source-character-extraction-v1'


def token_bounds(source_words, native_words):
    require(isinstance(source_words, list) and isinstance(native_words, list)
        and 1 <= len(source_words) <= 20000 and 1 <= len(native_words) <= 20000
        and all(isinstance(w, str) and w and not any(c.isspace() for c in w)
                for w in source_words + native_words), 'bounded_literal_words_required')
    source = ' '.join(source_words)
    require(len(source) <= 100000, 'bounded_literal_source_required')
    original, cursor = [], 0
    for word in source_words:
        original.append([cursor, cursor + len(word)])
        cursor += len(word) + 1
    native, cursor = [], 0
    for word in native_words:
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        require(source.startswith(word, cursor), 'lossless_native_characters_required')
        native.append([cursor, cursor + len(word)])
        cursor += len(word)
    require(cursor == len(source), 'complete_literal_source_consumption_required')
    return original, native


def char_graph(record, bounds, scope):
    validate_record(record)
    require(len(bounds) == record['token_count'], 'native_token_inventory_binding_required')
    all_entities = {tuple(e): (bounds[e[0]][0], bounds[e[1] - 1][1], e[2])
                    for e in record['entities']}
    entities = {e for e in all_entities.values() if e[2] in scope}
    relations = {(r[0], *all_entities[tuple(r[1:4])], *all_entities[tuple(r[4:])])
                 for r in record['relations'] if r[3] in scope and r[6] in scope}
    return entities, relations


def compare_character_annotation(adapter, prediction, *, source_words, prediction_words):
    validate_adapter(adapter)
    validate_record(prediction)
    gold = adapter['common_label_record']
    require(gold['origin'] in ('declared_human_annotation', 'authored_fixture')
            and prediction['origin'] in ('declared_frozen_prediction', 'authored_fixture')
            and gold['reader_id'] is not None and prediction['reader_id'] is None
            and all(gold[k] == prediction[k] for k in ('report_id', 'source_report_sha256')),
            'same_declared_report_separate_gold_and_prediction_required')
    require(digest(source_words) == gold['token_sequence_sha256']
            and digest(prediction_words) == prediction['token_sequence_sha256'],
            'exact_word_inventory_hashes_required')
    original, native = token_bounds(source_words, prediction_words)
    scope = set(gold['supported_labels'])
    ge, gr = char_graph(gold, original, scope)
    pe, pr = char_graph(prediction, native, scope)
    go = {e[:2]: STATES[e[2]] for e in ge if e[2] in STATES}
    po = {e[:2]: STATES[e[2]] for e in pe if e[2] in STATES}
    shared = go.keys() & po.keys()
    confusion = {a: {b: 0 for b in STATES.values()} for a in STATES.values()}
    for span in shared:
        confusion[go[span]][po[span]] += 1
    outside = {tuple(e) for e in prediction['entities'] if e[2] not in scope}
    common = {'report_id': gold['report_id'],
        'entity_metrics_in_gold_label_scope': exact_metrics(ge, pe),
        'relation_metrics_in_gold_label_scope': exact_metrics(gr, pr),
        'entity_metrics_by_label': {label: exact_metrics({e for e in ge if e[2] == label},
            {e for e in pe if e[2] == label}) for label in sorted(scope)},
        'relation_metrics_by_type': {kind: exact_metrics({r for r in gr if r[0] == kind},
            {r for r in pr if r[0] == kind}) for kind in RELATIONS},
        'polarity': {'gold_observation_spans': len(go), 'prediction_observation_spans': len(po),
            'matched_observation_spans': len(shared),
            'confusion_on_matched_observation_spans': confusion},
        'out_of_gold_label_scope': {'prediction_entities': len(outside),
            'prediction_relations': sum(r[3] not in scope or r[6] not in scope for r in prediction['relations']),
            'labels': sorted({e[2] for e in outside})}}
    payload = {'schema_version': VERSION, 'report_id': gold['report_id'],
        'offset_unit': 'exact_source_character_end_exclusive',
        'source_representation': 'released_words_joined_with_single_spaces',
        'gold_record_sha256': gold['record_sha256'],
        'prediction_record_sha256': prediction['record_sha256'],
        'token_boundaries_identical': source_words == prediction_words,
        'source_character_count': len(' '.join(source_words)),
        'gold_character_entities': sorted(ge), 'prediction_character_entities': sorted(pe),
        'gold_character_relations': sorted(gr), 'prediction_character_relations': sorted(pr),
        'common_comparison': common,
        'metric_role': 'lossless_exact_character_extraction_development_diagnostic',
        'clinical_score': None, 'gold_boundary_snapping': False,
        'source_characters_modified': False, 'full_official_cxrgraph_metric': False}
    return {**payload, 'comparison_sha256': digest(payload)}


def aggregate_character_comparisons(comparisons, cohort):
    result = {}
    for domain in ('all', 'mimic', 'chexpert'):
        ids = {r['report_id'] for r in cohort if domain == 'all' or r['source_domain'] == domain}
        require(bool(ids), 'nonempty_attempted_domain_required')
        selected = [c for c in comparisons if c['report_id'] in ids]
        require(len({c['report_id'] for c in selected}) == len(selected),
                'unique_character_comparisons_required')
        require(all(c['schema_version'] == VERSION and c['comparison_sha256'] ==
                    digest({k: v for k, v in c.items() if k != 'comparison_sha256'}) for c in selected),
                'sealed_character_comparisons_required')
        counts = {name: Counter() for name in ('entity', 'relation')}
        by_label = {label: Counter() for label in sorted(V1_LABELS.values())}
        by_relation = {kind: Counter() for kind in RELATIONS}
        confusion = {a: {b: 0 for b in STATES.values()} for a in STATES.values()}
        ng = np = nm = ne = nr = 0
        outside_labels = set()
        for comparison in selected:
            r = comparison['common_comparison']
            for group in counts:
                m = r[group + '_metrics_in_gold_label_scope']
                counts[group].update({key: m[key] for key in ('tp', 'fp', 'fn')})
            for source, target in ((r['entity_metrics_by_label'], by_label),
                                   (r['relation_metrics_by_type'], by_relation)):
                for name, metric in source.items():
                    target[name].update({key: metric[key] for key in ('tp', 'fp', 'fn')})
            p = r['polarity']
            ng += p['gold_observation_spans']
            np += p['prediction_observation_spans']
            nm += p['matched_observation_spans']
            for a in confusion:
                for b in confusion:
                    confusion[a][b] += p['confusion_on_matched_observation_spans'][a][b]
            ne += r['out_of_gold_label_scope']['prediction_entities']
            nr += r['out_of_gold_label_scope']['prediction_relations']
            outside_labels.update(r['out_of_gold_label_scope']['labels'])
        def metric(c):
            return count_metrics(c['tp'], c['fp'], c['fn']) if selected else None
        result[domain] = {'schema_version': VERSION + '-aggregate',
            'attempted_reports': len(ids), 'eligible_reports': len(selected),
            'comparison_availability': len(selected) / len(ids),
            'unavailable_report_ids': sorted(ids - {c['report_id'] for c in selected}),
            'entity_micro': metric(counts['entity']), 'relation_micro': metric(counts['relation']),
            'entity_by_label': {name: metric(c) for name, c in by_label.items()},
            'relation_by_type': {name: metric(c) for name, c in by_relation.items()},
            'out_of_gold_label_scope': {'prediction_entities': ne, 'prediction_relations': nr,
                                      'labels': sorted(outside_labels)},
            'polarity': {'gold_observation_spans': ng, 'prediction_observation_spans': np,
                'matched_observation_spans': nm, 'gold_span_coverage': nm / ng if ng else None,
                'conditional_accuracy': sum(confusion[a][a] for a in confusion) / nm if nm else None,
                'confusion_on_matched_observation_spans': confusion},
            'clinical_score': None, 'full_official_cxrgraph_metric': False}
    return result

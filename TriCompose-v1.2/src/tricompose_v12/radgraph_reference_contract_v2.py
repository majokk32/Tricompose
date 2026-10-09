"""Full native XL ontology; no training/inference/IO or invented reward."""
import hashlib
import re

from .radgraph_reference_contract import (
    METRICS, MODEL_TYPE, POLICY, finite_score, require,
)

VERSION = 'tricompose-radgraph-reference-v2'
# Exact nonempty vocabulary in the SHA256-pinned official XL archive.
NATIVE_LABELS = (
    'Anatomy::definitely present', 'Observation::definitely present',
    'Observation::definitely absent', 'Observation::uncertain',
    'Observation::measurement::definitely present', 'Anatomy::uncertain',
    'Anatomy::definitely absent', 'Anatomy::measurement::definitely present',
    'Observation::measurement::definitely absent',
    'Observation::measurement::uncertain', 'Anatomy::measurement::uncertain',
)


def graph_metadata(graph):
    require(isinstance(graph, dict) and isinstance(graph.get('text'), str) and
            isinstance(graph.get('entities'), dict), 'native_graph_required')
    require(len(graph['text']) <= 100000 and len(graph['entities']) <= 4096,
            'bounded_graph_required')
    tokens, entities = graph['text'].split(), graph['entities']
    counts = {label: 0 for label in NATIVE_LABELS}
    relation_count = 0
    for key, entity in entities.items():
        require(isinstance(key, str) and re.fullmatch(r'[1-9][0-9]*', key) and
                isinstance(entity, dict), 'native_entity_required')
        label = entity.get('label')
        require(isinstance(label, str) and label in counts, 'unsupported_xl_label')
        start, end = entity.get('start_ix'), entity.get('end_ix')
        require(type(start) is int and type(end) is int and
                0 <= start <= end < len(tokens), 'valid_word_offsets_required')
        require(entity.get('tokens') == ' '.join(tokens[start:end + 1]),
                'entity_token_span_mismatch')
        outgoing = entity.get('relations')
        require(isinstance(outgoing, list) and len(outgoing) <= 4096,
                'bounded_native_relations_required')
        for relation in outgoing:
            require(isinstance(relation, (tuple, list)) and len(relation) == 2 and
                    isinstance(relation[0], str) and 0 < len(relation[0]) <= 128 and
                    isinstance(relation[1], str) and relation[1] in entities,
                    'valid_relation_destination_required')
        counts[label] += 1
        relation_count += len(outgoing)
    return {'tokenized_text_sha256': hashlib.sha256(graph['text'].encode()).hexdigest(),
            'entity_count': len(entities), 'relation_count': relation_count,
            'native_label_counts': counts,
            'nonmeasurement_observation_state_counts': {
                'positive': counts['Observation::definitely present'],
                'negative': counts['Observation::definitely absent'],
                'uncertain': counts['Observation::uncertain']},
            'measurement_entity_count': sum(counts[label] for label in counts
                                            if '::measurement::' in label),
            'scope_verified': False}


def score_table(pair_ids, nonempty_inputs, official_result):
    require(isinstance(pair_ids, list) and 1 <= len(pair_ids) <= 1024 and
            all(isinstance(p, str) and re.fullmatch(r'pair_[0-9]{4}', p)
                for p in pair_ids) and len(set(pair_ids)) == len(pair_ids),
            'bounded_unique_opaque_pair_ids_required')
    require(isinstance(nonempty_inputs, list) and len(nonempty_inputs) == len(pair_ids)
            and all(type(v) is bool for v in nonempty_inputs),
            'pre_call_input_availability_required')
    require(isinstance(official_result, (tuple, list)) and len(official_result) == 4,
            'official_all_result_required')
    means, components, hypotheses, references = official_result
    require(isinstance(means, (tuple, list)) and len(means) == 3 and
            isinstance(components, (tuple, list)) and len(components) == 3,
            'three_native_reward_components_required')
    for mean, scores in zip(means, components):
        require(isinstance(scores, (tuple, list)) and len(scores) == len(pair_ids) and
                all(finite_score(v) for v in scores) and finite_score(mean),
                'finite_complete_reward_inventory_required')
        require(abs(float(mean) - sum(map(float, scores)) / len(scores)) < 1e-7,
                'native_mean_inventory_mismatch')
    available = sum(nonempty_inputs)
    require(isinstance(hypotheses, list) and isinstance(references, list) and
            len(hypotheses) == len(references) == available,
            'complete_nonempty_annotation_inventory_required')
    rows, graph_index = [], 0
    for index, (pair_id, nonempty) in enumerate(zip(pair_ids, nonempty_inputs)):
        scores = [float(component[index]) for component in components]
        if nonempty:
            hypothesis = graph_metadata(hypotheses[graph_index])
            reference = graph_metadata(references[graph_index])
            graph_index += 1
            status = 'complete'
        else:
            require(all(v == 0 for v in scores), 'upstream_empty_pair_zero_required')
            hypothesis = reference = None
            status = 'empty_input_not_eligible'
        rows.append({'item_id': pair_id, 'status': status,
                     'scores': {name: value if nonempty else None
                                for name, value in zip(METRICS, scores)},
                     'hypothesis_graph': hypothesis, 'reference_graph': reference})
    return {'schema_version': VERSION, 'model_type': MODEL_TYPE,
            'metric_provenance': 'official_frozen_reference_comparison',
            'policy': dict(POLICY), 'all_attempted_pairs': len(rows),
            'eligible_pairs': available, 'records': rows}

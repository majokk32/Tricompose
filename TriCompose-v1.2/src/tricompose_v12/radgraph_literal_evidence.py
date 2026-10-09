"""Literal native-observation evidence, not clinical scoring or fault truth.

No disease dictionary, synonym guessing, ontology remapping, thresholds,
inference, filesystem IO or report text exports. All clinical scopes remain
unverified. Same-image reports are dependent proposals, never truth votes.
"""
from collections import defaultdict
import hashlib
import json

from .radgraph_reference_contract import require
from .radgraph_reference_contract_v2 import graph_metadata

VERSION = 'tricompose-radgraph-literal-evidence-v1'
STATES = {'Observation::definitely present': 'positive',
          'Observation::definitely absent': 'negative',
          'Observation::uncertain': 'uncertain'}
RELATIONS = ('native_state_agreement', 'explicit_polarity_opposition_proposal',
             'uncertain_or_mixed_state', 'anatomy_context_not_definite',
             'unmentioned_in_left', 'unmentioned_in_right')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split())


def modifier_family(root, incoming):
    """Finite native modifier closure; cycles cannot cause unbounded traversal."""
    visited, pending = set(), [root]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(incoming.get(node, ()))
    return visited


def extract(graph):
    metadata = graph_metadata(graph)
    entities = graph['entities']
    incoming = defaultdict(set)
    modifiers, unhandled = set(), 0
    for source, entity in entities.items():
        for relation, destination in entity['relations']:
            if relation == 'modify':
                incoming[destination].add(source)
                if entity['label'] in STATES and entities[destination]['label'].startswith('Observation::'):
                    modifiers.add(source)
            elif relation != 'located_at':
                unhandled += 1
    atoms = {}
    for entity_id, entity in entities.items():
        if entity['label'] not in STATES or entity_id in modifiers:
            continue
        anatomy_nodes = set()
        for relation, destination in entity['relations']:
            if relation == 'located_at' and entities[destination]['label'].startswith('Anatomy::'):
                anatomy_nodes.update(modifier_family(destination, incoming))
        anatomy_tokens = sorted({normalize(entities[node]['tokens']) for node in anatomy_nodes})
        definite = all(entities[node]['label'] == 'Anatomy::definitely present' for node in anatomy_nodes)
        observation_modifiers = modifier_family(entity_id, incoming) - {entity_id}
        modifier_tokens = sorted({normalize(entities[node]['tokens']) for node in observation_modifiers})
        concept_hash = digest(['literal_observation', normalize(entity['tokens'])])
        context_hash = digest(['native_anatomy_token_set', anatomy_tokens])
        atom_id = 'atom_' + digest([concept_hash, context_hash])
        atom = atoms.setdefault(atom_id, {'atom_id': atom_id, 'concept_sha256': concept_hash,
            'anatomy_context_sha256': context_hash, 'has_explicit_anatomy': bool(anatomy_nodes),
            'anatomy_context_all_definite': True, 'native_states': [],
            'modifier_set_hashes': [], 'occurrences': []})
        atom['anatomy_context_all_definite'] &= definite
        atom['native_states'].append(STATES[entity['label']])
        modifier_hash = digest(['native_observation_modifier_token_set', modifier_tokens])
        atom['modifier_set_hashes'].append(modifier_hash)
        atom['occurrences'].append({'entity_id': entity_id, 'word_start': entity['start_ix'],
            'word_end': entity['end_ix'], 'native_state': STATES[entity['label']],
            'modifier_set_sha256': modifier_hash})
    for atom in atoms.values():
        atom['native_states'] = sorted(set(atom['native_states']))
        atom['modifier_set_hashes'] = sorted(set(atom['modifier_set_hashes']))
        atom['occurrences'].sort(key=lambda r: int(r['entity_id']))
    return {'schema_version': VERSION + '-report', 'native_graph_sha256': digest(graph),
        'tokenized_text_sha256': metadata['tokenized_text_sha256'],
        'native_entity_count': metadata['entity_count'],
        'excluded_measurement_entities': metadata['measurement_entity_count'],
        'observation_modifiers_not_counted_as_findings': len(modifiers),
        'unhandled_native_relations': unhandled, 'atoms': {k: atoms[k] for k in sorted(atoms)},
        'current_patient_temporal_scope_verified': False, 'clinical_qualified': False,
        'no_finding_expanded_to_negative_labels': False}


def compare(left, right):
    require(all(isinstance(side, dict) and side.get('schema_version') == VERSION + '-report'
                and side['clinical_qualified'] is False and
                side['current_patient_temporal_scope_verified'] is False
                for side in (left, right)), 'literal_unqualified_evidence_contract_required')
    counts = dict.fromkeys(RELATIONS, 0)
    details, modifier_differences = [], 0
    for atom_id in sorted(set(left['atoms']) | set(right['atoms'])):
        a, b = left['atoms'].get(atom_id), right['atoms'].get(atom_id)
        if a is None or b is None:
            relation = 'unmentioned_in_left' if a is None else 'unmentioned_in_right'
        elif not (a['anatomy_context_all_definite'] and b['anatomy_context_all_definite']):
            relation = 'anatomy_context_not_definite'
        elif len(a['native_states']) != 1 or len(b['native_states']) != 1 or \
                'uncertain' in a['native_states'] + b['native_states']:
            relation = 'uncertain_or_mixed_state'
        elif a['native_states'] == b['native_states']:
            relation = 'native_state_agreement'
        else:
            relation = 'explicit_polarity_opposition_proposal'
        different_modifiers = bool(a and b and a['modifier_set_hashes'] != b['modifier_set_hashes'])
        modifier_differences += different_modifiers
        counts[relation] += 1
        details.append({'atom_id': atom_id, 'relation': relation,
            'left_native_states': a['native_states'] if a else None,
            'right_native_states': b['native_states'] if b else None,
            'left_entity_ids': [r['entity_id'] for r in a['occurrences']] if a else [],
            'right_entity_ids': [r['entity_id'] for r in b['occurrences']] if b else [],
            'modifier_token_sets_differ': different_modifiers,
            'clinical_contradiction_verified': False})
    contexts = []
    for side in (left, right):
        by_concept = defaultdict(set)
        for atom in side['atoms'].values():
            by_concept[atom['concept_sha256']].add(atom['anatomy_context_sha256'])
        contexts.append(by_concept)
    differing_contexts = [concept for concept in sorted(set(contexts[0]) & set(contexts[1]))
                          if contexts[0][concept] != contexts[1][concept]]
    return {'schema_version': VERSION + '-pair', 'status': 'complete',
        'left_native_graph_sha256': left['native_graph_sha256'],
        'right_native_graph_sha256': right['native_graph_sha256'],
        'union_literal_atoms': len(details), 'counts': counts,
        'modifier_token_set_difference_atoms': modifier_differences,
        'different_anatomy_context_concepts': len(differing_contexts),
        'different_anatomy_context_concept_hashes': differing_contexts,
        'details': details, 'clinical_qualified': False,
        'current_patient_temporal_scope_verified': False, 'independent_truth_votes': False,
        'missing_mention_means_negative': False, 'selection_changed': False,
        'clinical_score': None, 'regeneration_authorized': False}


def candidate_overlay(plan, extracted, comparisons):
    """Bind dependent peer counts to exact immutable report/image identities."""
    require(len(comparisons) == len(plan['pairs']) and set(extracted) ==
            {g['graph_id'] for g in plan['graphs']}, 'complete_attempted_graph_and_pair_inventory_required')
    graph_ids = {g['report_sha256']: g['graph_id'] for g in plan['graphs']}
    peers = defaultdict(list)
    for expected, row in zip(plan['pairs'], comparisons):
        require(all(row.get(k) == v for k, v in expected.items()), 'exact_same_image_pair_lineage_required')
        available = all(extracted[row[k]] is not None for k in ('left_graph_id', 'right_graph_id'))
        require(row['status'] == ('complete' if available else 'unavailable_graph'), 'pair_availability_mismatch')
        require((isinstance(row.get('counts'), dict) and set(row['counts']) == set(RELATIONS)
                 and all(type(v) is int and v >= 0 for v in row['counts'].values())
                 and sum(row['counts'].values()) == row['union_literal_atoms']) if available else
                row.get('counts') is None, 'complete_literal_counts_or_unavailable_null_required')
        if available:
            require(row['clinical_qualified'] is False and row['regeneration_authorized'] is False
                    and row['clinical_score'] is None and
                    all(row[side + '_native_graph_sha256'] == extracted[row[side + '_graph_id']]['native_graph_sha256']
                        for side in ('left', 'right')), 'unqualified_hash_bound_pair_evidence_required')
        for side in ('left', 'right'):
            peers[row[side + '_triple_candidate_id']].append((side, row))
    output = []
    for candidate in plan['candidates']:
        current = extracted[graph_ids[candidate['report_sha256']]]
        rows = peers[candidate['triple_candidate_id']]
        require(len(rows) == 3, 'three_attempted_same_image_peers_required')
        available = [(side, row) for side, row in rows if row['status'] == 'complete']
        def total(name):
            return sum(row['counts'][name] for _, row in available) if available else None
        atoms = list(current['atoms'].values()) if current is not None else None
        item = {'triple_candidate_id': candidate['triple_candidate_id'],
            'rg_literal_status': 'complete' if current is not None else 'unavailable_graph',
            'rg_literal_atom_count': len(atoms) if atoms is not None else None,
            'rg_literal_attempted_peers': 3, 'rg_literal_available_peers': len(available),
            'rg_literal_state_agreement_occurrences': total('native_state_agreement'),
            'rg_literal_opposition_proposal_occurrences': total('explicit_polarity_opposition_proposal'),
            'rg_literal_uncertain_comparison_occurrences':
                sum(row['counts']['uncertain_or_mixed_state'] + row['counts']['anatomy_context_not_definite']
                    for _, row in available) if available else None,
            'rg_literal_own_atoms_unmentioned_by_peer_occurrences':
                sum(row['counts']['unmentioned_in_right' if side == 'left' else 'unmentioned_in_left']
                    for side, row in available) if available else None,
            'rg_literal_peer_atoms_unmentioned_by_self_occurrences':
                sum(row['counts']['unmentioned_in_left' if side == 'left' else 'unmentioned_in_right']
                    for side, row in available) if available else None,
            'rg_literal_anatomy_context_difference_occurrences':
                sum(row['different_anatomy_context_concepts'] for _, row in available) if available else None,
            'rg_literal_modifier_difference_occurrences':
                sum(row['modifier_token_set_difference_atoms'] for _, row in available) if available else None,
            'rg_literal_clinical_score': None, 'rg_literal_scope_verified': False,
            'rg_literal_clinical_qualified': False, 'rg_literal_regeneration_authorized': False}
        output.append(item)
    return output

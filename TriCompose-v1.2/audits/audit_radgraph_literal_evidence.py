"""Independent source-bound literal atom, pair and old-cell replay.

Uses only cached fully synthetic graphs internally. No model, source patient
record, generated report file, image, weights, external API or selector.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radgraph_bank_runs/native_pool960_12714150_001'
RUN = BASE / 'radgraph_literal_evidence_runs/literal_pool960_12714150_001'
OUT = BASE / 'radgraph_literal_evidence_audits/literal_pool960_12714150_001'
STATES = {'Observation::definitely present': 'positive',
          'Observation::definitely absent': 'negative', 'Observation::uncertain': 'uncertain'}
RELATIONS = ('native_state_agreement', 'explicit_polarity_opposition_proposal',
             'uncertain_or_mixed_state', 'anatomy_context_not_definite',
             'unmentioned_in_left', 'unmentioned_in_right')


def require(condition):
    if not condition:
        raise ValueError('literal_evidence_audit_failed')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    require(sha256(RUN / 'manifest.json') ==
            'da1bfb5c82f1ce51d000843ee11471c6edddaafc20a5ab075227a4c675eb57c9')
    receipt = json.loads((RUN / 'manifest.json').read_text())
    require(sha256(SOURCE / 'manifest.json') == receipt['source_manifest_sha256'])
    hash_checks = 0
    for pin in receipt['pins']:
        require(sha256(ROOT / pin['path']) == pin['sha256'])
        hash_checks += 1
    for pin in receipt['artifacts']:
        path = RUN / pin['path']
        require(path.resolve().is_relative_to(RUN) and sha256(path) == pin['sha256'])
        hash_checks += 1
    source_plan = json.loads((SOURCE / 'frozen_plan.json').read_text())
    native = json.loads((SOURCE / 'native_graphs.json').read_text())
    require(native['scope'] == 'synthetic_only')
    literal = json.loads((RUN / 'graph_literal_evidence.json').read_text())
    require(len(literal) == len(source_plan['graphs']) == 428)
    by_graph, atom_checks, occurrence_checks = {}, 0, 0
    for row, request in zip(literal, source_plan['graphs']):
        require((row['graph_id'], row['report_sha256']) == (request['graph_id'], request['report_sha256']))
        require(row['status'] == 'complete' and row['failure_type'] is None)
        graph = native['graphs'][row['graph_id']]
        entities, evidence = graph['entities'], row['evidence']
        require(evidence['native_graph_sha256'] == digest(graph))
        require(evidence['tokenized_text_sha256'] == hashlib.sha256(graph['text'].encode()).hexdigest())
        incoming, attributes, unhandled = defaultdict(set), set(), 0
        for node, entity in entities.items():
            for relation, destination in entity['relations']:
                if relation == 'modify':
                    incoming[destination].add(node)
                    if entity['label'] in STATES and entities[destination]['label'].startswith('Observation::'):
                        attributes.add(node)
                elif relation != 'located_at':
                    unhandled += 1
        def closure(node):
            current = {node}
            while True:
                expanded = current | {child for parent in current for child in incoming[parent]}
                if expanded == current:
                    return current
                current = expanded
        expected = defaultdict(list)
        for node, entity in entities.items():
            if entity['label'] not in STATES or node in attributes:
                continue
            anatomy = set()
            for relation, destination in entity['relations']:
                if relation == 'located_at' and entities[destination]['label'].startswith('Anatomy::'):
                    anatomy |= closure(destination)
            normalized = lambda nodes: sorted({' '.join(entities[n]['tokens'].casefold().split()) for n in nodes})
            concept_hash = digest(['literal_observation', ' '.join(entity['tokens'].casefold().split())])
            context_hash = digest(['native_anatomy_token_set', normalized(anatomy)])
            modifier_hash = digest(['native_observation_modifier_token_set', normalized(closure(node)-{node})])
            key = 'atom_' + digest([concept_hash, context_hash])
            expected[key].append({'entity_id': node, 'word_start': entity['start_ix'],
                'word_end': entity['end_ix'], 'native_state': STATES[entity['label']],
                'modifier_set_sha256': modifier_hash, 'concept': concept_hash, 'context': context_hash,
                'anatomy': bool(anatomy), 'definite': all(entities[n]['label'] == 'Anatomy::definitely present'
                                                        for n in anatomy)})
        require(set(evidence['atoms']) == set(expected))
        for key, occurrences in expected.items():
            atom = evidence['atoms'][key]
            require(atom['atom_id'] == key and atom['concept_sha256'] == occurrences[0]['concept']
                    and atom['anatomy_context_sha256'] == occurrences[0]['context'])
            require(atom['has_explicit_anatomy'] == occurrences[0]['anatomy'] and
                    atom['anatomy_context_all_definite'] == all(o['definite'] for o in occurrences))
            require(atom['native_states'] == sorted({o['native_state'] for o in occurrences}) and
                    atom['modifier_set_hashes'] == sorted({o['modifier_set_sha256'] for o in occurrences}))
            require(atom['occurrences'] == sorted([{k: o[k] for k in
                ('entity_id', 'word_start', 'word_end', 'native_state', 'modifier_set_sha256')}
                 for o in occurrences], key=lambda r: int(r['entity_id'])))
            atom_checks += 1
            occurrence_checks += len(occurrences)
        require(evidence['observation_modifiers_not_counted_as_findings'] == len(attributes))
        require(evidence['excluded_measurement_entities'] == sum('::measurement::' in e['label'] for e in entities.values()))
        require(evidence['unhandled_native_relations'] == unhandled)
        require(evidence['clinical_qualified'] is False and evidence['current_patient_temporal_scope_verified'] is False)
        require(evidence['no_finding_expanded_to_negative_labels'] is False)
        by_graph[row['graph_id']] = evidence
    pairs = json.loads((RUN / 'same_image_literal_evidence.json').read_text())
    require(len(pairs) == len(source_plan['pairs']) == 1440)
    peers, model_counts = defaultdict(list), defaultdict(lambda: dict(pairs=0, pairs_with_opposition_proposals=0,
                                                                     opposition_proposal_atoms=0))
    relation_totals, detail_checks = Counter(), 0
    for row, planned in zip(pairs, source_plan['pairs']):
        require(all(row[k] == v for k, v in planned.items()) and row['status'] == 'complete')
        a, b = by_graph[row['left_graph_id']], by_graph[row['right_graph_id']]
        require(row['left_native_graph_sha256'] == a['native_graph_sha256'] and
                row['right_native_graph_sha256'] == b['native_graph_sha256'])
        expected_counts, modifier_diffs = Counter(), 0
        details = {r['atom_id']: r for r in row['details']}
        require(len(details) == row['union_literal_atoms'] == len(set(a['atoms']) | set(b['atoms'])))
        for key in set(a['atoms']) | set(b['atoms']):
            left, right = a['atoms'].get(key), b['atoms'].get(key)
            if left is None or right is None:
                relation = 'unmentioned_in_left' if left is None else 'unmentioned_in_right'
            elif not all(s['anatomy_context_all_definite'] for s in (left, right)):
                relation = 'anatomy_context_not_definite'
            elif any(len(s['native_states']) != 1 or s['native_states'] == ['uncertain'] for s in (left, right)):
                relation = 'uncertain_or_mixed_state'
            else:
                relation = ('native_state_agreement' if left['native_states'] == right['native_states']
                            else 'explicit_polarity_opposition_proposal')
            detail = details[key]
            different_modifiers = bool(left and right and left['modifier_set_hashes'] != right['modifier_set_hashes'])
            require(detail['relation'] == relation and detail['modifier_token_sets_differ'] == different_modifiers)
            require(detail['left_native_states'] == (left['native_states'] if left else None) and
                    detail['right_native_states'] == (right['native_states'] if right else None))
            require(detail['left_entity_ids'] == ([o['entity_id'] for o in left['occurrences']] if left else []) and
                    detail['right_entity_ids'] == ([o['entity_id'] for o in right['occurrences']] if right else []))
            require(detail['clinical_contradiction_verified'] is False)
            expected_counts[relation] += 1
            modifier_diffs += different_modifiers
            detail_checks += 1
        require(row['counts'] == {name: expected_counts[name] for name in RELATIONS})
        require(row['modifier_token_set_difference_atoms'] == modifier_diffs)
        context_sets = []
        for side in (a, b):
            groups = defaultdict(set)
            for atom in side['atoms'].values():
                groups[atom['concept_sha256']].add(atom['anatomy_context_sha256'])
            context_sets.append(groups)
        different_contexts = sorted(k for k in set(context_sets[0]) & set(context_sets[1])
                                    if context_sets[0][k] != context_sets[1][k])
        require(row['different_anatomy_context_concept_hashes'] == different_contexts and
                row['different_anatomy_context_concepts'] == len(different_contexts))
        for flag in ('clinical_qualified', 'current_patient_temporal_scope_verified', 'independent_truth_votes',
                     'missing_mention_means_negative', 'selection_changed', 'regeneration_authorized'):
            require(row[flag] is False)
        require(row['clinical_score'] is None)
        relation_totals.update(expected_counts)
        for side in ('left', 'right'):
            peers[row[side+'_triple_candidate_id']].append((side, row))
        model = '|'.join(sorted((row['left_report_model_id'], row['right_report_model_id'])))
        model_counts[model]['pairs'] += 1
        model_counts[model]['pairs_with_opposition_proposals'] += bool(expected_counts['explicit_polarity_opposition_proposal'])
        model_counts[model]['opposition_proposal_atoms'] += expected_counts['explicit_polarity_opposition_proposal']
    original, current = rows(SOURCE / 'candidate_score_table.csv'), rows(RUN / 'candidate_score_table.csv')
    require(len(original) == len(current) == 960)
    graph_by_hash = {r['report_sha256']: r['evidence'] for r in literal}
    cell_checks = 0
    for old, new in zip(original, current):
        require(all(new[k] == v for k, v in old.items()))
        source = graph_by_hash[old['report_sha256']]
        compared = peers[old['triple_candidate_id']]
        require(len(compared) == 3)
        expected = {'rg_literal_status': 'complete', 'rg_literal_atom_count': len(source['atoms']),
            'rg_literal_attempted_peers': 3, 'rg_literal_available_peers': 3,
            'rg_literal_state_agreement_occurrences': sum(r['counts']['native_state_agreement'] for _, r in compared),
            'rg_literal_opposition_proposal_occurrences': sum(r['counts']['explicit_polarity_opposition_proposal'] for _, r in compared),
            'rg_literal_uncertain_comparison_occurrences': sum(r['counts']['uncertain_or_mixed_state']+
                r['counts']['anatomy_context_not_definite'] for _, r in compared),
            'rg_literal_own_atoms_unmentioned_by_peer_occurrences': sum(
                r['counts']['unmentioned_in_right' if side == 'left' else 'unmentioned_in_left'] for side, r in compared),
            'rg_literal_peer_atoms_unmentioned_by_self_occurrences': sum(
                r['counts']['unmentioned_in_left' if side == 'left' else 'unmentioned_in_right'] for side, r in compared),
            'rg_literal_anatomy_context_difference_occurrences': sum(r['different_anatomy_context_concepts'] for _, r in compared),
            'rg_literal_modifier_difference_occurrences': sum(r['modifier_token_set_difference_atoms'] for _, r in compared),
            'rg_literal_clinical_score': None, 'rg_literal_scope_verified': False,
            'rg_literal_clinical_qualified': False, 'rg_literal_regeneration_authorized': False}
        require(set(new)-set(old) == set(expected))
        require(all(new[k] == ('' if v is None else str(v)) for k, v in expected.items()))
        cell_checks += len(old)+len(expected)
    summary = json.loads((RUN / 'summary.json').read_text())
    require(summary['pair_relation_atom_occurrences'] == {name: relation_totals[name] for name in RELATIONS})
    require(summary['report_model_pair_diagnostics'] == model_counts)
    oppositions = [r for r in pairs if r['counts']['explicit_polarity_opposition_proposal']]
    require(summary['pairs_with_opposition_proposals'] == len(oppositions))
    require(summary['image_slots_with_opposition_proposals'] == len({r['cxr_candidate_id'] for r in oppositions}))
    require(summary['cases_with_opposition_proposals'] == len({r['case_id'] for r in oppositions}))
    require(summary['candidate_slots_with_opposition_proposals'] == sum(
        any(r['counts']['explicit_polarity_opposition_proposal'] for _, r in peer) for peer in peers.values()))
    require(summary['source_columns_preserved'] == len(original[0]) and summary['new_columns'] == 15)
    require(summary['different_anatomy_context_concept_occurrences'] == sum(r['different_anatomy_context_concepts'] for r in pairs))
    require(summary['different_modifier_token_set_atom_occurrences'] == sum(r['modifier_token_set_difference_atoms'] for r in pairs))
    for flag in ('source_patient_inputs_read', 'clinical_qualified', 'selection_changed', 'regeneration_authorized'):
        require(summary[flag] is False)
    require(summary['new_model_calls'] == summary['new_slurm_submissions'] == 0)
    permissions = 0
    for path in (RUN, *RUN.iterdir()):
        require(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660))
        permissions += 1
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    payload = {'status': 'passed', 'source_manifest_sha256': sha256(RUN / 'manifest.json'),
        'source_code_output_hash_checks': hash_checks, 'native_graphs_replayed': len(literal),
        'literal_atoms_reconstructed': atom_checks, 'source_bound_entity_occurrences': occurrence_checks,
        'same_image_pairs_replayed': len(pairs), 'pair_atom_relations_replayed': detail_checks,
        'original_plus_new_candidate_cells_replayed': cell_checks, 'protected_permissions_checked': permissions,
        'new_model_calls': 0, 'source_patient_inputs_read': False, 'clinical_qualified': False, 'selection_changed': False}
    with (OUT / 'audit.json').open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status': 'literal_evidence_audit_passed', 'audit_sha256': sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)

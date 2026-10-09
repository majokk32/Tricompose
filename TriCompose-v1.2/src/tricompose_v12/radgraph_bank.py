"""Pure synthetic-bank inventory and same-image report agreement.

No inference, medical truth, thresholds, weighting fit, or winner changes.
All native reward values must come from unchanged official RadGraph code.
"""
from collections import defaultdict
from itertools import combinations
import re

from .radgraph_reference_contract import METRICS, finite_score, require

SCHEMA = 'tricompose-radgraph-same-image-bank-v1'
CXR_MODELS = ('chexgenbench_pixart', 'chexgenbench_sana', 'roentgen_v2')
REPORT_MODELS = ('chexagent2', 'cxrmate_single', 'llavarad', 'maira2')
FIELDS = ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'cxr_model_id',
          'cxr_seed', 'report_candidate_id', 'report_model_id', 'report_path',
          'report_sha256', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')
HASH = re.compile(r'[0-9a-f]{64}\Z')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,255}\Z')
GRAPH_STATUSES = ('complete', 'empty_input', 'failed_unavailable')
POLICY = {'clinical_qualified': False, 'image_factuality_verified': False,
          'ehr_consistency_verified': False, 'selection_changed': False,
          'regeneration_authorized': False, 'new_training': False,
          'independent_model_votes': False, 'reference_is_real_target': False,
          'missing_or_empty_is_zero': False,
          'identical_text_is_independent_confirmation': False}


def inventory(rows, *, expected_cases=80):
    require(type(expected_cases) is int and 1 <= expected_cases <= 80 and
            isinstance(rows, list) and len(rows) == expected_cases * 12,
            'complete_bounded_synthetic_grid_required')
    cases, images, hashes, triples, report_ids = {}, {}, {}, set(), set()
    projected = []
    for source in sorted(rows, key=lambda r: r['triple_candidate_id']):
        require(all(field in source for field in FIELDS), 'bank_lineage_fields_required')
        row = {field: source[field] for field in FIELDS}
        require(all(isinstance(row[k], str) and ID.fullmatch(row[k]) for k in
                    ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'report_candidate_id')),
                'opaque_bank_ids_required')
        require(all(isinstance(row[k], str) and HASH.fullmatch(row[k]) for k in
                    ('report_sha256', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')),
                'bank_sha256_required')
        require(isinstance(row['report_path'], str) and row['report_path'],
                'explicit_report_artifact_path_required')
        require(row['cxr_model_id'] in CXR_MODELS and row['report_model_id'] in REPORT_MODELS
                and str(row['cxr_seed']) == '0', 'historical_three_by_four_seed_zero_required')
        require(row['triple_candidate_id'] not in triples and
                row['report_candidate_id'] not in report_ids, 'unique_candidate_slots_required')
        triples.add(row['triple_candidate_id'])
        report_ids.add(row['report_candidate_id'])
        anchor = (row['ehr_sha256'], row['ehr_facts_sha256'])
        case = cases.setdefault(row['case_id'], {'anchor': anchor, 'slots': set()})
        slot = (row['cxr_model_id'], row['report_model_id'])
        require(case['anchor'] == anchor and slot not in case['slots'],
                'fixed_ehr_unique_model_grid_required')
        case['slots'].add(slot)
        image = images.setdefault(row['cxr_candidate_id'], {'identity':
            (row['case_id'], row['cxr_model_id'], row['cxr_sha256']), 'reports': []})
        require(image['identity'] == (row['case_id'], row['cxr_model_id'], row['cxr_sha256']),
                'same_image_lineage_required')
        image['reports'].append(row)
        hashes.setdefault(row['report_sha256'], []).append(row)
        projected.append(row)
    expected = {(c, r) for c in CXR_MODELS for r in REPORT_MODELS}
    require(len(cases) == expected_cases and len(images) == expected_cases * 3 and
            all(case['slots'] == expected for case in cases.values()),
            'complete_case_inventory_required')
    graphs, lookup = [], {}
    for index, digest in enumerate(sorted(hashes)):
        members = hashes[digest]
        graph_id = f'report_graph_{index:04d}'
        graphs.append({'graph_id': graph_id, 'report_sha256': digest,
                       'source_path': members[0]['report_path'],
                       'candidate_slots': len(members)})
        lookup[digest] = graph_id
    pairs = []
    for image_id in sorted(images):
        members = sorted(images[image_id]['reports'], key=lambda r: r['report_model_id'])
        require(len(members) == 4 and len({r['report_model_id'] for r in members}) == 4,
                'four_experts_per_image_required')
        for left, right in combinations(members, 2):
            pairs.append({'pair_id': f'pair_{len(pairs):04d}',
                          'case_id': left['case_id'], 'cxr_candidate_id': image_id,
                          'left_triple_candidate_id': left['triple_candidate_id'],
                          'right_triple_candidate_id': right['triple_candidate_id'],
                          'left_report_model_id': left['report_model_id'],
                          'right_report_model_id': right['report_model_id'],
                          'left_graph_id': lookup[left['report_sha256']],
                          'right_graph_id': lookup[right['report_sha256']],
                          'identical_report_bytes': left['report_sha256'] == right['report_sha256']})
    return {'schema_version': SCHEMA + '-plan', 'policy': dict(POLICY),
            'case_count': len(cases), 'image_slots': len(images), 'candidate_slots': len(rows),
            'unique_report_graphs': len(graphs), 'same_image_pairs': len(pairs),
            'graphs': graphs, 'candidates': projected, 'pairs': pairs,
            'cohort_role': 'historical_inspected_development_bank'}


def summarize(plan, graphs, pair_rows):
    require(isinstance(graphs, list) and len(graphs) == len(plan['graphs']),
            'all_attempted_graphs_required')
    lookup = {}
    for graph, request in zip(graphs, plan['graphs']):
        require(graph['graph_id'] == request['graph_id'] and
                graph['report_sha256'] == request['report_sha256'] and
                graph['status'] in GRAPH_STATUSES, 'graph_receipt_inventory_required')
        require(graph['graph_id'] not in lookup, 'unique_graph_receipt_required')
        require((isinstance(graph.get('metadata'), dict) and graph.get('failure_reason') is None)
                if graph['status'] == 'complete' else
                (graph.get('metadata') is None and isinstance(graph.get('failure_reason'), str)),
                'complete_metadata_or_explicit_unavailable_required')
        lookup[graph['graph_id']] = graph
    require(len(pair_rows) == len(plan['pairs']), 'all_attempted_pairs_required')
    peers = defaultdict(list)
    for row, pair in zip(pair_rows, plan['pairs']):
        require(all(row.get(k) == pair[k] for k in pair), 'same_image_pair_identity_required')
        eligible = all(lookup[pair[k]]['status'] == 'complete' for k in
                       ('left_graph_id', 'right_graph_id'))
        require(row['status'] == ('complete' if eligible else 'unavailable_graph'),
                'pair_availability_must_follow_graphs')
        require(isinstance(row['scores'], dict) and set(row['scores']) == set(METRICS),
                'three_official_reward_components_required')
        require(all(finite_score(v) if eligible else v is None for v in row['scores'].values()),
                'finite_score_or_unavailable_null_required')
        for side in ('left', 'right'):
            peers[pair[side + '_triple_candidate_id']].append(row)
    by_hash = {graph['report_sha256']: graph for graph in graphs}
    output = []
    for candidate in plan['candidates']:
        graph = by_hash[candidate['report_sha256']]
        comparisons = peers[candidate['triple_candidate_id']]
        require(len(comparisons) == 3, 'three_attempted_peers_per_candidate_required')
        available = [row for row in comparisons if row['status'] == 'complete']
        row = {key: candidate[key] for key in
               ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'report_candidate_id',
                'report_model_id', 'report_sha256')}
        row.update({'radgraph_graph_id': graph['graph_id'], 'radgraph_status': graph['status'],
                    'radgraph_entity_count': (graph['metadata'] or {}).get('entity_count'),
                    'radgraph_relation_count': (graph['metadata'] or {}).get('relation_count'),
                    'radgraph_attempted_peers': 3, 'radgraph_available_peers': len(available),
                    'radgraph_same_text_peers': sum(p['identical_report_bytes'] for p in comparisons),
                    'radgraph_agreement_clinical_qualified': False})
        for metric in METRICS:
            row['radgraph_peer_mean_' + metric.removeprefix('radgraph_')] = (
                sum(p['scores'][metric] for p in available) / len(available) if available else None)
        output.append(row)
    return output

"""Independent bank receipt, source preservation and official-reward replay.

No neural inference, source MIMIC reads, medical adjudication, or new winners.
The synthetic graphs are consumed internally; public output is status/hash only.
"""
from collections import defaultdict
import csv
import hashlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'radgraph_bank_runs/native_pool960_12714150_001'
OUT = BASE / 'radgraph_bank_audits/native_pool960_12714150_001'
OLD = BASE / 'deliverables/first_version_12714150_001/candidate_index.csv'
METRICS = ('radgraph_entity_f1', 'radgraph_relation_presence_f1', 'radgraph_full_relation_f1')


def require(condition):
    if not condition:
        raise ValueError('radgraph_bank_audit_failed')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    receipt = json.loads((RUN / 'manifest.json').read_text())
    pins = receipt['pins']
    hash_checks = 0
    for pin in pins:
        source = ROOT / pin['path']
        require(source.resolve().is_relative_to(ROOT) and sha256(source) == pin['sha256'])
        hash_checks += 1
    for artifact in receipt['artifacts']:
        path = RUN / artifact['path']
        require(path.resolve().is_relative_to(RUN) and sha256(path) == artifact['sha256'])
        hash_checks += 1
    for path, digest in receipt['source_report_hashes'].items():
        source = Path(path)
        require(source.resolve().is_relative_to(ROOT / 'artifacts/protected')
                and sha256(source) == digest)
        hash_checks += 1
    old = csv_rows(OLD)
    new = csv_rows(RUN / 'candidate_score_table.csv')
    require(len(old) == len(new) == 960)
    for original, current in zip(old, new):
        require(all(current[key] == value for key, value in original.items()))
    # Scores, selection flags, source paths and original ordering are untouched.
    same_images = defaultdict(list)
    for row in old:
        same_images[row['cxr_candidate_id']].append(row)
    require(len(same_images) == 240 and all(len(v) == 4 for v in same_images.values()))
    expected_pairs = {}
    for image, rows in same_images.items():
        for left, right in combinations(sorted(rows, key=lambda r: r['report_model_id']), 2):
            expected_pairs[(left['triple_candidate_id'], right['triple_candidate_id'])] = (image, left, right)
    pairs = json.loads((RUN / 'same_image_agreement.json').read_text())
    graphs = json.loads((RUN / 'graph_receipts.json').read_text())
    native = json.loads((RUN / 'native_graphs.json').read_text())
    require(native['scope'] == 'synthetic_only')
    by_id = {row['graph_id']: row for row in graphs}
    by_hash = {row['report_sha256']: row for row in graphs}
    require(len(graphs) == len(by_id) == len(by_hash) == 428)
    require(len(pairs) == len(expected_pairs) == 1440)
    require(set(native['graphs']) == {g['graph_id'] for g in graphs if g['status'] == 'complete'})
    graph_checks = 0
    for graph in graphs:
        if graph['status'] == 'complete':
            prediction = native['graphs'][graph['graph_id']]
            meta = graph['metadata']
            counts = {label: 0 for label in meta['native_label_counts']}
            for entity in prediction['entities'].values():
                require(entity['label'] in counts)
                counts[entity['label']] += 1
            require(counts == meta['native_label_counts'])
            require(meta['entity_count'] == len(prediction['entities']))
            require(meta['relation_count'] == sum(len(e['relations']) for e in prediction['entities'].values()))
            require(meta['tokenized_text_sha256'] == hashlib.sha256(prediction['text'].encode()).hexdigest())
            require(meta['scope_verified'] is False)
            graph_checks += 5
        else:
            require(graph['metadata'] is None and graph['failure_reason'])
    spec = importlib.util.spec_from_file_location('official_bank_reward', ROOT / 'RadGraph/radgraph/rewards.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    peer_scores = defaultdict(list)
    pair_checks = 0
    seen = set()
    for pair in pairs:
        key = (pair['left_triple_candidate_id'], pair['right_triple_candidate_id'])
        require(key in expected_pairs and key not in seen)
        seen.add(key)
        image, left, right = expected_pairs[key]
        require(pair['cxr_candidate_id'] == image and pair['case_id'] == left['case_id'] == right['case_id'])
        left_graph, right_graph = by_hash[left['report_sha256']], by_hash[right['report_sha256']]
        require(pair['left_graph_id'] == left_graph['graph_id'] and pair['right_graph_id'] == right_graph['graph_id'])
        require(pair['identical_report_bytes'] == (left['report_sha256'] == right['report_sha256']))
        available = left_graph['status'] == right_graph['status'] == 'complete'
        require(pair['status'] == ('complete' if available else 'unavailable_graph'))
        if available:
            values = official.compute_reward(native['graphs'][left_graph['graph_id']],
                                             native['graphs'][right_graph['graph_id']], 'all')
            for name, value in zip(METRICS, values):
                require(abs(pair['scores'][name] - value) < 1e-12)
                pair_checks += 1
        else:
            require(all(value is None for value in pair['scores'].values()))
        for candidate in key:
            peer_scores[candidate].append(pair)
    require(seen == set(expected_pairs))
    mean_checks = 0
    for row in new:
        graph = by_hash[row['report_sha256']]
        require(row['radgraph_graph_id'] == graph['graph_id'] and row['radgraph_status'] == graph['status'])
        peer = peer_scores[row['triple_candidate_id']]
        available = [pair for pair in peer if pair['status'] == 'complete']
        require(len(peer) == 3 and int(row['radgraph_attempted_peers']) == 3
                and int(row['radgraph_available_peers']) == len(available)
                and int(row['radgraph_same_text_peers']) == sum(p['identical_report_bytes'] for p in peer))
        require(row['radgraph_agreement_clinical_qualified'] == 'False')
        for metric in METRICS:
            cell = row['radgraph_peer_mean_' + metric.removeprefix('radgraph_')]
            if available:
                require(abs(float(cell) - sum(p['scores'][metric] for p in available) / len(available)) < 1e-12)
            else:
                require(cell == '')
            mean_checks += 1
    permission_checks = 0
    for path in [RUN, *RUN.iterdir()]:
        require(path.stat().st_gid in (96293, 65534) and
                path.stat().st_mode & 0o7777 == (0o2770 if path.is_dir() else 0o660))
        permission_checks += 1
    require(receipt['policy']['clinical_qualified'] is False and
            receipt['policy']['selection_changed'] is False)
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    with (OUT / 'audit.json').open('x') as stream:
        json.dump({'status': 'passed', 'source_manifest_sha256': sha256(RUN / 'manifest.json'),
                   'source_output_hash_checks': hash_checks, 'graph_metadata_checks': graph_checks,
                   'original_rows_and_columns_preserved': 960, 'same_image_pair_inventory': 1440,
                   'official_reward_components_recomputed': pair_checks,
                   'candidate_peer_means_recomputed': mean_checks,
                   'protected_permissions_checked': permission_checks,
                   'new_neural_model_calls': 0, 'clinical_qualified': False,
                   'selection_changed': False}, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status': 'radgraph_bank_audit_passed', 'audit_sha256': sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)

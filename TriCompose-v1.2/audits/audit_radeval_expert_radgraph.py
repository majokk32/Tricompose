"""Independent private source/count/reward replay, no neural model calls.

Real texts/reader/source keys are transient inside the approved CPU job.
Public output is status/hash only; derived audit contains counts, not content.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

from tricompose_v12.report_metric_alignment import correlation, clustered_spearman_interval

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'report_metric_sources/radeval_expert_source_12714150_001'
RUN = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
OUT = BASE / 'radeval_expert_radgraph_audits/expert_radgraph_12714150_001'
METRICS = ('radgraph_entity_f1', 'radgraph_relation_presence_f1', 'radgraph_full_relation_f1')
SEVERITIES = ('clinically_significant', 'clinically_insignificant')
LABELS = (
    'false prediction of finding', 'omission of finding',
    'incorrect location position of finding', 'incorrect severity of finding',
    'mention of comparison that is not present in the reference',
    'omission of a change from a previous study',
    'inarticulate report grammar and readability issues',
)


def require(condition):
    if not condition:
        raise ValueError('expert_benchmark_audit_failed')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_counts(text):
    cells = {s: [None] * 7 for s in SEVERITIES}
    current, seen_headers, seen_categories = None, [], defaultdict(set)
    structural = False
    for line in text.splitlines():
        if not line.strip():
            continue
        header = re.fullmatch(r'[^a-zA-Z0-9]*(Significant|Insignificant)[^a-zA-Z0-9]*', line, re.I)
        if header:
            current = SEVERITIES[0 if header[1].lower() == 'significant' else 1]
            seen_headers.append(current)
            continue
        match = re.fullmatch(r'\s*([1-7])[.)]\s+(.+?)\s*:\s*(\d+)\s*[.;]?\s*', line)
        if match and current is not None:
            index = int(match[1]) - 1
            label = ' '.join(re.sub(r'[^a-z ]', ' ', match[2].lower()).split())
            if label != LABELS[index] or index in seen_categories[current]:
                structural = True
            seen_categories[current].add(index)
            cells[current][index] = int(match[3]) if int(match[3]) <= 1000 else None
        # Malformed numerical lines retain the initialized null category.
    if Counter(seen_headers) != Counter(SEVERITIES) or structural:
        cells = {s: [None] * 7 for s in SEVERITIES}
    return cells


def group_key(source):
    for strategy, pattern in (
        ('patient_folder', r'(?:^|[/\\])p([0-9]{6,})(?=[/\\]|$)'),
        ('chexpert_patient_folder', r'(?:^|[/\\])patient([0-9]+)(?=[/\\]|$)'),
        ('study_folder', r'(?:^|[/\\])s([0-9]{6,})(?=[/\\]|$)'),
    ):
        keys = set(re.findall(pattern, source))
        if len(keys) == 1:
            return strategy, next(iter(keys))
    return 'author_source_key', source.strip()


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    receipt = json.loads((RUN / 'manifest.json').read_text())
    hash_checks = 0
    for pin in receipt['pins']:
        path = ROOT / pin['path']
        require(path.resolve().is_relative_to(ROOT) and sha256(path) == pin['sha256'])
        hash_checks += 1
    for item in receipt['artifacts']:
        path = RUN / item['path']
        require(path.resolve().is_relative_to(RUN) and sha256(path) == item['sha256'])
        hash_checks += 1
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    data = (SOURCE / source_manifest['file']).read_bytes()
    require(len(data) == source_manifest['bytes'] == 691815)
    require(hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == source_manifest['git_blob'])
    with (SOURCE / source_manifest['file']).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    plan = json.loads((RUN / 'frozen_plan.json').read_text())['inventory']
    scores = json.loads((RUN / 'scores.json').read_text())
    graphs = json.loads((RUN / 'graph_receipts.json').read_text())
    native = json.loads((RUN / 'native_graphs.json').read_text())
    result = json.loads((RUN / 'evaluation.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    require(len(rows) == 208 and len(plan['records']) == len(scores) == 624)
    require(native['scope'] == 'authorized_real_reference_benchmark_only')
    source_ids, section_ids, groups = {}, {}, {}
    count_cells, pair_keys, report_texts, row_cells = defaultdict(list), {}, {}, []
    for row_index, row in enumerate(rows):
        source_id = source_ids.setdefault(row['images_path'].strip(), f'source_{len(source_ids):04d}')
        section_id = section_ids.setdefault(row['type'].strip().lower(), f'section_{len(section_ids):02d}')
        key = group_key(row['images_path'])
        group = groups.setdefault(key, f'group_{len(groups):04d}')
        reference = hashlib.sha256(row['ground_truth'].encode()).hexdigest()
        report_texts[reference] = row['ground_truth']
        for slot in range(1, 4):
            hypothesis = hashlib.sha256(row[f'prediction{slot}'].encode()).hexdigest()
            report_texts[hypothesis] = row[f'prediction{slot}']
            pair = (source_id, section_id, slot, reference, hypothesis)
            item_id = pair_keys.setdefault(pair, f'pair_{len(pair_keys):04d}')
            errors = independent_counts(row[f'annotation{slot}'])
            count_cells[item_id].append(errors)
            row_cells.append((item_id, row_index, errors))
            stored = plan['records'][int(item_id.split('_')[1])]
            require(stored['item_id'] == item_id and stored['source_group_id'] == group)
    require(len(groups) == plan['source_groups'] == 180 and len(source_ids) == plan['source_keys'] == 203)
    require(dict(Counter(k[0] for k in groups)) == plan['group_key_strategy_counts'])
    require(len(pair_keys) == 624 and len(report_texts) == len(graphs) == 762)
    count_checks = 0
    for stored, (item_id, row_index, errors) in zip(plan['annotation_cells'], row_cells):
        require(stored['item_id'] == item_id and stored['source_row_index'] == row_index
                and stored['errors'] == errors)
        count_checks += 14
    by_id = {g['graph_id']: g for g in graphs}
    require(len(by_id) == len(graphs) and set(native['graphs']) == {
        g['graph_id'] for g in graphs if g['status'] == 'complete'})
    graph_checks = 0
    for request, graph in zip(plan['graphs'], graphs):
        require(request['graph_id'] == graph['graph_id'] and request['text_sha256'] == graph['text_sha256']
                and graph['text_sha256'] in report_texts)
        if graph['status'] == 'complete':
            actual, meta = native['graphs'][graph['graph_id']], graph['metadata']
            require(meta['tokenized_text_sha256'] == hashlib.sha256(actual['text'].encode()).hexdigest())
            require(meta['entity_count'] == len(actual['entities']))
            require(meta['relation_count'] == sum(len(e['relations']) for e in actual['entities'].values()))
            require(dict(Counter(e['label'] for e in actual['entities'].values())) == {
                k: v for k, v in meta['native_label_counts'].items() if v})
            graph_checks += 4
        else:
            require(graph['metadata'] is None)
    spec = importlib.util.spec_from_file_location('expert_official_reward', ROOT / 'RadGraph/radgraph/rewards.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    replay_scores, primary_errors, secondary_errors = {}, {}, {}
    reward_checks = 0
    for record, score in zip(plan['records'], scores):
        require(record['item_id'] == score['item_id'])
        expected = {}
        cells = count_cells[record['item_id']]
        require(len(cells) == record['released_reader_cells'])
        for severity in SEVERITIES:
            expected[severity] = [sum(c[severity][i] for c in cells) / len(cells)
                if all(c[severity][i] is not None for c in cells) else None for i in range(7)]
        require(expected == record['errors'])
        count_checks += 14
        primary_errors[record['item_id']] = sum(expected[SEVERITIES[0]]) if None not in expected[SEVERITIES[0]] else None
        secondary_errors[record['item_id']] = sum(v for values in expected.values() for v in values) if all(
            v is not None for values in expected.values() for v in values) else None
        ref, hyp = record['reference_graph_id'], record['hypothesis_graph_id']
        available = by_id[ref]['status'] == by_id[hyp]['status'] == 'complete'
        require(score['status'] == ('complete' if available else 'unavailable_graph'))
        values = official.compute_reward(native['graphs'][hyp], native['graphs'][ref], 'all') if available else (None,)*3
        replay_scores[record['item_id']] = dict(zip(METRICS, values))
        for name, value in zip(METRICS, values):
            require(score['scores'][name] == value)
            reward_checks += 1
    require(sum(v is not None for v in primary_errors.values()) == 623)
    require(sum(v is not None for v in secondary_errors.values()) == 622)
    statistic_checks = 0
    for metric in METRICS:
        for outcome, errors in [('clinically_significant_total', primary_errors), ('all_errors_total', secondary_errors)]:
            values = [(r['source_group_id'], replay_scores[r['item_id']][metric], -errors[r['item_id']])
                for r in plan['records'] if replay_scores[r['item_id']][metric] is not None and errors[r['item_id']] is not None]
            correlation_result = correlation([(v[1], v[2]) for v in values])
            stored = result['metrics'][metric]['outcomes'][outcome]
            for name, value in correlation_result.items():
                require(stored[name] == value)
            require(stored['cluster_bootstrap'] == clustered_spearman_interval(values, 1000, 0))
            require(stored['paired_coverage'] == len(values)/624)
            statistic_checks += 1
    # CSV rows must reproduce the typed score and primary reference contracts.
    with (RUN / 'score_table.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    require(len(table) == 624)
    for row, score, record in zip(table, scores, plan['records']):
        require(row['item_id'] == score['item_id'] and row['source_group_id'] == record['source_group_id'])
        require(row['model_status'] == score['status'])
        for metric in METRICS:
            require(float(row[metric]) == score['scores'][metric] if score['status'] == 'complete' else row[metric] == '')
        expected = primary_errors[record['item_id']]
        require(float(row['clinically_significant_total']) == expected if expected is not None
                else row['clinically_significant_total'] == '')
    # No source path, reader name, or complete clinical text in metadata.
    leaked_checks = 0
    forbidden = {r['images_path'] for r in rows} | {r['annotator'] for r in rows}
    forbidden |= {text for text in report_texts.values() if len(text.strip()) >= 20}
    for path in RUN.iterdir():
        if path.suffix in ('.json', '.csv') and path.name != 'native_graphs.json':
            exported = path.read_text()
            require(not any(v and v in exported for v in forbidden))
            leaked_checks += 1
    permission_checks = 0
    for root in (SOURCE, RUN):
        for path in (root, *root.iterdir()):
            require(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 ==
                    (0o2770 if path.is_dir() else 0o660))
            permission_checks += 1
    require(summary['policy']['clinical_qualified'] is False and summary['policy']['selection_changed'] is False)
    require(summary['frozen_parameters'] is True and summary['network_disabled_during_scoring'] is True
            and summary['source_ehr_or_images_read'] is False)
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    payload = {'status': 'passed', 'source_manifest_sha256': sha256(RUN / 'manifest.json'),
        'source_output_hash_checks': hash_checks, 'independent_category_count_checks': count_checks,
        'expert_pairs_reconstructed': len(pair_keys), 'dependence_groups_reconstructed': len(groups),
        'graph_metadata_checks': graph_checks, 'official_reward_components_recomputed': reward_checks,
        'primary_secondary_correlations_and_cluster_intervals_recomputed': statistic_checks,
        'score_csv_rows_replayed': len(table), 'metadata_content_leak_checks': leaked_checks,
        'protected_permissions_checked': permission_checks, 'new_neural_model_calls': 0,
        'clinical_qualified': False, 'selection_changed': False}
    with (OUT / 'audit.json').open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status': 'expert_reference_benchmark_audit_passed', 'audit_sha256': sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)

"""Independent set/count replay of protected character-span diagnostics.

No report strings, source identifiers, native graph payloads or model calls.
Character-alignment semantics are tested separately with authored probes.
"""
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
OLD = BASE / 'cxrgraph_extraction_runs/manual_xl_12766754_001'
RUN = BASE / 'cxrgraph_extraction_runs/manual_xl_12766754_002'
EXPECTED = 'a56743b0a3a940184faebbc7a39b0947d227d7f9ed3ffadc00592e10bed7898c'
STATES = {'Observation::definitely present': 'positive',
          'Observation::definitely absent': 'negative', 'Observation::uncertain': 'uncertain'}


def metric(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return {'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'pinned_completed_character_run_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'output_hash_mismatch')
    for path, value in manifest['pins'].items():
        require(sha256(WORKSPACE / path) == value, 'source_pin_mismatch')
    require(sha256(OLD / 'manifest.json') ==
            '4e08b084dac22f3524b93af2212d4773839733ca585dd0370a1e862b3b3eadb8',
            'historical_first_run_changed')
    comparisons = json.loads((RUN / 'comparisons.json').read_text())
    receipts = json.loads((RUN / 'prediction_receipts.json').read_text())
    evaluation = json.loads((RUN / 'evaluation.json').read_text())
    old = json.loads((OLD / 'comparisons.json').read_text())
    old_receipts = json.loads((OLD / 'prediction_receipts.json').read_text())
    require(len(comparisons) == len(receipts) == 100 and all(r['status'] == 'complete' for r in receipts),
            'all_fixed_manual_reports_complete_required')
    by_id = {c['report_id']: c for c in comparisons}
    receipt_map = {r['report_id']: r for r in receipts}
    require(set(by_id) == set(receipt_map) == {f'report_{i:04d}' for i in range(100)},
            'fixed_opaque_inventory_required')
    invariant_pairs = 0
    for c in old:
        previous = c['common_comparison']
        current = by_id[previous['report_id']]['common_comparison']
        require(by_id[previous['report_id']]['token_boundaries_identical'],
                'old_exact_token_pairs_not_preserved')
        for group in ('entity', 'relation'):
            key = group + '_metrics_in_gold_label_scope'
            require(previous[key] == current[key], 'historical_exact_metric_changed')
        invariant_pairs += 1
    for r in old_receipts:
        if r['status'] == 'complete':
            require(r['prediction_record_sha256'] == receipt_map[r['report_id']]['prediction_record_sha256'],
                    'frozen_native_prediction_changed')
    replay_cells = 0
    for domain, aggregate in evaluation.items():
        rows = [c for c in comparisons if domain == 'all'
                or receipt_map[c['report_id']]['source_domain'] == domain]
        require(aggregate['eligible_reports'] == aggregate['attempted_reports'] == len(rows)
                and aggregate['comparison_availability'] == 1
                and aggregate['unavailable_report_ids'] == [], 'aggregate_coverage_mismatch')
        totals = {name: Counter() for name in ('entity', 'relation')}
        labels = {label: Counter() for label in aggregate['entity_by_label']}
        relation_types = {kind: Counter() for kind in aggregate['relation_by_type']}
        confusion = {a: {b: 0 for b in STATES.values()} for a in STATES.values()}
        ng = np = nm = 0
        for c in rows:
            for group, partitions, label_index in (('entity', labels, 2), ('relation', relation_types, 0)):
                plural = 'entities' if group == 'entity' else 'relations'
                g = {tuple(v) for v in c['gold_character_' + plural]}
                p = {tuple(v) for v in c['prediction_character_' + plural]}
                counts = {'tp': len(g & p), 'fp': len(p - g), 'fn': len(g - p)}
                require(metric(counts) == c['common_comparison'][group + '_metrics_in_gold_label_scope'],
                        'per_report_set_metric_mismatch')
                totals[group].update(counts)
                for name, acc in partitions.items():
                    gg, pp = ({v for v in values if v[label_index] == name} for values in (g, p))
                    acc.update({'tp': len(gg & pp), 'fp': len(pp - gg), 'fn': len(gg - pp)})
                replay_cells += 1
            g = {tuple(e[:2]): STATES[e[2]] for e in c['gold_character_entities'] if e[2] in STATES}
            p = {tuple(e[:2]): STATES[e[2]] for e in c['prediction_character_entities'] if e[2] in STATES}
            shared = g.keys() & p.keys()
            ng, np, nm = ng + len(g), np + len(p), nm + len(shared)
            for span in shared:
                confusion[g[span]][p[span]] += 1
        for group, counts in totals.items():
            require(metric(counts) == aggregate[group + '_micro'], 'aggregate_set_metric_mismatch')
        require({name: metric(counts) for name, counts in labels.items()} == aggregate['entity_by_label']
                and {name: metric(counts) for name, counts in relation_types.items()} == aggregate['relation_by_type'],
                'partitioned_aggregate_metric_mismatch')
        polarity = {'gold_observation_spans': ng, 'prediction_observation_spans': np,
            'matched_observation_spans': nm, 'gold_span_coverage': nm / ng if ng else None,
            'conditional_accuracy': sum(confusion[a][a] for a in confusion) / nm if nm else None,
            'confusion_on_matched_observation_spans': confusion}
        require(polarity == aggregate['polarity'], 'polarity_replay_mismatch')
    root = BASE / 'cxrgraph_extraction_audits'
    private_dir(root)
    output = root / 'numeric_12766754_001'
    private_dir(output, fresh=True)
    result = {'schema_version': 'cxrgraph-character-numeric-audit-v1', 'status': 'passed',
        'run_manifest_sha256': EXPECTED, 'replayed_per_report_entity_relation_cells': replay_cells,
        'all_reports_evaluated': 100, 'domains_replayed': 3,
        'first_run_exact_pairs_preserved': invariant_pairs,
        'historical_native_predictions_unchanged': True, 'source_and_artifact_hashes_verified': True,
        'raw_report_or_native_graph_content_read': False,
        'alignment_semantics_independently_replayed': False,
        'audit_scope': 'stored_character_set_counts_partitions_polarity_and_old_prediction_invariance',
        'clinical_qualified': False, 'checkpoint_training_overlap': None,
        'model_calls': 0, 'worker_sha256': sha256(Path(__file__)),
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    write_json(output / 'audit.json', result)
    return output / 'audit.json', result


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_scope_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_allocation_required')
        os.umask(0o007)
        path, result = execute()
        print(json.dumps({'status': 'protected_manual_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3), 'audit_sha256': sha256(path)}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_manual_numeric_audit_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

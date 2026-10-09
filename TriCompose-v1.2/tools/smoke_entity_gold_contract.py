"""Authored exact-span fixtures only; no model or clinical-file inputs accepted.

This is an engineering smoke, not an independent annotation benchmark.
Requires an actual existing Slurm cgroup and a new protected run directory.
"""
import json
import os
from pathlib import Path
import re
import resource
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, sha256, write_json
from tricompose_v12.entity_gold_contract import (
    POLICY, V1_LABELS, aggregate_reader, compare_annotations, digest, normalize_graph,
)

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
TEXT = 'fixture_finding fixture_anatomy fixture_other .'


def graph(specs, text=TEXT):
    words = text.split()
    return {'text': text, 'entities': {str(i + 1): {'start_ix': start, 'end_ix': end,
        'label': label, 'tokens': ' '.join(words[start:end + 1]), 'relations': outgoing}
        for i, (start, end, label, outgoing) in enumerate(specs)}}


def main():
    require(len(os.sys.argv) == 1, 'authored_worker_accepts_no_input_paths')
    job = os.environ.get('SLURM_JOB_ID', '')
    require(re.fullmatch(r'[1-9][0-9]{0,11}', job) and
        re.search(r'/job_' + re.escape(job) + r'(?:/|\n|$)', Path('/proc/self/cgroup').read_text()),
        'actual_existing_slurm_allocation_required')
    root = BASE / 'entity_gold_contract_smokes'
    out = root / ('authored_' + job + '_001')
    require(not out.exists() and not out.is_symlink(), 'refuse_existing_fixture_run')
    started = time.monotonic()
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_contract.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_entity_gold_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json',
        BASE / 'radeval_length_control_runs/length_controls_12766754_001/manifest.json',
        BASE / 'gold_readiness_audits/gold_availability_12766754_001/audit.json']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    os.umask(0o007)
    private_dir(root)
    private_dir(out, fresh=True)
    write_json(out / 'frozen_plan.json', {'schema_version': 'authored-span-fixture-plan-v1',
        'pins': pins, 'input_scope': 'wholly_invented_graphs',
        'clinical_gold': False, 'clinical_file_inputs_accepted': False,
        'models_executed': 0, 'policy': dict(POLICY)})
    simple = [(0, 0, 'OBS-DP', [])]
    exact = [(0, 0, 'OBS-DP', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])]
    specs = [
        ('exact', exact, exact, 'radgraph_v1', (2, 0, 0), (1, 0, 0)),
        ('polarity_difference', simple, [(0, 0, 'OBS-DA', [])], 'radgraph_v1', (0, 1, 1), (0, 0, 0)),
        ('uncertain_difference', [(0, 0, 'OBS-U', [])], [(0, 0, 'OBS-DA', [])], 'radgraph_v1', (0, 1, 1), (0, 0, 0)),
        ('valid_zero_prediction_entities', simple, [], 'radgraph_v1', (0, 0, 1), (0, 0, 0)),
        ('both_zero_entities', [], [], 'radgraph_v1', (0, 0, 0), (0, 0, 0)),
        ('xl_outside_gold_scope', simple, [(0, 0, 'Observation::measurement::definitely present', [])],
            'radgraph_xl', (0, 0, 1), (0, 0, 0)),
        ('relation_direction', [(0, 0, 'OBS-DP', [['modify', '2']]), (2, 2, 'OBS-DP', [])],
            [(0, 0, 'OBS-DP', []), (2, 2, 'OBS-DP', [['modify', '1']])], 'radgraph_v1', (2, 0, 0), (0, 1, 1)),
        ('span_boundaries', simple, [(0, 1, 'OBS-DP', [])], 'radgraph_v1', (0, 1, 1), (0, 0, 0)),
    ]
    records, checks, comparisons = [], [], {'reader_01': [], 'reader_02': []}
    for index, (kind, gs, ps, schema, expected_entity, expected_relation) in enumerate(specs):
        report = f'report_{index:04d}'
        source = digest(['authored_source', report])
        prediction = normalize_graph(graph(ps), native_schema=schema, report_id=report,
            source_report_sha256=source, origin='authored_fixture')
        records.append(prediction)
        for reader in comparisons:
            # Intentional authored disagreement, never a simulated clinician consensus.
            local = gs if reader == 'reader_01' else [
                (a, b, 'OBS-DA' if label == 'OBS-DP' else label, outgoing)
                for a, b, label, outgoing in gs]
            raw = graph(local)
            kwargs = dict(native_schema='radgraph_v1', report_id=report,
                source_report_sha256=source, origin='authored_fixture', reader_id=reader)
            gold = normalize_graph(raw, **kwargs)
            require(gold == normalize_graph(raw, **kwargs), 'deterministic_authored_normalization_required')
            records.append(gold)
            result = compare_annotations(gold, prediction)
            comparisons[reader].append(result)
            if reader == 'reader_01':
                for key, expected in (('entity_metrics_in_gold_label_scope', expected_entity),
                                      ('relation_metrics_in_gold_label_scope', expected_relation)):
                    require(tuple(result[key][k] for k in ('tp', 'fp', 'fn')) == expected,
                        'predeclared_authored_count_check_required')
            require(result['clinical_score'] is None and result['policy'] == POLICY,
                'no_clinical_qualification_from_fixture')
            checks.append({'fixture_id': f'fixture_{index:03d}', 'kind': kind, 'reader_id': reader,
                'comparison_sha256': result['comparison_sha256']})
    # A binding failure is unavailable, not a fabricated zero-entity prediction.
    g = normalize_graph(graph(simple), native_schema='radgraph_v1', report_id='report_0008',
        source_report_sha256=digest(['authored_binding_failure']), origin='authored_fixture', reader_id='reader_01')
    p = normalize_graph(graph(simple, TEXT + ' extra_fixture'), native_schema='radgraph_v1', report_id='report_0008',
        source_report_sha256=g['source_report_sha256'], origin='authored_fixture')
    try:
        compare_annotations(g, p)
    except ValueError as error:
        require(str(error) == 'same_declared_report_and_exact_token_sequence_required',
            'authored_binding_failure_code_required')
    else:
        raise ValueError('binding_failure_must_not_be_scored')
    attempted = [f'report_{i:04d}' for i in range(9)]
    aggregates = {reader: aggregate_reader(rows, reader_id=reader,
        attempted_report_ids=attempted, supported_labels=sorted(V1_LABELS.values()))
        for reader, rows in comparisons.items()}
    for value in aggregates.values():
        require(value['attempted_reports'] == 9 and value['eligible_reports'] == 8
            and value['unavailable_report_ids'] == ['report_0008'] and value['clinical_score'] is None,
            'fixed_attempted_inventory_and_null_clinical_score_required')
    write_json(out / 'normalized_records.json', records)
    write_json(out / 'comparisons.json', comparisons)
    write_json(out / 'reader_aggregates.json', aggregates)
    write_json(out / 'fixture_checks.json', checks)
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()),
        'frozen_source_or_historical_manifest_changed')
    summary = {'status': 'authored_interface_smoke_complete', 'fixture_inputs': 9,
        'eligible_fixture_comparisons': 16, 'separate_authored_readers': 2,
        'predeclared_count_checks': 16, 'intentional_unavailable_fixture_ids': ['fixture_008'],
        'actual_human_annotation_data_evaluated': False, 'clinical_benchmark_completed': False,
        'raw_or_synthetic_patient_artifacts_read': False, 'model_calls': 0, 'downloads': 0,
        'new_slurm_submissions': 0, 'policy': dict(POLICY),
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)}
    write_json(out / 'summary.json', summary)
    write_json(out / 'manifest.json', {'schema_version': 'authored-entity-gold-contract-smoke-v1',
        'pins': pins, 'artifacts': [{'path': path.name, 'sha256': sha256(path)}
                                  for path in sorted(out.glob('*.json'))]})
    print(json.dumps({'status': summary['status'], 'runtime_seconds': round(summary['runtime_seconds'], 3),
        'peak_rss_gib': round(summary['peak_rss_gib'], 3), 'manifest_sha256': sha256(out / 'manifest.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'authored_entity_gold_contract_smoke_failed',
            'error_type': type(error).__name__}))
        raise SystemExit(2)

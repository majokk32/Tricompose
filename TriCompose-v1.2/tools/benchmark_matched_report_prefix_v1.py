#!/usr/bin/env python3
"""Cache-only matched-guard control; seal new choices before secondary readout."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_probe_repair_v1 as b
import diagnose_probe_gap_v1 as gap
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12 import matched_report_prefix_v1 as control
from tricompose_v12.live_workers import check_pins

PROTOCOL = ROOT / 'configs/matched_report_prefix_v1.json'


def replay(grid, policy, cap):
    initial = (*policy['image_order'][0], policy['report_order'][0])
    runner = control.FixedReportPrefix(b.policy_module.snapshot(grid[initial]), policy['report_order'], cap)
    while True:
        request = runner.propose()
        if request['action'] == 'stop':
            result = runner.result(); result['terminal_reason'] = request['reason']; return result
        # Simulator opens only the reserved metadata slot, never weights/bodies.
        runner.complete(request, b.policy_module.snapshot(grid[tuple(request['slot'])]))


def endpoint_values(records, observations):
    if len(records) != len(observations): raise ValueError('complete_secondary_inventory_required')
    output = {}
    fields = ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256',
        'cxr_candidate_id', 'report_candidate_id')
    for record in records:
        cid = record['triple_candidate_id']
        if cid not in observations or cid in output: raise ValueError('unique_secondary_candidate_ids_required')
        view = observations[cid]
        if (record['case_id'] != view['case_id'] or record['calibrated'] is not False
                or any(record[k] != view['lineage'][k] for k in fields)):
            raise ValueError('same_uncalibrated_secondary_artifact_lineage_required')
        value = record['biovil_raw_cosine']
        if value is None:
            if record['status'] != 'not_available' or not record['reason']: raise ValueError('explicit_secondary_na_required')
        elif (type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1
                or record['status'] != 'computed_secondary_uncalibrated' or record['reason'] is not None):
            raise ValueError('finite_raw_secondary_cosine_required')
        output[cid] = value
    return output


def run(args):
    b.cpu_guard()
    sources = {}; bank, policy = b.load_primary(sources)
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol['caps'] != list(b.CAPS) or protocol['initial_image'] != policy['image_order'][0]
            or protocol['report_order'] != policy['report_order']
            or protocol['training_allowed'] is not False or protocol['clinical_qualified'] is not False
            or protocol['secondary_endpoint_used_for_selection'] is not False):
        raise ValueError('same_declared_order_and_diagnostic_protocol_required')
    pins = {str(path): sha256_file(path) for path in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_matched_report_prefix_v1.py', Path(control.__file__),
        Path(b.__file__), Path(b.policy_module.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        outcomes = [replay(grid, policy, cap) for _, grid in sorted(bank.items()) for cap in b.CAPS]
        choices = write_private_text(temporary / 'control_outcomes.jsonl',
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in outcomes))
        seal = sha256_file(choices)
        # Existing comparison choices and secondary scores cannot affect control.
        pm = json.loads(b.checked(gap.SOURCE / 'manifest.json', gap.SOURCE_SHA, 1024**2, sources))
        probes = [json.loads(line) for line in b.artifact(gap.SOURCE, pm, 'probe_outcomes.jsonl', sources).splitlines()]
        probes = [r for r in probes if r['method'] == b.NEW_METHODS[0]]
        expected = {(case, cap) for case in bank for cap in b.CAPS}
        if len(probes) != 400 or {(r['case_id'], r['model_call_budget']) for r in probes} != expected:
            raise ValueError('complete_sealed_probe_comparisons_required')
        old_root, om = b.parent(b.REPLAY, sources)
        old = [json.loads(line) for line in b.artifact(old_root, om, 'replay_outcomes.jsonl', sources).splitlines()]
        statics = [r for r in old if r['method'] == 'static_rerank']
        if len(statics) != 400 or {(r['case_id'], r['model_call_budget']) for r in statics} != expected:
            raise ValueError('complete_sealed_static_comparisons_required')
        observations = {c['score_record']['triple_candidate_id']: b.policy_module.snapshot(c)
            for grid in bank.values() for c in grid.values()}
        er, em = b.parent(b.ENDPOINT, sources)
        endpoint = json.loads(b.artifact(er, em, 'scores.json', sources))
        endpoints = endpoint_values(endpoint['records'], observations)
        trials = [{'case_id': row['case_id'], 'method': row['method'],
            'model_call_budget': row['model_call_budget'], 'selected_candidate_id': row['selected_candidate_id'],
            'calls': row['simulated_model_calls'] if row['method'] == 'static_rerank' else row['simulated_calls']}
            for row in (*outcomes, *probes, *statics)]
        cases, table = b.aggregate(trials, observations, endpoints)
        if len(cases) != 1200 or len(table) != 15: raise ValueError('all_three_methods_all_cases_all_caps_required')
        probe_index = {(r['case_id'], r['model_call_budget']): r for r in probes}
        pairs = [{'case_id': r['case_id'], 'model_call_budget': r['model_call_budget'],
            'same_final_candidate': r['selected_candidate_id'] == probe_index[r['case_id'], r['model_call_budget']]['selected_candidate_id'],
            'fixed_prefix_simulated_calls': r['simulated_calls'],
            'probe_simulated_calls': probe_index[r['case_id'], r['model_call_budget']]['simulated_calls'],
            'commit_guard_changed': False, 'clinical_repair_success': None} for r in outcomes]
        write_private_text(temporary / 'case_means.csv', b.csv_text(cases))
        write_private_text(temporary / 'method_comparison.csv', b.csv_text(table))
        write_private_text(temporary / 'matched_case_controls.csv', b.csv_text(pairs))
        summary = {'schema_version': control.VERSION + '-benchmark', 'status': 'matched_guard_development_control_complete',
            'fixed_ehr_cases': 80, 'control_trials': 400, 'all_caps': list(b.CAPS),
            'same_final_as_probe_cases_by_cap': {str(cap): sum(r['same_final_candidate'] for r in pairs if r['model_call_budget'] == cap) for cap in b.CAPS},
            'accepted_control_transitions_by_cap': {str(cap): sum(h['accepted'] for r in outcomes if r['model_call_budget'] == cap for h in r['history']) for cap in b.CAPS},
            'control_selection_sha256_before_endpoint': seal, 'commit_guard_changed': False,
            'new_model_calls': 0, 'actual_regeneration_executed': False, 'clinical_qualified': False,
            'source_bodies_pixels_or_weights_read': False, 'old_choices_changed': False,
            'secondary_used_for_control_selection': False, 'posthoc_development_control': True,
            'simulated_cost_is_measured_gpu_time': False}
        write_private_json(temporary / 'summary.json', summary)
        if sha256_file(choices) != seal or any(sha256_file(path) != pin for path, pin in sources.items()):
            raise ValueError('sealed_choices_or_sources_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': control.VERSION + '-benchmark-manifest',
            'sources': sources, 'code_pins': pins, 'control_selection_sha256_before_endpoint': seal,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in temporary.iterdir() if path.is_file()},
            'original_selection_changed': False, 'clinical_qualified': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'matched_guard_controls'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        root, summary = run(args)
        print(json.dumps({'status': summary['status'], 'new_model_calls': 0,
            'manifest_sha256': sha256_file(root / 'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__})); return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())

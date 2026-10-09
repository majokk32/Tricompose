"""Cached seven-mask diagnostic only; no models, source reports or selection.

Readers are dependent evidence proposals, not independent truth votes. This
post-hoc development analysis evaluates every fixed mask without choosing one.
"""
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import time

import report_assertion_challenge as challenge
from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from tricompose_v12.assertion_agreement_diagnostic import POLICIES, evaluate

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
BENCH = BASE / 'radgraph_assertion_runs/authored56_12766754_001'
EXPECTED = '317ba52eda0e258ec13bba5d64a1cb2e9763e194263cc7d2ebadeacbee20a993'
BANK = BASE / 'benchmarks/authored_assertions_20261002_001'


def execute(run):
    started = time.monotonic()
    require(sha256(BENCH / 'manifest.json') == EXPECTED, 'pinned_completed_assertion_benchmark_required')
    manifest = json.loads((BENCH / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(BENCH / name) == value, 'benchmark_artifact_changed')
    for group in ('pins', 'analysis_input_pins'):
        for path, value in manifest[group].items():
            require(sha256(WORKSPACE / path) == value, 'benchmark_source_changed')
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/assertion_agreement_diagnostic.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_assertion_agreement_diagnostic.py',
        BENCH / 'manifest.json', BANK / 'resolver.jsonl', BANK / 'references.jsonl']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    write_json(run / 'frozen_plan.json', {'schema_version': 'authored-agreement-mask-plan-v1',
        'pins': pins, 'policies': POLICIES, 'fixed_designated_checks': 80,
        'fit_or_best_policy_selection': False, 'post_hoc_development_diagnostic': True,
        'model_calls': 0, 'clinical_qualified': False})
    resolver = [json.loads(line) for line in (BANK / 'resolver.jsonl').read_text().splitlines() if line]
    references, _ = challenge.load_references(BANK, resolver)
    require(len(references) == 56 and sum(len(r['evaluation_findings']) for r in references) == 80,
            'all_fixed_designated_checks_required')
    native = json.loads((BENCH / 'predictions.json').read_text())
    predictions = {'radgraph': native['records']}
    for name, root in (
        ('chexbert', BASE / 'verification_runs/authored_assertions_chexbert_12580901'),
        ('qwen_span_v2', BASE / 'verification_runs/authored_span_v2_12646689')):
        predictions[name] = json.loads((root / 'predictions.json').read_text())['records']
    result, details = evaluate(references, predictions)
    # Arithmetic replay from original predictions, not a second call to decision().
    indexed = {name: {r['item_id']: r for r in records} for name, records in predictions.items()}
    keys = {(r['policy'], r['item_id'], r['finding']): r for r in details}
    replayed = 0
    for policy, readers in POLICIES.items():
        attempted = accepted = correct = unsafe = flips = known_count = known_accepted = 0
        for ref in references:
            for finding in ref['evaluation_findings']:
                expected = ref['expected_states'][finding]
                reader_rows = [indexed[name][ref['item_id']] for name in readers]
                available = all(r['status'] == 'complete' for r in reader_rows)
                values = [r['finding_states'][finding] for r in reader_rows] if available else []
                proposal = values[0] if values and len(set(values)) == 1 and values[0] in ('positive', 'negative') else None
                require(keys[(policy, ref['item_id'], finding)]['state'] == proposal,
                        'independent_mask_decision_mismatch')
                attempted += 1
                known = expected in ('positive', 'negative')
                known_count += known
                if proposal is not None:
                    accepted += 1
                    correct += proposal == expected
                    unsafe += not known
                    known_accepted += known
                    flips += known and proposal != expected
                replayed += 1
        row = result['policies'][policy]
        require((attempted, accepted, correct, unsafe, flips, known_count, known_accepted) == (
            row['attempted_designated_checks'], row['accepted_determinate_proposals'],
            row['correct_determinate_proposals'], row['determinate_on_authored_uncertain_or_unknown'],
            row['hard_positive_negative_flips'], row['known_determinate_reference_checks'],
            round(row['known_reference_proposal_coverage'] * known_count)),
            'independent_risk_count_mismatch')
        require(row['accepted_authored_state_errors'] == accepted - correct
            and row['proposal_coverage'] == accepted / attempted
            and row['accepted_authored_error_rate'] == ((accepted - correct) / accepted if accepted else None)
            and row['correct_determinate_recall'] == correct / known_count,
            'independent_risk_rate_mismatch')
    # Independently replay the primary 80-check matrices from all original readers.
    benchmark = json.loads((BENCH / 'evaluation.json').read_text())
    matrix_checks = 0
    for name, records in predictions.items():
        matrix = {a: dict.fromkeys((*challenge.STATES, 'unavailable'), 0) for a in challenge.STATES}
        for ref in references:
            prediction = indexed[name][ref['item_id']]
            for finding in ref['evaluation_findings']:
                predicted = prediction['finding_states'][finding] if prediction['status'] == 'complete' else 'unavailable'
                matrix[ref['expected_states'][finding]][predicted] += 1
                matrix_checks += 1
        recorded = benchmark['radgraph' if name == 'radgraph' else 'baselines']
        if name != 'radgraph':
            recorded = recorded[name]
        require(matrix == recorded['designated_targets']['confusion_matrix'],
                'independent_primary_state_matrix_mismatch')
    require(all(sha256(WORKSPACE / path) == value for path, value in pins.items()),
            'input_or_consumed_code_changed')
    write_json(run / 'evaluation.json', result)
    write_json(run / 'details.json', {'records': details})
    summary = {'schema_version': 'authored-agreement-mask-run-v1', 'status': 'complete',
        'model_calls': 0, 'source_reports_read': False,
        'independently_replayed_mask_decisions': replayed,
        'independently_replayed_primary_state_cells': matrix_checks,
        'policies_evaluated': 7, 'best_policy_selected': False,
        'clinical_qualified': False, 'regeneration_authorized': False,
        'selection_changed': False, 'thresholds_fitted': False,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    write_json(run / 'summary.json', summary)
    write_json(run / 'manifest.json', {'schema_version': 'authored-agreement-mask-receipt-v1',
        'pins': pins, 'source_assertion_benchmark_sha256': EXPECTED,
        'artifacts': {p.name: sha256(p) for p in sorted(run.glob('*.json'))}})
    return summary


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_scope_worker_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_allocation_required')
        os.umask(0o007)
        parent = BASE / 'assertion_agreement_runs'
        private_dir(parent)
        run = parent / 'seven_masks_12766754_001'
        private_dir(run, fresh=True)
        summary = execute(run)
        print(json.dumps({'status': 'protected_agreement_diagnostic_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': sha256(run / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_agreement_diagnostic_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

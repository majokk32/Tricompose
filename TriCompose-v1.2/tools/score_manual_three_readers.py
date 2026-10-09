"""Cached fixed seven-mask analysis after real-manual native parsing is sealed.

No raw source report, original annotation row, image or model is loaded here.
Native labels are primary; mention-conflict is a separate secondary readout,
not another independent reader and never chosen to improve observed metrics.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from tricompose_v12.manual_three_reader_agreement import FINDINGS, STATES, POLICIES, evaluate

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
PARENT = BASE / 'manual_literal_reader_runs/manual100_12766754_001'
PARENT_SHA = '030ac27c56217db264bb09c93cba6a3ec2ae7b5322c17323a918fd89ded01a5f'
NATIVE = BASE / 'manual_negbio_runs/manual100_12766754_001'


def convert(records, view):
    require(view in ('native_labels', 'mention_conflict_view'), 'fixed_native_or_secondary_conflict_view_required')
    result = []
    for row in records:
        output = row['output']
        require(output['status'] in ('complete', 'failed_unavailable'), 'native_availability_required')
        states = None
        if output['status'] == 'complete':
            require(output['original_text_sha256'] == row['report_sha256'], 'native_original_input_hash_required')
            require(output['hard_action_eligible'] is False and output['clinical_score'] is None,
                    'native_output_remains_unqualified')
            states = {f: output[view][f]['state'] for f in FINDINGS}
            require(all(v in STATES for v in states.values()), 'native_named_four_states_required')
        else:
            require(output['native_labels'] is None and output['mention_conflict_view'] is None,
                    'failed_native_labels_must_be_null')
        result.append({'report_id': row['item_id'], 'source_sha256': row['report_sha256'],
            'status': output['status'], 'finding_states': states,
            'failure_reason': output['failure_reason']})
    return result


def execute(expected):
    started = time.monotonic()
    require(sha256(NATIVE / 'manifest.json') == expected and sha256(PARENT / 'manifest.json') == PARENT_SHA,
            'pinned_completed_native_and_reference_runs_required')
    manifests = {}
    for root in (NATIVE, PARENT):
        m = json.loads((root / 'manifest.json').read_text())
        for name, value in m['artifacts'].items():
            require(sha256(root / name) == value, 'immutable_completed_output_required')
        for name, value in m['pins'].items():
            path = Path(name) if Path(name).is_absolute() else WORKSPACE / name
            require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value,
                    'consumed_program_or_source_changed')
        manifests[root] = m
    ns = json.loads((NATIVE / 'summary.json').read_text())
    require(ns['status'] == 'complete' and ns['attempted_reports'] == 100
            and ns['parser_passes_including_replay'] == 200 and ns['clinical_qualified'] is False,
            'all_fixed_native_passes_and_scope_required')
    receipt = json.loads((NATIVE / 'prediction_freeze_receipt.json').read_text())
    require(receipt['reference_projection_or_other_reader_states_decoded'] is False
            and all(sha256(NATIVE / name) == value for name, value in receipt['sha256'].items()),
            'predictions_closed_before_reference_analysis_required')
    parent = BASE / 'manual_three_reader_runs'
    private_dir(parent)
    target = parent / 'seven_masks_12766754_001'
    private_dir(target, fresh=True)
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_three_reader_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_literal_assertions.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/assertion_agreement_diagnostic.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_three_reader_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_score_manual_three_readers.py',
        NATIVE / 'manifest.json', NATIVE / 'predictions.json', NATIVE / 'summary.json',
        NATIVE / 'prediction_freeze_receipt.json', PARENT / 'manifest.json',
        PARENT / 'reference_projection.json', PARENT / 'evaluation.json',
        PARENT / 'chexbert_predictions.json', PARENT / 'radgraph_predictions.json']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    write_json(target / 'frozen_plan.json', {'schema_version': 'manual-three-reader-mask-plan-v1',
        'pins': pins, 'native_manifest_sha256': expected, 'reference_manifest_sha256': PARENT_SHA,
        'all_reference_reports': 100, 'findings': FINDINGS, 'fixed_masks': POLICIES,
        'primary_negbio_readout': 'native_labels', 'secondary_readout': 'mention_conflict_view',
        'views_are_not_independent_readers': True, 'no_view_or_best_mask_selection': True,
        'post_hoc_development_analysis': True, 'clinical_qualified': False,
        'model_calls': 0, 'selection_changed': False, 'regeneration_authorized': False})
    references = json.loads((PARENT / 'reference_projection.json').read_text())['records']
    old_predictions = {name: json.loads((PARENT / (name + '_predictions.json')).read_text())['records']
                       for name in ('radgraph', 'chexbert')}
    old = json.loads((PARENT / 'evaluation.json').read_text())
    native = json.loads((NATIVE / 'predictions.json').read_text())['records']
    require(len(references) == len(native) == 100, 'all_hundred_manual_entries_required')
    results, domains, details, table = {}, {}, [], []
    replayed = 0
    for view in ('native_labels', 'mention_conflict_view'):
        predictions = {**old_predictions, 'chexpert_negbio': convert(native, view)}
        result, rows = evaluate(references, predictions)
        for reader in ('radgraph', 'chexbert'):
            require(result['reader_metrics'][reader] == old['readers'][reader],
                    'historical_reader_metrics_must_remain_identical')
        for policy in ('radgraph', 'chexbert', 'agree_radgraph_chexbert'):
            for key, value in old['masks'][policy].items():
                if key in result['masks'][policy]:
                    require(result['masks'][policy][key] == value, 'historical_mask_metrics_must_remain_identical')
        results[view] = result
        details.extend({'readout': view, **row} for row in rows)
        domains[view] = {}
        for domain in ('mimic', 'chexpert'):
            ids = {r['report_id'] for r in references if r['source_domain'] == domain}
            require(len(ids) == 50, 'fixed_disjoint_domain_denominators_required')
            domains[view][domain], _ = evaluate([r for r in references if r['report_id'] in ids],
                {name: [r for r in records if r['report_id'] in ids] for name, records in predictions.items()})
        indexed = {name: {r['report_id']: r for r in records} for name, records in predictions.items()}
        row_lookup = {(r['policy'], r['report_id'], r['finding']): r for r in rows}
        for domain, current in (('all', result), *domains[view].items()):
            refs = [r for r in references if domain == 'all' or r['source_domain'] == domain]
            for policy, readers in POLICIES.items():
                accepted = correct = ak = flips = uncertain = unknown = known = 0
                for ref in refs:
                    for f in FINDINGS:
                        rs = [indexed[name][ref['report_id']] for name in readers]
                        values = [r['finding_states'][f] for r in rs] if all(r['status'] == 'complete' for r in rs) else []
                        proposal = values[0] if values and len(set(values)) == 1 and values[0] in ('positive', 'negative') else None
                        expected_state = ref['finding_states'][f]
                        is_known = expected_state in ('positive', 'negative')
                        known += is_known
                        if proposal is not None:
                            accepted += 1
                            ak += is_known
                            correct += is_known and proposal == expected_state
                            flips += is_known and proposal != expected_state
                            uncertain += expected_state == 'uncertain'
                            unknown += expected_state == 'unknown'
                        require(row_lookup[(policy, ref['report_id'], f)]['state'] == proposal,
                                'independent_mask_proposal_mismatch')
                        replayed += 1
                m = current['masks'][policy]
                require((accepted, correct, ak, flips, uncertain, unknown, known) == (
                    m['accepted_determinate_proposals'], m['correct_known_proposals'], m['accepted_known_proposals'],
                    m['hard_positive_negative_flips'], m['determinate_on_uncertain_reference'],
                    m['determinate_on_unknown_literal_reference'], m['positive_negative_reference_checks']),
                    'independent_mask_reference_counts_mismatch')
                table.append({'negbio_view': view, 'domain': domain, 'policy': policy,
                    'attempted_checks': m['attempted_checks'], 'determinate_proposals': accepted,
                    'positive_support': m['reference_support_by_state']['positive'],
                    'correct_positive_proposals': m['correct_positive_proposals'],
                    'negative_support': m['reference_support_by_state']['negative'],
                    'correct_negative_proposals': m['correct_negative_proposals'],
                    'uncertain_support': m['reference_support_by_state']['uncertain'],
                    'determinate_on_uncertain': uncertain, 'literal_unknown_support': m['reference_support_by_state']['unknown'],
                    'determinate_on_literal_unknown': unknown, 'correct_known_proposals': correct,
                    'known_reference_checks': known, 'hard_positive_negative_flips': flips,
                    'known_proposal_recall': m['correct_known_reference_recall'], 'proposal_coverage': m['proposal_coverage'],
                    'unavailable_reader_checks': m['unavailable_reader_checks'], 'clinical_qualified': False})
    write_json(target / 'evaluation.json', results)
    write_json(target / 'domain_evaluation.json', domains)
    write_json(target / 'mask_details.json', {'records': details})
    with (target / 'score_table.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        stream.flush()
        os.fsync(stream.fileno())
    (target / 'score_table.csv').chmod(0o660)
    summary = {'schema_version': 'manual-three-reader-mask-run-v1', 'status': 'complete',
        'model_calls': 0, 'attempted_reports': 100, 'independently_replayed_mask_checks': replayed,
        'policies_per_view': 7, 'readouts': 2, 'native_primary': True,
        'reference_uncertain_examples': 10, 'native_runtime': ns,
        'historical_two_reader_outputs_and_metrics_unchanged': True,
        'source_rows_reports_images_or_native_graph_text_read': False,
        'clinical_qualified': False, 'best_policy_selected': False, 'thresholds_fitted': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'consumed_inputs_or_program_changed')
    write_json(target / 'summary.json', summary)
    write_json(target / 'manifest.json', {'schema_version': 'manual-three-reader-mask-receipt-v1', 'pins': pins,
        'artifacts': {p.name: sha256(p) for p in sorted(target.iterdir())},
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--negbio-manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute(args.negbio_manifest_sha256)
        print(json.dumps({'status': 'protected_manual_three_reader_analysis_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_manual_three_reader_analysis_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

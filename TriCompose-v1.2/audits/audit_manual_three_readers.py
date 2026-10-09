"""Independent arithmetic/evidence audit of the sealed three-reader diagnostic.

No model, original source row, report text, image or native graph is decoded.
Source pins are hashed as bytes only. All decisions are replayed without the
production decision/statistics functions; no best mask or readout is selected.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'manual_three_reader_runs/seven_masks_12766754_001'
RUN_SHA = 'de2439533772e30a68714a7eef6c18b3970c52afee03360aa8e50b45be2fef78'
NATIVE = BASE / 'manual_negbio_runs/manual100_12766754_001'
NATIVE_SHA = '97878f07cdd76853d983fd9c778a02e32d23cedef63a699397a17d3025a7ae82'
PARENT = BASE / 'manual_literal_reader_runs/manual100_12766754_001'
PARENT_SHA = '030ac27c56217db264bb09c93cba6a3ec2ae7b5322c17323a918fd89ded01a5f'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
HEADS = ('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax')
STATES = ('positive', 'negative', 'uncertain', 'unknown')
READERS = ('radgraph', 'chexbert', 'chexpert_negbio')
MASKS = {
    'radgraph': ('radgraph',), 'chexbert': ('chexbert',),
    'chexpert_negbio': ('chexpert_negbio',),
    'agree_radgraph_chexbert': ('radgraph', 'chexbert'),
    'agree_radgraph_chexpert_negbio': ('radgraph', 'chexpert_negbio'),
    'agree_chexbert_chexpert_negbio': ('chexbert', 'chexpert_negbio'),
    'agree_all_three': READERS,
}


def proposal(rows, finding):
    require(rows and all(r['status'] in ('complete', 'failed_unavailable') for r in rows),
            'explicit_availability_required')
    if any(r['status'] != 'complete' for r in rows):
        return None, 'unavailable_reader'
    values = [r['finding_states'][finding] for r in rows]
    require(all(v in STATES for v in values), 'four_states_required')
    if 'unknown' in values:
        return None, 'abstain_unknown'
    if 'uncertain' in values:
        return None, 'abstain_uncertain'
    if len(set(values)) != 1:
        return None, 'abstain_disagreement'
    return values[0], 'accepted_determinate_proposal'


def counts(pairs):
    matrix = {s: dict.fromkeys((*STATES, 'unavailable'), 0) for s in STATES}
    for truth, prediction in pairs:
        require(truth in STATES and prediction in (*STATES, 'unavailable'), 'named_states_required')
        matrix[truth][prediction] += 1
    known = sum(sum(matrix[s].values()) for s in ('positive', 'negative'))
    correct = matrix['positive']['positive'] + matrix['negative']['negative']
    flips = matrix['positive']['negative'] + matrix['negative']['positive']
    return matrix, known, correct, flips


def verify_manifest(root, expected):
    require(sha256(root / 'manifest.json') == expected, 'completed_run_pin_required')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(root / name) == value, 'completed_artifact_changed')
    for name, value in manifest['pins'].items():
        path = Path(name) if Path(name).is_absolute() else WORKSPACE / name
        require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value,
                'consumed_code_or_source_changed')
    return manifest


def execute():
    started = time.monotonic()
    for root, value in ((RUN, RUN_SHA), (NATIVE, NATIVE_SHA), (PARENT, PARENT_SHA)):
        verify_manifest(root, value)
    require(sha256(BANK) == BANK_SHA, 'original_synthetic_bank_unchanged_required')
    refs = json.loads((PARENT / 'reference_projection.json').read_text())['records']
    ids = {f'report_{i:04d}' for i in range(100)}
    require(len(refs) == 100 and {r['report_id'] for r in refs} == ids, 'all_hundred_references_required')
    indexed = {name: {r['report_id']: r for r in
        json.loads((PARENT / (name + '_predictions.json')).read_text())['records']}
        for name in ('radgraph', 'chexbert')}
    native = json.loads((NATIVE / 'predictions.json').read_text())['records']
    replays = json.loads((NATIVE / 'replays.json').read_text())['records']
    ns = json.loads((NATIVE / 'summary.json').read_text())
    require(len(native) == len(replays) == 100 and {r['item_id'] for r in native} == ids
            and {r['item_id'] for r in replays} == ids, 'all_primary_and_repeat_attempts_required')
    native_by_id = {r['item_id']: r for r in native}
    refs_by_id = {r['report_id']: r for r in refs}
    status_counts, failure_counts, failure_reference_counts = Counter(), Counter(), Counter()
    differences = both_complete = 0
    for replay in replays:
        row = native_by_id[replay['item_id']]
        require(row['report_sha256'] == replay['report_sha256'] == refs_by_id[row['item_id']]['source_sha256'],
                'reference_primary_repeat_source_hash_match_required')
        same = row['output'] == replay['replay_output']
        complete = row['output']['status'] == replay['replay_output']['status'] == 'complete'
        require(replay['same_full_evidence'] == same and replay['both_complete'] == complete,
                'full_repeat_receipt_mismatch')
        differences += not same
        both_complete += complete
        out = row['output']
        require(out['status'] in ('complete', 'failed_unavailable') and out['hard_action_eligible'] is False
                and out['clinical_score'] is None, 'native_results_unqualified_and_explicit_required')
        status_counts[out['status']] += 1
        if out['status'] == 'complete':
            require(out['original_text_sha256'] == row['report_sha256'], 'completed_original_source_binding_required')
        else:
            require(out['native_labels'] is None and out['mention_conflict_view'] is None,
                    'failed_label_vectors_must_remain_null')
            failure_counts[out['failure_reason']] += 1
            failure_reference_counts.update(refs_by_id[row['item_id']]['finding_states'].values())
    require(dict(status_counts) == ns['primary_status_counts'] and dict(failure_counts) == ns['primary_failure_reasons']
            and differences == ns['full_evidence_replay_differences'] and both_complete == ns['replays_both_complete']
            and ns['parser_passes_including_replay'] == 200, 'independent_native_accounting_mismatch')
    evaluations = json.loads((RUN / 'evaluation.json').read_text())
    domains = json.loads((RUN / 'domain_evaluation.json').read_text())
    details = json.loads((RUN / 'mask_details.json').read_text())['records']
    detail_map = {(r['readout'], r['policy'], r['report_id'], r['finding']): r for r in details}
    require(len(details) == len(detail_map) == 5600, 'all_seven_masks_both_views_required')
    with (RUN / 'score_table.csv').open() as stream:
        table = list(csv.DictReader(stream))
    table_map = {(r['negbio_view'], r['domain'], r['policy']): r for r in table}
    require(len(table) == len(table_map) == 42, 'all_seven_masks_three_domains_two_views_required')
    matrix_checks = mask_checks = 0
    tradeoffs = {}
    for view in ('native_labels', 'mention_conflict_view'):
        indexed['chexpert_negbio'] = {r['item_id']: {
            'status': r['output']['status'], 'source_sha256': r['report_sha256'],
            'finding_states': {f: r['output'][view][f]['state'] for f in HEADS}
                if r['output']['status'] == 'complete' else None} for r in native}
        require(all(set(rows) == ids for rows in indexed.values()), 'no_failed_case_dropping_required')
        for domain, result in (('all', evaluations[view]), *domains[view].items()):
            selected = [r for r in refs if domain == 'all' or r['source_domain'] == domain]
            require(len(selected) == (100 if domain == 'all' else 50), 'fixed_domain_sizes_required')
            for reader in READERS:
                metrics = result['reader_metrics'][reader]
                all_pairs = []
                for f in HEADS:
                    pairs = []
                    for ref in selected:
                        r = indexed[reader][ref['report_id']]
                        require(r['source_sha256'] == ref['source_sha256'], 'same_readout_input_required')
                        p = r['finding_states'][f] if r['status'] == 'complete' else 'unavailable'
                        pairs.append((ref['finding_states'][f], p))
                        matrix_checks += 1
                    require(counts(pairs)[0] == metrics['per_finding'][f]['confusion_matrix'],
                            'per_head_reader_matrix_mismatch')
                    all_pairs.extend(pairs)
                matrix, known, correct, flips = counts(all_pairs)
                m = metrics['overall']
                require(matrix == m['confusion_matrix'] and known == m['positive_negative_reference_checks']
                        and correct == m['correct_positive_negative_states'] and flips == m['hard_positive_negative_flips']
                        and m['failure_aware_positive_negative_recovery'] == (correct / known if known else None),
                        'overall_reader_counts_mismatch')
            for policy, readers in MASKS.items():
                support, correct_by_state, commits = Counter(), Counter(), Counter()
                accepted = ak = flips = unavailable = 0
                for ref in selected:
                    for f in HEADS:
                        state, status = proposal([indexed[n][ref['report_id']] for n in readers], f)
                        truth = ref['finding_states'][f]
                        recorded = detail_map[(view, policy, ref['report_id'], f)]
                        require((state, status, truth) == (recorded['state'], recorded['status'],
                                recorded['human_literal_reference_state']), 'mask_decision_or_reference_mismatch')
                        support[truth] += 1
                        unavailable += status == 'unavailable_reader'
                        if state is not None:
                            accepted += 1
                            commits[truth] += 1
                            correct_by_state[truth] += state == truth
                            ak += truth in ('positive', 'negative')
                            flips += truth in ('positive', 'negative') and state != truth
                        mask_checks += 1
                correct = correct_by_state['positive'] + correct_by_state['negative']
                known = support['positive'] + support['negative']
                m = result['masks'][policy]
                expected = {'attempted_checks': len(selected) * 4, 'accepted_determinate_proposals': accepted,
                    'accepted_known_proposals': ak, 'correct_known_proposals': correct,
                    'hard_positive_negative_flips': flips, 'positive_negative_reference_checks': known,
                    'correct_positive_proposals': correct_by_state['positive'],
                    'correct_negative_proposals': correct_by_state['negative'],
                    'determinate_on_uncertain_reference': commits['uncertain'],
                    'determinate_on_unknown_literal_reference': commits['unknown'],
                    'unavailable_reader_checks': unavailable}
                require(all(m[k] == v for k, v in expected.items())
                        and m['reference_support_by_state'] == {s: support[s] for s in STATES}
                        and m['proposal_coverage'] == accepted / (len(selected) * 4)
                        and m['correct_known_reference_recall'] == (correct / known if known else None)
                        and m['conditional_known_error_rate'] == (flips / ak if ak else None),
                        'independent_mask_counts_or_rates_mismatch')
                t = table_map[(view, domain, policy)]
                fields = {'attempted_checks': len(selected) * 4, 'determinate_proposals': accepted,
                    'correct_known_proposals': correct, 'known_reference_checks': known,
                    'hard_positive_negative_flips': flips, 'correct_positive_proposals': correct_by_state['positive'],
                    'positive_support': support['positive'], 'correct_negative_proposals': correct_by_state['negative'],
                    'negative_support': support['negative'], 'uncertain_support': support['uncertain'],
                    'determinate_on_uncertain': commits['uncertain'], 'literal_unknown_support': support['unknown'],
                    'determinate_on_literal_unknown': commits['unknown'], 'unavailable_reader_checks': unavailable}
                require(all(int(t[k]) == v for k, v in fields.items())
                        and float(t['known_proposal_recall']) == m['correct_known_reference_recall']
                        and float(t['proposal_coverage']) == m['proposal_coverage'], 'aggregate_csv_mismatch')
        lost_correct, shared_uncertain = Counter(), 0
        for ref in refs:
            for f in HEADS:
                pair = detail_map[(view, 'agree_radgraph_chexbert', ref['report_id'], f)]['state']
                triple = detail_map[(view, 'agree_all_three', ref['report_id'], f)]['state']
                truth = ref['finding_states'][f]
                if truth in ('positive', 'negative') and pair == truth and triple != truth:
                    lost_correct[truth] += 1
                shared_uncertain += truth == 'uncertain' and pair is not None and triple == pair
        tradeoffs[view] = {'correct_pair_proposals_lost_after_third_reader': dict(lost_correct),
            'uncertain_reference_commitments_retained_by_both': shared_uncertain,
            'automatic_mask_selection': False}
    audited_paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_three_reader_audit.py']
    audit_pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in audited_paths}
    for root in (RUN, NATIVE, PARENT):
        for path in root.rglob('*'):
            require(not path.is_symlink() and path.stat().st_gid in (96293, 65534)
                    and path.stat().st_mode & 0o7777 == (0o2770 if path.is_dir() else 0o660),
                    'private_artifact_modes_required')
        require(root.stat().st_gid in (96293, 65534) and root.stat().st_mode & 0o7777 == 0o2770,
                'private_root_mode_required')
    target_parent = BASE / 'manual_three_reader_audits'
    private_dir(target_parent)
    target = target_parent / 'numeric_12766754_001'
    private_dir(target, fresh=True)
    audit = {'schema_version': 'manual-three-reader-independent-arithmetic-audit-v1', 'status': 'passed',
        'model_calls': 0, 'completed_analysis_manifest_sha256': RUN_SHA,
        'native_run_manifest_sha256': NATIVE_SHA, 'reference_run_manifest_sha256': PARENT_SHA,
        'reader_matrix_checks': matrix_checks, 'mask_checks': mask_checks,
        'full_native_replay_bindings': 100, 'native_replay_differences': differences,
        'native_failed_reference_state_counts': dict(failure_reference_counts),
        'score_table_rows_checked': 42, 'pair_to_three_reader_tradeoffs': tradeoffs,
        'original_source_rows_or_reports_decoded': False, 'native_graph_text_decoded': False,
        'semantic_literal_gold_reprojected': False, 'source_pins_hashed_as_bytes_only': True,
        'protected_modes_and_group_verified': True, 'original_synthetic_bank_unchanged': True,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False,
        'audit_code_pins': audit_pins, 'runtime_seconds': time.monotonic() - started}
    require(all(sha256(WORKSPACE / n) == v for n, v in audit_pins.items()), 'audit_code_changed_during_execution')
    write_json(target / 'audit.json', audit)
    write_json(target / 'manifest.json', {'schema_version': 'manual-three-reader-audit-receipt-v1',
        'pins': audit_pins, 'completed_analysis_manifest_sha256': RUN_SHA,
        'artifacts': {'audit.json': sha256(target / 'audit.json')}, 'clinical_qualified': False})
    return target, audit


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_three_reader_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_three_reader_numeric_audit_failed',
                         'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

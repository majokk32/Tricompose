"""Independent numeric replay of frozen manual-literal reader results.

Consumes only protected hashes, projected states and character coordinates.
Never reopens original reports, raw annotation rows, native graph text or images.
Literal-head semantics are unit-tested, not independently reconstructed here.
"""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'manual_literal_reader_runs/manual100_12766754_001'
EXPECTED = '030ac27c56217db264bb09c93cba6a3ec2ae7b5322c17323a918fd89ded01a5f'
OLD = BASE / 'cxrgraph_extraction_runs/manual_xl_12766754_002'
OLD_SHA = 'a56743b0a3a940184faebbc7a39b0947d227d7f9ed3ffadc00592e10bed7898c'
FINDINGS = ('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax')
STATES = ('positive', 'negative', 'uncertain', 'unknown')
LABELS = {'positive': 'Observation::definitely present',
          'negative': 'Observation::definitely absent', 'uncertain': 'Observation::uncertain'}
POLICIES = {'radgraph': ('radgraph',), 'chexbert': ('chexbert',),
            'agree_radgraph_chexbert': ('radgraph', 'chexbert')}


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED
            and sha256(OLD / 'manifest.json') == OLD_SHA, 'pinned_complete_manual_reader_run_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'completed_output_artifact_changed')
    for name, value in manifest['pins'].items():
        require(sha256(WORKSPACE / name) == value, 'consumed_source_or_program_changed')
    references = json.loads((RUN / 'reference_projection.json').read_text())['records']
    predictions = {name: json.loads((RUN / (name + '_predictions.json')).read_text())['records']
                   for name in ('radgraph', 'chexbert')}
    result = json.loads((RUN / 'evaluation.json').read_text())
    domains = json.loads((RUN / 'domain_evaluation.json').read_text())
    details = json.loads((RUN / 'mask_details.json').read_text())['records']
    summary = json.loads((RUN / 'summary.json').read_text())
    ids = {f'report_{i:04d}' for i in range(100)}
    require(len(references) == 100 and {r['report_id'] for r in references} == ids,
            'all_hundred_reference_indices_required')
    indexed = {name: {r['report_id']: r for r in rows} for name, rows in predictions.items()}
    require(all(len(rows) == 100 and set(indexed[name]) == ids for name, rows in predictions.items()),
            'all_hundred_frozen_predictions_required')
    require(summary['encoder_examples'] == 100 and summary['forward_batches'] == 25
            and summary['new_radgraph_forward_calls'] == 0
            and summary['truncation_used'] is False
            and summary['predictions_fsynced_before_reference_projection'] is True,
            'fixed_completed_call_accounting_required')
    old = json.loads((OLD / 'manifest.json').read_text())
    require(sha256(OLD / 'comparisons.json') == old['artifacts']['comparisons.json'],
            'unchanged_original_manual_span_coordinates_required')
    comparisons = {r['report_id']: r for r in json.loads((OLD / 'comparisons.json').read_text())}
    evidence_checks = 0
    for ref in references:
        gold = {tuple(e) for e in comparisons[ref['report_id']]['gold_character_entities']}
        for name in predictions:
            require(indexed[name][ref['report_id']]['source_sha256'] == ref['source_sha256'],
                    'shared_input_source_receipts_required')
        for finding in FINDINGS:
            values = set()
            for evidence in ref['human_span_references'][finding]:
                start, end = evidence['char_start'], evidence['char_end_exclusive']
                state = evidence['human_native_state']
                require((start, end, LABELS[state]) in gold
                        and start <= evidence['head_start'] < evidence['head_end_exclusive'] <= end,
                        'derived_human_state_must_bind_exact_manual_span')
                values.add(state)
                evidence_checks += 1
            expected = 'unknown' if not values else 'uncertain' if 'uncertain' in values or len(values) > 1 else next(iter(values))
            require(ref['finding_states'][finding] == expected, 'independent_human_state_pool_mismatch')
    require(len(details) == 1200, 'all_fixed_three_mask_checks_required')
    detail_map = {(r['policy'], r['report_id'], r['finding']): r for r in details}
    require(len(detail_map) == len(details), 'unique_mask_details_required')
    matrix_checks = mask_checks = 0
    table = []
    for domain, evaluation in (('all', result), *domains.items()):
        selected = [r for r in references if domain == 'all' or r['source_domain'] == domain]
        require(len(selected) == (100 if domain == 'all' else 50), 'unchanged_disjoint_domains_required')
        for reader, metrics in evaluation['readers'].items():
            by_finding = {}
            for finding in FINDINGS:
                matrix = {t: dict.fromkeys((*STATES, 'unavailable'), 0) for t in STATES}
                for ref in selected:
                    row = indexed[reader][ref['report_id']]
                    predicted = row['finding_states'][finding] if row['status'] == 'complete' else 'unavailable'
                    matrix[ref['finding_states'][finding]][predicted] += 1
                    matrix_checks += 1
                require(matrix == metrics['per_finding'][finding]['confusion_matrix'],
                        'independent_per_head_matrix_mismatch')
                by_finding[finding] = matrix
            overall = {t: {p: sum(m[t][p] for m in by_finding.values())
                           for p in (*STATES, 'unavailable')} for t in STATES}
            require(overall == metrics['overall']['confusion_matrix'], 'independent_overall_matrix_mismatch')
            recorded = metrics['overall']
            known = sum(sum(overall[t].values()) for t in ('positive', 'negative'))
            correct = overall['positive']['positive'] + overall['negative']['negative']
            flips = overall['positive']['negative'] + overall['negative']['positive']
            require(recorded['positive_negative_reference_checks'] == known
                    and recorded['correct_positive_negative_states'] == correct
                    and recorded['hard_positive_negative_flips'] == flips
                    and recorded['failure_aware_positive_negative_recovery'] == correct / known,
                    'independent_primary_reader_counts_mismatch')
        for policy, readers in POLICIES.items():
            accepted = correct = known_accepted = flips = uncertain = unknown = known = 0
            statuses = Counter()
            by_state = Counter()
            for ref in selected:
                for finding in FINDINGS:
                    rows = [indexed[name][ref['report_id']] for name in readers]
                    values = [row['finding_states'][finding] for row in rows] if all(
                        row['status'] == 'complete' for row in rows) else []
                    if not values:
                        state, status = None, 'unavailable_reader'
                    elif 'unknown' in values:
                        state, status = None, 'abstain_unknown'
                    elif 'uncertain' in values:
                        state, status = None, 'abstain_uncertain'
                    elif len(set(values)) != 1:
                        state, status = None, 'abstain_disagreement'
                    else:
                        state, status = values[0], 'accepted_determinate_proposal'
                    expected = ref['finding_states'][finding]
                    statuses[status] += 1
                    by_state[(expected, 'support')] += 1
                    is_known = expected in ('positive', 'negative')
                    known += is_known
                    if state is not None:
                        accepted += 1
                        known_accepted += is_known
                        correct += is_known and state == expected
                        flips += is_known and state != expected
                        uncertain += expected == 'uncertain'
                        unknown += expected == 'unknown'
                        by_state[(expected, 'correct')] += state == expected
                    detail = detail_map[(policy, ref['report_id'], finding)]
                    require(detail['state'] == state and detail['status'] == status
                            and detail['human_literal_reference_state'] == expected,
                            'independent_mask_decision_or_reference_mismatch')
                    mask_checks += 1
            row = evaluation['masks'][policy]
            require((accepted, known_accepted, correct, flips, uncertain, unknown, known) == (
                row['accepted_determinate_proposals'], row['accepted_known_proposals'], row['correct_known_proposals'],
                row['hard_positive_negative_flips'], row['determinate_on_uncertain_reference'],
                row['determinate_on_unknown_literal_reference'], row['positive_negative_reference_checks'])
                and dict(statuses) == row['decision_status_counts'], 'independent_mask_counts_mismatch')
            require(row['proposal_coverage'] == accepted / (len(selected) * 4)
                    and row['conditional_known_error_rate'] == (flips / known_accepted if known_accepted else None)
                    and row['correct_known_reference_recall'] == correct / known,
                    'independent_risk_coverage_rate_mismatch')
            table.append({'domain': domain, 'policy': policy, 'attempted_reports': len(selected),
                'attempted_checks': len(selected) * 4, 'determinate_proposals': accepted,
                'positive_support': by_state[('positive', 'support')],
                'correct_positive_proposals': by_state[('positive', 'correct')],
                'negative_support': by_state[('negative', 'support')],
                'correct_negative_proposals': by_state[('negative', 'correct')],
                'uncertain_reference_support': by_state[('uncertain', 'support')],
                'determinate_on_uncertain_reference': uncertain,
                'unknown_literal_reference_support': by_state[('unknown', 'support')],
                'determinate_on_unknown_literal_reference': unknown,
                'hard_positive_negative_flips': flips, 'known_proposal_recall': correct / known,
                'proposal_coverage': accepted / (len(selected) * 4), 'clinical_qualified': False})
    parent = BASE / 'manual_literal_reader_audits'
    private_dir(parent)
    target = parent / 'numeric_12766754_001'
    private_dir(target, fresh=True)
    audit = {'schema_version': 'manual-literal-reader-count-audit-v1', 'status': 'passed',
        'completed_run_manifest_sha256': EXPECTED, 'replayed_reader_matrix_checks': matrix_checks,
        'replayed_mask_decisions': mask_checks, 'human_exact_span_evidence_bindings_checked': evidence_checks,
        'raw_reports_annotations_images_or_native_graph_text_read': False,
        'literal_regex_semantics_independently_reconstructed': False,
        'clinical_qualified': False, 'new_model_calls': 0, 'selection_changed': False,
        'regeneration_authorized': False, 'worker_sha256': sha256(Path(__file__)),
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    write_json(target / 'audit.json', audit)
    with (target / 'score_table.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        stream.flush()
        os.fsync(stream.fileno())
    (target / 'score_table.csv').chmod(0o660)
    write_json(target / 'manifest.json', {'schema_version': 'manual-literal-reader-count-audit-receipt-v1',
        'completed_run_manifest_sha256': EXPECTED, 'worker_sha256': sha256(Path(__file__)),
        'artifacts': {p.name: sha256(p) for p in sorted(target.iterdir())}})
    return target, audit


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_manual_reader_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        target, audit = execute()
        print(json.dumps({'status': 'protected_manual_reader_numeric_audit_passed',
            'runtime_seconds': round(audit['runtime_seconds'], 3),
            'peak_rss_gib': round(audit['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_manual_reader_numeric_audit_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

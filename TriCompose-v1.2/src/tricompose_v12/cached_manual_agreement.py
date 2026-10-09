"""Count-only real-manual agreement audit, never a learned/clinical policy.

No source reports, annotation rows, model calls or per-record gold reconstruction.
Marginal gold confusions generally do NOT identify agreement-mask accuracy.
Only an identity confusion on the exact shared cohort permits the special
aggregate count proof below. Otherwise the joint reference metrics stay null.
"""
from collections import Counter
import re

from .assertion_agreement_diagnostic import decision
from .radgraph_reference_contract import require

FINDINGS = ('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax')
STATES = ('positive', 'negative', 'uncertain', 'unknown')
READERS = ('chexbert', 'qwen_span_v2')
POLICIES = {'chexbert': ('chexbert',), 'qwen_span_v2': ('qwen_span_v2',),
            'agree_chexbert_qwen_span_v2': READERS}
DETERMINATE = ('positive', 'negative')


def matrix():
    return {a: dict.fromkeys(STATES, 0) for a in STATES}


def checked_matrix(value, total):
    require(isinstance(value, dict) and set(value) == set(STATES), 'four_reference_states_required')
    for row in value.values():
        require(isinstance(row, dict) and set(row) == set(STATES), 'four_prediction_states_required')
        require(all(type(n) is int and n >= 0 for n in row.values()), 'nonnegative_integer_counts_required')
    require(sum(sum(row.values()) for row in value.values()) == total,
            'complete_cohort_confusion_denominator_required')
    return value


def single_metrics(value):
    known = sum(value[t][p] for t in DETERMINATE for p in STATES)
    accepted_known = sum(value[t][p] for t in DETERMINATE for p in DETERMINATE)
    correct = sum(value[t][t] for t in DETERMINATE)
    flips = accepted_known - correct
    return {'known_positive_negative_reference_checks': known,
        'accepted_known_reference_proposals': accepted_known,
        'correct_known_reference_proposals': correct,
        'hard_positive_negative_flips': flips,
        'conditional_known_error_rate': flips / accepted_known if accepted_known else None,
        'known_reference_proposal_coverage': accepted_known / known if known else None,
        'correct_known_reference_recall': correct / known if known else None,
        'determinate_on_unknown_reference': sum(value['unknown'][p] for p in DETERMINATE),
        'determinate_on_uncertain_reference': sum(value['uncertain'][p] for p in DETERMINATE),
        'uncertain_reference_checks': sum(value['uncertain'].values()),
        'unknown_reference_checks': sum(value['unknown'].values()),
        'reference_metrics_status': 'existing_manual_confusion_count_replay',
        'unknown_promotions_are_clinical_errors': False}


def evaluate(predictions, confusions):
    require(set(predictions) == set(confusions) == set(READERS), 'exact_two_cached_readers_required')
    indexed = {}
    for name, rows in predictions.items():
        require(isinstance(rows, list) and rows, 'nonempty_full_annotation_inventory_required')
        require(all(isinstance(r, dict) and isinstance(r.get('item_id'), str)
                    and re.fullmatch(r'item_[0-9]{4}', r['item_id']) for r in rows),
                'opaque_annotation_indices_required')
        require(len({r['item_id'] for r in rows}) == len(rows), 'unique_prediction_inventory_required')
        indexed[name] = {r['item_id']: r for r in rows}
        for row in rows:
            require(isinstance(row.get('status'), str), 'explicit_prediction_status_required')
            if row['status'] == 'complete':
                require(isinstance(row.get('finding_states'), dict)
                        and all(row['finding_states'].get(f) in STATES for f in FINDINGS),
                        'complete_four_head_states_required')
                require(all(isinstance(row.get(key), str) and re.fullmatch(r'[a-f0-9]{64}', row[key])
                        for key in ('selected_input_sha256', 'source_report_sha256')),
                        'complete_source_and_impression_hashes_required')
            else:
                require(row.get('finding_states') is None, 'failed_or_skipped_is_not_unknown_success')
    ids = set(indexed[READERS[0]])
    require(ids == set(indexed[READERS[1]]), 'all_annotation_entries_retained_required')
    complete_ids = {name: {key for key, row in rows.items() if row['status'] == 'complete'}
                    for name, rows in indexed.items()}
    require(complete_ids[READERS[0]] == complete_ids[READERS[1]],
            'exact_same_completed_cohort_required')
    completed = sorted(complete_ids[READERS[0]])
    require(completed, 'nonempty_shared_complete_cohort_required')
    for key in completed:
        a, b = (indexed[name][key] for name in READERS)
        require(all(a[field] == b[field] for field in ('selected_input_sha256', 'source_report_sha256')),
                'same_original_report_and_impression_bytes_required')
    observed = {f: matrix() for f in FINDINGS}
    counts = {policy: {f: Counter() for f in FINDINGS} for policy in POLICIES}
    for key in completed:
        rows = {name: indexed[name][key] for name in READERS}
        for finding in FINDINGS:
            observed[finding][rows['chexbert']['finding_states'][finding]][
                rows['qwen_span_v2']['finding_states'][finding]] += 1
            for policy, readers in POLICIES.items():
                counts[policy][finding][decision(rows, readers, finding)['status']] += 1
    certified = {}
    for finding in FINDINGS:
        require(all(set(confusions[name]) == set(FINDINGS) for name in READERS),
                'fixed_four_head_confusions_required')
        cm = {name: checked_matrix(confusions[name][finding], len(completed)) for name in READERS}
        require(all(sum(cm['chexbert'][t].values()) == sum(cm['qwen_span_v2'][t].values())
                    for t in STATES), 'same_manual_reference_marginals_required')
        for name in READERS:
            native = Counter(indexed[name][key]['finding_states'][finding] for key in completed)
            require(all(sum(cm[name][t][p] for t in STATES) == native[p] for p in STATES),
                    'native_prediction_marginals_match_frozen_confusion_required')
        identity = all(cm['chexbert'][t][p] == 0 for t in STATES for p in STATES if t != p)
        certified[finding] = identity
        if identity:
            # Aggregate consequence of the pinned independent manual summary.
            # No CheXbert output is installed as a new per-record gold label.
            require(observed[finding] == cm['qwen_span_v2'],
                    'identity_count_proof_must_replay_full_joint_matrix')
    results = {}
    for policy, readers in POLICIES.items():
        by_finding = {}
        for finding in FINDINGS:
            statuses = dict(counts[policy][finding])
            accepted = statuses.get('accepted_determinate_proposal', 0)
            row = {'attempted_shared_complete_checks': len(completed),
                'accepted_determinate_proposals': accepted,
                'proposal_coverage': accepted / len(completed), 'decision_status_counts': statuses}
            if len(readers) == 1:
                row.update(single_metrics(confusions[readers[0]][finding]))
            else:
                parent = single_metrics(confusions['qwen_span_v2'][finding])
                row.update({key: None for key in parent})
                for key in ('known_positive_negative_reference_checks', 'uncertain_reference_checks',
                            'unknown_reference_checks'):
                    row[key] = parent[key]
                row['unknown_promotions_are_clinical_errors'] = False
                if certified[finding]:
                    correct = parent['correct_known_reference_proposals']
                    known = parent['known_positive_negative_reference_checks']
                    require(accepted == correct, 'identity_mask_count_proof_required')
                    row.update(accepted_known_reference_proposals=correct,
                        correct_known_reference_proposals=correct, hard_positive_negative_flips=0,
                        conditional_known_error_rate=0.0 if correct else None,
                        known_reference_proposal_coverage=correct / known if known else None,
                        correct_known_reference_recall=correct / known if known else None,
                        determinate_on_unknown_reference=0, determinate_on_uncertain_reference=0,
                        reference_metrics_status='identity_baseline_aggregate_count_proof')
                else:
                    row['reference_metrics_status'] = 'not_identifiable_from_cached_marginals'
            by_finding[finding] = row
        total = {'attempted_shared_complete_checks': len(completed) * len(FINDINGS),
            'accepted_determinate_proposals': sum(r['accepted_determinate_proposals'] for r in by_finding.values()),
            'decision_status_counts': dict(sum((Counter(r['decision_status_counts']) for r in by_finding.values()), Counter()))}
        total['proposal_coverage'] = total['accepted_determinate_proposals'] / total['attempted_shared_complete_checks']
        for field in ('known_positive_negative_reference_checks', 'accepted_known_reference_proposals',
                      'correct_known_reference_proposals', 'hard_positive_negative_flips',
                      'determinate_on_unknown_reference', 'determinate_on_uncertain_reference',
                      'uncertain_reference_checks', 'unknown_reference_checks'):
            values = [r[field] for r in by_finding.values()]
            total[field] = sum(values) if all(v is not None for v in values) else None
        nk, ak, correct, flips = (total[k] for k in ('known_positive_negative_reference_checks',
            'accepted_known_reference_proposals', 'correct_known_reference_proposals', 'hard_positive_negative_flips'))
        total.update(known_reference_proposal_coverage=ak / nk if ak is not None and nk else None,
            correct_known_reference_recall=correct / nk if correct is not None and nk else None,
            conditional_known_error_rate=flips / ak if flips is not None and ak else None,
            unknown_promotions_are_clinical_errors=False)
        results[policy] = {'readers': list(readers), 'overall': total, 'per_finding': by_finding}
    return {'schema_version': 'cached-manual-reader-mask-count-audit-v1',
        'annotation_inventory': len(ids), 'shared_complete_reports': len(completed),
        'not_comparable_inventory_entries': len(ids) - len(completed),
        'reader_execution_status_counts': {name: dict(Counter(r['status'] for r in rows))
                                           for name, rows in predictions.items()},
        'shared_prediction_joint_matrices': observed, 'identity_count_proof_by_finding': certified,
        'policies': results, 'per_record_gold_reconstructed': False,
        'unknown_agreement_is_factual_support': False, 'independent_truth_votes': False,
        'checkpoint_training_overlap_status': 'unverified', 'untouched_final_test': False,
        'span_or_temporal_scope_gold': False, 'image_or_ehr_truth': False,
        'clinical_qualified': False, 'best_policy_selected': False,
        'selection_changed': False, 'regeneration_authorized': False, 'thresholds_fitted': False}

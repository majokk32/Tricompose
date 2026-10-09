"""Seven fixed masks on a human-literal reference; dependent readers, not votes."""
from itertools import combinations

from .assertion_agreement_diagnostic import decision
from .manual_literal_assertions import FINDINGS, STATES, statistics
from .radgraph_reference_contract import require

READERS = ('radgraph', 'chexbert', 'chexpert_negbio')
POLICIES = {name: (name,) for name in READERS}
POLICIES.update({'agree_' + '_'.join(pair): pair for pair in combinations(READERS, 2)})
POLICIES['agree_all_three'] = READERS


def evaluate(references, predictions):
    require(set(predictions) == set(READERS) and references, 'fixed_three_readers_and_reference_inventory_required')
    ids = {r['report_id'] for r in references}
    require(len(ids) == len(references), 'unique_reference_inventory_required')
    indexed = {}
    for name, records in predictions.items():
        require(len(records) == len(ids) and {r['report_id'] for r in records} == ids,
                'all_failed_or_complete_predictions_retained_required')
        indexed[name] = {r['report_id']: r for r in records}
        for ref in references:
            require(set(ref['finding_states']) == set(FINDINGS)
                    and all(v in STATES for v in ref['finding_states'].values()), 'complete_human_reference_states_required')
            row = indexed[name][ref['report_id']]
            require(row['source_sha256'] == ref['source_sha256'], 'same_source_bytes_required')
            require(row['status'] in ('complete', 'failed_unavailable'), 'explicit_reader_availability_required')
            if row['status'] == 'complete':
                require(set(row['finding_states']) == set(FINDINGS)
                        and all(v in STATES for v in row['finding_states'].values()), 'complete_native_four_states_required')
            else:
                require(row['finding_states'] is None, 'failed_is_null_not_unknown_success')
    reader_metrics = {}
    for name in READERS:
        by_head = {f: [(ref['finding_states'][f], indexed[name][ref['report_id']]['finding_states'][f]
            if indexed[name][ref['report_id']]['status'] == 'complete' else 'unavailable')
            for ref in references] for f in FINDINGS}
        reader_metrics[name] = {'overall': statistics([v for rows in by_head.values() for v in rows]),
                               'per_finding': {f: statistics(rows) for f, rows in by_head.items()}}
    results, details = {}, []
    for policy, readers in POLICIES.items():
        rows = []
        for ref in references:
            native = {name: indexed[name][ref['report_id']] for name in READERS}
            for finding in FINDINGS:
                rows.append({'report_id': ref['report_id'], 'finding': finding,
                    'human_literal_reference_state': ref['finding_states'][finding],
                    **decision(native, readers, finding)})
        accepted = [r for r in rows if r['state'] is not None]
        known = [r for r in rows if r['human_literal_reference_state'] in ('positive', 'negative')]
        ak = [r for r in accepted if r['human_literal_reference_state'] in ('positive', 'negative')]
        correct = sum(r['state'] == r['human_literal_reference_state'] for r in ak)
        counts = {s: sum(r['human_literal_reference_state'] == s for r in rows) for s in STATES}
        results[policy] = {'readers': list(readers), 'attempted_checks': len(rows),
            'accepted_determinate_proposals': len(accepted), 'proposal_coverage': len(accepted) / len(rows),
            'positive_negative_reference_checks': len(known), 'accepted_known_proposals': len(ak),
            'correct_known_proposals': correct, 'hard_positive_negative_flips': len(ak) - correct,
            'conditional_known_error_rate': (len(ak) - correct) / len(ak) if ak else None,
            'correct_known_reference_recall': correct / len(known) if known else None,
            'reference_support_by_state': counts,
            'correct_positive_proposals': sum(r['human_literal_reference_state'] == 'positive' and r['state'] == 'positive' for r in accepted),
            'correct_negative_proposals': sum(r['human_literal_reference_state'] == 'negative' and r['state'] == 'negative' for r in accepted),
            'determinate_on_uncertain_reference': sum(r['human_literal_reference_state'] == 'uncertain' for r in accepted),
            'determinate_on_unknown_literal_reference': sum(r['human_literal_reference_state'] == 'unknown' for r in accepted),
            'unavailable_reader_checks': sum(r['status'] == 'unavailable_reader' for r in rows),
            'unknown_promotions_are_clinical_errors': False, 'independent_truth_votes': False,
            'clinical_qualified': False}
        details.extend({'policy': policy, **r} for r in rows)
    return {'schema_version': 'manual-literal-three-reader-seven-mask-diagnostic-v1',
        'attempted_reports': len(ids), 'reader_metrics': reader_metrics, 'masks': results,
        'reference_role': 'fixed_human_literal_projection_not_full_official_report_gold',
        'best_policy_selected': False, 'thresholds_fitted': False, 'clinical_qualified': False,
        'scope_image_ehr_truth_verified': False, 'checkpoint_training_overlap': 'unresolved',
        'selection_changed': False, 'regeneration_authorized': False}, details

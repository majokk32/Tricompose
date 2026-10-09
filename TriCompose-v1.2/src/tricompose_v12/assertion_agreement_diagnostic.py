"""Fixed reader-agreement risk/coverage masks, not a clinical decision policy.

No fitting, model calls, thresholds, text parsing or truth votes. Evaluate all
seven masks; never pick one by its result. Unknown agreement cannot authorize
a determinate statement. These language endpoints do not authorize repairs.
"""
from itertools import combinations

from .radgraph_reference_contract import require

READERS = ('radgraph', 'chexbert', 'qwen_span_v2')
POLICIES = {name: (name,) for name in READERS}
POLICIES.update({'agree_' + '_'.join(pair): pair for pair in combinations(READERS, 2)})
POLICIES['agree_all_three'] = READERS


def decision(records, readers, finding):
    values = []
    for name in readers:
        row = records[name]
        require(row['status'] in ('complete', 'failed_unavailable'), 'declared_prediction_status_required')
        if row['status'] != 'complete':
            return {'status': 'unavailable_reader', 'state': None}
        state = row['finding_states'][finding]
        require(state in ('positive', 'negative', 'uncertain', 'unknown'), 'four_native_states_required')
        values.append(state)
    if 'unknown' in values:
        return {'status': 'abstain_unknown', 'state': None}
    if 'uncertain' in values:
        return {'status': 'abstain_uncertain', 'state': None}
    if len(set(values)) != 1:
        return {'status': 'abstain_disagreement', 'state': None}
    return {'status': 'accepted_determinate_proposal', 'state': values[0]}


def evaluate(references, predictions):
    require(set(predictions) == set(READERS), 'all_three_readers_required')
    ids = {r['item_id'] for r in references}
    require(len(ids) == len(references), 'unique_reference_inventory_required')
    indexed = {}
    for name, records in predictions.items():
        require(len(records) == len(ids) and {r['item_id'] for r in records} == ids,
                'failed_predictions_must_remain_attempted')
        indexed[name] = {r['item_id']: r for r in records}
        require(all(indexed[name][r['item_id']]['report_sha256'] == r['report_sha256'] for r in references),
                'same_input_report_hash_binding_required')
    results = {}
    details = []
    for name, readers in POLICIES.items():
        rows = []
        for ref in references:
            records = {reader: indexed[reader][ref['item_id']] for reader in READERS}
            for finding in ref['evaluation_findings']:
                value = decision(records, readers, finding)
                rows.append({'item_id': ref['item_id'], 'finding': finding,
                    'authored_reference_state': ref['expected_states'][finding], **value})
        accepted = [r for r in rows if r['state'] is not None]
        known = [r for r in rows if r['authored_reference_state'] in ('positive', 'negative')]
        known_accepted = [r for r in accepted if r['authored_reference_state'] in ('positive', 'negative')]
        correct = sum(r['state'] == r['authored_reference_state'] for r in accepted)
        hard_flips = sum(r['state'] != r['authored_reference_state'] for r in known_accepted)
        results[name] = {'readers': list(readers), 'attempted_designated_checks': len(rows),
            'accepted_determinate_proposals': len(accepted),
            'proposal_coverage': len(accepted) / len(rows),
            'accepted_authored_state_errors': len(accepted) - correct,
            'accepted_authored_error_rate': (len(accepted) - correct) / len(accepted) if accepted else None,
            'correct_determinate_proposals': correct,
            'hard_positive_negative_flips': hard_flips,
            'determinate_on_authored_uncertain_or_unknown': len(accepted) - len(known_accepted),
            'known_determinate_reference_checks': len(known),
            'known_reference_proposal_coverage': len(known_accepted) / len(known) if known else None,
            'correct_determinate_recall': correct / len(known) if known else None,
            'unavailable_reader_checks': sum(r['status'] == 'unavailable_reader' for r in rows),
            'abstained_checks': sum(r['status'].startswith('abstain_') for r in rows),
            'uncertain_or_unknown_is_not_an_agreement': True,
            'clinical_qualified': False, 'thresholds_fitted': False,
            'independent_truth_votes': False, 'regeneration_authorized': False}
        details.extend({'policy': name, **r} for r in rows)
    return {'schema_version': 'authored-assertion-agreement-risk-coverage-v1',
        'policies': results, 'clinical_qualified': False,
        'post_hoc_development_diagnostic': True, 'best_policy_selected': False,
        'semantic_scope_verified': False}, details

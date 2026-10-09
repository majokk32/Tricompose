"""Fixed literal-head projection of human spans and reader risk/coverage.

Not an official CheXpert report-label dataset or a clinical truth scorer.
The unchanged four literal patterns exclude synonyms and generic normality.
Human character spans, not model predictions, supply the reference states.
Unknown means no annotated literal-head observation in this projection only.
Temporal/person scope and image/EHR factuality remain unverified.
"""
from collections import Counter
import hashlib
import re

from .assertion_agreement_diagnostic import decision
from .entity_gold_contract import STATES as OBSERVATIONS
from .radgraph_assertion_readout import PATTERNS, pool
from .radgraph_reference_contract import require

VERSION = 'manual-literal-four-head-assertion-diagnostic-v1'
FINDINGS = tuple(PATTERNS)
STATES = ('positive', 'negative', 'uncertain', 'unknown')
READERS = ('radgraph', 'chexbert')
POLICIES = {'radgraph': ('radgraph',), 'chexbert': ('chexbert',),
            'agree_radgraph_chexbert': READERS}


def project_reference(text, entities, source_sha256):
    require(isinstance(text, str) and 0 < len(text) <= 100000
            and text == ' '.join(text.split()), 'canonical_released_source_required')
    require(hashlib.sha256(text.encode()).hexdigest() == source_sha256,
            'human_span_source_hash_binding_required')
    require(isinstance(entities, list) and len(entities) <= 4096, 'bounded_human_character_entities_required')
    seen = set()
    for entity in entities:
        require(isinstance(entity, list) and len(entity) == 3
                and all(type(v) is int for v in entity[:2])
                and 0 <= entity[0] < entity[1] <= len(text)
                and entity[2] in (*OBSERVATIONS, 'Anatomy::definitely present'),
                'exact_human_character_entity_required')
        require(tuple(entity[:2]) not in seen, 'unique_human_spans_required')
        seen.add(tuple(entity[:2]))
    evidence = {f: [] for f in FINDINGS}
    mentions = dict.fromkeys(FINDINGS, 0)
    for finding, pattern in PATTERNS.items():
        for mention in re.finditer(pattern, text, flags=re.IGNORECASE):
            left, right = mention.span('head')
            mentions[finding] += 1
            for start, end, label in entities:
                if label in OBSERVATIONS and start <= left and right <= end:
                    evidence[finding].append({'char_start': start, 'char_end_exclusive': end,
                        'head_start': left, 'head_end_exclusive': right,
                        'human_native_state': OBSERVATIONS[label]})
    return {'schema_version': VERSION + '-reference', 'source_sha256': source_sha256,
        'finding_states': {f: pool(e['human_native_state'] for e in evidence[f]) for f in FINDINGS},
        'literal_mention_counts': mentions, 'human_span_references': evidence,
        'unknown_means': 'no_annotated_literal_head_observation_not_clinical_absence',
        'normality_expanded': False, 'synonyms_added': False, 'scope_verified': False,
        'official_report_label_reference': False, 'clinical_qualified': False}


def statistics(rows):
    matrix = {a: dict.fromkeys((*STATES, 'unavailable'), 0) for a in STATES}
    for expected, actual in rows:
        matrix[expected][actual] += 1
    explicit = sum(matrix[t][p] for t in ('positive', 'negative') for p in (*STATES, 'unavailable'))
    correct = matrix['positive']['positive'] + matrix['negative']['negative']
    flips = matrix['positive']['negative'] + matrix['negative']['positive']
    uncertain = sum(matrix['uncertain'].values())
    return {'attempted_checks': len(rows), 'confusion_matrix': matrix,
        'positive_negative_reference_checks': explicit,
        'correct_positive_negative_states': correct,
        'failure_aware_positive_negative_recovery': correct / explicit if explicit else None,
        'hard_positive_negative_flips': flips,
        'uncertain_reference_checks': uncertain,
        'uncertain_reference_recovered': matrix['uncertain']['uncertain'],
        'failure_aware_uncertain_recovery': matrix['uncertain']['uncertain'] / uncertain if uncertain else None,
        'determinate_on_uncertain_reference': matrix['uncertain']['positive'] + matrix['uncertain']['negative'],
        'unknown_literal_reference_checks': sum(matrix['unknown'].values()),
        'determinate_on_unknown_literal_reference': matrix['unknown']['positive'] + matrix['unknown']['negative'],
        'unavailable_checks': sum(matrix[t]['unavailable'] for t in STATES),
        'unknown_promotions_are_clinical_errors': False,
        'clinical_qualified': False}


def evaluate(references, predictions):
    require(isinstance(references, list) and references and set(predictions) == set(READERS),
            'fixed_two_reader_complete_reference_inventory_required')
    ids = {r['report_id'] for r in references}
    require(len(ids) == len(references) and all(re.fullmatch(r'report_[0-9]{4}', key) for key in ids),
            'unique_opaque_manual_report_indices_required')
    for ref in references:
        require(set(ref['finding_states']) == set(FINDINGS)
                and all(v in STATES for v in ref['finding_states'].values()),
                'complete_human_projected_four_states_required')
    indexed = {}
    for name, records in predictions.items():
        require(len(records) == len(ids) and {r['report_id'] for r in records} == ids,
                'failed_predictions_retained_in_denominator_required')
        indexed[name] = {r['report_id']: r for r in records}
        for ref in references:
            row = indexed[name][ref['report_id']]
            require(row['source_sha256'] == ref['source_sha256'], 'same_literal_source_hash_required')
            require(row['status'] in ('complete', 'failed_unavailable'), 'explicit_prediction_availability_required')
            if row['status'] == 'complete':
                require(isinstance(row['finding_states'], dict)
                        and set(row['finding_states']) == set(FINDINGS)
                        and all(v in STATES for v in row['finding_states'].values()),
                        'complete_native_four_states_required')
            else:
                require(row['finding_states'] is None, 'failed_is_not_all_unknown_success')
    metrics = {}
    for name in READERS:
        by_head = {f: [(r['finding_states'][f], indexed[name][r['report_id']]['finding_states'][f]
                       if indexed[name][r['report_id']]['status'] == 'complete' else 'unavailable')
                      for r in references] for f in FINDINGS}
        metrics[name] = {'overall': statistics([pair for rows in by_head.values() for pair in rows]),
                         'per_finding': {f: statistics(rows) for f, rows in by_head.items()}}
    masks = {}
    details = []
    for policy, readers in POLICIES.items():
        rows = []
        for ref in references:
            native = {name: indexed[name][ref['report_id']] for name in READERS}
            for finding in FINDINGS:
                value = decision(native, readers, finding)
                rows.append({'report_id': ref['report_id'], 'finding': finding,
                    'human_literal_reference_state': ref['finding_states'][finding], **value})
        accepted = [r for r in rows if r['state'] is not None]
        known = [r for r in rows if r['human_literal_reference_state'] in ('positive', 'negative')]
        accepted_known = [r for r in accepted if r['human_literal_reference_state'] in ('positive', 'negative')]
        flips = sum(r['state'] != r['human_literal_reference_state'] for r in accepted_known)
        correct = len(accepted_known) - flips
        masks[policy] = {'attempted_checks': len(rows), 'accepted_determinate_proposals': len(accepted),
            'proposal_coverage': len(accepted) / len(rows),
            'positive_negative_reference_checks': len(known),
            'accepted_known_proposals': len(accepted_known), 'correct_known_proposals': correct,
            'hard_positive_negative_flips': flips,
            'conditional_known_error_rate': flips / len(accepted_known) if accepted_known else None,
            'correct_known_reference_recall': correct / len(known) if known else None,
            'determinate_on_uncertain_reference': sum(r['human_literal_reference_state'] == 'uncertain' for r in accepted),
            'determinate_on_unknown_literal_reference': sum(r['human_literal_reference_state'] == 'unknown' for r in accepted),
            'decision_status_counts': dict(Counter(r['status'] for r in rows)),
            'unknown_promotions_are_clinical_errors': False, 'clinical_qualified': False}
        details.extend({'policy': policy, **r} for r in rows)
    reference_counts = {f: dict(Counter(r['finding_states'][f] for r in references)) for f in FINDINGS}
    return {'schema_version': VERSION + '-evaluation', 'attempted_reports': len(ids),
        'reference_state_counts_by_finding': reference_counts, 'readers': metrics, 'masks': masks,
        'reference_role': 'human_span_literal_projection_not_full_clinical_finding_gold',
        'checkpoint_training_overlap_status': 'unresolved', 'untouched_final_test': False,
        'scope_image_ehr_truth_verified': False, 'clinical_qualified': False,
        'best_policy_selected': False, 'thresholds_fitted': False,
        'selection_changed': False, 'regeneration_authorized': False}, details

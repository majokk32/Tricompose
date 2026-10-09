"""Literal human opacity states versus a broader frozen report-label head.

This is a vocabulary/scope-limited development diagnostic, not image truth or
clinical qualification. Missing predictions and references stay in denominator.
Unknown literal references do not mean disease absent; no overall accuracy/F1
is used to hide this mismatch. Tests use wholly invented reference records.
"""
from collections import Counter
import re

VERSION = 'manual-opacity-chexbert-diagnostic-v1'
STATES = ('positive', 'negative', 'uncertain', 'unknown')
HEADS = ('enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity',
         'lung_lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis',
         'pneumothorax', 'pleural_effusion', 'pleural_other', 'fracture',
         'support_devices', 'no_finding')


def require(condition, code):
    if not condition:
        raise ValueError(code)


def evaluate(references, predictions):
    require(isinstance(references, list) and isinstance(predictions, list)
            and 0 < len(references) == len(predictions) <= 1024,
            'complete_fixed_reference_and_prediction_inventory_required')
    require(len({p['report_id'] for p in predictions}) == len(predictions),
            'unique_prediction_slots_required')
    by_id = {p['report_id']: p for p in predictions}
    matrix = {truth: {pred: 0 for pred in (*STATES, 'prediction_unavailable')}
              for truth in (*STATES, 'reference_unavailable')}
    details = []
    for index, ref in enumerate(references):
        key = f'report_{index:04d}'
        require(ref['report_id'] == key and ref['source_index'] == index and key in by_id,
                'all_release_slots_in_fixed_order_required')
        pred = by_id[key]
        require(isinstance(ref['source_sha256'], str)
                and re.fullmatch(r'[a-f0-9]{64}', ref['source_sha256']),
                'sealed_reference_text_hash_required')
        require(pred['source_sha256'] == ref['source_sha256'] or
                (pred['source_sha256'] is None and pred['status'] == 'failed_unavailable'),
                'same_source_report_for_reference_and_prediction_required')
        require(ref['status'] in ('complete', 'failed_unavailable') and
                pred['status'] in ('complete', 'failed_unavailable'),
                'explicit_reference_and_prediction_status_required')
        if ref['status'] == 'complete':
            projection = ref['projection']
            require(projection['manual_literal_state'] in STATES
                    and projection['source_sha256'] == ref['source_sha256']
                    and projection['current_lung_opacity_reference'] is None
                    and projection['clinical_qualified'] is False,
                    'literal_projection_cannot_be_promoted_to_global_clinical_gold')
            truth = projection['manual_literal_state']
        else:
            require(ref['projection'] is None and ref['failure_type'] is not None,
                    'unavailable_reference_cannot_be_unknown_or_negative')
            truth = 'reference_unavailable'
        if pred['status'] == 'complete':
            states = pred['finding_states']
            require(isinstance(states, dict) and set(states) == set(HEADS)
                    and all(state in STATES for state in states.values())
                    and states['no_finding'] in ('unknown', 'positive')
                    and pred['failure_type'] is None, 'complete_named_official_head_states_required')
            prediction = states['lung_opacity']
        else:
            require(pred['finding_states'] is None and pred['failure_type'] is not None,
                    'unavailable_prediction_cannot_be_unknown_or_negative')
            prediction = 'prediction_unavailable'
        matrix[truth][prediction] += 1
        assessment = ('reference_unavailable' if truth == 'reference_unavailable'
                      else 'prediction_unavailable' if prediction == 'prediction_unavailable'
                      else 'broader_head_vs_literal_unknown_not_clinical_error' if truth == 'unknown'
                      else 'literal_state_match' if truth == prediction
                      else 'literal_state_mismatch_scope_unverified')
        details.append({'report_id': key, 'source_sha256': ref['source_sha256'],
            'reference_status': ref['status'], 'prediction_status': pred['status'],
            'reference_literal_state': None if truth == 'reference_unavailable' else truth,
            'prediction_lung_opacity_state': None if prediction == 'prediction_unavailable' else prediction,
            'assessment': assessment, 'clinical_error_adjudicated': False})
    per_state = {}
    for state in STATES:
        row = matrix[state]
        support = sum(row.values())
        completed = support - row['prediction_unavailable']
        per_state[state] = {
            'reference_support_all_attempted': support,
            'prediction_available': completed,
            'prediction_unavailable': row['prediction_unavailable'],
            'literal_state_matches': row[state],
            'match_fraction_all_reference_supported': row[state] / support if support else None,
            'match_fraction_prediction_available': row[state] / completed if completed else None,
            'interpretation': ('vocabulary_difference_not_annotated_absence' if state == 'unknown'
                               else 'literal_annotation_state_not_full_finding_truth')}
    known = sum(sum(matrix[s].values()) for s in ('positive', 'negative', 'uncertain'))
    known_available = known - sum(matrix[s]['prediction_unavailable']
                                  for s in ('positive', 'negative', 'uncertain'))
    known_matches = sum(matrix[s][s] for s in ('positive', 'negative', 'uncertain'))
    flips = matrix['positive']['negative'] + matrix['negative']['positive']
    explicit = sum(sum(matrix[s].values()) for s in ('positive', 'negative'))
    explicit_available = explicit - sum(matrix[s]['prediction_unavailable'] for s in ('positive', 'negative'))
    available_refs = sum(sum(matrix[s].values()) for s in STATES)
    jointly_available = available_refs - sum(matrix[s]['prediction_unavailable'] for s in STATES)
    return {'schema_version': VERSION + '-evaluation',
        'status': 'development_literal_vs_broader_report_head_diagnostic_only',
        'attempted_reports': len(references), 'reference_available': available_refs,
        'reference_unavailable': len(references) - available_refs,
        'prediction_status_counts': dict(Counter(p['status'] for p in predictions)),
        'jointly_available': jointly_available,
        'confusion_matrix_all_attempted': matrix,
        'per_reference_state': per_state,
        'known_literal_reference_support': known,
        'known_literal_prediction_available': known_available,
        'known_literal_state_matches': known_matches,
        'known_literal_match_fraction_all_supported': known_matches / known if known else None,
        'known_literal_match_fraction_prediction_available': known_matches / known_available if known_available else None,
        'explicit_literal_reference_support': explicit,
        'explicit_literal_prediction_available': explicit_available,
        'known_literal_polarity_flips': flips,
        'known_literal_polarity_flip_fraction_prediction_available': flips / explicit_available if explicit_available else None,
        'determinate_predictions_on_literal_unknown': matrix['unknown']['positive'] + matrix['unknown']['negative'],
        'literal_unknown_determinate_is_clinical_hallucination': False,
        'reference_vocab': 'human_annotated_literal_opacity_or_opacities',
        'prediction_vocab': 'official_chexbert_lung_opacity_head',
        'vocabulary_and_scope_equivalence_verified': False,
        'other_thirteen_heads_have_reference': False,
        'global_lung_opacity_gold_available': False, 'independent_image_truth': False,
        'uncertainty_qualification_possible': False,
        'class_support_is_not_four_state_qualification': True,
        'patient_disjointness_verified': False, 'checkpoint_training_overlap_verified': False,
        'untouched_final_test': False, 'primary_metric_eligible': False,
        'clinical_qualified': False, 'selection_changed': False,
        'regeneration_authorized': False, 'thresholds_fitted': False}, details

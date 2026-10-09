"""Forty-eight wholly authored development texts, not clinical gold.

Frozen before model execution; six known prior controls are explicitly marked.
No patient data, paraphrases of protected reports or hidden clinical labels.
"""
from __future__ import annotations

VERSION = 'authored_opacity_controls_48_v2'
# (text, authored report state, relevant mechanical segment IDs, legacy control)
FAMILIES = {
    'explicit_presence': [
        ('Lung opacity is present.', 'positive', [0], 'control_000'),
        ('There is a focal lung opacity.', 'positive', [0], None),
        ('Bilateral airspace opacities are present.', 'positive', [0], None),
        ('The lungs show patchy opacities.', 'positive', [0], None)],
    'explicit_absence': [
        ('No lung opacity is present.', 'negative', [0], 'control_001'),
        ('There are no lung opacities.', 'negative', [0], None),
        ('Lung opacities are absent.', 'negative', [0], None),
        ('No airspace opacity is seen.', 'negative', [0], None)],
    'possible_finding': [
        ('Possible lung opacity.', 'uncertain', [0], 'control_002'),
        ('A lung opacity may be present.', 'uncertain', [0], None),
        ('Cannot exclude a focal lung opacity.', 'uncertain', [0], None),
        ('Suspected airspace opacity.', 'uncertain', [0], None)],
    'unmentioned_finding': [
        ('The heart size is normal.', 'unknown', [], 'control_003'),
        ('A pacemaker is present.', 'unknown', [], None),
        ('No pleural effusion.', 'unknown', [], None),
        ('No pneumothorax is seen.', 'unknown', [], None)],
    'qualified_absence': [
        ('No large lung opacity.', 'uncertain', [0], 'control_005'),
        ('No new lung opacity.', 'uncertain', [0], None),
        ('No significant lung opacity.', 'uncertain', [0], None),
        ('There is no worsening lung opacity.', 'uncertain', [0], None)],
    'opposing_assertions': [
        ('Lung opacity is present. No lung opacity is present.', 'uncertain', [0, 1], 'control_004'),
        ('No lung opacity is seen. A lung opacity is present.', 'uncertain', [0, 1], None),
        ('FINDINGS: Lung opacity is present.\nIMPRESSION: No lung opacity.', 'uncertain', [0, 1], None),
        ('Lung opacity is absent. Bilateral lung opacities are present.', 'uncertain', [0, 1], None)],
    'negation_other_finding': [
        ('No pleural effusion. Lung opacity is present.', 'positive', [1], None),
        ('Lung opacity is present. No pneumothorax.', 'positive', [0], None),
        ('A pleural effusion is present. No lung opacity.', 'negative', [1], None),
        ('No lung opacity. A pacemaker is present.', 'negative', [0], None)],
    'other_disease_only': [
        ('Pneumonia is diagnosed.', 'unknown', [], None),
        ('The patient has heart failure.', 'unknown', [], None),
        ('Pulmonary edema is suspected.', 'unknown', [], None),
        ('Atelectasis is noted.', 'unknown', [], None)],
    'generic_summary': [
        ('No acute cardiopulmonary abnormality.', 'unknown', [], None),
        ('There is no acute disease.', 'unknown', [], None),
        ('Normal chest radiograph.', 'unknown', [], None),
        ('No acute findings.', 'unknown', [], None)],
    'change_language': [
        ('No change in lung opacity.', 'uncertain', [0], None),
        ('No worsening of lung opacity.', 'uncertain', [0], None),
        ('Unchanged lung opacities are present.', 'positive', [0], None),
        ('The lung opacity has resolved and no lung opacity remains.', 'negative', [0], None)],
    'multisegment_context': [
        ('The heart size is normal. Lung opacity is present. No pleural effusion.', 'positive', [1], None),
        ('FINDINGS:\nNo lung opacity.\nIMPRESSION:\nNo lung opacity.', 'negative', [1, 3], None),
        ('Possible lung opacity. No pneumothorax. A pacemaker is present.', 'uncertain', [0], None),
        ('No pleural effusion. The heart size is normal. A pacemaker is present.', 'unknown', [], None)],
    'repeated_assertion': [
        ('Lung opacity is present. Lung opacity is present.', 'positive', [0, 1], None),
        ('No lung opacity. No lung opacity.', 'negative', [0, 1], None),
        ('Possible lung opacity. Possible lung opacity.', 'uncertain', [0, 1], None),
        ('The heart size is normal. The heart size is normal.', 'unknown', [], None)],
}


def cases():
    result = []
    for family, rows in FAMILIES.items():
        if len(rows) != 4:
            raise ValueError('four_cases_per_family_required')
        for text, expected, relevant, legacy in rows:
            result.append({'item_id': f'authored_{len(result):04d}', 'text': text,
                'family': family, 'expected_state': expected,
                'relevant_segment_ids': list(relevant), 'legacy_control_id': legacy})
    if len(result) != 48 or len({r['text'] for r in result}) != 48:
        raise ValueError('fixed_48_distinct_authored_cases_required')
    return result

"""64 prospectively frozen, wholly invented context stress texts.

Not clinician annotations, independent clinical gold, or unseen to the author.
No protected source was used. Qualifier/change conventions are task definitions.
"""
VERSION = 'authored-opacity-context64-v1'
GROUPS = {
    'morphology_presence': [
        ('A pulmonary opacity is visible.', 'positive'),
        ('Diffuse airspace opacities are demonstrated.', 'positive'),
        ('Patchy lung opacities are evident.', 'positive'),
        ('A persistent lung OPACITY is present.', 'positive')],
    'explicit_negation': [
        ('The lungs are without airspace opacities.', 'negative'),
        ('Negative for lung opacities.', 'negative'),
        ('There is no evidence of pulmonary opacity.', 'negative'),
        ('Lung opacity is not visualized.', 'negative')],
    'uncertainty': [
        ('Questionable pulmonary opacity.', 'uncertain'),
        ('A probable airspace opacity is described.', 'uncertain'),
        ('Lung opacities cannot be excluded.', 'uncertain'),
        ('There may be pulmonary opacities.', 'uncertain')],
    'historical_only': [
        ('History of pulmonary opacity.', 'unknown'),
        ('Past medical history includes lung opacities.', 'unknown'),
        ('Previously noted pulmonary opacities.', 'unknown'),
        ('Prior examination documented a lung opacity.', 'unknown')],
    'family_only': [
        ('Family history of lung opacity.', 'unknown'),
        ('The mother had pulmonary opacities.', 'unknown'),
        ('The father has a lung opacity.', 'unknown'),
        ('The sister was diagnosed with pulmonary opacities.', 'unknown')],
    'hypothetical_only': [
        ('If lung opacities develop, repeat the examination.', 'unknown'),
        ('Monitor for pulmonary opacity.', 'unknown'),
        ('Evaluate for lung opacities.', 'unknown'),
        ('Rule out pulmonary opacity.', 'unknown')],
    'resolved_only': [
        ('The previously seen lung opacity has resolved.', 'unknown'),
        ('Resolved pulmonary opacities.', 'unknown'),
        ('History of now resolved airspace opacity.', 'unknown'),
        ('Previous lung opacities are no longer seen.', 'unknown')],
    'history_then_current': [
        ('History of pulmonary opacity. Current lung opacities are visible.', 'positive'),
        ('Previously noted lung opacities. Currently no pulmonary opacity is present.', 'negative'),
        ('Past medical history: lung opacity. Current airspace opacities are possible.', 'uncertain'),
        ('Prior lung opacity was documented. The current heart size is normal.', 'unknown')],
    'opposed_current': [
        ('Airspace opacity is visible. Pulmonary opacity is not seen.', 'uncertain'),
        ('No lung opacities are demonstrated. A pulmonary opacity is evident.', 'uncertain'),
        ('FINDINGS: Lung opacities are visible. IMPRESSION: No pulmonary opacity.', 'uncertain'),
        ('Pulmonary opacity is present, but lung opacities are absent.', 'uncertain')],
    'qualified_absence': [
        ('No sizable pulmonary opacities.', 'uncertain'),
        ('No additional lung opacity is visible.', 'uncertain'),
        ('No enlarging pulmonary opacity.', 'uncertain'),
        ('No severe airspace opacities are present.', 'uncertain')],
    'change_only': [
        ('No interval increase in pulmonary opacities.', 'uncertain'),
        ('No progression of the lung opacity.', 'uncertain'),
        ('No interval change in airspace opacities.', 'uncertain'),
        ('There is no further worsening of the pulmonary opacity.', 'uncertain')],
    'other_entity_negation': [
        ('No edema, but a pulmonary opacity is visible.', 'positive'),
        ('Without pleural fluid. Airspace opacities are visible.', 'positive'),
        ('A pleural effusion is seen, but no lung opacities are present.', 'negative'),
        ('A support device is present. Pulmonary opacities are not seen.', 'negative')],
    'generic_summary': [
        ('No acute intrathoracic process.', 'unknown'),
        ('Clear lungs on this examination.', 'unknown'),
        ('Stable chest examination.', 'unknown'),
        ('No significant cardiopulmonary disease.', 'unknown')],
    'other_disease_only': [
        ('Pulmonary infection is suspected.', 'unknown'),
        ('Congestive heart failure is noted.', 'unknown'),
        ('A small pneumothorax is present.', 'unknown'),
        ('Pulmonary vascular congestion is described.', 'unknown')],
    'sentence_boundaries': [
        ('No pleural fluid.\nA 1.5 cm pulmonary opacity is visible.', 'positive'),
        ('FINDINGS:\nAirspace opacities are evident.\nIMPRESSION:\nPulmonary opacity is present.', 'positive'),
        ('A device is seen.\nNo lung opacities are demonstrated.', 'negative'),
        ('Airspace opacity is absent. Pulmonary opacity is absent.', 'negative')],
    'nonpulmonary_mentions': [
        ('A corneal opacity is present.', 'unknown'),
        ('Lens opacities are visible.', 'unknown'),
        ('Dental opacity is described.', 'unknown'),
        ('A vitreous opacity is documented.', 'unknown')],
}


def cases():
    rows = []
    for family, group in GROUPS.items():
        if len(group) != 4:
            raise ValueError('four_fixed_cases_per_family_required')
        for text, state in group:
            rows.append({'item_id': f'context_{len(rows):04d}', 'text': text,
                         'family': family, 'expected_state': state})
    if len(rows) != 64 or len({r['text'] for r in rows}) != 64:
        raise ValueError('64_distinct_authored_context_cases_required')
    return rows

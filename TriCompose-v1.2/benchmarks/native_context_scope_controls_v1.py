"""88 wholly invented, prospectively sealed multi-finding scope probes.

Known investigator-authored development templates, not clinical gold/held-out.
No protected source text. Expected states express the conservative task only.
"""
VERSION = 'authored-native-context-scope88-v1'
FINDINGS = (
    ('lung_opacity', 'pulmonary opacity'),
    ('pneumonia', 'pneumonia'),
    ('edema', 'pulmonary edema'),
    ('cardiomegaly', 'cardiomegaly'),
    ('pleural_effusion', 'pleural effusion'),
    ('pneumothorax', 'pneumothorax'),
    ('atelectasis', 'atelectasis'),
    ('consolidation', 'pulmonary consolidation'),
)
TEMPLATES = (
    ('current_presence', 'The examination demonstrates {term}.', 'positive'),
    ('current_absence', 'The examination does not demonstrate {term}.', 'negative'),
    ('uncertain_current', 'There is a possibility of {term}.', 'uncertain'),
    ('historical_only', 'Medical history: {term}.', 'unknown'),
    ('family_only', 'Family history includes {term}.', 'unknown'),
    ('hypothetical_only', 'Should {term} develop, obtain another examination.', 'unknown'),
    ('resolved_only', 'The earlier {term} has completely resolved.', 'unknown'),
    ('qualified_absence', 'No marked {term} is demonstrated.', 'uncertain'),
    ('change_only', 'There is no interval increase in {term}.', 'uncertain'),
    ('opposed_current', 'The examination demonstrates {term}. {term} is absent.', 'uncertain'),
)
NONPULMONARY = (
    'A lenticular opacity is demonstrated.',
    'An opacity of the crystalline lens is demonstrated.',
    'An opacity is noted within the corneal tissue.',
    'An opacity of the vitreous is seen.',
    'There is a possible opacity of the ocular lens.',
    'No opacity of the corneal tissue is demonstrated.',
    'An opacity of the tooth is demonstrated.',
    'An opacity is seen in dental enamel.',
)


def cases():
    rows = []
    for family, template, state in TEMPLATES:
        for finding, term in FINDINGS:
            rows.append({'item_id': f'scope_{len(rows):04d}', 'text': template.format(term=term),
                         'finding': finding, 'family': family, 'expected_state': state})
    for text in NONPULMONARY:
        rows.append({'item_id': f'scope_{len(rows):04d}', 'text': text,
                     'finding': 'lung_opacity', 'family': 'nonpulmonary_mentions', 'expected_state': 'unknown'})
    if len(rows) != 88 or len({row['text'] for row in rows}) != 88:
        raise ValueError('88_distinct_fixed_authored_probes_required')
    return rows

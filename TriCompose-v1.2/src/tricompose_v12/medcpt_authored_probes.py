"""Small wholly authored representation probes, not clinical gold or a scorer.

No medical aliases are learned from the candidate bank. A high cosine is
relatedness only; neither polarity nor current-patient scope is adjudicated.
"""
import hashlib
import math
from collections import defaultdict

VERSION = 'tricompose-medcpt-authored-probes-v1'
POLICY = {'clinical_qualified': False, 'clinical_score': None,
    'independent_clinical_gold': False, 'current_patient_scope_verified': False,
    'image_factuality_verified': False, 'ehr_consistency_verified': False,
    'new_training': False, 'weight_or_threshold_fitting': False,
    'selection_changed': False, 'regeneration_authorized': False}

# Investigator-authored language contrasts; no clinical records or bank text.
BASES = (
    ('edema', 'Pulmonary edema is present.', 'Fluid accumulation is visible in the lungs.',
     'No pulmonary edema is present.', 'Pulmonary edema was present on an earlier study.',
     'If pulmonary edema develops, obtain another examination.'),
    ('cardiomegaly', 'Cardiomegaly is present.', 'The heart is enlarged.',
     'There is no cardiomegaly.', 'Cardiomegaly was present on an earlier study.',
     'If cardiomegaly develops, obtain another examination.'),
    ('effusion', 'A pleural effusion is present.', 'Fluid is present in the pleural space.',
     'There is no pleural effusion.', 'A pleural effusion was present on an earlier study.',
     'If a pleural effusion develops, obtain another examination.'),
    ('pneumothorax', 'A pneumothorax is present.', 'Air is present in the pleural space.',
     'There is no pneumothorax.', 'A pneumothorax was present on an earlier study.',
     'If a pneumothorax develops, obtain another examination.'),
    ('atelectasis', 'Pulmonary atelectasis is present.', 'A portion of the lung is collapsed.',
     'There is no pulmonary atelectasis.', 'Pulmonary atelectasis was present on an earlier study.',
     'If pulmonary atelectasis develops, obtain another examination.'),
    ('consolidation', 'Pulmonary consolidation is present.', 'Airspaces are filled with material rather than air.',
     'There is no pulmonary consolidation.', 'Pulmonary consolidation was present on an earlier study.',
     'If pulmonary consolidation develops, obtain another examination.'),
)


def probes():
    result = []
    for index, (name, reference, paraphrase, negation, history, hypothetical) in enumerate(BASES):
        variants = (('identity', reference), ('paraphrase', paraphrase),
                    ('negation', negation), ('history', history), ('hypothetical', hypothetical),
                    ('different_finding', BASES[(index + 1) % len(BASES)][1]))
        result.extend({'group': name, 'variant': kind, 'reference': reference, 'candidate': text}
                      for kind, text in variants)
    result.extend([
        {'group': 'detail_0', 'variant': 'laterality',
         'reference': 'There is a left pleural effusion.', 'candidate': 'There is a right pleural effusion.'},
        {'group': 'detail_1', 'variant': 'location',
         'reference': 'An opacity is in the left lower lung.', 'candidate': 'An opacity is in the right upper lung.'},
        {'group': 'detail_2', 'variant': 'severity',
         'reference': 'There is a small pleural effusion.', 'candidate': 'There is a large pleural effusion.'},
        {'group': 'detail_3', 'variant': 'device_position',
         'reference': 'The endotracheal tube tip is above the carina.',
         'candidate': 'The endotracheal tube tip is in the right main bronchus.'},
        {'group': 'empty_control', 'variant': 'empty', 'reference': BASES[0][1], 'candidate': ''},
    ])
    return result


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def inventory(source):
    rows, texts = [], {}
    for index, pair in enumerate(source):
        if set(pair) != {'group', 'variant', 'reference', 'candidate'} or \
                not all(isinstance(v, str) for v in pair.values()):
            raise ValueError('bounded_authored_probe_schema_required')
        for text in (pair['reference'], pair['candidate']):
            if len(text) > 1024:
                raise ValueError('authored_probe_size_exceeded')
            if text.strip():
                key = digest(text)
                if texts.setdefault(key, text) != text:
                    raise ValueError('text_hash_collision')
        rows.append({'pair_id': f'probe_{index:04d}', 'group': pair['group'],
            'variant': pair['variant'], 'reference_sha256': digest(pair['reference']),
            'candidate_sha256': digest(pair['candidate']),
            'input_nonempty': bool(pair['reference'].strip() and pair['candidate'].strip())})
    return {'schema_version': VERSION + '-inventory', 'pairs': rows,
            'pair_count': len(rows), 'distinct_nonempty_texts': len(texts),
            'policy': dict(POLICY)}, texts


def cosine(left, right):
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or \
            not left or len(left) != len(right) or len(left) > 4096 or \
            any(type(v) not in (int, float) or not math.isfinite(v) for v in (*left, *right)):
        raise ValueError('finite_equal_dimension_vectors_required')
    a, b = math.fsum(v * v for v in left), math.fsum(v * v for v in right)
    if a <= 0 or b <= 0:
        raise ValueError('nonzero_embedding_norm_required')
    value = math.fsum(x * y for x, y in zip(left, right)) / math.sqrt(a * b)
    if not math.isfinite(value) or not -1.00000001 <= value <= 1.00000001:
        raise ValueError('finite_cosine_required')
    return max(-1.0, min(1.0, value))


def score_pairs(plan, embeddings):
    rows = []
    for pair in plan['pairs']:
        value, status = None, 'empty_input_not_comparable'
        if pair['input_nonempty']:
            left = embeddings.get(pair['reference_sha256'])
            right = embeddings.get(pair['candidate_sha256'])
            if left is None or right is None:
                status = 'unavailable_embedding'
            else:
                try:
                    value, status = cosine(left, right), 'complete'
                except ValueError:
                    status = 'invalid_embedding'
        rows.append({**pair, 'status': status, 'query_query_cosine': value,
                     'clinical_score': None, 'clinical_qualified': False,
                     'regeneration_authorized': False})
    return rows


def diagnostic_summary(rows):
    by_variant, by_group, status_counts = defaultdict(list), defaultdict(dict), defaultdict(int)
    for row in rows:
        status_counts[row['status']] += 1
        if row['status'] == 'complete':
            by_variant[row['variant']].append(row['query_query_cosine'])
            by_group[row['group']][row['variant']] = row['query_query_cosine']
    ordering = {}
    for alternative in ('different_finding', 'negation', 'history', 'hypothetical'):
        counts = {'paraphrase_higher': 0, 'paraphrase_lower': 0, 'tie': 0, 'unavailable': 0}
        for name, *_ in BASES:
            values = by_group[name]
            if 'paraphrase' not in values or alternative not in values:
                counts['unavailable'] += 1
                continue
            difference = values['paraphrase'] - values[alternative]
            key = 'tie' if abs(difference) < 1e-12 else 'paraphrase_higher' if difference > 0 else 'paraphrase_lower'
            counts[key] += 1
        ordering[alternative] = counts
    return {'all_attempted_pairs': len(rows), 'status_counts': dict(status_counts),
        'cosine_by_variant': {key: {'n': len(values), 'mean': math.fsum(values) / len(values),
                                  'minimum': min(values), 'maximum': max(values)}
                              for key, values in sorted(by_variant.items())},
        'authored_ordering_diagnostics_not_clinical_accuracy': ordering,
        'policy': dict(POLICY)}

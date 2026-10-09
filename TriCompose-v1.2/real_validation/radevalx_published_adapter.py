"""Released CSV columns -> opaque derived numeric contracts only.

Never accesses ground_truth/M2Tr-Generation. This is a published-score
benchmark, not local model verification or a reference-free selection score.
"""
import math

from tricompose_v12.report_metric_alignment import VERSION

ANNOTATION_FIELDS = ('report_id', 'ground_truth', 'M2Tr-Generation',
                     '1','2','3','4','5','6','7','8')
METRIC_FIELDS = ('report_id', 'id', 'bleu4', 'bleu_2', 'bertscore', 'CheXbert', 'radgraph_f1', 'radcliq')
METRICS = {'bleu4': ('published_bleu4', 'higher_is_better'),
    'bleu_2': ('published_bleu2', 'higher_is_better'),
    'bertscore': ('published_bertscore', 'higher_is_better'),
    'CheXbert': ('published_chexbert', 'higher_is_better'),
    'radgraph_f1': ('published_radgraph_f1', 'higher_is_better'),
    'radcliq': ('published_radcliq', 'lower_is_better')}


def cell(value, count=False):
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError('bounded_numeric_cell_required')
    value = value.strip()
    if value.lower() in ('', 'nan'):
        return None
    try:
        number = float(value)
    except ValueError:
        raise ValueError('invalid_released_numeric_cell') from None
    if not math.isfinite(number):
        raise ValueError('finite_released_numeric_cell_required')
    if count:
        if number < 0 or not number.is_integer() or number > 10000:
            raise ValueError('bounded_nonnegative_consensus_count_required')
        return int(number)
    return number


def key(row):
    value = row['report_id']
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError('bounded_internal_join_key_required')
    return value.strip()


def predictions(rows, expected_count=100):
    if len(rows) != expected_count or not 1 <= expected_count <= 1024:
        raise ValueError('all_released_pairs_required')
    identities, records = {}, []
    for index, row in enumerate(rows):
        if set(row) != set(METRIC_FIELDS):
            raise ValueError('exact_published_metric_csv_schema_required')
        original = key(row)
        if original in identities:
            raise ValueError('duplicate_released_pair_refused')
        identity = {'item_id': 'pair_%04d' % index, 'source_group_id': 'group_%04d' % index}
        identities[original] = identity
        scores = {}
        for column, (name, _) in METRICS.items():
            number = cell(row[column])
            scores[name] = {'status': 'complete' if number is not None else 'not_available_in_source', 'value': number}
        records.append({**identity, 'scores': scores})
    return {'schema_version': VERSION + '-predictions', 'benchmark': 'radevalx-1.0.0',
        'metric_definitions': {name: {'orientation': direction, 'provenance': 'published_cached'}
            for name, direction in METRICS.values()}, 'records': records}, identities


def annotation_counts(rows, identities):
    if len(rows) != len(identities):
        raise ValueError('all_released_reference_pairs_required')
    result = {}
    for row in rows:
        if set(row) != set(ANNOTATION_FIELDS):
            raise ValueError('exact_released_annotation_csv_schema_required')
        original = key(row)
        if original not in identities or original in result:
            raise ValueError('unique_same_source_reference_join_required')
        # Unused report fields are never accessed, inspected or hashed separately.
        result[original] = [cell(row[str(index)], count=True) for index in range(1,9)]
    if set(result) != set(identities):
        raise ValueError('same_complete_reference_inventory_required')
    return result


def references(significant_rows, insignificant_rows, identities):
    significant = annotation_counts(significant_rows, identities)
    insignificant = annotation_counts(insignificant_rows, identities)
    return {'schema_version': VERSION + '-references', 'benchmark': 'radevalx-1.0.0',
        'reference_policy': 'two_reader_consensus', 'records': [
            {**identity, 'errors': {'clinically_significant': significant[original],
                                   'clinically_insignificant': insignificant[original]}}
            for original, identity in identities.items()]}

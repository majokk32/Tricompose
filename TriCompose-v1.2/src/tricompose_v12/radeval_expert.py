"""Strict author expert-count adapter and reference-based evaluation.

No models, filesystem IO, fitting, or clinical text exports. Source strings
are consumed internally; returned contracts contain opaque IDs/hashes/counts.
The benchmark is not an image/EHR or reference-free correctness test.
"""
from collections import Counter, defaultdict
import hashlib
import math
import re

from .radgraph_reference_contract import METRICS
from .report_metric_alignment import correlation, clustered_spearman_interval

VERSION = 'tricompose-radeval-expert-v1'
POLICY = {'reference_free': False, 'new_training': False,
    'weight_or_threshold_fitting': False, 'clinical_qualified': False,
    'selection_changed': False, 'regeneration_authorized': False,
    'checkpoint_training_overlap_verified': False,
    'image_factuality_verified': False, 'ehr_consistency_verified': False,
    'missing_annotation_or_score_is_zero': False,
    'expert_count_is_a_binary_finding_label': False}
FIELDS = {'annotator', 'type', 'ground_truth', 'images_path',
          'prediction1', 'prediction2', 'prediction3',
          'annotation1', 'annotation2', 'annotation3'}
CATEGORIES = ('false_prediction', 'omission', 'incorrect_location',
              'incorrect_severity', 'unsupported_comparison', 'omitted_change',
              'inarticulate_report')
SEVERITIES = ('clinically_significant', 'clinically_insignificant')
# Anchored category descriptions identify schema labels, not patient findings.
LABELS = (
    r'false prediction of finding',
    r'omission of finding',
    r'incorrect location(?: position)?(?: of finding)?',
    r'incorrect severity(?: of finding)?',
    r'mention(?: of)? comparison (?:that is )?not (?:present )?in (?:the )?reference(?: report)?',
    r'omission(?: of)? (?:a )?change (?:from|in) (?:(?:the|a) )?previous study',
    r'inarticulate report(?: grammar(?: and)? readability(?: issues)?)?',
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def annotation_counts(value):
    """Only explicit integers; missing/malformed cells never become zeros.

Keep known category counts when one other category is malformed. A total is
eligible only when all seven categories of that severity are explicit.
"""
    require(isinstance(value, str) and len(value) <= 100000, 'bounded_annotation_required')
    result = {s: [None] * 7 for s in SEVERITIES}
    if not value.strip():
        return {'status': 'blank_unavailable', 'errors': result, 'issues': ['blank_cell']}
    current, seen, issues = None, set(), []
    for line in value.splitlines():
        if not line.strip():
            continue
        header = re.fullmatch(r'[^A-Za-z0-9]*(Significant|Insignificant)[^A-Za-z0-9]*', line, re.I)
        if header:
            current = SEVERITIES[header[1].lower() == 'insignificant']
            if current in seen:
                issues.append('duplicate_severity_header')
            seen.add(current)
            continue
        numbered = re.fullmatch(r'\s*([1-7])[.)]\s+(.+?)\s*:\s*([0-9]+)\s*[.;]?\s*', line)
        if numbered and current is not None:
            category = int(numbered[1]) - 1
            label = ' '.join(re.sub(r'[^a-z ]', ' ', numbered[2].lower()).split())
            if not re.fullmatch(LABELS[category], label):
                issues.append('unsupported_category_description')
                continue
            if result[current][category] is not None:
                issues.append('duplicate_category')
                # Conflicting duplicates must not retain one arbitrary count.
                result[current][category] = None
                continue
            count = int(numbered[3])
            if count > 1000:
                issues.append('count_out_of_bounds')
                continue
            result[current][category] = count
        else:
            issues.append('unparsed_annotation_line')
    if seen != set(SEVERITIES):
        issues.append('missing_severity_header')
    if any(v is None for counts in result.values() for v in counts):
        issues.append('missing_explicit_category_count')
    # Structural ambiguity invalidates the whole cell; a single missing count
    # only invalidates totals using that category, without inventing zeros.
    structural = {'duplicate_severity_header', 'duplicate_category',
                  'missing_severity_header', 'unsupported_category_description'}
    if structural.intersection(issues):
        result = {s: [None] * 7 for s in SEVERITIES}
    return {'status': 'complete' if not issues else 'partial_or_invalid',
            'errors': result, 'issues': sorted(set(issues))}


def internal_group(path):
    """Cluster by an explicit MIMIC patient folder if present, then study.

    Raw identifiers are transient keys only and never leave inventory(). If
    neither folder is recognizable, use the complete author source key. This
    fallback is declared, not falsely described as patient-level splitting.
    """
    subjects = set(re.findall(r'(?:^|[/\\])p([0-9]{6,})(?=[/\\]|$)', path))
    if len(subjects) == 1:
        return ('patient_folder', next(iter(subjects)))
    chexpert = set(re.findall(r'(?:^|[/\\])patient([0-9]+)(?=[/\\]|$)', path))
    if len(chexpert) == 1:
        return ('chexpert_patient_folder', next(iter(chexpert)))
    studies = set(re.findall(r'(?:^|[/\\])s([0-9]{6,})(?=[/\\]|$)', path))
    if len(studies) == 1:
        return ('study_folder', next(iter(studies)))
    return ('author_source_key', path.strip())


def inventory(rows):
    require(isinstance(rows, list) and 1 <= len(rows) <= 1024, 'bounded_author_rows_required')
    groups, studies, readers, section_ids = {}, {}, {}, {}
    pairs, graph_ids, cells = {}, {}, []
    source_slots = defaultdict(set)
    issues = Counter()
    for row_index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == FIELDS and
                all(isinstance(v, str) and len(v) <= 100000 for v in row.values()),
                'exact_bounded_author_schema_required')
        require(row['images_path'].strip() and row['annotator'].strip() and row['type'].strip(),
                'source_group_reader_section_required')
        study = studies.setdefault(row['images_path'].strip(), f'source_{len(studies):04d}')
        key = internal_group(row['images_path'])
        group = groups.setdefault(key, f'group_{len(groups):04d}')
        reader = readers.setdefault(row['annotator'].strip(), f'reader_{len(readers):03d}')
        source_section = row['type'].strip().lower()
        section = section_ids.setdefault(source_section, f'section_{len(section_ids):02d}')
        section_name = source_section if source_section in ('findings', 'impression') else 'other_author_section'
        reference = row['ground_truth']
        reference_sha = digest(reference)
        ref_id = graph_ids.setdefault(reference_sha, f'graph_{len(graph_ids):04d}')
        for slot in range(1, 4):
            hypothesis = row[f'prediction{slot}']
            hyp_sha = digest(hypothesis)
            hyp_id = graph_ids.setdefault(hyp_sha, f'graph_{len(graph_ids):04d}')
            # Exact bytes + source section + released candidate position.
            # Inconsistent source texts aren't silently merged as reader votes.
            pair_key = (study, section, slot, reference_sha, hyp_sha)
            source_slots[(study, section, slot)].add((reference_sha, hyp_sha))
            if pair_key not in pairs:
                pairs[pair_key] = {'item_id': f'pair_{len(pairs):04d}',
                    'source_group_id': group, 'source_id': study, 'section_id': section,
                    'section_name': section_name, 'candidate_slot': slot,
                    'reference_sha256': reference_sha, 'hypothesis_sha256': hyp_sha,
                    'reference_graph_id': ref_id, 'hypothesis_graph_id': hyp_id,
                    'input_nonempty': bool(reference.strip() and hypothesis.strip()),
                    'reader_cells': []}
            pair = pairs[pair_key]
            require(not any(c['reader_id'] == reader for c in pair['reader_cells']),
                    'duplicate_exact_reader_pair_rejected')
            parsed = annotation_counts(row[f'annotation{slot}'])
            cell = {'cell_id': f'cell_{len(cells):04d}', 'item_id': pair['item_id'],
                    'source_row_index': row_index, 'reader_id': reader, **parsed}
            cells.append(cell)
            pair['reader_cells'].append(cell)
            issues.update(parsed['issues'])
    records = []
    for pair in pairs.values():
        observed = pair.pop('reader_cells')
        errors = {}
        for severity in SEVERITIES:
            errors[severity] = [
                sum(c['errors'][severity][i] for c in observed) / len(observed)
                if all(c['errors'][severity][i] is not None for c in observed) else None
                for i in range(7)]
        records.append({**pair, 'released_reader_cells': len(observed),
            'complete_reader_cells': sum(c['status'] == 'complete' for c in observed),
            'errors': errors})
    graph_rows = [{'graph_id': graph_id, 'text_sha256': text_sha}
                  for text_sha, graph_id in graph_ids.items()]
    return {'schema_version': VERSION + '-inventory', 'source_rows': len(rows),
        'released_annotation_cells': len(cells), 'source_keys': len(studies),
        'source_groups': len(groups), 'reader_count': len(readers),
        'group_key_strategy_counts': dict(Counter(k[0] for k in groups)),
        'section_counts': dict(Counter(p['section_name'] for p in records)),
        'distinct_author_section_values': len(section_ids),
        'source_slot_text_conflicts': sum(len(v) > 1 for v in source_slots.values()),
        'reader_cell_status_counts': dict(Counter(c['status'] for c in cells)),
        'annotation_issue_counts': dict(issues),
        'reader_aggregation': 'mean_of_all_released_cells_for_exact_source_section_candidate_text_pair',
        'missing_reader_category_invalidates_aggregate_category': True,
        'two_readers_per_pair_assumed': False, 'records': records,
        'annotation_cells': cells, 'graphs': graph_rows, 'policy': dict(POLICY)}


def outcomes(errors):
    result = {}
    totals = {}
    for severity in SEVERITIES:
        values = errors[severity]
        require(isinstance(values, list) and len(values) == 7 and all(v is None or
            (type(v) in (int, float) and math.isfinite(v) and v >= 0) for v in values),
            'seven_nonnegative_or_missing_categories_required')
        totals[severity] = sum(values) if all(v is not None for v in values) else None
        result[severity + '_total'] = totals[severity]
        result.update({severity + '_' + CATEGORIES[i]: value for i, value in enumerate(values)})
    result['all_errors_total'] = sum(totals.values()) if all(v is not None for v in totals.values()) else None
    return result


def evaluate(plan, scores, *, resamples=1000, seed=0):
    require(plan['schema_version'] == VERSION + '-inventory' and plan['policy'] == POLICY,
            'frozen_expert_contract_required')
    require(len(scores) == len(plan['records']) and all(s['item_id'] == p['item_id']
            for s, p in zip(scores, plan['records'])), 'exact_all_pair_join_required')
    all_outcomes = [outcomes(p['errors']) for p in plan['records']]
    results = {}
    for metric in METRICS:
        per_outcome = {}
        for name in all_outcomes[0]:
            values = []
            for p, s, reference in zip(plan['records'], scores, all_outcomes):
                require(s['status'] in ('complete', 'unavailable_graph', 'empty_input'),
                        'explicit_model_availability_required')
                value = s['scores'][metric]
                require((type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)
                        if s['status'] == 'complete' else value is None, 'finite_score_or_null_required')
                if value is not None and reference[name] is not None:
                    values.append((p['source_group_id'], value, -reference[name]))
            result = correlation([(v[1], v[2]) for v in values])
            result['paired_coverage'] = len(values) / len(scores)
            result['reference_available'] = sum(o[name] is not None for o in all_outcomes)
            if name in ('clinically_significant_total', 'all_errors_total'):
                result['cluster_bootstrap'] = clustered_spearman_interval(values, resamples, seed)
            per_outcome[name] = result
        by_section = {}
        for section in sorted({p['section_id'] for p in plan['records']}):
            eligible = [(s['scores'][metric], -o['clinically_significant_total'])
                for p, s, o in zip(plan['records'], scores, all_outcomes)
                if p['section_id'] == section and s['status'] == 'complete'
                and o['clinically_significant_total'] is not None]
            by_section[section] = correlation(eligible)
        results[metric] = {'orientation': 'higher_is_better', 'provenance': 'local_frozen_official',
            'complete_scores': sum(s['status'] == 'complete' for s in scores),
            'outcomes': per_outcome, 'descriptive_section_correlations': by_section}
    return {'schema_version': VERSION + '-evaluation', 'policy': dict(POLICY),
        'all_attempted_pairs': len(scores), 'source_groups': plan['source_groups'],
        'expert_annotation_cells': plan['released_annotation_cells'],
        'primary_outcome': 'clinically_significant_total',
        'secondary_outcome': 'all_errors_total', 'bootstrap_resamples': resamples, 'seed': seed,
        'correlation_direction': 'quality_score_vs_negative_expert_error_burden',
        'reader_aggregation': plan['reader_aggregation'], 'metrics': results,
        'local_implementation_qualified': False, 'image_or_ehr_truth_verified': False}

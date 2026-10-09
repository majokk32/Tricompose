"""Lossless frozen-mask annotation; no clinical score, selection or action.

Same cached image proposals as the real-reference diagnostic, but no reference
labels or reference-set error rates are transferred to synthetic candidates.
"""
from collections import Counter, defaultdict
import math

from .image_reader_consensus import POLICIES, CONTRASTS, decision
from .radgraph_reference_contract import require

PREFIX = 'image_mask_'
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
STATES = ('positive', 'negative', 'uncertain', 'unknown')
LINEAGE = ('case_id', 'cxr_candidate_id', 'cxr_sha256')


def number(value):
    if value in ('', None):
        return None
    require(type(value) in (str, int, float), 'finite_cached_numeric_metadata_required')
    value = float(value)
    require(math.isfinite(value), 'finite_cached_numeric_metadata_required')
    return value


def same_number(actual, expected):
    value = number(actual)
    require((value is None) == (expected is None)
            and (expected is None or math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)),
            'cached_image_value_changed')


def relation(image_state, report_state):
    require(report_state in STATES and image_state in (None, 'positive', 'negative'), 'explicit_or_unavailable_state_required')
    if image_state is None or report_state not in ('positive', 'negative'):
        return 'not_comparable'
    return 'proxy_support' if image_state == report_state else 'proxy_opposition'


def overlay(rows, xrv_records, biovil_records):
    require(rows and len(rows) <= 960 and len({r['triple_candidate_id'] for r in rows}) == len(rows),
            'bounded_unique_complete_candidate_rows_required')
    columns = list(rows[0])
    require(all(list(r) == columns and not any(k.startswith(PREFIX) for k in r) for r in rows),
            'same_unannotated_source_columns_required')
    x = {r['cxr_candidate_id']: r for r in xrv_records}
    b = {r['cxr_candidate_id']: r for r in biovil_records}
    require(len(x) == len(xrv_records) and len(b) == len(biovil_records)
            and set(x) == set(b) == {r['cxr_candidate_id'] for r in rows}, 'all_unique_bound_image_score_outcomes_required')
    images = {}
    for key in x:
        a, c = x[key], b[key]
        require(all(a[f] == c[f] for f in (*LINEAGE, 'cxr_model_id')), 'same_cached_image_lineage_required')
        require(a['status'] in ('scored', 'failed_without_replacement')
                and c['status'] in ('scored', 'failed_without_replacement'), 'explicit_image_attempt_receipts_required')
        score = a['exact_lung_opacity_score']
        require((score is None) == (a['status'] == 'failed_without_replacement'), 'failure_cannot_be_a_negative_score')
        pairs = c['score_pairs']
        require((pairs is None) == (c['status'] == 'failed_without_replacement'), 'failed_biovil_requires_null_pairs')
        margins = None
        if pairs is not None:
            require(set(pairs) == set(FAMILIES)
                    and all(set(pair) == {'positive_cosine', 'negative_cosine'}
                            and all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1
                                    for v in pair.values()) for pair in pairs.values()), 'all_fixed_finite_cosine_pairs_required')
            margins = [pairs[f]['positive_cosine'] - pairs[f]['negative_cosine'] for f in FAMILIES]
        decisions = {p: decision({'xrv_score': score, 'margins': margins}, p) for p in POLICIES}
        images[key] = {**{f: a[f] for f in (*LINEAGE, 'cxr_model_id')},
            'xrv_score': score, 'pairs': pairs, 'margins': margins, 'decisions': decisions}
    fixed, report_labels, annotated = {}, {}, []
    for r in rows:
        image = images[r['cxr_candidate_id']]
        require(all(r[f] == image[f] for f in LINEAGE), 'same_fixed_candidate_image_required')
        value = r['ehr_sha256'], r['ehr_facts_sha256']
        require(fixed.setdefault(r['case_id'], value) == value, 'fixed_ehr_and_facts_required')
        state = r['opacity_cached_report_state']
        require(state in STATES and report_labels.setdefault(r['report_sha256'], state) == state,
                'same_report_artifact_keeps_same_proposal_state')
        require(r['opacity_cached_ehr_state'] == 'unknown' and r['opacity_selector_used'] == 'False'
                and r['opacity_biovil_selector_used'] == 'False'
                and r['opacity_clinical_accuracy'] == r['opacity_biovil_clinical_accuracy'] == '',
                'original_unqualified_opacity_profile_required')
        same_number(r['opacity_exact_score'], image['xrv_score'])
        expected = image['decisions']['xrv_exact_0_5']['state'] or 'unknown'
        require(r['opacity_exact_state_0_5'] == expected
                and r['opacity_biovil_preference_state'] == (image['decisions']['biovil_fixed_mean']['state'] or 'unknown'),
                'unchanged_exact_head_and_fixed_mean_states_required')
        for i, family in enumerate(FAMILIES):
            pair = image['pairs'][family] if image['pairs'] is not None else None
            for polarity in ('positive_cosine', 'negative_cosine'):
                same_number(r['opacity_biovil_' + family + '_' + polarity], pair[polarity] if pair else None)
            same_number(r['opacity_biovil_' + family + '_margin'], image['margins'][i] if pair else None)
        same_number(r['opacity_biovil_mean_margin'], sum(image['margins']) / 3 if image['margins'] is not None else None)
        added = {}
        for policy in POLICIES:
            d = image['decisions'][policy]
            added.update({PREFIX + policy + '_state': d['state'], PREFIX + policy + '_status': d['status'],
                          PREFIX + policy + '_report_relation': relation(d['state'], state)})
        added.update({PREFIX + 'scope': 'exact_opacity_image_proposals_and_cached_report_proposal',
            PREFIX + 'clinical_primary_eligible': False, PREFIX + 'reference_metric_transferred': False,
            PREFIX + 'synthetic_domain_transport_validated': False, PREFIX + 'selector_used': False,
            PREFIX + 'regeneration_authorized': False})
        annotated.append({**r, **added})
    summaries = []
    for policy in POLICIES:
        ds = [image['decisions'][policy] for image in images.values()]
        relations = Counter(r[PREFIX + policy + '_report_relation'] for r in annotated)
        comparable = relations['proxy_support'] + relations['proxy_opposition']
        by_report = {}
        for model in sorted({r['report_model_id'] for r in rows}):
            subset = [r for r in annotated if r['report_model_id'] == model]
            counts = Counter(r[PREFIX + policy + '_report_relation'] for r in subset)
            by_report[model] = {'candidate_rows': len(subset), **{s: counts[s] for s in ('proxy_support', 'proxy_opposition', 'not_comparable')}}
        summaries.append({'policy': policy, 'image_slots': len(images),
            'accepted_image_slots': sum(d['state'] is not None for d in ds),
            'abstained_image_slots': sum(d['status'].startswith('abstain_') for d in ds),
            'unavailable_image_slots': sum(d['status'] == 'unavailable_reader' for d in ds),
            'accepted_image_states': dict(Counter(d['state'] for d in ds if d['state'] is not None)),
            'status_counts': dict(Counter(d['status'] for d in ds)), 'candidate_rows': len(rows),
            **{s: relations[s] for s in ('proxy_support', 'proxy_opposition', 'not_comparable')},
            'comparable_coverage': comparable / len(rows),
            'agreement_over_comparable': relations['proxy_support'] / comparable if comparable else None,
            'by_report_model': by_report, 'clinical_qualified': False})
    losses = []
    for left, right in CONTRASTS:
        removed = {key for key, image in images.items()
                   if image['decisions'][left]['state'] is not None and image['decisions'][right]['state'] is None}
        require(all(image['decisions'][right]['state'] is None
                    or image['decisions'][right]['state'] == image['decisions'][left]['state'] for image in images.values()),
                'agreement_masks_must_be_nested_not_changed_states')
        source_relations = Counter(r[PREFIX + left + '_report_relation'] for r in annotated if r['cxr_candidate_id'] in removed)
        losses.append({'left': left, 'right': right, 'image_slots_withheld': len(removed),
            'proxy_support_rows_withheld': source_relations['proxy_support'],
            'proxy_opposition_rows_withheld': source_relations['proxy_opposition'],
            'not_comparable_rows_withheld': source_relations['not_comparable'],
            'removed_correct_or_incorrect_cases_adjudicated': False})
    fingerprints = defaultdict(set)
    for image in images.values():
        fingerprints[image['cxr_sha256']].add(tuple((image['decisions'][p]['state'], image['decisions'][p]['status']) for p in POLICIES))
    summary = {'schema_version': 'candidate-fixed-image-mask-overlay-v1', 'fixed_ehr_cases': len(fixed),
        'image_slots': len(images), 'distinct_image_artifacts': len(fingerprints), 'candidate_rows': len(rows),
        'distinct_report_artifacts': len(report_labels), 'source_columns': len(columns),
        'added_columns': len(annotated[0]) - len(columns), 'policy_summaries': summaries,
        'nested_mask_withholding': losses, 'duplicate_image_hash_decision_variations': sum(len(v) > 1 for v in fingerprints.values()),
        'original_cells_preserved': all(all(out[k] == r[k] for k in columns) for out, r in zip(annotated, rows)),
        'original_row_order_preserved': [r['triple_candidate_id'] for r in rows] == [r['triple_candidate_id'] for r in annotated],
        'ehr_opacity_reference_available': False, 'new_model_calls': 0, 'cases_dropped': 0,
        'clinical_qualified': False, 'reference_metric_transferred': False,
        'synthetic_domain_transport_validated': False, 'selection_changed': False, 'regeneration_authorized': False}
    return annotated, list(images.values()), summary

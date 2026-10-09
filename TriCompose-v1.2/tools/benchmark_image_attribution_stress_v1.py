#!/usr/bin/env python3
"""Frozen image-evidence transport and unsafe attribution shortcut diagnostic.

Reuse exact-opacity numeric caches only. A detected image/report opposition
is not a clinical fault label or permission to regenerate either modality.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
import diagnose_localization_observability_v1 as evidence
from tricompose_v12 import image_reader_consensus as reader
from tricompose_v12.live_workers import check_pins
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

b = evidence.b
VERSION = 'tricompose-image-attribution-stress-v1'
PROTOCOL = ROOT / 'configs/image_attribution_stress_v1.json'
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
STATES = {'positive', 'negative', 'uncertain', 'unknown'}
EXPLICIT = {'positive', 'negative'}
XRV = (b.BASE / 'candidate_opacity_runs/opacity_pool240_12668204',
    '778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94')
BIOVIL = (b.BASE / 'candidate_opacity_biovil_runs/biovil_opacity240_12670345',
    '6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828')
REFERENCE = (b.BASE / 'image_reader_consensus_runs/ricord50_12784259_002',
    'add0c915921d0e2fa0741d604608186ce67faf83d94ad81f370eddc3c5997c29')


def validate_protocol(protocol):
    if (protocol != json.loads(PROTOCOL.read_text())
            or protocol['image_policies'] != list(reader.POLICIES)
            or protocol['finding'] != 'exact_lung_opacity'
            or protocol['biovil_templates'] != list(FAMILIES)
            or protocol['xrv_threshold'] != .5 or protocol['biovil_threshold'] != 0
            or protocol['all_fixed_ehr_opacity_states'] != 'unknown'
            or protocol['key_decoded_after_prediction_seal'] is not True
            or protocol['exact_zero_is_unknown'] is not True
            or type(protocol['new_model_calls']) is not int or protocol['new_model_calls'] != 0
            or any(protocol[k] is not False for k in (
                'shortcut_is_authorized_action', 'historical_final_role_is_untouched_test',
                'synthetic_transport_validated', 'same_image_votes_are_independent',
                'mechanical_target_is_clinical_fault', 'report_assertions_clinically_qualified',
                'old_choices_or_scores_changed', 'secondary_endpoint_used_for_prediction',
                'clinical_qualified', 'training_allowed', 'regeneration_authorized',
                'actual_regeneration_executed'))):
        raise ValueError('complete_frozen_opacity_only_nonclinical_protocol_required')


def cache_records(spec, schema, sources):
    root, pin = spec
    meta = json.loads(b.checked(root / 'manifest.json', pin, 1024**2, sources))
    value = meta['artifacts']['image_scores.json']
    digest = value['sha256'] if isinstance(value, dict) else value
    data = json.loads(b.checked(root / 'image_scores.json', digest, 4*1024**2, sources))
    if data['schema_version'] != schema or not isinstance(data['records'], list):
        raise ValueError('declared_numeric_score_cache_required')
    return data['records'], meta


def score_margins(row):
    if row['status'] == 'failed_without_replacement' and row['score_pairs'] is None:
        return None
    if row['status'] != 'scored' or not isinstance(row['score_pairs'], dict) or set(row['score_pairs']) != set(FAMILIES):
        raise ValueError('all_declared_score_pairs_or_explicit_failure_required')
    values = []
    for family in FAMILIES:
        pair = row['score_pairs'][family]
        if (not isinstance(pair, dict) or set(pair) != {'positive_cosine', 'negative_cosine'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1.01 for v in pair.values())):
            raise ValueError('finite_fixed_polarity_cosine_pairs_required')
        values.append(pair['positive_cosine'] - pair['negative_cosine'])
    return values


def bind_image_evidence(bank, xrv, biovil):
    """No image bytes, source EHRs or report contents; exact recorded lineage."""
    expected, reports, cases = {}, {}, {}
    for grid in bank.values():
        for candidate in grid.values():
            view = b.policy_module.snapshot(candidate)
            line = view['lineage']
            if view['states']['ehr']['lung_opacity'] != 'unknown':
                raise ValueError('missing_ehr_opacity_cannot_be_filled_from_other_modalities')
            anchor = (line['ehr_sha256'], line['ehr_facts_sha256'])
            if cases.setdefault(view['case_id'], anchor) != anchor:
                raise ValueError('fixed_ehr_lineage_required')
            image = {'case_id': view['case_id'], 'cxr_candidate_id': line['cxr_candidate_id'],
                'cxr_sha256': line['cxr_sha256'], 'cxr_model_id': line['cxr_model_id']}
            if expected.setdefault(line['cxr_candidate_id'], image) != image:
                raise ValueError('unchanged_shared_image_lineage_required')
            report = {'report_sha256': line['report_sha256'], 'state': view['states']['chexbert']['lung_opacity']}
            if reports.setdefault(line['report_candidate_id'], report) != report:
                raise ValueError('unchanged_report_opacity_state_and_hash_required')
    def index(rows):
        result = {}
        for row in rows:
            item = row['cxr_candidate_id']
            if item in result or item not in expected or any(row[k] != v for k, v in expected[item].items()):
                raise ValueError('complete_unique_same_image_score_lineage_required')
            result[item] = row
        if set(result) != set(expected):
            raise ValueError('all_existing_image_scores_required')
        return result
    xs, bs = index(xrv), index(biovil)
    result = {}
    for item, line in expected.items():
        row = xs[item]
        if row['status'] == 'failed_without_replacement' and row['exact_lung_opacity_score'] is None:
            score = None
        elif row['status'] == 'scored':
            score = row['exact_lung_opacity_score']
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('exact_frozen_head_score_required')
        else:
            raise ValueError('explicit_xrv_score_or_failure_required')
        values = {'xrv_score': score, 'margins': score_margins(bs[item])}
        for policy in reader.POLICIES:
            reader.decision(values, policy)
        result[item] = {**line, **values}
    return result, reports


def validate_packet(packet):
    if (not isinstance(packet, dict) or set(packet) != {'item_id', 'ehr_opacity_state', 'report_opacity_state', 'image_evidence'}
            or not isinstance(packet['item_id'], str) or not packet['item_id']
            or packet['ehr_opacity_state'] != 'unknown' or packet['report_opacity_state'] not in STATES
            or not isinstance(packet['image_evidence'], dict) or set(packet['image_evidence']) != {'xrv_score', 'margins'}):
        raise ValueError('opaque_metadata_only_missing_ehr_opacity_packet_required')


def judge(packet, policy):
    validate_packet(packet)
    value = reader.decision(packet['image_evidence'], policy)
    state, report = value['state'], packet['report_opacity_state']
    if value['status'] == 'unavailable_reader':
        relation, request = 'not_comparable_image_unavailable', 'image_evidence_missing'
    elif state is None:
        relation, request = 'not_comparable_image_abstained', 'resolve_image_reader_uncertainty'
    elif report not in EXPLICIT:
        relation, request = 'not_comparable_report_state', 'report_scope_evidence_missing'
    elif state == report:
        relation, request = 'proxy_support', 'no_action_authorized_by_proxy_agreement'
    else:
        relation, request = 'proxy_opposition', 'verify_image_finding_and_report_assertion'
    return {'item_id': packet['item_id'], 'image_policy': policy, 'finding': 'exact_lung_opacity',
        'image_state': state, 'image_status': value['status'], 'cached_report_state': report,
        'ehr_state': 'unknown', 'proxy_relation': relation,
        'unsafe_report_blame_shortcut': 'report' if relation == 'proxy_opposition' else None,
        'evidence_request_reason': request, 'confirmed_faulty_modality': None,
        'clinical_localization_accuracy': None, 'clinical_qualified': False,
        'regeneration_authorized': False, 'image_votes_are_independent': False}


def item_packets(resolver, images, reports):
    result, seen = [], set()
    for row in resolver:
        if row['item_id'] in seen or row['benchmark_split'] not in evidence.ROLES:
            raise ValueError('unique_original_intervention_items_required')
        seen.add(row['item_id'])
        image, report = images[row['displayed_cxr_candidate_id']], reports[row['displayed_report_candidate_id']]
        if image['cxr_sha256'] != row['displayed_cxr_sha256'] or report['report_sha256'] != row['displayed_report_sha256']:
            raise ValueError('actual_displayed_artifacts_required_not_original_parents')
        result.append({'item_id': row['item_id'], 'ehr_opacity_state': 'unknown',
            'report_opacity_state': report['state'],
            'image_evidence': {k: image[k] for k in ('xrv_score', 'margins')}})
    return result


def summarize(rows, resolver, key):
    lookup = {r['item_id']: r for r in resolver}
    truth = {r['item_id']: r for r in key}
    groups = defaultdict(list)
    for row in rows:
        for role in ('overall', lookup[row['item_id']]['benchmark_split']):
            groups[row['image_policy'], role, truth[row['item_id']]['intervention_type']].append(row)
    result = []
    for (policy, role, arm), subset in sorted(groups.items()):
        flags = [r for r in subset if r['unsafe_report_blame_shortcut'] is not None]
        counts = Counter(r['proxy_relation'] for r in subset)
        comparable = counts['proxy_support'] + counts['proxy_opposition']
        result.append({'image_policy': policy, 'source_role': role, 'mechanical_arm': arm,
            'available_items': len(subset), 'fixed_ehr_cases': len({lookup[r['item_id']]['case_id'] for r in subset}),
            'accepted_image_judgments': sum(r['image_state'] is not None for r in subset),
            'comparable_image_report_pairs': comparable, 'proxy_support': counts['proxy_support'],
            'proxy_opposition': counts['proxy_opposition'], 'comparison_coverage': comparable / len(subset),
            'conditional_proxy_support': counts['proxy_support'] / comparable if comparable else None,
            'image_unavailable': counts['not_comparable_image_unavailable'],
            'image_abstained': counts['not_comparable_image_abstained'],
            'report_not_comparable': counts['not_comparable_report_state'],
            'unsafe_shortcut_matches_mechanical_target': sum(truth[r['item_id']]['intervention_target'] == 'report' for r in flags),
            'unsafe_shortcut_disagrees_with_mechanical_target': sum(truth[r['item_id']]['intervention_target'] != 'report' for r in flags),
            'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None})
    return result


def run(args):
    b.cpu_guard()
    sources = {}
    bank, _ = b.load_primary(sources)
    protocol = json.loads(PROTOCOL.read_text()); validate_protocol(protocol)
    xrv, xm = cache_records(XRV, 'tricompose-cached-exact-opacity-sidecar-v1-scores', sources)
    biovil, bm = cache_records(BIOVIL, 'tricompose-cached-opacity-biovil-evidence-v1', sources)
    if xm['original_selection_changed'] is not False or bm['source_manifest_sha256'] != XRV[1]:
        raise ValueError('same_unchanged_primary_bank_opacity_sidecars_required')
    images, reports = bind_image_evidence(bank, xrv, biovil)
    if len(images) != protocol['image_slots'] or len(reports) != protocol['candidate_slots']:
        raise ValueError('complete_existing_numeric_evidence_inventory_required')
    meta = json.loads(b.checked(evidence.BANK / 'manifest.json', evidence.BANK_SHA, 1024**2, sources))
    if meta['counts'] != protocol['available_arms'] or meta['source']['coverage']['unavailable'] != protocol['unavailable_arms']:
        raise ValueError('all_available_and_unavailable_arms_required')
    items = {}
    for name in ('blind_items', 'resolver'):
        text = b.checked(evidence.BANK / (name + '.jsonl'), meta['artifact_sha256'][name], 4*1024**2, sources)
        items[name] = [json.loads(line) for line in text.splitlines()]
    # Also authenticates unchanged fixed EHRs and the original displayed state inventory.
    evidence.bind_packets(bank, items['resolver'])
    packets = item_packets(items['resolver'], images, reports)
    if (len(packets) != protocol['available_intervention_items']
            or {p['item_id'] for p in packets} != {r['item_id'] for r in items['blind_items']}
            or len({r['case_id'] for r in items['resolver']}) != protocol['fixed_ehr_cases']):
        raise ValueError('all_fixed_ehrs_and_intervention_items_required')
    reference_root, reference_sha = REFERENCE
    reference = json.loads(b.checked(reference_root / 'manifest.json', reference_sha, 1024**2, sources))
    reader_path = Path(reader.__file__).resolve()
    reader_relative = str(reader_path.relative_to(ROOT.parent))
    if reference['pins'][reader_relative] != sha256_file(reader_path) or reference['clinical_qualified'] is not False:
        raise ValueError('unchanged_reference_tested_but_unqualified_image_rules_required')
    pins = {str(p): sha256_file(p) for p in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_image_attribution_stress_v1.py', reader_path, Path(evidence.__file__),
        Path(evidence.old.__file__), Path(evidence.interventions.__file__), Path(b.__file__),
        Path(b.policy_module.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        image_rows = [{k: row[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_sha256', 'cxr_model_id')}
            | {'image_policy': policy, **reader.decision(row, policy), 'clinical_qualified': False}
            for _, row in sorted(images.items()) for policy in reader.POLICIES]
        write_private_text(temporary / 'image_policy_table.csv', b.csv_text(image_rows))
        candidate_rows = []
        for grid in bank.values():
            for candidate in grid.values():
                view = b.policy_module.snapshot(candidate); line = view['lineage']; image = images[line['cxr_candidate_id']]
                packet = {'item_id': view['candidate_id'], 'ehr_opacity_state': 'unknown',
                    'report_opacity_state': view['states']['chexbert']['lung_opacity'],
                    'image_evidence': {k: image[k] for k in ('xrv_score', 'margins')}}
                for policy in reader.POLICIES:
                    candidate_rows.append({'case_id': view['case_id'], **line, **judge(packet, policy)})
        write_private_text(temporary / 'candidate_verification_table.csv', b.csv_text(candidate_rows))
        outcomes = [judge(packet, policy) for packet in packets for policy in reader.POLICIES]
        sealed = write_private_text(temporary / 'blind_intervention_outcomes.jsonl',
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in outcomes))
        seal = sha256_file(sealed)
        # References/key are evaluation readouts, not judge inputs.
        text = b.checked(evidence.BANK / 'intervention_key.jsonl', meta['artifact_sha256']['intervention_key'], 4*1024**2, sources)
        items['intervention_key'] = [json.loads(line) for line in text.splitlines()]
        evidence.interventions.validate_items(items)
        comparisons = summarize(outcomes, items['resolver'], items['intervention_key'])
        write_private_text(temporary / 'mechanical_arm_comparison.csv', b.csv_text(comparisons))
        risk = reference['artifacts']['risk_coverage_table.csv']
        risk_pin = risk['sha256'] if isinstance(risk, dict) else risk
        risk_text = b.checked(reference_root / 'risk_coverage_table.csv', risk_pin, 1024**2, sources)
        reference_rows = list(csv.DictReader(io.StringIO(risk_text)))
        if len(reference_rows) != 4 or {r['policy'] for r in reference_rows} != set(reader.POLICIES):
            raise ValueError('all_fixed_reference_policies_required_no_best_mask_selection')
        write_private_json(temporary / 'reference_scope_record.json', {
            'reference_manifest_sha256': reference_sha, 'finding': 'exact_lung_opacity',
            'all_fixed_reference_policies': reference_rows, 'clinical_qualified': False,
            'synthetic_transport_validated': False, 'reference_supplied_by_image_agreement': False})
        summary = {'schema_version': VERSION, 'status': 'image_attribution_stress_diagnostic_complete',
            'fixed_ehr_cases': protocol['fixed_ehr_cases'], 'image_policy_rows': len(image_rows),
            'candidate_verification_rows': len(candidate_rows), 'intervention_outcomes': len(outcomes),
            'available_items': len(packets), 'intended_items': protocol['intended_intervention_items'],
            'missing_arms': protocol['unavailable_arms'], 'image_policies': {},
            'predictions_sha256_before_key': seal, 'clinical_localization_accuracy': None,
            'clinical_false_repair_rate': None, 'new_model_calls': 0, 'actual_regeneration_executed': False,
            'old_choices_or_scores_changed': False, 'raw_reference_or_patient_bodies_read': False,
            'synthetic_bodies_pixels_or_weights_read': False, 'secondary_endpoint_used_for_prediction': False}
        for policy in reader.POLICIES:
            rows = [r for r in outcomes if r['image_policy'] == policy]
            summary['image_policies'][policy] = {
                'intervention_relation_counts': dict(Counter(r['proxy_relation'] for r in rows)),
                'image_slot_status_counts': dict(Counter(r['status'] for r in image_rows if r['image_policy'] == policy)),
                'clinical_faults_confirmed': 0, 'automatic_regeneration_authorized': False}
        write_private_json(temporary / 'summary.json', summary)
        if sha256_file(sealed) != seal or any(sha256_file(p) != h for p, h in sources.items()):
            raise ValueError('sealed_predictions_or_original_sources_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION + '-manifest',
            'sources': sources, 'code_pins': pins, 'clinical_qualified': False,
            'original_selection_changed': False, 'predictions_sha256_before_key': seal,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir() if p.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'image_attribution_stress'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({'status': 'completed', 'new_model_calls': 0}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Numeric cache-only same-finding opposition/attribution stress diagnostic.

All six findings enter every prediction. The intervention target finding is
opened only after sealing and is used for a separate evaluation-only scope
readout. No donor identities, case/model IDs or construction labels enter
the score-only judge. Artifact replacement is not a clinical fault reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
import score_finding_matched_biovil_v1 as scoring
import diagnose_localization_observability_v1 as evidence
from contracts import (PROTECTED_ROOT, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

b = evidence.b
SCHEMA = 'tricompose-finding-matched-stress-v1'
PROTOCOL = ROOT / 'configs/finding_matched_stress_v1.json'
TESTS = ROOT / 'tests/test_finding_matched_stress_v1.py'
POLICIES = ('cached_xrv', 'biovil_mean', 'agree_mean', 'agree_all_templates')
CONTROLS = ('unsafe_report_blame', 'ehr_asymmetry_only')
STATES = scoring.STATES
EXPLICIT = scoring.EXPLICIT
FINDINGS = scoring.FINDINGS
PACKET_FACT_FIELDS = {'ehr', 'report', 'cached_xrv', 'score_pairs'}


def protocol():
    value = json.loads(PROTOCOL.read_text())
    if (value['schema_version'] != SCHEMA + '-protocol' or value['findings'] != list(FINDINGS)
            or value['image_policies'] != list(POLICIES) or value['attribution_controls'] != list(CONTROLS)
            or value['comparison_rule'] != 'any_same_finding_explicit_image_report_opposition'
            or value['attribution_rule'] != 'explicit_three_way_asymmetry_unanimous_target_else_abstain'
            or (value['fixed_ehr_cases'], value['image_slots'], value['candidate_triples'], value['available_items']) != (80, 240, 960, 236)
            or value['available_arms'] != {'no_corruption': 80, 'cxr_swap': 80, 'report_swap': 76}
            or value['unavailable_arms'] != {'report_swap': 4}
            or value['source_roles'] != list(evidence.ROLES)
            or value['predictions_sealed_before_key'] is not True or value['post_hoc_development'] is not True
            or type(value['new_model_calls']) is not int or value['new_model_calls'] != 0
            or any(value[k] is not False for k in ('key_target_finding_is_predictor_input',
                'historical_final_role_is_untouched_test', 'construction_reference_is_independent',
                'mechanical_target_is_clinical_fault', 'report_assertion_scope_qualified',
                'unknown_is_negative', 'threshold_fitting', 'method_selection_after_results',
                'training_allowed', 'clinical_qualified', 'regeneration_authorized', 'original_selection_changed'))):
        raise ValueError('fixed_all_finding_diagnostic_protocol_required')
    return value


def validate_packet(packet):
    if (not isinstance(packet, dict) or set(packet) != {'item_id', 'findings'}
            or not isinstance(packet['item_id'], str) or not re.fullmatch(r'item_[0-9]{4}', packet['item_id'])
            or not isinstance(packet['findings'], dict) or set(packet['findings']) != set(FINDINGS)):
        raise ValueError('opaque_item_and_complete_finding_packet_required')
    for finding, fact in packet['findings'].items():
        if (not isinstance(fact, dict) or set(fact) != PACKET_FACT_FIELDS
                or any(fact[k] not in STATES for k in ('ehr', 'report', 'cached_xrv'))):
            raise ValueError('scores_and_four_states_only_required')
        if finding == 'support_devices' and fact['cached_xrv'] != 'unknown':
            raise ValueError('device_head_unavailable_not_invented')
        scoring.score_readout(fact['score_pairs'])


def image_state(fact, policy):
    if policy not in POLICIES:
        raise ValueError('predeclared_image_policy_required')
    if policy == 'cached_xrv':
        return fact['cached_xrv']
    readout = scoring.score_readout(fact['score_pairs'])
    state = readout['preference_state']
    if policy == 'biovil_mean':
        return state
    if state not in EXPLICIT or fact['cached_xrv'] != state:
        return 'unknown'
    if policy == 'agree_all_templates' and readout['template_pattern'] not in ('all_positive_preference', 'all_negative_preference'):
        return 'unknown'
    return state


def predict(packet, policy):
    validate_packet(packet)
    facts = []
    for finding in FINDINGS:
        source = packet['findings'][finding]
        image = image_state(source, policy)
        ehr, report = source['ehr'], source['report']
        hint = None
        if all(s in EXPLICIT for s in (ehr, image, report)):
            if ehr == report != image:
                hint = 'cxr'
            elif ehr == image != report:
                hint = 'report'
        facts.append({'finding': finding, 'ehr_state': ehr, 'image_state': image,
            'report_state': report, 'ehr_image_relation': scoring.relation(ehr, image),
            'ehr_report_relation': scoring.relation(ehr, report),
            'image_report_relation': scoring.relation(image, report),
            'three_way_comparable': all(s in EXPLICIT for s in (ehr, image, report)),
            'tentative_ehr_asymmetry_target': hint})
    comparable = [r['finding'] for r in facts if r['image_report_relation'] != 'not_comparable']
    oppositions = [r['finding'] for r in facts if r['image_report_relation'] == 'proxy_opposition']
    hints = {r['tentative_ehr_asymmetry_target'] for r in facts if r['tentative_ehr_asymmetry_target'] is not None}
    target = next(iter(hints)) if len(hints) == 1 else None
    return {'item_id': packet['item_id'], 'image_policy': policy, 'facts': facts,
        'comparable_findings': comparable, 'opposition_findings': oppositions,
        'pair_status': 'proxy_opposition_detected' if oppositions else
                       'no_opposition_on_available_subset' if comparable else 'no_comparable_findings',
        'proxy_mismatch_detected': bool(oppositions),
        'unsafe_report_blame': 'report' if oppositions else None,
        'ehr_asymmetry_only': target, 'ehr_asymmetry_status': 'single_target_proxy_hint' if target else
            'conflicting_target_hints_abstain' if len(hints) > 1 else 'no_three_way_asymmetry',
        'clinical_fault_confirmed': None, 'clinical_localization_accuracy': None,
        'clinical_qualified': False, 'regeneration_authorized': False}


def numeric_images(records, bank):
    expected = {}
    for grid in bank.values():
        for candidate in grid.values():
            obs = b.policy_module.snapshot(candidate)
            line = obs['lineage']
            image = {'case_id': obs['case_id'], **{k: line[k] for k in
                ('cxr_candidate_id', 'cxr_model_id', 'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256')}}
            if expected.setdefault(line['cxr_candidate_id'], image) != image:
                raise ValueError('unchanged_shared_image_lineage_required')
    result, same_bytes = {}, {}
    for record in records:
        item = record['cxr_candidate_id']
        if (item in result or item not in expected or any(record[k] != value for k, value in expected[item].items())
                or set(record) != set(scoring.IMAGE_FIELDS) | {'status', 'score_pairs'}
                or record['status'] not in ('scored', 'failed_without_replacement')
                or (record['status'] == 'scored') != (record['score_pairs'] is not None)):
            raise ValueError('exact_unique_numeric_image_outcome_required')
        pairs = record['score_pairs']
        if pairs is not None:
            if set(pairs) != set(FINDINGS):
                raise ValueError('complete_six_disease_score_inventory_required')
            for value in pairs.values():
                scoring.score_readout(value)
        signature = (record['status'], pairs)
        if same_bytes.setdefault(record['cxr_sha256'], signature) != signature:
            raise ValueError('same_image_hash_same_scores_required')
        # Strip paths/model/case IDs from predictor evidence, retain hash for binding only.
        result[item] = {'cxr_sha256': record['cxr_sha256'], 'score_pairs': deepcopy(pairs)}
    if set(result) != set(expected):
        raise ValueError('outcome_for_every_original_image_required')
    return result


def packets(resolver, bank, images):
    original = evidence.bind_packets(bank, resolver)
    by_id = {r['item_id']: r for r in original}
    result = []
    for row in resolver:
        image = images[row['displayed_cxr_candidate_id']]
        if image['cxr_sha256'] != row['displayed_cxr_sha256']:
            raise ValueError('actual_displayed_image_not_original_parent_required')
        old = by_id[row['item_id']]['states']
        result.append({'item_id': row['item_id'], 'findings': {finding: {
            'ehr': old['ehr'][finding], 'cached_xrv': old['xrv'][finding],
            'report': old['chexbert'][finding],
            'score_pairs': deepcopy(image['score_pairs'][finding]) if image['score_pairs'] is not None else None}
            for finding in FINDINGS}})
        validate_packet(result[-1])
    return result


def summarize(outcomes, resolver, key):
    roles = {r['item_id']: r['benchmark_split'] for r in resolver}
    truth = {r['item_id']: r for r in key}
    expected = {(item, policy) for item in roles for policy in POLICIES}
    if ({(r['item_id'], r['image_policy']) for r in outcomes} != expected
            or len(outcomes) != len(expected) or set(truth) != set(roles)):
        raise ValueError('complete_blind_outcomes_and_evaluation_keys_required')
    groups = defaultdict(list)
    for row in outcomes:
        for role in ('overall', roles[row['item_id']]):
            groups[row['image_policy'], role, truth[row['item_id']]['intervention_type']].append(row)
    rows = []
    for (policy, role, arm), group in sorted(groups.items()):
        targeted = []
        for row in group:
            # Evaluation-only diagnostic; NEVER passed to predict().
            finding = truth[row['item_id']]['targeted_finding']
            if finding is not None:
                if finding not in FINDINGS:
                    raise ValueError('construction_finding_outside_predeclared_scope')
                targeted.append(next(f for f in row['facts'] if f['finding'] == finding))
        for control in CONTROLS:
            flags = [r for r in group if r[control] is not None]
            matches = sum(r[control] == truth[r['item_id']]['intervention_target'] for r in flags)
            rows.append({'image_policy': policy, 'attribution_control': control, 'source_role': role,
                'mechanical_arm': arm, 'available_items': len(group),
                'proxy_mismatch_items': sum(r['proxy_mismatch_detected'] for r in group),
                'no_comparable_items': sum(r['pair_status'] == 'no_comparable_findings' for r in group),
                'tentative_target_items': len(flags), 'abstentions_not_clean_judgments': len(group)-len(flags),
                'matches_known_replacement_target': matches,
                'disagrees_with_known_replacement_target': len(flags)-matches,
                'target_recovery_over_all_available_intervened': matches/len(group) if arm != 'no_corruption' else None,
                'evaluation_only_target_finding_items': len(targeted),
                'evaluation_only_target_finding_comparable': sum(r['image_report_relation'] != 'not_comparable' for r in targeted),
                'evaluation_only_target_finding_opposition': sum(r['image_report_relation'] == 'proxy_opposition' for r in targeted),
                'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None,
                'regeneration_authorized': False})
    return rows


def load_scores(root, expected, sources):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    meta = json.loads(b.checked(root / 'manifest.json', expected, 1024**2, sources))
    if (meta['schema_version'] != scoring.SCHEMA + '-manifest'
            or meta['clinical_qualified'] is not False or meta['original_selection_changed'] is not False
            or meta['regeneration_authorized'] is not False or meta['new_training_calls'] != 0 or meta['new_generation_calls'] != 0):
        raise ValueError('completed_unqualified_frozen_scoring_run_required')
    data = json.loads(b.checked(root / 'image_scores.json', meta['artifacts']['image_scores.json'], 4*1024**2, sources))
    if data['schema_version'] != scoring.SCHEMA + '-image-scores':
        raise ValueError('same_finding_numeric_score_schema_required')
    # Do not reopen body/pixel/weight source pins on this cache-only CPU worker.
    return data['records'], meta


def run(args):
    b.cpu_guard()
    rules = protocol()
    sources = {}
    bank, _ = b.load_primary(sources)
    records, scoring_manifest = load_scores(args.scoring_run, args.scoring_manifest_sha256, sources)
    images = numeric_images(records, bank)
    if len(images) != rules['image_slots']:
        raise ValueError('full_240_image_evidence_required')
    meta = json.loads(b.checked(evidence.BANK / 'manifest.json', evidence.BANK_SHA, 1024**2, sources))
    if meta['counts'] != rules['available_arms'] or meta['source']['coverage']['unavailable'] != rules['unavailable_arms']:
        raise ValueError('all_available_and_missing_arms_required')
    items = {}
    for name in ('blind_items', 'resolver'):
        text = b.checked(evidence.BANK / (name + '.jsonl'), meta['artifact_sha256'][name], 4*1024**2, sources)
        items[name] = [json.loads(line) for line in text.splitlines()]
    values = packets(items['resolver'], bank, images)
    if (len(values) != rules['available_items'] or len({r['case_id'] for r in items['resolver']}) != rules['fixed_ehr_cases']
            or {r['item_id'] for r in values} != {r['item_id'] for r in items['blind_items']}):
        raise ValueError('complete_original_intervention_inventory_required')
    code_pins = {str(p): sha256_file(p) for p in (Path(__file__), PROTOCOL, TESTS,
        Path(scoring.__file__), scoring.PROTOCOL, Path(evidence.__file__),
        Path(evidence.interventions.__file__), Path(b.__file__), Path(b.policy_module.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', rules)
        outcomes = [predict(value, policy) for value in values for policy in POLICIES]
        sealed = write_private_text(temporary / 'blind_predictions.jsonl',
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in outcomes))
        seal = sha256_file(sealed)
        # No source-role, intervention-type or targeted-finding key is used above.
        text = b.checked(evidence.BANK / 'intervention_key.jsonl', meta['artifact_sha256']['intervention_key'], 4*1024**2, sources)
        items['intervention_key'] = [json.loads(line) for line in text.splitlines()]
        evidence.interventions.validate_items(items)
        table = summarize(outcomes, items['resolver'], items['intervention_key'])
        write_private_text(temporary / 'mechanical_comparison_table.csv', b.csv_text(table))
        summary = {'schema_version': SCHEMA, 'fixed_ehr_cases': 80, 'available_items': len(values),
            'unavailable_arms': rules['unavailable_arms'], 'blind_prediction_rows': len(outcomes),
            'finding_diagnostics': len(outcomes)*len(FINDINGS), 'comparison_rows': len(table),
            'scoring_manifest_sha256': args.scoring_manifest_sha256,
            'predictions_sha256_before_key': seal, 'new_model_calls': 0,
            'raw_or_synthetic_bodies_pixels_or_weights_read': False,
            'key_target_finding_used_for_prediction': False, 'clinical_qualified': False,
            'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None,
            'regeneration_authorized': False, 'original_selection_changed': False,
            'mechanical_replacement_target_is_clinical_fault': False,
            'interpretation': 'Same-finding proxy comparisons, not clinical attribution. Construction used historical classifier proposals and the pool has been inspected. Any-finding flags can increase with scope, including unchanged controls. Evaluation-only target-finding readouts are not deployable predictor inputs. No method or threshold is selected after these results.'}
        summary['image_policies'] = {policy: {
            'pair_status_counts': dict(Counter(r['pair_status'] for r in outcomes if r['image_policy'] == policy)),
            'ehr_asymmetry_status_counts': dict(Counter(r['ehr_asymmetry_status'] for r in outcomes if r['image_policy'] == policy))}
            for policy in POLICIES}
        write_private_json(temporary / 'summary.json', summary)
        if sha256_file(sealed) != seal or any(sha256_file(path) != pin for path, pin in {**sources, **code_pins}.items()):
            raise ValueError('sealed_predictions_or_sources_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'sources': sources, 'code_pins': code_pins, 'predictions_sha256_before_key': seal,
            'scoring_plan_manifest_sha256': scoring_manifest['plan_manifest_sha256'],
            'scoring_manifest_sha256': args.scoring_manifest_sha256,
            'clinical_qualified': False, 'regeneration_authorized': False,
            'original_selection_changed': False,
            'artifacts': {p.name: sha256_file(p) for p in temporary.iterdir() if p.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring-run', required=True)
    parser.add_argument('--scoring-manifest-sha256', required=True)
    parser.add_argument('--output-root', default=str(b.BASE / 'finding_matched_stress'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target = run(args)
        print(json.dumps({'status': 'completed_cached_same_finding_diagnostic', 'new_model_calls': 0,
                          'manifest_sha256': sha256_file(target / 'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

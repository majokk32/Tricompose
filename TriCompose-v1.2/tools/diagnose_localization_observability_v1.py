#!/usr/bin/env python3
"""Blinded proxy signals and empirical mechanical-target ambiguity, not gold."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
import benchmark_probe_repair_v1 as b
import audit_localization_evidence as old
import build_targeted_interventions as interventions
from tricompose_v12.live_workers import check_pins
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

VERSION = 'tricompose-localization-observability-v1'
PROTOCOL = ROOT / 'configs/localization_observability_v1.json'
BANK = b.BASE / 'benchmarks/pool80_label_targeted_stratified_20261001_001'
BANK_SHA = '0f8f7c901c636596217b94f5eab9a3b81558a48427e9e6bec82bfe86d0139aea'
ENDPOINT = b.BASE / 'benchmarks/pool80_targeted_biovil_20261001_001/scores.json'
ENDPOINT_SHA = 'f91f6c186ff98777cace43b598710f062a1efc81a08596d2fbc6d9c8b0cf7645'
TARGETS = ('none', 'cxr', 'report')
FEATURES = ('edge_counts_only', 'named_finding_states')
ROLES = ('development', 'calibration', 'final_test')


def validate_packet(packet):
    if (not isinstance(packet, dict) or set(packet) != {'item_id', 'states'}
            or not isinstance(packet['item_id'], str) or not re.fullmatch(r'item_[0-9]{4}', packet['item_id'])
            or not isinstance(packet['states'], dict) or set(packet['states']) != {'ehr', 'xrv', 'chexbert'}
            or any(not isinstance(vector, dict) or set(vector) != set(b.policy_module.FINDINGS)
                or any(state not in ('positive', 'negative', 'uncertain', 'unknown') for state in vector.values())
                for vector in packet['states'].values())):
        raise ValueError('only_complete_opaque_item_and_four_state_evidence_allowed')


def bind_packets(bank, resolver):
    ehr, images, reports = {}, {}, {}
    for case, grid in bank.items():
        for source in grid.values():
            view = b.policy_module.snapshot(source)
            line = view['lineage']
            for index, key, value in (
                (ehr, case, (line['ehr_sha256'], view['states']['ehr'])),
                (images, line['cxr_candidate_id'], (line['cxr_sha256'], view['states']['xrv'])),
                (reports, line['report_candidate_id'], (line['report_sha256'], view['states']['chexbert']))):
                if index.setdefault(key, value) != value:
                    raise ValueError('shared_artifact_evidence_must_remain_unchanged')
    packets = []
    seen = set()
    for row in resolver:
        if row['item_id'] in seen or row['benchmark_split'] not in ROLES:
            raise ValueError('unique_items_and_declared_source_roles_required')
        seen.add(row['item_id'])
        sources = (ehr[row['case_id']], images[row['displayed_cxr_candidate_id']], reports[row['displayed_report_candidate_id']])
        expected = (row['ehr_sha256'], row['displayed_cxr_sha256'], row['displayed_report_sha256'])
        if tuple(value[0] for value in sources) != expected:
            raise ValueError('exact_displayed_artifact_and_fixed_ehr_binding_required')
        packet = {'item_id': row['item_id'], 'states': {k: deepcopy(v[1]) for k, v in zip(('ehr', 'xrv', 'chexbert'), sources)}}
        validate_packet(packet)
        packets.append(packet)
    return packets


def edge_counts(packet):
    validate_packet(packet)
    vectors = packet['states']
    explicit = {'positive', 'negative'}
    result = {}
    for name, left, right in (('ehr_cxr', 'ehr', 'xrv'), ('ehr_report', 'ehr', 'chexbert'), ('cxr_report', 'xrv', 'chexbert')):
        known = {f for f, s in vectors[left].items() if s in explicit}
        pairs = {f for f in known if vectors[right][f] in explicit}
        supported = {f for f in pairs if vectors[left][f] == vectors[right][f]}
        result[name] = {'known': len(known), 'comparable': len(pairs), 'support': len(supported),
            'positive_support': sum(vectors[left][f] == 'positive' for f in supported), 'opposition': len(pairs - supported)}
    return result


def signature(packet, feature):
    validate_packet(packet)
    if feature not in FEATURES:
        raise ValueError('declared_evidence_projection_required')
    value = edge_counts(packet) if feature == FEATURES[0] else packet['states']
    return hashlib.sha256(json.dumps({'feature_view': feature, 'evidence': value}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def predict(packet):
    validate_packet(packet)
    states = packet['states']
    evidence = old.triad_pattern(states['ehr'], states['xrv'], states['chexbert'])
    return {'item_id': packet['item_id'], **evidence,
        'tentative_proxy_target': {'report_candidate_signal': 'report', 'cxr_candidate_signal': 'cxr'}.get(evidence['pattern']),
        'confirmed_faulty_modality': None, 'clinical_qualified': False, 'regeneration_authorized': False}


def collision_summary(rows, feature, role):
    """A sample-dependent full-information ceiling, not a trained classifier."""
    groups = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        if row['target'] not in TARGETS:
            raise ValueError('mechanical_target_class_required')
        groups[row['signature']][row['target']] += 1
        totals[row['target']] += 1
    count = len(rows)
    correct_ceiling = sum(max(values.values()) for values in groups.values())
    balanced = (sum(max(values.get(target, 0) / totals[target] for target in TARGETS)
        for values in groups.values()) / len(TARGETS)) if all(totals[t] for t in TARGETS) else None
    summary = {'feature_view': feature, 'source_role': role, 'available_items': count,
        'signature_groups': len(groups), 'mixed_target_signature_groups': sum(len(v) > 1 for v in groups.values()),
        'items_in_mixed_target_signatures': sum(sum(v.values()) for v in groups.values() if len(v) > 1),
        'unavoidable_empirical_mechanical_target_errors': count - correct_ceiling,
        'empirical_target_reconstruction_upper_bound': correct_ceiling / count if count else None,
        'empirical_balanced_target_reconstruction_upper_bound': balanced,
        'majority_target_baseline': max(totals.values()) / count if count else None,
        'clinical_localization_accuracy': None, 'new_policy_trained': False,
        'bound_applies_to_arbitrary_new_cases': False}
    cells = [{'feature_view': feature, 'source_role': role, 'signature_sha256': key,
        'items': sum(values.values()), **{target + '_items': values.get(target, 0) for target in TARGETS}}
        for key, values in sorted(groups.items())]
    return summary, cells


def validate_protocol(protocol):
    if (protocol['feature_views'] != list(FEATURES) or protocol['mechanical_target_classes'] != list(TARGETS)
            or protocol['source_roles'] != list(ROLES) or protocol['fixed_ehr_cases'] != 80
            or protocol['available_items'] != 236 or protocol['intended_items'] != 240
            or protocol['prediction_rule'] != 'unchanged_audit_localization_evidence_triad_pattern'
            or protocol['key_read_after_prediction_seal'] is not True
            or any(protocol[key] is not False for key in ('historical_final_role_is_untouched_test',
                'secondary_endpoint_used_for_prediction', 'construction_labels_are_independent_evaluation',
                'mechanical_target_is_clinical_fault', 'clinical_qualified', 'training_allowed',
                'regeneration_authorized', 'actual_regeneration_executed'))
            or type(protocol['new_model_calls']) is not int or protocol['new_model_calls'] != 0):
        raise ValueError('frozen_diagnostic_projection_and_claim_boundary_required')


def run(args):
    b.cpu_guard()
    sources = {}
    bank, _ = b.load_primary(sources)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    metadata = json.loads(b.checked(BANK / 'manifest.json', BANK_SHA, 1024**2, sources))
    if metadata['counts'] != protocol['available_arms'] or metadata['source']['coverage']['unavailable'] != protocol['unavailable_arms']:
        raise ValueError('all_available_and_missing_intervention_arms_required')
    items = {}
    for name in ('blind_items', 'resolver'):
        text = b.checked(BANK / (name + '.jsonl'), metadata['artifact_sha256'][name], 4*1024**2, sources)
        items[name] = [json.loads(line) for line in text.splitlines()]
    if {r['item_id'] for r in items['blind_items']} != {r['item_id'] for r in items['resolver']}:
        raise ValueError('blind_item_inventory_required')
    packets = bind_packets(bank, items['resolver'])
    if len(packets) != protocol['available_items'] or len({r['case_id'] for r in items['resolver']}) != protocol['fixed_ehr_cases']:
        raise ValueError('complete_eighty_case_available_inventory_required')
    pins = {str(path): sha256_file(path) for path in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_localization_observability_v1.py', Path(old.__file__),
        Path(interventions.__file__), Path(b.__file__), Path(b.policy_module.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        predictions = [predict(packet) for packet in packets]
        sealed = write_private_text(temporary / 'blind_predictions.jsonl',
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in predictions))
        seal = sha256_file(sealed)
        # Key and independent scalar endpoint only enter post-seal diagnostics.
        text = b.checked(BANK / 'intervention_key.jsonl', metadata['artifact_sha256']['intervention_key'], 4*1024**2, sources)
        items['intervention_key'] = [json.loads(line) for line in text.splitlines()]
        validation = interventions.validate_items(items)
        truth = {r['item_id']: r for r in items['intervention_key']}
        resolver = {r['item_id']: r for r in items['resolver']}
        endpoint = json.loads(b.checked(ENDPOINT, ENDPOINT_SHA, 4*1024**2, sources))
        if (endpoint['scorer_read_intervention_key'] is not False or endpoint['source']['bank_manifest_sha256'] != BANK_SHA
                or endpoint['clinical_localization_accuracy'] is not None or endpoint['probability_calibration_performed'] is not False):
            raise ValueError('same_blind_uncalibrated_artifact_swap_endpoint_required')
        scores = {}
        for row in endpoint['records']:
            item = resolver[row['item_id']]
            if (row['item_id'] in scores or any(row[k] != item[k] for k in ('case_id', 'benchmark_split', 'displayed_cxr_sha256', 'displayed_report_sha256'))
                    or type(row['biovil_raw_cosine']) not in (int, float) or not math.isfinite(row['biovil_raw_cosine'])
                    or not -1 <= row['biovil_raw_cosine'] <= 1):
                raise ValueError('unique_finite_same_artifact_secondary_scores_required')
            scores[row['item_id']] = row['biovil_raw_cosine']
        if set(scores) != set(resolver):
            raise ValueError('complete_secondary_inventory_required')
        collisions, cells = [], []
        for feature in FEATURES:
            for role in ('overall', *ROLES):
                rows = [{'signature': signature(packet, feature), 'target': truth[packet['item_id']]['intervention_target']}
                    for packet in packets if role == 'overall' or resolver[packet['item_id']]['benchmark_split'] == role]
                summary, group_cells = collision_summary(rows, feature, role)
                collisions.append(summary); cells.extend(group_cells)
        diagnostic = []
        for row in predictions:
            gold = truth[row['item_id']]
            diagnostic.append({**row, 'source_role': resolver[row['item_id']]['benchmark_split'],
                'mechanical_target': gold['intervention_target'], 'mechanical_intervention': gold['intervention_type'],
                'tentative_proxy_target_matches_mechanical_target': row['tentative_proxy_target'] == gold['intervention_target'] if row['tentative_proxy_target'] else None,
                'existing_biovil_raw_cosine_secondary': scores[row['item_id']], 'clinical_localization_accuracy': None})
        summary = {'schema_version': VERSION, 'status': 'cached_localization_observability_diagnostic_complete',
            'available_items': len(packets), 'intended_items': protocol['intended_items'], 'missing_arms': protocol['unavailable_arms'],
            'source_fixed_ehr_cases': validation['cases'], 'source_arm_counts': validation['arm_counts'],
            'pattern_counts': dict(Counter(r['pattern'] for r in predictions)),
            'tentative_proxy_target_items': sum(r['tentative_proxy_target'] is not None for r in predictions),
            'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None,
            'predictions_sha256_before_key': seal, 'mechanical_targets_are_clinical_faults': False,
            'construction_labels_are_independent_evaluation': False,
            'historical_final_role_is_untouched_test': False, 'new_model_calls': 0,
            'actual_regeneration_executed': False, 'old_choices_or_rules_changed': False,
            'source_bodies_pixels_or_weights_read': False, 'secondary_endpoint_used_for_prediction': False}
        write_private_text(temporary / 'item_diagnostics.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in diagnostic))
        write_private_text(temporary / 'collision_summary.csv', b.csv_text(collisions))
        write_private_text(temporary / 'signature_cells.csv', b.csv_text(cells))
        write_private_json(temporary / 'summary.json', summary)
        if sha256_file(sealed) != seal or any(sha256_file(path) != pin for path, pin in sources.items()):
            raise ValueError('sealed_predictions_or_sources_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION + '-manifest',
            'sources': sources, 'code_pins': pins, 'original_selection_changed': False, 'clinical_qualified': False,
            'predictions_sha256_before_key': seal, 'artifacts': {f.name: {'sha256': sha256_file(f)} for f in temporary.iterdir() if f.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'localization_observability'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, _ = run(args)
        print(json.dumps({'status': 'cached_localization_observability_diagnostic_complete', 'manifest_sha256': sha256_file(target / 'manifest.json'), 'new_model_calls': 0}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

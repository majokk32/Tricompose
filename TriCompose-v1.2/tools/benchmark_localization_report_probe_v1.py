#!/usr/bin/env python3
"""Cache-only same-image report-probe corroboration; never clinical fault gold."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
import diagnose_localization_observability_v1 as evidence
from tricompose_v12.live_workers import check_pins
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

b = evidence.b
VERSION = 'tricompose-localization-report-probe-v1'
PROTOCOL = ROOT / 'configs/localization_report_probe_v1.json'
METHODS = ('no_report_probe', 'uniform_report_probe', 'anchored_report_probe')
EXPLICIT = {'positive', 'negative'}


def validate_protocol(protocol):
    expected = json.loads(PROTOCOL.read_text())
    if protocol != expected or protocol.get('methods') != list(METHODS):
        raise ValueError('unchanged_complete_protocol_required')
    if (protocol['probe_report_model'] != 'maira2'
            or protocol['initial_report_model'] != 'chexagent2'
            or protocol['simulated_charge_per_requested_report'] != 2
            or protocol['key_decoded_after_prediction_seal'] is not True
            or any(protocol[k] is not False for k in (
                'historical_final_role_is_untouched_test', 'can_create_a_new_target_from_probe',
                'reports_are_independent_clinical_evidence', 'mechanical_target_is_clinical_fault',
                'construction_labels_are_independent_evaluation', 'secondary_endpoint_used_for_prediction',
                'old_choices_or_rules_changed', 'clinical_qualified', 'training_allowed',
                'regeneration_authorized', 'actual_regeneration_executed'))
            or type(protocol['new_model_calls']) is not int or protocol['new_model_calls'] != 0):
        raise ValueError('no_new_targets_independent_truth_or_execution_allowed')


def wants_probe(packet, method):
    evidence.validate_packet(packet)
    if method not in METHODS:
        raise ValueError('declared_probe_method_required')
    return method == 'uniform_report_probe' or (
        method == 'anchored_report_probe' and any(s in EXPLICIT for s in packet['states']['ehr'].values()))


def target_facts(packet, target):
    evidence.validate_packet(packet)
    vectors = packet['states']
    result = []
    for finding in b.policy_module.FINDINGS:
        e, x, r = (vectors[k][finding] for k in ('ehr', 'xrv', 'chexbert'))
        if all(s in EXPLICIT for s in (e, x, r)) and (
                (target == 'report' and e == x != r) or (target == 'cxr' and e == r != x)):
            result.append(finding)
    return sorted(result)


def validate_probe(packet, probe):
    if (not isinstance(probe, dict) or set(probe) != {'report_states', 'distinct_report_bytes', 'same_image_binding_verified'}
            or type(probe['distinct_report_bytes']) is not bool
            or probe['same_image_binding_verified'] is not True):
        raise ValueError('exact_same_image_distinctness_and_state_packet_required')
    changed = deepcopy(packet)
    changed['states']['chexbert'] = probe['report_states']
    evidence.validate_packet(changed)


def verify(packet, method, probe=None):
    """A probe can corroborate an existing signal, never manufacture a target."""
    initial = evidence.predict(packet)
    requested = wants_probe(packet, method)
    if not requested and probe is not None:
        raise ValueError('unrequested_probe_evidence_forbidden')
    target = initial['tentative_proxy_target']
    facts = target_facts(packet, target)
    result = {'item_id': packet['item_id'], 'method': method,
        'initial_pattern': initial['pattern'], 'initial_tentative_proxy_target': target,
        'known_direct_fact_count': initial['known_direct_fact_count'],
        'probe_requested': requested, 'simulated_additional_report_charges': 2 if requested else 0,
        'probe_status': 'not_requested', 'original_target_fact_ids': facts,
        'corroborated_fact_ids': [], 'challenged_fact_ids': [], 'uninformative_fact_ids': [],
        'corroborated_proxy_target': None, 'confirmed_faulty_modality': None,
        'clinical_localization_accuracy': None, 'reports_are_independent_clinical_evidence': False,
        'clinical_qualified': False, 'regeneration_authorized': False}
    if not requested:
        return result
    if probe is None:
        result['probe_status'] = 'unavailable'
        return result
    validate_probe(packet, probe)
    if not probe['distinct_report_bytes']:
        result['probe_status'] = 'duplicate_report_bytes'
        return result
    if target is None:
        result['probe_status'] = 'observed_without_unique_initial_target'
        return result
    for finding in facts:
        state = probe['report_states'][finding]
        key = ('corroborated_fact_ids' if state == packet['states']['ehr'][finding]
            else 'challenged_fact_ids' if state in EXPLICIT else 'uninformative_fact_ids')
        result[key].append(finding)
    if result['challenged_fact_ids']:
        result['probe_status'] = 'challenged_or_mixed'
    elif result['uninformative_fact_ids']:
        result['probe_status'] = 'unconfirmed_missing_explicit_evidence'
    elif facts:
        result['probe_status'] = 'all_original_target_facts_corroborated'
        result['corroborated_proxy_target'] = target
    else:
        raise ValueError('unique_original_target_must_have_explicit_finding_witness')
    return result


def probe_store(bank, resolver, packets):
    """Adapter only: bind MAIRA-2 to displayed CXR, not to the original donor EHR."""
    images, reports = {}, {}
    for grid in bank.values():
        for raw in grid.values():
            view = b.policy_module.snapshot(raw)
            line = view['lineage']
            report_identity = (line['report_model_id'], line['report_sha256'])
            if reports.setdefault(line['report_candidate_id'], report_identity) != report_identity:
                raise ValueError('unchanged_original_report_expert_and_hash_required')
            if line['report_model_id'] != 'maira2':
                continue
            key = line['cxr_candidate_id']
            value = {'cxr_sha256': line['cxr_sha256'], 'xrv_states': view['states']['xrv'],
                'report_id': line['report_candidate_id'], 'report_sha256': line['report_sha256'],
                'report_states': view['states']['chexbert']}
            if images.setdefault(key, value) != value:
                raise ValueError('unique_same_image_fixed_maira_probe_required')
    lookup = {r['item_id']: r for r in resolver}
    result = {}
    for packet in packets:
        row = lookup[packet['item_id']]
        if reports.get(row['displayed_report_candidate_id']) != ('chexagent2', row['displayed_report_sha256']):
            raise ValueError('initial_displayed_report_must_be_fixed_chexagent2')
        source = images.get(row['displayed_cxr_candidate_id'])
        if source is None or source['cxr_sha256'] != row['displayed_cxr_sha256'] or source['xrv_states'] != packet['states']['xrv']:
            raise ValueError('all_probe_parents_must_match_displayed_image_hash_and_states')
        result[packet['item_id']] = source
    return result


def summarize(rows, resolver, key):
    groups = defaultdict(list)
    lookup = {r['item_id']: r for r in resolver}
    truth = {r['item_id']: r for r in key}
    for row in rows:
        item = row['item_id']
        for role in ('overall', lookup[item]['benchmark_split']):
            groups[row['method'], role, truth[item]['intervention_type']].append(row)
    result = []
    for (method, role, arm), records in sorted(groups.items()):
        corroborated = [r for r in records if r['corroborated_proxy_target'] is not None]
        result.append({'method': method, 'source_role': role, 'mechanical_arm': arm,
            'available_items': len(records), 'fixed_ehr_cases': len({lookup[r['item_id']]['case_id'] for r in records}),
            'initial_tentative_signals': sum(r['initial_tentative_proxy_target'] is not None for r in records),
            'requested_report_probes': sum(r['probe_requested'] for r in records),
            'simulated_additional_report_charges': sum(r['simulated_additional_report_charges'] for r in records),
            'corroborated_proxy_signals': len(corroborated),
            'corroborated_matches_mechanical_target': sum(r['corroborated_proxy_target'] == truth[r['item_id']]['intervention_target'] for r in corroborated),
            'challenged_or_mixed_signals': sum(r['probe_status'] == 'challenged_or_mixed' for r in records),
            'missing_explicit_corroboration': sum(r['probe_status'] == 'unconfirmed_missing_explicit_evidence' for r in records),
            'unavailable_probes': sum(r['probe_status'] == 'unavailable' for r in records),
            'duplicate_byte_probes': sum(r['probe_status'] == 'duplicate_report_bytes' for r in records),
            'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None})
    return result


def run(args):
    b.cpu_guard()
    sources = {}
    bank, _ = b.load_primary(sources)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    meta = json.loads(b.checked(evidence.BANK / 'manifest.json', evidence.BANK_SHA, 1024**2, sources))
    if meta['counts'] != protocol['available_arms'] or meta['source']['coverage']['unavailable'] != protocol['unavailable_arms']:
        raise ValueError('complete_available_and_unavailable_arm_inventory_required')
    items = {}
    for name in ('blind_items', 'resolver'):
        text = b.checked(evidence.BANK / (name + '.jsonl'), meta['artifact_sha256'][name], 4*1024**2, sources)
        items[name] = [json.loads(line) for line in text.splitlines()]
    packets = evidence.bind_packets(bank, items['resolver'])
    if (len(packets) != protocol['available_items']
            or {p['item_id'] for p in packets} != {r['item_id'] for r in items['blind_items']}
            or len({r['case_id'] for r in items['resolver']}) != protocol['fixed_ehr_cases']):
        raise ValueError('unchanged_complete_blind_inventory_required')
    probes = probe_store(bank, items['resolver'], packets)
    lookup = {r['item_id']: r for r in items['resolver']}
    pins = {str(p): sha256_file(p) for p in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_localization_report_probe_v1.py', Path(evidence.__file__),
        Path(evidence.old.__file__), Path(evidence.interventions.__file__), Path(b.__file__),
        Path(b.policy_module.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        outcomes, requests = [], []
        for packet in packets:
            for method in METHODS:
                probe = None
                if wants_probe(packet, method):
                    source = probes[packet['item_id']]
                    row = lookup[packet['item_id']]
                    distinct = source['report_sha256'] != row['displayed_report_sha256']
                    probe = {'report_states': deepcopy(source['report_states']),
                        'distinct_report_bytes': distinct, 'same_image_binding_verified': True}
                    requests.append({'item_id': packet['item_id'], 'method': method,
                        'displayed_cxr_candidate_id': row['displayed_cxr_candidate_id'],
                        'displayed_cxr_sha256': row['displayed_cxr_sha256'],
                        'probe_report_model': 'maira2', 'probe_report_candidate_id': source['report_id'],
                        'probe_report_sha256': source['report_sha256'], 'distinct_report_bytes': distinct,
                        'simulated_additional_report_charges': 2, 'actual_new_model_calls': 0})
                outcomes.append(verify(packet, method, probe))
        prediction = write_private_text(temporary / 'blind_probe_outcomes.jsonl',
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in outcomes))
        seal = sha256_file(prediction)
        write_private_text(temporary / 'probe_requests.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in requests))
        # Mechanical key is decoded only after all three methods' outcomes seal.
        text = b.checked(evidence.BANK / 'intervention_key.jsonl', meta['artifact_sha256']['intervention_key'], 4*1024**2, sources)
        items['intervention_key'] = [json.loads(line) for line in text.splitlines()]
        validation = evidence.interventions.validate_items(items)
        comparisons = summarize(outcomes, items['resolver'], items['intervention_key'])
        write_private_text(temporary / 'method_by_arm.csv', b.csv_text(comparisons))
        result = {'schema_version': VERSION, 'status': 'cached_report_probe_diagnostic_complete',
            'available_items': len(packets), 'intended_items': protocol['intended_items'],
            'missing_arms': protocol['unavailable_arms'], 'fixed_ehr_cases': validation['cases'],
            'outcome_rows': len(outcomes), 'methods': {}, 'predictions_sha256_before_key': seal,
            'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None,
            'same_image_reports_are_independent': False, 'historical_final_role_is_untouched_test': False,
            'new_model_calls': 0, 'actual_regeneration_executed': False,
            'old_choices_or_rules_changed': False, 'source_bodies_pixels_or_weights_read': False,
            'secondary_endpoint_read': False}
        for method in METHODS:
            rows = [r for r in outcomes if r['method'] == method]
            result['methods'][method] = {'items': len(rows),
                'initial_tentative_signals': sum(r['initial_tentative_proxy_target'] is not None for r in rows),
                'requested_report_probes': sum(r['probe_requested'] for r in rows),
                'simulated_additional_report_charges': sum(r['simulated_additional_report_charges'] for r in rows),
                'corroborated_proxy_signals': sum(r['corroborated_proxy_target'] is not None for r in rows),
                'probe_status_counts': dict(Counter(r['probe_status'] for r in rows))}
        write_private_json(temporary / 'summary.json', result)
        if sha256_file(prediction) != seal or any(sha256_file(path) != pin for path, pin in sources.items()):
            raise ValueError('sealed_outcomes_or_original_sources_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION + '-manifest',
            'sources': sources, 'code_pins': pins, 'original_selection_changed': False,
            'clinical_qualified': False, 'predictions_sha256_before_key': seal,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir() if p.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'localization_report_probes'))
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

#!/usr/bin/env python3
"""Cache-only joint conditioning/mismatch heuristic stress test, not clinical gold.

Always keep the recipient EHR/final prompt fixed after an artifact swap.
Prediction consumes numeric scores only, never the known donor/parent identity.
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
import benchmark_image_attribution_stress_v1 as image
import score_prompt_image_conditioning_v1 as conditioning
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.live_workers import check_pins

b, evidence, reader = image.b, image.evidence, image.reader
VERSION = 'tricompose-conditioning-attribution-v1'
PROTOCOL = ROOT / 'configs/conditioning_attribution_v1.json'
SOURCE = b.BASE / 'prompt_image_conditioning_runs/conditioning_pool240_12805302'
SOURCE_SHA = '241ecf98b9be6dbe2da9c8bba2460cc3ad28dbc61f33827d393f034034bd65bb'
METHODS = ('report_blame_on_opposition', 'conditioning_margin_only',
           'joint_margin0', 'joint_rank1', 'joint_rank5')
CONDITION_FIELDS = ('status', 'matched_cosine', 'matched_minus_other_mean',
                    'best_rank', 'worst_rank', 'prompt_groups')


def validate_protocol(value):
    if (value != json.loads(PROTOCOL.read_text()) or value['methods'] != list(METHODS)
            or value['image_policies'] != list(reader.POLICIES)
            or value['rank_cutoffs'] != [1, 5] or value['margin_threshold'] != 0
            or (value['fixed_ehr_cases'], value['image_slots'], value['candidate_slots'], value['available_items']) != (80, 240, 960, 236)
            or value['predictions_sealed_before_key'] is not True
            or value['all_fixed_ehr_opacity_states'] != 'unknown'
            or value['post_hoc_development'] is not True
            or type(value['new_model_calls']) is not int or value['new_model_calls'] != 0
            or any(value[k] is not False for k in ('clinical_qualified', 'regeneration_authorized',
                'mechanical_target_is_clinical_fault', 'prompt_retrieval_is_finding_truth',
                'same_biovil_architecture_is_independent_evidence', 'historical_final_role_is_untouched_test',
                'threshold_fitting', 'method_selection_after_results', 'old_choices_or_scores_changed',
                'training_allowed', 'actual_regeneration_executed'))):
        raise ValueError('fixed_nonclinical_ablation_protocol_required')


def source_cache(sources):
    meta = json.loads(b.checked(SOURCE / 'manifest.json', SOURCE_SHA, 1024**2, sources))
    if (meta['schema_version'] != conditioning.SCHEMA + '-manifest'
            or meta['selection_changed'] is not False or meta['clinical_qualified'] is not False):
        raise ValueError('completed_unchanged_synthetic_conditioning_sidecar_required')
    def get(name, cap):
        return b.checked(SOURCE / name, meta['artifacts'][name], cap, sources)
    rows = list(csv.DictReader(io.StringIO(get('image_conditioning_table.csv', 1024**2))))
    pairs = list(csv.DictReader(io.StringIO(get('within_generator_pair_cosines.csv', 16*1024**2))))
    texts = json.loads(get('text_encoding_receipts.json', 1024**2))
    summary = json.loads(get('summary.json', 1024**2))
    if (summary['image_slot_status_counts'] != {'scored_complete_inventory': 240}
            or summary['generation_calls'] != 0 or summary['training_calls'] != 0
            or summary['real_data_read'] is not False or summary['selection_changed'] is not False
            or summary['regeneration_authorized'] is not False):
        raise ValueError('all_completed_unqualified_conditioning_scores_required')
    # Manifest pins for body/pixel/weight files are not reopened on the CPU.
    return rows, pairs, texts


def bind_conditioning(rows, pairs, texts, bank, *, full=True):
    expected = {}
    for grid in bank.values():
        for source in grid.values():
            view = b.policy_module.snapshot(source); line = view['lineage']
            projection = {'case_id': view['case_id'], **{k: str(line[k]) for k in (
                'cxr_candidate_id', 'cxr_model_id', 'cxr_seed', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')}}
            if expected.setdefault(line['cxr_candidate_id'], projection) != projection:
                raise ValueError('unchanged_shared_image_anchor_required')
    frames, anchors = {}, {}
    for row in rows:
        key = row['cxr_candidate_id']
        if key in frames or key not in expected or any(row[k] != v for k, v in expected[key].items()):
            raise ValueError('exact_unique_conditioning_image_ehr_binding_required')
        group = row['prompt_group_id']
        anchor_key = row['case_id'], row['cxr_model_id']
        if anchor_key in anchors:
            raise ValueError('one_fixed_prompt_per_case_generator_required')
        anchors[anchor_key] = row
        frames[key] = row
    if set(frames) != set(expected) or full and len(frames) != 240:
        raise ValueError('all_original_image_slots_required')
    encodings = {r['prompt_sha256']: r for r in texts['records']}
    if len(encodings) != len(texts['records']):
        raise ValueError('unique_text_encoding_receipts_required')
    groups = defaultdict(set)
    for row in rows:
        encoded = encodings[row['prompt_sha256']]
        if encoded['status'] != 'encoded' or row['prompt_group_id'] != 'tokens_' + encoded['token_sequence_sha256']:
            raise ValueError('actual_critic_token_equivalence_group_required')
        groups[row['cxr_model_id']].add(row['prompt_group_id'])
    scores = defaultdict(dict)
    for row in pairs:
        frame = frames[row['cxr_candidate_id']]
        if any(row[k] != frame[k] for k in conditioning.IMAGE_FIELDS):
            raise ValueError('pair_matrix_image_lineage_changed')
        group = row['prompt_group_id']
        if group not in groups[frame['cxr_model_id']] or group in scores[frame['cxr_candidate_id']]:
            raise ValueError('complete_unique_within_generator_matrix_required')
        if (row['clinical_negative_validated'] != 'False'
                or row['is_intended_encoder_prompt_group'] != str(group == frame['prompt_group_id'])):
            raise ValueError('retrieval_not_clinical_negatives_required')
        value = float(row['raw_cosine']) if row['raw_cosine'] != '' else None
        if value is not None and (not math.isfinite(value) or abs(value) > 1.00001):
            raise ValueError('finite_cached_cosine_required')
        scores[frame['cxr_candidate_id']][group] = value
    if set(scores) != set(frames) or any(set(scores[k]) != groups[f['cxr_model_id']] for k, f in frames.items()):
        raise ValueError('every_image_every_within_generator_prompt_group_required')
    return frames, anchors, scores


def recipient_conditioning(case, displayed_id, frames, anchors, scores):
    current = frames[displayed_id]
    # Critically this is the RECIPIENT prompt, not the donor's cached match.
    original = anchors[case, current['cxr_model_id']]
    values = conditioning.retrieval(scores[displayed_id], original['prompt_group_id'])
    return {k: values[k] for k in CONDITION_FIELDS}, original


def validate_packet(packet):
    if not isinstance(packet, dict) or set(packet) != {'item_id', 'ehr_opacity_state',
            'report_opacity_state', 'image_evidence', 'conditioning'}:
        raise ValueError('opaque_scores_only_packet_required')
    image.validate_packet({k: v for k, v in packet.items() if k != 'conditioning'})
    value = packet['conditioning']
    if not isinstance(value, dict) or set(value) != set(CONDITION_FIELDS):
        raise ValueError('numeric_conditioning_packet_required')
    if type(value['prompt_groups']) is not int or value['prompt_groups'] < 1:
        raise ValueError('explicit_group_inventory_required')
    for key, limit in (('matched_cosine', 1.00001), ('matched_minus_other_mean', 2.00002)):
        v = value[key]
        if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or abs(v) > limit):
            raise ValueError('finite_conditioning_or_explicit_missing_required')
    if value['status'] == 'scored_complete_inventory':
        a, z = value['best_rank'], value['worst_rank']
        if (value['matched_cosine'] is None or value['matched_minus_other_mean'] is None
                or type(a) is not int or type(z) is not int or not 1 <= a <= z <= value['prompt_groups']):
            raise ValueError('complete_scores_and_valid_rank_interval_required')
    elif value['status'] in ('incomplete_prompt_inventory_no_rank', 'single_prompt_group_not_discriminating'):
        if any(value[k] is not None for k in ('matched_minus_other_mean', 'best_rank', 'worst_rank')):
            raise ValueError('missing_conditioning_is_not_a_negative')
    else:
        raise ValueError('explicit_conditioning_status_required')


def conditioning_preference(value, method):
    if value['status'] != 'scored_complete_inventory':
        return 'unavailable'
    if method in ('conditioning_margin_only', 'joint_margin0', 'report_blame_on_opposition'):
        margin = value['matched_minus_other_mean']
        return 'high' if margin > 0 else 'low' if margin < 0 else 'tie'
    cutoff = 1 if method == 'joint_rank1' else 5
    return 'high' if value['worst_rank'] <= cutoff else 'low' if value['best_rank'] > cutoff else 'boundary_tie'


def predict(packet, policy, method):
    validate_packet(packet)
    if method not in METHODS:
        raise ValueError('fixed_ablation_method_required')
    old = image.judge({k: v for k, v in packet.items() if k != 'conditioning'}, policy)
    preference = conditioning_preference(packet['conditioning'], method)
    target, reason = None, 'no_target_authorized_by_proxy_support_or_missingness'
    if method == 'report_blame_on_opposition':
        target = old['unsafe_report_blame_shortcut']
        reason = 'naive_mismatch_report_blame_control'
    elif method == 'conditioning_margin_only':
        target = 'cxr' if preference == 'low' else None
        reason = 'conditioning_only_nonclinical_cxr_hint' if target else 'conditioning_only_no_localization_signal'
    elif old['proxy_relation'] == 'proxy_opposition':
        target = 'report' if preference == 'high' else 'cxr' if preference == 'low' else None
        reason = 'joint_nonclinical_conditioning_hint' if target else 'conditioning_missing_or_tied_abstention'
    return {'item_id': packet['item_id'], 'image_policy': policy, 'method': method,
        'image_proxy_relation': old['proxy_relation'], 'image_status': old['image_status'],
        'conditioning_preference': preference, **{'conditioning_' + k: v for k, v in packet['conditioning'].items()},
        'tentative_target': target, 'reason': reason, 'confirmed_faulty_modality': None,
        'clinical_localization_accuracy': None, 'clinical_qualified': False,
        'regeneration_authorized': False, 'prompt_matching_is_finding_truth': False,
        'same_biovil_evidence_is_independent': False, 'ehr_opacity_state': 'unknown'}


def packets_and_bindings(resolver, images, reports, frames, anchors, scores):
    base = image.item_packets(resolver, images, reports)
    lookup = {r['item_id']: r for r in resolver}
    packets, bindings = [], []
    for packet in base:
        row = lookup[packet['item_id']]
        values, original = recipient_conditioning(row['case_id'], row['displayed_cxr_candidate_id'], frames, anchors, scores)
        if original['ehr_sha256'] != row['ehr_sha256']:
            raise ValueError('recipient_ehr_must_not_change_for_donor_image')
        current = frames[row['displayed_cxr_candidate_id']]
        if current['cxr_sha256'] != row['displayed_cxr_sha256']:
            raise ValueError('actual_displayed_image_matrix_required')
        packets.append({**packet, 'conditioning': values})
        validate_packet(packets[-1])
        bindings.append({'item_id': row['item_id'], 'case_id': row['case_id'], 'ehr_sha256': row['ehr_sha256'],
            'recipient_prompt_sha256': original['prompt_sha256'], 'recipient_prompt_group_id': original['prompt_group_id'],
            'displayed_cxr_candidate_id': current['cxr_candidate_id'], 'displayed_cxr_sha256': current['cxr_sha256'],
            'displayed_image_original_prompt_group_id': current['prompt_group_id'],
            'displayed_report_sha256': row['displayed_report_sha256'],
            **{'conditioning_' + k: v for k, v in values.items()}})
    return packets, bindings


def summarize(outcomes, resolver, key):
    truth = {r['item_id']: r for r in key}
    roles = {r['item_id']: r['benchmark_split'] for r in resolver}
    groups = defaultdict(list)
    for row in outcomes:
        for role in ('overall', roles[row['item_id']]):
            groups[row['image_policy'], row['method'], role, truth[row['item_id']]['intervention_type']].append(row)
    table = []
    for (policy, method, role, arm), rows in sorted(groups.items()):
        flags = [r for r in rows if r['tentative_target'] is not None]
        match = sum(r['tentative_target'] == truth[r['item_id']]['intervention_target'] for r in flags)
        table.append({'image_policy': policy, 'method': method, 'source_role': role, 'mechanical_arm': arm,
            'available_items': len(rows), 'tentative_targets': len(flags),
            'tentative_cxr_targets': sum(r['tentative_target'] == 'cxr' for r in flags),
            'tentative_report_targets': sum(r['tentative_target'] == 'report' for r in flags),
            'abstentions_not_clean_judgments': len(rows) - len(flags),
            'matches_known_replacement_target': match,
            'disagrees_with_known_replacement_target': len(flags) - match,
            'target_recovery_over_all_available_intervened': match / len(rows) if arm != 'no_corruption' else None,
            'conditional_target_match': match / len(flags) if flags else None,
            'clinical_localization_accuracy': None, 'clinical_false_repair_rate': None,
            'regeneration_authorized': False})
    return table


def run(args):
    b.cpu_guard()
    protocol = json.loads(PROTOCOL.read_text()); validate_protocol(protocol)
    sources = {}
    bank, _ = b.load_primary(sources)
    xrv, xm = image.cache_records(image.XRV, 'tricompose-cached-exact-opacity-sidecar-v1-scores', sources)
    biovil, bm = image.cache_records(image.BIOVIL, 'tricompose-cached-opacity-biovil-evidence-v1', sources)
    if xm['original_selection_changed'] is not False or bm['source_manifest_sha256'] != image.XRV[1]:
        raise ValueError('same_unchanged_image_sidecars_required')
    images, reports = image.bind_image_evidence(bank, xrv, biovil)
    frames, anchors, scores = bind_conditioning(*source_cache(sources), bank)
    meta = json.loads(b.checked(evidence.BANK / 'manifest.json', evidence.BANK_SHA, 1024**2, sources))
    if meta['counts'] != protocol['available_arms'] or meta['source']['coverage']['unavailable'] != protocol['unavailable_arms']:
        raise ValueError('all_available_and_missing_intervention_arms_required')
    items = {}
    for name in ('blind_items', 'resolver'):
        text = b.checked(evidence.BANK / (name + '.jsonl'), meta['artifact_sha256'][name], 4*1024**2, sources)
        items[name] = [json.loads(line) for line in text.splitlines()]
    evidence.bind_packets(bank, items['resolver'])
    packets, bindings = packets_and_bindings(items['resolver'], images, reports, frames, anchors, scores)
    if (len(packets) != 236 or len({r['case_id'] for r in items['resolver']}) != 80
            or {r['item_id'] for r in packets} != {r['item_id'] for r in items['blind_items']}):
        raise ValueError('complete_fixed80_intervention_inventory_required')
    pins = {str(p): sha256_file(p) for p in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_conditioning_attribution_v1.py', Path(image.__file__), Path(evidence.__file__),
        Path(conditioning.__file__), Path(b.__file__), Path(b.policy_module.__file__), Path(reader.__file__),
        Path(evidence.interventions.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        write_private_text(temporary / 'conditioning_bindings.csv', b.csv_text(bindings))
        outcomes = [predict(packet, policy, method) for packet in packets for policy in reader.POLICIES for method in METHODS]
        sealed = write_private_text(temporary / 'blind_predictions.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in outcomes))
        seal = sha256_file(sealed)
        # Key is only used after all numeric-only predictions have been sealed.
        text = b.checked(evidence.BANK / 'intervention_key.jsonl', meta['artifact_sha256']['intervention_key'], 4*1024**2, sources)
        items['intervention_key'] = [json.loads(line) for line in text.splitlines()]
        evidence.interventions.validate_items(items)
        table = summarize(outcomes, items['resolver'], items['intervention_key'])
        write_private_text(temporary / 'mechanical_ablation_table.csv', b.csv_text(table))
        # A separate original-bank sidecar; no change to any old winner/score.
        candidates = []
        for grid in bank.values():
            for source in grid.values():
                view = b.policy_module.snapshot(source); line = view['lineage']; img = images[line['cxr_candidate_id']]
                values, original = recipient_conditioning(view['case_id'], line['cxr_candidate_id'], frames, anchors, scores)
                packet = {'item_id': view['candidate_id'], 'ehr_opacity_state': 'unknown',
                    'report_opacity_state': view['states']['chexbert']['lung_opacity'],
                    'image_evidence': {k: img[k] for k in ('xrv_score', 'margins')}, 'conditioning': values}
                for policy in reader.POLICIES:
                    row = {'case_id': view['case_id'], 'triple_candidate_id': view['candidate_id'], **line,
                        'image_policy': policy, 'prompt_sha256': original['prompt_sha256'],
                        **{'conditioning_' + k: v for k, v in values.items()}}
                    judgments = [predict(packet, policy, method) for method in METHODS]
                    row['image_proxy_relation'] = judgments[0]['image_proxy_relation']
                    row.update({r['method'] + '_tentative_target': r['tentative_target'] for r in judgments})
                    row.update(clinical_localization_accuracy=None, regeneration_authorized=False, selector_used=False)
                    candidates.append(row)
        write_private_text(temporary / 'candidate_joint_evidence_table.csv', b.csv_text(candidates))
        lookup = {r['item_id']: r for r in items['intervention_key']}
        alias = Counter((lookup[r['item_id']]['intervention_type'],
            r['recipient_prompt_group_id'] == r['displayed_image_original_prompt_group_id']) for r in bindings)
        summary = {'schema_version': VERSION, 'status': 'cached_conditioning_attribution_diagnostic_complete',
            'available_items': 236, 'missing_arms': protocol['unavailable_arms'], 'fixed_ehr_cases': 80,
            'candidate_joint_evidence_rows': len(candidates), 'prediction_rows': len(outcomes), 'ablation_rows': len(table),
            'conditioning_matrix_reference': SOURCE_SHA, 'predictions_sha256_before_key': seal,
            'recipient_vs_displayed_original_prompt_group_counts': [
                {'mechanical_arm': arm, 'same_critic_token_group': same, 'items': count}
                for (arm, same), count in sorted(alias.items())],
            'source_roles_are_untouched_test': False, 'new_model_calls': 0,
            'body_pixel_or_weight_reads': False, 'old_choices_or_scores_changed': False,
            'conditioning_secondary_score_is_predictor_input': True,
            'same_score_cannot_be_independent_success_endpoint': True,
            'clinical_qualified': False, 'clinical_localization_accuracy': None,
            'clinical_false_repair_rate': None, 'regeneration_authorized': False,
            'actual_regeneration_executed': False, 'thresholds_fitted': False}
        write_private_json(temporary / 'summary.json', summary)
        if sha256_file(sealed) != seal or any(sha256_file(p) != pin for p, pin in sources.items()):
            raise ValueError('sealed_predictions_or_numeric_sources_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION + '-manifest',
            'sources': sources, 'code_pins': pins, 'clinical_qualified': False,
            'original_selection_changed': False, 'predictions_sha256_before_key': seal,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir() if p.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'conditioning_attribution'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target = run(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 1
    print(json.dumps({'status': 'cached_conditioning_attribution_diagnostic_complete',
        'manifest_sha256': sha256_file(target / 'manifest.json'), 'new_model_calls': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

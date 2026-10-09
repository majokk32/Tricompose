#!/usr/bin/env python3
"""CPU metadata diagnostic only; never decode clinical text or load a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ('TriCompose-v1.2/src', 'TriCompose-v1.1/src', 'TriCompose-v1.0/src',
                 'src', 'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(ROOT.parent / relative))
sys.path.insert(0, str(ROOT / 'tools'))
from score_free_random_control import cpu_guard
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, read_json,
    write_private_json, write_private_text, new_atomic_run, commit_atomic_run, discard_atomic_run)
from tricompose_v11.cxr_contracts import canonical_json_sha256
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt
from tricompose_v12.live_workers import check_pins

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SCHEMA = 'tricompose-online-conflict-metadata-diagnostic-v1'
SOURCES = {
    'plan': ('online_report_plans/online2_12654973_001', 'fca888e160ce692eaf100aa5cddf3aabd504c283a1b17e35ce6bf64c628dafac'),
    'run': ('online_report_runs/online2_12657042', 'aaf6da59ce052de0722e2e1c47e4813f8ad773a95153442803816bff79f8c2fb'),
    'audit': ('online_report_audits/online2_12657042', '0fce53d6d3faff1d74f8b7f14d2c2d2d93ef0c93e5c06b586a4b08b0387270af'),
    'parent_plan': ('bounded_regeneration_plans/retry2_12625457_001', '5a4e3bfbfd7e87879e14ea545932e3fef6331f3440bb4ef2973e52355fec8542'),
    'reliability': ('ehr_cxr_reliability_audits/retry2_reliability_12625457_001', '583fbae662f9ff98c4b9bd43de192127748aa11f29b810ce905018d902f9ca15'),
    'scorecard': ('real_validation/evaluator_scorecards/scorecard50_12645021_001', '91a2899919a14cb03e6f43ac0bc58f37bb0290e553628831b80da689ec2833f1'),
}
BODY_FILES = {'synthetic_ehr.json', 'ehr_facts.json', 'tokenizer_trace.json', 'tokenizer_input.txt'}


def metadata(path, pins, expected=None):
    path = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if path.name in BODY_FILES or path.suffix not in ('.json', '.csv') or path.stat().st_size > 4 * 1024**2:
        raise ValueError('bounded_metadata_only_required')
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError('metadata_source_hash_changed')
    pins[str(path)] = actual
    return read_json(path) if path.suffix == '.json' else list(csv.DictReader(io.StringIO(path.read_text())))


def hash_only(path, expected, pins):
    path = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if sha256_file(path) != expected:
        raise ValueError('unchanged_fixed_input_hash_required')
    pins[str(path)] = expected


def bundle(label, pins):
    relative, wanted = SOURCES[label]; root = BASE / relative
    return root, metadata(root / 'manifest.json', pins, wanted)


def transfer(case, original, candidate, prompt_manifest):
    request = case['requests'][0]['request']; final = request['inputs']['final_prompt']
    if ({k: v for k, v in request.items() if k not in ('seed', 'request_id')} !=
            {k: v for k, v in original.items() if k not in ('seed', 'request_id')}):
        raise ValueError('only_seed_and_request_id_may_change')
    anchor = anchor_from_record(case['anchor'])
    known = [(n, s, categories) for n, s, categories in anchor.findings if s in ('positive', 'negative')]
    if known != [('pneumonia', 'positive', ('diagnosis',))]:
        raise ValueError('same_diagnosis_intent_scope_required')
    for key, wanted in {'case_id': anchor.case_id, 'model_id': 'roentgen_v2', 'seed': 2,
            'frozen_model': True, 'adapter_added_prefix': False,
            'input_request_sha256': canonical_json_sha256(request),
            'ehr_sha256': anchor.ehr_sha256, 'ehr_facts_sha256': anchor.ehr_facts_sha256,
            'clinical_intent_sha256': final['clinical_intent_sha256'], 'prompt_sha256': final['sha256']}.items():
        if candidate.get(key) != wanted:
            raise ValueError('actual_generator_transport_binding_changed')
    model = prompt_manifest['models']['roentgen_v2']
    if (prompt_manifest['case_id'] != anchor.case_id
            or prompt_manifest['ehr_facts_sha256'] != anchor.ehr_facts_sha256
            or prompt_manifest['one_shared_clinical_intent'] is not True
            or prompt_manifest['adapter_must_not_add_prefix'] is not True):
        raise ValueError('shared_prompt_metadata_changed')
    for key in ('prompt_sha256', 'clinical_intent_sha256', 'included_direct_fact_ids', 'renderer_version',
                'available_context_ids', 'included_context_ids', 'omitted_context_ids'):
        wanted = final['sha256'] if key == 'prompt_sha256' else final[key]
        if model[key] != wanted:
            raise ValueError('declared_prompt_metadata_differs_from_frozen_request')
    if ('pneumonia' not in final['included_direct_fact_ids']
            or 'pneumonia_to_airspace_opacity_prior_v1' not in model['derived_rule_ids']):
        raise ValueError('declared_diagnosis_and_renderer_prior_required')
    observed = candidate['tokenizer_input']; count = candidate['cost']['prompt_token_count']
    if observed['runtime_observed'] is not True or type(count) is not int or not 0 < count <= 77:
        raise ValueError('runtime_tokenizer_attestation_and_length_guard_required')
    exact = observed['sha256'] == final['sha256']
    if observed['pipeline_changed_text'] != (not exact):
        raise ValueError('tokenizer_text_change_attestation_inconsistent')
    return {'opaque_source_index': case['opaque_source_index'], 'ehr_and_facts_hashes_bound': True,
        'seed_only_change': True, 'declared_pneumonia_intent_included': True,
        'derived_image_text_prior_marked': True, 'adapter_added_prefix': False,
        'runtime_tokenizer_observed': True, 'tokenizer_bytes_equal_supplied_prompt': exact,
        'unpadded_token_count': count, 'rejection_limit': 77,
        'tokenizer_boundary_not_encoder_hook': True,
        'included_context_count': len(final['included_context_ids']),
        'omitted_context_count': len(final['omitted_context_ids']),
        'extraction_semantics_independently_checked': False, 'clinical_image_truth_available': False}


def run(args):
    cpu_guard()  # Before any protected source read, output or model-related work.
    started = time.monotonic(); pins = {}
    loaded = {name: bundle(name, pins) for name in SOURCES}
    root, manifest = loaded['plan']; plan = metadata(root / 'plan.json', pins, manifest['plan_sha256'])
    check_pins(plan['source_pins'])  # Known source files only, never weights.
    pins.update(plan['source_pins'])
    root, manifest = loaded['parent_plan']
    parent = metadata(root / 'plan.json', pins, manifest['plan_sha256'])
    originals = {c['case_id']: c['original_request'] for c in parent['cases']}
    root, manifest = loaded['run']
    for filename, entry in manifest['artifacts'].items():
        hash_only(root / filename, entry['sha256'], pins)
    triples = metadata(root / 'completed_triplets.json', pins)['records']
    rows = metadata(root / 'score_rows.json', pins)['records']
    selections = metadata(root / 'selection.json', pins)
    audit_root, audit_manifest = loaded['audit']
    audit = metadata(audit_root / 'audit.json', pins, audit_manifest['audit_sha256'])
    if (audit['status'] != 'metadata_audit_passed_not_clinical'
            or audit['source_manifest_sha256'] != SOURCES['run'][1] or audit['recomputed_completed_receipts'] != 4
            or len(plan['cases']) != 2 or len(triples) != 4 or len(rows) != 4):
        raise ValueError('exact_audited_two_case_scope_required')
    old_root, old_manifest = loaded['reliability']
    old = metadata(old_root / 'audit.json', pins, old_manifest['artifacts']['audit.json']['sha256'])
    scores = metadata(old_root / 'synthetic_score_comparison.csv', pins,
        old_manifest['artifacts']['synthetic_score_comparison.csv']['sha256'])
    weak = metadata(old_root / 'weak_reference_head_metrics.csv', pins,
        old_manifest['artifacts']['weak_reference_head_metrics.csv']['sha256'])
    if (old['xrv_checkpoint_sha256'] != plan['workers']['xrv']['checkpoint_sha256']
            or old['threshold_sha256'] != plan['workers']['xrv']['thresholds_sha256']
            or old['independent_image_ground_truth'] is not False):
        raise ValueError('same_frozen_weak_reference_evaluator_required')
    transports = []; state_counts = {'pneumonia': Counter(), 'lung_opacity': Counter()}
    for case in plan['cases']:
        request = case['requests'][0]['request']; inputs = request['inputs']
        for value in inputs.values(): hash_only(value['path'], value['sha256'], pins)
        pm = metadata(Path(inputs['final_prompt']['path']).parent / 'prompt_manifest.json', pins)
        triple = next(t for t in triples if t['case_id'] == case['case_id'] and t['report_model_id'] == 'cxrmate_single')
        image_root = Path(triple['cxr_run']); cm = metadata(image_root / 'manifest.json', pins)
        if len(cm['candidates']) != 1: raise ValueError('one_new_image_per_ehr_required')
        item = cm['candidates'][0]
        candidate = metadata(image_root / item['path'], pins, item['sha256'])
        transports.append(transfer(case, originals[case['case_id']], candidate, pm))
        observation = candidate['tokenizer_input']
        hash_only(observation['path'], observation['sha256'], pins)
        hash_only(observation['trace_path'], observation['trace_sha256'], pins)  # bytes only
        label_path = root / 'cases' / case['case_id'] / 'operations/xrv_0_a1/scored/cxr_finding_labels.json'
        label_hash = next(r['receipt']['xrv_labels_sha256'] for r in rows if r['case_id'] == case['case_id'])
        labels = metadata(label_path, pins, label_hash)
        image_receipt(anchor_from_record(case['anchor']), candidate, labels,
            label_sha256=label_hash, thresholds_sha256=plan['workers']['xrv']['thresholds_sha256'],
            checkpoint_sha256=plan['workers']['xrv']['checkpoint_sha256'])
        vector = labels['records'][0]
        value = vector['finding_probabilities']['pneumonia']; threshold = labels['thresholds']['pneumonia']
        scores.append({'opaque_source_index': case['opaque_source_index'], 'finding': 'pneumonia',
            'ehr_state': 'positive', 'source_categories': 'diagnosis', 'image_seed': 2,
            'xrv_op_norm_score': value, 'negative_max': threshold['negative_max'], 'positive_min': threshold['positive_min'],
            'xrv_state': vector['finding_states']['pneumonia'], 'score_minus_positive_min': value - threshold['positive_min'],
            'historical_default_0_5_state': 'positive' if value >= .5 else 'negative'})
        for name in state_counts: state_counts[name][vector['finding_states'][name]] += 1
    if len(scores) != 6 or {(int(r['opaque_source_index']), int(r['image_seed'])) for r in scores} != {(c['opaque_source_index'], s) for c in plan['cases'] for s in (0, 1, 2)}:
        raise ValueError('complete_same_ehr_seed_history_required')
    card_root, card_manifest = loaded['scorecard']
    card = metadata(card_root / 'scorecard.csv', pins, card_manifest['artifacts']['scorecard.csv']['sha256'])
    report_states = {m: {n: dict(Counter(next(f['chexbert'] for f in r['receipt']['fact_states'] if f['finding'] == n)
        for r in rows if r['report_model_id'] == m)) for n in state_counts} for m in ('cxrmate_single', 'chexagent2')}
    weak_pneumonia = next(r for r in weak if r['finding'] == 'pneumonia')
    result = {'schema_version': SCHEMA, 'status': 'metadata_diagnosis_completed_unresolved_clinical_truth',
        'fixed_cases': 2, 'matched_seed_score_records': len(scores),
        'exact_tokenizer_text_cases': sum(r['tokenizer_bytes_equal_supplied_prompt'] for r in transports),
        'prompt_token_count_range': [min(r['unpadded_token_count'] for r in transports), max(r['unpadded_token_count'] for r in transports)],
        'new_image_state_counts': {k: dict(v) for k, v in state_counts.items()}, 'report_state_counts': report_states,
        'matched_score_range': [min(float(r['xrv_op_norm_score']) for r in scores), max(float(r['xrv_op_norm_score']) for r in scores)],
        'all_six_below_original_and_default_threshold': all(float(r['score_minus_positive_min']) < 0 and r['historical_default_0_5_state'] == 'negative' for r in scores),
        'weak_reference_pneumonia': weak_pneumonia, 'separate_public_proxy_scorecard': card,
        'static_veto_reason_counts': dict(Counter(reason for c in selections['cases'] for reason in c['static_decision']['rejection_reason_codes'])),
        'diagnosis_is_generation_intent_not_adjudicated_image_truth': True,
        'opacity_is_supplemental_renderer_prior_not_new_ehr_fact': True,
        'semantic_extraction_re_adjudicated': False, 'weights_reopened': False,
        'global_prompt_collapse_assessed': False, 'encoder_conditioning_use_proven': False,
        'clinical_fault_localization_established': False, 'new_model_calls': 0, 'new_slurm_submissions': 0,
        'training_or_threshold_fitting': False, 'source_bodies_or_pixels_parsed': False,
        'historical_scores_or_choices_changed': False, 'clinical_acceptance': False,
        'elapsed_cpu_seconds_before_serialization': round(time.monotonic() - started, 6)}
    for path in (Path(__file__), ROOT / 'tests/test_online_conflict_diagnostic.py', ROOT.parent / 'docs/online_conflict_diagnostic_protocol.md'):
        pins[str(path)] = sha256_file(path)
    check_pins(pins)
    temp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temp / 'diagnostic.json', result)
        write_private_json(temp / 'transport.json', {'records': transports})
        csv_out = io.StringIO(); writer = csv.DictWriter(csv_out, fieldnames=list(scores[0]))
        writer.writeheader(); writer.writerows(sorted(scores, key=lambda r: (int(r['opaque_source_index']), int(r['image_seed']))))
        write_private_text(temp / 'matched_seed_scores.csv', csv_out.getvalue())
        write_private_json(temp / 'manifest.json', {'schema_version': SCHEMA, 'status': result['status'],
            'sources': pins, 'new_model_calls': 0, 'clinical_acceptance': False,
            'artifacts': {name: {'sha256': sha256_file(temp / name)} for name in ('diagnostic.json', 'transport.json', 'matched_seed_scores.csv')}})
        commit_atomic_run(temp, target)
    except BaseException:
        discard_atomic_run(temp); raise
    return target, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True); parser.add_argument('--run-id', required=True)
    try:
        root, result = run(parser.parse_args())
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__})); return 1
    print(json.dumps({'status': result['status'], 'manifest_sha256': sha256_file(root / 'manifest.json'),
        'fixed_cases': 2, 'new_model_calls': 0})); return 0


if __name__ == '__main__':
    raise SystemExit(main())

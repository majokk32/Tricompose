"""Independent sealed-cache arithmetic only; no production scorer imports.

No real source reports, native annotations, model weights or inference are
opened here. Provenance of those bytes comes from the approved worker receipt.
This audit cannot qualify clinical correctness or vocabulary equivalence.
"""
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from contracts import new_atomic_run, commit_atomic_run, discard_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'manual_opacity_reader_runs/manual_dev_12788366_001'
EXPECTED = '61622a2986dec980a22063a47da23c54eb327346c33f460c5a1875feadcfcf34'
REFERENCE = BASE / 'manual_opacity_reference_inventories/released_dev_12787886_001'
REFERENCE_SHA = 'c698dc6c047090142633c886219d98064ed049428fe8bd77d110b4eb191a7dd5'
NO_REOPEN = {
    'artifacts/protected/tricompose_v1_2/reference_datasets/cxrgraph_source_12766754_001/source/manual_data/dev.json',
    '.cache/tricompose_report_eval/models/chexbert/chexbert.pth'}
STATES = ('positive', 'negative', 'uncertain', 'unknown')
HEADS = {'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion',
         'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax',
         'pleural_effusion', 'pleural_other', 'fracture', 'support_devices', 'no_finding'}


def replay(references, predictions):
    require(isinstance(references, list) and isinstance(predictions, list)
            and 0 < len(references) == len(predictions) <= 1024, 'complete_cache_inventory_required')
    require(len({p['report_id'] for p in predictions}) == len(predictions), 'unique_cache_prediction_required')
    indexed = {p['report_id']: p for p in predictions}
    pairs, known, explicit, details = [], [], [], []
    for i, ref in enumerate(references):
        key = f'report_{i:04d}'
        require(ref['report_id'] == key and ref['source_index'] == i and key in indexed,
                'complete_opaque_release_slots_required')
        pred = indexed[key]
        require(re.fullmatch(r'[a-f0-9]{64}', ref['source_sha256'])
                and (pred['source_sha256'] == ref['source_sha256'] or
                     pred['status'] == 'failed_unavailable' and pred['source_sha256'] is None),
                'sealed_same_source_slot_required')
        if ref['status'] == 'complete':
            projection = ref['projection']
            require(projection['source_sha256'] == ref['source_sha256']
                and projection['manual_literal_state'] in STATES
                and projection['current_lung_opacity_reference'] is None
                and projection['clinical_qualified'] is False, 'literal_reference_not_global_gold_required')
            truth = projection['manual_literal_state']
        else:
            require(ref['status'] == 'failed_unavailable' and ref['projection'] is None
                    and ref['failure_type'] is not None, 'failed_reference_not_unknown_required')
            truth = 'reference_unavailable'
        if pred['status'] == 'complete':
            labels = pred['finding_states']
            require(isinstance(labels, dict) and set(labels) == HEADS and
                all(value in STATES for value in labels.values()) and
                labels['no_finding'] in ('unknown', 'positive') and pred['failure_type'] is None,
                'fourteen_valid_official_heads_required')
            decision = labels['lung_opacity']
        else:
            require(pred['status'] == 'failed_unavailable' and pred['finding_states'] is None
                    and pred['failure_type'] is not None, 'failed_prediction_not_unknown_required')
            decision = 'prediction_unavailable'
        pair = (truth, decision)
        pairs.append(pair)
        if truth in STATES[:3]:
            known.append(pair)
        if truth in STATES[:2]:
            explicit.append(pair)
        details.append({'report_id': key, 'source_sha256': ref['source_sha256'],
            'reference_status': ref['status'], 'prediction_status': pred['status'],
            'reference_literal_state': truth if truth in STATES else None,
            'prediction_lung_opacity_state': decision if decision in STATES else None})
    counts = Counter(pairs)
    matrix = {a: {b: counts[a, b] for b in (*STATES, 'prediction_unavailable')}
              for a in (*STATES, 'reference_unavailable')}
    per_state = {}
    for state in STATES:
        subset = [pair for pair in pairs if pair[0] == state]
        completed = [pair for pair in subset if pair[1] in STATES]
        matches = sum(a == b for a, b in subset)
        per_state[state] = {'reference_support_all_attempted': len(subset),
            'prediction_available': len(completed), 'prediction_unavailable': len(subset) - len(completed),
            'literal_state_matches': matches,
            'match_fraction_all_reference_supported': matches / len(subset) if subset else None,
            'match_fraction_prediction_available': matches / len(completed) if completed else None}
    known_complete = [p for p in known if p[1] in STATES]
    explicit_complete = [p for p in explicit if p[1] in STATES]
    known_matches = sum(a == b for a, b in known)
    flips = sum(a != b and b in STATES[:2] for a, b in explicit)
    ref_available = sum(a in STATES for a, _ in pairs)
    numeric = {'attempted_reports': len(pairs), 'reference_available': ref_available,
        'reference_unavailable': len(pairs) - ref_available,
        'jointly_available': sum(a in STATES and b in STATES for a, b in pairs),
        'prediction_status_counts': dict(Counter(p['status'] for p in predictions)),
        'confusion_matrix_all_attempted': matrix, 'known_literal_reference_support': len(known),
        'known_literal_prediction_available': len(known_complete), 'known_literal_state_matches': known_matches,
        'known_literal_match_fraction_all_supported': known_matches / len(known) if known else None,
        'known_literal_match_fraction_prediction_available': known_matches / len(known_complete) if known_complete else None,
        'explicit_literal_reference_support': len(explicit),
        'explicit_literal_prediction_available': len(explicit_complete), 'known_literal_polarity_flips': flips,
        'known_literal_polarity_flip_fraction_prediction_available': flips / len(explicit_complete) if explicit_complete else None,
        'determinate_predictions_on_literal_unknown': sum(a == 'unknown' and b in STATES[:2] for a, b in pairs)}
    return numeric, per_state, details


def check_evaluation(evaluation, predictions, references, comparison_records):
    numeric, states, details = replay(references, predictions)
    require(all(evaluation.get(k) == v for k, v in numeric.items()), 'independent_full_denominator_count_mismatch')
    require(all(all(evaluation['per_reference_state'][s][k] == v for k, v in row.items())
                for s, row in states.items()), 'independent_per_state_denominator_mismatch')
    require(len(comparison_records) == len(details) and all(
        all(row[k] == value for k, value in expected.items()) and row['clinical_error_adjudicated'] is False
        for row, expected in zip(comparison_records, details)), 'comparison_slots_or_clinical_promotion_mismatch')
    for flag in ('clinical_qualified', 'primary_metric_eligible', 'selection_changed', 'regeneration_authorized',
                 'untouched_final_test', 'thresholds_fitted', 'vocabulary_and_scope_equivalence_verified',
                 'independent_image_truth', 'literal_unknown_determinate_is_clinical_hallucination'):
        require(evaluation[flag] is False, 'diagnostic_cannot_authorize_clinical_or_repair_claim')
    return numeric


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED
            and sha256(REFERENCE / 'manifest.json') == REFERENCE_SHA, 'sealed_completed_reader_and_reference_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    reference_manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'sealed_reader_artifact_changed')
    require(sha256(REFERENCE / 'reference_inventory.json') == reference_manifest['artifacts']['reference_inventory.json'],
            'sealed_cached_reference_required')
    plan = json.loads((RUN / 'frozen_plan.json').read_text())
    require(plan['pins'] == manifest['pins'] and plan['operator_blinded'] is False,
            'recorded_consumed_plan_and_nonblind_design_required')
    # Recheck only code, cached metadata and small public tokenizer assets.
    # DO NOT reopen raw dev reports or the heavy checkpoint.
    for name, value in manifest['pins'].items():
        if name not in NO_REOPEN:
            path = WORKSPACE / name
            require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value, 'consumed_code_or_cache_changed')
    predictions = json.loads((RUN / 'chexbert_predictions.json').read_text())['records']
    references = json.loads((REFERENCE / 'reference_inventory.json').read_text())['records']
    evaluation = json.loads((RUN / 'evaluation.json').read_text())
    details = json.loads((RUN / 'comparison_records.json').read_text())['records']
    numeric = check_evaluation(evaluation, predictions, references, details)
    summary = json.loads((RUN / 'summary.json').read_text())
    eligible = sum(type(p['input_token_count']) is int and 0 < p['input_token_count'] <= 512 for p in predictions)
    require(summary['attempted_reports'] == 75 and numeric['attempted_reports'] == 75
            and summary['encoder_examples'] == eligible <= 75
            and summary['forward_batches'] == math.ceil(eligible / 4)
            and summary['prediction_status_counts'] == numeric['prediction_status_counts']
            and summary['frozen_parameters'] is True and summary['device'] == 'cpu'
            and summary['retries'] == 0 and summary['truncation_used'] is False
            and summary['model_receives_annotation_labels'] is False
            and summary['predictions_fsynced_and_sealed_before_reference_decode'] is True,
            'recorded_call_budget_or_frozen_execution_mismatch')
    paths = [Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/tests/test_manual_opacity_reader_audit.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    temporary, target = new_atomic_run(BASE / 'manual_opacity_reader_audits', 'numeric_12784259_001')
    try:
        write_json(temporary / 'frozen_plan.json', {'schema_version': 'manual-opacity-reader-independent-audit-plan-v1',
            'pins': pins, 'completed_reader_manifest_sha256': EXPECTED, 'reference_manifest_sha256': REFERENCE_SHA,
            'production_evaluator_imported': False, 'raw_source_or_model_weights_reopened': False})
        result = {'schema_version': 'manual-opacity-reader-independent-audit-v1', 'status': 'passed',
            'verified_counts': numeric, 'reader_manifest_sha256': EXPECTED, 'reference_manifest_sha256': REFERENCE_SHA,
            'raw_source_or_model_weights_reopened': False, 'vocabulary_or_scope_equivalence_adjudicated': False,
            'clinical_qualified': False, 'new_model_calls': 0, 'selection_changed': False,
            'regeneration_authorized': False, 'runtime_seconds': time.monotonic() - started}
        write_json(temporary / 'audit.json', result)
        require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'audit_program_changed')
        write_json(temporary / 'manifest.json', {'schema_version': 'manual-opacity-reader-audit-receipt-v1',
            'pins': pins, 'reader_manifest_sha256': EXPECTED,
            'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())}, 'clinical_qualified': False})
        for p in [temporary, *temporary.rglob('*')]:
            require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                    and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_cached_audit_required')
        commit_atomic_run(temporary, target)
        return target, result
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_cache_audit_no_arguments_required')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS')
                and not os.environ.get('SLURM_STEP_GPUS'), 'existing_actual_cache_only_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_manual_opacity_reader_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_manual_opacity_reader_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

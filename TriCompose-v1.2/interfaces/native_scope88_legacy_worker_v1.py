#!/usr/bin/env python
"""Reuse sealed native inference on a new fixed authored-only scope88 plan."""
from __future__ import print_function

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'interfaces/chexpert_negbio_legacy_worker_v1.py'
spec = importlib.util.spec_from_file_location('_unchanged_native_worker', str(path))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
SCHEMA = 'tricompose-native-context-scope88-v1'
BASE = native.PROTECTED/'tricompose_v1_2'


def validate_inputs(inputs):
    if set(inputs) != {'records'} or not isinstance(inputs['records'], list) or len(inputs['records']) != 88:
        raise ValueError('fixed_authored88_inventory_required')
    for index, row in enumerate(inputs['records']):
        if set(row) != {'item_id', 'text', 'report_sha256'} or row['item_id'] != 'scope_' + str(index).zfill(4) or \
                not isinstance(row['text'], str) or not row['text'].strip() or len(row['text']) > 8192 or \
                native.hashlib.sha256(row['text'].encode()).hexdigest() != row['report_sha256']:
            raise ValueError('bounded_hash_bound_authored_only_input_required')
    if len({row['report_sha256'] for row in inputs['records']}) != 88:
        raise ValueError('distinct_authored88_inputs_required')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan-root', required=True, type=Path)
    cli.add_argument('--plan-manifest-sha256', required=True)
    cli.add_argument('--output-directory', required=True, type=Path)
    args = cli.parse_args()
    tmp = native.guard()
    if not native.inside(args.plan_root, BASE/'native_context_scope_plans') or \
            not native.inside(args.output_directory, BASE/'native_context_scope_runs') or \
            not args.output_directory.is_dir() or any(args.output_directory.iterdir()):
        raise ValueError('sealed_authored_plan_and_empty_private_output_required')
    manifest = native.metadata(args.plan_root/'manifest.json', args.plan_manifest_sha256)
    native.verify_pins(manifest['sources'])
    if manifest['sources'].get(str(Path(__file__))) != native.sha(Path(__file__)):
        raise ValueError('frozen_authored_scope_worker_required')
    plan = native.metadata(args.plan_root/'plan.json', manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA+'-plan' or plan['authored_count'] != 88 or \
            plan['policy']['clinical_qualified'] is not False or plan['deployment_manifest_sha256'] != native.DEPLOYMENT_SHA:
        raise ValueError('frozen_authored_scope_diagnostic_policy_required')
    deployment = native.metadata(native.DEPLOYMENT/'manifest.json', native.DEPLOYMENT_SHA)
    native.verify_pins(deployment['sources'])
    runtime = native.metadata(native.DEPLOYMENT/'runtime.json', deployment['artifacts']['runtime.json'])
    started = time.monotonic()
    components = native.initialize(runtime, tmp)
    initialized = time.monotonic()
    inputs = native.metadata(args.plan_root/'inputs.json', manifest['artifacts']['inputs.json'])
    validate_inputs(inputs)
    predictions, replays = [], []
    for item in inputs['records']:
        primary, replay = native.parse_item(item, components), native.parse_item(item, components)
        predictions.append(primary)
        replays.append({'item_id': item['item_id'], 'report_sha256': item['report_sha256'],
                        'same_full_evidence': primary == replay,
                        'both_complete': primary['output']['status'] == replay['output']['status'] == 'complete',
                        'replay_output': replay['output']})
    parsed = time.monotonic()
    native.verify_pins(manifest['sources'])
    native.verify_pins(deployment['sources'])
    native.private_json(args.output_directory/'native_predictions.json', {'records': predictions})
    native.private_json(args.output_directory/'native_replays.json', {'records': replays})
    native.private_json(args.output_directory/'native_freeze_receipt.json', {
        'sha256': {name: native.sha(args.output_directory/name) for name in ('native_predictions.json', 'native_replays.json')},
        'reference_semantics_decoded_before_prediction_fsync': False, 'reference_json_payload_read_by_worker': False,
        'source_provenance_bytes_hashed': True, 'authored_development_not_clinical_gold': True})
    native.private_json(args.output_directory/'native_runtime.json', {
        'schema_version': SCHEMA+'-native-runtime', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'environment': str(native.ENV), 'python': sys.version.split()[0],
        'initialization_seconds': round(initialized-started, 6), 'parsing_seconds': round(parsed-initialized, 6),
        'authored_texts': 88, 'parser_passes': 176,
        'native_parser_or_rules_changed': False, 'patient_or_candidate_inputs_read': False,
        'clinical_qualified': False, 'gpu_calls': 0, 'external_api': False,
        'selection_changed': False, 'regeneration_authorized': False})
    print(json.dumps({'status': 'native_scope88_predictions_frozen', 'authored_texts': 88,
                      'parser_passes': 176, 'parsing_seconds': round(parsed-initialized, 6)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'native_scope88_failed_closed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)

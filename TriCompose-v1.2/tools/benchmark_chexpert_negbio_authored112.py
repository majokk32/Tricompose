#!/usr/bin/env python3
"""Prospective authored-only parsing diagnostic, not clinical qualification."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'),
               str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, sha256_file, require_inside, new_atomic_run,
                       write_private_json, write_private_text, commit_atomic_run, discard_atomic_run)
from tricompose_v12 import chexpert_negbio_contract as contract
from tricompose_v12.opacity_context import metrics, TARGET_PATTERN
import score_cached_opacity_candidates as guard

BASE = PROTECTED_ROOT/'tricompose_v1_2'
DEPLOYMENT = BASE/'chexpert_negbio_deployments/deployment_12682821_001'
DEPLOYMENT_SHA = 'f8c8d9a02455780de22e913ce7a1c056450c5e75eafffea6279d869583d9ef85'
OLD_PLAN = BASE/'opacity_assertion_stages_plans/authored48_stages_v3_12666569_001'
OLD_PLAN_SHA = '238a8e44133c2edc9c69d460a49fcbfbb5e62ef38ed64672699a808eaa088551'
INPUT_PLAN = BASE/'opacity_context_plans/context112_12666569_001'
INPUT_PLAN_SHA = '380f452755911a9d8d0393a649e956dbb4af7831d9e6e8467c898c7c08f8f7e0'
ENV = WORKSPACE/'runtime/venvs/chexpert-negbio-py36-12682821-v1'
WORKER = ROOT/'interfaces/chexpert_negbio_legacy_worker_v1.py'
PROTOCOL = WORKSPACE/'docs/chexpert_negbio_authored112_protocol.md'
TEST = ROOT/'tests/test_chexpert_negbio_legacy_worker.py'
SCHEMA = 'tricompose-chexpert-negbio-authored112-v1'
POLICY = {'authored_only': True, 'old48': 48, 'new64': 64, 'replay_all': True,
          'selective_retry': False, 'native_classifier_unchanged': True,
          'official_sections_to_extract': [], 'official_extract_strict': False,
          'literal_head_ontology_matches_official': False,
          'native_labels_and_conflict_view_separate': True,
          'failures_are_unavailable_not_unknown': True, 'no_finding_expands_negatives': False,
          'current_scope_or_anatomy_verified': False,
          'contract_transform': 'remove_only_unused_future_annotations_import_in_memory',
          'literal_slice_pattern': TARGET_PATTERN, 'heldout_clinical_gold': False,
          'investigator_knows_prior_results': True, 'references_loaded_by_worker': False,
          'clinical_qualified': False, 'primary_metric_eligible': False,
          'selection_changed': False, 'regeneration_authorized': False,
          'patient_or_candidate_inputs_read': False, 'gpu_calls': 0,
          'training': False, 'external_api': False, 'new_slurm_submissions': 0}


def dump(path, value):
    write_private_json(path, value)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def private_permissions(root):
    for path in (root, *root.iterdir()):
        st = path.stat()
        if st.st_gid not in (96293, 65534) or st.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
            raise ValueError('protected_project_permissions_required')


def copy_references(source, expected, target, pins):
    source = require_inside(source, PROTECTED_ROOT, must_exist=True)
    if source.suffix != '.json' or not 0 < source.stat().st_size < 1024**2 or sha256_file(source) != expected:
        raise ValueError('sealed_authored_reference_file_required')
    # Copy bytes-as-text without using reference semantics during preparation.
    write_private_text(target, source.read_text(encoding='utf-8'))
    if sha256_file(target) != expected or sha256_file(source) != expected:
        raise ValueError('reference_copy_changed')
    pins[str(source)] = expected


def prepare(run_id):
    guard.guard()
    pins = {str(path): sha256_file(path) for path in (
        Path(__file__), WORKER, PROTOCOL, TEST, Path(contract.__file__),
        ROOT/'src/tricompose_v12/opacity_context.py', ROOT/'src/tricompose_v12/report_assertions.py',
        ROOT/'tools/score_cached_opacity_candidates.py',
        WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py')}
    deployment = guard.metadata(DEPLOYMENT/'manifest.json', pins, DEPLOYMENT_SHA)
    guard.verify_pins(deployment['sources'])
    pins.update(deployment['sources'])
    guard.metadata(DEPLOYMENT/'runtime.json', pins, deployment['artifacts']['runtime.json'])
    prior = guard.metadata(INPUT_PLAN/'manifest.json', pins, INPUT_PLAN_SHA)
    inputs = guard.metadata(INPUT_PLAN/'inputs.json', pins, prior['artifacts']['inputs.json'])
    # Reuse the worker's validation only, without importing/initializing models.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_legacy_worker_validation', WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_inputs(inputs)
    old = guard.metadata(OLD_PLAN/'manifest.json', pins, OLD_PLAN_SHA)
    temporary, target = new_atomic_run(BASE/'chexpert_negbio_authored_plans', run_id)
    try:
        dump(temporary/'inputs.json', inputs)
        copy_references(OLD_PLAN/'references.json', old['artifacts']['references.json'],
                        temporary/'references_old48.json', pins)
        copy_references(INPUT_PLAN/'references_new64.json', prior['artifacts']['references_new64.json'],
                        temporary/'references_new64.json', pins)
        dump(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'policy': POLICY,
            'deployment_manifest_sha256': DEPLOYMENT_SHA, 'input_plan_manifest_sha256': INPUT_PLAN_SHA,
            'inputs_parsed_during_preparation': 0, 'references_semantics_used_for_preparation': False})
        guard.verify_pins(pins)
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())}})
        private_permissions(temporary)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'authored112_plan_frozen', 'authored_texts': 112,
                    'manifest_sha256': sha256_file(target/'manifest.json')}


def diagnostic_rows(references, predictions, inputs, view):
    if view not in ('native', 'mention_conflict') or not len(references) == len(predictions) == len(inputs):
        raise ValueError('exact_authored_reference_join_required')
    result = []
    for ref, pred, item in zip(references, predictions, inputs):
        if any(ref[key] != pred[key] or ref[key] != item[key] for key in ('item_id', 'report_sha256')):
            raise ValueError('hash_bound_authored_join_required')
        output = pred['output']
        status = output['status']
        if status not in ('complete', 'failed_unavailable'):
            raise ValueError('explicit_availability_required')
        if status == 'complete':
            labels = output['native_labels'] if view == 'native' else output['mention_conflict_view']
            state = labels['lung_opacity']['state']
            if state not in contract.STATES:
                raise ValueError('four_state_native_or_conflict_output_required')
        else:
            if output['native_labels'] is not None or output['mention_conflict_view'] is not None:
                raise ValueError('unavailable_is_null_not_unknown_required')
            state = None
        result.append({'item_id': ref['item_id'], 'report_sha256': ref['report_sha256'],
            'family': ref['family'], 'expected_state': ref['expected_state'], 'status': status,
            'state': state, 'failure_reason': output['failure_reason'],
            'literal_target_present': re.search(TARGET_PATTERN, item['text'], re.I) is not None})
    return result


def markdown(summary):
    lines = ['# CheXpert/NegBio authored112 end-to-end / 端到端解析诊断', '',
        '已知开发用人工文本；不是 clinical gold 或最终报告准确性。', '',
        '| Set | View | Authored-task matches | Macro F1 | Unsafe certainty | Unavailable |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for dataset, views in summary['readouts'].items():
        for view, row in views.items():
            lines.append(f"| {dataset} | {view} | {row['exact_matches']}/{row['rows']} | "
                         f"{row['macro_f1_present_classes']:.5f} | {row['determinate_on_uncertain_unknown']} | {row['unavailable']} |")
    lines += ['', 'Native 官方结果与 TriCompose mention-conflict view 分开；后者不是另一个临床评分器。',
        '官方 Lung Opacity ontology 较宽；与 literal/current-finding 参考语义并不相同。', '',
        f"Replay differences: {summary['replay_changed']}/112; both complete: {summary['replays_both_complete']}/112.",
        f"Initialization seconds: {summary['runtime']['initialization_seconds']}; parsing seconds: {summary['runtime']['parsing_seconds']}.",
        'Predictions/replays fsynced before reference evaluation; no patient/candidate bodies.', '',
        '## Every family / 每个文本组', '',
        '| Set | Family | Native matches | Conflict matches | Native unsafe certainty |',
        '| --- | --- | ---: | ---: | ---: |']
    for dataset, families in summary['per_family'].items():
        for family, views in families.items():
            native, conflict = views['native'], views['mention_conflict']
            lines.append(f"| {dataset} | {family} | {native['exact_matches']}/{native['rows']} | "
                         f"{conflict['exact_matches']}/{conflict['rows']} | {native['determinate_on_uncertain_unknown']} |")
    lines += ['', '不可用为 unavailable/null；不是 unknown 或隐式 positive。',
        'No changes to old labels, scores, selected triples, EHRs, models, rules or gates.',
        'No clinical qualification, selector promotion or regeneration. No new sbatch/GPU/training/API.', '']
    return '\n'.join(lines)


def evaluate(plan_root, expected, run_id):
    guard.guard()
    pins = {}
    root = require_inside(plan_root, BASE/'chexpert_negbio_authored_plans', must_exist=True)
    manifest = guard.metadata(root/'manifest.json', pins, expected)
    guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan = guard.metadata(root/'plan.json', pins, manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA+'-plan' or plan['policy'] != POLICY:
        raise ValueError('prospectively_frozen_authored_policy_required')
    temporary, target = new_atomic_run(BASE/'chexpert_negbio_authored_runs', run_id)
    try:
        tmp = WORKSPACE/'.tmp'/f'chexpert_negbio_authored_{run_id}'
        tmp.mkdir(mode=0o2770, exist_ok=False)
        os.chmod(tmp, 0o2770)
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.update(TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [str(ENV/'bin/python'), str(WORKER), '--plan-root', str(root),
                   '--plan-manifest-sha256', expected, '--output-directory', str(temporary)]
        # Wrapper suppresses native output and never prints source-bearing errors.
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError('legacy_authored_parser_failed_closed')
        predictions = guard.metadata(temporary/'predictions.json', pins)['datasets']
        replays = guard.metadata(temporary/'replays.json', pins)['records']
        receipt = guard.metadata(temporary/'prediction_freeze_receipt.json', pins)
        if receipt['references_read_before_prediction_fsync'] is not False:
            raise ValueError('predictions_before_references_required')
        closed = receipt['sha256']
        if set(closed) != {'predictions.json', 'replays.json'} or \
                any(sha256_file(temporary/name) != digest for name, digest in closed.items()):
            raise ValueError('closed_predictions_and_replays_required')
        runtime = guard.metadata(temporary/'runtime.json', pins)
        inputs = guard.metadata(root/'inputs.json', pins, manifest['artifacts']['inputs.json'])
        # Only now load reference semantics. None are passed to the legacy worker.
        references = {dataset: guard.metadata(root/f'references_{dataset}.json', pins,
                    manifest['artifacts'][f'references_{dataset}.json'])['records']
                    for dataset in ('old48', 'new64')}
        if set(predictions) != {'old48', 'new64'} or len(replays) != 112:
            raise ValueError('all_attempted_authored_rows_required')
        checks = {}
        for dataset, size in (('old48', 48), ('new64', 64)):
            if any(len(table[dataset]) != size for table in (references, predictions, inputs)):
                raise ValueError('all_fixed_authored_rows_required')
            checks[dataset] = {view: diagnostic_rows(references[dataset], predictions[dataset], inputs[dataset], view)
                               for view in ('native', 'mention_conflict')}
        failure_counts = Counter(row['output']['failure_reason'] for rows in predictions.values()
                                 for row in rows if row['output']['status'] != 'complete')
        stages = {stage: {'attempted': 0, 'complete': 0, 'errors': 0} for stage in
                  ('load', 'extract', 'parse', 'dependency', 'syntax_gate', 'detector', 'aggregate', 'serialize')}
        for rows in predictions.values():
            for row in rows:
                for stage, counts in row['output']['stage_availability'].items():
                    for key in stages[stage]:
                        stages[stage][key] += counts[key]
        summary = {'schema_version': SCHEMA, 'policy': POLICY, 'runtime': runtime,
            'readouts': {d: {v: metrics(rows) for v, rows in views.items()} for d, views in checks.items()},
            'per_family': {d: {f: {v: metrics([r for r in rows if r['family'] == f]) for v, rows in views.items()}
                for f in sorted({r['family'] for r in views['native']})} for d, views in checks.items()},
            'literal_present_slice': {d: {v: metrics([r for r in rows if r['literal_target_present']])
                for v, rows in views.items()} for d, views in checks.items()},
            'native_vs_conflict_differences': {d: sum(a['state'] != b['state'] for a, b in
                zip(views['native'], views['mention_conflict'])) for d, views in checks.items()},
            'failure_reasons': dict(sorted(failure_counts.items())), 'stage_availability': stages,
            'replay_changed': sum(not row['same_full_evidence'] for row in replays),
            'replays_both_complete': sum(row['both_complete'] for row in replays),
            'reference_semantics_read_after_prediction_fsync': True,
            'clinical_qualification_passed': False, 'clinical_accuracy': None,
            'patient_or_candidate_inputs_read': False, 'selection_changed': False,
            'old_labels_scores_winners_changed': False, 'regeneration_authorized': False}
        dump(temporary/'scored_checks.json', checks)
        dump(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        # Pins referring to the temporary run become relative artifact pins below.
        guard.verify_pins(pins)
        source_pins = {p: digest for p, digest in pins.items() if not Path(p).is_relative_to(temporary)}
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-run-manifest',
            'sources': source_pins, 'plan_manifest_sha256': expected,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        private_permissions(temporary)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'authored112_parser_diagnostic_complete', 'authored_texts': 112,
        'parser_passes': 224, 'parsing_seconds': runtime['parsing_seconds'],
        'replay_changed': summary['replay_changed'], 'clinical_qualified': False,
        'manifest_sha256': sha256_file(target/'manifest.json')}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('prepare', 'evaluate'))
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--plan-root', type=Path)
    cli.add_argument('--plan-manifest-sha256')
    args = cli.parse_args()
    os.umask(0o007)
    try:
        _, status = prepare(args.run_id) if args.action == 'prepare' else \
            evaluate(args.plan_root, args.plan_manifest_sha256, args.run_id)
        print(json.dumps(status, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'authored112_failed_closed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

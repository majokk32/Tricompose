#!/usr/bin/env python3
"""Frozen official rule-parser diagnostic on authored texts, CPU Slurm only."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools'), str(ROOT / 'benchmarks'),
    str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'), str(ROOT.parent / 'src')]
from contracts import (WORKSPACE, PROTECTED_ROOT, sha256_file, require_inside,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_text)
from tricompose_v12 import opacity_context as parser
from tricompose_v12 import report_assertions as mapping
from tricompose_v12.assertion_abstention import authored_readout
import score_cached_opacity_candidates as guard

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
ENV = WORKSPACE / 'runtime/venvs/report-context-v12-12576792'
VERSIONS = {'medspacy': '1.3.1', 'spacy': '3.7.5', 'PyRuSH': '1.0.12', 'numpy': '1.26.4'}
OLD_PLAN = BASE / 'opacity_assertion_stages_plans/authored48_stages_v3_12666569_001'
OLD_PLAN_SHA = '238a8e44133c2edc9c69d460a49fcbfbb5e62ef38ed64672699a808eaa088551'
OLD_RUN = BASE / 'opacity_assertion_stages_runs/authored48_stages_v3_12677513'
OLD_RUN_SHA = '73dd5b1a26a0043caac1e4822ff901a496167947d873b12201e6d7c19e4bacce'
TESTS = ROOT / 'tests/test_opacity_context.py'
PROTOCOL = ROOT.parent / 'docs/opacity_context_parser_protocol.md'
FIXTURES = ROOT / 'benchmarks/opacity_context_controls_v1.py'
SCHEMA = 'tricompose-opacity-context-diagnostic-v1'
POLICY = {'old_authored_texts': 48, 'new_authored_texts': 64, 'finding': 'lung_opacity',
    'target_pattern': parser.TARGET_PATTERN, 'rule_parser': 'medspacy_context_official_default',
    'pipeline': ['medspacy_pyrush', 'medspacy_context'], 'context_rule_count': 102,
    'official_rules_modified': False, 'trained_model_components': False,
    'replay_all_texts': True, 'retry_failed': False, 'max_characters': 8192,
    'qualifier_and_change_only_authored_convention': 'uncertain',
    'historical_family_hypothetical_only_convention': 'unknown',
    'resolved_prior_finding_does_not_globally_negate_target': True,
    'target_noun_is_not_lung_anatomy_disambiguation': True,
    'model_calls': 0, 'external_api': False, 'primary_metric_eligible': False,
    'independent_clinical_qualification': False, 'selector_enabled': False,
    'regeneration_authorized': False, 'heldout_test': False,
    'investigator_known_previous_results': True, 'new_fixture_frozen_before_parser_execution': True}
ASSETS = {
    'context_rules': ('resources/en/context_rules.json', '5a0096af3761232288475a2b1f8cfbcc1aa2b3eb50c92b8bd12f8032cafec7c2'),
    'sentence_rules': ('resources/en/rush_rules.tsv', '35652fa4c72f95fe1a10279c6a88bec814c2b34dea229f60fbfde63c6af4571a'),
    'context_code': ('medspacy/context/context.py', '31a33afd5769908f527ef6fb8a38ab7d19e175c14754f88a05a4c80e4aed00a1'),
    'modifier_code': ('medspacy/context/context_modifier.py', 'a4ddb1f6344279d34c53ac46da6cf731b64a7118a43e7a165dacb980e00459db')}


def dump(path, payload):
    # Explicit fsync makes the prediction/key phase boundary inspectable.
    from contracts import write_private_json
    write_private_json(path, payload)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def initialize(pins):
    if Path(sys.prefix).resolve() != ENV.resolve() or \
            {name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS:
        raise RuntimeError('unchanged_pinned_rule_parser_environment_required')
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        import medspacy
        nlp = medspacy.load(medspacy_enable=POLICY['pipeline'])
    context = nlp.get_pipe('medspacy_context')
    if nlp.pipe_names != POLICY['pipeline'] or nlp.vocab.vectors.size or len(context.rules) != 102:
        raise ValueError('blank_official_pipeline_required')
    packages = Path(medspacy.__file__).resolve().parents[1]
    for name, (relative, expected) in ASSETS.items():
        path = packages / relative
        if sha256_file(path) != expected:
            raise ValueError('previously_frozen_official_asset_changed')
        pins[str(path)] = expected
    # Pin the complete installed parser source trees, not just the entry module.
    for name in ('medspacy', 'PyRuSH', 'PyFastNER', 'pyfastner'):
        package = packages / name
        if package.is_dir():
            for path in sorted(package.rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.pyx', '.so'):
                    pins[str(path)] = sha256_file(path)
    identity = {'versions': VERSIONS, 'environment_path': str(ENV),
        'pipeline': nlp.pipe_names, 'context_rule_count': len(context.rules),
        'official_asset_sha256s': {name: expected for name, (_, expected) in ASSETS.items()},
        'rule_dictionary_sha256': hashlib.sha256(json.dumps([r.to_dict() for r in context.rules],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'tokenizer_configuration_sha256': hashlib.sha256(nlp.tokenizer.to_bytes(exclude=['vocab'])).hexdigest(),
        'pipeline_config_sha256': hashlib.sha256(nlp.config.to_str().encode()).hexdigest(),
        'trained_models_loaded': False, 'official_rule_source': 'https://github.com/medspacy/medspacy/tree/1.3.1'}
    return nlp, identity


def fresh_inputs():
    from opacity_context_controls_v1 import cases
    inputs, refs = [], []
    for row in cases():
        parser.targets(row['text'])  # Validate bounds, not target-dependent selection.
        inputs.append({key: row[key] for key in ('item_id', 'text')})
        inputs[-1]['report_sha256'] = parser.digest(row['text'])
        refs.append({key: row[key] for key in ('item_id', 'family', 'expected_state')})
        refs[-1]['report_sha256'] = inputs[-1]['report_sha256']
    return inputs, refs


def validate_inputs(inputs, count):
    if not isinstance(inputs, list) or len(inputs) != count or len({r['item_id'] for r in inputs}) != count:
        raise ValueError('fixed_unique_authored_inventory_required')
    for row in inputs:
        if set(row) != {'item_id', 'text', 'report_sha256'} or parser.digest(row['text']) != row['report_sha256']:
            raise ValueError('bounded_hash_bound_authored_text_required')
        parser.targets(row['text'])


def prepare(args):
    guard.guard()
    pins = {str(path): sha256_file(path) for path in (Path(__file__), Path(parser.__file__),
        Path(mapping.__file__), FIXTURES, TESTS, PROTOCOL, ROOT/'src/tricompose_v12/assertion_abstention.py',
        ROOT/'tools/score_cached_opacity_candidates.py', WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py')}
    _, runtime = initialize(pins)  # Initialize rules only; no text has been parsed.
    manifest = guard.metadata(OLD_PLAN/'manifest.json', pins, OLD_PLAN_SHA)
    old_inputs = guard.metadata(OLD_PLAN/'inputs.json', pins, manifest['artifacts']['inputs.json'])['records']
    old_inputs = [{k: r[k] for k in ('item_id', 'text', 'report_sha256')} for r in old_inputs]
    guard.metadata(OLD_RUN/'manifest.json', pins, OLD_RUN_SHA)
    new_inputs, new_refs = fresh_inputs()
    validate_inputs(old_inputs, 48)
    validate_inputs(new_inputs, 64)
    if {r['report_sha256'] for r in old_inputs} & {r['report_sha256'] for r in new_inputs}:
        raise ValueError('fresh_texts_must_not_duplicate_old_texts')
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        dump(temporary/'inputs.json', {'old48': old_inputs, 'new64': new_inputs})
        dump(temporary/'references_new64.json', {'records': new_refs})
        dump(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'policy': POLICY,
            'runtime': runtime, 'parser_version': parser.VERSION,
            'old_manifest_sha256': OLD_RUN_SHA, 'old_plan_manifest_sha256': OLD_PLAN_SHA,
            'old_references_sha256': manifest['artifacts']['references.json'],
            'new_references_sha256': sha256_file(temporary/'references_new64.json'),
            'new_model_calls': 0, 'parsed_texts': 0, 'patient_or_candidate_inputs_read': False})
        guard.verify_pins(pins)
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {path.name: sha256_file(path) for path in sorted(temporary.iterdir())}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'prepared_frozen_rule_parser_plan', 'authored_texts': 112, 'new_model_calls': 0}


def parse_text(nlp, text):
    try:
        doc = nlp.make_doc(text)
        entities = []
        for target in parser.targets(text):
            entity = doc.char_span(target['char_start'], target['char_end'], label='lung_opacity', alignment_mode='strict')
            if entity is None:
                raise ValueError('exact_literal_token_alignment_required')
            entities.append(entity)
        doc.ents = entities
        return parser.serialize(nlp(doc), text)
    except Exception as error:
        return {'status': 'failed_unavailable', 'state': None, 'failure_reason': type(error).__name__,
            'literal_target_mentions': None, 'mentions': None, 'semantic_scope_verified': False,
            'independent_clinical_validation': False, 'hard_action_eligible': False,
            'regeneration_authorized': False, 'target_inventory_is_not_domain_disambiguation': True}


def join(references, records, count):
    if len(references) != count or len(records) != count or len({r['item_id'] for r in references}) != count:
        raise ValueError('all_authored_reference_rows_required')
    checks = []
    for ref, record in zip(references, records):
        if any(ref[k] != record[k] for k in ('item_id', 'report_sha256')):
            raise ValueError('exact_authored_join_required')
        checks.append({**record, 'expected_state': ref['expected_state'], 'family': ref['family']})
    return checks


def markdown(summary):
    lines = ['# Official ConText opacity / 官方语境解析诊断', '',
        'Wholly authored known development controls, not clinical gold or repaired triples.', '',
        '| Set / 文本集 | Matches / all attempted | Macro F1 | Unsafe certainty | Unavailable |',
        '| --- | ---: | ---: | ---: | ---: |']
    for dataset, r in summary['readouts'].items():
        lines.append(f"| {dataset} | {r['exact_matches']}/{r['rows']} | {r['macro_f1_present_classes']:.5f} | {r['determinate_on_uncertain_unknown']} | {r['unavailable']} |")
    r = summary['old48_staged_veto']
    lines += ['', '## Veto unchanged V3 signed proposals / 对旧 V3 的旁路弃权', '',
        f"Original four-state matches remain {r['raw_four_state_matches']}/48.",
        f"Soft retained {r['soft_retained']}/48; correct {r['soft_correct']}; incorrect {r['soft_incorrect']}.",
        f"Incorrect determinate withheld {r['incorrect_determinate_withheld']}; correct lost {r['correct_determinate_withheld']}.",
        'No retained state was corrected, no abstention credited as a correct unknown, no hard action eligible.', '',
        '## Every family / 全部文本组', '',
        '| Set | Family | Matches | Unsafe certainty |', '| --- | --- | ---: | ---: |']
    for dataset, families in summary['per_family'].items():
        for family, r in families.items():
            lines.append(f"| {dataset} | {family} | {r['exact_matches']}/{r['rows']} | {r['determinate_on_uncertain_unknown']} |")
    lines += ['', 'Official context triggers are unchanged; opacity/opacities target nouns are a disclosed custom inventory.',
        'This is a different algorithm over the same report, not independent clinical evidence.',
        'No patient data, GPU/model call, new primary score, winner replacement or regeneration.', '']
    return '\n'.join(lines)


def evaluate(args):
    guard.guard()
    if not isinstance(args.plan_manifest_sha256, str) or len(args.plan_manifest_sha256) != 64 or \
            any(ch not in '0123456789abcdef' for ch in args.plan_manifest_sha256):
        raise ValueError('explicit_sealed_plan_manifest_hash_required')
    pins = {}
    root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    manifest = guard.metadata(root/'manifest.json', pins, args.plan_manifest_sha256)
    guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan = guard.metadata(root/'plan.json', pins, manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA+'-plan' or plan['policy'] != POLICY or plan['parser_version'] != parser.VERSION:
        raise ValueError('frozen_rule_parser_policy_required')
    nlp, runtime = initialize(pins)
    if runtime != plan['runtime']:
        raise ValueError('frozen_runtime_identity_required')
    inputs = guard.metadata(root/'inputs.json', pins, manifest['artifacts']['inputs.json'])
    if set(inputs) != {'old48', 'new64'}:
        raise ValueError('two_separate_authored_sets_required')
    for dataset, count in (('old48', 48), ('new64', 64)):
        validate_inputs(inputs[dataset], count)
    rules = [r.to_dict() for r in nlp.get_pipe('medspacy_context').rules]
    tokenizer = hashlib.sha256(nlp.tokenizer.to_bytes(exclude=['vocab'])).hexdigest()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started, predictions, replays = time.monotonic(), {}, []
    try:
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            for dataset in ('old48', 'new64'):
                predictions[dataset] = []
                for item in inputs[dataset]:
                    primary, replay = parse_text(nlp, item['text']), parse_text(nlp, item['text'])
                    predictions[dataset].append({k: item[k] for k in ('item_id', 'report_sha256')})
                    predictions[dataset][-1].update(primary)
                    replays.append({'dataset': dataset, 'item_id': item['item_id'],
                        'same_full_evidence': primary == replay,
                        'both_complete': primary['status'] == replay['status'] == 'complete',
                        'replay_outcome': replay})
        if rules != [r.to_dict() for r in nlp.get_pipe('medspacy_context').rules] or \
                tokenizer != hashlib.sha256(nlp.tokenizer.to_bytes(exclude=['vocab'])).hexdigest():
            raise ValueError('unchanged_rules_and_tokenizer_configuration_required')
        dump(temporary/'predictions.json', {'schema_version': SCHEMA, **predictions})
        dump(temporary/'replays.json', {'records': replays})
        closed = {name: sha256_file(temporary/name) for name in ('predictions.json', 'replays.json')}
        dump(temporary/'prediction_freeze_receipt.json', {'sha256': closed,
            'authored_reference_semantics_read_before_prediction_fsync': False,
            'existing_qwen_predictions_read_before_prediction_fsync': False,
            'investigator_knows_previous_development_results': True})
        # No gold state/family or Qwen proposal is used by the rule parser.
        old_refs = guard.metadata(OLD_PLAN/'references.json', pins, plan['old_references_sha256'])['records']
        new_refs = guard.metadata(root/'references_new64.json', pins, plan['new_references_sha256'])['records']
        checks = {'old48': join(old_refs, predictions['old48'], 48),
                  'new64': join(new_refs, predictions['new64'], 64)}
        old_manifest = guard.metadata(OLD_RUN/'manifest.json', pins, OLD_RUN_SHA)
        old_predictions = guard.metadata(OLD_RUN/'predictions.json', pins,
                                        old_manifest['artifacts']['predictions.json'])['records']
        veto = []
        for old, current, ref in zip(old_predictions, predictions['old48'], old_refs):
            if any(old[k] != current[k] or old[k] != ref[k] for k in ('item_id', 'report_sha256')):
                raise ValueError('exact_same_report_veto_join_required')
            veto.append({k: current[k] for k in ('item_id', 'report_sha256')})
            veto[-1].update(parser.veto(old['staged'], current),
                expected_state=ref['expected_state'], family=ref['family'])
        if len(old_predictions) != 48 or len(veto) != 48:
            raise ValueError('all_original_proposals_required')
        summary = {'schema_version': SCHEMA, 'policy': POLICY, 'runtime': runtime,
            'readouts': {d: parser.metrics(r) for d, r in checks.items()},
            'per_family': {d: {f: parser.metrics([r for r in rows if r['family'] == f])
                for f in sorted({r['family'] for r in rows})} for d, rows in checks.items()},
            'old48_staged_veto': authored_readout(veto),
            'old48_veto_decisions': dict(sorted(Counter(r['decision'] for r in veto).items())),
            'old48_veto_per_family': {f: authored_readout([r for r in veto if r['family'] == f])
                for f in sorted({r['family'] for r in veto})},
            'literal_mention_coverage': {d: {'texts': len(rows),
                'texts_with_literal_mentions': sum(r['status'] == 'complete' and r['literal_target_mentions'] > 0 for r in rows)}
                for d, rows in predictions.items()},
            'parser_passes_including_replay': 224, 'replay_changed': sum(not r['same_full_evidence'] for r in replays),
            'replays_both_complete': sum(r['both_complete'] for r in replays),
            'elapsed_cpu_seconds_before_serialization': round(time.monotonic()-started, 6),
            'reference_read_after_prediction_fsync': True, 'new_model_calls': 0,
            'patient_inputs_read': False, 'candidate_bodies_read': False, 'selection_changed': False,
            'regeneration_authorized': False, 'clinical_fault_localization': False,
            'original_language_gate_changed': False, 'clinical_qualification_passed': False}
        dump(temporary/'scored_checks.json', checks)
        dump(temporary/'staged_veto_checks.json', {'records': veto})
        dump(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        guard.verify_pins(pins)
        if any(sha256_file(temporary/name) != expected for name, expected in closed.items()):
            raise ValueError('closed_parser_predictions_changed')
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-manifest',
            'sources': pins, 'plan_manifest_sha256': args.plan_manifest_sha256,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False,
            'selection_changed': False, 'regeneration_authorized': False})
        for path in (temporary, *temporary.iterdir()):
            stat = path.stat()
            if stat.st_gid not in (96293, 65534) or stat.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'completed_rule_parser_development_diagnostic', 'new_model_calls': 0,
        'parsed_texts': 112, 'parser_passes_including_replay': 224,
        'elapsed_cpu_seconds': summary['elapsed_cpu_seconds_before_serialization']}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('prepare', 'evaluate'))
    cli.add_argument('--plan-root')
    cli.add_argument('--plan-manifest-sha256')
    cli.add_argument('--output-root', required=True)
    cli.add_argument('--run-id', required=True)
    args = cli.parse_args(argv)
    os.umask(0o007)
    try:
        target, status = prepare(args) if args.action == 'prepare' else evaluate(args)
        print(json.dumps({**status, 'manifest_sha256': sha256_file(target/'manifest.json')}, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

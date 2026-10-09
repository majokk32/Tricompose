#!/usr/bin/env python3
"""Official aggregation-only check and offline readiness audit, CPU Slurm.

No report files, dataset, parser model, Java process, installer or API is used.
Only source and authored mock annotations are consumed. No clinical score.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'),
               str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, sha256_file, new_atomic_run,
                       commit_atomic_run, discard_atomic_run, write_private_json)
from tricompose_v12 import chexpert_negbio_contract as contract
import score_cached_opacity_candidates as guard

SCHEMA = 'tricompose-chexpert-negbio-readiness-v1'
CHEXPERT = WORKSPACE/'chexpert-labeler'
NEGBIO = WORKSPACE/'NegBio'
MODEL = WORKSPACE/'runtime/eval_models/chexpert_negbio/GENIA+PubMed'
NLTK_ROOT = WORKSPACE/'runtime/eval_resources/chexpert_negbio/nltk_data'
ENV = WORKSPACE/'runtime/venvs/report-context-v12-12576792'
MODULES = ('numpy', 'tqdm', 'bioc', 'pandas', 'bllipparser', 'StanfordDependencies', 'jpype', 'nltk')
RESOURCES = ('tokenizers/punkt', 'taggers/universal_tagset/en-ptb.map', 'corpora/wordnet')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@contextlib.contextmanager
def native_aggregation_module():
    # Avoid importing stages.__init__, which imports the unavailable parser.
    constants = module(CHEXPERT/'constants/constants.py', '_upstream_chexpert_constants')
    if tuple(constants.CATEGORIES) != contract.CATEGORIES or \
            (constants.POSITIVE, constants.NEGATIVE, constants.UNCERTAIN) != (1, 0, -1):
        raise ValueError('upstream_category_or_label_definition_changed')
    previous = sys.modules.get('constants')
    sys.modules['constants'] = constants
    try:
        yield module(CHEXPERT/'stages/aggregate.py', '_upstream_chexpert_aggregator'), constants
    finally:
        if previous is None:
            del sys.modules['constants']
        else:
            sys.modules['constants'] = previous


def annotation(category, value, text='authored_marker', both=False):
    infons = {'observation': category}
    if value == 0:
        infons['negation'] = 'True'
    if value == -1 or both:
        infons['uncertainty'] = 'True'
    return SimpleNamespace(infons=infons, text=text)


def aggregate(aggregator, annotations):
    collection = SimpleNamespace(documents=[SimpleNamespace(
        passages=[SimpleNamespace(annotations=annotations)])])
    result = aggregator.aggregate(collection)
    if result.shape != (1, 14):
        raise ValueError('exact_native_output_shape_required')
    return contract.native_vector(result[0])


def expected_numeric(values):
    # Independent enumeration of the precise published branch order.
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if 0 in values and -1 in values:
        return -1
    if 0 in values and 1 in values:
        return 1
    if -1 in values and 1 in values:
        return 1
    return values[0]


def component_checks():
    rows, special = [], []
    with native_aggregation_module() as (upstream, constants):
        aggregator = upstream.Aggregator(constants.CATEGORIES, verbose=False)
        for count in range(4):
            for sequence in itertools.product((1, 0, -1), repeat=count):
                key = {'Lung Opacity': list(sequence)} if sequence else {}
                value = contract.native_value(aggregator.dict_to_vec(key)[4])
                annotations = [annotation('Lung Opacity', v) for v in sequence]
                whole = aggregate(aggregator, annotations)
                if value['value'] != expected_numeric(sequence) or whole['lung_opacity'] != value:
                    raise ValueError('native_aggregation_component_check_failed')
                view = contract.conflict_view([contract.mention_state(a.infons) for a in annotations])
                expected_no_finding = None if any(v in (-1, 1) for v in sequence) else 1
                if whole['no_finding']['value'] != expected_no_finding:
                    raise ValueError('native_no_finding_component_check_failed')
                rows.append({'case_id': 'aggregation_' + str(len(rows)).zfill(3),
                             'native_mention_values': list(sequence), 'native_label': value,
                             'mention_conflict_view': view,
                             'views_differ': value['state'] != view['state'],
                             'native_no_finding_value': expected_no_finding,
                             'component_contract_passed': True})
        cases = (
            ('support_device_only', [annotation('Support Devices', 1)],
             {'support_devices': 1, 'no_finding': 1, 'cardiomegaly': None}),
            ('positive_chf_exception', [annotation('Edema', 1, 'chf')],
             {'edema': 1, 'cardiomegaly': -1, 'no_finding': None}),
            ('uncertain_heart_failure_exception', [annotation('Edema', -1, 'heart failure')],
             {'edema': -1, 'cardiomegaly': -1, 'no_finding': None}),
            ('negative_chf_no_exception', [annotation('Edema', 0, 'chf')],
             {'edema': 0, 'cardiomegaly': None, 'no_finding': 1}),
            ('two_flags_negation_precedence', [annotation('Lung Opacity', 0, both=True)],
             {'lung_opacity': 0, 'no_finding': 1}),
            ('explicit_no_finding_annotation_not_a_disease_assertion', [annotation('No Finding', 1)],
             {'no_finding': None, 'lung_opacity': None}),
        )
        for case_id, anns, expected in cases:
            actual = aggregate(aggregator, anns)
            if any(actual[finding]['value'] != value for finding, value in expected.items()):
                raise ValueError('native_special_aggregation_component_check_failed')
            special.append({'case_id': case_id, 'expected_native_values': expected,
                            'actual_native_values': {k: actual[k]['value'] for k in expected},
                            'component_contract_passed': True})
    parser = module(CHEXPERT/'args/arg_parser.py', '_upstream_chexpert_arguments').ArgParser().parser
    defaults = parser.parse_args(['--reports_path', 'unused_authored_component_path'])
    if defaults.sections_to_extract != [] or defaults.extract_strict is not False:
        raise ValueError('native_default_section_selection_changed')
    return {'component': 'unmodified_official_aggregation_only',
            'sequence_checks': rows, 'special_aggregation_checks': special,
            'official_default_sections_to_extract': [], 'official_default_extract_strict': False,
            'sample_or_patient_reports_loaded': False, 'report_parser_executed': False,
            'native_source_modified': False, 'clinical_accuracy': None}


def source_pins():
    pins = {}
    for directory, expected in ((CHEXPERT, contract.UPSTREAM['chexpert_labeler']),
                                 (NEGBIO, contract.UPSTREAM['negbio'])):
        head = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
        if head != expected:
            raise ValueError('fixed_upstream_revision_required')
        # A sparse checkout must not expose upstream clinical example files.
        for name in ('examples', 'sample_reports.csv', 'labeled_reports.csv', 'tests'):
            if (directory/name).exists():
                raise ValueError('source_only_sparse_checkout_required')
        subprocess.run(['git', '-C', str(directory), 'diff', '--exit-code', 'HEAD'],
                       check=True, capture_output=True)
        for path in sorted(directory.rglob('*')):
            if '.git' not in path.relative_to(directory).parts and path.is_file():
                if path.suffix not in ('.py', '.txt', '.md', '.rst', '.yml') and \
                        path.name not in ('LICENSE', 'Dockerfile') and \
                        path != NEGBIO/'negbio/ngrex/parser.out':
                    raise ValueError('upstream_source_or_rule_files_only')
                pins[str(path)] = sha256_file(path)
    for path in (Path(__file__), Path(contract.__file__),
                 ROOT/'tests/test_chexpert_negbio_contract.py',
                 WORKSPACE/'docs/chexpert_negbio_adapter_protocol.md',
                 ROOT/'tools/score_cached_opacity_candidates.py',
                 WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py'):
        pins[str(path)] = sha256_file(path)
    return pins


def readiness():
    available = {name: importlib.util.find_spec(name) is not None for name in MODULES}
    versions = {}
    for distribution in ('numpy', 'tqdm', 'bioc', 'pandas', 'bllipparser',
                          'pystanforddependencies', 'jpype1', 'nltk'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    resources = {}
    if available['nltk']:
        import nltk.data
        for resource in RESOURCES:
            try:
                nltk.data.find(resource, paths=[str(NLTK_ROOT)])
                resources[resource] = True
            except LookupError:
                resources[resource] = False
    else:
        resources = dict.fromkeys(RESOURCES, False)
    # Presence alone would not qualify resource integrity or inference stability.
    blockers = ['isolated_legacy_environment_not_validated',
                'stanford_dependency_jar_identity_not_validated',
                'frozen_parser_end_to_end_smoke_not_run']
    blockers += ['missing_module:' + name for name, present in available.items() if not present]
    java = shutil.which('java')
    if java is None:
        blockers.append('java_not_on_current_path')
    if not MODEL.is_dir():
        blockers.append('workspace_genia_pubmed_model_missing')
    blockers += ['missing_workspace_nltk:' + name for name, present in resources.items() if not present]
    return {'python': sys.version.split()[0], 'environment_path': sys.prefix,
            'module_available': available, 'installed_versions': versions,
            'java_on_path': java, 'workspace_parsing_model_path': str(MODEL),
            'workspace_parsing_model_directory_present': MODEL.is_dir(),
            'workspace_nltk_resource_root': str(NLTK_ROOT), 'nltk_resource_available': resources,
            'dependency_parser_ready': False, 'blockers': blockers,
            'automatic_resource_download_permitted': False, 'official_environment_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=PROTECTED_ROOT/'tricompose_v1_2/chexpert_negbio_readiness_runs')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    guard.guard()
    if Path(sys.prefix).resolve() != ENV.resolve():
        raise RuntimeError('existing_read_only_component_environment_required')
    pins = source_pins()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        started = time.monotonic()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            checks, status = component_checks(), readiness()
        elapsed = time.monotonic() - started
        write_private_json(temporary/'component_checks.json', checks)
        write_private_json(temporary/'dependency_readiness.json', status)
        summary = {'schema_version': SCHEMA, 'upstream_revisions': contract.UPSTREAM,
                   'aggregation_sequence_checks': len(checks['sequence_checks']),
                   'aggregation_special_checks': len(checks['special_aggregation_checks']),
                   'native_vs_conflict_view_difference_cases': sum(r['views_differ'] for r in checks['sequence_checks']),
                   'all_component_checks_passed': True, 'elapsed_seconds': round(elapsed, 6),
                   'report_parser_ready': status['dependency_parser_ready'],
                   'report_parser_calls': 0, 'gpu_calls': 0, 'patient_or_candidate_bodies_read': False,
                   'model_downloads': 0, 'environment_installations': 0,
                   'primary_metric_eligible': False, 'clinical_accuracy': None,
                   'existing_scores_or_winners_changed': False, 'regeneration_authorized': False}
        write_private_json(temporary/'summary.json', summary)
        guard.verify_pins(pins)
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA+'-manifest', 'sources': pins,
                           'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    print(json.dumps({'status': 'aggregation_contract_checked_parser_dependencies_unavailable',
                      'run_path': str(target), 'manifest_sha256': sha256_file(target/'manifest.json'),
                      'report_parser_calls': 0, 'model_downloads': 0}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)

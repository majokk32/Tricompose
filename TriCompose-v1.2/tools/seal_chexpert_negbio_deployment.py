#!/usr/bin/env python3
"""Seal metadata of the newly installed frozen parser; no report input."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src'),
               str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, sha256_file, new_atomic_run,
                       write_private_json, write_private_text, commit_atomic_run, discard_atomic_run)
from tricompose_v12 import chexpert_negbio_contract as contract
import score_cached_opacity_candidates as guard

ENV = WORKSPACE/'runtime/venvs/chexpert-negbio-py36-12682821-v1'
BUNDLE = WORKSPACE/'runtime/eval_models/chexpert_negbio/frozen_assets_12682821_001'
BUNDLE_SHA = 'c3fa079110adef7392e0363cbad745204b638f23cd391699834f047b2e1326b6'
INITIAL = PROTECTED_ROOT/'tricompose_v1_2/chexpert_negbio_deployment_runs/initialization_12682821_002/initialization.json'
INITIAL_SHA = 'd3add7ef1a12a36c471af9efa64621a30d915f7e4db36c3c6d7a2e8220995cfa'
CORE = {'numpy': '1.15.4', 'pandas': '0.23.4', 'networkx': '1.11', 'nltk': '3.3',
        'ply': '3.11', 'JPype1': '0.6.3', 'bioc': '1.1.dev3',
        'bllipparser': '2016.9.11', 'pystanforddependencies': '0.3.1', 'lxml': '3.7.3'}


def metadata(path, expected, pins):
    if not path.resolve().is_relative_to(WORKSPACE.resolve()) or path.suffix != '.json' or \
            path.stat().st_size > 2*1024**2 or sha256_file(path) != expected:
        raise ValueError('sealed_workspace_metadata_required')
    result = json.loads(path.read_text())
    pins[str(path)] = expected
    return result


def gather():
    guard.guard()
    pins = {}
    initial = metadata(INITIAL, INITIAL_SHA, pins)
    bundle = metadata(BUNDLE/'resource_manifest.json', BUNDLE_SHA, pins)
    if initial['status'] != 'initialized_without_reports' or initial['versions'] != CORE or \
            initial['backend'] != 'jpype' or initial['report_parser_calls'] != 0 or \
            initial['end_to_end_report_smoke_passed'] is not False:
        raise ValueError('exact_zero_report_initialization_receipt_required')
    for relative, expected in bundle['files'].items():
        path = BUNDLE/relative
        if not path.resolve().is_relative_to(BUNDLE.resolve()) or sha256_file(path) != expected:
            raise ValueError('frozen_resource_changed')
        pins[str(path)] = expected
    for directory, expected in ((WORKSPACE/'chexpert-labeler', contract.UPSTREAM['chexpert_labeler']),
                                 (WORKSPACE/'NegBio', contract.UPSTREAM['negbio'])):
        head = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
        if head != expected:
            raise ValueError('pinned_upstream_revision_required')
        subprocess.run(['git', '-C', str(directory), 'diff', '--exit-code', 'HEAD'], check=True, capture_output=True)
        for path in sorted(directory.rglob('*')):
            if '.git' not in path.relative_to(directory).parts and path.is_file():
                pins[str(path)] = sha256_file(path)
    # Conda-package provenance and builds; no unrelated environment inspected.
    conda = []
    for path in sorted((ENV/'conda-meta').glob('*.json')):
        row = json.loads(path.read_text())
        conda.append({k: row.get(k) for k in ('name', 'version', 'build', 'channel', 'sha256', 'md5')})
        pins[str(path)] = sha256_file(path)
    wanted = {}
    for line in (ROOT/'configs/chexpert_negbio_conda_exact_v1.yml').read_text().splitlines():
        if line.startswith('  - ') and '=' in line:
            name, version, build = line[4:].split('=')
            wanted[name] = (version, build)
    actual = {row['name']: (row['version'], row['build']) for row in conda}
    if any(actual.get(name) != pair for name, pair in wanted.items()):
        raise ValueError('official_conda_constraints_changed')
    # Pin all source/native-extension files in the installed parser dependencies.
    site = ENV/'lib/python3.6/site-packages'
    for package in ('bllipparser', 'bioc', 'StanfordDependencies', 'jpype', 'nltk',
                    'numpy', 'networkx', 'pandas', 'ply', 'lxml'):
        root = site/package
        if not root.is_dir():
            raise ValueError('installed_package_directory_required')
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.so'):
                pins[str(path)] = sha256_file(path)
    for path in (site/'_jpype.cpython-36m-x86_64-linux-gnu.so', ENV/'bin/python',
                 ENV/'lib/libstdc++.so.6',
                 ENV/'libexec/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/cc1plus'):
        if not path.is_file():
            raise ValueError('installed_runtime_binary_required')
        pins[str(path)] = sha256_file(path)
    for path in (Path(__file__), Path(contract.__file__),
                 ROOT/'tools/setup_chexpert_negbio_resources.py',
                 ROOT/'interfaces/initialize_chexpert_negbio_legacy_v1.py',
                 ROOT/'interfaces/initialize_chexpert_negbio_legacy_v2.py',
                 ROOT/'configs/chexpert_negbio_condarc_v1.yml',
                 ROOT/'configs/chexpert_negbio_conda_exact_v1.yml',
                 ROOT/'configs/chexpert_negbio_pip_exact_v1.txt',
                 ROOT/'tests/test_chexpert_negbio_setup.py'):
        pins[str(path)] = sha256_file(path)
    for version in range(1, 6):
        path = ROOT/f'tools/install_chexpert_negbio_pip_v{version}.sh'
        pins[str(path)] = sha256_file(path)
    check = subprocess.run([str(ENV/'bin/python'), '-m', 'pip', 'check'],
                           check=True, capture_output=True, text=True)
    if check.stdout.strip() != 'No broken requirements found.':
        raise ValueError('complete_declared_dependency_check_required')
    return pins, conda, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    pins, conda, initial = gather()
    temporary, target = new_atomic_run(PROTECTED_ROOT/'tricompose_v1_2/chexpert_negbio_deployments', args.run_id)
    try:
        write_private_json(temporary/'runtime.json', {
            'schema_version': 'tricompose-chexpert-negbio-frozen-runtime-v1',
            'python': '3.6.7', 'environment': str(ENV), 'core_versions': CORE,
            'conda_packages': conda, 'official_conda_constraints_verified': True,
            'official_pip_constraints_preserved': True,
            'additional_declared_dependency': {'Cython': '0.29.36'},
            'pip_check_passed': True, 'initialization_receipt': str(INITIAL),
            'initialization_receipt_sha256': INITIAL_SHA, 'resource_root': str(BUNDLE),
            'resource_manifest_sha256': BUNDLE_SHA, 'upstream_revisions': contract.UPSTREAM,
            'jvm_module': 'openjdk/11.0.20.1_1', 'stanford_corenlp': '3.5.2',
            'backend': 'jpype', 'universal_dependencies': True, 'representation': 'CCprocessed',
            'compiler': 'exact_conda_gcc_7.3.0', 'compiler_explicit_B_and_sysroot_paths': True,
            'workspace_resource_paths_and_ply_disk_options_explicit': True,
            'native_ply_grammar_namespace_explicit': 'negbio.ngrex.parser',
            'official_rule_or_model_changes': False, 'environment_installation_authorized': True,
            'new_slurm_submissions': 0, 'gpu_jobs': 0, 'report_inputs_read': 0,
            'report_parser_calls': 0, 'clinical_qualified': False,
            'end_to_end_report_smoke_passed': False, 'selector_or_regeneration_enabled': False})
        write_private_json(temporary/'setup_attempts.json', {'attempts': [
            {'stage': 'classic_solver_dry_run', 'status': 'cancelled_without_environment_write'},
            {'stage': 'libmamba_official_dry_run', 'status': 'solved_exact_original_constraints'},
            {'stage': 'conda_create', 'status': 'installed_exact_constraints',
             'home_registry_verify_write': 'sandbox_denied_despite_register_envs_false'},
            {'stage': 'pip_v1', 'status': 'failed_old_activation_nounset'},
            {'stage': 'pip_v2', 'status': 'failed_cc1plus_search'},
            {'stage': 'pip_v3', 'status': 'failed_linker_startup_object_search'},
            {'stage': 'pip_v4', 'status': 'cancelled_wrong_search_paths',
             'inherited_GCC_EXEC_PREFIX_hypothesis_confirmed': False},
            {'stage': 'pip_v5', 'status': 'success_exact_compiler_with_explicit_paths'},
            {'stage': 'cython_declared_dependency', 'status': 'installed_0_29_36'},
            {'stage': 'initialization_v1', 'status': 'failed_yacc_wrapper_caller_namespace'},
            {'stage': 'initialization_v2', 'status': 'initialized_without_reports',
             'elapsed_seconds': initial['elapsed_seconds']}], 'failed_artifacts_overwritten': False})
        write_private_text(temporary/'STATUS_CN_EN.md', '\n'.join([
            '# Frozen CheXpert/NegBio environment / 冻结解析环境', '',
            'Exact official Conda/pip constraints retained; new isolated Python 3.6.7 environment.',
            'Frozen GENIA+PubMed, CoreNLP 3.5.2 and pinned NLTK resources acquired and sealed.',
            'Offline JPype/universal/CCprocessed initialization passed; pip dependency check passed.',
            'Official grammar/rules/model weights unchanged. Cython 0.29.36 added for declared MKL dependency.',
            '初始化通过，不等于报告解析准确性通过。未输入任何报告；未读取患者数据。',
            'No report inputs, training, GPU jobs, new Slurm submissions, score/winner changes or regeneration.',
            'Next: freeze a separate authored-control plan and verify end-to-end parsing before candidate scoring.', '']))
        guard.verify_pins(pins)
        write_private_json(temporary/'manifest.json', {
            'schema_version': 'tricompose-chexpert-negbio-deployment-manifest-v1',
            'sources': pins, 'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    print(json.dumps({'status': 'frozen_parser_environment_sealed', 'path': str(target),
                      'manifest_sha256': sha256_file(target/'manifest.json'), 'report_parser_calls': 0}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'deployment_seal_failed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)

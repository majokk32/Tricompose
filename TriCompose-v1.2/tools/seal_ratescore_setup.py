"""Verify public asset bytes and dependency metadata; no model/report import."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

from prepare_ratescore_assets import (
    WORKSPACE, CONFIG, GID, inventory, require, require_slurm, sha256, write_json,
)


def expected_versions():
    text = (WORKSPACE / 'TriCompose-v1.2/configs/ratescore_runtime_v1.txt').read_text()
    result = {'torch': '2.6.0+cpu'}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*==[0-9][A-Za-z0-9.+-]*', line),
                'exact_top_level_dependency_required')
        name, value = line.split('==')
        require(name not in result, 'duplicate_dependency')
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--status-run', required=True)
    parser.add_argument('--environment', required=True)
    args = parser.parse_args()
    job_id = os.environ.get('SLURM_JOB_ID', '')
    require_slurm(Path('/proc/self/cgroup').read_text(), job_id)
    require(args.run_id == 'assets_' + job_id + '_001', 'exact_job_asset_id_required')
    status = WORKSPACE / ('artifacts/protected/tricompose_v1_2/ratescore_setup_' + job_id + '_001')
    env = WORKSPACE / ('runtime/venvs/ratescore-v12-' + job_id + '-001')
    require(Path(args.status_run) == status and Path(args.environment) == env
            and not status.is_symlink() and not env.is_symlink()
            and Path(sys.prefix) == env and status.resolve().is_relative_to(WORKSPACE.resolve())
            and env.resolve().is_relative_to(WORKSPACE.resolve()), 'exact_private_runtime_required')
    require(status.stat().st_gid in GID and status.stat().st_mode & 0o7777 == 0o2770,
            'private_status_directory_required')
    config = json.loads(CONFIG.read_text())
    entries = inventory(config)
    model_root = WORKSPACE / 'runtime/models/ratescore_v12' / args.run_id
    asset_manifest = model_root / 'asset_manifest.json'
    require(not model_root.is_symlink() and model_root.resolve().is_relative_to(WORKSPACE.resolve())
            and asset_manifest.is_file() and not asset_manifest.is_symlink(), 'fresh_asset_manifest_required')
    manifest = json.loads(asset_manifest.read_text())
    require(manifest['status'] == 'complete' and manifest['user_confirmed_umls_snomed_license'] is True
            and manifest['pins'] == {'config': sha256(CONFIG),
                'worker': sha256(WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py')},
            'approved_asset_receipt_required')
    verified = []
    for entry in entries:
        relative = entry['relative_path']
        path = (WORKSPACE / 'RaTEScore' / relative[len('source/'):]
                if relative.startswith('source/') else model_root / relative)
        require(not path.is_symlink() and path.resolve().is_relative_to(WORKSPACE.resolve())
                and path.stat().st_size == entry['bytes'] and path.stat().st_gid in GID
                and path.stat().st_mode & 0o7777 == 0o660 and sha256(path) == entry['sha256'],
                'official_asset_bytes_and_modes_required')
        verified.append({'path': str(path.relative_to(WORKSPACE)),
                         'bytes': entry['bytes'], 'sha256': entry['sha256']})
    require(manifest['assets'] == verified and manifest['total_bytes'] == sum(e['bytes'] for e in verified),
            'exact_asset_receipt_inventory_required')
    wanted = expected_versions()
    actual = {name: version(name) for name in wanted}
    require(actual == wanted, 'pinned_runtime_versions_required')
    require((status / 'pip_freeze.txt').is_file(), 'dependency_snapshot_required')
    pins = {}
    for relative in ('TriCompose-v1.2/configs/ratescore_assets_v1.json',
                     'TriCompose-v1.2/configs/ratescore_runtime_v1.txt',
                     'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
                     'TriCompose-v1.2/tools/seal_ratescore_setup.py',
                     'TriCompose-v1.2/slurm/61_ratescore_setup_cpu.sbatch',
                     'TriCompose-v1.2/tests/test_ratescore_setup.py',
                     'docs/ratescore_readiness.md'):
        pins[relative] = sha256(WORKSPACE / relative)
    receipt = {'schema_version': 'tricompose-ratescore-setup-receipt-v1',
        'status': 'assets_and_dependencies_verified_only', 'versions': actual,
        'environment': str(env.relative_to(WORKSPACE)), 'pins': pins,
        'asset_manifest_sha256': sha256(asset_manifest),
        'dependency_snapshot_sha256': sha256(status / 'pip_freeze.txt'),
        'transitive_dependency_versions_preregistered': False,
        'module_import_verified': False, 'model_loading_verified': False,
        'stable_inference_verified': False, 'clinical_qualified': False,
        'current_patient_scope_verified': False, 'benchmark_metrics_computed': False,
        'clinical_inputs_read': False, 'model_inference_calls': 0,
        'training': False, 'selection_changed': False, 'regeneration_authorized': False}
    os.umask(0o007)
    write_json(status / 'setup_receipt.json', receipt)
    # setup.log is still open and intentionally excluded from the sealed artifact set.
    write_json(status / 'manifest.json', {
        'schema_version': 'tricompose-ratescore-setup-manifest-v1', 'pins': pins,
        'asset_manifest_sha256': sha256(asset_manifest),
        'artifacts': [{'path': name, 'sha256': sha256(status / name)}
                      for name in ('setup_receipt.json', 'pip_freeze.txt')]})
    print(json.dumps({'status': receipt['status'], 'manifest_sha256': sha256(status / 'manifest.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'ratescore_setup_seal_failed', 'error_type': type(error).__name__}),
              file=sys.stderr)
        raise SystemExit(2)

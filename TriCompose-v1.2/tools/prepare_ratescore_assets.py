"""Pinned public source/weight acquisition only; never accepts clinical inputs.

Requires an actual Slurm allocation and separate acquisition/license approval.
The inspection command is strictly read-only and never accesses the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
CONFIG = WORKSPACE / 'TriCompose-v1.2/configs/ratescore_assets_v1.json'
HASH = re.compile(r'[a-f0-9]{64}')
REVISION = re.compile(r'[a-f0-9]{40}')
GID = (96293, 65534)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_asset(value):
    require(isinstance(value, str) and value and '\\' not in value,
            'relative_public_asset_required')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '.' not in path.parts
            and path.as_posix() == value and all(re.fullmatch(r'[A-Za-z0-9_.-]+', p)
                                               for p in path.parts),
            'safe_asset_path_required')
    return path


def inventory(config):
    require(config['schema_version'] == 'tricompose-ratescore-public-assets-v1',
            'pinned_asset_schema_required')
    source = config['source']
    require(source['repository'] == 'MAGIC-AI4Med/RaTEScore' and
            REVISION.fullmatch(source['revision']), 'official_pinned_source_required')
    require(len(config['models']) == 2 and
            {(m['name'], m['repository']) for m in config['models']} ==
            {('ner', 'Angelakeke/RaTE-NER-Deberta'),
             ('encoder', 'FremyCompany/BioLORD-2023-C')},
            'only_two_official_models_allowed')
    entries = []
    for component in (source, *config['models']):
        require(REVISION.fullmatch(component['revision']), 'pinned_revision_required')
        prefix = 'source' if component is source else component['name']
        for asset in component['files']:
            path = relative_asset(asset['path'])
            require(type(asset['bytes']) is int and 0 < asset['bytes'] <= 800_000_000
                    and HASH.fullmatch(asset['sha256']), 'bounded_hash_pinned_asset_required')
            if prefix == 'source':
                url = ('https://raw.githubusercontent.com/' + component['repository'] +
                       '/' + component['revision'] + '/' + path.as_posix())
            else:
                url = ('https://huggingface.co/' + component['repository'] +
                       '/resolve/' + component['revision'] + '/' + path.as_posix())
            entries.append({'relative_path': prefix + '/' + path.as_posix(),
                            'url': url, 'bytes': asset['bytes'], 'sha256': asset['sha256']})
    require(len({e['relative_path'] for e in entries}) == len(entries), 'duplicate_asset_path')
    total = sum(e['bytes'] for e in entries)
    require(type(config['maximum_asset_bytes']) is int and
            0 < total <= config['maximum_asset_bytes'] <= 1_300_000_000,
            'approved_asset_download_cap_required')
    return entries


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)


def private_dir(path, *, fresh=False, allow_existing_readonly=False):
    require(path.is_relative_to(WORKSPACE) and not path.is_symlink(),
            'workspace_non_symlink_target_required')
    require(path.parent.resolve().is_relative_to(WORKSPACE.resolve()),
            'workspace_parent_boundary_required')
    if not path.exists():
        path.mkdir(mode=0o2770)
        path.chmod(0o2770)
    else:
        require(not fresh, 'refuse_existing_run_or_source')
    allowed_modes = (0o2750, 0o2770) if allow_existing_readonly else (0o2770,)
    require(path.is_dir() and path.stat().st_gid in GID and
            path.stat().st_mode & 0o7777 in allowed_modes, 'project_private_directory_required')


def require_slurm(cgroup, job_id):
    require(re.fullmatch(r'[1-9][0-9]{0,11}', job_id or '') and
            re.search(r'/job_' + re.escape(job_id) + r'(?:/|\n|$)', cgroup),
            'actual_slurm_cgroup_required')


def require_approval(download_approved, license_confirmed):
    require(download_approved is True, 'explicit_asset_download_approval_required')
    require(license_confirmed is True, 'user_confirmed_umls_snomed_license_required')


def fetch(entry, target, opener):
    require(not target.exists() and not target.is_symlink(), 'fresh_file_required')
    partial = target.with_name(target.name + '.partial')
    require(not partial.exists() and not partial.is_symlink(), 'fresh_partial_required')
    digest, received = hashlib.sha256(), 0
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'TriCompose-public-asset-acquisition'})
    # No HF client, implicit token, cookie jar, credential helper or patient payload.
    with opener.open(request, timeout=60) as response, partial.open('xb') as stream:
        require(response.geturl().startswith('https://'), 'https_asset_response_required')
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            received += len(block)
            require(received <= entry['bytes'], 'asset_size_limit_exceeded')
            digest.update(block)
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    partial.chmod(0o660)
    require(received == entry['bytes'] and digest.hexdigest() == entry['sha256'],
            'upstream_asset_hash_or_size_mismatch')
    # Target is private, fresh and never intentionally replaced; preserve failure partials.
    require(not target.exists() and not target.is_symlink(), 'target_appeared_during_acquisition')
    partial.rename(target)


def acquire(args, entries):
    require_approval(args.download_approved, args.license_confirmed)
    job_id = os.environ.get('SLURM_JOB_ID', '')
    require_slurm(Path('/proc/self/cgroup').read_text(), job_id)
    require(args.run_id == 'assets_' + job_id + '_001', 'fresh_job_bound_asset_id_required')
    os.umask(0o007)
    model_parent = WORKSPACE / 'runtime/models/ratescore_v12'
    source_target = WORKSPACE / 'RaTEScore'
    require(not source_target.exists() and not source_target.is_symlink(),
            'refuse_existing_official_source')
    # Existing runtime parents are group-traversable/readable but may be 2750.
    # Preserve their permissions; require 2770 only for our new asset directories.
    private_dir(WORKSPACE / 'runtime', allow_existing_readonly=True)
    private_dir(WORKSPACE / 'runtime/models', allow_existing_readonly=True)
    private_dir(model_parent)
    target = model_parent / args.run_id
    private_dir(target, fresh=True)
    private_dir(target / 'source', fresh=True)
    private_dir(target / 'source/RaTEScore', fresh=True)
    for name in ('ner', 'encoder'):
        private_dir(target / name, fresh=True)
    pins = {'config': sha256(CONFIG), 'worker': sha256(Path(__file__))}
    write_json(target / 'acquisition_plan.json', {
        'schema_version': 'ratescore-acquisition-plan-v1', 'pins': pins,
        'assets': entries, 'license_confirmation': True, 'public_only': True,
        'clinical_inputs_read': False, 'model_inference_calls': 0,
        'clinical_qualified': False, 'selection_changed': False})
    # Disable implicit .netrc/proxy credential use. Public assets require no authentication.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for entry in entries:
            fetch(entry, target / entry['relative_path'], opener)
        require(pins == {'config': sha256(CONFIG), 'worker': sha256(Path(__file__))},
                'preparation_source_changed')
        # All public files were verified first; root-level official checkout is byte-identical.
        require(not source_target.exists() and not source_target.is_symlink(),
                'source_appeared_during_acquisition')
        (target / 'source').rename(source_target)
        receipt_entries = []
        for entry in entries:
            name = entry['relative_path']
            path = source_target / name[len('source/'):] if name.startswith('source/') else target / name
            require(path.stat().st_gid in GID and path.stat().st_mode & 0o7777 == 0o660
                    and sha256(path) == entry['sha256'], 'verified_private_asset_required')
            receipt_entries.append({'path': str(path.relative_to(WORKSPACE)),
                                    'bytes': path.stat().st_size, 'sha256': sha256(path)})
        write_json(target / 'asset_manifest.json', {
            'schema_version': 'ratescore-public-assets-complete-v1', 'status': 'complete',
            'pins': pins, 'source_revision': 'a970406b6f4155f7c7a9cb0d29b4fe94437e3361',
            'assets': receipt_entries, 'total_bytes': sum(e['bytes'] for e in entries),
            'user_confirmed_umls_snomed_license': True,
            'inference_readiness_verified': False, 'clinical_qualified': False,
            'clinical_inputs_read': False, 'model_inference_calls': 0,
            'selection_changed': False, 'regeneration_authorized': False})
    except BaseException as error:
        write_json(target / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
            'existing_runs_overwritten': False, 'partial_assets_preserved': True})
        raise
    return {'status': 'public_assets_complete_without_inference',
            'asset_manifest_sha256': sha256(target / 'asset_manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'acquire'))
    parser.add_argument('--run-id')
    parser.add_argument('--download-approved', action='store_true')
    parser.add_argument('--license-confirmed', action='store_true')
    args = parser.parse_args()
    entries = inventory(json.loads(CONFIG.read_text()))
    if args.mode == 'inspect':
        print(json.dumps({'status': 'read_only_preflight', 'asset_count': len(entries),
            'asset_bytes': sum(e['bytes'] for e in entries),
            'weight_bytes': sum(e['bytes'] for e in entries if e['relative_path'].endswith('model.safetensors')),
            'config_sha256': sha256(CONFIG), 'network_calls': 0, 'model_calls': 0,
            'requires_download_approval': True, 'requires_umls_snomed_license_confirmation': True,
            'already_deployed': (WORKSPACE / 'RaTEScore').exists()}))
    else:
        print(json.dumps(acquire(args, entries)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never print URLs with query parameters, response bodies, credentials or tracebacks.
        print(json.dumps({'status': 'asset_preparation_failed', 'error_type': type(error).__name__}),
              file=sys.stderr)
        raise SystemExit(2)

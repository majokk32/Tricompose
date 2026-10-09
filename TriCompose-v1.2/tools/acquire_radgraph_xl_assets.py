"""Acquire the approved, pinned public XL archive; no clinical inputs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import urllib.request

from tricompose_v12.radgraph_reference_contract import (
    CODE_REVISION, MODEL_TYPE, WEIGHT_BYTES, WEIGHT_FILENAME,
    WEIGHT_REVISION, WEIGHT_SHA256,
)

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')


def acquire(destination):
    cgroup = Path('/proc/self/cgroup').read_text()
    if not re.search(r'/slurm/[^\n]*/job_[0-9]+/', cgroup):
        raise ValueError('actual_slurm_cgroup_required')
    destination = Path(destination).resolve()
    if not destination.is_relative_to(WORKSPACE / 'runtime/models'):
        raise ValueError('workspace_model_destination_required')
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    url = (f'https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/'
           f'{WEIGHT_REVISION}/{WEIGHT_FILENAME}')
    archive = destination / WEIGHT_FILENAME
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={'User-Agent': 'TriCompose-pinned-public-assets'})
    with urllib.request.urlopen(request, timeout=60) as source, archive.open('xb') as output:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > WEIGHT_BYTES:
                raise ValueError('archive_download_size_limit_exceeded')
            digest.update(chunk)
            output.write(chunk)
    if size != WEIGHT_BYTES or digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError('archive_size_or_sha256_mismatch')
    model_dir = destination / 'model_cache' / MODEL_TYPE
    model_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as package:
        members = package.getmembers()
        if len(members) > 1024 or sum(item.size for item in members) > 2_000_000_000:
            raise ValueError('unpacked_archive_limit_exceeded')
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in seen or \
                    not (member.isdir() or member.isfile()):
                raise ValueError('unsafe_archive_member')
            seen.add(member.name)
        for member in members:
            target = model_dir / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
    config_path = model_dir / 'config.json'
    config = json.loads(config_path.read_text())
    encoder = config['model']['embedder']['token_embedders']['bert']['model_name']
    tokenizer = config['dataset_reader']['token_indexers']['bert']['model_name']
    assets = []
    for path in sorted(model_dir.rglob('*')):
        if path.is_file():
            file_digest = hashlib.sha256()
            with path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    file_digest.update(chunk)
            assets.append({'path': str(path.relative_to(destination)),
                           'bytes': path.stat().st_size, 'sha256': file_digest.hexdigest()})
    manifest = {'schema_version': 'radgraph-xl-public-assets-v1',
                'model_type': MODEL_TYPE, 'code_revision': CODE_REVISION,
                'weight_revision': WEIGHT_REVISION,
                'archive_bytes': size, 'archive_sha256': digest.hexdigest(),
                'encoder_model_name': encoder, 'tokenizer_model_name': tokenizer,
                'assets': assets, 'download_total_limit_bytes': 2_000_000_000,
                'encoder_weights_required_separately': False,
                'new_model_calls': 0, 'clinical_qualified': False}
    with (destination / 'asset_manifest.json').open('x') as output:
        json.dump(manifest, output, sort_keys=True, indent=2)
        output.write('\n')
    print(json.dumps({'status': 'archive_verified_and_extracted',
                      'runtime_seconds': round(time.monotonic() - started, 3),
                      'archive_sha256': digest.hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    try:
        acquire(args.output_root)
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)

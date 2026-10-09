"""Pinned public config/tokenizer only, with an explicit offline cache alias."""
import hashlib
import json
from pathlib import Path
import urllib.request

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
CACHE = WORKSPACE / '.cache/radgraph_v12/huggingface/hub'
REPOSITORY = 'microsoft/BiomedVLP-CXR-BERT-general'
REVISION = '6172dbfa7c061d635a4b86761b80e324e1995496'
FILES = {
    'config.json': (473, '976cc9c59dc0efa7a5d349cef15f3fa5402340b3'),
    'tokenizer_config.json': (228, 'bcab7319baa9b2d0620a92808581c48343e73bcb'),
    'vocab.txt': (235402, 'a106caa20ff3c3d8feb06454feb2deaa93a3b69a'),
}


def acquire():
    root = CACHE / ('models--' + REPOSITORY.replace('/', '--'))
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / 'snapshots' / REVISION
    snapshot.mkdir(parents=True)
    assets = []
    for filename, (expected_size, expected_blob) in FILES.items():
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}'
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(expected_size + 1)
        git_blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if len(data) != expected_size or git_blob != expected_blob:
            raise ValueError('public_tokenizer_blob_mismatch')
        with (snapshot / filename).open('xb') as output:
            output.write(data)
        assets.append({'path': filename, 'bytes': len(data), 'git_blob': git_blob,
                       'sha256': hashlib.sha256(data).hexdigest()})
    (root / 'refs').mkdir()
    with (root / 'refs/main').open('x') as output:
        output.write(REVISION)
    manifest = {'schema_version': 'radgraph-xl-public-tokenizer-v1',
                'repository': REPOSITORY, 'revision': REVISION, 'assets': assets,
                'offline_main_alias': REVISION,
                'download_bytes': sum(item['bytes'] for item in assets),
                'encoder_weights_downloaded': False}
    with (root / 'asset_manifest.json').open('x') as output:
        json.dump(manifest, output, sort_keys=True, indent=2)
        output.write('\n')
    print(json.dumps({'status': 'pinned_tokenizer_assets_verified',
                      'manifest_sha256': hashlib.sha256(
                          (root / 'asset_manifest.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    try:
        acquire()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)

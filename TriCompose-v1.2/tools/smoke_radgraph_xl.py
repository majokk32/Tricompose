"""Approved existing-CPU-allocation smoke on invented text, never MIMIC."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from tricompose_v12.radgraph_reference_contract import CODE_REVISION, WEIGHT_SHA256
from tricompose_v12.radgraph_reference_contract_v2 import NATIVE_LABELS, score_table

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
ASSETS = WORKSPACE / 'runtime/models/radgraph-xl-v12-12714150-001'
HUB_CACHE = WORKSPACE / '.cache/radgraph_v12/huggingface/hub'
TOKENIZER_ROOT = HUB_CACHE / 'models--microsoft--BiomedVLP-CXR-BERT-general'
TOKENIZER_REVISION = '6172dbfa7c061d635a4b86761b80e324e1995496'
OUTPUT_ROOT = WORKSPACE / 'artifacts/protected/tricompose_v1_2/radgraph_interface_runs'

# Invented interface probes. Not expert-annotated clinical validation examples.
PAIRS = (
    ('identical', 'Mild pulmonary edema is present.', 'Mild pulmonary edema is present.'),
    ('negation', 'Mild pulmonary edema is present.', 'No pulmonary edema is present.'),
    ('uncertainty', 'Mild pulmonary edema is present.', 'Possible mild pulmonary edema.'),
    ('laterality', 'There is a small left pleural effusion.',
                  'There is a small right pleural effusion.'),
    ('measurement', 'The heart measures 14 cm in width.', 'The heart measures 18 cm in width.'),
    ('empty', 'Mild pulmonary edema is present.', ''),
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with path.open('x') as output:
        json.dump(payload, output, sort_keys=True, indent=2)
        output.write('\n')
    path.chmod(0o660)


def prepare(run_id):
    require(re.fullmatch(r'native_xl_12714150_[0-9]{3}', run_id), 'opaque_run_id_required')
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(),
            'approved_existing_slurm_allocation_required')
    os.umask(0o007)
    OUTPUT_ROOT.mkdir(mode=0o2770, exist_ok=True)
    OUTPUT_ROOT.chmod(0o2770)
    require(OUTPUT_ROOT.stat().st_gid in (96293, 65534), 'project_mount_group_required')
    run = OUTPUT_ROOT / run_id
    run.mkdir(mode=0o2770, exist_ok=False)
    run.chmod(0o2770)
    return run


def verify_assets():
    manifest = json.loads((ASSETS / 'asset_manifest.json').read_text())
    require(manifest['archive_sha256'] == WEIGHT_SHA256 and
            sha256(ASSETS / 'radgraph-xl.tar.gz') == WEIGHT_SHA256,
            'pinned_archive_required')
    for item in manifest['assets']:
        path = ASSETS / item['path']
        require(path.resolve().is_relative_to(ASSETS) and path.is_file() and
                sha256(path) == item['sha256'], 'model_asset_hash_mismatch')
    tokenizer = json.loads((TOKENIZER_ROOT / 'asset_manifest.json').read_text())
    require(tokenizer['revision'] == TOKENIZER_REVISION and
            (TOKENIZER_ROOT / 'refs/main').read_text() == TOKENIZER_REVISION,
            'pinned_offline_tokenizer_revision_required')
    for item in tokenizer['assets']:
        path = TOKENIZER_ROOT / 'snapshots' / TOKENIZER_REVISION / item['path']
        require(path.resolve().is_relative_to(TOKENIZER_ROOT) and
                sha256(path) == item['sha256'], 'tokenizer_asset_hash_mismatch')
    labels = (ASSETS / 'model_cache/radgraph-xl/vocabulary/radgraph-xl__ner_labels.txt').read_text()
    require(tuple(label for label in labels.splitlines() if label) == NATIVE_LABELS,
            'complete_official_xl_ontology_required')
    revision = subprocess.check_output(['git', '-C', str(WORKSPACE / 'RadGraph'),
                                        'rev-parse', 'HEAD'], text=True).strip()
    require(revision == CODE_REVISION, 'pinned_official_source_required')
    return manifest, tokenizer


def run_smoke(run):
    started = time.monotonic()
    manifest, tokenizer = verify_assets()
    os.environ.update({
        'HF_HOME': str(HUB_CACHE.parent), 'HF_HUB_CACHE': str(HUB_CACHE),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '',
        'XDG_CACHE_HOME': str(WORKSPACE / '.cache/radgraph_v12/xdg'),
        'TMPDIR': str(WORKSPACE / '.tmp/radgraph_deploy_12714150_001'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
    })
    require(importlib.metadata.version('radgraph') == '0.1.18', 'official_package_version_required')
    # Offline mode plus blocked sockets: no hidden download or clinical API.
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        import radgraph
        from radgraph import F1RadGraph

        installed = Path(radgraph.__file__).parent
        source = WORKSPACE / 'RadGraph/radgraph'
        matched = 0
        for path in source.rglob('*.py'):
            counterpart = installed / path.relative_to(source)
            require(counterpart.is_file() and sha256(path) == sha256(counterpart),
                    'installed_official_python_source_mismatch')
            matched += 1
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        loaded_at = time.monotonic()
        metric = F1RadGraph(reward_level='all', model_type='radgraph-xl', cuda=-1,
                           model_cache_dir=str(ASSETS / 'model_cache'),
                           tokenizer_cache_dir=str(HUB_CACHE))
        metric.eval()
        metric.requires_grad_(False)
        require(not any(parameter.requires_grad for parameter in metric.parameters()),
                'frozen_parameters_required')
        initialization_seconds = time.monotonic() - loaded_at
        inference_started = time.monotonic()
        references, hypotheses = [pair[1] for pair in PAIRS], [pair[2] for pair in PAIRS]
        with torch.inference_mode():
            native = metric(refs=references, hyps=hypotheses)
        inference_seconds = time.monotonic() - inference_started
    ids = [f'pair_{index:04d}' for index in range(len(PAIRS))]
    nonempty = [bool(ref.strip() and hyp.strip()) for ref, hyp in zip(references, hypotheses)]
    table = score_table(ids, nonempty, native)
    for row, pair in zip(table['records'], PAIRS):
        row['authored_probe_type'] = pair[0]
    # Mechanical identity and empty-input invariants, not clinical accuracy.
    require(all(value == 1 for value in table['records'][0]['scores'].values()),
            'identical_nonempty_graph_reward_invariant_failed')
    require(table['records'][-1]['status'] == 'empty_input_not_eligible',
            'empty_pair_eligibility_invariant_failed')
    write_json(run / 'score_table.json', table)
    write_json(run / 'native_annotations.json', {
        'scope': 'invented_interface_probes_not_patient_data',
        'hypotheses': native[2], 'references': native[3]})
    summary = {'schema_version': 'radgraph-xl-interface-smoke-v1',
               'status': 'complete', 'job_id': 12714150, 'device': 'cpu',
               'authored_pairs': len(PAIRS), 'eligible_pairs': sum(nonempty),
               'report_inference_passes': 2 * sum(nonempty),
               'frozen_parameters': True, 'network_disabled': True,
               'clinical_qualified': False, 'independent_expert_benchmark': False,
               'selection_changed': False, 'regeneration_authorized': False,
               'code_revision': CODE_REVISION, 'installed_source_files_verified': matched,
               'archive_sha256': manifest['archive_sha256'],
               'tokenizer_revision': tokenizer['revision'],
               'model_download_bytes': manifest['archive_bytes'] + tokenizer['download_bytes'],
               'initialization_seconds': initialization_seconds,
               'inference_seconds': inference_seconds,
               'runtime_seconds': time.monotonic() - started,
               'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
               'versions': {name: importlib.metadata.version(name) for name in
                            ('radgraph', 'torch', 'transformers', 'numpy', 'huggingface-hub')}}
    write_json(run / 'summary.json', summary)
    pins = [{'path': str(path.relative_to(WORKSPACE)), 'sha256': sha256(path)}
            for path in (Path(__file__).resolve(),
                         WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
                         WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
                         ASSETS / 'asset_manifest.json', TOKENIZER_ROOT / 'asset_manifest.json')]
    outputs = [{'path': path.name, 'sha256': sha256(path)}
               for path in sorted(run.glob('*.json'))]
    write_json(run / 'run_manifest.json', {'schema_version': 'radgraph-xl-interface-receipt-v1',
                                         'pins': pins, 'outputs': outputs})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = None
    try:
        run = prepare(args.run_id)
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    summary = run_smoke(run)
                except Exception:
                    traceback.print_exc()
                    raise
        print(json.dumps({'status': summary['status'],
                          'runtime_seconds': round(summary['runtime_seconds'], 3),
                          'peak_rss_gib': round(summary['peak_rss_gib'], 3),
                          'manifest_sha256': sha256(run / 'run_manifest.json')}))
    except Exception as error:
        if run is not None:
            write_json(run / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                             'clinical_qualified': False})
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)

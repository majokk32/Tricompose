#!/usr/bin/env python3
"""Fetch explicitly authorized frozen parsing resources; no report inference.

HTTPS only, fixed upstream NLTK revision, bounded downloads/extraction, no
symlinks or archive traversal. Never call a library's automatic downloader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import urllib.request
from urllib.parse import urlparse
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import WORKSPACE, sha256_file, write_private_json
import score_cached_opacity_candidates as guard

NLTK_REV = '550b6625bcef1f2abff2ff770a5a0d272c9c6b2a'
MODEL_URL = 'https://www.dropbox.com/s/ev3h78gq7526xdj/BLLIP-GENIA-PubMed.tar.bz2?dl=1'
MAVEN = 'https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/3.5.2/stanford-corenlp-3.5.2.jar'
ASSETS = {
    'genia_pubmed': (MODEL_URL, 'BLLIP-GENIA-PubMed.tar.bz2', 128*1024**2),
    'stanford_corenlp_3_5_2': (MAVEN, 'stanford-corenlp-3.5.2.jar', 32*1024**2),
    'stanford_corenlp_published_sha1': (MAVEN+'.sha1', 'stanford-corenlp-3.5.2.jar.sha1', 4096),
}
for name, group in (('punkt', 'tokenizers'), ('universal_tagset', 'taggers'), ('wordnet', 'corpora')):
    ASSETS[name] = (f'https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_REV}/packages/{group}/{name}.zip',
                   name+'.zip', 32*1024**2)


def private_dir(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o2770)
    os.chmod(path, 0o2770)


class TrustedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urlparse(newurl)
        host = target.hostname or ''
        if target.scheme != 'https' or not (host == 'www.dropbox.com' or
                host == 'dropboxusercontent.com' or host.endswith('.dropboxusercontent.com')):
            raise ValueError('unexpected_resource_redirect')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download(url, path, cap):
    if path.exists() or urlparse(url).scheme != 'https':
        raise ValueError('fresh_https_resource_destination_required')
    opener = urllib.request.build_opener(TrustedRedirect())
    request = urllib.request.Request(url, headers={'User-Agent': 'TriCompose-resource-setup/1'})
    total = 0
    with opener.open(request, timeout=60) as response, path.open('xb') as output:
        declared = response.headers.get('Content-Length')
        if declared and int(declared) > cap:
            raise ValueError('resource_download_size_cap_exceeded')
        while True:
            chunk = response.read(1024**2)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise ValueError('resource_download_size_cap_exceeded')
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o660)
    if not total or (declared and total != int(declared)):
        raise ValueError('incomplete_resource_download')
    return {'source_url': url, 'bytes': total, 'sha256': sha256_file(path),
            'upstream_published_sha256_available': False}


def destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or not relative.parts:
        raise ValueError('unsafe_resource_archive_path')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe_resource_archive_path')
    return path


def copy_stream(source, target, expected):
    private_dir(target.parent)
    written = 0
    with target.open('xb') as output:
        while True:
            chunk = source.read(1024**2)
            if not chunk:
                break
            written += len(chunk)
            if written > expected:
                raise ValueError('resource_archive_size_mismatch')
            output.write(chunk)
    if written != expected:
        raise ValueError('resource_archive_size_mismatch')
    os.chmod(target, 0o660)


def extract_tar(archive, root):
    private_dir(root)
    with tarfile.open(archive, 'r:bz2') as stream:
        members = stream.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 384*1024**2:
            raise ValueError('resource_archive_size_cap_exceeded')
        for entry in members:
            target = destination(root, entry.name)
            if entry.isdir():
                private_dir(target)
            elif entry.isfile():
                with stream.extractfile(entry) as source:
                    copy_stream(source, target, entry.size)
            else:
                raise ValueError('resource_archive_links_or_special_files_forbidden')


def extract_zip(archive, root):
    private_dir(root)
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        if len(members) > 20000 or sum(m.file_size for m in members) > 192*1024**2:
            raise ValueError('resource_archive_size_cap_exceeded')
        for entry in members:
            mode = entry.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError('resource_archive_links_forbidden')
            target = destination(root, entry.filename)
            if entry.is_dir():
                private_dir(target)
            else:
                with stream.open(entry) as source:
                    copy_stream(source, target, entry.file_size)


def install(args):
    guard.guard()
    if not args.approved_resource_setup or not args.run_id.isascii() or \
            not args.run_id.replace('_', '').replace('-', '').isalnum() or len(args.run_id) > 100:
        raise ValueError('explicit_resource_authorization_and_opaque_id_required')
    parent = WORKSPACE/'runtime/eval_models/chexpert_negbio'
    private_dir(parent)
    target = parent/args.run_id
    if target.exists():
        raise FileExistsError('resource_bundle_already_exists')
    temporary = WORKSPACE/'.tmp'/('chexpert_resources_' + uuid.uuid4().hex)
    temporary.mkdir(mode=0o2770)
    os.chmod(temporary, 0o2770)
    private_dir(temporary/'archives')
    acquired = {}
    for name, (url, filename, cap) in ASSETS.items():
        acquired[name] = download(url, temporary/'archives'/filename, cap)
        print(json.dumps({'stage': 'resource_downloaded', 'asset': name,
                          'bytes': acquired[name]['bytes'], 'sha256': acquired[name]['sha256']}), flush=True)
    jar = temporary/'archives/stanford-corenlp-3.5.2.jar'
    published = (temporary/'archives/stanford-corenlp-3.5.2.jar.sha1').read_text().strip().split()[0]
    if len(published) != 40 or hashlib.sha1(jar.read_bytes()).hexdigest() != published:
        raise ValueError('published_maven_sha1_mismatch')
    acquired['stanford_corenlp_3_5_2']['published_sha1_matches'] = True
    if not zipfile.is_zipfile(jar):
        raise ValueError('valid_java_archive_required')
    extract_tar(temporary/'archives/BLLIP-GENIA-PubMed.tar.bz2', temporary/'GENIA+PubMed')
    for name, group in (('punkt', 'tokenizers'), ('universal_tagset', 'taggers'), ('wordnet', 'corpora')):
        extract_zip(temporary/'archives'/(name+'.zip'), temporary/'nltk_data'/group)
    files = {str(p.relative_to(temporary)): sha256_file(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
    write_private_json(temporary/'resource_manifest.json', {
        'schema_version': 'tricompose-chexpert-negbio-resource-bundle-v1',
        'nltk_data_revision': NLTK_REV, 'acquired_assets': acquired, 'files': files,
        'setup_script_sha256': sha256_file(Path(__file__)), 'patient_data_present': False,
        'automatic_library_download_used': False, 'model_trained_or_modified': False,
        'parser_inference_run': False, 'approved_resource_setup': True})
    if target.exists():
        raise FileExistsError('resource_bundle_already_exists')
    os.rename(temporary, target)
    print(json.dumps({'status': 'frozen_resources_acquired', 'resource_root': str(target),
                      'manifest_sha256': sha256_file(target/'resource_manifest.json'),
                      'parser_inference_run': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--approved-resource-setup', action='store_true')
    try:
        install(parser.parse_args())
    except Exception as error:
        print(json.dumps({'status': 'resource_setup_failed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)

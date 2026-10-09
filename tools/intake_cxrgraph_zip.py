#!/usr/bin/env python3
"""Receive a user-uploaded CXRGraph ZIP without decoding clinical payloads.

Only a fixed manual-data allowlist is unpacked. An existing intake is never
overwritten. Checksums authenticate consistency with the *embedded* list, not
the archive's remote origin or the checkpoint's held-out training membership.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from datetime import datetime, timezone
import zipfile


PROTECTED = Path('/project2/ruishanl_1185/inference_3mod/artifacts/protected')
FILES = (
    'LICENSE.txt', 'SHA256SUMS.txt', 'data_dictionary.md',
    'manual_data/train.json', 'manual_data/dev.json', 'manual_data/test.json',
    'manual_data/test_mimic.json', 'manual_data/test_chexpert.json',
)
GROUPS = {96293, 65534}  # Project group or its CARC NFS representation.


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or path.is_absolute()
            or '..' in path.parts or str(path) != name.rstrip('/')):
        raise ValueError('unsafe_package_path')
    return name


def member_inventory(archive):
    entries = archive.infolist()
    if len(entries) > 100 or sum(e.file_size for e in entries) > 800_000_000:
        raise ValueError('package_size_limit')
    members = {}
    for entry in entries:
        name = safe_name(entry.filename)
        kind = stat.S_IFMT(entry.external_attr >> 16)
        if (name in members or entry.flag_bits & 1
                or kind not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError('unsafe_package_entry')
        members[name] = entry
    return members


def parse_checksums(payload):
    sums = {}
    for line in payload.decode('ascii').splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+)', line)
        if not match:
            raise ValueError('invalid_checksum_metadata')
        name = match[2].removeprefix('./')
        safe_name(name)
        if name in sums:
            raise ValueError('duplicate_checksum_entry')
        sums[name] = match[1].lower()
    return sums


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def private_mode(path, directory=False):
    os.chmod(path, 0o2770 if directory else 0o660)
    if path.stat().st_gid not in GROUPS:
        raise ValueError('project_group_required')


def make_directory(path):
    path.mkdir(mode=0o2770)
    private_mode(path, directory=True)


def write_receipt(path, receipt):
    with path.open('x', encoding='utf-8') as stream:
        private_mode(path)
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')


def intake(root):
    root = root.resolve(strict=True)
    if not root.is_relative_to(PROTECTED.resolve(strict=True)):
        raise ValueError('protected_destination_required')
    private_mode(root, directory=True)
    archive_path = root / 'archive/cxrgraph_1.0.0.zip'
    if (archive_path.is_symlink() or not archive_path.is_file()
            or archive_path.resolve(strict=True) != archive_path):
        raise ValueError('regular_archive_required')
    private_mode(archive_path)
    if any((root / name).exists() for name in
           ('source', 'source.partial', 'intake_started.json', 'intake_manifest.json')):
        raise ValueError('existing_intake_refused')

    with zipfile.ZipFile(archive_path) as archive:
        members = member_inventory(archive)
        checksum_names = [name for name in members
                          if PurePosixPath(name).name == 'SHA256SUMS.txt']
        if len(checksum_names) != 1:
            raise ValueError('unique_checksum_list_required')
        checksum_name = checksum_names[0]
        prefix = checksum_name[:-len('SHA256SUMS.txt')]
        if any(prefix + name not in members for name in FILES):
            raise ValueError('required_member_missing')
        if (members[checksum_name].file_size > 65536
                or any(members[prefix + name].is_dir() for name in FILES)
                or any(members[prefix + name].file_size > 32_000_000 for name in FILES)):
            raise ValueError('selected_member_size_or_type_invalid')
        checksums = parse_checksums(archive.read(checksum_name))
        if any(name not in checksums for name in FILES if name != 'SHA256SUMS.txt'):
            raise ValueError('required_checksum_missing')

        # Exclusive marker keeps reruns from replacing even an interrupted intake.
        write_receipt(root / 'intake_started.json', {
            'schema_version': 'cxrgraph-zip-intake-start-v1',
            'started_utc': datetime.now(timezone.utc).isoformat(),
        })
        partial = root / 'source.partial'
        make_directory(partial)
        make_directory(partial / 'manual_data')
        receipts = []
        for name in FILES:
            info = members[prefix + name]
            target = partial / name
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as incoming, target.open('xb') as outgoing:
                private_mode(target)
                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                    size += len(block)
                    if size > info.file_size:
                        raise ValueError('member_size_mismatch')
                    digest.update(block)
                    outgoing.write(block)
            actual = digest.hexdigest()
            if size != info.file_size or (name in checksums and actual != checksums[name]):
                raise ValueError('member_checksum_or_size_mismatch')
            receipts.append({
                'path': 'source/' + name, 'bytes': size, 'sha256': actual,
                'zip_crc_verified': True,
                'embedded_sha256_verified': name in checksums,
            })
        if (root / 'source').exists():
            raise ValueError('existing_destination_refused')
        partial.rename(root / 'source')

    manifest = {
        'schema_version': 'cxrgraph-zip-intake-v1',
        'intake_id': root.name,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_release': 'CXRGraph 1.0.0',
        'source_origin': 'user_uploaded_declared_official_physionet_archive',
        'archive': {'path': 'archive/cxrgraph_1.0.0.zip',
                    'bytes': archive_path.stat().st_size, 'sha256': sha256(archive_path)},
        'files': receipts,
        'archive_entries': len(members),
        'unextracted_entries': len(members) - len(FILES),
        'bulk_inference_extracted': False, 'examples_pdf_extracted': False,
        'annotation_json_parsed': False, 'report_content_inspected': False,
        'remote_archive_authenticity_verified': False,
        'integrity_scope': 'selected_members_crc_and_embedded_sha256_only',
        'checksum_list_self_authenticated': False,
        'checkpoint_training_overlap': None,
        'extraction_accuracy': None, 'model_calls': 0,
        'worker_sha256': sha256(Path(__file__)),
        'status': 'manual_files_available_content_validation_not_run',
    }
    write_receipt(root / 'intake_manifest.json', manifest)
    return root / 'intake_manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        receipt = intake(args.root)
    except Exception:
        # Never echo an archive-provided path, exception payload or patient content.
        print('status=intake_failed_no_existing_intake_overwritten', file=sys.stderr)
        return 1
    print('status=intake_complete')
    print('manifest_sha256=' + sha256(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

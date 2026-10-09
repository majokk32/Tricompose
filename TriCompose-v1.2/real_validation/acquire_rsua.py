#!/usr/bin/env python3
"""Acquire the small public RSUA validated ZIP and inspect metadata only.

No extraction, numpy.load, pixel decoding, model imports or job submission.
The published clinical folder classes are not adjudicated disease gold labels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import zipfile

WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROTECTED = WORKSPACE / "artifacts/protected"
sys.path.insert(0, str(WORKSPACE / "TriCompose-v1.0/eval/report_v1_1"))
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       sha256_file, write_private_json)

PAGE = "https://data.mendeley.com/datasets/2jg8vfdmpm/1"
FILES_URL = ("https://data.mendeley.com/public-api/datasets/2jg8vfdmpm/files"
             "?folder_id=root&version=1&$start=0&$limit=1000")
S3_HOST = "prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
SCHEMA = "tricompose-rsua-public-acquisition-v1"
HEADERS = {"User-Agent": "TriCompose-public-metadata-audit/1.0",
           "Accept": "application/vnd.mendeley-public-dataset.1+json"}


def allowed_url(url):
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.hostname in ("data.mendeley.com", S3_HOST)
            and parsed.port in (None, 443) and parsed.username is None and parsed.password is None)


class PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError("public_download_redirect_boundary_rejected")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def public_opener():
    return build_opener(ProxyHandler({}), PublicRedirect())


def bounded_public_get(opener, url, limit):
    if not allowed_url(url):
        raise ValueError("public_metadata_boundary_rejected")
    with opener.open(Request(url, headers=HEADERS), timeout=15) as response:
        if response.status != 200:
            raise ValueError("public_metadata_status_rejected")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("public_metadata_size_bound_exceeded")
    return data


def select_validated_archive(rows):
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("public_file_list_schema_rejected")
    candidates = [row for row in rows if isinstance(row, dict)
                  and isinstance(row.get("filename"), str)
                  and "validated" in row["filename"].casefold()
                  and row["filename"].casefold().endswith(".zip")]
    if len(candidates) != 1:
        raise ValueError("unique_validated_archive_required")
    row = candidates[0]
    details = row.get("content_details", {})
    size, checksum, url = details.get("size"), details.get("sha256_hash"), details.get("download_url")
    if (type(size) is not int or not 0 < size <= MAX_DOWNLOAD_BYTES or row.get("size") != size
            or not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)
            or not isinstance(url, str) or not allowed_url(url)
            or urlparse(url).hostname != "data.mendeley.com"
            or not urlparse(url).path.startswith("/public-files/datasets/2jg8vfdmpm/files/")
            or not urlparse(url).path.endswith("/file_downloaded")
            or urlparse(url).query):
        raise ValueError("published_archive_contract_rejected")
    return {"size_bytes": size, "published_sha256": checksum, "download_url": url}


def download_verified(opener, contract, target):
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o660)
    digest, received = hashlib.sha256(), 0
    started = time.monotonic()
    with os.fdopen(descriptor, "wb") as handle:
        with opener.open(Request(contract["download_url"], headers=HEADERS), timeout=15) as response:
            if response.status != 200 or not allowed_url(response.geturl()):
                raise ValueError("public_archive_response_rejected")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != contract["size_bytes"]:
                raise ValueError("archive_content_length_mismatch")
            for chunk in iter(lambda: response.read(256 * 1024), b""):
                received += len(chunk)
                if received > contract["size_bytes"] or time.monotonic() - started > 180:
                    raise ValueError("archive_download_bound_exceeded")
                digest.update(chunk)
                handle.write(chunk)
    os.chmod(target, 0o660)
    if received != contract["size_bytes"] or digest.hexdigest() != contract["published_sha256"]:
        raise ValueError("published_archive_integrity_mismatch")
    return {"size_bytes": received, "sha256": digest.hexdigest(),
            "published_checksum_verified": True,
            "download_seconds": round(time.monotonic() - started, 6)}


def read_npy_header(handle):
    prefix = handle.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY" or prefix[6:] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise ValueError("unsupported_npy_header")
    width = 2 if prefix[6] == 1 else 4
    raw_length = handle.read(width)
    if len(raw_length) != width:
        raise ValueError("incomplete_npy_header")
    length = struct.unpack("<H" if width == 2 else "<I", raw_length)[0]
    if not 0 < length <= 16 * 1024:
        raise ValueError("npy_header_size_bound_exceeded")
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("incomplete_npy_header")
    try:
        fields = ast.literal_eval(raw.decode("utf-8" if prefix[6] == 3 else "latin1"))
    except (SyntaxError, ValueError, UnicodeError):
        raise ValueError("npy_header_schema_rejected") from None
    if (not isinstance(fields, dict) or set(fields) != {"descr", "fortran_order", "shape"}
            or type(fields["fortran_order"]) is not bool or not isinstance(fields["shape"], tuple)
            or not 1 <= len(fields["shape"]) <= 5
            or any(type(dim) is not int or dim <= 0 or dim > 100000 for dim in fields["shape"])):
        raise ValueError("npy_header_schema_rejected")
    numeric = isinstance(fields["descr"], str) and re.fullmatch(r"[<>=|][uibf](1|2|4|8)", fields["descr"]) is not None
    return {"shape": list(fields["shape"]), "numeric_plain_dtype": numeric,
            "dtype": fields["descr"] if numeric else None,
            "unsafe_or_non_numeric_dtype": not numeric, "fortran_order": fields["fortran_order"],
            "header_sha256": hashlib.sha256(prefix + raw_length + raw).hexdigest(),
            "bytes_read": 8 + width + length, "array_values_read": False}


def archive_inventory(path):
    extensions, npy = Counter(), []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 5000 or sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
            raise ValueError("archive_inventory_bound_exceeded")
        for index, member in enumerate(members):
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or member.flag_bits & 1 or stat.S_ISLNK(mode)):
                raise ValueError("archive_member_boundary_rejected")
            if member.is_dir():
                continue
            suffix = name.suffix.casefold()
            extensions[suffix if suffix in (".npy", ".bmp", ".png", ".jpg", ".jpeg", ".txt", ".pdf") else "other"] += 1
            if suffix == ".npy":
                with archive.open(member) as handle:
                    header = read_npy_header(handle)
                npy.append({"member_index": index, "member_name_sha256": hashlib.sha256(member.filename.encode()).hexdigest(),
                            "uncompressed_bytes": member.file_size, **header})
    return {"member_count": len(members), "uncompressed_bytes": sum(member.file_size for member in members),
            "file_extensions": dict(extensions), "npy_headers": npy,
            "archive_extracted": False, "pixel_values_decoded": False,
            "clinical_class_mapping_verified": False, "patient_grouping_verified": False,
            "age_population_verified": False, "primary_metric_eligible": False}


def run(*, output_root, run_id, allow_public_download):
    if allow_public_download is not True:
        raise RuntimeError("explicit_public_download_approval_required")
    temporary, target = new_atomic_run(output_root, run_id)
    started = time.monotonic()
    try:
        opener = public_opener()
        page = bounded_public_get(opener, PAGE, 2 * 1024 * 1024)
        match = re.search(rb"INITIAL_STATE\s*=\s*", page)
        if match is None:
            raise ValueError("public_license_state_missing")
        state, _ = json.JSONDecoder().raw_decode(page[match.end():].decode("utf-8"))
        snapshot = state["dataset"]["snapshot"]
        if (snapshot.get("id") != "2jg8vfdmpm" or snapshot.get("version") != 1
                or snapshot.get("is_confidential") is not False or snapshot.get("is_metadata_only") is not False
                or snapshot.get("licence", {}).get("short_name") != "CC BY 4.0"):
            raise ValueError("public_license_or_dataset_version_rejected")
        raw_files = bounded_public_get(opener, FILES_URL, 2 * 1024 * 1024)
        contract = select_validated_archive(json.loads(raw_files))
        integrity = download_verified(opener, contract, temporary / "validated_archive.zip")
        inventory = archive_inventory(temporary / "validated_archive.zip")
        summary = {"schema_version": SCHEMA, "status": "acquired_metadata_only_pending_class_mapping",
                   "dataset": "rsua", "dataset_version": 1, "doi": "10.17632/2jg8vfdmpm.1",
                   "official_page": PAGE, "license": "CC BY 4.0", "archive_role": "validated",
                   "public_page_sha256": hashlib.sha256(page).hexdigest(),
                   "public_file_list_sha256": hashlib.sha256(raw_files).hexdigest(),
                   "source_download_url_sha256": hashlib.sha256(contract["download_url"].encode()).hexdigest(),
                   "archive_integrity": integrity, "inventory": inventory, "credentials_read": False,
                   "model_calls": 0, "slurm_submissions": 0, "selection_changed": False,
                   "thresholds_changed": False, "reference_kind": "published_cohort_classes_not_independently_adjudicated",
                   "segmentation_validation_is_not_disease_adjudication": True,
                   "benchmark_executed": False, "primary_metric_eligible": False,
                   "elapsed_seconds": round(time.monotonic() - started, 6)}
        write_private_json(temporary / "acquisition.json", summary)
        write_private_json(temporary / "manifest.json", {
            "schema_version": SCHEMA + "-manifest", "run_id": run_id,
            "program_sha256": sha256_file(Path(__file__)),
            "artifacts": {name: sha256_file(temporary / name)
                          for name in ("validated_archive.zip", "acquisition.json")}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(PROTECTED / "tricompose_v1_2/reference_datasets/rsua"))
    parser.add_argument("--allow-public-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary = run(output_root=args.output_root, run_id=args.run_id,
                              allow_public_download=args.allow_public_download)
    except Exception as error:
        # Native URL/ZIP/parser errors may include member names or signed URLs.
        print(json.dumps({"status": "acquisition_failed_closed", "error_type": type(error).__name__, "model_calls": 0}))
        return 2
    print(json.dumps({"status": summary["status"], "archive_bytes": summary["archive_integrity"]["size_bytes"],
                      "published_checksum_verified": True, "model_calls": 0,
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""VinDr access/schema readiness, not a dataset reader or model benchmark.

The default run opens no real source. Optional authorized inspection reads
only one bounded physical CSV header line, never rows or image pixels. An
attestation records a user's confirmations; it does not verify a DUA remotely
or replace approval of a future Slurm script. No credentials are accepted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROTECTED = WORKSPACE / "artifacts/protected"
sys.path.insert(0, str(WORKSPACE / "TriCompose-v1.0/eval/report_v1_1"))
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       sha256_file, write_private_json, write_private_text)

SCHEMA = "tricompose-vindr-readiness-v1"
ATTESTATION_SCHEMA = "tricompose-vindr-access-attestation-v1"
OFFICIAL_PAGE = "https://physionet.org/content/vindr-cxr/1.0.0/"
PROBE_URL = "https://physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv"
ANNOTATION_RELATIVE_PATH = Path("annotations/image_labels_test.csv")
MAX_HEADER_BYTES = 16 * 1024
MAX_ATTESTATION_BYTES = 8 * 1024
FINDINGS = ("atelectasis", "cardiomegaly", "consolidation", "edema",
            "lung_opacity", "pleural_effusion", "pneumonia", "pneumothorax")
# Explicit candidate aliases, NOT a claim that the official header was seen.
# No substring matching, inferred vector positions or pathology synonyms.
ALIASES = {
    "image_id": "image_id", "rad_ID": "rad_ID",
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation", "Edema": "edema",
    "Lung opacity": "lung_opacity", "Lung Opacity": "lung_opacity",
    "Pleural effusion": "pleural_effusion", "Pleural Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax",
    "class_name": "class_name", "x_min": "x_min", "y_min": "y_min",
    "x_max": "x_max", "y_max": "y_max", "labels": "labels", "label": "label",
}
REQUIRED_CONFIRMATIONS = (
    "dataset_specific_dua_confirmed", "required_training_confirmed",
    "all_project_group_readers_authorized_confirmed",
)
ATTESTATION_KEYS = {"schema_version", "dataset", "dataset_version", "purpose",
                    "local_dataset_root", *REQUIRED_CONFIRMATIONS}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def header_inventory(line):
    """Sanitized schema only; any arbitrary column text is hashed, not echoed."""
    if (not isinstance(line, bytes) or not line or len(line) > MAX_HEADER_BYTES
            or not line.endswith(b"\n")):
        raise ValueError("bounded_complete_header_required")
    try:
        header = next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")], strict=True))
    except (UnicodeError, csv.Error, StopIteration):
        raise ValueError("invalid_header_encoding_or_csv") from None
    if not header or any(not name for name in header):
        raise ValueError("empty_header_column")
    recognized, unrecognized, seen = [], [], set()
    duplicate = len(header) != len(set(header))
    for index, name in enumerate(header):
        canonical = ALIASES.get(name)
        if canonical is None:
            unrecognized.append({"column_index": index,
                                 "column_name_sha256": digest_bytes(name.encode("utf-8"))})
        else:
            duplicate |= canonical in seen
            seen.add(canonical)
            recognized.append({"column_index": index, "canonical_name": canonical})
    missing = [name for name in ("image_id", *FINDINGS) if name not in seen]
    if duplicate:
        status = "blocked_duplicate_or_aliased_columns"
    elif "rad_ID" in seen:
        status = "blocked_train_reader_schema_not_test_consensus"
    elif seen.intersection({"class_name", "x_min", "y_min", "x_max", "y_max"}):
        status = "blocked_bounding_box_source_not_global_labels"
    elif seen.intersection({"labels", "label"}):
        status = "blocked_vector_schema_requires_documented_order"
    elif missing:
        status = "blocked_missing_named_reference_heads"
    else:
        status = "named_header_candidate_requires_semantics_review"
    return {"status": status, "column_count": len(header),
            "recognized_columns": recognized, "unrecognized_columns": unrecognized,
            "missing_required_names": missing, "header_sha256": digest_bytes(line),
            "annotation_rows_read": False, "reference_states_inferred": False,
            "missing_reference_policy": "unknown_not_negative",
            "official_schema_byte_authenticity_verified": False}


def validate_attestation(payload):
    """No person/account/token fields; exact true confirmations, never defaults."""
    if (not isinstance(payload, dict) or set(payload) != ATTESTATION_KEYS
            or payload.get("schema_version") != ATTESTATION_SCHEMA
            or payload.get("dataset") != "vindr-cxr"
            or payload.get("dataset_version") != "1.0.0"
            or payload.get("purpose") != "header_only_readiness"
            or any(payload.get(key) is not True for key in REQUIRED_CONFIRMATIONS)
            or not isinstance(payload.get("local_dataset_root"), str)
            or not payload["local_dataset_root"].startswith("/")):
        raise ValueError("invalid_access_attestation")
    return payload["local_dataset_root"]


def inspect_authorized_header(attestation_path, *, allow_authorized_header):
    # Approval/allocation guard must precede EVERY supplied path operation.
    if allow_authorized_header is not True or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("explicit_header_approval_and_slurm_required")
    protected = PROTECTED.resolve(strict=True)
    receipt = Path(attestation_path).resolve(strict=True)
    info = receipt.stat()
    if (not receipt.is_relative_to(protected) or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o660 or info.st_gid not in (96293, 65534)
            or info.st_size > MAX_ATTESTATION_BYTES):
        raise ValueError("private_attestation_boundary_required")
    parent_info = receipt.parent.stat()
    if (stat.S_IMODE(parent_info.st_mode) != 0o2770
            or parent_info.st_gid not in (96293, 65534)):
        raise ValueError("private_attestation_boundary_required")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid_access_attestation") from None
    dataset_root = validate_attestation(payload)
    root = Path(dataset_root).resolve(strict=True)
    project = WORKSPACE.parent.resolve(strict=True)
    if not root.is_relative_to(project) or not root.is_dir():
        raise ValueError("readonly_project_source_boundary_required")
    source = (root / ANNOTATION_RELATIVE_PATH).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("annotation_source_boundary_required")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    with os.fdopen(descriptor, "rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("regular_annotation_file_required")
        # FileIO has no buffered readahead: only this first physical line is read.
        line = handle.readline(MAX_HEADER_BYTES + 1)
        after = os.fstat(handle.fileno())
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("annotation_changed_during_header_read")
    inventory = header_inventory(line)
    return {"header": inventory, "annotation_size_bytes": before.st_size,
            "attestation_sha256": sha256_file(receipt),
            "source_header_only": True, "source_full_file_hashed": False,
            "dataset_access": "user_attested_not_remotely_verified"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _head_result(code, content_type):
    if code in (401, 403):
        status = "unauthenticated_access_denied"
    elif 300 <= code < 400:
        status = "redirect_not_followed"
    elif code == 200 and "html" in content_type.lower():
        status = "html_response_not_dataset_access"
    elif code == 200:
        status = "head_responded_not_data_or_dua_verified"
    elif code == 404:
        status = "resource_unavailable_at_probed_url"
    else:
        status = "other_http_status_not_access_verified"
    return {"status": status, "http_status": code}


def probe_unauthenticated_head():
    """Hardcoded official URL; no proxy/auth/cookie/redirect/body downloads."""
    request = Request(PROBE_URL, method="HEAD",
                      headers={"User-Agent": "TriCompose-metadata-readiness/1.0"})
    opener = build_opener(ProxyHandler({}), NoRedirect())
    started = time.monotonic()
    try:
        with opener.open(request, timeout=15) as response:
            result = _head_result(response.status, response.headers.get("Content-Type", ""))
    except HTTPError as error:
        result = _head_result(error.code, error.headers.get("Content-Type", "") if error.headers else "")
        error.close()
    except (URLError, TimeoutError, OSError):
        result = {"status": "network_unavailable_not_access_verified", "http_status": None}
    result.update({"url": PROBE_URL, "method": "HEAD", "response_body_read": False,
                   "authentication_used": False, "redirects_followed": False,
                   "timeout_seconds": 15, "elapsed_seconds": round(time.monotonic() - started, 6),
                   "checked_at_utc": datetime.now(timezone.utc).isoformat()})
    return result


def build_readiness(*, access_attestation=None, allow_authorized_header=False, probe=False):
    summary = {"schema_version": SCHEMA, "dataset": "vindr-cxr", "dataset_version": "1.0.0",
               "official_page": OFFICIAL_PAGE, "status": "blocked_authorized_local_data_required",
               "annotation_header": None, "annotation_rows_read": False,
               "image_pixels_read": False, "patient_keys_written": False,
               "credentials_read": False, "model_calls": 0, "downloads": 0,
               "slurm_submissions": 0, "thresholds_changed": False,
               "selection_changed": False, "benchmark_executed": False,
               "primary_metric_eligible": False, "can_prepare_gpu_job": False,
               "patient_grouping": "unverified_image_id_is_not_patient_id",
               "checkpoint_training_overlap": "unverified",
               "dicom_decoder": "not_implemented_or_validated_for_this_dataset",
               "label_encoding": "not_verified_no_label_rows_read",
               "missing_reference_policy": "unknown_not_negative"}
    if access_attestation is not None:
        checked = inspect_authorized_header(access_attestation,
                                            allow_authorized_header=allow_authorized_header)
        summary["annotation_header"] = checked
        summary["status"] = checked["header"]["status"]
        if summary["status"] == "named_header_candidate_requires_semantics_review":
            summary["status"] = "pending_label_semantics_and_dicom_preflight"
    elif allow_authorized_header:
        raise ValueError("access_attestation_required")
    summary["public_head_probe"] = probe_unauthenticated_head() if probe else None
    return summary


def populate_run(summary, temporary, run_id):
    write_private_json(temporary / "readiness.json", summary)
    head = summary["public_head_probe"]
    probe_status = head["status"] if head else "not_requested"
    report = ("# VinDr-CXR readiness / 数据入口预检\n\n"
                  f"Status / 状态: `{summary['status']}`.\n\n"
                  f"Unauthenticated HEAD / 未登录 HEAD: `{probe_status}`.\n\n"
                  "No dataset rows, pixels or credentials were read. No dataset was downloaded; "
                  "no model, threshold fitting, reselection or Slurm submission ran.\n\n"
                  "没有读取数据行、图像像素或凭证；没有下载数据、运行模型、调整阈值或更改择优。\n\n"
                  "This is a readiness receipt, NOT benchmark scores or clinical qualification. "
                  "An HTTP status does not verify a user's DUA or authenticated access.\n\n"
                  "这是数据入口预检记录，不是 benchmark 得分。HTTP 状态不能证明用户已签 DUA。\n\n"
                  "Next: confirm dataset-specific DUA/training and eligibility of every project-group "
                  "reader; provide an authorized local root; review the actual image-label schema and "
                  "encoding, DICOM protocol and provenance; then review a separate exact Slurm script.\n")
    # The optional branch DOES read a header, but never any annotation row.
    write_private_text(temporary / "RESULTS_CN_EN.md", report)
    manifest = {"schema_version": SCHEMA + "-manifest", "run_id": run_id,
                "source_code_sha256": sha256_file(Path(__file__)),
                "status": summary["status"], "model_calls": 0,
                "artifacts": {name: sha256_file(temporary / name)
                              for name in ("readiness.json", "RESULTS_CN_EN.md")}}
    write_private_json(temporary / "manifest.json", manifest)


def write_run(summary, *, output_root, run_id):
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        populate_run(summary, temporary, run_id)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(PROTECTED / "tricompose_v1_2/vindr_readiness_runs"))
    parser.add_argument("--access-attestation")
    parser.add_argument("--allow-authorized-header", action="store_true")
    parser.add_argument("--probe-public-access", action="store_true")
    args = parser.parse_args(argv)
    try:
        # Reserve before network/source checks: an existing run is never rerun.
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        try:
            summary = build_readiness(access_attestation=args.access_attestation,
                                      allow_authorized_header=args.allow_authorized_header,
                                      probe=args.probe_public_access)
            populate_run(summary, temporary, args.run_id)
            commit_atomic_run(temporary, target)
        except BaseException:
            discard_atomic_run(temporary)
            raise
    except (ValueError, RuntimeError, OSError, TypeError):
        print(json.dumps({"status": "readiness_failed_closed", "model_calls": 0,
                          "benchmark_executed": False}))
        return 2
    print(json.dumps({"status": summary["status"], "model_calls": 0,
                      "benchmark_executed": False, "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

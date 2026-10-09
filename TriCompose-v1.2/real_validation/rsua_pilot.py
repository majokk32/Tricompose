#!/usr/bin/env python3
"""Seal a metadata-only RSUA cohort; score it only in approved GPU Slurm.

This is published pneumonia-vs-normal cohort discrimination, not adjudicated
image truth, an adult-domain guarantee or probability calibration. No training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import time
import zipfile

WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
sys.path.insert(0, str(WORKSPACE / "TriCompose-v1.0/eval/report_v1_1"))
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, read_json, require_inside, sha256_file,
                       write_private_json, write_private_text)
from extract_cxr_labels_xrv import FrozenXRVRuntime, XRV_LABELS
from xrv_calibration import SCORE_SPACE, validate_bundle
from biovil_matched_pairs import auc_ap

COHORT_SCHEMA = "tricompose-rsua-fixed-pilot-cohort-v1"
SCORE_SCHEMA = "tricompose-rsua-xrv-pilot-v1"
REFERENCE_KIND = "published_pneumonia_vs_paper_described_normal_cohort_classes"
PAPER = "https://doi.org/10.1016/j.dib.2023.109640"
CHECKPOINT_SHA = "56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"
THRESHOLDS_SHA = "630c25c4f3dc0446e3ed4fe02806e0a761f498a518f70af19546bc71c62d6be4"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metadata_role(member):
    name = PurePosixPath(member.filename)
    if member.is_dir() or name.suffix.casefold() != ".bmp" or len(name.parts) != 4:
        return None
    if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
        raise ValueError("unsafe_archive_member")
    parent, family = name.parent.name.casefold(), name.parent.parent.name.casefold()
    if "image" in parent and "mask" not in parent:
        role = "image"
    elif "mask" in parent and "image" not in parent:
        role = "mask"
    else:
        raise ValueError("unrecognized_image_mask_directory_schema")
    label = ("pneumonia" if "pneumonia" in family else "non_covid" if "non_covid" in family
             else "covid" if "covid" in family else None)
    if label is None or role not in name.stem.casefold():
        raise ValueError("source_class_or_role_marker_unrecognized")
    return label, role


def bmp_header(header, *, expected_size):
    if len(header) != 54 or header[:2] != b"BM":
        raise ValueError("unsupported_bmp_header")
    size, offset = struct.unpack_from("<I", header, 2)[0], struct.unpack_from("<I", header, 10)[0]
    dib = struct.unpack_from("<I", header, 14)[0]
    width, height = struct.unpack_from("<ii", header, 18)
    planes, bits = struct.unpack_from("<HH", header, 26)
    compression = struct.unpack_from("<I", header, 30)[0]
    if (size != expected_size or not 54 <= offset < size or dib != 40
            or (width, height, planes, bits, compression) != (256, 256, 1, 8, 0)):
        raise ValueError("bmp_protocol_mismatch")
    return {"width": width, "height": height, "bits_per_pixel": bits,
            "compression": compression, "header_sha256": hashlib.sha256(header).hexdigest()}


def select_metadata_cohort(archive, *, per_class=25, seed=0):
    if type(per_class) is not int or per_class != 25 or type(seed) is not int:
        raise ValueError("fixed_25_per_class_protocol_required")
    pools = {"pneumonia": [], "non_covid": []}
    counts = Counter()
    seen_names = set()
    for index, member in enumerate(archive.infolist()):
        if member.filename in seen_names:
            raise ValueError("duplicate_archive_member_name")
        seen_names.add(member.filename)
        tag = metadata_role(member)
        if tag is None:
            continue
        label, role = tag
        counts[(label, role)] += 1
        if role != "image" or label == "covid":
            continue
        if not 0 < member.file_size <= 1024 * 1024:
            raise ValueError("source_image_size_bound_exceeded")
        with archive.open(member) as source:
            dimensions = bmp_header(source.read(54), expected_size=member.file_size)
        name_hash = hashlib.sha256(member.filename.encode()).hexdigest()
        rank = hashlib.sha256(f"rsua|{seed}|{name_hash}".encode()).hexdigest()
        pools[label].append((rank, {"member_index": index, "member_name_sha256": name_hash,
                                    "size_bytes": member.file_size, "published_class": label,
                                    "reference_state": "positive" if label == "pneumonia" else "negative",
                                    "bmp_header": dimensions}))
    expected = {(label, role): count for label, count in (("pneumonia", 53), ("non_covid", 32), ("covid", 207))
                for role in ("image", "mask")}
    if dict(counts) != expected:
        raise ValueError("published_image_mask_class_counts_mismatch")
    chosen = []
    for label in ("pneumonia", "non_covid"):
        for _, row in sorted(pools[label], key=lambda pair: pair[0])[:per_class]:
            chosen.append({"case_id": f"case_{len(chosen):04d}", **row})
    return chosen, {f"{label}_{role}": count for (label, role), count in sorted(counts.items())}


def source_archive(source_run, source_manifest_sha256):
    source = require_inside(source_run, PROTECTED_ROOT, must_exist=True)
    manifest_path = source / "manifest.json"
    if sha256_file(manifest_path) != source_manifest_sha256:
        raise ValueError("acquisition_manifest_hash_mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "tricompose-rsua-public-acquisition-v1-manifest":
        raise ValueError("acquisition_schema_mismatch")
    for name in ("validated_archive.zip", "acquisition.json"):
        if sha256_file(source / name) != manifest["artifacts"][name]:
            raise ValueError("acquisition_artifact_hash_mismatch")
    receipt = read_json(source / "acquisition.json")
    if (receipt.get("license") != "CC BY 4.0" or receipt.get("archive_role") != "validated"
            or receipt["archive_integrity"].get("published_checksum_verified") is not True):
        raise ValueError("public_dataset_integrity_contract_missing")
    return source / "validated_archive.zip", manifest["artifacts"]["validated_archive.zip"]


def stage(args):
    archive_path, archive_hash = source_archive(args.source_run, args.source_manifest_sha256)
    with zipfile.ZipFile(archive_path) as archive:
        records, counts = select_metadata_cohort(archive, seed=args.seed)
    cohort = {"schema_version": COHORT_SCHEMA, "records": records, "counts": counts,
              "source_run": str(archive_path.parent), "source_manifest_sha256": args.source_manifest_sha256,
              "archive_sha256": archive_hash, "seed": args.seed, "per_class": 25,
              "reference_kind": REFERENCE_KIND, "reference_description_source": PAPER,
              "non_covid_negative_is_paper_described_normal_proxy": True,
              "independent_image_disease_adjudication": False, "patient_grouping_verified": False,
              "age_population_verified": False, "training_overlap_verified": False,
              "image_level_selection": True, "label_stratified": True, "natural_prevalence_estimate": False,
              "cohort_selected_before_model_scores": True, "pixel_values_decoded": False,
              "mask_images_excluded": True, "covid_images_excluded": True,
              "model_calls": 0, "primary_metric_eligible": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / "cohort.json", cohort)
        write_private_json(temporary / "manifest.json", {
            "schema_version": COHORT_SCHEMA + "-manifest", "run_id": args.run_id,
            "program_sha256": sha256_file(Path(__file__)), "case_count": len(records),
            "cohort_sha256": sha256_file(temporary / "cohort.json")})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, len(records)


def summarize_binary(records, threshold):
    labels, scores = [row["reference_state"] == "positive" for row in records], [row["pneumonia_score"] for row in records]
    if (not records or any(row["reference_state"] not in ("positive", "negative") for row in records)
            or any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores)
            or type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("invalid_binary_diagnostic_inputs")
    positive, negative = sum(labels), len(labels) - sum(labels)
    tp = sum(label and score >= threshold for label, score in zip(labels, scores))
    fp = sum(not label and score >= threshold for label, score in zip(labels, scores))
    tn, fn = negative - fp, positive - tp
    sensitivity = tp / positive if positive else None
    specificity = tn / negative if negative else None
    return {**auc_ap([int(label) for label in labels], scores), "threshold": threshold,
            "positive": positive, "negative": negative, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if positive and negative else None,
            "precision_on_balanced_cohort_only": tp / (tp + fp) if tp + fp else None,
            "f1_on_balanced_cohort_only": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate(args):
    # No path resolution, source or torch read before allocation/approval guard.
    if args.allow_rsua_pixels is not True or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved_rsua_pixel_slurm_required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_allocation_required")
    cohort_path = require_inside(args.cohort, PROTECTED_ROOT, must_exist=True)
    if sha256_file(cohort_path) != args.cohort_sha256:
        raise ValueError("sealed_cohort_hash_mismatch")
    cohort = read_json(cohort_path)
    if cohort.get("schema_version") != COHORT_SCHEMA:
        raise ValueError("sealed_cohort_schema_mismatch")
    cohort_manifest = read_json(cohort_path.parent / "manifest.json")
    if (cohort_manifest.get("schema_version") != COHORT_SCHEMA + "-manifest"
            or cohort_manifest.get("cohort_sha256") != args.cohort_sha256
            or cohort_manifest.get("program_sha256") != sha256_file(Path(__file__))):
        raise ValueError("sealed_cohort_program_or_manifest_changed")
    archive_path, archive_hash = source_archive(cohort["source_run"], cohort["source_manifest_sha256"])
    with zipfile.ZipFile(archive_path) as archive:
        selected, counts = select_metadata_cohort(archive, seed=cohort["seed"])
    if selected != cohort["records"] or counts != cohort["counts"] or archive_hash != cohort["archive_sha256"]:
        raise ValueError("sealed_cohort_selection_changed")
    weight_dir = require_inside(args.cache_dir, PROTECTED_ROOT, must_exist=True)
    weight = require_inside(weight_dir / args.weight_filename, weight_dir, must_exist=True)
    thresholds_path = require_inside(args.thresholds, PROTECTED_ROOT, must_exist=True)
    if sha256_file(weight) != CHECKPOINT_SHA or sha256_file(thresholds_path) != THRESHOLDS_SHA:
        raise ValueError("unchanged_scorer_or_threshold_hash_mismatch")
    bundle = read_json(thresholds_path)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    try:
        image_root = temporary / "inputs"
        image_root.mkdir(mode=0o2770)
        os.chmod(image_root, 0o2770)
        records, seen = [], set()
        torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            runtime = FrozenXRVRuntime(cache_dir=weight_dir, weight_filename=args.weight_filename,
                                       model_name="densenet121-res224-all")
            if runtime.model.training or any(parameter.requires_grad for parameter in runtime.model.parameters()):
                raise RuntimeError("unfrozen_scorer_rejected")
            preprocessing = {"image": "PIL_L_float32", "normalize_maxval": 255,
                             "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
                             "xrv_models_source_sha256": sha256_file(Path(runtime.xrv.models.__file__)),
                             "xrv_datasets_source_sha256": sha256_file(Path(runtime.xrv.datasets.__file__))}
            validated = validate_bundle(bundle, checkpoint_sha256=CHECKPOINT_SHA,
                                        expected_provenance={"preprocessing_sha256": fingerprint(preprocessing),
                                                             "finding_mapping_sha256": fingerprint(XRV_LABELS),
                                                             "score_space": SCORE_SPACE})
            head = validated["pneumonia"]
            if not head["enabled"] or head["positive_min"] != head["negative_max"]:
                raise ValueError("unchanged_pneumonia_threshold_must_be_enabled_single_point")
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                for record in selected:
                    member = members[record["member_index"]]
                    if metadata_role(member) != (record["published_class"], "image"):
                        raise ValueError("image_mask_or_class_lineage_changed")
                    with archive.open(member) as source:
                        data = source.read(record["size_bytes"] + 1)
                    if len(data) != record["size_bytes"] or bmp_header(data[:54], expected_size=len(data)) != record["bmp_header"]:
                        raise ValueError("selected_bmp_bytes_or_header_changed")
                    image_hash = hashlib.sha256(data).hexdigest()
                    if image_hash in seen:
                        raise ValueError("duplicate_selected_image_bytes")
                    seen.add(image_hash)
                    path = image_root / (record["case_id"] + ".bmp")
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o660)
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(data)
                    os.chmod(path, 0o660)
                    scores = runtime.predict(path)
                    records.append({"case_id": record["case_id"], "reference_state": record["reference_state"],
                                    "image_sha256": image_hash, "pneumonia_score": scores["pneumonia"]})
            torch.cuda.synchronize()
        summary = {"schema_version": SCORE_SCHEMA, "status": "completed_single_finding_cohort_diagnostic",
                   "cohort_sha256": args.cohort_sha256, "checkpoint_sha256": CHECKPOINT_SHA,
                   "thresholds_sha256": THRESHOLDS_SHA, "frozen": True, "score_space": SCORE_SPACE,
                   "probability_semantics": False, "reference_kind": REFERENCE_KIND,
                   "primary_metric_eligible": False, "independent_image_disease_adjudication": False,
                   "patient_grouping_verified": False, "age_population_verified": False,
                   "checkpoint_training_overlap_verified": False, "label_stratified": True,
                   "natural_prevalence_estimate": False, "thresholds_fitted_on_this_cohort": False,
                   "selection_changed": False, "generation_calls": 0, "model_calls": len(records),
                   "preprocessing": preprocessing,
                   "default_0_5": summarize_binary(records, 0.5),
                   "unchanged_weak_reference_threshold_transport": summarize_binary(records, head["positive_min"]),
                   "elapsed_seconds": round(time.monotonic() - started, 6),
                   "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
        write_private_json(temporary / "summary.json", summary)
        write_private_json(temporary / "scores.json", {**summary, "records": records})
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=("case_id", "reference_state", "pneumonia_score", "image_sha256"))
        writer.writeheader()
        writer.writerows(records)
        write_private_text(temporary / "score_table.csv", stream.getvalue())
        write_private_json(temporary / "manifest.json", {"schema_version": SCORE_SCHEMA + "-manifest",
            "run_id": args.run_id, "program_sha256": sha256_file(Path(__file__)),
            "artifacts": {name: sha256_file(temporary / name) for name in ("summary.json", "scores.json", "score_table.csv")}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    stage_parser = modes.add_parser("stage")
    stage_parser.add_argument("--source-run", required=True)
    stage_parser.add_argument("--source-manifest-sha256", required=True)
    stage_parser.add_argument("--seed", type=int, default=0)
    eval_parser = modes.add_parser("evaluate")
    for name in ("cohort", "cohort-sha256", "cache-dir", "weight-filename", "thresholds"):
        eval_parser.add_argument("--" + name, required=True)
    eval_parser.add_argument("--allow-rsua-pixels", action="store_true")
    for subparser in (stage_parser, eval_parser):
        subparser.add_argument("--output-root", required=True)
        subparser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        target, count = stage(args) if args.mode == "stage" else evaluate(args)
    except Exception as error:
        print(json.dumps({"status": "failed_closed", "error_type": type(error).__name__}))
        return 2
    print(json.dumps({"status": "cohort_sealed" if args.mode == "stage" else "completed",
                      "case_count": count, "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

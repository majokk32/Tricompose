#!/usr/bin/env python3
"""Fit operating points from precomputed protected validation scores; Slurm only."""
import argparse
import json
import os

from contracts import (read_json, new_atomic_run, commit_atomic_run,
                       discard_atomic_run, write_private_json, sha256_file)
from xrv_calibration import fit_bundle, validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--uncertainty-margin", type=float, default=0.0)
    args = parser.parse_args()
    # A prepared real calibration bundle remains protected patient-derived data.
    # Do not inspect it on the login node or print records in exception messages.
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    try:
        payload = read_json(args.input_bundle)
        if payload.get("schema_version") != "tricompose-xrv-calibration-input-v1":
            raise ValueError("unsupported calibration input")
        provenance = {**payload["provenance"], "calibration_input_sha256": sha256_file(args.input_bundle)}
        bundle = fit_bundle(payload["records"], provenance,
                            heldout_group_hashes=payload["heldout_group_hashes"],
                            min_per_class=args.min_per_class, uncertainty_margin=args.uncertainty_margin)
        validate_bundle(bundle)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        artifact = write_private_json(temporary / "thresholds.json", bundle)
        artifact_hash = sha256_file(artifact)
        write_private_json(temporary / "manifest.json", {
            "schema_version": "tricompose-xrv-calibration-run-v1", "run_id": args.run_id,
            "thresholds_sha256": artifact_hash, "counts": bundle["counts"],
            "primary_metric_eligible": False, "model_training_used": False,
            "model_inference_used": False})
        commit_atomic_run(temporary, target)
        temporary = None
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed", "thresholds_sha256": artifact_hash,
                      "primary_metric_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Small synthetic-only PNG integrity check; not anatomy or realism scoring.

Bounded to 16 images for lightweight inspection. Larger cohorts require a
separately approved compute job, not increasing this limit on the login node.
"""
import argparse
import json
import os

from contracts import (load_cxr_candidates, new_atomic_run, commit_atomic_run,
                       discard_atomic_run, sha256_file, write_private_json,
                       write_private_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    candidates = load_cxr_candidates(args.cxr_run)
    if len(candidates) > 16:
        raise ValueError("lightweight image audit is restricted to at most 16 images")
    from PIL import Image, ImageStat
    rows = []
    for candidate_id, candidate in sorted(candidates.items()):
        row = {"candidate_id": candidate_id, "image_sha256": candidate["artifact"]["sha256"],
               "corrupted": False, "blank": None, "width": None, "height": None,
               "anatomy_validity": "not_evaluated", "realism": "not_evaluated"}
        try:
            with Image.open(candidate["artifact"]["path"]) as image:
                if image.width * image.height > 4096 * 4096:
                    raise ValueError("image exceeds lightweight inspection bound")
                gray = image.convert("L")
                stats = ImageStat.Stat(gray)
                low, high = gray.getextrema()
                row.update(width=image.width, height=image.height,
                           mean_intensity=stats.mean[0], std_intensity=stats.stddev[0],
                           min_intensity=low, max_intensity=high,
                           blank=low == high)
        except (OSError, SyntaxError):
            row["corrupted"] = True
        rows.append(row)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        output = write_private_text(temporary / "cxr_basic_validity.json", json.dumps(rows, sort_keys=True, indent=2) + "\n")
        summary = {"schema_version": "tricompose-small-cxr-validity-v1",
                   "count": len(rows), "corrupted": sum(r["corrupted"] for r in rows),
                   "blank": sum(r["blank"] is True for r in rows),
                   "blank_rule": "constant_8bit_grayscale_only_not_anatomy_or_crop_detection",
                   "artifact_sha256": sha256_file(output), "gpu_inference_used": False}
        write_private_json(temporary / "manifest.json", summary)
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(json.dumps({"status": "completed", **summary}))


if __name__ == "__main__":
    main()

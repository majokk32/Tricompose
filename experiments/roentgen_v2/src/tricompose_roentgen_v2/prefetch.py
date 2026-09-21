"""Download and verify the audited gated RoentGen-v2 snapshot.

Run only after the user has accepted the Hugging Face model conditions and
explicitly approved the download job.  This module never handles patient data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .model import OFFICIAL_MODEL_REVISION, validate_model_snapshot


def prefetch(destination: str | Path, staging_dir: str | Path) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    destination = Path(destination).resolve(strict=False)
    staging_dir = Path(staging_dir).resolve(strict=False)
    if destination.exists():
        raise FileExistsError("final RoentGen-v2 snapshot already exists")
    if staging_dir.exists():
        raise FileExistsError("RoentGen-v2 staging directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="stanfordmimi/RoentGen-v2",
        revision=OFFICIAL_MODEL_REVISION,
        local_dir=staging_dir,
        token=True,
    )
    audit = validate_model_snapshot(staging_dir)
    os.rename(staging_dir, destination)
    return {
        "status": "completed",
        "model_revision": OFFICIAL_MODEL_REVISION,
        "weight_bytes": audit["weight_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--staging-dir", required=True)
    args = parser.parse_args()
    try:
        result = prefetch(args.destination, args.staging_dir)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__},
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch one pinned public SynEHRgy checkpoint into workspace-managed storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tricompose.privacy import PROJECT_ROOT, require_inside

from .model import MODEL_SPECS, get_model_spec, validate_model_snapshot


def _set_tree_modes(root: Path) -> None:
    for directory, _, filenames in os.walk(root):
        directory_path = Path(directory)
        os.chmod(directory_path, 0o700)
        for filename in filenames:
            os.chmod(directory_path / filename, 0o600)


def fetch_model(
    *,
    model_variant: str,
    target_dir: Path,
    cache_dir: Path,
) -> dict[str, object]:
    """Download only inference files, verify them, then publish atomically."""

    from huggingface_hub import snapshot_download

    spec = get_model_spec(model_variant)
    target = require_inside(target_dir, PROJECT_ROOT, must_exist=False)
    cache = require_inside(cache_dir, PROJECT_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError("refusing to overwrite existing model directory")

    target.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.partial.{os.getpid()}"
    if stage.exists():
        raise FileExistsError("refusing to overwrite existing model staging directory")
    stage.mkdir(mode=0o700)

    snapshot = Path(
        snapshot_download(
            repo_id=spec.repo,
            revision=spec.revision,
            cache_dir=str(cache),
            allow_patterns=list(spec.expected_file_sizes),
            token=False,
        )
    ).resolve(strict=True)
    for relative in spec.expected_file_sizes:
        source = snapshot / relative
        if not source.is_file():
            raise FileNotFoundError(f"downloaded snapshot is missing: {relative}")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)

    _set_tree_modes(stage)
    audit = validate_model_snapshot(stage, model_variant)
    os.rename(stage, target)
    return {
        "status": "complete",
        "variant": audit["variant"],
        "repo": audit["repo"],
        "revision": audit["revision"],
        "checkpoint": audit["checkpoint"],
        "weight_bytes": audit["weight_bytes"],
        "vocab_size": audit["vocab_size"],
        "model_sha256": audit["sha256"][
            f"{audit['checkpoint']}/model.safetensors"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = fetch_model(
            model_variant=args.model_variant,
            target_dir=args.target_dir,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

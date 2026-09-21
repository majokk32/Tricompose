"""Download and atomically validate the pinned RadEdit generation bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .model import (
    RADEDIT_REPO,
    RADEDIT_REVISION,
    TEXT_ENCODER_REPO,
    TEXT_ENCODER_REVISION,
    VAE_REPO,
    VAE_REVISION,
    validate_model_bundle,
)


def prefetch(destination: str | Path, staging_dir: str | Path) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    destination = Path(destination).resolve(strict=False)
    staging_dir = Path(staging_dir).resolve(strict=False)
    if destination.exists():
        raise FileExistsError("final RadEdit bundle already exists")
    if staging_dir.exists():
        raise FileExistsError("RadEdit staging directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=False)

    snapshot_download(
        repo_id=RADEDIT_REPO,
        revision=RADEDIT_REVISION,
        local_dir=staging_dir / "radedit",
        allow_patterns=["unet/config.json", "unet/diffusion_pytorch_model.safetensors"],
        token=True,
        max_workers=4,
    )
    snapshot_download(
        repo_id=VAE_REPO,
        revision=VAE_REVISION,
        local_dir=staging_dir / "sdxl-vae",
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
        token=False,
        max_workers=4,
    )
    snapshot_download(
        repo_id=TEXT_ENCODER_REPO,
        revision=TEXT_ENCODER_REVISION,
        local_dir=staging_dir / "biovil-t",
        allow_patterns=[
            "config.json",
            "configuration_cxrbert.py",
            "model.safetensors",
            "modeling_cxrbert.py",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
        token=False,
        max_workers=4,
    )
    audit = validate_model_bundle(staging_dir)
    os.rename(staging_dir, destination)
    return {
        "status": "completed",
        "radedit_revision": RADEDIT_REVISION,
        "weight_bytes": audit["weight_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--staging-dir", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prefetch(**vars(args)), sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

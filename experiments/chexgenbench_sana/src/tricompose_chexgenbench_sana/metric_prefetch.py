"""Prefetch and cryptographically freeze official CheXGenBench scorers.

This stage uses only public model artifacts. It must run through Slurm because
it instantiates the public scorers to validate their snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


OFFICIAL_REPO_COMMIT = "cc7e91ebcee946836090acb4c399e4f3912280ac"
RAD_DINO_REPO = "microsoft/rad-dino"
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
BIOVIL_T_REPO = "microsoft/BiomedVLP-BioViL-T"
BIOVIL_T_REVISION = "301f526e823b805d3fe712d0cf06a66f042789c5"
BIOVIL_T_IMAGE_WEIGHT = "biovil_t_image_model_proj_size_128.pt"
FREEZE_SCHEMA = "tricompose.chexgenbench_sana.metric_models.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise ValueError("public scorer snapshot has no files")
    return records


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_rad_dino(path: Path) -> None:
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(path, local_files_only=True)
    AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("RadDINO did not remain frozen")


def _validate_biovil_t(path: Path) -> None:
    import torch
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer

    CXRBertTokenizer.from_pretrained(path, local_files_only=True)
    model = CXRBertModel.from_pretrained(path, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("BioViL-T text encoder did not remain frozen")
    image_state = torch.load(
        path / BIOVIL_T_IMAGE_WEIGHT,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(image_state, dict) or not image_state:
        raise ValueError("BioViL-T image checkpoint is invalid")


def _warm_inception(torch_home: Path) -> list[dict[str, Any]]:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid = FrechetInceptionDistance(feature=2048)
    inception = InceptionScore(feature=2048)
    fid.eval().requires_grad_(False)
    inception.eval().requires_grad_(False)
    del fid, inception

    checkpoint_root = torch_home / "hub" / "checkpoints"
    records = []
    for path in sorted(checkpoint_root.glob("*inception*")):
        if path.is_file():
            records.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not records:
        raise FileNotFoundError("torchmetrics Inception weights were not cached")
    return records


def prefetch(destination: str | Path, staging: str | Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    destination = Path(destination).resolve(strict=False)
    staging = Path(staging).resolve(strict=False)
    if destination.exists():
        raise FileExistsError("official metric scorer directory already exists")
    if staging.exists():
        raise FileExistsError("official metric staging directory already exists")
    torch_home = Path(os.environ["TORCH_HOME"]).resolve(strict=False)
    staging.mkdir(parents=True, mode=0o700)
    rad_dino = staging / "rad-dino" / RAD_DINO_REVISION
    biovil_t = staging / "biovil-t" / BIOVIL_T_REVISION
    rad_dino.parent.mkdir(parents=True, mode=0o700)
    biovil_t.parent.mkdir(parents=True, mode=0o700)

    snapshot_download(
        repo_id=RAD_DINO_REPO,
        revision=RAD_DINO_REVISION,
        local_dir=rad_dino,
        token=False,
        max_workers=4,
    )
    snapshot_download(
        repo_id=BIOVIL_T_REPO,
        revision=BIOVIL_T_REVISION,
        local_dir=biovil_t,
        token=False,
        max_workers=4,
    )
    _validate_rad_dino(rad_dino)
    _validate_biovil_t(biovil_t)
    inception_files = _warm_inception(torch_home)

    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen",
        "official_repo_commit": OFFICIAL_REPO_COMMIT,
        "mutation_policy": "read-only inference; reject any file hash change",
        "scorers": {
            "rad_dino": {
                "repository": RAD_DINO_REPO,
                "revision": RAD_DINO_REVISION,
                "files": _snapshot_files(rad_dino),
            },
            "biovil_t": {
                "repository": BIOVIL_T_REPO,
                "revision": BIOVIL_T_REVISION,
                "files": _snapshot_files(biovil_t),
            },
            "inception": {"files": inception_files},
        },
        "software": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "torchmetrics",
                "torch-fidelity",
                "transformers",
                "hi-ml-multimodal",
            )
        },
    }
    freeze_path = staging / "freeze.json"
    _write_exclusive_json(freeze_path, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, destination)
    final_freeze = destination / "freeze.json"
    return {
        "status": "completed",
        "official_repo_commit": OFFICIAL_REPO_COMMIT,
        "rad_dino_revision": RAD_DINO_REVISION,
        "biovil_t_revision": BIOVIL_T_REVISION,
        "freeze_sha256": _sha256(final_freeze),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--staging", required=True)
    args = parser.parse_args()
    try:
        result = prefetch(args.destination, args.staging)
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

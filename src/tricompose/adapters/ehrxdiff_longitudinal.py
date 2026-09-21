"""Generate one follow-up CXR with the frozen longitudinal EHRXDiff model.

The adapter deliberately does not instantiate the upstream evaluation dataset:
that dataset loads the real target image and exposes source identifiers. This
module internally selects one condition by opaque index, loads only its previous
frontal CXR and prepared condition tensor, and emits sanitized status metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    create_private_stage_dir,
    enforce_private_file_mode,
    sha256_file,
    write_private_json,
)


STAGE_NAME = "ehrxdiff_longitudinal_to_cxr"
CONDITION_CONTRACT = "previous_report_hash_plus_interval_ehr_hash_plus_previous_cxr"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument("--clip-bootstrap-checkpoint", required=True)
    parser.add_argument("--condition-h5", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-steps", type=int, default=35)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-event-len", type=int, default=1024)
    return parser


def _resolve_existing(path: str, *, directory: bool = False) -> Path:
    resolved = Path(path).resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError("required external directory is missing")
    if not directory and not resolved.is_file():
        raise ValueError("required external file is missing")
    return resolved


def _find_previous_image(
    image_root: Path,
    previous_image_key: str,
) -> Path:
    """Locate one copied input image by filename without reading metadata rows."""
    if not previous_image_key or not all(
        character.isalnum() or character == "-"
        for character in previous_image_key
    ):
        raise ValueError("private image key contains unsupported characters")
    matches = list(image_root.rglob(f"{previous_image_key}.jpg"))
    if len(matches) != 1:
        raise LookupError("selected previous image did not resolve uniquely")
    candidate = matches[0].resolve(strict=True)
    candidate.relative_to(image_root)
    if not candidate.is_file():
        raise ValueError("mapped previous image is not a regular file")
    return candidate


def _run_private(args: argparse.Namespace, stage_dir: Path) -> dict[str, Any]:
    # Heavy and external imports remain inside the silenced private computation.
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    from torchvision.transforms.functional import to_pil_image

    external_root = _resolve_existing(args.external_root, directory=True)
    checkpoint = _resolve_existing(args.checkpoint)
    autoencoder = _resolve_existing(args.autoencoder_checkpoint)
    clip_bootstrap = _resolve_existing(args.clip_bootstrap_checkpoint)
    condition_h5 = _resolve_existing(args.condition_h5)
    image_root = _resolve_existing(args.image_root, directory=True)

    if args.sample_index < 0:
        raise ValueError("sample index must be non-negative")
    if args.sampling_steps < 1:
        raise ValueError("sampling steps must be positive")
    if args.max_event_len < 1:
        raise ValueError("max event length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this adapter through Slurm")

    sys.path.insert(0, str(external_root))
    # This legacy environment contains two editable installs whose generated
    # .pth finder is not active on CARC compute nodes. Add their existing source
    # roots explicitly; both remain read-only.
    editable_roots = [
        Path(sys.prefix) / "src" / "taming-transformers",
        Path(sys.prefix) / "src" / "clip",
    ]
    for editable_root in editable_roots:
        if editable_root.is_dir():
            sys.path.insert(0, str(editable_root))
    from cheff.ldm.inference import CheffLDMTab2IAdaptor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    with h5py.File(condition_h5, "r", driver="sec2") as handle:
        if "ehr" not in handle:
            raise ValueError("condition file does not contain the expected group")
        private_keys = sorted(handle["ehr"].keys())
        if args.sample_index >= len(private_keys):
            raise IndexError("sample index is outside the condition split")
        private_key = private_keys[args.sample_index]
        condition_array = np.asarray(handle["ehr"][private_key], dtype=np.float32)

    if condition_array.ndim != 2 or condition_array.shape[1] != 1536:
        raise ValueError("condition tensor has an incompatible shape")
    if condition_array.shape[0] > args.max_event_len:
        raise ValueError("condition sequence exceeds max event length")

    key_parts = private_key.split("_", 1)
    if len(key_parts) != 2:
        raise ValueError("private condition key has an incompatible schema")
    previous_image_key = key_parts[1]
    previous_image_path = _find_previous_image(
        image_root,
        previous_image_key,
    )

    transform = Compose(
        [
            Resize(256),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    with Image.open(previous_image_path) as image:
        previous_image = transform(image.convert("RGB")).unsqueeze(0)

    event_count = condition_array.shape[0]
    table = torch.zeros((1, args.max_event_len, 1536), dtype=torch.float32)
    table[0, :event_count] = torch.from_numpy(condition_array)
    attention_mask = torch.ones((1, args.max_event_len), dtype=torch.bool)
    attention_mask[0, :event_count] = False

    # Upstream initializes its CLIP visual tower through clip.load("ViT-B/32")
    # before the full EHRXDiff checkpoint is applied. Pin that bootstrap read to
    # an existing local file so it cannot download or write to an external cache.
    import clip as openai_clip

    original_clip_load = openai_clip.load

    def load_local_clip(
        model_name: str,
        device: str = "cuda",
        jit: bool = False,
        download_root: str | None = None,
    ) -> Any:
        del model_name, download_root
        return original_clip_load(str(clip_bootstrap), device=device, jit=jit)

    openai_clip.load = load_local_clip
    try:
        adapter = CheffLDMTab2IAdaptor(
            model_path=str(checkpoint),
            ae_path=str(autoencoder),
            device="cuda",
            num_layer=2,
            max_event_len=args.max_event_len,
            n_embed=1536,
            context_dim=768,
            condition_feat_dim=1024,
            condition_type="table, prev_img",
            conditioning_key="crossattn",
            tab_encoder_architecture="MultiModalTransformerAdaptor",
        )
    finally:
        openai_clip.load = original_clip_load
    conditioning = {
        "c_crossattn": {
            "prev_img": previous_image,
            "attn_mask": attention_mask,
            "table": table,
        }
    }

    with torch.inference_mode():
        sample = adapter.sample(
            conditioning=conditioning,
            sampling_steps=args.sampling_steps,
            eta=args.eta,
            decode=True,
            batch_size=1,
        )

    if sample.ndim != 4 or sample.shape[0] != 1:
        raise ValueError("EHRXDiff returned an incompatible image tensor")
    if not torch.isfinite(sample).all():
        raise ValueError("EHRXDiff returned non-finite pixels")

    image_tensor = ((sample[0].detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0)
    generated_image = to_pil_image(image_tensor)
    output_path = stage_dir / "generated_cxr.png"
    generated_image.save(output_path, format="PNG")
    enforce_private_file_mode(output_path)

    return {
        "checkpoint_file": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "external_inference_source_sha256": sha256_file(
            external_root / "cheff" / "ldm" / "inference.py"
        ),
        "condition_contract": CONDITION_CONTRACT,
        "input_selector_index": args.sample_index,
        "seed": args.seed,
        "sampling_steps": args.sampling_steps,
        "eta": args.eta,
        "output_file": output_path.name,
        "output_dimensions": list(generated_image.size),
        "output_sha256": sha256_file(output_path),
        "peak_vram_gib": round(
            torch.cuda.max_memory_allocated() / (1024**3),
            3,
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    started = time.monotonic()
    stage_dir: Path | None = None

    try:
        stage_dir = create_private_stage_dir(args.output_dir)
        with Path(os.devnull).open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                manifest = _run_private(args, stage_dir)

        manifest.update(
            {
                "stage": STAGE_NAME,
                "status": "ok",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        write_private_json(stage_dir / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "stage": STAGE_NAME,
                    "status": "ok",
                    "artifact": "generated_cxr.png",
                    "sha256": manifest["output_sha256"],
                    "dimensions": manifest["output_dimensions"],
                    "elapsed_seconds": manifest["elapsed_seconds"],
                    "peak_vram_gib": manifest["peak_vram_gib"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:  # Deliberately omit exception messages and tracebacks.
        failure = {
            "stage": STAGE_NAME,
            "status": "failed",
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            failure["missing_module"] = exc.name
        if stage_dir is not None:
            try:
                write_private_json(stage_dir / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

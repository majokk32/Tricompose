"""MeDiM-paper FID and Inception Score for an existing protected run.

This stage is a small-sample pilot. It consumes matched real CXRs internally,
never copies them, and writes aggregate statistics only.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medim_ehr_prompt.common import (
    load_config,
    read_private_json,
    require_private_file,
    require_run_directory,
    sha256_file,
    validate_case_id,
    write_private_json,
)
from medim_ehr_prompt.consistency import (
    _load_real_cxr_index,
    _resolve_evaluation_layout,
    _resolve_real_cxr,
    _validate_generation_prompt,
)
from tricompose.privacy import create_private_stage_dir


SUMMARY_SCHEMA = "medim_ehr_prompt.official_image_metrics.v1"
EXPECTED_SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
EXPECTED_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
EXPECTED_GENERATION_SCHEMA = "medim_ehr_prompt.generation.v1"
FID_WEIGHT_FILENAME = "pt_inception-2015-12-05-6726825d.pth"


def _inception_score_from_probabilities(
    probabilities: np.ndarray,
    *,
    splits: int,
) -> tuple[float, float]:
    """Reproduce the split-wise formula in MeDiM's official is_score.py."""
    from scipy.stats import entropy

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Inception probabilities must be a non-empty matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Inception probabilities are invalid")
    if splits <= 0 or splits > values.shape[0]:
        raise ValueError("invalid Inception Score split count")
    if values.shape[0] % splits != 0:
        raise ValueError("sample count must be divisible by Inception Score splits")

    split_size = values.shape[0] // splits
    split_scores: list[float] = []
    for split_index in range(splits):
        part = values[split_index * split_size : (split_index + 1) * split_size]
        marginal = np.mean(part, axis=0)
        divergences = [entropy(conditional, marginal) for conditional in part]
        split_scores.append(float(np.exp(np.mean(divergences))))
    return float(np.mean(split_scores)), float(np.std(split_scores))


def _collect_image_paths(
    config: dict[str, Any],
    run_dir: Path,
    source_run_id: str | None = None,
) -> tuple[list[Path], list[Path]]:
    source_run, cases, expected_generation_schema = _resolve_evaluation_layout(
        run_dir, source_run_id
    )

    real_cxr_index = _load_real_cxr_index(config)
    real_paths: list[Path] = []
    generated_paths: list[Path] = []
    for record in cases:
        case_id = validate_case_id(str(record["case_id"]))
        case_input = read_private_json(source_run / "cases" / case_id / "input.json")
        if case_input.get("schema_version") != EXPECTED_CASE_SCHEMA:
            raise ValueError("unsupported case input artifact")
        generation_dir = run_dir / "cases" / case_id / "generation"
        generation = read_private_json(generation_dir / "generation.json")
        _validate_generation_prompt(
            generation, case_input, expected_generation_schema
        )

        real_paths.append(
            _resolve_real_cxr(real_cxr_index, int(case_input["source_row_index"]))
        )
        generated_paths.append(
            require_private_file(generation_dir / "generated_cxr.png")
        )

    if len(real_paths) != len(generated_paths):
        raise RuntimeError("real/generated image count mismatch")
    return real_paths, generated_paths


def _fid_score(
    real_paths: Sequence[Path],
    generated_paths: Sequence[Path],
    *,
    device: Any,
    batch_size: int,
    num_workers: int,
    mode: str,
) -> float:
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import fid_from_feats, get_files_features

    if mode != "legacy_pytorch":
        raise ValueError("MeDiM-compatible FID mode must be legacy_pytorch")
    feature_model = build_feature_extractor(
        mode,
        device=device,
        use_dataparallel=False,
    )
    try:
        real_features = get_files_features(
            list(real_paths),
            model=feature_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            verbose=False,
        )
        generated_features = get_files_features(
            list(generated_paths),
            model=feature_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            verbose=False,
        )
    finally:
        del feature_model
    return float(fid_from_feats(generated_features, real_features))


def _load_official_is_model(checkpoint_path: Path, device: Any) -> Any:
    import torch
    from torchvision.models.inception import inception_v3

    model = inception_v3(
        weights=None,
        aux_logits=True,
        transform_input=False,
        init_weights=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def _inception_probabilities(
    paths: Sequence[Path],
    *,
    model: Any,
    device: Any,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class ImagePathDataset(Dataset[Any]):
        def __init__(self, image_paths: Sequence[Path]) -> None:
            self.paths = list(image_paths)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256), Image.Resampling.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> Any:
            with Image.open(self.paths[index]) as image:
                return self.transform(image.convert("RGB"))

    loader = DataLoader(
        ImagePathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images in loader:
            images = functional.interpolate(
                images.to(device),
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )
            logits = model(images)
            batches.append(functional.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(batches, axis=0)
    if probabilities.shape != (len(paths), 1000):
        raise RuntimeError("unexpected Inception prediction shape")
    return probabilities


def evaluate_run(
    config_path: str | Path,
    run_id: str,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    import torch

    os.umask(0o077)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run official metrics through Slurm")

    config = load_config(config_path)
    run_dir = require_run_directory(run_id)
    stage_path = run_dir / "official_image_metrics"
    if stage_path.exists():
        raise FileExistsError("official image metric stage already exists")

    metric_config = config["official_image_metrics"]
    batch_size = int(metric_config["batch_size"])
    num_workers = int(metric_config["num_workers"])
    splits = int(metric_config["is_splits"])
    is_checkpoint = Path(metric_config["is_checkpoint"]).resolve(strict=True)
    if not is_checkpoint.is_file():
        raise ValueError("configured Inception Score checkpoint is not a file")

    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    real_paths, generated_paths = _collect_image_paths(
        config, run_dir, source_run_id
    )

    fid_value = _fid_score(
        real_paths,
        generated_paths,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        mode=str(metric_config["fid_mode"]),
    )
    torch.cuda.empty_cache()

    is_model = _load_official_is_model(is_checkpoint, device)
    try:
        generated_probabilities = _inception_probabilities(
            generated_paths,
            model=is_model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        real_probabilities = _inception_probabilities(
            real_paths,
            model=is_model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    finally:
        del is_model

    generated_is_mean, generated_is_std = _inception_score_from_probabilities(
        generated_probabilities,
        splits=splits,
    )
    real_is_mean, real_is_std = _inception_score_from_probabilities(
        real_probabilities,
        splits=splits,
    )

    torch_home = Path(os.environ["TORCH_HOME"]).resolve(strict=True)
    fid_checkpoint = torch_home / "hub" / "checkpoints" / FID_WEIGHT_FILENAME
    if not fid_checkpoint.is_file():
        raise FileNotFoundError("FID Inception checkpoint was not cached as expected")

    stage_dir = create_private_stage_dir(stage_path)
    summary_path = write_private_json(
        stage_dir / "summary.json",
        {
            "schema_version": SUMMARY_SCHEMA,
            "run_id": run_id,
            "sample_count": len(generated_paths),
            "scope": (
                f"{len(generated_paths)}-sample pilot; not directly comparable "
                "to paper full-test-set values"
            ),
            "paper_reference": {
                "mimic_cxr_fid": 16.60,
                "mimic_cxr_inception_score": 2.87,
            },
            "fid": {
                "value": round(fid_value, 8),
                "direction": "lower_is_better",
                "real_set": "matched real CXR cohort",
                "generated_set": "MeDiM EHR-prompt-conditioned CXR cohort",
                "protocol": "official repository pytorch-fid compatible legacy_pytorch",
            },
            "inception_score": {
                "direction": "higher_is_better",
                "splits": splits,
                "generated_mean": round(generated_is_mean, 8),
                "generated_std": round(generated_is_std, 8),
                "real_mean": round(real_is_mean, 8),
                "real_std": round(real_is_std, 8),
                "protocol": "official repository is_score.py preprocessing and formula",
            },
            "software": {
                "clean_fid": importlib.metadata.version("clean-fid"),
                "torch": torch.__version__,
                "torchvision": importlib.metadata.version("torchvision"),
            },
            "checkpoints": {
                "fid_inception_sha256": sha256_file(fid_checkpoint),
                "is_inception_sha256": sha256_file(is_checkpoint),
            },
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    return {
        "stage": "official_image_metrics",
        "status": "ok",
        "sample_count": len(generated_paths),
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "summary_sha256": sha256_file(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_run(args.config, args.run_id, args.source_run_id),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

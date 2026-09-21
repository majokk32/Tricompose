"""Protected, official-inspired evaluation for a RoentGen-v2 real-anchor run.

This module is GPU inference and must run through a user-approved Slurm job. It
loads matched real CXRs internally but never copies them or writes source paths,
patient identifiers, EHR rows, source reports, or real images to an artifact.
Only opaque case IDs, numeric scores, aggregate statistics, and hashes are saved.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from medim_ehr_prompt.prompting import EHR_TO_CXR_TARGETS
from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)

from .common import (
    PROTECTED_EXPERIMENT_ROOT,
    PROTECTED_ROENTGEN_ROOT,
    read_private_json,
    require_source_run,
    validate_case_id,
    validate_run_id,
)


CONFIG_SCHEMA = "tricompose.roentgen_v2.eval_config.v1"
RUN_SCHEMA = "tricompose.roentgen_v2.run.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
ROENTGEN_CASE_SCHEMA = "tricompose.roentgen_v2.case_input.v1"
GENERATION_SCHEMA = "tricompose.roentgen_v2.generation.v1"
EVAL_SCHEMA = "tricompose.roentgen_v2.evaluation.v1"
CASE_METRICS_SCHEMA = "tricompose.roentgen_v2.case_metrics.v1"
PROTECTED_EVAL_ROOT = PROTECTED_ROENTGEN_ROOT / "evaluations"

OFFICIAL_FIVE = (
    "atelectasis",
    "cardiomegaly",
    "edema",
    "effusion",
    "pneumothorax",
)

AGE_INTERVALS: dict[str, tuple[float, float]] = {
    "young adult": (18.0, 30.0),
    "middle-aged adult": (30.0, 50.0),
    "older adult": (50.0, 70.0),
    "elderly adult": (70.0, math.inf),
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    real_cxr: Path
    generated_cxr: Path
    facts: Mapping[str, Any]


def _normalized_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("evaluation config must be a mapping")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported evaluation config")
    return payload


def _require_generation_run(run_id: str) -> Path:
    validate_run_id(run_id)
    resolved = require_inside(
        PROTECTED_EXPERIMENT_ROOT / run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    if not resolved.is_dir():
        raise ValueError("protected generation run is not a directory")
    enforce_private_directory_mode(resolved)
    return resolved


def _load_real_cxr_index(config: Mapping[str, Any]) -> Any:
    """Load only the private path column; values never leave process memory."""
    import pandas as pd

    dataset_root = Path(config["dataset"]["root"]).resolve(strict=True)
    return pd.read_csv(
        dataset_root / "manifest.csv",
        usecols=["cxr_path"],
        dtype={"cxr_path": str},
    )["cxr_path"]


def _resolve_real_cxr(paths: Any, source_row_index: int) -> Path:
    if source_row_index < 0 or source_row_index >= len(paths):
        raise ValueError("opaque source row index is outside the private manifest")
    value = paths.iloc[source_row_index]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("matched real CXR path is unavailable")
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError("matched real CXR is not a file")
    return path


def _collect_cases(
    config: Mapping[str, Any], generation_run_id: str
) -> tuple[list[CaseSpec], str]:
    generation_run = _require_generation_run(generation_run_id)
    manifest = read_private_json(generation_run / "manifest.json")
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unsupported RoentGen-v2 run manifest")
    if manifest.get("run_id") != generation_run_id:
        raise ValueError("generation run ID mismatch")
    if manifest.get("loads_real_target_cxr") is not False:
        raise ValueError("generation manifest does not preserve the target boundary")

    source_run_id = validate_run_id(str(manifest["source_run_id"]))
    source_run = require_source_run(source_run_id)
    records = manifest.get("cases")
    if not isinstance(records, list) or not records:
        raise ValueError("generation run has no cases")
    expected_count = int(config["evaluation"]["expected_case_count"])
    if len(records) != expected_count:
        raise ValueError("generation case count does not match evaluation config")

    real_index = _load_real_cxr_index(config)
    specs: list[CaseSpec] = []
    seen: set[str] = set()
    for record in records:
        case_id = validate_case_id(str(record["case_id"]))
        if case_id in seen:
            raise ValueError("duplicate opaque case ID")
        seen.add(case_id)

        source_case = read_private_json(source_run / "cases" / case_id / "input.json")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case")
        facts = source_case.get("ehr_facts")
        if not isinstance(facts, dict):
            raise TypeError("source case has no protected EHR facts")

        roentgen_case = read_private_json(generation_run / "cases" / case_id / "input.json")
        if roentgen_case.get("schema_version") != ROENTGEN_CASE_SCHEMA:
            raise ValueError("unsupported protected RoentGen-v2 case")
        if roentgen_case.get("source_prompt_sha256") != source_case.get("prompt_sha256"):
            raise ValueError("source prompt hash mismatch")

        generation_dir = generation_run / "cases" / case_id / "generation"
        generation = read_private_json(generation_dir / "generation.json")
        if generation.get("schema_version") != GENERATION_SCHEMA:
            raise ValueError("unsupported generation artifact")
        if generation.get("prompt_sha256") != roentgen_case.get("prompt_sha256"):
            raise ValueError("generation prompt hash mismatch")

        generated_cxr = require_private_file(generation_dir / "generated_cxr.png")
        real_cxr = _resolve_real_cxr(real_index, int(source_case["source_row_index"]))
        specs.append(
            CaseSpec(
                case_id=case_id,
                real_cxr=real_cxr,
                generated_cxr=generated_cxr,
                facts=facts,
            )
        )
    return specs, source_run_id


def _load_grayscale(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2 or min(array.shape) < 32 or not np.isfinite(array).all():
        raise ValueError("invalid CXR image array")
    return np.clip(array, 0.0, 1.0)


def _resize_gray(array: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L")
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def age_group_match(age_group: str, predicted_age: float) -> bool | None:
    """Return age-bin agreement; generic ``adult`` is intentionally unscored."""
    interval = AGE_INTERVALS.get(age_group)
    if interval is None or not math.isfinite(predicted_age):
        return None
    low, high = interval
    return predicted_age >= low and predicted_age < high


def age_distance_to_group(age_group: str, predicted_age: float) -> float | None:
    interval = AGE_INTERVALS.get(age_group)
    if interval is None or not math.isfinite(predicted_age):
        return None
    low, high = interval
    if predicted_age < low:
        return low - predicted_age
    if predicted_age >= high and math.isfinite(high):
        return predicted_age - high
    return 0.0


def _ehr_support(
    observable_targets: Mapping[str, Any],
    confidences: Mapping[str, float],
    *,
    operating_point: float,
) -> dict[str, Any]:
    scores: list[float] = []
    per_fact: dict[str, Any] = {}
    for fact in sorted(observable_targets):
        raw_targets = observable_targets[fact]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("invalid protected EHR target mapping")
        labels = [_normalized_label(str(label)) for label in raw_targets]
        missing = [label for label in labels if label not in confidences]
        if missing:
            raise ValueError("EHR target is unavailable from XRV")
        score = max(float(confidences[label]) for label in labels)
        scores.append(score)
        per_fact[str(fact)] = {
            "candidate_labels": labels,
            "support": round(score, 8),
            "supported": bool(score >= operating_point),
        }
    if not scores:
        return {
            "status": "not_applicable",
            "fact_count": 0,
            "mean_support": None,
            "support_rate": None,
            "per_fact": {},
        }
    values = np.asarray(scores, dtype=np.float64)
    return {
        "status": "scored",
        "fact_count": len(scores),
        "mean_support": round(float(values.mean()), 8),
        "support_rate": round(float(np.mean(values >= operating_point)), 8),
        "per_fact": per_fact,
    }


def _bootstrap_mean_ci(
    values: Sequence[float], *, repetitions: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return [round(float(low), 8), round(float(high), 8)]


def _summary_stats(
    values: Sequence[float], *, repetitions: int, seed: int
) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "bootstrap_mean_95ci": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
        "bootstrap_mean_95ci": _bootstrap_mean_ci(
            list(array), repetitions=repetitions, seed=seed
        ),
    }


def _safe_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return round(float(np.corrcoef(left, right)[0, 1]), 8)


def pseudo_label_auroc(
    real_scores: Sequence[float],
    generated_scores: Sequence[float],
    *,
    operating_point: float,
) -> float | None:
    """AUROC against binarized real-XRV predictions, not clinical ground truth."""
    from sklearn.metrics import roc_auc_score

    target = np.asarray(real_scores, dtype=np.float64) >= operating_point
    prediction = np.asarray(generated_scores, dtype=np.float64)
    if len(np.unique(target)) != 2:
        return None
    return round(float(roc_auc_score(target.astype(np.int8), prediction)), 8)


def _quality_features(array: np.ndarray) -> dict[str, float]:
    from scipy import ndimage

    image = _resize_gray(array, 512).astype(np.float64)
    histogram = np.histogram(image, bins=256, range=(0.0, 1.0))[0].astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    sobel_x = ndimage.sobel(image, axis=1, mode="reflect")
    sobel_y = ndimage.sobel(image, axis=0, mode="reflect")
    gradient = np.hypot(sobel_x, sobel_y)
    laplacian = ndimage.laplace(image, mode="reflect")

    centered = image - image.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    height, width = spectrum.shape
    yy, xx = np.ogrid[
        -0.5:0.5:complex(0, height),
        -0.5:0.5:complex(0, width),
    ]
    radius = np.sqrt(xx * xx + yy * yy)
    total_energy = float(spectrum[radius > 0.01].sum())
    high_energy = float(spectrum[radius >= 0.25].sum())
    high_frequency_ratio = high_energy / total_energy if total_energy > 0 else 0.0

    return {
        "intensity_std": float(np.std(image)),
        "clipped_fraction": float(np.mean((image <= 1.0 / 255.0) | (image >= 254.0 / 255.0))),
        "entropy_bits": entropy,
        "gradient_p95": float(np.percentile(gradient, 95.0)),
        "laplacian_variance": float(np.var(laplacian)),
        "high_frequency_ratio": high_frequency_ratio,
    }


def robust_quality_flags(
    real_features: Sequence[Mapping[str, float]],
    generated_features: Sequence[Mapping[str, float]],
    *,
    z_threshold: float,
    extreme_z_threshold: float,
) -> list[dict[str, Any]]:
    if not real_features or len(real_features) != len(generated_features):
        raise ValueError("quality feature cohorts are invalid")
    names = tuple(sorted(real_features[0]))
    references: dict[str, tuple[float, float]] = {}
    for name in names:
        values = np.asarray([row[name] for row in real_features], dtype=np.float64)
        median = float(np.median(values))
        mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
        q25, q75 = np.percentile(values, [25.0, 75.0])
        iqr_scale = float(q75 - q25) / 1.349
        scale = max(mad_scale, iqr_scale, 1e-6)
        references[name] = (median, scale)

    results: list[dict[str, Any]] = []
    for row in generated_features:
        z_scores = {
            name: abs(float(row[name]) - references[name][0]) / references[name][1]
            for name in names
        }
        high_count = sum(value >= z_threshold for value in z_scores.values())
        hard_failure = (
            row["intensity_std"] < 0.03
            or row["clipped_fraction"] > 0.60
            or row["entropy_bits"] < 3.0
        )
        flagged = bool(
            hard_failure
            or high_count >= 2
            or max(z_scores.values()) >= extreme_z_threshold
        )
        results.append(
            {
                "flagged": flagged,
                "hard_failure": bool(hard_failure),
                "outlying_feature_count": int(high_count),
                "max_abs_robust_z": round(float(max(z_scores.values())), 8),
                "features": {name: round(float(row[name]), 8) for name in names},
                "abs_robust_z": {
                    name: round(float(z_scores[name]), 8) for name in names
                },
            }
        )
    return results


class FrozenClinicalModels:
    def __init__(self, config: Mapping[str, Any], device: Any) -> None:
        import torch
        import torch.nn as nn
        import torchvision
        import torchxrayvision as xrv

        self.torch = torch
        self.torchvision = torchvision
        self.xrv = xrv
        self.device = device
        model_config = config["models"]

        disease_checkpoint = Path(model_config["xrv_disease_checkpoint"]).resolve(strict=True)
        self.disease_model = xrv.models.DenseNet(
            weights=str(model_config["xrv_disease_name"]),
            cache_dir=str(disease_checkpoint.parent),
        ).to(device)
        self.disease_model.eval().requires_grad_(False)
        self.pathologies = [
            _normalized_label(name) for name in self.disease_model.pathologies
        ]
        self.crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(224)

        sex_checkpoint = Path(model_config["sex_checkpoint"]).resolve(strict=True)
        sex_model = torchvision.models.resnet34(weights=None)
        sex_model.fc = nn.Linear(sex_model.fc.in_features, 2)
        checkpoint = torch.load(sex_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        cleaned = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in state.items()
        }
        sex_model.load_state_dict(cleaned, strict=True)
        self.sex_model = sex_model.to(device).eval().requires_grad_(False)

        age_checkpoint = Path(model_config["age_checkpoint"]).resolve(strict=True)
        self.age_model = torch.jit.load(str(age_checkpoint), map_location=device).eval()
        self.age_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.age_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.checkpoints = {
            "xrv_disease": sha256_file(disease_checkpoint),
            "sex_resnet34": sha256_file(sex_checkpoint),
            "age_senet154": sha256_file(age_checkpoint),
        }

    def _xrv_array(self, path: Path) -> np.ndarray:
        array = _load_grayscale(path) * 255.0
        return self.xrv.datasets.normalize(array.astype(np.float32), 255)

    def predict(self, paths: Sequence[Path], batch_size: int) -> list[dict[str, Any]]:
        import torch.nn.functional as functional

        disease_inputs: list[Any] = []
        sex_inputs: list[Any] = []
        age_inputs: list[Any] = []
        for path in paths:
            normalized = self._xrv_array(path)
            disease = self.crop(normalized[None, :, :])
            disease = self.resize(disease)
            disease_inputs.append(self.torch.from_numpy(disease))

            raw_array = _load_grayscale(path)
            raw_low = float(raw_array.min())
            raw_high = float(raw_array.max())
            if raw_high <= raw_low:
                raise ValueError("constant image is invalid for sex classification")
            raw_array = (raw_array - raw_low) / (raw_high - raw_low)
            raw = self.torch.from_numpy(_resize_gray(raw_array, 224))
            sex_inputs.append(raw.unsqueeze(0).repeat(3, 1, 1) * 255.0)

            age = self.torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
            age = functional.interpolate(
                age, size=(320, 320), mode="bilinear", align_corners=False
            ).repeat(1, 3, 1, 1)
            age = (age + 1024.0) / 2048.0
            age = (age - self.age_mean) / self.age_std
            age_inputs.append(age[0])

        results: list[dict[str, Any]] = []
        with self.torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                stop = min(start + batch_size, len(paths))
                disease_batch = self.torch.stack(disease_inputs[start:stop]).to(self.device)
                sex_batch = self.torch.stack(sex_inputs[start:stop]).to(self.device)
                age_batch = self.torch.stack(age_inputs[start:stop]).to(self.device)
                disease_output = self.disease_model(disease_batch).detach().float().cpu().numpy()
                sex_output = self.torch.softmax(self.sex_model(sex_batch), dim=1).detach().float().cpu().numpy()
                age_output = self.age_model(age_batch).detach().float().cpu().numpy().reshape(-1)
                for disease_row, sex_row, age_value in zip(
                    disease_output, sex_output, age_output, strict=True
                ):
                    confidences = {
                        label: float(value)
                        for label, value in zip(self.pathologies, disease_row, strict=True)
                    }
                    results.append(
                        {
                            "disease": confidences,
                            "sex_prediction": "female" if int(np.argmax(sex_row)) == 1 else "male",
                            "sex_confidence": float(np.max(sex_row)),
                            "predicted_age": float(age_value),
                        }
                    )
        if len(results) != len(paths):
            raise RuntimeError("clinical prediction count mismatch")
        return results


class FrozenBioViL:
    def __init__(self, checkpoint: Path, device: Any) -> None:
        from health_multimodal.image.model.model import ImageModel
        from health_multimodal.image.model.types import ImageEncoderType

        self.torch = __import__("torch")
        self.device = device
        self.model = ImageModel(
            img_encoder_type=ImageEncoderType.RESNET50,
            joint_feature_size=128,
            pretrained_model_path=checkpoint,
        ).to(device)
        self.model.eval().requires_grad_(False)
        self.checkpoint_sha256 = sha256_file(checkpoint)

    def embed(self, paths: Sequence[Path], batch_size: int) -> np.ndarray:
        import torch.nn.functional as functional
        from health_multimodal.image.data.transforms import (
            create_chest_xray_transform_for_inference,
        )
        from PIL import Image

        # Match hi-ml-multimodal's released BioViL inference helper exactly:
        # resize the shorter side to 512, center-crop to 480, convert to a
        # tensor, and repeat the grayscale channel three times.
        transform = create_chest_xray_transform_for_inference(
            resize=512,
            center_crop_size=480,
        )
        tensors: list[Any] = []
        for path in paths:
            array = _load_grayscale(path)
            low = float(array.min())
            high = float(array.max())
            if high <= low:
                raise ValueError("constant image is invalid for BioViL")
            remapped = np.rint((array - low) / (high - low) * 255.0).astype(np.uint8)
            image = Image.fromarray(remapped, mode="L")
            tensors.append(transform(image))

        batches: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = self.torch.stack(tensors[start : start + batch_size]).to(self.device)
                output = self.model(batch).projected_global_embedding
                output = functional.normalize(output, dim=-1)
                batches.append(output.detach().float().cpu().numpy())
        result = np.concatenate(batches, axis=0)
        if result.shape != (len(paths), 128) or not np.isfinite(result).all():
            raise RuntimeError("invalid BioViL embedding matrix")
        return result


def _paired_ms_ssim(
    real_paths: Sequence[Path], generated_paths: Sequence[Path], device: Any
) -> list[float]:
    import torch
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

    scores: list[float] = []
    with torch.inference_mode():
        for real_path, generated_path in zip(real_paths, generated_paths, strict=True):
            real = torch.from_numpy(_resize_gray(_load_grayscale(real_path), 512)).view(1, 1, 512, 512).to(device)
            generated = torch.from_numpy(_resize_gray(_load_grayscale(generated_path), 512)).view(1, 1, 512, 512).to(device)
            score = multiscale_structural_similarity_index_measure(
                generated,
                real,
                data_range=1.0,
            )
            scores.append(float(score.detach().cpu()))
    return scores


def _fid_score(
    real_paths: Sequence[Path],
    generated_paths: Sequence[Path],
    *,
    device: Any,
    batch_size: int,
    num_workers: int,
) -> float:
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import fid_from_feats, get_files_features

    mode = "legacy_pytorch"
    model = build_feature_extractor(mode, device=device, use_dataparallel=False)
    try:
        real_features = get_files_features(
            list(real_paths),
            model=model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            verbose=False,
        )
        generated_features = get_files_features(
            list(generated_paths),
            model=model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            verbose=False,
        )
    finally:
        del model
    return float(fid_from_feats(generated_features, real_features))


def _checkpoint_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        "xrv_disease": Path(config["models"]["xrv_disease_checkpoint"]),
        "sex_resnet34": Path(config["models"]["sex_checkpoint"]),
        "age_senet154": Path(config["models"]["age_checkpoint"]),
        "biovil": Path(config["models"]["biovil_checkpoint"]),
    }
    result: dict[str, Any] = {}
    for name, raw_path in files.items():
        path = raw_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("configured evaluation checkpoint is not a file")
        result[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def evaluate(
    *,
    config_path: str | Path,
    generation_run_id: str,
    evaluation_run_id: str,
) -> dict[str, Any]:
    import torch

    os.umask(0o077)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run evaluation through Slurm")
    validate_run_id(generation_run_id)
    validate_run_id(evaluation_run_id)
    config = _load_config(config_path)
    audit = _checkpoint_audit(config)
    specs, source_run_id = _collect_cases(config, generation_run_id)

    evaluation_root = create_private_stage_dir(PROTECTED_EVAL_ROOT / evaluation_run_id)
    log_path = evaluation_root / "evaluation.log"
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    with log_path.open("x", encoding="utf-8") as protected_log:
        with contextlib.redirect_stdout(protected_log), contextlib.redirect_stderr(protected_log):
            real_paths = [spec.real_cxr for spec in specs]
            generated_paths = [spec.generated_cxr for spec in specs]
            batch_size = int(config["evaluation"]["batch_size"])
            operating_point = float(config["evaluation"]["operating_point"])
            repetitions = int(config["evaluation"]["bootstrap_repetitions"])
            bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])

            clinical = FrozenClinicalModels(config, device)
            real_predictions = clinical.predict(real_paths, batch_size)
            generated_predictions = clinical.predict(generated_paths, batch_size)

            biovil_checkpoint = Path(config["models"]["biovil_checkpoint"]).resolve(strict=True)
            biovil = FrozenBioViL(biovil_checkpoint, device)
            real_embeddings = biovil.embed(real_paths, batch_size)
            generated_embeddings = biovil.embed(generated_paths, batch_size)
            biovil_cosines = np.sum(real_embeddings * generated_embeddings, axis=1)
            del biovil
            torch.cuda.empty_cache()

            ms_ssim = _paired_ms_ssim(real_paths, generated_paths, device)
            fid = _fid_score(
                real_paths,
                generated_paths,
                device=device,
                batch_size=int(config["evaluation"]["fid_batch_size"]),
                num_workers=int(config["evaluation"]["num_workers"]),
            )

            real_quality = [_quality_features(_load_grayscale(path)) for path in real_paths]
            generated_quality = [
                _quality_features(_load_grayscale(path)) for path in generated_paths
            ]
            quality_flags = robust_quality_flags(
                real_quality,
                generated_quality,
                z_threshold=float(config["quality"]["z_threshold"]),
                extreme_z_threshold=float(config["quality"]["extreme_z_threshold"]),
            )

            case_rows: list[dict[str, Any]] = []
            for index, spec in enumerate(specs):
                real_prediction = real_predictions[index]
                generated_prediction = generated_predictions[index]
                observable = spec.facts.get("cxr_observable_targets")
                if not isinstance(observable, dict):
                    raise TypeError("protected EHR observable targets must be an object")
                real_support = _ehr_support(
                    observable,
                    real_prediction["disease"],
                    operating_point=operating_point,
                )
                generated_support = _ehr_support(
                    observable,
                    generated_prediction["disease"],
                    operating_point=operating_point,
                )
                support_gap = None
                if real_support["status"] == "scored":
                    support_gap = round(
                        float(generated_support["mean_support"])
                        - float(real_support["mean_support"]),
                        8,
                    )

                target_sex = str(spec.facts.get("sex", "unspecified-sex"))
                sex_scored = target_sex in {"female", "male"}
                target_age_group = str(spec.facts.get("age_group", "adult"))
                real_age_match = age_group_match(
                    target_age_group, float(real_prediction["predicted_age"])
                )
                generated_age_match = age_group_match(
                    target_age_group, float(generated_prediction["predicted_age"])
                )
                official_scores = {
                    label: {
                        "real": round(float(real_prediction["disease"][label]), 8),
                        "generated": round(
                            float(generated_prediction["disease"][label]), 8
                        ),
                    }
                    for label in OFFICIAL_FIVE
                }
                case_rows.append(
                    {
                        "case_id": spec.case_id,
                        "ehr_positive_support": {
                            "real": real_support,
                            "generated": generated_support,
                            "generated_minus_real": support_gap,
                        },
                        "official_five_xrv": official_scores,
                        "demographics": {
                            "sex_scored": sex_scored,
                            "real_sex_match": (
                                real_prediction["sex_prediction"] == target_sex
                                if sex_scored
                                else None
                            ),
                            "generated_sex_match": (
                                generated_prediction["sex_prediction"] == target_sex
                                if sex_scored
                                else None
                            ),
                            "real_age_group_match": real_age_match,
                            "generated_age_group_match": generated_age_match,
                            "real_age_distance_to_group": age_distance_to_group(
                                target_age_group,
                                float(real_prediction["predicted_age"]),
                            ),
                            "generated_age_distance_to_group": age_distance_to_group(
                                target_age_group,
                                float(generated_prediction["predicted_age"]),
                            ),
                        },
                        "paired_similarity": {
                            "biovil_cosine": round(float(biovil_cosines[index]), 8),
                            "ms_ssim": round(float(ms_ssim[index]), 8),
                            "ms_ssim_below_official_failure_threshold": bool(
                                ms_ssim[index]
                                < float(config["evaluation"]["ms_ssim_failure_threshold"])
                            ),
                        },
                        "quality": quality_flags[index],
                        "generated_image_sha256": sha256_file(spec.generated_cxr),
                    }
                )

            scored_support = [
                row for row in case_rows if row["ehr_positive_support"]["real"]["status"] == "scored"
            ]
            real_support_values = [
                float(row["ehr_positive_support"]["real"]["mean_support"])
                for row in scored_support
            ]
            generated_support_values = [
                float(row["ehr_positive_support"]["generated"]["mean_support"])
                for row in scored_support
            ]
            support_gaps = [
                float(row["ehr_positive_support"]["generated_minus_real"])
                for row in scored_support
            ]

            per_label: dict[str, Any] = {}
            flattened_real: list[float] = []
            flattened_generated: list[float] = []
            for label in OFFICIAL_FIVE:
                real_values = [
                    float(row["official_five_xrv"][label]["real"])
                    for row in case_rows
                ]
                generated_values = [
                    float(row["official_five_xrv"][label]["generated"])
                    for row in case_rows
                ]
                flattened_real.extend(real_values)
                flattened_generated.extend(generated_values)
                per_label[label] = {
                    "paired_probability_mae": round(
                        float(np.mean(np.abs(np.asarray(real_values) - np.asarray(generated_values)))),
                        8,
                    ),
                    "paired_probability_correlation": _safe_correlation(
                        real_values, generated_values
                    ),
                    "binary_agreement_at_operating_point": round(
                        float(
                            np.mean(
                                (np.asarray(real_values) >= operating_point)
                                == (np.asarray(generated_values) >= operating_point)
                            )
                        ),
                        8,
                    ),
                    "real_xrv_pseudo_label_auroc": pseudo_label_auroc(
                        real_values,
                        generated_values,
                        operating_point=operating_point,
                    ),
                }
            available_aurocs = [
                float(value["real_xrv_pseudo_label_auroc"])
                for value in per_label.values()
                if value["real_xrv_pseudo_label_auroc"] is not None
            ]

            def _accuracy(field: str) -> dict[str, Any]:
                values = [
                    bool(row["demographics"][field])
                    for row in case_rows
                    if row["demographics"][field] is not None
                ]
                return {
                    "count": len(values),
                    "accuracy": round(float(np.mean(values)), 8) if values else None,
                }

            below_threshold = sum(
                bool(row["paired_similarity"]["ms_ssim_below_official_failure_threshold"])
                for row in case_rows
            )
            quality_flag_count = sum(bool(row["quality"]["flagged"]) for row in case_rows)

            case_metrics_path = write_private_json(
                evaluation_root / "case_metrics.json",
                {
                    "schema_version": CASE_METRICS_SCHEMA,
                    "evaluation_run_id": evaluation_run_id,
                    "generation_run_id": generation_run_id,
                    "cases": case_rows,
                },
            )

            summary_path = write_private_json(
                evaluation_root / "summary.json",
                {
                    "schema_version": EVAL_SCHEMA,
                    "evaluation_run_id": evaluation_run_id,
                    "generation_run_id": generation_run_id,
                    "source_run_id": source_run_id,
                    "sample_count": len(specs),
                    "scientific_scope": {
                        "path": (
                            "structured EHR -> deterministic radiology-style prompt -> "
                            "frozen RoentGen-v2 -> synthetic CXR"
                        ),
                        "direct_ehr_to_cxr": False,
                        "real_anchor": True,
                        "loads_real_cxr_internally": True,
                        "copies_or_serializes_real_cxr": False,
                        "loads_real_report": False,
                    },
                    "disease_alignment": {
                        "ehr_absence_semantics": "unknown_not_negative",
                        "official_prompt_auroc": (
                            "not_estimable_without_explicit_negative_prompt labels"
                        ),
                        "ehr_positive_support_real": _summary_stats(
                            real_support_values,
                            repetitions=repetitions,
                            seed=bootstrap_seed,
                        ),
                        "ehr_positive_support_generated": _summary_stats(
                            generated_support_values,
                            repetitions=repetitions,
                            seed=bootstrap_seed + 1,
                        ),
                        "generated_minus_real_support": _summary_stats(
                            support_gaps,
                            repetitions=repetitions,
                            seed=bootstrap_seed + 2,
                        ),
                        "official_five": {
                            "labels": list(OFFICIAL_FIVE),
                            "per_label": per_label,
                            "macro_real_xrv_pseudo_label_auroc": (
                                round(float(np.mean(available_aurocs)), 8)
                                if available_aurocs
                                else None
                            ),
                            "pooled_probability_mae": round(
                                float(
                                    np.mean(
                                        np.abs(
                                            np.asarray(flattened_real)
                                            - np.asarray(flattened_generated)
                                        )
                                    )
                                ),
                                8,
                            ),
                            "pooled_probability_correlation": _safe_correlation(
                                flattened_real, flattened_generated
                            ),
                            "warning": (
                                "real-XRV pseudo labels are scorer-derived anchors, "
                                "not radiologist ground truth"
                            ),
                        },
                    },
                    "demographic_alignment": {
                        "sex_real": _accuracy("real_sex_match"),
                        "sex_generated": _accuracy("generated_sex_match"),
                        "age_group_real": _accuracy("real_age_group_match"),
                        "age_group_generated": _accuracy("generated_age_group_match"),
                        "age_metric_note": (
                            "age-bin accuracy is used because the EHR prompt stores an age group, "
                            "not exact age; official age RMSE is therefore not claimed"
                        ),
                        "race_metric": "not_applicable_race_not_used_in_prompt",
                    },
                    "real_synthetic_similarity": {
                        "biovil_cosine": _summary_stats(
                            list(map(float, biovil_cosines)),
                            repetitions=repetitions,
                            seed=bootstrap_seed + 3,
                        ),
                        "ms_ssim": _summary_stats(
                            list(map(float, ms_ssim)),
                            repetitions=repetitions,
                            seed=bootstrap_seed + 4,
                        ),
                        "ms_ssim_failure_threshold": float(
                            config["evaluation"]["ms_ssim_failure_threshold"]
                        ),
                        "ms_ssim_below_threshold_count": int(below_threshold),
                        "ms_ssim_below_threshold_rate": round(
                            float(below_threshold / len(specs)), 8
                        ),
                    },
                    "fid": {
                        "value": round(float(fid), 8),
                        "protocol": "clean-fid legacy_pytorch Inception-v3 features",
                        "direction": "lower_is_better",
                        "sample_count_per_set": len(specs),
                        "scope": "exploratory_small_sample_not_paper_comparable",
                        "paper_reference_roentgen_v2": 76.8,
                    },
                    "quality_gate": {
                        "protocol": (
                            "deterministic image-statistic outlier gate calibrated on the "
                            "matched real cohort; exploratory and not clinically validated"
                        ),
                        "flagged_count": int(quality_flag_count),
                        "flagged_rate": round(float(quality_flag_count / len(specs)), 8),
                        "z_threshold": float(config["quality"]["z_threshold"]),
                        "extreme_z_threshold": float(
                            config["quality"]["extreme_z_threshold"]
                        ),
                    },
                    "checkpoints": audit,
                    "software": {
                        "torch": torch.__version__,
                        "torchvision": importlib.metadata.version("torchvision"),
                        "torchxrayvision": importlib.metadata.version("torchxrayvision"),
                        "clean_fid": importlib.metadata.version("clean-fid"),
                        "torchmetrics": importlib.metadata.version("torchmetrics"),
                        "hi_ml_multimodal": importlib.metadata.version(
                            "hi-ml-multimodal"
                        ),
                    },
                    "case_metrics_sha256": sha256_file(case_metrics_path),
                    "runtime_seconds": round(time.perf_counter() - started, 4),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                },
            )

    enforce_private_file_mode(log_path)
    return {
        "stage": "roentgen_v2_evaluation",
        "status": "ok",
        "sample_count": len(specs),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "summary_sha256": sha256_file(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--generation-run-id", required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    args = parser.parse_args()
    try:
        result = evaluate(
            config_path=args.config,
            generation_run_id=args.generation_run_id,
            evaluation_run_id=args.evaluation_run_id,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "roentgen_v2_evaluation",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

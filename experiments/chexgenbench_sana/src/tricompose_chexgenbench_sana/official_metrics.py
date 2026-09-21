"""Privacy-preserving CheXGenBench official metrics for a frozen Sana run.

The upstream metric encoders, preprocessing, and formulas are preserved. This
wrapper fixes upstream CLI defects and never copies, serializes, or logs a real
patient CXR. GPU execution is allowed only through a user-approved Slurm job.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)

from .common import (
    PROTECTED_EXPERIMENT_ROOT,
    read_private_json,
    require_source_run,
    validate_case_id,
    validate_run_id,
)
from .metric_prefetch import (
    BIOVIL_T_IMAGE_WEIGHT,
    BIOVIL_T_REVISION,
    FREEZE_SCHEMA,
    OFFICIAL_REPO_COMMIT,
    RAD_DINO_REVISION,
)


CONFIG_SCHEMA = "tricompose.chexgenbench_sana.official_metrics_config.v1"
RUN_SCHEMA = "tricompose.chexgenbench_sana.run.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
CASE_INPUT_SCHEMA = "tricompose.chexgenbench_sana.case_input.v1"
GENERATION_SCHEMA = "tricompose.chexgenbench_sana.generation.v1"
FROZEN_RUN_SCHEMA = "tricompose.chexgenbench_sana.frozen_run.v1"
EVALUATION_SCHEMA = "tricompose.chexgenbench_sana.official_metrics.v1"
CASE_SCORE_SCHEMA = "tricompose.chexgenbench_sana.biovil_case_scores.v1"
PROTECTED_EVALUATION_ROOT = PROTECTED_ROOT / "chexgenbench_sana" / "evaluations"
METRIC_SEED = 42


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    prompt: str
    prompt_sha256: str
    real_cxr: Path
    synthetic_cxr: Path


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    resolved = Path(path).resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported official metric configuration")
    return payload


def _load_real_cxr_index(dataset_root: Path) -> Any:
    """Read only the real-path column; its values stay in process memory."""
    import pandas as pd

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


def _require_generation_run(run_id: str) -> Path:
    validate_run_id(run_id)
    run = require_inside(
        PROTECTED_EXPERIMENT_ROOT / run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    if not run.is_dir():
        raise ValueError("protected Sana run is not a directory")
    return run


def _validate_frozen_run(run: Path, expected_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_private_json(run / "manifest.json")
    summary = read_private_json(run / "summary.json")
    frozen = read_private_json(run / "frozen.json")
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unsupported Sana run manifest")
    if frozen.get("schema_version") != FROZEN_RUN_SCHEMA or frozen.get("status") != "frozen":
        raise ValueError("Sana run is not frozen")
    if summary.get("status") != "completed":
        raise ValueError("Sana run is incomplete")
    for payload in (manifest, summary, frozen):
        if payload.get("case_count") != expected_count:
            raise ValueError("Sana run does not contain the fixed case count")
    if manifest.get("run_id") != run.name or frozen.get("run_id") != run.name:
        raise ValueError("Sana run identity mismatch")
    if frozen.get("manifest_sha256") != sha256_file(run / "manifest.json"):
        raise ValueError("frozen Sana manifest hash mismatch")
    if frozen.get("summary_sha256") != sha256_file(run / "summary.json"):
        raise ValueError("frozen Sana summary hash mismatch")
    model_audit = frozen.get("model_snapshot_audit")
    if not isinstance(model_audit, dict) or not model_audit.get("weight_sha256"):
        raise ValueError("Sana frozen model audit is unavailable")
    return manifest, frozen


def _collect_cases(config: Mapping[str, Any], run_id: str) -> tuple[list[CaseSpec], str]:
    expected_count = int(config["evaluation"]["expected_case_count"])
    run = _require_generation_run(run_id)
    manifest, frozen = _validate_frozen_run(run, expected_count)
    source_run_id = validate_run_id(str(manifest["source_run_id"]))
    source_run = require_source_run(source_run_id)
    dataset_root = Path(config["dataset"]["root"]).resolve(strict=True)
    real_index = _load_real_cxr_index(dataset_root)

    manifest_rows = manifest.get("cases")
    frozen_rows = frozen.get("cases")
    if not isinstance(manifest_rows, list) or not isinstance(frozen_rows, list):
        raise ValueError("Sana fixed case lists are unavailable")
    if len(manifest_rows) != expected_count or len(frozen_rows) != expected_count:
        raise ValueError("Sana fixed case lists have the wrong length")

    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for manifest_row, frozen_row in zip(manifest_rows, frozen_rows, strict=True):
        case_id = validate_case_id(str(manifest_row["case_id"]))
        if case_id in seen or frozen_row.get("case_id") != case_id:
            raise ValueError("duplicate or reordered opaque case ID")
        seen.add(case_id)

        source_case = read_private_json(source_run / "cases" / case_id / "input.json")
        sana_case = read_private_json(run / "cases" / case_id / "input.json")
        generation_dir = run / "cases" / case_id / "generation"
        generation_path = generation_dir / "generation.json"
        generation = read_private_json(generation_path)
        image_path = require_private_file(generation_dir / "generated_cxr.png")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case")
        if sana_case.get("schema_version") != CASE_INPUT_SCHEMA:
            raise ValueError("unsupported protected Sana case")
        if generation.get("schema_version") != GENERATION_SCHEMA:
            raise ValueError("unsupported protected Sana generation")

        prompt = str(sana_case["sana_prompt"])
        prompt_hash = _hash_text(prompt)
        if prompt_hash != sana_case.get("prompt_sha256"):
            raise ValueError("Sana prompt content hash mismatch")
        if prompt_hash != manifest_row.get("prompt_sha256"):
            raise ValueError("Sana manifest prompt hash mismatch")
        if prompt_hash != frozen_row.get("prompt_sha256"):
            raise ValueError("Sana frozen prompt hash mismatch")
        if prompt_hash != generation.get("prompt_sha256"):
            raise ValueError("Sana generation prompt hash mismatch")
        if sha256_file(generation_path) != frozen_row.get("generation_sha256"):
            raise ValueError("frozen generation record hash mismatch")
        image_hash = sha256_file(image_path)
        if image_hash != generation.get("output", {}).get("image_sha256"):
            raise ValueError("generated image record hash mismatch")
        if image_hash != frozen_row.get("image_sha256"):
            raise ValueError("frozen generated image hash mismatch")
        if int(sana_case["seed"]) != int(frozen_row["seed"]):
            raise ValueError("frozen Sana seed mismatch")

        cases.append(
            CaseSpec(
                case_id=case_id,
                prompt=prompt,
                prompt_sha256=prompt_hash,
                real_cxr=_resolve_real_cxr(real_index, int(source_case["source_row_index"])),
                synthetic_cxr=image_path,
            )
        )
    return cases, sha256_file(run / "frozen.json")


def _validate_file_records(root: Path, records: Any) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("frozen scorer file list is unavailable")
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid frozen scorer relative path")
        path = (root / relative).resolve(strict=True)
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError("frozen scorer file size mismatch")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("frozen scorer file hash mismatch")


def _validate_scorer_freeze(root: Path, workspace: Path) -> tuple[dict[str, Path], str]:
    freeze_path = root / "freeze.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FREEZE_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("official scorer bundle is not frozen")
    if payload.get("official_repo_commit") != OFFICIAL_REPO_COMMIT:
        raise ValueError("official scorer source commit mismatch")
    scorers = payload.get("scorers")
    if not isinstance(scorers, dict):
        raise ValueError("frozen scorer records are unavailable")

    rad = scorers["rad_dino"]
    bio = scorers["biovil_t"]
    if rad.get("revision") != RAD_DINO_REVISION:
        raise ValueError("RadDINO revision mismatch")
    if bio.get("revision") != BIOVIL_T_REVISION:
        raise ValueError("BioViL-T revision mismatch")
    rad_root = root / "rad-dino" / RAD_DINO_REVISION
    bio_root = root / "biovil-t" / BIOVIL_T_REVISION
    _validate_file_records(rad_root, rad.get("files"))
    _validate_file_records(bio_root, bio.get("files"))

    inception_records = scorers.get("inception", {}).get("files")
    if not isinstance(inception_records, list) or not inception_records:
        raise ValueError("frozen Inception file list is unavailable")
    for record in inception_records:
        path = require_inside(record["path"], workspace, must_exist=True)
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError("frozen Inception file size mismatch")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("frozen Inception file hash mismatch")
    return {"rad_dino": rad_root, "biovil_t": bio_root}, sha256_file(freeze_path)


class _ImageDataset:
    """Factory wrapper that avoids importing torchvision on the login node."""

    @staticmethod
    def build(paths: Sequence[Path]) -> Any:
        from PIL import Image
        from torch.utils.data import Dataset
        from torchvision import transforms

        class DatasetImpl(Dataset[Any]):
            def __init__(self, image_paths: Sequence[Path]) -> None:
                self.paths = list(image_paths)
                self.transform = transforms.Compose(
                    [transforms.Resize((299, 299)), transforms.ToTensor()]
                )

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, index: int) -> Any:
                with Image.open(self.paths[index]) as image:
                    return self.transform(image.convert("RGB"))

        return DatasetImpl(paths)


def _distribution_metrics(
    real_paths: Sequence[Path],
    synthetic_paths: Sequence[Path],
    *,
    rad_dino_path: Path,
    batch_size: int,
    num_workers: int,
    device: Any,
) -> dict[str, float]:
    import torch
    from prdc import compute_prdc
    from torch.utils.data import DataLoader
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.kid import KernelInceptionDistance
    from transformers import AutoImageProcessor, AutoModel

    class RadDinoFeatureExtractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = AutoModel.from_pretrained(rad_dino_path, local_files_only=True)
            self.processor = AutoImageProcessor.from_pretrained(
                rad_dino_path, local_files_only=True
            )
            self.model.eval().requires_grad_(False)

        def forward(self, images: Any) -> Any:
            cpu_images = images.detach().cpu()
            inputs = self.processor(images=cpu_images, return_tensors="pt").to(device)
            with torch.inference_mode():
                return self.model(**inputs).pooler_output

    rad_dino = RadDinoFeatureExtractor().to(device)
    rad_dino.eval().requires_grad_(False)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid_rad = FrechetInceptionDistance(feature=rad_dino).to(device)
    subset_size = min(len(real_paths), len(synthetic_paths))
    kid = KernelInceptionDistance(subset_size=subset_size, feature=2048).to(device)
    kid_rad = KernelInceptionDistance(subset_size=subset_size, feature=rad_dino).to(device)
    synthetic_is = InceptionScore(feature=2048).to(device)
    real_is = InceptionScore(feature=2048).to(device)
    metric_modules = (fid, fid_rad, kid, kid_rad, synthetic_is, real_is)
    for metric in metric_modules:
        metric.eval().requires_grad_(False)

    real_loader = DataLoader(
        _ImageDataset.build(real_paths),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    synthetic_loader = DataLoader(
        _ImageDataset.build(synthetic_paths),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    real_features: list[Any] = []
    synthetic_features: list[Any] = []
    with torch.inference_mode():
        for batch in real_loader:
            batch = (batch.to(device) * 255).to(torch.uint8)
            fid.update(batch, real=True)
            fid_rad.update(batch, real=True)
            kid.update(batch, real=True)
            kid_rad.update(batch, real=True)
            real_is.update(batch)
            real_features.append(rad_dino(batch).cpu())
        for batch in synthetic_loader:
            batch = (batch.to(device) * 255).to(torch.uint8)
            fid.update(batch, real=False)
            fid_rad.update(batch, real=False)
            kid.update(batch, real=False)
            kid_rad.update(batch, real=False)
            synthetic_is.update(batch)
            synthetic_features.append(rad_dino(batch).cpu())

    kid_mean, kid_std = kid.compute()
    kid_rad_mean, kid_rad_std = kid_rad.compute()
    synthetic_is_mean, synthetic_is_std = synthetic_is.compute()
    real_is_mean, real_is_std = real_is.compute()
    prdc = compute_prdc(
        real_features=torch.cat(real_features).numpy(),
        fake_features=torch.cat(synthetic_features).numpy(),
        nearest_k=5,
    )
    return {
        "fid": float(fid.compute().item()),
        "fid_rad_dino": float(fid_rad.compute().item()),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "kid_rad_dino_mean": float(kid_rad_mean.item()),
        "kid_rad_dino_std": float(kid_rad_std.item()),
        "inception_score_synthetic_mean": float(synthetic_is_mean.item()),
        "inception_score_synthetic_std": float(synthetic_is_std.item()),
        "inception_score_real_mean": float(real_is_mean.item()),
        "inception_score_real_std": float(real_is_std.item()),
        "precision_rad_dino": float(prdc["precision"]),
        "recall_rad_dino": float(prdc["recall"]),
        "density_rad_dino": float(prdc["density"]),
        "coverage_rad_dino": float(prdc["coverage"]),
    }


def _biovil_metrics(cases: Sequence[CaseSpec], model_path: Path, device: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    import torch
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
    from health_multimodal.image.inference_engine import ImageInferenceEngine
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    from health_multimodal.text.inference_engine import TextInferenceEngine
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer
    from health_multimodal.vlp import ImageTextInferenceEngine

    tokenizer = CXRBertTokenizer.from_pretrained(model_path, local_files_only=True)
    text_model = CXRBertModel.from_pretrained(model_path, local_files_only=True)
    text_model.eval().requires_grad_(False)
    text_engine = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)
    text_engine.model.to(device)

    image_model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=128,
        pretrained_model_path=model_path / BIOVIL_T_IMAGE_WEIGHT,
    )
    image_model.eval().requires_grad_(False)
    transform = create_chest_xray_transform_for_inference(
        resize=512,
        center_crop_size=448,
    )
    image_engine = ImageInferenceEngine(image_model=image_model, transform=transform)
    image_engine.model.to(device)
    engine = ImageTextInferenceEngine(image_engine, text_engine)
    if image_engine.model.training or text_engine.model.training:
        raise RuntimeError("BioViL-T scorer entered training mode")
    if any(parameter.requires_grad for parameter in image_engine.model.parameters()):
        raise RuntimeError("BioViL-T image encoder is not frozen")
    if any(parameter.requires_grad for parameter in text_engine.model.parameters()):
        raise RuntimeError("BioViL-T text encoder is not frozen")

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in cases:
            real_score = engine.get_similarity_score_from_raw_data(
                case.real_cxr, case.prompt
            )
            synthetic_score = engine.get_similarity_score_from_raw_data(
                case.synthetic_cxr, case.prompt
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "real_alignment": float(real_score),
                    "synthetic_alignment": float(synthetic_score),
                }
            )
    real = np.asarray([row["real_alignment"] for row in rows], dtype=np.float64)
    synthetic = np.asarray(
        [row["synthetic_alignment"] for row in rows], dtype=np.float64
    )
    correlation = float(np.corrcoef(real, synthetic)[0, 1])
    return (
        {
            "real_mean": float(real.mean()),
            "real_std": float(real.std()),
            "synthetic_mean": float(synthetic.mean()),
            "synthetic_std": float(synthetic.std()),
            "paired_mean_difference_synthetic_minus_real": float(
                np.mean(synthetic - real)
            ),
            "paired_correlation": correlation,
        },
        rows,
    )


def _radiomics_one(arguments: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
    opaque_name, image_path, metrics_root = arguments
    from PIL import Image

    sys.path.insert(0, metrics_root)
    from radiomics_utils import compute_slice_radiomics

    with Image.open(image_path) as image:
        image_slice = np.asarray(image.convert("L")).copy()
    mask = np.ones_like(image_slice)
    mask[0, 0] = 0
    features = compute_slice_radiomics(image_slice, mask)
    return opaque_name, dict(features)


def _frd_metric(
    real_paths: Sequence[Path],
    synthetic_paths: Sequence[Path],
    *,
    case_ids: Sequence[str],
    metrics_root: Path,
    workers: int,
) -> float:
    import pandas as pd

    real_args = [
        (case_id, str(path), str(metrics_root))
        for case_id, path in zip(case_ids, real_paths, strict=True)
    ]
    synthetic_args = [
        (case_id, str(path), str(metrics_root))
        for case_id, path in zip(case_ids, synthetic_paths, strict=True)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        real_rows = list(executor.map(_radiomics_one, real_args))
        synthetic_rows = list(executor.map(_radiomics_one, synthetic_args))
    if len(real_rows) != len(case_ids) or len(synthetic_rows) != len(case_ids):
        raise RuntimeError("FRD did not preserve the fixed cohort")

    def frame(rows: Sequence[tuple[str, dict[str, Any]]]) -> Any:
        result = pd.DataFrame([features for _, features in rows])
        result.insert(0, "img_fname", [name for name, _ in rows])
        return result

    sys.path.insert(0, str(metrics_root))
    from radiomics_utils import convert_radiomic_dfs_to_vectors
    from utils import frechet_distance

    real_features, synthetic_features = convert_radiomic_dfs_to_vectors(
        frame(real_rows),
        frame(synthetic_rows),
        match_sample_count=True,
    )
    distance = float(frechet_distance(real_features, synthetic_features))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("FRD distance is not positive and finite")
    return float(np.log(distance))


def _rounded(values: Mapping[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in values.items():
        result[key] = round(float(value), 8) if math.isfinite(float(value)) else None
    return result


def evaluate(
    *,
    config_path: str | Path,
    generation_run_id: str,
    evaluation_run_id: str,
) -> dict[str, Any]:
    import torch

    os.umask(0o077)
    validate_run_id(generation_run_id)
    validate_run_id(evaluation_run_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run official metrics through Slurm")
    config = _load_config(config_path)
    workspace = Path(config["workspace"]).resolve(strict=True)
    official_repo = Path(config["official_repo"]["root"]).resolve(strict=True)
    scorer_root = Path(config["scorers"]["root"]).resolve(strict=True)
    if config["official_repo"].get("commit") != OFFICIAL_REPO_COMMIT:
        raise ValueError("configured CheXGenBench source commit mismatch")
    if config["scorers"].get("rad_dino_revision") != RAD_DINO_REVISION:
        raise ValueError("configured RadDINO revision mismatch")
    if config["scorers"].get("biovil_t_revision") != BIOVIL_T_REVISION:
        raise ValueError("configured BioViL-T revision mismatch")
    if int(config["evaluation"].get("metric_seed", -1)) != METRIC_SEED:
        raise ValueError("configured official metric seed mismatch")
    head = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != OFFICIAL_REPO_COMMIT:
        raise ValueError("CheXGenBench source commit mismatch")
    subprocess.run(
        ["git", "-C", str(official_repo), "diff", "--quiet", "--", "metrics"],
        check=True,
    )
    scorer_paths, scorer_freeze_sha = _validate_scorer_freeze(
        scorer_root, workspace
    )
    cases, generation_freeze_sha = _collect_cases(config, generation_run_id)
    expected_count = int(config["evaluation"]["expected_case_count"])
    if len(cases) != expected_count:
        raise ValueError("fixed evaluation cohort count mismatch")

    evaluation_root = create_private_stage_dir(
        PROTECTED_EVALUATION_ROOT / evaluation_run_id
    )
    log_dir = create_private_stage_dir(evaluation_root / "logs")
    protected_log = log_dir / "metrics.log"
    started = time.perf_counter()
    torch.manual_seed(METRIC_SEED)
    np.random.seed(METRIC_SEED)
    torch.cuda.manual_seed_all(METRIC_SEED)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    metric_config = config["evaluation"]

    try:
        with protected_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                distribution = _distribution_metrics(
                    [case.real_cxr for case in cases],
                    [case.synthetic_cxr for case in cases],
                    rad_dino_path=scorer_paths["rad_dino"],
                    batch_size=int(metric_config["batch_size"]),
                    num_workers=int(metric_config["num_workers"]),
                    device=torch.device("cuda:0"),
                )
                torch.cuda.empty_cache()
                biovil, case_scores = _biovil_metrics(
                    cases,
                    scorer_paths["biovil_t"],
                    torch.device("cuda:0"),
                )
                torch.cuda.empty_cache()
                frd = _frd_metric(
                    [case.real_cxr for case in cases],
                    [case.synthetic_cxr for case in cases],
                    case_ids=[case.case_id for case in cases],
                    metrics_root=official_repo / "metrics",
                    workers=int(metric_config["frd_workers"]),
                )
        enforce_private_file_mode(protected_log)
        case_score_path = write_private_json(
            evaluation_root / "biovil_case_scores.json",
            {
                "schema_version": CASE_SCORE_SCHEMA,
                "sample_count": len(case_scores),
                "scores": [
                    {
                        "case_id": row["case_id"],
                        "real_alignment": round(row["real_alignment"], 8),
                        "synthetic_alignment": round(row["synthetic_alignment"], 8),
                    }
                    for row in case_scores
                ],
            },
        )
        source_files = {
            name: sha256_file(official_repo / "metrics" / name)
            for name in (
                "fid.py",
                "img_text_alignment_scores.py",
                "frd.py",
                "radiomics_utils.py",
                "utils.py",
            )
        }
        elapsed = time.perf_counter() - started
        summary_path = write_private_json(
            evaluation_root / "summary.json",
            {
                "schema_version": EVALUATION_SCHEMA,
                "status": "completed",
                "evaluation_run_id": evaluation_run_id,
                "generation_run_id": generation_run_id,
                "sample_count": len(cases),
                "scope": (
                    "fixed 50-case paired pilot; values are not directly comparable "
                    "to the CheXGenBench full-test leaderboard"
                ),
                "data_boundary": {
                    "real_cxr_role": "private matched reference only",
                    "real_cxr_copied_or_serialized": False,
                    "real_report_loaded": False,
                    "patient_identifier_written": False,
                },
                "metrics": {
                    "distribution_and_quality": _rounded(distribution),
                    "biovil_t_prompt_alignment": _rounded(biovil),
                    "frechet_radiomics_distance": round(frd, 8),
                },
                "protocol": {
                    "official_repo_commit": OFFICIAL_REPO_COMMIT,
                    "official_metric_source_sha256": source_files,
                    "rad_dino_revision": RAD_DINO_REVISION,
                    "biovil_t_revision": BIOVIL_T_REVISION,
                    "prdc_nearest_k": 5,
                    "metric_seed": METRIC_SEED,
                    "wrapper_note": (
                        "official encoders, preprocessing, and formulas with a "
                        "privacy-safe wrapper that fixes upstream CLI/temp-copy defects"
                    ),
                },
                "freeze": {
                    "all_models_eval_mode": True,
                    "all_model_parameters_require_grad": False,
                    "generation_freeze_sha256": generation_freeze_sha,
                    "scorer_freeze_sha256": scorer_freeze_sha,
                    "biovil_case_scores_sha256": sha256_file(case_score_path),
                },
                "software": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "torchvision",
                        "torchmetrics",
                        "torch-fidelity",
                        "transformers",
                        "PyRadiomics",
                    )
                },
                "runtime_seconds": round(elapsed, 4),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
        )
        return {
            "status": "completed",
            "sample_count": len(cases),
            "runtime_seconds": round(elapsed, 4),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "summary_sha256": sha256_file(summary_path),
        }
    except Exception as exc:
        if protected_log.exists():
            enforce_private_file_mode(protected_log)
        write_private_json(
            evaluation_root / "failure.json",
            {
                "schema_version": EVALUATION_SCHEMA,
                "status": "failed",
                "error_type": type(exc).__name__,
            },
        )
        raise


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
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

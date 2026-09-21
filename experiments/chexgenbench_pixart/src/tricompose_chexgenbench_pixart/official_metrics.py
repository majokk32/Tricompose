"""Privacy-preserving official CheXGenBench metrics for frozen PixArt-50."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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
from tricompose_chexgenbench_sana.metric_prefetch import (
    BIOVIL_T_REVISION,
    OFFICIAL_REPO_COMMIT,
    RAD_DINO_REVISION,
)
from tricompose_chexgenbench_sana.official_metrics import (
    CaseSpec,
    METRIC_SEED,
    _biovil_metrics,
    _distribution_metrics,
    _frd_metric,
    _load_real_cxr_index,
    _resolve_real_cxr,
    _rounded,
    _validate_scorer_freeze,
)

from .common import (
    PROTECTED_EXPERIMENT_ROOT,
    read_private_json,
    require_source_run,
    validate_case_id,
    validate_run_id,
)


CONFIG_SCHEMA = "tricompose.chexgenbench_pixart.official_metrics_config.v1"
RUN_SCHEMA = "tricompose.chexgenbench_pixart.run.v1"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
CASE_INPUT_SCHEMA = "tricompose.chexgenbench_pixart.case_input.v1"
GENERATION_SCHEMA = "tricompose.chexgenbench_pixart.generation.v1"
FROZEN_RUN_SCHEMA = "tricompose.chexgenbench_pixart.frozen_run.v1"
EVALUATION_SCHEMA = "tricompose.chexgenbench_pixart.official_metrics.v1"
CASE_SCORE_SCHEMA = "tricompose.chexgenbench_pixart.biovil_case_scores.v1"
PROTECTED_EVALUATION_ROOT = PROTECTED_ROOT / "chexgenbench_pixart" / "evaluations"


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported PixArt metric configuration")
    return payload


def _require_generation_run(run_id: str) -> Path:
    validate_run_id(run_id)
    run = require_inside(
        PROTECTED_EXPERIMENT_ROOT / run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    if not run.is_dir():
        raise ValueError("protected PixArt run is not a directory")
    return run


def _validate_frozen_run(run: Path, expected_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_private_json(run / "manifest.json")
    summary = read_private_json(run / "summary.json")
    frozen = read_private_json(run / "frozen.json")
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unsupported PixArt manifest")
    if frozen.get("schema_version") != FROZEN_RUN_SCHEMA or frozen.get("status") != "frozen":
        raise ValueError("PixArt run is not frozen")
    if summary.get("status") != "completed":
        raise ValueError("PixArt run is incomplete")
    for payload in (manifest, summary, frozen):
        if payload.get("case_count") != expected_count:
            raise ValueError("PixArt run does not contain the fixed case count")
    if manifest.get("run_id") != run.name or frozen.get("run_id") != run.name:
        raise ValueError("PixArt run identity mismatch")
    if frozen.get("manifest_sha256") != sha256_file(run / "manifest.json"):
        raise ValueError("frozen PixArt manifest hash mismatch")
    if frozen.get("summary_sha256") != sha256_file(run / "summary.json"):
        raise ValueError("frozen PixArt summary hash mismatch")
    model_audit = frozen.get("model_snapshot_audit")
    if not isinstance(model_audit, dict) or not model_audit.get("weight_sha256"):
        raise ValueError("PixArt frozen model audit is unavailable")
    return manifest, frozen


def _collect_cases(config: Mapping[str, Any], run_id: str) -> tuple[list[CaseSpec], str]:
    expected_count = int(config["evaluation"]["expected_case_count"])
    run = _require_generation_run(run_id)
    manifest, frozen = _validate_frozen_run(run, expected_count)
    source_run = require_source_run(validate_run_id(str(manifest["source_run_id"])))
    real_index = _load_real_cxr_index(
        Path(config["dataset"]["root"]).resolve(strict=True)
    )
    manifest_rows = manifest.get("cases")
    frozen_rows = frozen.get("cases")
    if not isinstance(manifest_rows, list) or not isinstance(frozen_rows, list):
        raise ValueError("PixArt fixed case lists are unavailable")
    if len(manifest_rows) != expected_count or len(frozen_rows) != expected_count:
        raise ValueError("PixArt fixed case lists have the wrong length")

    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for manifest_row, frozen_row in zip(manifest_rows, frozen_rows, strict=True):
        case_id = validate_case_id(str(manifest_row["case_id"]))
        if case_id in seen or frozen_row.get("case_id") != case_id:
            raise ValueError("duplicate or reordered opaque case ID")
        seen.add(case_id)
        source_case = read_private_json(source_run / "cases" / case_id / "input.json")
        pixart_case = read_private_json(run / "cases" / case_id / "input.json")
        generation_dir = run / "cases" / case_id / "generation"
        generation_path = generation_dir / "generation.json"
        generation = read_private_json(generation_path)
        image_path = require_private_file(generation_dir / "generated_cxr.png")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case")
        if pixart_case.get("schema_version") != CASE_INPUT_SCHEMA:
            raise ValueError("unsupported protected PixArt case")
        if generation.get("schema_version") != GENERATION_SCHEMA:
            raise ValueError("unsupported protected PixArt generation")
        prompt = str(pixart_case["pixart_prompt"])
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected_prompt_hashes = (
            pixart_case.get("prompt_sha256"),
            manifest_row.get("prompt_sha256"),
            frozen_row.get("prompt_sha256"),
            generation.get("prompt_sha256"),
        )
        if any(value != prompt_hash for value in expected_prompt_hashes):
            raise ValueError("PixArt prompt hash mismatch")
        if sha256_file(generation_path) != frozen_row.get("generation_sha256"):
            raise ValueError("frozen PixArt generation record hash mismatch")
        image_hash = sha256_file(image_path)
        if image_hash != generation.get("output", {}).get("image_sha256"):
            raise ValueError("PixArt image record hash mismatch")
        if image_hash != frozen_row.get("image_sha256"):
            raise ValueError("frozen PixArt image hash mismatch")
        if int(pixart_case["seed"]) != int(frozen_row["seed"]):
            raise ValueError("frozen PixArt seed mismatch")
        cases.append(
            CaseSpec(
                case_id=case_id,
                prompt=prompt,
                prompt_sha256=prompt_hash,
                real_cxr=_resolve_real_cxr(
                    real_index, int(source_case["source_row_index"])
                ),
                synthetic_cxr=image_path,
            )
        )
    return cases, sha256_file(run / "frozen.json")


def evaluate(config_path: str | Path, generation_run_id: str, evaluation_run_id: str) -> dict[str, Any]:
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
        raise ValueError("configured CheXGenBench commit mismatch")
    if config["scorers"].get("rad_dino_revision") != RAD_DINO_REVISION:
        raise ValueError("configured RadDINO revision mismatch")
    if config["scorers"].get("biovil_t_revision") != BIOVIL_T_REVISION:
        raise ValueError("configured BioViL-T revision mismatch")
    if int(config["evaluation"].get("metric_seed", -1)) != METRIC_SEED:
        raise ValueError("configured metric seed mismatch")
    head = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != OFFICIAL_REPO_COMMIT:
        raise ValueError("CheXGenBench source commit mismatch")
    subprocess.run(
        ["git", "-C", str(official_repo), "diff", "--quiet", "--", "metrics"],
        check=True,
    )
    scorer_paths, scorer_freeze_sha = _validate_scorer_freeze(scorer_root, workspace)
    cases, generation_freeze_sha = _collect_cases(config, generation_run_id)
    expected_count = int(config["evaluation"]["expected_case_count"])
    if len(cases) != expected_count:
        raise ValueError("fixed PixArt cohort count mismatch")

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
                    cases, scorer_paths["biovil_t"], torch.device("cuda:0")
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
                    "fixed 50-case paired pilot; not directly comparable to the "
                    "CheXGenBench full-test leaderboard"
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
        print(
            json.dumps(
                evaluate(args.config, args.generation_run_id, args.evaluation_run_id),
                sort_keys=True,
            ),
            flush=True,
        )
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

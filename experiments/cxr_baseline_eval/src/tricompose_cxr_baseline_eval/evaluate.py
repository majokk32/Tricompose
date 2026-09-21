"""Run a lightweight frozen-XRV evaluation on frozen CXR-generator outputs.

This is GPU inference and must run through a user-approved Slurm job. Real CXR
paths remain in process memory and are never copied, serialized, or logged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_file_mode,
    require_inside,
    require_private_file,
    sha256_file,
    write_private_json,
)
from tricompose_roentgen_v2 import candidate_selector as xrv_reference
from tricompose_roentgen_v2 import evaluation as metric_reference


CONFIG_SCHEMA = "tricompose.cxr_baseline_eval.config.v2"
EVAL_SCHEMA = "tricompose.cxr_baseline_eval.evaluation.v2"
CASE_METRICS_SCHEMA = "tricompose.cxr_baseline_eval.case_metrics.v2"
SOURCE_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
SOURCE_RUN_ROOT = PROTECTED_ROOT / "medim" / "runs"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    producer: str
    run_root: Path
    evaluation_root: Path
    run_schema: str
    case_schema: str
    generation_schema: str
    frozen_schema: str
    prompt_field: str


MODEL_SPECS = {
    "sana": ModelSpec(
        name="sana",
        producer="frozen CheXGenBench Sana",
        run_root=PROTECTED_ROOT / "chexgenbench_sana" / "runs",
        evaluation_root=PROTECTED_ROOT / "chexgenbench_sana" / "evaluations",
        run_schema="tricompose.chexgenbench_sana.run.v1",
        case_schema="tricompose.chexgenbench_sana.case_input.v1",
        generation_schema="tricompose.chexgenbench_sana.generation.v1",
        frozen_schema="tricompose.chexgenbench_sana.frozen_run.v1",
        prompt_field="sana_prompt",
    ),
    "pixart": ModelSpec(
        name="pixart",
        producer="frozen CheXGenBench PixArt-Sigma",
        run_root=PROTECTED_ROOT / "chexgenbench_pixart" / "runs",
        evaluation_root=PROTECTED_ROOT / "chexgenbench_pixart" / "evaluations",
        run_schema="tricompose.chexgenbench_pixart.run.v1",
        case_schema="tricompose.chexgenbench_pixart.case_input.v1",
        generation_schema="tricompose.chexgenbench_pixart.generation.v1",
        frozen_schema="tricompose.chexgenbench_pixart.frozen_run.v1",
        prompt_field="pixart_prompt",
    ),
    "radedit": ModelSpec(
        name="radedit",
        producer="frozen RadEdit text-to-image backbone",
        run_root=PROTECTED_ROOT / "radedit" / "runs",
        evaluation_root=PROTECTED_ROOT / "radedit" / "evaluations",
        run_schema="tricompose.radedit.run.v1",
        case_schema="tricompose.radedit.case_input.v1",
        generation_schema="tricompose.radedit.generation.v1",
        frozen_schema="tricompose.radedit.frozen_run.v1",
        prompt_field="radedit_prompt",
    ),
}


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported shared evaluation config")
    return payload


def _read_private_json(path: Path) -> dict[str, Any]:
    resolved = require_private_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


def _validate_id(value: str) -> str:
    import re

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value) is None:
        raise ValueError("invalid opaque run ID")
    return value


def _validate_case_id(value: str) -> str:
    import re

    if re.fullmatch(r"case_[0-9]{3}", value) is None:
        raise ValueError("invalid opaque case ID")
    return value


def _load_real_index(config: Mapping[str, Any]) -> Any:
    import pandas as pd

    dataset_root = Path(config["dataset"]["root"]).resolve(strict=True)
    return pd.read_csv(
        dataset_root / "manifest.csv",
        usecols=["cxr_path"],
        dtype={"cxr_path": str},
    )["cxr_path"]


def _real_cxr(paths: Any, index: int) -> Path:
    if index < 0 or index >= len(paths):
        raise ValueError("opaque source index is outside the private manifest")
    value = paths.iloc[index]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("matched real CXR is unavailable")
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError("matched real CXR is not a file")
    return path


def _collect_cases(
    config: Mapping[str, Any], spec: ModelSpec, generation_run_id: str
) -> tuple[list[metric_reference.CaseSpec], str, str]:
    generation_run = require_inside(
        spec.run_root / _validate_id(generation_run_id),
        PROTECTED_ROOT,
        must_exist=True,
    )
    manifest = _read_private_json(generation_run / "manifest.json")
    summary = _read_private_json(generation_run / "summary.json")
    frozen = _read_private_json(generation_run / "frozen.json")
    if manifest.get("schema_version") != spec.run_schema:
        raise ValueError("unsupported generation manifest")
    if manifest.get("run_id") != generation_run_id:
        raise ValueError("generation run identity mismatch")
    if summary.get("status") != "completed":
        raise ValueError("generation run is incomplete")
    if frozen.get("schema_version") != spec.frozen_schema or frozen.get("status") != "frozen":
        raise ValueError("generation run is not frozen")
    expected_count = int(config["evaluation"]["expected_case_count"])
    for payload in (manifest, summary, frozen):
        if payload.get("case_count") != expected_count:
            raise ValueError("generation run has the wrong fixed count")
    if frozen.get("manifest_sha256") != sha256_file(generation_run / "manifest.json"):
        raise ValueError("frozen manifest hash mismatch")
    if frozen.get("summary_sha256") != sha256_file(generation_run / "summary.json"):
        raise ValueError("frozen summary hash mismatch")
    if not isinstance(frozen.get("model_snapshot_audit"), dict):
        raise ValueError("frozen model audit is unavailable")

    source_run_id = _validate_id(str(manifest["source_run_id"]))
    source_run = require_inside(
        SOURCE_RUN_ROOT / source_run_id,
        PROTECTED_ROOT,
        must_exist=True,
    )
    manifest_rows = manifest.get("cases")
    frozen_rows = frozen.get("cases")
    if not isinstance(manifest_rows, list) or not isinstance(frozen_rows, list):
        raise ValueError("fixed case lists are unavailable")
    if len(manifest_rows) != expected_count or len(frozen_rows) != expected_count:
        raise ValueError("fixed case lists have the wrong length")

    real_index = _load_real_index(config)
    cases: list[metric_reference.CaseSpec] = []
    seen: set[str] = set()
    for manifest_row, frozen_row in zip(manifest_rows, frozen_rows, strict=True):
        case_id = _validate_case_id(str(manifest_row["case_id"]))
        if case_id in seen or frozen_row.get("case_id") != case_id:
            raise ValueError("duplicate or reordered opaque case")
        seen.add(case_id)
        source_case = _read_private_json(source_run / "cases" / case_id / "input.json")
        model_case = _read_private_json(generation_run / "cases" / case_id / "input.json")
        generation_dir = generation_run / "cases" / case_id / "generation"
        generation_path = generation_dir / "generation.json"
        generation = _read_private_json(generation_path)
        generated_cxr = require_private_file(generation_dir / "generated_cxr.png")
        if source_case.get("schema_version") != SOURCE_CASE_SCHEMA:
            raise ValueError("unsupported protected source case")
        if model_case.get("schema_version") != spec.case_schema:
            raise ValueError("unsupported protected model case")
        if generation.get("schema_version") != spec.generation_schema:
            raise ValueError("unsupported protected generation record")
        prompt = str(model_case[spec.prompt_field])
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected_hashes = (
            model_case.get("prompt_sha256"),
            manifest_row.get("prompt_sha256"),
            frozen_row.get("prompt_sha256"),
            generation.get("prompt_sha256"),
        )
        if any(value != prompt_hash for value in expected_hashes):
            raise ValueError("frozen prompt hash mismatch")
        if sha256_file(generation_path) != frozen_row.get("generation_sha256"):
            raise ValueError("frozen generation hash mismatch")
        image_hash = sha256_file(generated_cxr)
        if image_hash != generation.get("output", {}).get("image_sha256"):
            raise ValueError("generation image hash mismatch")
        if image_hash != frozen_row.get("image_sha256"):
            raise ValueError("frozen image hash mismatch")
        if int(model_case["seed"]) != int(frozen_row["seed"]):
            raise ValueError("frozen seed mismatch")
        facts = source_case.get("ehr_facts")
        if not isinstance(facts, dict):
            raise TypeError("protected EHR facts are unavailable")
        cases.append(
            metric_reference.CaseSpec(
                case_id=case_id,
                real_cxr=_real_cxr(real_index, int(source_case["source_row_index"])),
                generated_cxr=generated_cxr,
                facts=facts,
            )
        )
    return cases, source_run_id, sha256_file(generation_run / "frozen.json")


def _checkpoint_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["models"]["xrv_disease_checkpoint"]).resolve(strict=True)
    if not path.is_file():
        raise ValueError("configured XRV checkpoint is not a file")
    return {
        "xrv_disease": {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    }


def _xrv_runtime(config: Mapping[str, Any], device: Any) -> Any:
    return xrv_reference.FrozenXRVSelector(
        {
            "model": {
                "name": str(config["models"]["xrv_disease_name"]),
                "checkpoint": str(config["models"]["xrv_disease_checkpoint"]),
            }
        },
        device,
    )


def evaluate(
    *,
    config_path: str | Path,
    model_name: str,
    generation_run_id: str,
    evaluation_run_id: str,
) -> dict[str, Any]:
    import torch

    os.umask(0o077)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run evaluation through Slurm")
    if model_name not in MODEL_SPECS:
        raise ValueError("unsupported frozen CXR generator")
    spec = MODEL_SPECS[model_name]
    config = _load_config(config_path)
    audit = _checkpoint_audit(config)
    cases, source_run_id, generation_freeze_sha = _collect_cases(
        config, spec, generation_run_id
    )
    evaluation_root = create_private_stage_dir(
        spec.evaluation_root / _validate_id(evaluation_run_id)
    )
    log_path = evaluation_root / "evaluation.log"
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    try:
        with log_path.open("x", encoding="utf-8") as protected_log:
            with contextlib.redirect_stdout(protected_log), contextlib.redirect_stderr(protected_log):
                settings = config["evaluation"]
                batch_size = int(settings["batch_size"])
                operating_point = float(settings["operating_point"])
                repetitions = int(settings["bootstrap_repetitions"])
                bootstrap_seed = int(settings["bootstrap_seed"])
                real_paths = [case.real_cxr for case in cases]
                generated_paths = [case.generated_cxr for case in cases]

                print("stage=xrv_load", flush=True)
                xrv = _xrv_runtime(config, device)
                print("stage=xrv_real", flush=True)
                real_predictions = xrv.predict(real_paths, batch_size)
                print("stage=xrv_generated", flush=True)
                generated_predictions = xrv.predict(generated_paths, batch_size)
                del xrv
                torch.cuda.empty_cache()

                print("stage=quality", flush=True)
                real_quality = [
                    metric_reference._quality_features(
                        metric_reference._load_grayscale(path)
                    )
                    for path in real_paths
                ]
                generated_quality = [
                    metric_reference._quality_features(
                        metric_reference._load_grayscale(path)
                    )
                    for path in generated_paths
                ]
                quality_flags = metric_reference.robust_quality_flags(
                    real_quality,
                    generated_quality,
                    z_threshold=float(config["quality"]["z_threshold"]),
                    extreme_z_threshold=float(config["quality"]["extreme_z_threshold"]),
                )

                rows: list[dict[str, Any]] = []
                for index, case in enumerate(cases):
                    observable = case.facts.get("cxr_observable_targets")
                    if not isinstance(observable, dict):
                        raise TypeError("protected EHR observable targets are unavailable")
                    real_support = metric_reference._ehr_support(
                        observable,
                        real_predictions[index],
                        operating_point=operating_point,
                    )
                    generated_support = metric_reference._ehr_support(
                        observable,
                        generated_predictions[index],
                        operating_point=operating_point,
                    )
                    support_gap = None
                    if real_support["status"] == "scored":
                        support_gap = round(
                            float(generated_support["mean_support"])
                            - float(real_support["mean_support"]),
                            8,
                        )
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "ehr_positive_support": {
                                "real": real_support,
                                "generated": generated_support,
                                "generated_minus_real": support_gap,
                            },
                            "official_five_xrv": {
                                label: {
                                    "real": round(float(real_predictions[index][label]), 8),
                                    "generated": round(
                                        float(generated_predictions[index][label]), 8
                                    ),
                                }
                                for label in metric_reference.OFFICIAL_FIVE
                            },
                            "quality": quality_flags[index],
                            "generated_image_sha256": sha256_file(case.generated_cxr),
                        }
                    )

                scored = [
                    row
                    for row in rows
                    if row["ehr_positive_support"]["real"]["status"] == "scored"
                ]
                real_support_values = [
                    float(row["ehr_positive_support"]["real"]["mean_support"])
                    for row in scored
                ]
                generated_support_values = [
                    float(row["ehr_positive_support"]["generated"]["mean_support"])
                    for row in scored
                ]
                support_gaps = [
                    float(row["ehr_positive_support"]["generated_minus_real"])
                    for row in scored
                ]

                per_label: dict[str, Any] = {}
                flattened_real: list[float] = []
                flattened_generated: list[float] = []
                for label in metric_reference.OFFICIAL_FIVE:
                    real_values = [
                        float(row["official_five_xrv"][label]["real"]) for row in rows
                    ]
                    generated_values = [
                        float(row["official_five_xrv"][label]["generated"])
                        for row in rows
                    ]
                    flattened_real.extend(real_values)
                    flattened_generated.extend(generated_values)
                    per_label[label] = {
                        "paired_probability_mae": round(
                            float(
                                np.mean(
                                    np.abs(
                                        np.asarray(real_values)
                                        - np.asarray(generated_values)
                                    )
                                )
                            ),
                            8,
                        ),
                        "paired_probability_correlation": metric_reference._safe_correlation(
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
                        "real_xrv_pseudo_label_auroc": metric_reference.pseudo_label_auroc(
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

                print("stage=write", flush=True)
                case_metrics_path = write_private_json(
                    evaluation_root / "case_metrics.json",
                    {
                        "schema_version": CASE_METRICS_SCHEMA,
                        "evaluation_run_id": evaluation_run_id,
                        "generation_run_id": generation_run_id,
                        "cases": rows,
                    },
                )
                elapsed = time.perf_counter() - started
                summary_path = write_private_json(
                    evaluation_root / "summary.json",
                    {
                        "schema_version": EVAL_SCHEMA,
                        "status": "completed",
                        "model_name": model_name,
                        "evaluation_run_id": evaluation_run_id,
                        "generation_run_id": generation_run_id,
                        "source_run_id": source_run_id,
                        "sample_count": len(cases),
                        "scientific_scope": {
                            "path": (
                                "structured EHR -> deterministic radiology-style prompt -> "
                                f"{spec.producer} -> synthetic CXR"
                            ),
                            "direct_ehr_to_cxr": False,
                            "real_anchor": True,
                            "copies_or_serializes_real_cxr": False,
                            "loads_real_report": False,
                        },
                        "disease_alignment": {
                            "ehr_absence_semantics": "unknown_not_negative",
                            "ehr_positive_support_real": metric_reference._summary_stats(
                                real_support_values,
                                repetitions=repetitions,
                                seed=bootstrap_seed,
                            ),
                            "ehr_positive_support_generated": metric_reference._summary_stats(
                                generated_support_values,
                                repetitions=repetitions,
                                seed=bootstrap_seed + 1,
                            ),
                            "generated_minus_real_support": metric_reference._summary_stats(
                                support_gaps,
                                repetitions=repetitions,
                                seed=bootstrap_seed + 2,
                            ),
                            "official_five": {
                                "labels": list(metric_reference.OFFICIAL_FIVE),
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
                                "pooled_probability_correlation": metric_reference._safe_correlation(
                                    flattened_real, flattened_generated
                                ),
                            },
                        },
                        "quality_gate": {
                            "flagged_count": sum(
                                bool(row["quality"]["flagged"]) for row in rows
                            ),
                            "flagged_rate": round(
                                float(
                                    np.mean(
                                        [bool(row["quality"]["flagged"]) for row in rows]
                                    )
                                ),
                                8,
                            ),
                            "z_threshold": float(config["quality"]["z_threshold"]),
                            "extreme_z_threshold": float(
                                config["quality"]["extreme_z_threshold"]
                            ),
                        },
                        "protocol": {
                            "name": "lightweight frozen-XRV real-anchor evaluation",
                            "operating_point": operating_point,
                            "batch_size": batch_size,
                            "bootstrap_repetitions": repetitions,
                            "bootstrap_seed": bootstrap_seed,
                            "omitted_metrics": [
                                "BioViL",
                                "MS-SSIM",
                                "FID",
                                "age",
                                "sex",
                            ],
                        },
                        "freeze": {
                            "generation_freeze_sha256": generation_freeze_sha,
                            "checkpoint_audit": audit,
                            "case_metrics_sha256": sha256_file(case_metrics_path),
                        },
                        "software": {
                            "torch": torch.__version__,
                            "torchxrayvision": importlib.metadata.version(
                                "torchxrayvision"
                            ),
                        },
                        "runtime_seconds": round(elapsed, 4),
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    },
                )
        enforce_private_file_mode(log_path)
        return {
            "status": "completed",
            "model_name": model_name,
            "sample_count": len(cases),
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "summary_sha256": sha256_file(summary_path),
        }
    except Exception as exc:
        if log_path.exists():
            enforce_private_file_mode(log_path)
        write_private_json(
            evaluation_root / "failure.json",
            {
                "schema_version": EVAL_SCHEMA,
                "status": "failed",
                "error_type": type(exc).__name__,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--generation-run-id", required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                evaluate(
                    config_path=args.config,
                    model_name=args.model_name,
                    generation_run_id=args.generation_run_id,
                    evaluation_run_id=args.evaluation_run_id,
                ),
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

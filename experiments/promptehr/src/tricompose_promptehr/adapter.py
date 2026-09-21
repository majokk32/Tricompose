"""Generate protected PromptEHR candidates from official public synthetic seeds."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    enforce_private_file_mode,
    sha256_file,
    write_private_json,
)

from .evaluation import MODALITIES, evaluate_cases


RUN_SCHEMA = "tricompose.promptehr.run.v1"
CASE_SCHEMA = "tricompose.promptehr.synthetic_ehr.v1"
OFFICIAL_SOURCE_COMMIT = "7e3751ed564881bb06abbdc5887aff056ba417b9"
OFFICIAL_ARCHIVE_SHA256 = "1dcffdefbfbf69459375d3d6e863664a9782d16cb220c1d80ef381583acfb322"
OFFICIAL_DEMO_SHA256 = "40260cbd6d43b666eb7e76bd86a4ecc7ad4dc583f26144403a757b3fd8c8c59f"
EXPECTED_CHECKPOINT_FILES = {
    "config.json": (1897, "627de9d63ea239f2a1c2dc7d338b734772c8ffdadee0b2e489429e77b03e0b76"),
    "data_tokenizer.pkl": (2260356, "bf6e27e9ff0c9c929aeb135f6c3516136323d8844e6957182bd29b32a0db7f80"),
    "latest.checkpoint.pth.tar": (581142171, "eec18f3e0b6337a43a15a70ed0f0c146a72472a58ce04d638c1a99a7ae7ccc1b"),
    "model_tokenizer.pkl": (44546, "8d2bcea1ae0f6ef7d33eba745e062af7118ba4d56df4b15f36a2de511f9344e3"),
    "promptehr_config.json": (299, "868285c0befa07d61039fb537e08523ab97c6510067c38581655268e3661ce15"),
}


def _validate_run_id(value: str) -> str:
    if not value or len(value) > 64:
        raise ValueError("invalid run ID")
    if any(not (character.isalnum() or character in "._-") for character in value):
        raise ValueError("invalid run ID")
    return value


def _validate_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    for name, (expected_size, expected_sha256) in EXPECTED_CHECKPOINT_FILES.items():
        path = checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint file: {name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected_size or digest != expected_sha256:
            raise ValueError(f"checkpoint validation failed: {name}")
        files[name] = {"size_bytes": size, "sha256": digest}
    return {
        "name": "PromptEHR",
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "dataset_schema": "MIMIC-III longitudinal code sequences",
        "architecture": "BART-base with learned demographic prompt embeddings",
        "code_types": list(MODALITIES),
        "checkpoint_files": files,
        "frozen": True,
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _selection_hash(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _event_counts(visits: list[Any], order: list[str]) -> dict[str, int]:
    counts = {modality: 0 for modality in MODALITIES}
    for visit in visits:
        for position, modality in enumerate(order):
            counts[modality] += len(visit[position])
    return counts


def _build_case(
    *,
    case_id: str,
    input_visits: list[Any],
    output_visits: list[Any],
    order: list[str],
    vocabularies: dict[str, Any],
    conditioning_feature: Any,
    copied_label: Any,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(input_visits) != len(output_visits):
        errors.append("visit_count_mismatch")
    event_indices: list[dict[str, list[int]]] = []
    event_codes: list[dict[str, list[str]]] = []
    prompt_comparison = {
        modality: {
            "input_count": 0,
            "output_count": 0,
            "overlap_count": 0,
            "novel_output_count": 0,
        }
        for modality in MODALITIES
    }

    for visit_index, output_visit in enumerate(output_visits):
        if len(output_visit) != len(order):
            errors.append(f"visit_{visit_index}_modality_count")
            continue
        index_visit: dict[str, list[int]] = {}
        code_visit: dict[str, list[str]] = {}
        for position, modality in enumerate(order):
            values = sorted({int(value) for value in output_visit[position]})
            input_values = (
                {int(value) for value in input_visits[visit_index][position]}
                if visit_index < len(input_visits)
                else set()
            )
            vocabulary = vocabularies[modality]
            codes: list[str] = []
            for value in values:
                if value not in vocabulary.idx2word:
                    errors.append(f"visit_{visit_index}_{modality}_out_of_vocab")
                    continue
                codes.append(str(vocabulary.idx2word[value]))
            index_visit[modality] = values
            code_visit[modality] = codes
            overlap = set(values) & input_values
            novel = set(values) - input_values
            prompt_comparison[modality]["input_count"] += len(input_values)
            prompt_comparison[modality]["output_count"] += len(values)
            prompt_comparison[modality]["overlap_count"] += len(overlap)
            prompt_comparison[modality]["novel_output_count"] += len(novel)
        event_indices.append(index_visit)
        event_codes.append(code_visit)

    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "input_contract": "official_public_synthetic_seed_prompt",
        "conditioning": {
            "baseline_feature": _to_builtin(conditioning_feature),
            "label_is_copied_not_generated": True,
            "copied_label": _to_builtin(copied_label),
            "source_prompt_content_stored": False,
        },
        "event_indices": event_indices,
        "event_codes": event_codes,
        "prompt_comparison": prompt_comparison,
        "validation": {
            "strict_valid": not errors and bool(event_indices),
            "visit_count": len(event_indices),
            "errors": errors,
        },
    }


def run_generation(
    *,
    checkpoint_dir: Path,
    demo_data_dir: Path,
    run_id: str,
    count: int,
    selection_seed: int,
    generation_seed: int,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not 1 <= count <= 1000:
        raise ValueError("count must be between 1 and 1000")
    checkpoint_audit = _validate_checkpoint(checkpoint_dir)
    demo_file = demo_data_dir.resolve(strict=True) / "data.pkl"
    if not demo_file.is_file() or sha256_file(demo_file) != OFFICIAL_DEMO_SHA256:
        raise ValueError("official synthetic demo validation failed")

    run_dir = create_private_stage_dir(PROTECTED_ROOT / "promptehr" / "runs" / run_id)
    cases_dir = create_private_stage_dir(run_dir / "cases")
    diagnostic_path = run_dir / "diagnostic.log"
    started = time.monotonic()
    os.umask(0o077)

    with diagnostic_path.open("x", encoding="utf-8") as diagnostic_handle:
        with contextlib.redirect_stdout(diagnostic_handle), contextlib.redirect_stderr(diagnostic_handle):
            import pdb

            def _abort_debugger() -> None:
                raise RuntimeError("official PromptEHR entered an interactive debugger")

            pdb.set_trace = _abort_debugger
            import tokenizers
            import transformers
            from promptehr import PromptEHR, SequencePatient, load_synthetic_data

            runtime_versions = {
                "python": ".".join(str(value) for value in sys.version_info[:3]),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
            }
            expected_versions = {
                "torch": "1.13.1+cu117",
                "transformers": "4.19.0",
                "tokenizers": "0.12.1",
            }
            for package, expected_version in expected_versions.items():
                if runtime_versions[package] != expected_version:
                    raise RuntimeError(f"unexpected {package} version")

            all_demo = load_synthetic_data(str(demo_data_dir.resolve(strict=True)))
            population_size = len(all_demo["visit"])
            if count > population_size:
                raise ValueError("count exceeds official synthetic demo size")
            selected_indices = random.Random(selection_seed).sample(
                range(population_size), count
            )
            selected_visits = [all_demo["visit"][index] for index in selected_indices]
            selected_features = np.asarray(all_demo["feature"])[selected_indices]
            selected_labels = [all_demo["y"][index] for index in selected_indices]
            order = list(all_demo["order"])
            if tuple(order) != MODALITIES:
                raise ValueError("unexpected official modality order")

            seed_dataset = SequencePatient(
                data={
                    "v": selected_visits,
                    "x": selected_features,
                    "y": selected_labels,
                },
                metadata={
                    "visit": {"mode": "dense", "order": order},
                    "label": {"mode": "tensor"},
                    "voc": all_demo["voc"],
                    "max_visit": 20,
                },
            )

            random.seed(generation_seed)
            np.random.seed(generation_seed)
            torch.manual_seed(generation_seed)
            torch.cuda.manual_seed_all(generation_seed)
            torch.cuda.reset_peak_memory_stats()
            model = PromptEHR(
                device="cuda:0",
                num_worker=0,
                output_dir=str(run_dir / "unused_training_logs"),
                seed=generation_seed,
            )
            model.from_pretrained(str(checkpoint_dir.resolve(strict=True)))
            model.eval()
            model.model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            total_parameters = sum(parameter.numel() for parameter in model.parameters())
            trainable_parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            if trainable_parameters != 0:
                raise RuntimeError("PromptEHR parameters are not frozen")

            generated = model.predict(
                seed_dataset,
                n=count,
                n_per_sample=1,
                sample_config={
                    "num_beams": 1,
                    "no_repeat_ngram_size": 1,
                    "do_sample": True,
                    "num_return_sequences": 1,
                    "top_k": 1,
                    "temperature": 1.0,
                },
                verbose=False,
            )
            torch.cuda.synchronize()
            peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

    enforce_private_file_mode(diagnostic_path)
    if len(generated["visit"]) != count:
        raise RuntimeError("unexpected generated case count")

    cases: list[dict[str, Any]] = []
    case_entries: list[dict[str, Any]] = []
    event_count_values: list[int] = []
    for index in range(count):
        case_id = f"case_{index:03d}"
        case = _build_case(
            case_id=case_id,
            input_visits=selected_visits[index],
            output_visits=generated["visit"][index],
            order=order,
            vocabularies=all_demo["voc"],
            conditioning_feature=generated["feature"][index],
            copied_label=generated["y"][index],
        )
        cases.append(case)
        counts = _event_counts(generated["visit"][index], order)
        event_count = sum(counts.values())
        event_count_values.append(event_count)
        case_path = write_private_json(cases_dir / f"{case_id}.json", case)
        case_entries.append(
            {
                "case_id": case_id,
                "relative_path": str(case_path.relative_to(run_dir)),
                "sha256": sha256_file(case_path),
                "visit_count": case["validation"]["visit_count"],
                "strict_valid": case["validation"]["strict_valid"],
                "event_counts": counts,
            }
        )

    evaluation = evaluate_cases(cases)
    elapsed = time.monotonic() - started
    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "generator": {
            **checkpoint_audit,
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "runtime_versions": runtime_versions,
        },
        "privacy": {
            "real_patient_input_used": False,
            "official_public_synthetic_seed_used": True,
            "external_api_used": False,
            "opaque_case_ids": True,
            "protected_output": True,
        },
        "generation": {
            "count": count,
            "selection_seed": selection_seed,
            "selection_index_sha256": _selection_hash(selected_indices),
            "selection_indices_stored": False,
            "generation_seed": generation_seed,
            "n_per_sample": 1,
            "top_k": 1,
            "temperature": 1.0,
            "official_demo_population_size": population_size,
            "input_contract": "partial structured synthetic EHR plus demographic features",
            "cold_start": False,
        },
        "aggregate": {
            "strict_valid_count": sum(
                bool(case["validation"]["strict_valid"]) for case in cases
            ),
            "event_count_min": min(event_count_values),
            "event_count_median": statistics.median(event_count_values),
            "event_count_max": max(event_count_values),
            "total_seconds": elapsed,
            "peak_vram_gib": peak_vram_gib,
        },
        "simple_evaluation": evaluation,
        "cases": case_entries,
    }
    manifest_path = write_private_json(run_dir / "run.json", manifest)
    return {
        "status": "complete",
        "run_id": run_id,
        "count": count,
        "strict_valid_count": manifest["aggregate"]["strict_valid_count"],
        "unique_sequence_rate": evaluation["unique_sequence_rate"],
        "visit_count_min": evaluation["visit_count"]["min"],
        "visit_count_max": evaluation["visit_count"]["max"],
        "total_seconds": round(elapsed, 3),
        "peak_vram_gib": round(peak_vram_gib, 3),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--demo-data-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=20260805)
    parser.add_argument("--generation-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = run_generation(
            checkpoint_dir=args.checkpoint_dir,
            demo_data_dir=args.demo_data_dir,
            run_id=args.run_id,
            count=args.count,
            selection_seed=args.selection_seed,
            generation_seed=args.generation_seed,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        diagnostic_path = (
            PROTECTED_ROOT / "promptehr" / "runs" / args.run_id / "diagnostic.log"
        )
        if diagnostic_path.is_file():
            with diagnostic_path.open("a", encoding="utf-8") as diagnostic_handle:
                traceback.print_exc(file=diagnostic_handle)
            enforce_private_file_mode(diagnostic_path)
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

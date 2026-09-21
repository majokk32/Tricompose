"""Batch-score frozen synthetic CXR-report candidates with one Qwen-VL load."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_json,
)
from tricompose.reporting.protected_cxr import load_frozen_cxr_source
from tricompose.verifiers.qwenvl_cxr_report import (
    PROMPT_VERSION,
    _load_model,
    _resolve_external_directory,
    _score_candidate,
    _validate_opaque_name,
)


SCHEMA_VERSION = "tricompose.qwenvl_cxr_report_batch.v1"
PRODUCER_VERSION = "1.0.0"
TEMPORAL_PATTERN = re.compile(
    r"\b(?:compared (?:to|with) (?:the )?(?:prior|previous)|"
    r"no (?:significant )?interval change|unchanged|re-?demonstrated|"
    r"remains stable|stable (?:appearance|position))\b",
    flags=re.IGNORECASE,
)
MEASUREMENT_PATTERN = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:cm|mm)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ReportRunSpec:
    candidate_id: str
    namespace: str
    run_schema: str
    frozen_schema: str


REPORT_SPECS = (
    ReportRunSpec(
        candidate_id="maira2",
        namespace="maira2_report",
        run_schema="tricompose.maira2_report.run.v1",
        frozen_schema="tricompose.maira2_report.frozen_run.v1",
    ),
    ReportRunSpec(
        candidate_id="cxrmate_single",
        namespace="cxrmate_single_report",
        run_schema="tricompose.cxrmate_single_report.run.v1",
        frozen_schema="tricompose.cxrmate_single_report.frozen_run.v1",
    ),
)


@dataclass(frozen=True)
class FrozenReportCase:
    case_id: str
    report_path: Path
    report_sha256: str
    source_image_sha256: str


def _read_private_json(path: Path) -> dict[str, Any]:
    resolved = require_private_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


def _load_report_run(
    *,
    spec: ReportRunSpec,
    run_id: str,
    expected_source_run_id: str,
    expected_case_ids: tuple[str, ...],
) -> tuple[str, tuple[FrozenReportCase, ...]]:
    validated_run_id = _validate_opaque_name(run_id, label="report run ID")
    run = PROTECTED_ROOT / spec.namespace / "runs" / validated_run_id
    manifest_path = require_private_file(run / "manifest.json")
    summary_path = require_private_file(run / "summary.json")
    frozen_path = require_private_file(run / "frozen.json")
    manifest = _read_private_json(manifest_path)
    summary = _read_private_json(summary_path)
    frozen = _read_private_json(frozen_path)
    if manifest.get("schema_version") != spec.run_schema:
        raise ValueError("unsupported report manifest schema")
    if frozen.get("schema_version") != spec.frozen_schema:
        raise ValueError("unsupported frozen report schema")
    if manifest.get("run_id") != validated_run_id or frozen.get("run_id") != validated_run_id:
        raise ValueError("report run identity mismatch")
    if manifest.get("source_cxr_run_id") != expected_source_run_id:
        raise ValueError("report run uses a different CXR source")
    if summary.get("status") != "completed" or frozen.get("status") != "frozen":
        raise ValueError("report run is incomplete")
    if frozen.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("report manifest hash mismatch")
    if frozen.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("report summary hash mismatch")

    manifest_rows = manifest.get("cases")
    frozen_rows = frozen.get("cases")
    if not isinstance(manifest_rows, list) or not isinstance(frozen_rows, list):
        raise ValueError("report case indexes are unavailable")
    if len(manifest_rows) < len(expected_case_ids) or len(frozen_rows) < len(expected_case_ids):
        raise ValueError("report run has fewer cases than requested")

    cases: list[FrozenReportCase] = []
    for expected_case_id, manifest_row, frozen_row in zip(
        expected_case_ids,
        manifest_rows[: len(expected_case_ids)],
        frozen_rows[: len(expected_case_ids)],
        strict=True,
    ):
        if not isinstance(manifest_row, dict) or not isinstance(frozen_row, dict):
            raise TypeError("report case index is malformed")
        if manifest_row.get("case_id") != expected_case_id:
            raise ValueError("report manifest cases are reordered")
        if frozen_row.get("case_id") != expected_case_id:
            raise ValueError("frozen report cases are reordered")
        report_path = require_private_file(
            run / "cases" / expected_case_id / "generated_report.txt"
        )
        report_hash = sha256_file(report_path)
        if frozen_row.get("report_sha256") != report_hash:
            raise ValueError("frozen report hash mismatch")
        source_hash = str(manifest_row.get("source_image_sha256", ""))
        if not source_hash:
            raise ValueError("report source image hash is unavailable")
        cases.append(
            FrozenReportCase(
                case_id=expected_case_id,
                report_path=report_path,
                report_sha256=report_hash,
                source_image_sha256=source_hash,
            )
        )
    return sha256_file(frozen_path), tuple(cases)


def deterministic_flags(text: str) -> dict[str, object]:
    words = re.findall(r"\b\w+(?:[-']\w+)*\b", text)
    sentences = [
        sentence.strip().lower()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    duplicate_sentences = len(sentences) - len(set(sentences))
    return {
        "empty": not bool(text.strip()),
        "word_count": len(words),
        "under_20_words": len(words) < 20,
        "unsupported_temporal_language": bool(TEMPORAL_PATTERN.search(text)),
        "exact_cm_mm_measurement": bool(MEASUREMENT_PATTERN.search(text)),
        "duplicate_sentence_count": duplicate_sentences,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-model", default="sana")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--maira-run-id", required=True)
    parser.add_argument("--cxrmate-run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    return parser


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    scores = [float(record["qwen_match_score"]) for record in records]
    return {
        "candidate_count": len(records),
        "qwen_score_mean": round(statistics.fmean(scores), 8),
        "qwen_score_median": round(statistics.median(scores), 8),
        "qwen_score_min": round(min(scores), 8),
        "qwen_score_max": round(max(scores), 8),
        "unsupported_temporal_count": sum(
            bool(record["deterministic"]["unsupported_temporal_language"])
            for record in records
        ),
        "under_20_words_count": sum(
            bool(record["deterministic"]["under_20_words"])
            for record in records
        ),
        "exact_measurement_count": sum(
            bool(record["deterministic"]["exact_cm_mm_measurement"])
            for record in records
        ),
    }


def _run(args: argparse.Namespace, progress: dict[str, object]) -> dict[str, object]:
    progress["phase"] = "dependency_imports"
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this verifier through Slurm")
    if args.limit < 1:
        raise ValueError("case limit must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("invalid pixel bounds")

    progress["phase"] = "validate_frozen_inputs"
    run_id = _validate_opaque_name(args.run_id, label="evaluation run ID")
    model_path = _resolve_external_directory(args.model_path)
    source = load_frozen_cxr_source(
        source_model=args.source_model,
        source_run_id=args.source_run_id,
        limit=args.limit,
    )
    case_ids = tuple(case.case_id for case in source.cases)
    report_run_ids = {
        "maira2": args.maira_run_id,
        "cxrmate_single": args.cxrmate_run_id,
    }
    report_runs: dict[str, tuple[FrozenReportCase, ...]] = {}
    report_frozen_hashes: dict[str, str] = {}
    for spec in REPORT_SPECS:
        frozen_hash, cases = _load_report_run(
            spec=spec,
            run_id=report_run_ids[spec.candidate_id],
            expected_source_run_id=source.run_id,
            expected_case_ids=case_ids,
        )
        report_frozen_hashes[spec.candidate_id] = frozen_hash
        report_runs[spec.candidate_id] = cases

    progress["phase"] = "create_output"
    output_run = create_private_stage_dir(args.output_dir)
    cases_root = create_private_stage_dir(output_run / "cases")

    progress["phase"] = "model_setup"
    model, processor, model_type = _load_model(
        model_path,
        torch,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    torch.cuda.reset_peak_memory_stats()

    by_model: dict[str, list[dict[str, object]]] = {
        spec.candidate_id: [] for spec in REPORT_SPECS
    }
    raw_wins = {spec.candidate_id: 0 for spec in REPORT_SPECS}
    raw_ties = 0
    progress["phase"] = "model_inference"
    for ordinal, source_case in enumerate(source.cases):
        progress["case_ordinal"] = ordinal
        with Image.open(source_case.image_path) as image_handle:
            image = image_handle.convert("RGB").copy()
            image_dimensions = list(image_handle.size)
        case_candidates: dict[str, dict[str, object]] = {}
        for spec in REPORT_SPECS:
            report_case = report_runs[spec.candidate_id][ordinal]
            if report_case.case_id != source_case.case_id:
                raise ValueError("candidate case identity mismatch")
            if report_case.source_image_sha256 != source_case.image_sha256:
                raise ValueError("candidate report references a different image")
            report = report_case.report_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if not report:
                raise ValueError("candidate report is empty")
            score, reason = _score_candidate(
                report=report,
                image=image,
                model=model,
                processor=processor,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
            )
            record: dict[str, object] = {
                "candidate_id": spec.candidate_id,
                "report_sha256": report_case.report_sha256,
                "qwen_match_score": round(score, 8),
                "qwen_reason": reason,
                "deterministic": deterministic_flags(report),
            }
            case_candidates[spec.candidate_id] = record
            by_model[spec.candidate_id].append(record)

        maira_score = float(case_candidates["maira2"]["qwen_match_score"])
        cxrmate_score = float(case_candidates["cxrmate_single"]["qwen_match_score"])
        if maira_score > cxrmate_score:
            raw_wins["maira2"] += 1
            raw_winner: str | None = "maira2"
        elif cxrmate_score > maira_score:
            raw_wins["cxrmate_single"] += 1
            raw_winner = "cxrmate_single"
        else:
            raw_ties += 1
            raw_winner = None
        case_dir = create_private_stage_dir(cases_root / source_case.case_id)
        write_private_json(
            case_dir / "scores.json",
            {
                "schema_version": f"{SCHEMA_VERSION}.case",
                "case_id": source_case.case_id,
                "image_sha256": source_case.image_sha256,
                "image_dimensions": image_dimensions,
                "candidates": case_candidates,
                "uncalibrated_raw_qwen_winner": raw_winner,
            },
        )

    progress["phase"] = "aggregate"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_uncalibrated",
        "run_id": run_id,
        "producer_version": PRODUCER_VERSION,
        "source_cxr_run_id": source.run_id,
        "source_cxr_frozen_sha256": source.frozen_sha256,
        "report_run_ids": report_run_ids,
        "report_frozen_sha256": report_frozen_hashes,
        "case_count": len(source.cases),
        "candidate_count": len(source.cases) * len(REPORT_SPECS),
        "verifier": {
            "model_directory": model_path.name,
            "model_type": model_type,
            "prompt_version": PROMPT_VERSION,
            "frozen": True,
            "score_contract": "uncalibrated generated VLM judge score",
        },
        "model_summary": {
            candidate_id: _aggregate(records)
            for candidate_id, records in by_model.items()
        },
        "paired_raw_score_comparison": {
            "maira2_wins": raw_wins["maira2"],
            "cxrmate_single_wins": raw_wins["cxrmate_single"],
            "ties": raw_ties,
            "not_a_final_selector": True,
        },
        "cost": {
            "model_loads": 1,
            "model_calls": len(source.cases) * len(REPORT_SPECS),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    started = time.monotonic()
    progress: dict[str, object] = {"phase": "start"}
    output_dir = Path(args.output_dir)
    try:
        with Path(os.devnull).open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                summary = _run(args, progress)
        summary["cost"]["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output_path = write_private_json(output_dir / "summary.json", summary)
        frozen_path = write_private_json(
            output_dir / "frozen.json",
            {
                "schema_version": f"{SCHEMA_VERSION}.frozen",
                "status": "frozen_uncalibrated",
                "run_id": summary["run_id"],
                "case_count": summary["case_count"],
                "candidate_count": summary["candidate_count"],
                "source_cxr_frozen_sha256": summary["source_cxr_frozen_sha256"],
                "report_frozen_sha256": summary["report_frozen_sha256"],
                "summary_sha256": sha256_file(output_path),
                "mutation_policy": "never overwrite or regenerate this run",
            },
        )
        print(
            json.dumps(
                {
                    "stage": "qwenvl_cxr_report_batch",
                    "status": "ok",
                    "case_count": summary["case_count"],
                    "candidate_count": summary["candidate_count"],
                    "elapsed_seconds": summary["cost"]["elapsed_seconds"],
                    "peak_vram_gib": summary["cost"]["peak_vram_gib"],
                    "frozen_sha256": sha256_file(frozen_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        failure = {
            "stage": "qwenvl_cxr_report_batch",
            "status": "failed",
            "error_type": type(exc).__name__,
            "phase": progress.get("phase"),
            "case_ordinal": progress.get("case_ordinal"),
        }
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

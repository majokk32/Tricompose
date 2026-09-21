"""Score frozen CheXagent-2 reports against synthetic CXRs with Qwen-VL."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from tricompose.privacy import (
    create_private_stage_dir,
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
from tricompose.verifiers.qwenvl_cxr_report_batch import (
    ReportRunSpec,
    _aggregate,
    _load_report_run,
    deterministic_flags,
)


SCHEMA_VERSION = "tricompose.qwenvl_chexagent2_report_batch.v1"
PRODUCER_VERSION = "1.0.0"
CHEXAGENT2_SPEC = ReportRunSpec(
    candidate_id="chexagent2_srrg_findings",
    namespace="chexagent2_report",
    run_schema="tricompose.chexagent2_report.run.v1",
    frozen_schema="tricompose.chexagent2_report.frozen_run.v1",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-model", default="sana")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--chexagent2-run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    return parser


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
    report_frozen_hash, reports = _load_report_run(
        spec=CHEXAGENT2_SPEC,
        run_id=args.chexagent2_run_id,
        expected_source_run_id=source.run_id,
        expected_case_ids=case_ids,
    )

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
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen verifier is not frozen")
    torch.cuda.reset_peak_memory_stats()

    records: list[dict[str, object]] = []
    progress["phase"] = "model_inference"
    for ordinal, (source_case, report_case) in enumerate(
        zip(source.cases, reports, strict=True)
    ):
        progress["case_ordinal"] = ordinal
        if report_case.case_id != source_case.case_id:
            raise ValueError("candidate case identity mismatch")
        if report_case.source_image_sha256 != source_case.image_sha256:
            raise ValueError("candidate report references a different image")
        report = report_case.report_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if not report:
            raise ValueError("candidate report is empty")
        with Image.open(source_case.image_path) as image_handle:
            image = image_handle.convert("RGB").copy()
            image_dimensions = list(image_handle.size)
        score, reason = _score_candidate(
            report=report,
            image=image,
            model=model,
            processor=processor,
            torch=torch,
            max_new_tokens=args.max_new_tokens,
        )
        record: dict[str, object] = {
            "candidate_id": CHEXAGENT2_SPEC.candidate_id,
            "report_sha256": report_case.report_sha256,
            "qwen_match_score": round(score, 8),
            "qwen_reason": reason,
            "deterministic": deterministic_flags(report),
        }
        records.append(record)
        case_dir = create_private_stage_dir(cases_root / source_case.case_id)
        write_private_json(
            case_dir / "scores.json",
            {
                "schema_version": f"{SCHEMA_VERSION}.case",
                "case_id": source_case.case_id,
                "image_sha256": source_case.image_sha256,
                "image_dimensions": image_dimensions,
                "candidate": record,
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
        "report_run_id": args.chexagent2_run_id,
        "report_frozen_sha256": report_frozen_hash,
        "case_count": len(source.cases),
        "candidate_count": len(records),
        "verifier": {
            "model_directory": model_path.name,
            "model_type": model_type,
            "prompt_version": PROMPT_VERSION,
            "frozen": True,
            "score_contract": "uncalibrated generated VLM judge score",
        },
        "model_summary": {CHEXAGENT2_SPEC.candidate_id: _aggregate(records)},
        "cost": {
            "model_loads": 1,
            "model_calls": len(records),
            "peak_vram_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            ),
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
                    "stage": "qwenvl_chexagent2_report_batch",
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
            "stage": "qwenvl_chexagent2_report_batch",
            "status": "failed",
            "error_type": type(exc).__name__,
            "phase": progress.get("phase"),
            "case_ordinal": progress.get("case_ordinal"),
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

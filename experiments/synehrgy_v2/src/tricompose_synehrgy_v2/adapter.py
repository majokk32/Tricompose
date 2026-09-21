"""Generate protected, fully synthetic EHR candidates with frozen SynEHRgy-v2."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    sha256_file,
    write_private_json,
)

from .evaluation import evaluate_synthetic_cases
from .model import MODEL_SPECS, FrozenSynEHRgyRuntime, validate_model_snapshot
from .structure import parse_token_sequence


RUN_SCHEMA = "tricompose.synehrgy_v2.run.v1"
CASE_SCHEMA = "tricompose.synehrgy_v2.synthetic_ehr.v1"


def _validate_run_id(value: str) -> str:
    if not value or len(value) > 64:
        raise ValueError("invalid run ID")
    if any(not (character.isalnum() or character in "._-") for character in value):
        raise ValueError("invalid run ID")
    return value


def run_generation(
    *,
    model_dir: Path,
    model_variant: str = "gpt2-10bins",
    run_id: str,
    count: int,
    seed: int,
    temperature: float,
    top_k: int,
) -> dict[str, object]:
    _validate_run_id(run_id)
    if not 1 <= count <= 1000:
        raise ValueError("count must be between 1 and 1000")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    model_audit = validate_model_snapshot(model_dir, model_variant)
    run_dir = create_private_stage_dir(
        PROTECTED_ROOT / "synehrgy_v2" / "runs" / run_id
    )
    cases_dir = create_private_stage_dir(run_dir / "cases")
    runtime = FrozenSynEHRgyRuntime(model_dir, model_variant)
    started = time.monotonic()
    case_entries: list[dict[str, object]] = []
    token_lengths: list[int] = []
    elapsed_values: list[float] = []
    valid_count = 0
    evaluation_cases: list[dict[str, object]] = []

    for index in range(count):
        case_id = f"case_{index:03d}"
        case_seed = seed + index
        tokens, elapsed = runtime.generate(
            seed=case_seed,
            temperature=temperature,
            top_k=top_k,
        )
        parsed = parse_token_sequence(tokens)
        if parsed.validation["strict_valid"]:
            valid_count += 1
        token_lengths.append(len(tokens))
        elapsed_values.append(elapsed)
        evaluation_cases.append(
            {
                "tokens": tokens,
                "structure": parsed.structure,
                "validation": parsed.validation,
            }
        )
        payload = {
            "schema": CASE_SCHEMA,
            "case_id": case_id,
            "generator": model_audit,
            "generation": {
                "cold_start": True,
                "input": "bos_token_only",
                "seed": case_seed,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": 1.0,
                "token_count": len(tokens),
                "elapsed_seconds": elapsed,
                "numeric_representation": "official_discrete_bin_tokens",
            },
            "raw_tokens": tokens,
            "structure": parsed.structure,
            "validation": parsed.validation,
        }
        case_path = write_private_json(cases_dir / f"{case_id}.json", payload)
        case_entries.append(
            {
                "case_id": case_id,
                "relative_path": str(case_path.relative_to(run_dir)),
                "sha256": sha256_file(case_path),
                "token_count": len(tokens),
                "visit_count": parsed.validation["visit_count"],
                "strict_valid": parsed.validation["strict_valid"],
            }
        )

    total_elapsed = time.monotonic() - started
    simple_evaluation = evaluate_synthetic_cases(evaluation_cases)
    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "generator": model_audit,
        "privacy": {
            "real_patient_input_used": False,
            "external_api_used": False,
            "opaque_case_ids": True,
            "protected_output": True,
        },
        "generation": {
            "count": count,
            "base_seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": 1.0,
            "cold_start": True,
            "input": "bos_token_only",
            "numeric_representation": "official_discrete_bin_tokens",
        },
        "aggregate": {
            "strict_valid_count": valid_count,
            "token_length_min": min(token_lengths),
            "token_length_median": statistics.median(token_lengths),
            "token_length_max": max(token_lengths),
            "generation_seconds_mean": statistics.mean(elapsed_values),
            "total_seconds": total_elapsed,
            "peak_vram_gib": runtime.peak_vram_gib,
        },
        "simple_evaluation": simple_evaluation,
        "cases": case_entries,
    }
    manifest_path = write_private_json(run_dir / "run.json", manifest)
    return {
        "status": "complete",
        "run_id": run_id,
        "count": count,
        "strict_valid_count": valid_count,
        "eos_completion_rate": simple_evaluation["eos_completion_rate"],
        "unique_sequence_rate": simple_evaluation["unique_sequence_rate"],
        "token_length_min": min(token_lengths),
        "token_length_median": statistics.median(token_lengths),
        "token_length_max": max(token_lengths),
        "total_seconds": round(total_elapsed, 3),
        "peak_vram_gib": round(runtime.peak_vram_gib, 3),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model-variant",
        choices=sorted(MODEL_SPECS),
        default="gpt2-10bins",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    try:
        result = run_generation(
            model_dir=args.model_dir,
            model_variant=args.model_variant,
            run_id=args.run_id,
            count=args.count,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

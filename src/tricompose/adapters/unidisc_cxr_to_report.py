"""Generate one report from a protected synthetic CXR with frozen UniDisc."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_json,
    write_private_text,
)


STAGE_NAME = "unidisc_cxr_to_report"
FIXED_PROMPT = "Generate a concise radiology findings report for this chest x-ray."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--cfg", type=float, default=3.5)
    return parser


def _resolve_external(path: str, *, directory: bool = False) -> Path:
    resolved = Path(path).resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError("required external directory is missing")
    if not directory and not resolved.is_file():
        raise ValueError("required external file is missing")
    return resolved


def _run_private(
    args: argparse.Namespace,
    stage_dir: Path,
    progress: dict[str, str],
) -> dict[str, Any]:
    progress["phase"] = "dependency_imports"
    import numpy as np
    import torch
    from PIL import Image

    progress["phase"] = "resolve_inputs"
    external_root = _resolve_external(args.external_root, directory=True)
    checkpoint = _resolve_external(args.checkpoint)
    tokenizer_path = _resolve_external(args.tokenizer_path, directory=True)
    input_image = require_private_file(args.input_image)

    if args.max_tokens < 1 or args.sampling_steps < 1:
        raise ValueError("generation lengths must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this adapter through Slurm")

    progress["phase"] = "external_imports"
    sys.path.insert(0, str(external_root))
    import demo.inference as unidisc_inference
    from demo.api_data_defs import ChatMessage, ChatRequest, ContentPart

    def no_visualization_write(input_array: Any, output_name: str, row_len: int | None = None) -> int:
        del output_name
        if row_len is not None:
            return row_len
        element_count = max(1, int(input_array.numel()))
        return max(1, math.ceil(math.sqrt(element_count)))

    # The upstream inference function writes three diagnostic PNGs by default.
    unidisc_inference.save_grid_image = no_visualization_write

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    progress["phase"] = "compose_config"
    overrides = [
        (
            "experiments=[large_scale_train,"
            "large_scale_train_high_res_interleaved,"
            "eval_unified,"
            "large_scale_high_res_interleaved_inference]"
        ),
        f"trainer.load_from_state_dict={checkpoint}",
        f"data.tokenizer_name_or_path={tokenizer_path}",
        f"eval.gen_ppl_eval_model_name_or_path={tokenizer_path}",
        "eval.static_img_txt_demo=false",
        "eval.visualize_sample=false",
    ]
    config = unidisc_inference.get_cfg(overrides)
    progress["phase"] = "model_setup"
    inference_fn = unidisc_inference.setup(
        config=config,
        profile_memory=False,
        device="cuda",
    )
    model = inference_fn.keywords["model"]
    wrapped_sample = model._sample
    unwrapped_sample = getattr(wrapped_sample, "__wrapped__", None)
    if unwrapped_sample is None:
        raise RuntimeError("UniDisc sample wrapper cannot be safely bypassed")
    model._sample = types.MethodType(unwrapped_sample, model)

    progress["phase"] = "open_input_image"
    with Image.open(input_image) as image_handle:
        image = image_handle.convert("RGB").copy()

    request = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[ContentPart(type="text", text=FIXED_PROMPT)],
            ),
            ChatMessage(
                role="assistant",
                content=[ContentPart(type="image_url", image_url=image)],
            ),
        ],
        max_tokens=args.max_tokens,
        resolution=args.resolution,
        sampling_steps=args.sampling_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg=args.cfg,
        use_reward_models=False,
    )

    progress["phase"] = "model_inference"
    original_working_directory = Path.cwd()
    os.chdir(stage_dir)
    try:
        with torch.inference_mode():
            response = inference_fn(request)
    finally:
        os.chdir(original_working_directory)

    progress["phase"] = "extract_report"
    text_candidates = [
        part.text.strip()
        for part in response.messages[-1].content
        if part.type == "text" and part.text is not None and part.text.strip()
    ]
    if not text_candidates:
        raise ValueError("UniDisc returned an empty report")
    report = max(text_candidates, key=len)

    progress["phase"] = "write_output"
    output_path = stage_dir / "generated_report.txt"
    write_private_text(output_path, report + "\n")
    return {
        "checkpoint_file": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "external_inference_source_sha256": sha256_file(
            external_root / "demo" / "inference.py"
        ),
        "tokenizer_source": "local_read_only_snapshot",
        "input_contract": "protected_synthetic_cxr_png",
        "input_sha256": sha256_file(input_image),
        "prompt_template": "fixed_cxr_to_findings_v1",
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "sampling_steps": args.sampling_steps,
        "resolution": args.resolution,
        "output_file": output_path.name,
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
    progress = {"phase": "create_stage_directory"}

    try:
        stage_dir = create_private_stage_dir(args.output_dir)
        # Upstream prints both input messages and generated text; discard both.
        with Path(os.devnull).open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                manifest = _run_private(args, stage_dir, progress)

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
                    "artifact": "generated_report.txt",
                    "sha256": manifest["output_sha256"],
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
            "phase": progress["phase"],
        }
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            failure["missing_module"] = exc.name
        if isinstance(exc, OSError) and exc.errno is not None:
            failure["errno"] = exc.errno
        private_failure = dict(failure)
        import traceback

        private_failure["traceback_frames"] = [
            {
                "file": Path(frame.filename).name,
                "function": frame.name,
                "line": frame.lineno,
            }
            for frame in traceback.extract_tb(exc.__traceback__)
        ]
        if progress["phase"] in {
            "dependency_imports",
            "external_imports",
            "compose_config",
            "model_setup",
        }:
            # These phases run before the input image is opened, so their
            # exception text cannot contain patient content. Keep it private.
            private_failure["safe_setup_detail"] = str(exc)[:1000]
        if stage_dir is not None:
            try:
                write_private_json(stage_dir / "failure.json", private_failure)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

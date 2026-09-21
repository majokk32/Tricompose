"""Generate one report from a protected synthetic CXR with frozen LLaVA-Rad."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_json,
    write_private_text,
)


STAGE_NAME = "llavarad_cxr_to_report"
FIXED_PROMPT = "Given the chest X-ray image, describe the findings in the image:"
DEFAULT_BIOMEDBERT_PATH = (
    "/home1/yikeyang/.cache/huggingface/hub/"
    "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/"
    "snapshots/d673b8835373c6fa116d6d8006b33d48734e305d"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument(
        "--biomedbert-path",
        default=DEFAULT_BIOMEDBERT_PATH,
    )
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--conv-mode", default="v1")
    return parser


def _resolve_external_directory(path: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("required external directory is missing")
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
    external_root = _resolve_external_directory(args.external_root)
    model_path = _resolve_external_directory(args.model_path)
    model_base = _resolve_external_directory(args.model_base)
    biomedbert_path = _resolve_external_directory(args.biomedbert_path)
    input_image = require_private_file(args.input_image)

    vision_config = model_path / "biomedclipcxr_518.json"
    vision_checkpoint = model_path / "biomedclipcxr_518_checkpoint.pt"
    for required_file in (
        model_path / "adapter_config.json",
        model_path / "adapter_model.bin",
        model_path / "config.json",
        model_path / "non_lora_trainables.bin",
        vision_config,
        vision_checkpoint,
        model_base / "config.json",
        model_base / "pytorch_model.bin.index.json",
        model_base / "tokenizer.model",
        biomedbert_path / "config.json",
        biomedbert_path / "tokenizer_config.json",
        biomedbert_path / "vocab.txt",
    ):
        if not required_file.is_file():
            raise FileNotFoundError("required local model asset is missing")

    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this adapter through Slurm")

    progress["phase"] = "external_imports"
    sys.path.insert(0, str(external_root))
    import llava.model.builder as llava_builder
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.utils import disable_torch_init

    if args.conv_mode not in conv_templates:
        raise ValueError("unknown conversation mode")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    progress["phase"] = "compose_local_vision_config"
    with vision_config.open("r", encoding="utf-8") as handle:
        local_vision_payload = json.load(handle)
    local_vision_payload["text_cfg"]["hf_model_name"] = str(biomedbert_path)
    local_vision_payload["text_cfg"]["hf_tokenizer_name"] = str(biomedbert_path)
    local_vision_config = stage_dir / "biomedclipcxr_518.local.json"
    write_private_json(local_vision_config, local_vision_payload)

    progress["phase"] = "model_setup"
    disable_torch_init()

    # The shared checkpoint stores the vision paths relative to the upstream
    # training repository. Override them in memory so inference remains fully
    # offline and every external model directory stays read-only.
    original_auto_config = llava_builder.AutoConfig

    class LocalAutoConfig:
        @staticmethod
        def from_pretrained(location: str | Path, *positional: Any, **keywords: Any) -> Any:
            config = original_auto_config.from_pretrained(
                location,
                *positional,
                **keywords,
            )
            if Path(location).resolve() == model_path:
                config.mm_vision_tower_config = str(local_vision_config)
                config.mm_vision_tower_checkpoint = str(vision_checkpoint)
            return config

    llava_builder.AutoConfig = LocalAutoConfig
    try:
        tokenizer, model, image_processor, _ = llava_builder.load_pretrained_model(
            str(model_path),
            str(model_base),
            "llavarad",
            load_8bit=False,
            load_4bit=False,
            device="cuda",
        )
    finally:
        llava_builder.AutoConfig = original_auto_config

    if image_processor is None:
        raise RuntimeError("LLaVA-Rad image processor was not initialized")
    model.eval()

    progress["phase"] = "open_input_image"
    with Image.open(input_image) as image_handle:
        image = image_handle.convert("RGB").copy()
    image_tensor = image_processor.preprocess(
        image,
        return_tensors="pt",
    )["pixel_values"][0]

    question = DEFAULT_IMAGE_TOKEN + "\n" + FIXED_PROMPT
    conversation = conv_templates[args.conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    progress["phase"] = "model_inference"
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    progress["phase"] = "extract_report"
    report = tokenizer.decode(
        output_ids[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    stop_text = (
        conversation.sep
        if conversation.sep_style != SeparatorStyle.TWO
        else conversation.sep2
    )
    if stop_text and report.endswith(stop_text):
        report = report[: -len(stop_text)].strip()
    if not report:
        raise ValueError("LLaVA-Rad returned an empty report")

    progress["phase"] = "write_output"
    output_path = stage_dir / "generated_report.txt"
    write_private_text(output_path, report + "\n")
    return {
        "external_builder_source_sha256": sha256_file(
            external_root / "llava" / "model" / "builder.py"
        ),
        "model_adapter_file": "adapter_model.bin",
        "model_adapter_bytes": (model_path / "adapter_model.bin").stat().st_size,
        "model_base_contract": "local_read_only_vicuna_7b_v1_5",
        "biomedbert_contract": "local_read_only_config_and_tokenizer",
        "vision_checkpoint_file": vision_checkpoint.name,
        "vision_checkpoint_bytes": vision_checkpoint.stat().st_size,
        "input_contract": "protected_synthetic_cxr_png",
        "input_sha256": sha256_file(input_image),
        "prompt_template": "fixed_llavarad_findings_v1",
        "decoding": "greedy",
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
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
        original_working_directory = Path.cwd()
        os.chdir(stage_dir)
        try:
            # Upstream prints model paths and may print generated text. Keep all
            # third-party output out of the public Slurm logs.
            with Path(os.devnull).open("w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    manifest = _run_private(args, stage_dir, progress)
        finally:
            os.chdir(original_working_directory)

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
    except Exception as exc:  # Deliberately omit private messages and values.
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
            "resolve_inputs",
            "external_imports",
            "compose_local_vision_config",
            "model_setup",
        }:
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

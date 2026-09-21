"""Thin local-only runtime around the deployed frozen UniDisc checkpoint."""

from __future__ import annotations

import contextlib
import gc
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_RESOLUTION = 256
DEFAULT_SAMPLING_STEPS = 35
DEFAULT_CFG = 3.5
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95
OFFLINE_EVAL_TOKENIZER = "NousResearch/Llama-2-7b-hf"
CHECKPOINT_RELATIVE = Path("ckpts/unidisc_interleaved/unidisc_interleaved.pt")
VQ_RELATIVE = Path("ckpts/unidisc_interleaved/vq_ds16_t2i.pt")
EXPECTED_SIZES = {
    CHECKPOINT_RELATIVE: 5_627_633_316,
    VQ_RELATIVE: 287_920_306,
}
REQUIRED_SOURCE_FILES = (
    Path("configs/config.yaml"),
    Path("demo/inference.py"),
    Path("demo/api_data_defs.py"),
)


@dataclass(frozen=True)
class UniDiscResult:
    image: Any
    elapsed_seconds: float
    peak_vram_gib: float


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    import os

    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def validate_unidisc_deployment(model_root: str | Path) -> dict[str, Any]:
    root = Path(model_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("UniDisc deployment root is not a directory")
    for relative in REQUIRED_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing UniDisc source component: {relative}")
    for relative, expected_size in EXPECTED_SIZES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing UniDisc checkpoint component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected UniDisc checkpoint size: {relative}")
    return {
        "checkpoint_bytes": EXPECTED_SIZES[CHECKPOINT_RELATIVE],
        "vq_bytes": EXPECTED_SIZES[VQ_RELATIVE],
        "checkpoint_layout_verified": True,
    }


def build_unidisc_overrides(model_root: str | Path) -> list[str]:
    """Compose the official inference config without an unused online dependency."""

    root = Path(model_root).resolve(strict=True)
    return [
        (
            "experiments=[large_scale_train,large_scale_train_high_res_interleaved,"
            "eval_unified,large_scale_high_res_interleaved_inference]"
        ),
        f"trainer.load_from_state_dict={root / CHECKPOINT_RELATIVE}",
        # UniDisc initializes this tokenizer even when generative perplexity is
        # disabled.  Reuse the already-cached generation tokenizer so offline
        # image inference does not try to fetch the unused default gpt2-large.
        f"eval.gen_ppl_eval_model_name_or_path={OFFLINE_EVAL_TOKENIZER}",
        "eval.compute_generative_perplexity=false",
    ]


class FrozenUniDiscRuntime:
    """Load UniDisc once and generate one image per protected case."""

    def __init__(self, model_root: str | Path, bootstrap_dir: str | Path) -> None:
        self.root = Path(model_root).resolve(strict=True)
        validate_unidisc_deployment(self.root)
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        import torch
        from demo.api_data_defs import ChatMessage, ChatRequest, ContentPart
        from demo.inference import get_cfg, setup

        if not torch.cuda.is_available():
            raise RuntimeError("UniDisc runtime requires a Slurm CUDA allocation")
        self.torch = torch
        self.ChatMessage = ChatMessage
        self.ChatRequest = ChatRequest
        self.ContentPart = ContentPart
        overrides = build_unidisc_overrides(self.root)
        bootstrap = Path(bootstrap_dir)
        self.load_attempts = 0
        for attempt in range(1, 3):
            self.load_attempts = attempt
            try:
                with _working_directory(bootstrap):
                    # Recompose because UniDisc mutates the config during setup.
                    config = get_cfg(overrides)
                    self.inference_fn = setup(config=config, profile_memory=False)
                break
            except OSError:
                if attempt == 2:
                    raise
                # Drop an incompletely initialized model before one bounded
                # retry for a transient filesystem read failure. All non-I/O
                # exceptions still fail immediately.
                gc.collect()
                torch.cuda.empty_cache()

    def generate(self, prompt: str, seed: int, runtime_dir: str | Path) -> UniDiscResult:
        import numpy as np

        torch = self.torch
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

        request = self.ChatRequest(
            messages=[
                self.ChatMessage(
                    role="user",
                    content=[self.ContentPart(type="text", text=prompt)],
                )
            ],
            resolution=DEFAULT_RESOLUTION,
            sampling_steps=DEFAULT_SAMPLING_STEPS,
            cfg=DEFAULT_CFG,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            use_reward_models=False,
        )
        torch.cuda.synchronize()
        started = time.monotonic()
        with _working_directory(Path(runtime_dir)), torch.inference_mode():
            response = self.inference_fn(request)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started

        image = None
        for part in response.messages[-1].content:
            if part.type == "image_url" and part.image_url is not None:
                image = part.image_url
                break
        if image is None or not hasattr(image, "save") or not hasattr(image, "size"):
            raise RuntimeError("UniDisc did not return a PIL-compatible image")
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return UniDiscResult(
            image=image,
            elapsed_seconds=elapsed,
            peak_vram_gib=peak,
        )

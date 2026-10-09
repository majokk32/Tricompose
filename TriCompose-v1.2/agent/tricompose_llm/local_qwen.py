"""Optional frozen LOCAL Qwen planner, loaded only inside approved GPU Slurm.

Text-only numeric planning, NOT a newly qualified multimodal clinical critic.
Reuses the existing audited Qwen loader; no new downloads or trained weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .contracts import ContractError, decode_json, decision_schema, require, validate_decision
from .planner import SYSTEM_PROMPT

EXPECTED_WEIGHT = "26fa644c8e61aa8185da5055b61aa28d6897fee102c2f43173acab657c4257a1"


def gpu_guard():
    from tricompose_v12.live_workers import require_gpu_slurm
    require_gpu_slurm()
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and f"/job_{job}/" in Path("/proc/self/cgroup").read_text(),
            "actual_gpu_slurm_cgroup_required")


class LocalQwenPlanner:
    def __init__(self, model_path):
        gpu_guard()  # Before torch imports, checkpoint reads or model construction.
        import torch
        from contracts import sha256_file
        from tricompose.verifiers.qwenvl_cxr_report import _load_model
        require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 24*1024**3,
                "local_planner_needs_at_least_24gib_gpu")
        root = Path(model_path).resolve(strict=True)
        files = [root/name for name in ("config.json", "generation_config.json", "processor_config.json",
            "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors")]
        self.asset_pins = {p.name: sha256_file(p) for p in files}
        require(self.asset_pins["model.safetensors"] == EXPECTED_WEIGHT, "audited_qwen_weight_required")
        self._asset_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        self.model, self.processor, self.model_type = _load_model(root, torch, min_pixels=256*28*28, max_pixels=512*28*28)
        self.model.eval().requires_grad_(False)
        require(not self.model.training and not any(p.requires_grad for p in self.model.parameters()), "frozen_qwen_required")
        self.torch = torch
        self.local_attempts = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def propose(self, state):
        schema = decision_schema(state)
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\nRequired JSON schema: " + json.dumps(schema)},
                    {"role": "user", "content": json.dumps(state, sort_keys=True, allow_nan=False)}]
        self.local_attempts += 1
        try:
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            length = int(inputs["input_ids"].shape[-1])
            require(length <= 8192, "bounded_local_planner_context")
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=384, do_sample=False)
            output_length = int(generated.shape[-1]) - length
            self.usage["input_tokens"] += length
            self.usage["output_tokens"] += output_length
            require(output_length < 384, "complete_local_planner_output_required")
            text = self.processor.batch_decode(generated[:, length:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
            return validate_decision(decode_json(text), state)
        except Exception:
            raise ContractError("local_planner_failed_or_invalid") from None

    def audit(self):
        require(all((p.stat().st_size, p.stat().st_mtime_ns) == expected for p, expected in self._asset_stats.items()),
                "local_model_assets_changed")
        return {"model_type": self.model_type, "asset_pins": self.asset_pins,
                "frozen": True, "local_attempts": self.local_attempts, "usage": self.usage,
                "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated()),
                "input_scope": "numeric_state_only_not_multimodal_critic"}

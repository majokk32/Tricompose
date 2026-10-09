#!/usr/bin/env python3
"""One protected numeric decision; subprocess exit releases local Qwen VRAM.

No external API, clinical artifact body, image input, retry or format repair.
Only usable in separately approved GPU Slurm, never on a login node.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "TriCompose-v1.2/src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT / relative))

from contracts import PROTECTED_ROOT, read_json, require_inside, sha256_file, write_private_json
from tricompose_llm.contracts import validate_public_state
from tricompose_llm.local_qwen_v2 import LocalQwenPlanner
from tricompose_llm.local_qwen import gpu_guard


def run(args):
    gpu_guard()
    state_path = require_inside(args.state, PROTECTED_ROOT, must_exist=True)
    output = require_inside(args.output, PROTECTED_ROOT, must_exist=False)
    if (state_path.stat().st_size > 32768 or sha256_file(state_path) != args.state_sha256
            or output.exists()):
        raise ValueError("bounded_pinned_state_and_exclusive_output_required")
    state = validate_public_state(read_json(state_path))
    if sha256_file(state_path) != args.state_sha256:
        raise ValueError("numeric_state_changed")
    planner = LocalQwenPlanner(args.model_path)
    try:
        decision = planner.propose(state)
    except Exception:
        write_private_json(output, {"status": "failed", "state_sha256": args.state_sha256,
                                   "audit": planner.audit(), "decision": None})
        return 1
    write_private_json(output, {"status": "completed", "state_sha256": args.state_sha256,
                               "audit": planner.audit(), "decision": decision})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        return run(args)
    except Exception:
        print(json.dumps({"stage": "one_local_numeric_policy_decision", "status": "failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Local frozen Qwen pilot with an explicit pre-call empty-tool-menu guard."""
import json
import os

from run import build_parser
from tricompose_llm.local_pilot import run_local_pilot
from tricompose_llm.local_qwen_bounded import INTERFACE_VERSION, LocalQwenPlanner


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(mode="cache-replay", planner="local-qwen")
    args = parser.parse_args()
    os.umask(0o007)
    try:
        return run_local_pilot(args, planner_factory=LocalQwenPlanner, interface_version=INTERFACE_VERSION)
    except Exception:
        print(json.dumps({"stage": "llm_bounded_routing_pilot", "status": "failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

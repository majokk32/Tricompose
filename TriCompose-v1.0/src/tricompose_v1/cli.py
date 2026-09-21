"""Lightweight inspection and planning CLI for TriCompose version 1.0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import CandidateGraph, SelectionPolicy
from .planner import build_exhaustive_plan
from .registry import ModelRegistry


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON input must be an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    registry = commands.add_parser("registry-summary")
    registry.add_argument("--registry", type=Path, required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument("--ehr-seed", type=int, action="append")
    plan.add_argument("--cxr-seed", type=int, action="append")
    plan.add_argument("--use-ehr-prompt", action="store_true")

    select = commands.add_parser("select")
    select.add_argument("--graph", type=Path, required=True)
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--budget-exhausted", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "registry-summary":
            result = ModelRegistry.from_path(args.registry).summary()
        elif args.command == "plan":
            registry = ModelRegistry.from_path(args.registry)
            result = build_exhaustive_plan(
                registry,
                ehr_seeds=args.ehr_seed or [42],
                cxr_seeds=args.cxr_seed or [0],
                use_ehr_prompt=args.use_ehr_prompt,
            )
        else:
            graph = CandidateGraph.from_dict(_read_object(args.graph))
            policy = SelectionPolicy.from_path(args.policy)
            result = graph.select_or_route(
                policy,
                budget_remaining=not args.budget_exhausted,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

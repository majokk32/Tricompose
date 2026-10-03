#!/usr/bin/env python3
"""GPU Slurm only: execute the preflighted two-case frozen-worker plan."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from tricompose_v12.live_execution import run
from contracts import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-run", "plan-manifest-sha256", "output-root", "run-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try: root, result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": result["status"],
        "runtime_seconds": result["runtime_seconds_includes_startup_and_io"],
        "manifest_sha256": sha256_file(root/"manifest.json")}))
    return 0 if result["completed_triplets"]==4 else 1


if __name__ == "__main__": raise SystemExit(main())

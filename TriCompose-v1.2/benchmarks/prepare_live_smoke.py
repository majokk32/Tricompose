#!/usr/bin/env python3
"""CPU Slurm preflight only: pin two original synthetic cases and frozen workers."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from tricompose_v12.live_plan import prepare
from contracts import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output-root", "run-id"): p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007); started = time.monotonic()
    try: root, _ = prepare(args)
    except Exception as exc:
        print(json.dumps({"status": "preflight_failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "completed_cpu_preflight_no_inference", "new_model_calls": 0,
        "runtime_seconds": round(time.monotonic()-started, 3), "manifest_sha256": sha256_file(root/"manifest.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())

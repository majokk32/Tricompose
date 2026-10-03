#!/usr/bin/env python3
"""Bounded observer in an existing CPU Slurm allocation; never submits jobs.

Wait for one approved GPU job, then invoke the metadata-only post-run auditor
once. No model execution, resumption, routing update, or automatic notification
service. Results and native auditor logs remain project-private.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, private_directory, write_private_json)
from tricompose_v12.runtime_dispatch import require_slurm, _new_private_handle

TERMINAL_FAILURES = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
AUDITOR = ROOT/"audits/audit_fixed_image_reports.py"


def wait_for_completion(status, has_manifest, *, max_wait, poll_seconds, now=time.monotonic, sleep=time.sleep):
    if type(max_wait) is not int or not 1 <= max_wait <= 10800:
        raise ValueError("wait bound required")
    if type(poll_seconds) is not int or not 10 <= poll_seconds <= 60:
        raise ValueError("bounded non-busy polling required")
    deadline = now()+max_wait
    while True:
        state = status()
        if state in TERMINAL_FAILURES: return "gpu_job_terminal_without_audit"
        if state == "COMPLETED":
            return "ready_for_metadata_audit" if has_manifest() else "completed_without_published_manifest"
        remaining = deadline-now()
        if remaining <= 0: return "observer_deadline_expired_no_audit"
        sleep(min(poll_seconds, remaining))


def job_status(job):
    # Fixed numeric job ID only; no clinical data or arbitrary scheduler mutation.
    if not re.fullmatch(r"[1-9][0-9]{0,11}", job): raise ValueError("numeric job ID required")
    try:
        result = subprocess.run(["sacct", "-j", job, "--noheader", "--parsable2", "--format=JobIDRaw,State"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=True, text=True)
    except (subprocess.SubprocessError, OSError): return "UNKNOWN"
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[0] == job:
            state = parts[1].split()[0].rstrip("+")
            return state if state in TERMINAL_FAILURES | {"COMPLETED", "PENDING", "RUNNING", "COMPLETING"} else "UNKNOWN"
    return "UNKNOWN"


def run(args):
    require_slurm()
    if not re.fullmatch(r"[1-9][0-9]{0,11}", args.gpu_job): raise ValueError("numeric GPU job ID required")
    root = require_inside(args.watch_root, PROTECTED_ROOT, must_exist=False)
    source = require_inside(args.run, PROTECTED_ROOT, must_exist=False)
    plan = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    private_directory(root)
    auditor_sha = sha256_file(AUDITOR)
    write_private_json(root/"start.json", {"status": "bounded_metadata_observer_started", "gpu_job_id": args.gpu_job,
        "cpu_allocation_id": os.environ["SLURM_JOB_ID"], "maximum_wait_seconds": args.max_wait,
        "poll_seconds": args.poll_seconds, "auditor_program_sha256": auditor_sha,
        "new_model_calls": 0, "job_submission_allowed": False, "automatic_model_retry_allowed": False})
    print("status=bounded_metadata_observer_started", flush=True)
    outcome = wait_for_completion(lambda: job_status(args.gpu_job), lambda: (source/"manifest.json").is_file(),
                                  max_wait=args.max_wait, poll_seconds=args.poll_seconds)
    if outcome == "ready_for_metadata_audit":
        if sha256_file(AUDITOR) != auditor_sha: raise ValueError("auditor source changed during wait")
        argv = [sys.executable, str(AUDITOR), "--plan-run", str(plan),
            "--plan-manifest-sha256", args.plan_manifest_sha256, "--run", str(source),
            "--output-root", args.audit_output_root, "--run-id", args.audit_run_id]
        with _new_private_handle(root/"auditor.stdout.log") as out, _new_private_handle(root/"auditor.stderr.log") as err:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=300, check=False)
        outcome = "metadata_audit_completed" if result.returncode == 0 else "metadata_audit_failed_private_log"
    write_private_json(root/"finish.json", {"status": outcome, "gpu_job_id": args.gpu_job,
        "new_model_calls": 0, "new_slurm_submissions": 0, "clinical_acceptance": False,
        "auditor_program_sha256": auditor_sha})
    print("status="+outcome, flush=True)
    return outcome


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gpu-job", "watch-root", "run", "plan-run", "plan-manifest-sha256", "audit-output-root", "audit-run-id"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--max-wait", type=int, default=10800); p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args(); os.umask(0o007)
    try: run(args)
    except Exception as exc:
        print(json.dumps({"status": "metadata_observer_failed", "error_type": type(exc).__name__})); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())

#!/bin/bash
# NOT an sbatch script. Use the current CPU allocation only after the user
# explicitly approves reading/copying these already generated synthetic reports.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PACKET="$ROOT/human_review/report_blind48_20261002_001"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/materialize_blinded_report_review.py"
test "${ALLOW_SYNTHETIC_REPORT_COPY:-0}" = 1
test -n "${SLURM_JOB_ID:-}"
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
TASK="$ROOT/task_runtime/blinded_report_copies_${SLURM_JOB_ID}"
test ! -e "$TASK"
test ! -e "$ROOT/human_review/report_blind48_text_${SLURM_JOB_ID}"
printf '%s  %s\n' \
  e11e76a9920872c675ef90361cc3ba643ec156183130f3a60e59d9896572b147 "$PACKET/manifest.json" \
  a20038ff9333ddfad05b9d51c7e2059ad3abc023fb78af681714b995a3efc546 "$PROGRAM" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
trap 'find "$TASK" -type d -exec chmod 2770 {} +; find "$TASK" -type f -exec chmod 0660 {} +' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/benchmarks:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export XDG_CACHE_HOME="$TASK/cache" TMPDIR="$TASK/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 120s \
  "$WORKSPACE/cxrmate/venv/bin/python" "$PROGRAM" \
  --packet-run "$PACKET" --output-root "$ROOT/human_review" \
  --run-id "report_blind48_text_${SLURM_JOB_ID}" --allow-synthetic-report-copy \
  >"$TASK/run.log" 2>&1
printf 'status=protected_synthetic_report_blind_copies_completed_no_inference\n'

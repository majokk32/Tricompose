#!/bin/bash
# NOT sbatch: approval-gated metadata audit in the existing CPU allocation.
# No report/EHR files, image pixels, model calls, downloads or external APIs.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PROGRAM="$WORKSPACE/TriCompose-v1.2/real_validation/audit_official_report_gold.py"
DATASET=/project2/ruishanl_1185/datasets
GOLD="$DATASET/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.1.0-test-set-labeled.csv"
LINKAGE="$DATASET/three_modalities/v2_labs_vitals/manifest.csv"
test "${ALLOW_REAL_GOLD_METADATA_AUDIT:-0}" = 1
test "${SLURM_JOB_ID:-}" = 12576792
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
TASK="$ROOT/task_runtime/official_gold_metadata_${SLURM_JOB_ID}"
RUN="official_report_gold_coverage_${SLURM_JOB_ID}"
test ! -e "$TASK"
test ! -e "$ROOT/real_validation/$RUN"
test -r "$GOLD"
test -r "$LINKAGE"
test -x "$WORKSPACE/cxrmate/venv/bin/python"
printf '%s  %s\n' \
  f52b57396ba7504e0a57710ccadad51a4edf8085687e1bc93352f84bcda5fff4 "$PROGRAM" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$WORKSPACE/TriCompose-v1.0/eval/report_v1_1/contracts.py" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
trap 'find "$TASK" -type d -exec chmod 2770 {} +; find "$TASK" -type f -exec chmod 0660 {} +' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export XDG_CACHE_HOME="$TASK/cache" TMPDIR="$TASK/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 120s \
  "$WORKSPACE/cxrmate/venv/bin/python" "$PROGRAM" \
  --gold-labels "$GOLD" --linkage-manifest "$LINKAGE" \
  --output-root "$ROOT/real_validation" --run-id "$RUN" \
  --allow-real-gold-metadata >"$TASK/run.log" 2>&1
printf 'status=protected_official_report_gold_metadata_audit_completed_no_inference\n'

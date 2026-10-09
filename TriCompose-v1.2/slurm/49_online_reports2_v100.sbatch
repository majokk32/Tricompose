#!/bin/bash
# PREPARED ONLY: explicit approval after this complete script is shown.
# Two fixed EHRs; at most 2 new images and 4 reports; no training or EHR edits.
# Shadow static-control work is paid; independent endpoint follows sealed choices.
# Thirty minutes is an allocation cap, not a runtime/start-time guarantee.
#SBATCH --job-name=tri_v12_online2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/online_report_plans/online2_12654973_001"
PLAN_SHA=fca888e160ce692eaf100aa5cddf3aabd504c283a1b17e35ce6bf64c628dafac
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/online_report_runs"
AUDITS="$WORKSPACE/artifacts/protected/tricompose_v1_2/online_report_audits"
RUN_ID="online2_${SLURM_JOB_ID}"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/$RUN_ID"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/online_report_smoke.py"
PROGRAM_SHA=5f9f2d1f7d90d8e686973e8afe160a989ec83d8d9024564dcb9674c214389cfc
cd "$WORKSPACE"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_online2_${SLURM_JOB_ID}.out" \
  "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_online2_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID" || test -e "$AUDITS/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
PROGRAM_ACTUAL="$(sha256sum "$PROGRAM")"
if test "${PROGRAM_ACTUAL%% *}" != "$PROGRAM_SHA"; then
  printf 'status=refused reason=controller_source_changed\n'
  exit 1
fi
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor" MPLCONFIGDIR="$TASK/cache/matplotlib"
export TMPDIR="$TASK/tmp" TEMP="$TASK/tmp" TMP="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
finish() {
  local task_exit=$?
  trap - EXIT
  if ! "$PYTHON" - "$WORKSPACE" "$TASK" \
    >"$TASK/permissions.stdout.log" 2>"$TASK/permissions.stderr.log" <<'PY'
from pathlib import Path
import sys
workspace, task = map(Path, sys.argv[1:])
boundary = workspace / 'artifacts/protected/tricompose_v1_2/job_runtime'
if task.is_symlink() or task.parent.resolve() != boundary.resolve():
    raise ValueError('exact_new_job_runtime_required')
sys.path.insert(0, str(workspace / 'TriCompose-v1.2/benchmarks'))
from run_fixed_image_reports import normalize_owned_modes
normalize_owned_modes(task)
PY
  then
    printf 'status=failed reason=private_runtime_permission_audit\n'
    task_exit=1
  fi
  exit "$task_exit"
}
trap finish EXIT
if ! "$PYTHON" "$PROGRAM" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=failed reason=see_protected_budget_journals\n'
  exit 1
fi
SOURCE_HASH="$(sha256sum "$OUTPUT/$RUN_ID/manifest.json")"
if ! "$PYTHON" "$PROGRAM" audit \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --source-run "$OUTPUT/$RUN_ID" --source-manifest-sha256 "${SOURCE_HASH%% *}" \
  --output-root "$AUDITS" --run-id "$RUN_ID" \
  >"$TASK/audit.stdout.log" 2>"$TASK/audit.stderr.log"; then
  printf 'status=completed_generation_but_metadata_audit_failed\n'
  exit 1
fi
printf 'status=completed_online_report_smoke_metadata_audited_unvalidated\n'

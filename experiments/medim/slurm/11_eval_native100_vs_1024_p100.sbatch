#!/bin/bash
#SBATCH --job-name=medim_eval_steps
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/medim"
CONFIG="$EXPERIMENT/configs/real_anchor_10_official.yaml"
XRV_PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
METRIC_PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
SOURCE_RUN_ID=medim10_official_20260803_001
RUN_100=medim10_official_native_20260803_001
RUN_1024=medim10_official_native1024_20260803_001

TASK_CACHE="$WORKSPACE/.cache/medim/${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/medim/${SLURM_JOB_ID}"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export TORCH_HOME="$WORKSPACE/.cache/medim/official_image_metrics/torch"
export TMPDIR="$TASK_TMP"
export WANDB_MODE=disabled

run_xrv() {
  local run_id="$1"
  local protected_log="$WORKSPACE/artifacts/protected/medim/runs/$run_id/logs/xrv_eval.log"
  if "$XRV_PYTHON" -m medim_ehr_prompt.consistency \
    --config "$CONFIG" \
    --run-id "$run_id" \
    --source-run-id "$SOURCE_RUN_ID" \
    >"$protected_log" 2>&1; then
    chmod 600 "$protected_log"
    printf 'stage=xrv run_id=%s status=completed\n' "$run_id"
  else
    chmod 600 "$protected_log"
    printf 'stage=xrv run_id=%s status=failed\n' "$run_id"
    return 1
  fi
}

run_official_metrics() {
  local run_id="$1"
  local protected_log="$WORKSPACE/artifacts/protected/medim/runs/$run_id/logs/official_metrics_eval.log"
  if "$METRIC_PYTHON" -m medim_ehr_prompt.official_image_metrics \
    --config "$CONFIG" \
    --run-id "$run_id" \
    --source-run-id "$SOURCE_RUN_ID" \
    >"$protected_log" 2>&1; then
    chmod 600 "$protected_log"
    printf 'stage=official_metrics run_id=%s status=completed\n' "$run_id"
  else
    chmod 600 "$protected_log"
    printf 'stage=official_metrics run_id=%s status=failed\n' "$run_id"
    return 1
  fi
}

run_xrv "$RUN_100"
run_official_metrics "$RUN_100"
run_xrv "$RUN_1024"
run_official_metrics "$RUN_1024"

printf 'status=completed compared_runs=2 sample_count_per_run=10\n'

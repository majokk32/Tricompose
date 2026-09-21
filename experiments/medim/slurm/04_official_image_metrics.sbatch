#!/bin/bash
#SBATCH --job-name=medim_official_metrics
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/medim"
CONFIG="$EXPERIMENT/configs/real_anchor_50.yaml"
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
RUN_ID="${1:?usage: sbatch 04_official_image_metrics.sbatch OPAQUE_RUN_ID}"

METRIC_CACHE="$WORKSPACE/.cache/medim/official_image_metrics"
TASK_TMP="$WORKSPACE/.tmp/medim/${SLURM_JOB_ID}"
mkdir -p "$METRIC_CACHE" "$TASK_TMP"
chmod 700 "$METRIC_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$METRIC_CACHE/xdg"
export TORCH_HOME="$METRIC_CACHE/torch"
export TMPDIR="$TASK_TMP"
export WANDB_MODE=disabled

"$PYTHON" -m medim_ehr_prompt.official_image_metrics \
  --config "$CONFIG" \
  --run-id "$RUN_ID"

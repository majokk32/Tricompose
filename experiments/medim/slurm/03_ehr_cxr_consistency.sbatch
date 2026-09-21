#!/bin/bash
#SBATCH --job-name=medim_ehr_score
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
CONFIG="$EXPERIMENT/configs/real_anchor_50.yaml"
PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
RUN_ID="${1:?usage: sbatch 03_ehr_cxr_consistency.sbatch OPAQUE_RUN_ID}"

TASK_CACHE="$WORKSPACE/.cache/medim/${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/medim/${SLURM_JOB_ID}"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"

"$PYTHON" -m medim_ehr_prompt.consistency \
  --config "$CONFIG" \
  --run-id "$RUN_ID"

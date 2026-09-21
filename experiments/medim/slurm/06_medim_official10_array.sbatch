#!/bin/bash
#SBATCH --job-name=medim_off10_cxr
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%A_%a.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%A_%a.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/medim"
CONFIG="$EXPERIMENT/configs/real_anchor_10_official.yaml"
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
RUN_ID="${1:?usage: sbatch 06_medim_official10_array.sbatch OPAQUE_RUN_ID}"
NUM_SHARDS=2

TASK_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TASK_CACHE="$WORKSPACE/.cache/medim/$TASK_KEY"
TASK_TMP="$WORKSPACE/.tmp/medim/$TASK_KEY"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

"$PYTHON" -m medim_ehr_prompt.medim_adapter \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS"

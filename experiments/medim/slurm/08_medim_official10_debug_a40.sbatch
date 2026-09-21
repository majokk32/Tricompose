#!/bin/bash
#SBATCH --job-name=medim_off10_dbg
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/medim"
CONFIG="$EXPERIMENT/configs/real_anchor_10_official_fast.yaml"
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
RUN_ID="${1:?usage: sbatch 08_medim_official10_debug_a40.sbatch OPAQUE_RUN_ID}"

TASK_KEY="${SLURM_JOB_ID}"
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
  --task-index 0 \
  --num-shards 1

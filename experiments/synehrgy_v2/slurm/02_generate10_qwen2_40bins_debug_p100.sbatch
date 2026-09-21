#!/bin/bash
#SBATCH --job-name=synehrgy_qwen_g10
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/synehrgy_v2/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/synehrgy_v2/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/synehrgy_v2"
PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
MODEL_DIR="$WORKSPACE/runtime/models/synehrgy_v2/qwen2-40bins"
RUN_ID="${1:?usage: sbatch 02_generate10_qwen2_40bins_debug_p100.sbatch RUN_ID}"
TASK_CACHE="$WORKSPACE/.cache/synehrgy_v2/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/synehrgy_v2/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$MODEL_DIR"; then
  printf 'status=failed reason=missing_model_snapshot\n' >&2
  exit 1
fi
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m tricompose_synehrgy_v2.adapter \
  --model-variant qwen2-40bins \
  --model-dir "$MODEL_DIR" \
  --run-id "$RUN_ID" \
  --count 10 \
  --seed 42 \
  --temperature 1.0 \
  --top-k 50

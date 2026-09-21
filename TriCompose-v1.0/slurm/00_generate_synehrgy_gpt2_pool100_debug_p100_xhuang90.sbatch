#!/bin/bash
#SBATCH --job-name=tri_ehr_gpt2_100
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/synehrgy_v2"
PYTHON="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/venvs/synehrgy-v2/bin/python"
MODEL_DIR="$WORKSPACE/SynEHRgy-v2/saved_models/gpt2-10bins"
RUN_ID="${1:?usage: sbatch 00_generate_synehrgy_qwen2_pool100_debug_p100.sbatch RUN_ID}"
TASK_CACHE="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/cache/synehrgy_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/tmp/synehrgy_${SLURM_JOB_ID}"

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

export PYTHONPATH="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/src:$WORKSPACE/TriCompose-v1.0/src:$EXPERIMENT/src"
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
  --model-variant gpt2-10bins \
  --model-dir "$MODEL_DIR" \
  --run-id "$RUN_ID" \
  --count 100 \
  --seed 4200 \
  --temperature 1.0 \
  --top-k 50

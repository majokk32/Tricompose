#!/bin/bash
#SBATCH --job-name=tri_p100_rgv2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
REVISION=c88d2bf041a448f505fc13dc009d6991be1c9b0f
MODEL_DIR="$WORKSPACE/runtime/models/roentgen-v2/$REVISION"
REQUEST_RUN="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/artifacts/protected/tricompose_v1/cxr_requests/cxr_requests_gpt2_pool100_s2_20260810_001"
OUTPUT_ROOT="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/artifacts/protected/tricompose_v1/cxr_candidates"
OUTPUT_RUN_ID=cxr_gpt2_pool100_roentgen_v2_s2_20260810_001
TASK_CACHE="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/cache/roentgen_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/tmp/roentgen_${SLURM_JOB_ID}"

for required in "$PYTHON" "$MODEL_DIR" "$REQUEST_RUN"
do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset path=%s\n' "$required" >&2
    exit 1
  fi
done

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/src:$WORKSPACE/TriCompose-v1.0/src:$WORKSPACE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export CUDA_CACHE_PATH="$TASK_CACHE/cuda"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_VERBOSITY=error
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" "$WORKSPACE/tools/run_tricompose_v1_cxr.py" \
  --request-run "$REQUEST_RUN" \
  --output-root "$OUTPUT_ROOT" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --model-id roentgen_v2 \
  --model-dir "$MODEL_DIR" \
  --batch-size 2

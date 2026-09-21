#!/bin/bash
#SBATCH --job-name=tri_p100_pixart
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
REVISION=ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/pixart-sigma/$REVISION"
REQUEST_RUN="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/artifacts/protected/tricompose_v1/cxr_requests/cxr_requests_gpt2_pool100_s2_20260810_001"
OUTPUT_ROOT="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/artifacts/protected/tricompose_v1/cxr_candidates"
OUTPUT_RUN_ID=cxr_gpt2_pool100_pixart_s2_20260810_001
TASK_CACHE="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/cache/pixart_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/TriCompose-v1.0/runtime_xhuang90/tmp/pixart_${SLURM_JOB_ID}"

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
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor"
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
  --model-id chexgenbench_pixart \
  --model-dir "$MODEL_DIR" \
  --batch-size 1

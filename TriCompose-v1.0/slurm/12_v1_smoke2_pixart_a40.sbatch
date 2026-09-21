#!/bin/bash
#SBATCH --job-name=tri_v1_pixart_s2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
REVISION=ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/pixart-sigma/$REVISION"
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1/cxr_requests/v1_smoke2_all_cxr_s2_20260806_001"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/cxr_candidates"
OUTPUT_RUN_ID=v1_smoke2_pixart_s2_20260806_001
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1/pixart_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/pixart_${SLURM_JOB_ID}"

for required in "$PYTHON" "$MODEL_DIR" "$REQUEST_RUN"
do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
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

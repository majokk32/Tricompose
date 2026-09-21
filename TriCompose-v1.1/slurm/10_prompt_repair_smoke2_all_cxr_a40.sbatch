#!/bin/bash
#SBATCH --job-name=tri_v11_cxr_s2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-2
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v11/%x_%A_%a.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v11/%x_%A_%a.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_requests/prompt_repair_smoke2_allmodels_s2_20260813_001"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    MODEL_ID=roentgen_v2
    PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
    REVISION=c88d2bf041a448f505fc13dc009d6991be1c9b0f
    MODEL_DIR="$WORKSPACE/runtime/models/roentgen-v2/$REVISION"
    BATCH_SIZE=2
    ;;
  1)
    MODEL_ID=chexgenbench_sana
    PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
    REVISION=6c8989d92e75fbcfc91a82c3570389293867ad6a
    MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/sana-e20/$REVISION"
    BATCH_SIZE=2
    ;;
  2)
    MODEL_ID=chexgenbench_pixart
    PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
    REVISION=ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727
    MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/pixart-sigma/$REVISION"
    BATCH_SIZE=1
    ;;
  *)
    printf 'status=failed reason=invalid_array_index\n' >&2
    exit 2
    ;;
esac

OUTPUT_RUN_ID="prompt_repair_smoke2_${MODEL_ID}_s2_20260813_001"
TASK_CACHE="$WORKSPACE/.cache/tricompose_v11/${MODEL_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v11/${MODEL_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

for required in "$PYTHON" "$MODEL_DIR" "$REQUEST_RUN"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 2770 "$TASK_CACHE" "$TASK_TMP"

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

"$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_cxr.py" \
  --request-run "$REQUEST_RUN" \
  --output-root "$OUTPUT_ROOT" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --model-id "$MODEL_ID" \
  --model-dir "$MODEL_DIR" \
  --batch-size "$BATCH_SIZE"

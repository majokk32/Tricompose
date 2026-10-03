#!/bin/bash
# Prepared only. Do not submit without user approval.
# 8 direct-fact cases x 3 frozen CXR generators x 2 seeds = 48 images.
# Each array task runs one model over its 16 requests.
#SBATCH --job-name=tri_v11_direct8_cxr
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/job_logs/%x_%A_%a.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/job_logs/%x_%A_%a.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_requests/directfacts8_all3_s2_20260922_001"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates"
TASK_RUNTIME="$WORKSPACE/artifacts/protected/tricompose_v1_1/job_logs/runtime_direct8_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

PRECISION_ARGS=()
case "$SLURM_ARRAY_TASK_ID" in
  0)
    MODEL_ID=roentgen_v2
    PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
    MODEL_DIR="$WORKSPACE/runtime/models/roentgen-v2/c88d2bf041a448f505fc13dc009d6991be1c9b0f"
    PRECISION_ARGS=(--precision float16)
    BATCH_SIZE=2
    ;;
  1)
    MODEL_ID=chexgenbench_sana
    PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
    MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/sana-e20/6c8989d92e75fbcfc91a82c3570389293867ad6a"
    BATCH_SIZE=2
    ;;
  2)
    MODEL_ID=chexgenbench_pixart
    PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
    MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/pixart-sigma/ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727"
    BATCH_SIZE=1
    ;;
  *)
    printf 'status=failed reason=invalid_array_index\n'
    exit 2
    ;;
esac

OUTPUT_RUN_ID="directfacts8_${MODEL_ID}_s2_${SLURM_ARRAY_JOB_ID}"
for required in "$PYTHON" "$MODEL_DIR" "$REQUEST_RUN/manifest.json"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n'
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID" || test -e "$TASK_RUNTIME"; then
  printf 'status=failed reason=output_run_exists\n'
  exit 1
fi
mkdir -m 2770 "$TASK_RUNTIME"
mkdir -m 2770 "$TASK_RUNTIME/cache" "$TASK_RUNTIME/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_RUNTIME/cache/xdg"
export HF_HOME="$TASK_RUNTIME/cache/huggingface"
export TORCH_HOME="$TASK_RUNTIME/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$TASK_RUNTIME/cache/torchinductor"
export CUDA_CACHE_PATH="$TASK_RUNTIME/cache/cuda"
export TMPDIR="$TASK_RUNTIME/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFUSERS_VERBOSITY=error TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if "$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_cxr.py" \
  --request-run "$REQUEST_RUN" \
  --output-root "$OUTPUT_ROOT" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --model-id "$MODEL_ID" --model-dir "$MODEL_DIR" \
  --batch-size "$BATCH_SIZE" "${PRECISION_ARGS[@]}" \
  >"$TASK_RUNTIME/model.stdout.log" 2>"$TASK_RUNTIME/model.stderr.log"; then
  printf 'status=completed model=%s\n' "$MODEL_ID"
else
  printf 'status=failed reason=see_protected_model_log model=%s\n' "$MODEL_ID"
  exit 1
fi

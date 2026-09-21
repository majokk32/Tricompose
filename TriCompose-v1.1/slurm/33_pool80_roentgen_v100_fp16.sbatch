#!/bin/bash
#SBATCH --job-name=tri_v11_c80_rg16
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
REVISION=c88d2bf041a448f505fc13dc009d6991be1c9b0f
MODEL_DIR="$WORKSPACE/runtime/models/roentgen-v2/$REVISION"
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_requests/pool80_all3_s1_20260813_b208"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates"
OUTPUT_RUN_ID=pool80_roentgen_v2_fp16_s1_20260813_0bd4
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1_1/roentgen_fp16_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1_1/roentgen_fp16_${SLURM_JOB_ID}"

for required in "$PYTHON" "$MODEL_DIR" "$REQUEST_RUN/manifest.json"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

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
  --model-id roentgen_v2 \
  --model-dir "$MODEL_DIR" \
  --precision float16 \
  --batch-size 4

#!/bin/bash
#SBATCH --job-name=tri_v11_r160_llava
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/llavarad/bin/python
EXTERNAL_ROOT=/project2/ruishanl_1185/SDP_for_VLM/SDP_for_VLM/llava-rad
MODEL_DIR=/project2/ruishanl_1185/SDP_for_VLM/models/llava_rad
MODEL_BASE=/project2/ruishanl_1185/SDP_for_VLM/models/vicuna-7b-v1.5
BIOMEDBERT_DIR=/home1/yikeyang/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_requests/pool160_four_experts_20260813_e5c4"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates"
OUTPUT_RUN_ID=pool160_llavarad_20260813_aa39
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1_1/llavarad_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1_1/llavarad_${SLURM_JOB_ID}"

for required in "$PYTHON" "$EXTERNAL_ROOT" "$MODEL_DIR" "$MODEL_BASE" \
  "$BIOMEDBERT_DIR" "$REQUEST_RUN/manifest.json"; do
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
export HUGGINGFACE_HUB_CACHE="$TASK_CACHE/huggingface/hub"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface/transformers"
export TORCH_HOME="$TASK_CACHE/torch"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export CUDA_CACHE_PATH="$TASK_CACHE/cuda"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_report.py" \
  --request-run "$REQUEST_RUN" \
  --output-root "$OUTPUT_ROOT" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --model-id llavarad \
  --model-dir "$MODEL_DIR" \
  --model-base "$MODEL_BASE" \
  --biomedbert-dir "$BIOMEDBERT_DIR" \
  --external-root "$EXTERNAL_ROOT" \
  --runtime-dir "$TASK_TMP/model_runtime"

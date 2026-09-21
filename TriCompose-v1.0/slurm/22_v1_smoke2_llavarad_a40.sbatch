#!/bin/bash
#SBATCH --job-name=tri_v1_llavarad_all
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/llavarad/bin/python
EXTERNAL_ROOT=/project2/ruishanl_1185/SDP_for_VLM/SDP_for_VLM/llava-rad
MODEL_DIR=/project2/ruishanl_1185/SDP_for_VLM/models/llava_rad
MODEL_BASE=/project2/ruishanl_1185/SDP_for_VLM/models/vicuna-7b-v1.5
BIOMEDBERT_DIR=/home1/yikeyang/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d
CXR_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/cxr_candidates"
REPORT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/report_candidates"
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1/llavarad_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/llavarad_${SLURM_JOB_ID}"

for required in "$PYTHON" "$EXTERNAL_ROOT" "$MODEL_DIR" "$MODEL_BASE" \
  "$BIOMEDBERT_DIR" \
  "$CXR_ROOT/v1_smoke2_roentgen_v2_s2_20260806_001" \
  "$CXR_ROOT/v1_smoke2_sana_s2_20260806_001" \
  "$CXR_ROOT/v1_smoke2_pixart_s2_20260806_001"
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

"$PYTHON" "$WORKSPACE/tools/run_tricompose_v1_report.py" \
  --cxr-run "$CXR_ROOT/v1_smoke2_roentgen_v2_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_sana_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_pixart_s2_20260806_001" \
  --output-root "$REPORT_ROOT" \
  --output-run-id v1_smoke2_llavarad_allcxr_20260806_001 \
  --model-id llavarad \
  --model-dir "$MODEL_DIR" \
  --model-base "$MODEL_BASE" \
  --biomedbert-dir "$BIOMEDBERT_DIR" \
  --external-root "$EXTERNAL_ROOT" \
  --runtime-dir "$TASK_TMP/model_runtime"

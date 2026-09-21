#!/bin/bash
#SBATCH --job-name=llavarad_rpt50
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/llavarad_report/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/llavarad_report/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/llavarad_report"
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/llavarad/bin/python
EXTERNAL_ROOT=/project2/ruishanl_1185/SDP_for_VLM/SDP_for_VLM/llava-rad
MODEL_DIR=/project2/ruishanl_1185/SDP_for_VLM/models/llava_rad
MODEL_BASE=/project2/ruishanl_1185/SDP_for_VLM/models/vicuna-7b-v1.5
BIOMEDBERT_DIR=/home1/yikeyang/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d
SOURCE_MODEL=sana
SOURCE_RUN_ID=cgb_sana50_fixed_20260804_002
OUTPUT_RUN_ID="${1:?usage: sbatch 01_report50_debug_a40.sbatch OUTPUT_RUN_ID}"
TASK_CACHE="$WORKSPACE/.cache/llavarad_report/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/llavarad_report/job_${SLURM_JOB_ID}"

for required in \
  "$PYTHON" \
  "$EXTERNAL_ROOT" \
  "$MODEL_DIR" \
  "$MODEL_BASE" \
  "$BIOMEDBERT_DIR"
do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$EXTERNAL_ROOT"
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

"$PYTHON" -m tricompose_llavarad_report.adapter \
  --external-root "$EXTERNAL_ROOT" \
  --model-dir "$MODEL_DIR" \
  --model-base "$MODEL_BASE" \
  --biomedbert-dir "$BIOMEDBERT_DIR" \
  --runtime-dir "$TASK_TMP/model_runtime" \
  --source-model "$SOURCE_MODEL" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --limit 50

#!/bin/bash
#SBATCH --job-name=qwen25vl_rpt50_v100
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/cxr_report_consistency/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/cxr_report_consistency/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
MODEL_DIR=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct
SOURCE_MODEL=sana
SOURCE_RUN_ID=cgb_sana50_fixed_20260804_002
MAIRA_RUN_ID=maira2_rpt50_20260805_001
CXRMATE_RUN_ID=cxrmate1_rpt50_20260805_001
OUTPUT_RUN_ID="${1:?usage: sbatch 02_qwen25vl_report50_v100.sbatch OUTPUT_RUN_ID}"
LIMIT="${2:-50}"
OUTPUT_DIR="$WORKSPACE/artifacts/protected/cxr_report_consistency/runs/$OUTPUT_RUN_ID"
TASK_CACHE="$WORKSPACE/.cache/cxr_report_consistency/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/cxr_report_consistency/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf '{"stage":"qwen25vl_cxr_report_batch","status":"failed","error_type":"MissingEnvironment"}\n' >&2
  exit 1
fi
if ! test -d "$MODEL_DIR"; then
  printf '{"stage":"qwen25vl_cxr_report_batch","status":"failed","error_type":"MissingModel"}\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export HF_HOME="$TASK_CACHE/huggingface"
export HUGGINGFACE_HUB_CACHE="$TASK_CACHE/huggingface/hub"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface/transformers"
export HF_MODULES_CACHE="$TASK_CACHE/huggingface/modules"
export TORCH_HOME="$TASK_CACHE/torch"
export CUDA_CACHE_PATH="$TASK_CACHE/cuda"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" \
  -m tricompose.verifiers.qwenvl_cxr_report_batch \
  --run-id "$OUTPUT_RUN_ID" \
  --model-path "$MODEL_DIR" \
  --source-model "$SOURCE_MODEL" \
  --source-run-id "$SOURCE_RUN_ID" \
  --maira-run-id "$MAIRA_RUN_ID" \
  --cxrmate-run-id "$CXRMATE_RUN_ID" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --max-new-tokens 128

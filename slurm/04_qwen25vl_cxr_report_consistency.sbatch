#!/bin/bash
#SBATCH --job-name=tc_qwen25vl_verify
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/qwen2_5_vl/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/qwen2_5_vl/%x_%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=/project2/ruishanl_1185/inference_3mod
QWEN_ENV=/project2/ruishanl_1185/SDP_for_VLM/envs/attack
QWEN_MODEL=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct
RUN_ID=${TRICOMPOSE_RUN_ID:-real_smoke_003}

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo '{"stage":"qwenvl_cxr_report_verifier","status":"failed","error_type":"InvalidRunId"}'
  exit 2
fi

RUNTIME_DIR="$PROJECT_ROOT/runtime/$SLURM_JOB_ID"
mkdir -p \
  "$RUNTIME_DIR/xdg" \
  "$RUNTIME_DIR/torch" \
  "$RUNTIME_DIR/huggingface" \
  "$RUNTIME_DIR/cuda" \
  "$RUNTIME_DIR/tmp"
chmod -R 700 "$RUNTIME_DIR"

export XDG_CACHE_HOME="$RUNTIME_DIR/xdg"
export TORCH_HOME="$RUNTIME_DIR/torch"
export HF_HOME="$RUNTIME_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$RUNTIME_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$RUNTIME_DIR/huggingface/transformers"
export CUDA_CACHE_PATH="$RUNTIME_DIR/cuda"
export TMPDIR="$RUNTIME_DIR/tmp"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

srun --ntasks=1 "$QWEN_ENV/bin/python" \
  -m tricompose.verifiers.qwenvl_cxr_report \
  --run-id "$RUN_ID" \
  --model-path "$QWEN_MODEL" \
  --input-image "$PROJECT_ROOT/artifacts/protected/$RUN_ID/stage1_ehrxdiff/generated_cxr.png" \
  --candidate unidisc "$PROJECT_ROOT/artifacts/protected/$RUN_ID/stage2_unidisc/generated_report.txt" \
  --candidate llavarad "$PROJECT_ROOT/artifacts/protected/$RUN_ID/stage2_llavarad/generated_report.txt" \
  --output-dir "$PROJECT_ROOT/artifacts/protected/$RUN_ID/verification/qwen25vl_v1" \
  --max-new-tokens 128

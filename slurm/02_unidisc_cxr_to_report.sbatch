#!/bin/bash
#SBATCH --job-name=tc_unidisc_s2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/unidisc/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/unidisc/%x_%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=/project2/ruishanl_1185/inference_3mod
UNIDISC_ROOT=/project2/ruishanl_1185/Baselines/UniDisc_baseline
RUN_ID=${TRICOMPOSE_RUN_ID:-real_smoke_001}

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo '{"stage":"unidisc_cxr_to_report","status":"failed","error_type":"InvalidRunId"}'
  exit 2
fi

RUNTIME_DIR="$PROJECT_ROOT/runtime/$SLURM_JOB_ID"
mkdir -p \
  "$RUNTIME_DIR/xdg" \
  "$RUNTIME_DIR/torch" \
  "$RUNTIME_DIR/huggingface" \
  "$RUNTIME_DIR/matplotlib" \
  "$RUNTIME_DIR/cuda" \
  "$RUNTIME_DIR/tmp"
chmod -R 700 "$RUNTIME_DIR"

export XDG_CACHE_HOME="$RUNTIME_DIR/xdg"
export TORCH_HOME="$RUNTIME_DIR/torch"
export HF_HOME="$RUNTIME_DIR/huggingface"
export MPLCONFIGDIR="$RUNTIME_DIR/matplotlib"
export CUDA_CACHE_PATH="$RUNTIME_DIR/cuda"
export TMPDIR="$RUNTIME_DIR/tmp"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_ROOT"
source "$UNIDISC_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src:$UNIDISC_ROOT${PYTHONPATH:+:$PYTHONPATH}"

srun --ntasks=1 python -m tricompose.adapters.unidisc_cxr_to_report \
  --external-root "$UNIDISC_ROOT" \
  --checkpoint "$UNIDISC_ROOT/ckpts/unidisc_interleaved/unidisc_interleaved.pt" \
  --tokenizer-path /home1/yikeyang/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --input-image "$PROJECT_ROOT/artifacts/protected/$RUN_ID/stage1_ehrxdiff/generated_cxr.png" \
  --output-dir "$PROJECT_ROOT/artifacts/protected/$RUN_ID/stage2_unidisc" \
  --seed 42 \
  --max-tokens 128 \
  --resolution 256 \
  --sampling-steps 35 \
  --temperature 0.9 \
  --top-p 0.95 \
  --cfg 3.5

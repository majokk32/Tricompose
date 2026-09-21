#!/bin/bash
#SBATCH --job-name=tc_ehrxdiff_s1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/ehrxdiff/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/ehrxdiff/%x_%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=/project2/ruishanl_1185/inference_3mod
EHRXDIFF_ROOT=/project2/ruishanl_1185/EHRXDiff_baseline/EHRXDiff
RUN_ID=${TRICOMPOSE_RUN_ID:-real_smoke_001}
SAMPLE_INDEX=${TRICOMPOSE_SAMPLE_INDEX:-0}

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo '{"stage":"ehrxdiff_longitudinal_to_cxr","status":"failed","error_type":"InvalidRunId"}'
  exit 2
fi
if [[ ! "$SAMPLE_INDEX" =~ ^[0-9]+$ ]]; then
  echo '{"stage":"ehrxdiff_longitudinal_to_cxr","status":"failed","error_type":"InvalidSampleIndex"}'
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
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_ROOT"
source /home1/yikeyang/envs/ehrxdiff_env/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src:$EHRXDIFF_ROOT${PYTHONPATH:+:$PYTHONPATH}"

srun --ntasks=1 python -m tricompose.adapters.ehrxdiff_longitudinal \
  --external-root "$EHRXDIFF_ROOT" \
  --checkpoint /project2/ruishanl_1185/EHRXDiff_baseline/local_data/ehrxdiff_outputs/train_logs/2026-06-05T00-39-27_plusreport_intervalehr_1k_conditioner_only_optfreeze_lr1e6_step200/checkpoints/last.ckpt \
  --autoencoder-checkpoint /project2/ruishanl_1185/EHRXDiff_baseline/local_data/ehrxdiff_work/trained_models/cheff_autoencoder.pt \
  --clip-bootstrap-checkpoint /home1/yikeyang/.cache/clip/ViT-B-32.pt \
  --condition-h5 /project2/ruishanl_1185/EHRXDiff_baseline/local_data/ehrxdiff_outputs/artifacts/ehrxdiff_plusreport_intervalehr_data_1k_trainval/tab/mimiciv-cxr-filtered_test_openai.h5 \
  --image-root /project2/ruishanl_1185/EHRXDiff_baseline/local_data/ehrxdiff_outputs/artifacts/intervalehr_1k_images/images \
  --sample-index "$SAMPLE_INDEX" \
  --output-dir "$PROJECT_ROOT/artifacts/protected/$RUN_ID/stage1_ehrxdiff" \
  --seed 42 \
  --sampling-steps 35 \
  --eta 1.0 \
  --max-event-len 1024

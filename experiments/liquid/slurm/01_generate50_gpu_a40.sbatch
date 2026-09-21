#!/bin/bash
#SBATCH --job-name=liquid_ehr50
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/liquid/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/liquid/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/liquid"
ROENTGEN_EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
LIQUID_ROOT=/project2/ruishanl_1185/Baselines/Liquid_baseline
EVALUATION_ROOT="$LIQUID_ROOT/evaluation"
MODEL_DIR=/project2/ruishanl_1185/MeDiM/pretrained_models/Liquid_V1_7B
VQ_CONFIG="$EVALUATION_ROOT/chameleon/vqgan.yaml"
VQ_CHECKPOINT="$EVALUATION_ROOT/chameleon/vqgan.ckpt"
PYTHON="$LIQUID_ROOT/liquid_env/bin/python"
SOURCE_RUN_ID=medim50_20260803_001
OUTPUT_RUN_ID="${1:?usage: sbatch 01_generate50_gpu_a40.sbatch OUTPUT_RUN_ID}"
TASK_CACHE="$WORKSPACE/.cache/liquid/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/liquid/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$MODEL_DIR"; then
  printf 'status=failed reason=missing_model_snapshot\n' >&2
  exit 1
fi
if ! test -f "$VQ_CHECKPOINT"; then
  printf 'status=failed reason=missing_vq_checkpoint\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$ROENTGEN_EXPERIMENT/src:$EVALUATION_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -m tricompose_liquid.adapter \
  --model-dir "$MODEL_DIR" \
  --evaluation-root "$EVALUATION_ROOT" \
  --vq-config "$VQ_CONFIG" \
  --vq-checkpoint "$VQ_CHECKPOINT" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --limit 50 \
  --batch-size 4

#!/bin/bash
#SBATCH --job-name=radedit_smoke2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/radedit/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/radedit/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/radedit"
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
BUNDLE_REVISION=e8ebd31396ff8553c34084b98b5defb8ebea2817
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/radedit/$BUNDLE_REVISION"
SOURCE_RUN_ID=medim50_20260803_001
OUTPUT_RUN_ID="${1:?usage: sbatch 01_smoke2_debug_a40.sbatch OUTPUT_RUN_ID}"
TASK_CACHE="$WORKSPACE/.cache/radedit/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/radedit/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$MODEL_DIR"; then
  printf 'status=failed reason=missing_model_bundle\n' >&2
  exit 1
fi
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_VERBOSITY=error
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m tricompose_radedit.adapter \
  --model-dir "$MODEL_DIR" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --limit 2 \
  --batch-size 1

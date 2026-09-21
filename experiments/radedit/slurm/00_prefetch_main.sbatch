#!/bin/bash
#SBATCH --job-name=radedit_fetch
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/radedit/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/radedit/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/radedit"
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
BUNDLE_REVISION=e8ebd31396ff8553c34084b98b5defb8ebea2817
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/radedit/$BUNDLE_REVISION"
STAGING_DIR="$WORKSPACE/.tmp/radedit/model_${SLURM_JOB_ID}"
CACHE_DIR="$WORKSPACE/.cache/radedit/huggingface"
TOKEN_PATH="$WORKSPACE/.cache/roentgen_v2/huggingface/token"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -f "$TOKEN_PATH"; then
  printf 'status=failed reason=missing_huggingface_token\n' >&2
  exit 1
fi
mkdir -p "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
chmod 700 "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="$CACHE_DIR"
export HF_TOKEN_PATH="$TOKEN_PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1

"$PYTHON" -m tricompose_radedit.prefetch \
  --destination "$MODEL_DIR" \
  --staging-dir "$STAGING_DIR"


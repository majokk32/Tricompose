#!/bin/bash
#SBATCH --job-name=rgv2_fetch
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/roentgen_v2/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/roentgen_v2/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
REVISION=c88d2bf041a448f505fc13dc009d6991be1c9b0f
MODEL_DIR="$WORKSPACE/runtime/models/roentgen-v2/$REVISION"
STAGING_DIR="$WORKSPACE/.tmp/roentgen_v2/model_${SLURM_JOB_ID}"
CACHE_DIR="$WORKSPACE/.cache/roentgen_v2/huggingface"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi

mkdir -p "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
chmod 700 "$CACHE_DIR" "$(dirname "$STAGING_DIR")"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="$CACHE_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1

"$PYTHON" -m tricompose_roentgen_v2.prefetch \
  --destination "$MODEL_DIR" \
  --staging-dir "$STAGING_DIR"

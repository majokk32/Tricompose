#!/bin/bash
#SBATCH --job-name=cgb_pix_fetch
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_pixart/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_pixart/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/chexgenbench_pixart"
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
REVISION=ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/pixart-sigma/$REVISION"
STAGING_DIR="$WORKSPACE/.tmp/chexgenbench_pixart/model_${SLURM_JOB_ID}"
CACHE_DIR="$WORKSPACE/.cache/chexgenbench_pixart/huggingface"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
mkdir -p "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
chmod 700 "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="$CACHE_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1

"$PYTHON" -m tricompose_chexgenbench_pixart.prefetch \
  --destination "$MODEL_DIR" \
  --staging-dir "$STAGING_DIR"


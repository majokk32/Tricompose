#!/bin/bash
#SBATCH --job-name=cgb_sana_fetch
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_sana/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_sana/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/chexgenbench_sana"
ROENTGEN_EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
REVISION=6c8989d92e75fbcfc91a82c3570389293867ad6a
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/sana-e20/$REVISION"
STAGING_DIR="$WORKSPACE/.tmp/chexgenbench_sana/model_${SLURM_JOB_ID}"
CACHE_DIR="$WORKSPACE/.cache/chexgenbench_sana/huggingface"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
mkdir -p "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
chmod 700 "$CACHE_DIR" "$(dirname "$STAGING_DIR")" "$(dirname "$MODEL_DIR")"
export PYTHONPATH="$WORKSPACE/src:$ROENTGEN_EXPERIMENT/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="$CACHE_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1

"$PYTHON" -m tricompose_chexgenbench_sana.prefetch \
  --destination "$MODEL_DIR" \
  --staging-dir "$STAGING_DIR"


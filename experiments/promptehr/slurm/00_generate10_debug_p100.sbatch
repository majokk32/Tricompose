#!/bin/bash
#SBATCH --job-name=promptehr_g10
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/promptehr/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/promptehr/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/promptehr"
PYTHON="$WORKSPACE/runtime/venvs/promptehr-cu117-v1/bin/python"
CHECKPOINT_DIR="$WORKSPACE/PromptEHR/checkpoints/official_mimiciii_20221017"
DEMO_DATA_DIR="$WORKSPACE/PromptEHR/demo_data/synthetic_ehr"
ASSET_ROOT="$WORKSPACE/PromptEHR/assets"
RUN_ID="${1:?usage: sbatch 00_generate10_debug_p100.sbatch RUN_ID}"
TASK_CACHE="$WORKSPACE/.cache/promptehr/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/promptehr/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$CHECKPOINT_DIR"; then
  printf 'status=failed reason=missing_checkpoint\n' >&2
  exit 1
fi
if ! test -f "$DEMO_DATA_DIR/data.pkl"; then
  printf 'status=failed reason=missing_official_synthetic_demo\n' >&2
  exit 1
fi
if ! test -f "$ASSET_ROOT/facebook/bart-base/config.json"; then
  printf 'status=failed reason=missing_bart_config_asset\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/PromptEHR:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONBREAKPOINT=0
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TRANSFORMERS_CACHE="$TASK_CACHE/transformers"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$ASSET_ROOT"
"$PYTHON" -m tricompose_promptehr.adapter \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --demo-data-dir "$DEMO_DATA_DIR" \
  --run-id "$RUN_ID" \
  --count 10 \
  --selection-seed 20260805 \
  --generation-seed 42


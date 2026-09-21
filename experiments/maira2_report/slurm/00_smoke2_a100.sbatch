#!/bin/bash
#SBATCH --job-name=maira2_rpt_smoke2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/maira2_report/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/maira2_report/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/maira2_report"
PYTHON="$WORKSPACE/maria-2/venv/bin/python"
MODEL_DIR=/project2/ruishanl_1185/Baselines/hf_cache/hub/models--microsoft--maira-2/snapshots/795a2b1cd4a310624b4e3d14b5a23e41fd273deb
SOURCE_MODEL=sana
SOURCE_RUN_ID=cgb_sana50_fixed_20260804_002
OUTPUT_RUN_ID="${1:?usage: sbatch 00_smoke2_a100.sbatch OUTPUT_RUN_ID}"
TASK_CACHE="$WORKSPACE/.cache/maira2_report/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/maira2_report/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$MODEL_DIR"; then
  printf 'status=failed reason=missing_model_snapshot\n' >&2
  exit 1
fi
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export HF_MODULES_CACHE="$TASK_CACHE/huggingface/modules"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m tricompose_maira2_report.adapter \
  --model-dir "$MODEL_DIR" \
  --source-model "$SOURCE_MODEL" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --limit 2

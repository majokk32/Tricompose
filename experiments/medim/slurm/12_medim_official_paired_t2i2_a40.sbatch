#!/bin/bash
#SBATCH --job-name=medim_pair2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/medim"
OFFICIAL_REPO=/project2/ruishanl_1185/MeDiM/MeDiM_original_infer
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
SOURCE_RUN_ID="${1:?usage: sbatch 12_medim_official_paired_t2i2_a40.sbatch SOURCE_RUN_ID OUTPUT_RUN_ID [SOURCE_CASE_ID]}"
OUTPUT_RUN_ID="${2:?usage: sbatch 12_medim_official_paired_t2i2_a40.sbatch SOURCE_RUN_ID OUTPUT_RUN_ID [SOURCE_CASE_ID]}"
SOURCE_CASE_ID="${3:-case_000}"

TASK_KEY="${SLURM_JOB_ID}"
TASK_CACHE="$WORKSPACE/.cache/medim/$TASK_KEY"
TASK_TMP="$WORKSPACE/.tmp/medim/$TASK_KEY"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$OFFICIAL_REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export UNIDISC_DIR="$OFFICIAL_REPO"

"$PYTHON" -m medim_ehr_prompt.official_paired_t2i \
  --source-run-id "$SOURCE_RUN_ID" \
  --source-case-id "$SOURCE_CASE_ID" \
  --output-run-id "$OUTPUT_RUN_ID"

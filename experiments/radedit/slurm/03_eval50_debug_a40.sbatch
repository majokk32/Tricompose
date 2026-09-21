#!/bin/bash
#SBATCH --job-name=radedit50_eval
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/radedit/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/radedit/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/cxr_baseline_eval"
ROENTGEN_EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
MEDIM_SOURCE="$WORKSPACE/experiments/medim/src"
XRV_SITE=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/lib/python3.11/site-packages
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
CONFIG="$EXPERIMENT/configs/eval_n50.yaml"
MODEL_NAME=radedit
GENERATION_RUN_ID=radedit50_20260804_001
EVALUATION_RUN_ID=radedit50_baseline_eval_20260804_001
TASK_CACHE="$WORKSPACE/.cache/cxr_baseline_eval/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/cxr_baseline_eval/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if test -e "$WORKSPACE/artifacts/protected/radedit/evaluations/$EVALUATION_RUN_ID"; then
  printf 'status=failed reason=evaluation_run_exists\n' >&2
  exit 1
fi
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$ROENTGEN_EXPERIMENT/src:$MEDIM_SOURCE:$XRV_SITE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$MPLCONFIGDIR"
chmod 700 "$MPLCONFIGDIR"

"$PYTHON" -m tricompose_cxr_baseline_eval.evaluate \
  --config "$CONFIG" \
  --model-name "$MODEL_NAME" \
  --generation-run-id "$GENERATION_RUN_ID" \
  --evaluation-run-id "$EVALUATION_RUN_ID"

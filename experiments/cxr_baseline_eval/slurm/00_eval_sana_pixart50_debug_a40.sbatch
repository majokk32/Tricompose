#!/bin/bash
#SBATCH --job-name=cxr_baseline_eval
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --array=0-1%1
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/cxr_baseline_eval/%x_%A_%a.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/cxr_baseline_eval/%x_%A_%a.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/cxr_baseline_eval"
ROENTGEN_EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
MEDIM_SOURCE="$WORKSPACE/experiments/medim/src"
XRV_SITE=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/lib/python3.11/site-packages
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
CONFIG="$EXPERIMENT/configs/eval_n50.yaml"
TASK_CACHE="$WORKSPACE/.cache/cxr_baseline_eval/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TASK_TMP="$WORKSPACE/.tmp/cxr_baseline_eval/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  MODEL_NAME=sana
  GENERATION_RUN_ID=cgb_sana50_fixed_20260804_002
  EVALUATION_RUN_ID=cgb_sana50_baseline_eval_20260804_001
elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  MODEL_NAME=pixart
  GENERATION_RUN_ID=cgb_pixart50_fixed_20260804_001
  EVALUATION_RUN_ID=cgb_pixart50_baseline_eval_20260804_001
else
  printf 'status=failed reason=invalid_array_index\n' >&2
  exit 1
fi

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
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

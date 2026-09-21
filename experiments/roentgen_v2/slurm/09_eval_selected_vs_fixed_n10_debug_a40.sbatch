#!/bin/bash
#SBATCH --job-name=rgv2_cmp10
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/roentgen_v2/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/roentgen_v2/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
XRV_SITE=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/lib/python3.11/site-packages
BIOVIL_VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
MEDIM_SOURCE="$WORKSPACE/experiments/medim/src"
CONFIG="$EXPERIMENT/configs/eval_n10_selected_official_inspired.yaml"
SWEEP_RUN_ID=rgv2_sweep_n10_cfg34_s4_20260804_001
SELECTION_RUN_ID=rgv2_select_n10_xrv_20260804_001
FIXED_RUN_ID=rgv2_fixed_n10_cfg3s0_20260804_001
SELECTED_RUN_ID=rgv2_selected_n10_xrv_20260804_001
FIXED_EVAL_ID=rgv2_eval_fixed_n10_cfg3s0_20260804_001
SELECTED_EVAL_ID=rgv2_eval_selected_n10_xrv_20260804_001

TASK_CACHE="$WORKSPACE/.cache/roentgen_v2/evaluation/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/roentgen_v2/evaluation/job_${SLURM_JOB_ID}"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$MEDIM_SOURCE:$BIOVIL_VENDOR:$XRV_SITE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export TORCH_HOME="$WORKSPACE/.cache/medim/official_image_metrics/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

mkdir -p "$MPLCONFIGDIR"
chmod 700 "$MPLCONFIGDIR"

"$PYTHON" -m tricompose_roentgen_v2.materialize_selection \
  --sweep-run-id "$SWEEP_RUN_ID" \
  --selection-run-id "$SELECTION_RUN_ID" \
  --output-run-id "$FIXED_RUN_ID" \
  --choice baseline

"$PYTHON" -m tricompose_roentgen_v2.materialize_selection \
  --sweep-run-id "$SWEEP_RUN_ID" \
  --selection-run-id "$SELECTION_RUN_ID" \
  --output-run-id "$SELECTED_RUN_ID" \
  --choice selected

"$PYTHON" -m tricompose_roentgen_v2.evaluation \
  --config "$CONFIG" \
  --generation-run-id "$FIXED_RUN_ID" \
  --evaluation-run-id "$FIXED_EVAL_ID"

"$PYTHON" -m tricompose_roentgen_v2.evaluation \
  --config "$CONFIG" \
  --generation-run-id "$SELECTED_RUN_ID" \
  --evaluation-run-id "$SELECTED_EVAL_ID"

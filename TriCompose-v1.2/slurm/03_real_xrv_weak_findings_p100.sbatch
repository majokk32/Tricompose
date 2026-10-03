#!/bin/bash
# Prepared only. Requires explicit user approval before sbatch.
# Weak report-label reference, not independent image truth.
#SBATCH --job-name=tri_real_xrvweak
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
DATASET=/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals
CHEXPERT=/project2/ruishanl_1185/datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz
PREVIOUS="$WORKSPACE/artifacts/protected/tricompose_v1_2/real_validation/real_biovil_valtest128_20260930_001/scores.json"
PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
WEIGHT_NAME=nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
WEIGHT_DIR="$WORKSPACE/artifacts/protected/tricompose_v1_1/evaluation/smoke48/scoring_12257177/cache/xrv"
PROGRAM="$WORKSPACE/TriCompose-v1.2/real_validation/xrv_real_findings.py"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2/real_validation"
OUTPUT_RUN_ID=real_xrv_weak_valtest128_20261001_001
TASK_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/xrv_weak_${SLURM_JOB_ID}"

for required in "$PYTHON" "$DATASET/manifest.csv" "$CHEXPERT" "$PREVIOUS" "$WEIGHT_DIR/$WEIGHT_NAME" "$PROGRAM"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

mkdir "$TASK_ROOT"
chmod 2770 "$TASK_ROOT"
mkdir "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
chmod 2770 "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export TORCH_HOME="$TASK_ROOT/cache/torch"
export TMPDIR="$TASK_ROOT/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$PROGRAM" \
  --dataset-root "$DATASET" \
  --chexpert-labels "$CHEXPERT" \
  --previous-scores "$PREVIOUS" \
  --cache-dir "$WEIGHT_DIR" \
  --weight-filename "$WEIGHT_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$OUTPUT_RUN_ID"

#!/bin/bash
# Show this complete script/resource request and obtain approval before sbatch.
# Frozen verifier only; no new generator outputs and no training.
#SBATCH --job-name=tri_v12_xrv208
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
DATASET=/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals
LABELS=/project2/ruishanl_1185/datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz
PREVIOUS="$ROOT/real_validation/real_xrv_weak_valtest128_20261001_001/scores.json"
COHORT="$ROOT/real_validation/real_reference_coverage_12548196/cohort.json"
COHORT_SHA=7c097915fda822876135b6e4a43b8db485bbab9f3f2c1d92fd33fa4ea4cc9ba0
WEIGHT_DIR="$WORKSPACE/artifacts/protected/tricompose_v1_1/evaluation/smoke48/scoring_12257177/cache/xrv"
WEIGHT_NAME=nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
SYNTHETIC="$WORKSPACE/artifacts/protected/tricompose_v1_1/evaluation/v12_explicit_two_12538545_001/scoring_12538856/xrv/cxr_finding_labels.json"
TASK_ROOT="$ROOT/job_runtime/xrv208_${SLURM_JOB_ID}"

for required in "$PYTHON" "$DATASET/manifest.csv" "$LABELS" "$PREVIOUS" \
  "$COHORT" "$WEIGHT_DIR/$WEIGHT_NAME" "$SYNTHETIC"; do
  test -r "$required"
done
test ! -e "$TASK_ROOT"
mkdir -m 2770 "$TASK_ROOT"
mkdir -m 2770 "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
mkdir -m 2770 "$TASK_ROOT/cache/cuda" "$TASK_ROOT/cache/matplotlib" "$TASK_ROOT/cache/kernels"
chmod 0660 "$ROOT/slurm_logs/tri_v12_xrv208_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_xrv208_${SLURM_JOB_ID}.err"

export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg" HF_HOME="$TASK_ROOT/cache/huggingface"
export TORCH_HOME="$TASK_ROOT/cache/torch" TMPDIR="$TASK_ROOT/tmp"
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export PYTORCH_KERNEL_CACHE_PATH="$TASK_ROOT/cache/kernels"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$PYTHON" "$WORKSPACE/TriCompose-v1.2/real_validation/score_fixed_reference_cohort.py" \
  --dataset-root "$DATASET" --chexpert-labels "$LABELS" \
  --previous-scores "$PREVIOUS" --cohort "$COHORT" --cohort-sha256 "$COHORT_SHA" \
  --cache-dir "$WEIGHT_DIR" --weight-filename "$WEIGHT_NAME" \
  --output-root "$ROOT/real_validation" --run-id "real_xrv_reference_${SLURM_JOB_ID}"

"$PYTHON" "$WORKSPACE/TriCompose-v1.2/real_validation/calibrate_cached_xrv.py" \
  --cached-real-scores "$ROOT/real_validation/real_xrv_reference_${SLURM_JOB_ID}/scores.json" \
  --cached-synthetic-labels "$SYNTHETIC" --dataset-manifest "$DATASET/manifest.csv" \
  --fixed-cohort "$COHORT" --min-per-class 20 \
  --output-root "$ROOT/real_validation" --run-id "real_xrv_thresholds_${SLURM_JOB_ID}"

# Normalize only this new task's generated caches; never follow symlinks.
find "$TASK_ROOT" -type d -exec chmod 2770 {} +
find "$TASK_ROOT" -type f -exec chmod 0660 {} +

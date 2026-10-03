#!/bin/bash
# Metadata-only: show this full script and obtain approval before submission.
#SBATCH --job-name=tri_v12_refcov
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
DATASET=/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals
LABELS=/project2/ruishanl_1185/datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PREVIOUS="$ROOT/real_validation/real_xrv_weak_valtest128_20261001_001/scores.json"
TASK_ROOT="$ROOT/job_runtime/refcov_${SLURM_JOB_ID}"

for required in "$PYTHON" "$DATASET/manifest.csv" "$LABELS" "$PREVIOUS"; do
  test -r "$required"
done
test ! -e "$TASK_ROOT"
mkdir -m 2770 "$TASK_ROOT"
mkdir -m 2770 "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
chmod 0660 "$ROOT/slurm_logs/tri_v12_refcov_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_refcov_${SLURM_JOB_ID}.err"

export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg" HF_HOME="$TASK_ROOT/cache/huggingface"
export TORCH_HOME="$TASK_ROOT/cache/torch" TMPDIR="$TASK_ROOT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$PYTHON" "$WORKSPACE/TriCompose-v1.2/real_validation/audit_reference_coverage.py" \
  --dataset-root "$DATASET" --chexpert-labels "$LABELS" \
  --previous-scores "$PREVIOUS" --seed 2711 --min-per-class 20 \
  --max-cases-per-split 512 --output-root "$ROOT/real_validation" \
  --run-id "real_reference_coverage_${SLURM_JOB_ID}"

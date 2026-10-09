#!/bin/bash
# Prepared only. Requires approval of this exact script and resource request.
# Reads authorized manual dev text internally; no model calls, GPU or training.
#SBATCH --job-name=tri_opacity_gold
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
cd "$WORKSPACE"
chmod 0660 "artifacts/protected/tricompose_v1_2/slurm_logs/tri_opacity_gold_${SLURM_JOB_ID}.out" \
  "artifacts/protected/tricompose_v1_2/slurm_logs/tri_opacity_gold_${SLURM_JOB_ID}.err"
TASK_RUNTIME="$WORKSPACE/artifacts/protected/tricompose_v1_2/task_runtime/manual_opacity_${SLURM_JOB_ID}"
test ! -e "$TASK_RUNTIME"
mkdir -m 2770 "$TASK_RUNTIME" "$TASK_RUNTIME/tmp" "$TASK_RUNTIME/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_RUNTIME/tmp" XDG_CACHE_HOME="$TASK_RUNTIME/cache"
export HF_HOME="$TASK_RUNTIME/cache/huggingface" TORCH_HOME="$TASK_RUNTIME/cache/torch"
export TRICOMPOSE_MANUAL_OPACITY_REQUEST=manual-opacity-inventory-request-v1
timeout --signal=TERM --kill-after=10s 240s \
  runtime/venvs/report-context-v12-12576792/bin/python \
  TriCompose-v1.2/tools/inventory_manual_opacity_dev.py

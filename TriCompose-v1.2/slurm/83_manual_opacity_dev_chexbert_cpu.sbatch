#!/bin/bash
# Prepared only; approval of this exact script/resources is required to submit.
# Authorized dev reports -> frozen CPU CheXbert. No training, GPU or download.
#SBATCH --job-name=tri_opacity_reader
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
cd "$WORKSPACE"
chmod 0660 "artifacts/protected/tricompose_v1_2/slurm_logs/tri_opacity_reader_${SLURM_JOB_ID}.out" \
  "artifacts/protected/tricompose_v1_2/slurm_logs/tri_opacity_reader_${SLURM_JOB_ID}.err"
TASK_RUNTIME="$WORKSPACE/artifacts/protected/tricompose_v1_2/task_runtime/manual_opacity_reader_${SLURM_JOB_ID}"
test ! -e "$TASK_RUNTIME"
mkdir -m 2770 "$TASK_RUNTIME" "$TASK_RUNTIME/tmp" "$TASK_RUNTIME/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$TASK_RUNTIME/tmp" XDG_CACHE_HOME="$TASK_RUNTIME/cache"
export HF_HOME="$TASK_RUNTIME/cache/huggingface" TORCH_HOME="$TASK_RUNTIME/cache/torch"
export MPLCONFIGDIR="$TASK_RUNTIME/cache/matplotlib" PYTHONPYCACHEPREFIX="$TASK_RUNTIME/cache/bytecode"
export TRICOMPOSE_MANUAL_OPACITY_READER=manual-opacity-chexbert-request-v1
timeout --signal=TERM --kill-after=10s 780s \
  cxrmate/venv/bin/python \
  TriCompose-v1.2/tools/benchmark_manual_opacity_dev_chexbert.py

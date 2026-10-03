#!/bin/bash
# PREPARED ONLY: requires explicit approval after full script/resource review.
# 2 fixed synthetic EHRs x (RoentGen-v2 + Sana) x CXRMate-single.
# Sequential generation/XRV/report/CheXbert; no training or adaptive repair.
#SBATCH --job-name=tri_v12_live_s2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/live_worker_plans/live_workers2_12605930_001"
PLAN_SHA=1e18eb7664b406551100ffad296dbbf4c75f8876fe349b062f1d1db8089cc55e
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/live_worker_runs"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/live_workers_${SLURM_JOB_ID}"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/run_live_smoke.py"
cd "$WORKSPACE"
if test -e "$TASK" || test -e "$OUTPUT/live_workers2_${SLURM_JOB_ID}"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" TMPDIR="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if "$PYTHON" "$PROGRAM" \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "live_workers2_${SLURM_JOB_ID}" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=completed_fixed_paths_unvalidated\n'
else
  printf 'status=failed reason=see_protected_execution_journal\n'
  exit 1
fi

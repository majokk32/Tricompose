#!/bin/bash
# PREPARED ONLY: explicit approval after complete script review required.
# Fresh XRV/CheXbert on the fixed full pool; no generation or training.
# Reuse historical BioViL only AFTER sealing choice; CPU audit before exit.
#SBATCH --job-name=tri_v12_pool80
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
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/full_pool_report_plans/pool80_fresh_12621834_001"
PLAN_SHA=ffa4691cb2df67ba6d7c8e8a7bbd235fff1b592d82620837f6b26e1011ae65ca
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/full_pool_report_runs"
AUDITS="$WORKSPACE/artifacts/protected/tricompose_v1_2/full_pool_report_audits"
RUN_ID="pool80_fresh_${SLURM_JOB_ID}"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/${RUN_ID}"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/run_full_pool_report_control.py"
AUDITOR="$WORKSPACE/TriCompose-v1.2/audits/audit_full_pool_report_control.py"
cd "$WORKSPACE"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_pool80_${SLURM_JOB_ID}.out" \
  "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_pool80_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID" || test -e "$AUDITS/$RUN_ID"; then
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
if ! "$PYTHON" "$PROGRAM" \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=failed reason=see_protected_cost_journal\n'
  exit 1
fi
if ! "$PYTHON" "$AUDITOR" \
  --source-run "$OUTPUT/$RUN_ID" --plan-run "$PLAN" \
  --output-root "$AUDITS" --run-id "$RUN_ID" \
  >"$TASK/audit.stdout.log" 2>"$TASK/audit.stderr.log"; then
  printf 'status=completed_scores_but_metadata_audit_failed\n'
  exit 1
fi
printf 'status=completed_full_pool_fresh_scores_metadata_audited_unvalidated\n'

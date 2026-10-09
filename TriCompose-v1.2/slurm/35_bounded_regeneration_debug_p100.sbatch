#!/bin/bash
# PREPARED ONLY: approval after complete script/resource review required.
# Fixed EHR/text; at most two new RoentGen images and two CXRMate reports.
# Four charged generation/verification attempts per case, no retries.
# Independent secondary evaluation only after sealing; then metadata audit.
#SBATCH --job-name=tri_v12_retry2
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
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/bounded_regeneration_plans/retry2_12625457_001"
PLAN_SHA=5a4e3bfbfd7e87879e14ea545932e3fef6331f3440bb4ef2973e52355fec8542
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/bounded_regeneration_runs"
AUDITS="$WORKSPACE/artifacts/protected/tricompose_v1_2/bounded_regeneration_audits"
RUN_ID="retry2_${SLURM_JOB_ID}"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/${RUN_ID}"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/run_bounded_regeneration.py"
AUDITOR="$WORKSPACE/TriCompose-v1.2/audits/audit_bounded_regeneration.py"
AUDIT_SHA=98755d9aaad7227f27da10f106a233a67e5712fad43a81d909c78286b1c86a5a
cd "$WORKSPACE"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_retry2_${SLURM_JOB_ID}.out" \
  "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_retry2_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID" || test -e "$AUDITS/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
AUDIT_ACTUAL="$(sha256sum "$AUDITOR")"
if test "${AUDIT_ACTUAL%% *}" != "$AUDIT_SHA"; then
  printf 'status=refused reason=auditor_source_changed\n'
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
  printf 'status=failed reason=see_protected_budget_journals\n'
  exit 1
fi
if ! "$PYTHON" "$AUDITOR" \
  --source-run "$OUTPUT/$RUN_ID" --plan-run "$PLAN" \
  --plan-manifest-sha256 "$PLAN_SHA" --output-root "$AUDITS" --run-id "$RUN_ID" \
  >"$TASK/audit.stdout.log" 2>"$TASK/audit.stderr.log"; then
  printf 'status=retry_completed_but_metadata_audit_failed\n'
  exit 1
fi
printf 'status=completed_bounded_retry_metadata_audited_unvalidated\n'

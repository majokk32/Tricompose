#!/bin/bash
# PREPARED ONLY. Submit only after the user sees this entire script and approves.
# Frozen, fully synthetic 2-case/2-seed/2-report-expert paired collection.
# <=24 generation/verification attempts. No training or online-repair claim.
# BioViL is NOT run here; only a sealed request for a later endpoint job.
#SBATCH --job-name=tri_paired2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/paired_probe_plans/paired2_12784259_001"
PLAN_SHA=ea4a1faf17b9d22a063bccd6d9482c4f598d017acb63920a59c2a0934ab0d37a
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/paired_probe_runs"
RUN_ID="paired2_${SLURM_JOB_ID}"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/$RUN_ID"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/collect_paired_probes_v1.py"
PROGRAM_SHA=f29979a407f68ae367972d42baa1d416bddbec7ac5169b1f21388fd05097720e
cd "$WORKSPACE"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_paired2_${SLURM_JOB_ID}.out" \
  "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_paired2_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
ACTUAL_SHA="$(sha256sum "$PROGRAM")"
if test "${ACTUAL_SHA%% *}" != "$PROGRAM_SHA"; then
  printf 'status=refused reason=worker_source_changed\n'
  exit 1
fi
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor" MPLCONFIGDIR="$TASK/cache/matplotlib"
export TMPDIR="$TASK/tmp" TEMP="$TASK/tmp" TMP="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if ! "$PYTHON" "$PROGRAM" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=failed reason=see_protected_receipts_and_budget_journals\n'
  exit 1
fi
printf 'status=completed_paired_collection_unvalidated\n'

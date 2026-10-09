#!/bin/bash
# PREPARED ONLY: approval after complete script/resource review required.
# BioViL recovery ONLY: saved images/reports, no generation or primary scoring.
# Restore existing vendor import bootstrap; preserve old choice and failed run.
#SBATCH --job-name=tri_v12_retry_bv
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/bounded_regeneration_endpoint_plans/retry2_endpoint_12625457_001"
PLAN_SHA=17a0e2de31bcf003723411ec32613a4cc1285d46be428cfd80c54d6d26610f6d
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/bounded_regeneration_endpoints"
RUN_ID="retry2_endpoint_${SLURM_JOB_ID}"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/${RUN_ID}"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/audits/complete_bounded_regeneration_endpoint.py"
cd "$WORKSPACE"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_retry_bv_${SLURM_JOB_ID}.out" \
  "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_retry_bv_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
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
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
if ! "$PYTHON" "$PROGRAM" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=secondary_recovery_failed_charged_work_retained\n'
  exit 1
fi
printf 'status=completed_secondary_recovery_source_choices_unchanged\n'

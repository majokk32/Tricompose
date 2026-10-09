#!/bin/bash
# PREPARED ONLY. Show the entire script/resources and obtain approval first.
# Existing synthetic artifacts only; no regeneration, primary scoring or training.
# BioViL is secondary-only: fixed/static/LLM choices are already sealed.
#SBATCH --job-name=tri_llm_fresh_bv
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/agent/run_fresh_report_secondary.py"
PROGRAM_SHA=1ebbdcbd4ead0aefaae854e6ce300216113bea102ebf5f63c78101227c31d5de
PLAN="$ROOT/llm_fresh_report_endpoint_plans/fresh_report2_biovil_12817261_001"
PLAN_SHA=c5f9435f49a15f5f43a0c58fe0a7fdf71869b265d24aa04f76bef218d6497745
OUTPUT="$ROOT/llm_fresh_report_endpoints"
case "${SLURM_JOB_ID:-}" in
  ""|*[!0-9]*) printf 'status=refused reason=slurm_job_id_required\n'; exit 1 ;;
esac
RUN_ID="fresh_report2_biovil_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
chmod 0660 "$ROOT/slurm_logs/tri_llm_fresh_bv_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_llm_fresh_bv_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
printf '%s  %s\n' "$PROGRAM_SHA" "$PROGRAM" | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
finish() {
  local task_exit=$?
  trap - EXIT
  find "$TASK" -type d -exec chmod 2770 {} +
  find "$TASK" -type f -exec chmod 0660 {} +
  exit "$task_exit"
}
trap finish EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda" MPLCONFIGDIR="$TASK/cache/matplotlib"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" TRITON_CACHE_DIR="$TASK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor" TORCH_EXTENSIONS_DIR="$TASK/cache/extensions"
export TMPDIR="$TASK/tmp" TMP="$TASK/tmp" TEMP="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
RESULT=0
timeout --signal=TERM --kill-after=30s 480s "$PYTHON" "$PROGRAM" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_and_charged_work_retained\n'
  exit "$RESULT"
fi
printf 'status=completed_secondary_scores_original_selections_unchanged\n'

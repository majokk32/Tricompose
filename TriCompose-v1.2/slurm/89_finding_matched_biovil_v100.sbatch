#!/bin/bash
# PREPARED ONLY. Show this entire script and obtain approval before sbatch.
# Frozen scoring of existing synthetic images; no generation or patient input.
#SBATCH --job-name=tri_v12_find6
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
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/score_finding_matched_biovil_v1.py"
PROGRAM_SHA=762951f59f175e8915ce0c83e45302c6a0cceb9baa61b2302274354a154f5046
PLAN="$ROOT/finding_matched_biovil_plans/finding6_pool240_12799642_001"
PLAN_SHA=63cd3e4ea1400782ee12f51c2b21a65951182c5b0752d3b94cb81c904d220923
OUTPUT="$ROOT/finding_matched_biovil_runs"
RUN_ID="finding6_pool240_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
chmod 0660 "$ROOT/slurm_logs/tri_v12_find6_${SLURM_JOB_ID}.out" "$ROOT/slurm_logs/tri_v12_find6_${SLURM_JOB_ID}.err"
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
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
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
timeout --signal=TERM --kill-after=30s 480s "$PYTHON" "$PROGRAM" evaluate \
  --plan-root "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --allow-synthetic-finding-images --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$RESULT"
fi
printf 'status=completed_finding_diagnostic_no_selection_change\n'

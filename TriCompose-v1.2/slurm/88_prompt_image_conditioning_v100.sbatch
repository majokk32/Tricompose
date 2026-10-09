#!/bin/bash
# PREPARED ONLY: display this complete script and obtain explicit approval.
# Existing synthetic prompts/CXRs only; frozen BioViL diagnostic, no selection.
#SBATCH --job-name=tri_v12_prompt_img
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
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/score_prompt_image_conditioning_v1.py"
PROGRAM_SHA=b89570362fd8d3f6a81a3d43ffebafed02315bd1f44e871d91ddcdddd56c0b27
PLAN="$ROOT/prompt_image_conditioning_plans/conditioning_pool240_12799642_001"
PLAN_SHA=8566f1a78abc5aac0a63fb19c03ac01979bebac462ffb9f0a87dbcd342f1646a
OUTPUT="$ROOT/prompt_image_conditioning_runs"
RUN_ID="conditioning_pool240_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
cd "$WORKSPACE"
chmod 0660 "$ROOT/slurm_logs/tri_v12_prompt_img_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_prompt_img_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
printf '%s  %s\n' "$PROGRAM_SHA" "$PROGRAM" | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
mkdir -m 2770 "$TASK/cache/xdg" "$TASK/cache/huggingface" "$TASK/cache/torch" \
  "$TASK/cache/cuda" "$TASK/cache/matplotlib" "$TASK/cache/kernels"
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
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export TORCH_EXTENSIONS_DIR="$TASK/cache/extensions"
export TMPDIR="$TASK/tmp" TMP="$TASK/tmp" TEMP="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
RESULT=0
timeout --signal=TERM --kill-after=30s 480s "$PYTHON" "$PROGRAM" evaluate \
  --plan-root "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --allow-synthetic-prompt-image --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$RESULT"
fi
printf 'status=completed_conditioning_diagnostic_no_selection_change\n'

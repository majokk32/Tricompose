#!/bin/bash
# PROPOSED ONLY: complete script/resource review and fresh approval required.
# Two fixed synthetic EHRs x two models x two seeds x two clause orders.
# Preserve all sealed legacy words, including unvalidated CHF image priors.
# Sixteen images and sixteen XRV attempts; no report/planner/API/training.
# No old data/scorer/winner change, overwrite, automatic resume or retry.
# Scheduling-only revision: accept validated V100 OR A40 on gpu/debug.
# Fifteen minutes is an allocation ceiling, not a runtime/queue guarantee.
#SBATCH --job-name=tri_prompt_order16
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100|a40
#SBATCH --no-requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
TRI_BASE=/project2/ruishanl_1185/inference_3mod
TRI_ROOT="$TRI_BASE/artifacts/protected/tricompose_v1_2"
TRI_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
TRI_PROGRAM="$TRI_BASE/TriCompose-v1.2/agent/run_fixed_ehr_prompt_order.py"
TRI_PROGRAM_SHA=52263a8840de0169cdd96d8c222a8338f8812db26af099474473fa6784003cb6
TRI_PLAN="$TRI_ROOT/fixed_ehr_prompt_order_plans/prompt_order16_12879534_003"
TRI_PLAN_SHA=f45718f3a2917ec24fd06c4b8551fd84b958528e9790511d4262c924d3bc06c7
TRI_OUTPUT="$TRI_ROOT/fixed_ehr_prompt_order_runs"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
TRI_RUN_ID="prompt_order_${SLURM_JOB_ID}"
TRI_TASK="$TRI_ROOT/job_runtime/$TRI_RUN_ID"
test -x "$TRI_PYTHON"
for directory in "$TRI_ROOT/slurm_logs" "$TRI_ROOT/job_runtime" "$TRI_PLAN" "$TRI_OUTPUT"; do
  test "$(stat -c %a "$directory")" = 2770
  task_gid=$(stat -c %g "$directory")
  test "$task_gid" = 96293 || test "$task_gid" = 65534
done
test ! -e "$TRI_TASK" && test ! -L "$TRI_TASK"
test ! -e "$TRI_OUTPUT/$TRI_RUN_ID" && test ! -L "$TRI_OUTPUT/$TRI_RUN_ID"
printf '%s  %s\n' "$TRI_PROGRAM_SHA" "$TRI_PROGRAM" | sha256sum --check --status
printf '%s  %s\n' "$TRI_PLAN_SHA" "$TRI_PLAN/manifest.json" | sha256sum --check --status
mkdir -m 2770 "$TRI_TASK" "$TRI_TASK/cache" "$TRI_TASK/tmp"
chmod 0660 "$TRI_ROOT/slurm_logs/tri_prompt_order16_${SLURM_JOB_ID}.out" \
  "$TRI_ROOT/slurm_logs/tri_prompt_order16_${SLURM_JOB_ID}.err"
finish() {
  local task_exit=$?
  trap - EXIT
  for task_run in "$TRI_TASK" "$TRI_OUTPUT/$TRI_RUN_ID"; do
    if test -d "$task_run"; then
      find "$task_run" -type d -exec chmod 2770 {} +
      find "$task_run" -type f -exec chmod 0660 {} +
    fi
  done
  exit "$task_exit"
}
trap finish EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TRI_TASK/cache/xdg" HF_HOME="$TRI_TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$TRI_TASK/cache/torch" CUDA_CACHE_PATH="$TRI_TASK/cache/cuda" MPLCONFIGDIR="$TRI_TASK/cache/matplotlib"
export TRITON_CACHE_DIR="$TRI_TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TRI_TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TRI_TASK/cache/kernels" TORCH_EXTENSIONS_DIR="$TRI_TASK/cache/extensions"
export HF_DATASETS_CACHE="$TRI_TASK/cache/datasets" TMPDIR="$TRI_TASK/tmp" TMP="$TRI_TASK/tmp" TEMP="$TRI_TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY TRICOMPOSE_LLM_API_KEY
TRI_RESULT=0
timeout --signal=INT --kill-after=15s 870s "$TRI_PYTHON" "$TRI_PROGRAM" run \
  --plan-run "$TRI_PLAN" --plan-manifest-sha256 "$TRI_PLAN_SHA" \
  --output-root "$TRI_OUTPUT" --run-id "$TRI_RUN_ID" \
  >"$TRI_TASK/worker.stdout.log" 2>"$TRI_TASK/worker.stderr.log" || TRI_RESULT=$?
if test "$TRI_RESULT" -ne 0; then
  printf 'status=failed_closed private_prefix_and_charged_journals_retained\n'
  exit "$TRI_RESULT"
fi
printf 'status=prompt_order_diagnostic_finished_review_counts_no_clinical_repair_claim\n'

#!/bin/bash
# PROPOSED ONLY: show full script/resources and obtain explicit approval.
# Same RSUA50; official frozen CheXagent SRRG -> frozen CheXbert.
# Sequential native environments, no Qwen/planner/CXR generation or training.
# At most 50 report requests + 50 label requests; zero retry/selection change.
# Ten minutes is a ceiling, not a runtime or queue promise.
#SBATCH --job-name=tri_rsua_medical50
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
TRI_BASE=/project2/ruishanl_1185/inference_3mod
TRI_ROOT="$TRI_BASE/artifacts/protected/tricompose_v1_2"
TRI_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
TRI_PROGRAM="$TRI_BASE/TriCompose-v1.2/real_validation/rsua_medical_report_observer.py"
TRI_PROGRAM_SHA=eaf078046434c20188174240b614c97a39d76a98555ab57e4c2c60c511e28a19
TRI_PLAN="$TRI_ROOT/real_validation/rsua_medical_report_plans/rsua_medical50_12851223_001"
TRI_PLAN_SHA=8edb55e03155b546acdce1d627407924e1bf54d50e1bcc716452d0e387f2331f
TRI_OUTPUT="$TRI_ROOT/real_validation/rsua_medical_report_runs"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
TRI_RUN_ID="rsua_medical50_${SLURM_JOB_ID}"
TRI_TASK="$TRI_ROOT/job_runtime/$TRI_RUN_ID"
test -x "$TRI_PYTHON"
for directory in "$TRI_ROOT/slurm_logs" "$TRI_ROOT/job_runtime" "$TRI_PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  task_gid=$(stat -c %g "$directory")
  test "$task_gid" = 96293 || test "$task_gid" = 65534
done
test ! -e "$TRI_TASK" && test ! -L "$TRI_TASK"
test ! -e "$TRI_OUTPUT/$TRI_RUN_ID" && test ! -L "$TRI_OUTPUT/$TRI_RUN_ID"
printf '%s  %s\n' "$TRI_PROGRAM_SHA" "$TRI_PROGRAM" | sha256sum --check --status
printf '%s  %s\n' "$TRI_PLAN_SHA" "$TRI_PLAN/manifest.json" | sha256sum --check --status
mkdir -m 2770 "$TRI_TASK" "$TRI_TASK/cache" "$TRI_TASK/tmp"
chmod 0660 "$TRI_ROOT/slurm_logs/tri_rsua_medical50_${SLURM_JOB_ID}.out" \
  "$TRI_ROOT/slurm_logs/tri_rsua_medical50_${SLURM_JOB_ID}.err"
finish() {
  local task_exit=$?
  trap - EXIT
  find "$TRI_TASK" -type d -exec chmod 2770 {} +
  find "$TRI_TASK" -type f -exec chmod 0660 {} +
  if test -d "$TRI_OUTPUT"; then
    while IFS= read -r -d '' task_run; do
      find "$task_run" -type d -exec chmod 2770 {} +
      find "$task_run" -type f -exec chmod 0660 {} +
    done < <(find "$TRI_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
      \( -name "$TRI_RUN_ID" -o -name ".$TRI_RUN_ID.*.tmp" \) -print0)
  fi
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
# SIGINT raises KeyboardInterrupt so the existing owned-child cleanup runs.
timeout --signal=INT --kill-after=15s 540s "$TRI_PYTHON" "$TRI_PROGRAM" run \
  --plan-run "$TRI_PLAN" --plan-manifest-sha256 "$TRI_PLAN_SHA" \
  --output-root "$TRI_OUTPUT" --run-id "$TRI_RUN_ID" --allow-public-reference-eval \
  >"$TRI_TASK/worker.stdout.log" 2>"$TRI_TASK/worker.stderr.log" || TRI_RESULT=$?
if test "$TRI_RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_and_charged_attempts_retained\n'
  exit "$TRI_RESULT"
fi
printf 'status=completed_medical_pipeline_proxy_diagnostic_selections_unchanged_review_results_required\n'

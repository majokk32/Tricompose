#!/bin/bash
# PROPOSED ONLY: display this complete script and obtain fresh explicit approval.
# Two NEW frozen numeric LLM decisions; cached image observations, no generation.
# Original selected triples remain unchanged; requests need separate approval.
# Five minutes is an allocation cap, not an inference-time promise.
#SBATCH --job-name=tri_fresh_pol2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
TRI_BASE=/project2/ruishanl_1185/inference_3mod
TRI_ROOT="$TRI_BASE/artifacts/protected/tricompose_v1_2"
TRI_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
TRI_PROGRAM="$TRI_BASE/TriCompose-v1.2/agent/run_guarded_fresh_policy.py"
TRI_PROGRAM_SHA=0b52267f4d29138aa94f069b36d1aae1be89b877c29f3ffc85edc3163b0114b1
TRI_PLAN="$TRI_ROOT/guarded_fresh_policy_plans/fresh_policy2_12827441_001"
TRI_PLAN_SHA=f575d3be72e16242b50e711d187dc9be9db59f929d380bb65e2dd771731a46ae
TRI_OUTPUT="$TRI_ROOT/guarded_fresh_policy_runs"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
TRI_RUN_ID="fresh_policy2_${SLURM_JOB_ID}"
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
mkdir -m 2770 "$TRI_TASK" "$TRI_TASK/cache" "$TRI_TASK/tmp"
chmod 0660 "$TRI_ROOT/slurm_logs/tri_fresh_pol2_${SLURM_JOB_ID}.out" \
  "$TRI_ROOT/slurm_logs/tri_fresh_pol2_${SLURM_JOB_ID}.err"
finish() {
  local task_exit=$?
  trap - EXIT
  find "$TRI_TASK" -type d -exec chmod 2770 {} +
  find "$TRI_TASK" -type f -exec chmod 0660 {} +
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
timeout --signal=TERM --kill-after=15s 240s "$TRI_PYTHON" "$TRI_PROGRAM" run \
  --plan-run "$TRI_PLAN" --plan-manifest-sha256 "$TRI_PLAN_SHA" \
  --output-root "$TRI_OUTPUT" --run-id "$TRI_RUN_ID" \
  >"$TRI_TASK/worker.stdout.log" 2>"$TRI_TASK/worker.stderr.log" || TRI_RESULT=$?
if test "$TRI_RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_and_charged_attempts_retained\n'
  exit "$TRI_RESULT"
fi
printf 'status=fresh_numeric_planning_completed_no_generation\n'

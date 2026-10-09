#!/bin/bash
# PROPOSED ONLY. Complete CPU preflight, show script/resources, obtain approval.
# Two fixed synthetic EHRs; real frozen generation; report-only Qwen proposals.
# At most 2 CXR / 8 report artifacts, 20 worker + 6 policy reservations.
# No training, API, real target, scorer change or historical overwrite.
#SBATCH --job-name=tri_llm_fresh2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
TRI_BASE=/project2/ruishanl_1185/inference_3mod
TRI_AGENT="$TRI_BASE/TriCompose-v1.2/agent"
TRI_ROOT="$TRI_BASE/artifacts/protected/tricompose_v1_2"
TRI_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
TRI_PLAN="$TRI_ROOT/llm_fresh_report_plans/fresh_report2_fullassets_12799642_001"
TRI_PLAN_SHA=01516d6965c3b0e287acff715da1735086580ddc8aa8d9ef39525ea8a9533e1f
[[ "$TRI_PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
TRI_RUN_ID="llm_fresh2_${SLURM_JOB_ID}"
TRI_OUTPUT="$TRI_ROOT/llm_fresh_report_runs"
TRI_TASK="$TRI_OUTPUT/task_${SLURM_JOB_ID}"
test -x "$TRI_PYTHON"
for directory in "$TRI_ROOT/slurm_logs" "$TRI_OUTPUT" "$TRI_PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  183846783cde57215e82dee2786a404fe65b7803200ae7e3f3d5fa940bad33bf "$TRI_AGENT/run_fresh_report_agent_v2.py" \
  335362cc8092b0c347025deba0b43009713a8b685059a4dbd7cac981e56740e3 "$TRI_AGENT/run_fresh_report_agent.py" \
  b720c793583480b7510c5533aa607f6bb20f9c5621345823634e5cd2d257f5fd "$TRI_AGENT/run_local_qwen_decision.py" \
  f154cbbe89c946e8c8d6a352946a8da75116ffe18cb150abf725ca84694ee4f8 "$TRI_AGENT/tricompose_llm/live_report_bridge.py" \
  bc07c02ab280e21d8f9d9101f2e7f494a970ba0e02b6d99df7f00534dd8f68e2 "$TRI_AGENT/tricompose_llm/fresh_report_worker.py" \
  | sha256sum --check --status
test ! -e "$TRI_TASK"
test ! -L "$TRI_TASK"
test ! -e "$TRI_OUTPUT/$TRI_RUN_ID"
test ! -L "$TRI_OUTPUT/$TRI_RUN_ID"
mkdir -m 2770 "$TRI_TASK" "$TRI_TASK/cache" "$TRI_TASK/tmp"
chmod 0660 "$TRI_ROOT/slurm_logs/tri_llm_fresh2_${SLURM_JOB_ID}.out" \
  "$TRI_ROOT/slurm_logs/tri_llm_fresh2_${SLURM_JOB_ID}.err"
finish_permissions() {
  for directory in "$TRI_TASK" "$TRI_OUTPUT/$TRI_RUN_ID"; do
    if [ -d "$directory" ]; then
      find "$directory" -type d -exec chmod 2770 {} +
      find "$directory" -type f -exec chmod 0660 {} +
    fi
  done
}
trap finish_permissions EXIT
unset OPENAI_API_KEY TRICOMPOSE_LLM_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TRI_TASK/cache/xdg" HF_HOME="$TRI_TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TRANSFORMERS_CACHE="$HF_HOME/transformers" TORCH_HOME="$TRI_TASK/cache/torch"
export CUDA_CACHE_PATH="$TRI_TASK/cache/cuda" TRITON_CACHE_DIR="$TRI_TASK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$TRI_TASK/cache/inductor" MPLCONFIGDIR="$TRI_TASK/cache/matplotlib"
export PYTORCH_KERNEL_CACHE_PATH="$TRI_TASK/cache/kernels" HF_DATASETS_CACHE="$TRI_TASK/cache/datasets"
export TMPDIR="$TRI_TASK/tmp" TMP="$TRI_TASK/tmp" TEMP="$TRI_TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$TRI_PYTHON" "$TRI_AGENT/run_fresh_report_agent_v2.py" run \
  --plan-run "$TRI_PLAN" --plan-manifest-sha256 "$TRI_PLAN_SHA" \
  --output-root "$TRI_OUTPUT" --run-id "$TRI_RUN_ID" \
  >"$TRI_TASK/driver.log" 2>&1
printf 'status=fresh_report_agent_contract_run_completed_unvalidated\n'

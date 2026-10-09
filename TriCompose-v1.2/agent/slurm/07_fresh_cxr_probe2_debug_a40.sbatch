#!/bin/bash
# PROPOSED ONLY: display complete script/resources and obtain approval.
# Two fixed synthetic anchors; at most one prospective image probe per case.
# Maximum 2 CXR + 2 reports, 8 worker + 2 numeric-only Qwen calls.
# Retain existing EHR/text/expert/scorers/gates; no training or external API.
# Thirty minutes is an allocation cap, not an expected inference duration.
#SBATCH --job-name=tri_llm_cxr2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
TRI_BASE=/project2/ruishanl_1185/inference_3mod
TRI_AGENT="$TRI_BASE/TriCompose-v1.2/agent"
TRI_ROOT="$TRI_BASE/artifacts/protected/tricompose_v1_2"
TRI_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
TRI_PLAN="$TRI_ROOT/llm_fresh_cxr_plans/fresh_cxr2_12817261_001"
TRI_PLAN_SHA=807f136342fd0c837962b3b860462df8e08df6a197a0899b6de3b7e5682f1163
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
TRI_RUN_ID="llm_cxr2_${SLURM_JOB_ID}"
TRI_OUTPUT="$TRI_ROOT/llm_fresh_cxr_runs"
TRI_TASK="$TRI_OUTPUT/task_${SLURM_JOB_ID}"
test -x "$TRI_PYTHON"
for directory in "$TRI_ROOT/slurm_logs" "$TRI_OUTPUT" "$TRI_PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  240258fd4b776dcb552e1557d6caed79970b397eb5f6944d175af141faef1e38 "$TRI_AGENT/run_fresh_cxr_agent.py" \
  71aa353497504f90e4184f1a72cf8f7f15496f46ddef64f12d4011bbb5df7b8d "$TRI_AGENT/tricompose_llm/live_cxr_bridge.py" \
  b720c793583480b7510c5533aa607f6bb20f9c5621345823634e5cd2d257f5fd "$TRI_AGENT/run_local_qwen_decision.py" \
  | sha256sum --check --status
test ! -e "$TRI_TASK"
test ! -L "$TRI_TASK"
test ! -e "$TRI_OUTPUT/$TRI_RUN_ID"
test ! -L "$TRI_OUTPUT/$TRI_RUN_ID"
mkdir -m 2770 "$TRI_TASK" "$TRI_TASK/cache" "$TRI_TASK/tmp"
chmod 0660 "$TRI_ROOT/slurm_logs/tri_llm_cxr2_${SLURM_JOB_ID}.out" \
  "$TRI_ROOT/slurm_logs/tri_llm_cxr2_${SLURM_JOB_ID}.err"
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
"$TRI_PYTHON" "$TRI_AGENT/run_fresh_cxr_agent.py" run \
  --plan-run "$TRI_PLAN" --plan-manifest-sha256 "$TRI_PLAN_SHA" \
  --output-root "$TRI_OUTPUT" --run-id "$TRI_RUN_ID" \
  >"$TRI_TASK/driver.log" 2>&1
printf 'status=fresh_cxr_probe_contract_run_completed_unvalidated\n'

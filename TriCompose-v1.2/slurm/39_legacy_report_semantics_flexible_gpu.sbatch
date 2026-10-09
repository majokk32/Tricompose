#!/bin/bash
# Display this complete script/resources and obtain approval before sbatch.
# Secondary assertion extraction only; no generation, reranking or training.
#SBATCH --job-name=tri_v12_scope_qwen
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --constraint="v100|a40|a100|l40s"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PLAN="$ROOT/legacy_report_semantic_plans/semantic_scope2_12632006_001"
WORKER="$WORKSPACE/TriCompose-v1.2/benchmarks/verify_legacy_report_semantics.py"
RUNS="$ROOT/verification_runs"
TASK="$ROOT/task_runtime/qwen_scope2_${SLURM_JOB_ID}"
RUN_ID="qwen_scope2_${SLURM_JOB_ID}"

test -x "$PYTHON"
for directory in "$ROOT" "$ROOT/slurm_logs" "$RUNS" "$ROOT/task_runtime" "$PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  2c76baa73aea4b0af0bd4745ff164387727107a627a519524122c1ce84a99d2c "$PLAN/manifest.json" \
  | sha256sum --check --status
test ! -e "$TASK"
test ! -e "$RUNS/$RUN_ID"
test ! -e "$RUNS/${RUN_ID}_analysis"
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
chmod 0660 "$ROOT/slurm_logs/tri_v12_scope_qwen_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_scope_qwen_${SLURM_JOB_ID}.err"
finish_permissions() {
  find "$TASK" -type d -exec chmod 2770 {} +
  find "$TASK" -type f -exec chmod 0660 {} +
}
trap finish_permissions EXIT

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$WORKSPACE/TriCompose-v1.2/benchmarks"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" MPLCONFIGDIR="$TASK/cache/matplotlib"
export TMPDIR="$TASK/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

"$PYTHON" "$WORKER" run --plan-run "$PLAN" \
  --output-root "$RUNS" --run-id "$RUN_ID" >"$TASK/logs/verify.log" 2>&1
printf 'status=frozen_report_assertion_review_completed\n'
"$PYTHON" "$WORKER" analyze --plan-run "$PLAN" --review-run "$RUNS/$RUN_ID" \
  --output-root "$RUNS" --run-id "${RUN_ID}_analysis" >"$TASK/logs/analysis.log" 2>&1
printf 'status=secondary_assertion_comparison_completed_no_selection_change\n'

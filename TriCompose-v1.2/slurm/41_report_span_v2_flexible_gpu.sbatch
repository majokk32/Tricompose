#!/bin/bash
# Show the complete script/resources and obtain explicit approval before sbatch.
# Separate JSON-hierarchy clarification pilot; no generation/training/ranking.
#SBATCH --job-name=tri_v12_span_v2
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
PLAN="$ROOT/report_span_v2_plans/span_v2_scope2_12645021_001"
BASELINE="$ROOT/verification_runs/qwen_scope2_12639326"
SPAN_V1="$ROOT/verification_runs/span_scope2_12639717"
WORKER="$WORKSPACE/TriCompose-v1.2/tools/verify_report_span_selection_v2.py"
RUNS="$ROOT/verification_runs"
TASK="$ROOT/task_runtime/span_v2_scope2_${SLURM_JOB_ID}"
RUN_ID="span_v2_scope2_${SLURM_JOB_ID}"

test -x "$PYTHON"
for directory in "$ROOT" "$ROOT/slurm_logs" "$RUNS" "$ROOT/task_runtime" "$PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc "$PLAN/manifest.json" \
  d543ac0287b40bc1596944f348a1780b76c742b60a5cdc7decaddc8d2352dca4 "$BASELINE/manifest.json" \
  6a0a56a76c04b2f60987cba98b8ef1b82bb999767dd3e98cb9f2ad345319b4d9 "$SPAN_V1/manifest.json" \
  | sha256sum --check --status
test ! -e "$TASK"
test ! -e "$RUNS/$RUN_ID"
test ! -e "$RUNS/${RUN_ID}_analysis"
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
chmod 0660 "$ROOT/slurm_logs/tri_v12_span_v2_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_span_v2_${SLURM_JOB_ID}.err"
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
printf 'status=frozen_span_v2_review_completed\n'
"$PYTHON" "$WORKER" analyze --plan-run "$PLAN" --review-run "$RUNS/$RUN_ID" \
  --quote-review-run "$BASELINE" --output-root "$RUNS" --run-id "${RUN_ID}_analysis" \
  >"$TASK/logs/analysis.log" 2>&1
printf 'status=secondary_span_v2_comparison_completed_no_selection_change\n'

#!/bin/bash
# PROPOSED ONLY: display this complete script and obtain explicit approval.
# Frozen BioViL-T only: existing synthetic images/reports, no generation/training.
# Maximum four image encodings and ten report encodings; choices remain sealed.
# Five minutes is an allocation cap, not an expected inference duration.
#SBATCH --job-name=tri_cxr2_bv
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
TRI_BASE=/project2/ruishanl_1185/inference_3mod
TRI_ROOT="$TRI_BASE/artifacts/protected/tricompose_v1_2"
TRI_PYTHON="$TRI_BASE/CheXGenBench/metrics_venv/bin/python"
TRI_PROGRAM="$TRI_BASE/TriCompose-v1.2/agent/run_fresh_cxr_secondary.py"
TRI_PROGRAM_SHA=c5c580d583e535475ea24e44c0ee3b82cf9370907de1d068cee70881cbd91005
TRI_PLAN="$TRI_ROOT/llm_fresh_cxr_endpoint_plans/fresh_cxr2_biovil_12817261_001"
TRI_PLAN_SHA=914e06ae76e3788523c4910d52b696e1d0bb3a62a63c1b8a8a9f65dbfa11e917
TRI_OUTPUT="$TRI_ROOT/llm_fresh_cxr_endpoints"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
TRI_RUN_ID="fresh_cxr2_biovil_${SLURM_JOB_ID}"
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
chmod 0660 "$TRI_ROOT/slurm_logs/tri_cxr2_bv_${SLURM_JOB_ID}.out" \
  "$TRI_ROOT/slurm_logs/tri_cxr2_bv_${SLURM_JOB_ID}.err"
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
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY TRICOMPOSE_LLM_API_KEY
TRI_RESULT=0
timeout --signal=TERM --kill-after=15s 240s "$TRI_PYTHON" "$TRI_PROGRAM" run \
  --plan-run "$TRI_PLAN" --plan-manifest-sha256 "$TRI_PLAN_SHA" \
  --output-root "$TRI_OUTPUT" --run-id "$TRI_RUN_ID" \
  >"$TRI_TASK/worker.stdout.log" 2>"$TRI_TASK/worker.stderr.log" || TRI_RESULT=$?
if test "$TRI_RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$TRI_RESULT"
fi
printf 'status=completed_secondary_scores_original_selections_unchanged\n'

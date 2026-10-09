#!/bin/bash
# PREPARED ONLY: show complete script/resources and obtain explicit approval.
# Sixteen synthetic report texts, two replays, six authored controls; no images.
# No generation, training, reranking or old-winner change.
#SBATCH --job-name=tri_v12_optext16
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/check_opacity_report_assertions.py"
PROGRAM_SHA=3f5493bbd5b4f4d5c661f308ec87eaae7dd1e5592f77b2f9a1ad9c1f8979606f
PLAN="$ROOT/report_opacity_check_plans/opacity_text16_12666569_001"
PLAN_SHA=a8d0530a6c646b52427c30caf35be0f076027543d96084d3dc43913deff095cc
OUTPUT="$ROOT/report_opacity_check_runs"
RUN_ID="opacity_text16_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
cd "$WORKSPACE"
chmod 0660 "$ROOT/slurm_logs/tri_v12_optext16_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_optext16_${SLURM_JOB_ID}.err"
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
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/benchmarks:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
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
export TRANSFORMERS_VERBOSITY=error WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
RESULT=0
"$PYTHON" "$PROGRAM" evaluate \
  --plan-root "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --allow-synthetic-text-check --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$RESULT"
fi
printf 'status=completed_text_assertion_proposals_no_selection_or_repair\n'

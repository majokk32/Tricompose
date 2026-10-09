#!/bin/bash
# PREPARED ONLY: display the complete script/resources and obtain approval.
# Same 50 public RSUA images, three fixed pneumonia polarity probe pairs.
# No training, generation, threshold fitting or historical selection change.
#SBATCH --job-name=tri_v12_rsua_bv50
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/real_validation/rsua_biovil.py"
PROGRAM_SHA=91556a13e97c1c64253b3ca6c15927ebfbba1c159966e994dfe289ec0fe08c0c
PLAN="$ROOT/real_validation/rsua_biovil_plans/plan_12636566_001"
PLAN_SHA=043e8f8ad708fa84fe2529a48ad0a0c2139e211e9814199fffff673ccb922e5b
OUTPUT="$ROOT/real_validation/rsua_biovil_pilots"
RUN_ID="rsua_biovil50_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
cd "$WORKSPACE"
chmod 0660 "$ROOT/slurm_logs/tri_v12_rsua_bv50_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_rsua_bv50_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
printf '%s  %s\n' "$PROGRAM_SHA" "$PROGRAM" | sha256sum --check --status
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
mkdir -m 2770 "$TASK/cache/xdg" "$TASK/cache/huggingface" "$TASK/cache/torch" \
  "$TASK/cache/cuda" "$TASK/cache/matplotlib" "$TASK/cache/kernels"
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export TORCH_EXTENSIONS_DIR="$TASK/cache/extensions" TMPDIR="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
RESULT=0
"$PYTHON" "$PROGRAM" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" --allow-rsua-pixels \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
# New job-local cache/logs only; never follow symlinks or change old runs.
find "$TASK" -type d -exec chmod 2770 {} +
find "$TASK" -type f -exec chmod 0660 {} +
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$RESULT"
fi
printf 'status=completed_secondary_diagnostic_choices_and_thresholds_unchanged\n'

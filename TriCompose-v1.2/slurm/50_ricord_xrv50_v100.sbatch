#!/bin/bash
# PREPARED ONLY: show complete script/resources and obtain explicit approval.
# Fixed RICORD 50; DICOM presentation + frozen XRV, no training/refitting.
# Ten minutes is an allocation cap, not a runtime or queue-time guarantee.
#SBATCH --job-name=tri_v12_ricord50
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
PROGRAM="$WORKSPACE/TriCompose-v1.2/real_validation/ricord_xrv50.py"
PLAN="$ROOT/real_validation/ricord_xrv_plans/display50_12666569_001"
PLAN_SHA=3732ed1592bed8dea7aa479f278bd14ddfb3d8a4304356039ba84080651bd742
OUTPUT="$ROOT/real_validation/ricord_xrv_pilots"
RUN_ID="ricord_xrv50_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
cd "$WORKSPACE"
chmod 0660 "$ROOT/slurm_logs/tri_v12_ricord50_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_ricord50_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
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
export PYTHONPATH="$WORKSPACE/.tmp/ricord_header_dependencies_12666569_001:$WORKSPACE/.tmp/ricord_pixel_dependencies_12666569_001:$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels"
export TORCHINDUCTOR_CACHE_DIR="$TASK/cache/kernels"
export TMPDIR="$TASK/tmp" TMP="$TASK/tmp" TEMP="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
RESULT=0
"$PYTHON" "$PROGRAM" evaluate \
  --plan-root "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --allow-ricord-pixels-and-xrv --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$RESULT"
fi
printf 'status=completed_fixed50_frozen_opacity_diagnostic_unvalidated\n'

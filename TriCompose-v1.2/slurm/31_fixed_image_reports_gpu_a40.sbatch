#!/bin/bash
# PREPARED ONLY. Explicit approval after complete script/resource review required.
# Same frozen plan as script 30; gpu partition because debug A40 is drained.
# No training, new CXR generation, original artifact overwrite or external API.
#SBATCH --job-name=tri_v12_fixed_rep4
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PLAN="$WORKSPACE/artifacts/protected/tricompose_v1_2/fixed_image_report_plans/fixed_reports4_12605930_001"
PLAN_SHA=81dbde3393c67daff229d495c137e596683b18f244b06dc4895b297e7186a838
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/fixed_image_report_runs"
TASK="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/fixed_reports_${SLURM_JOB_ID}"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/run_fixed_image_reports.py"
cd "$WORKSPACE"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_fixed_rep4_${SLURM_JOB_ID}.out" \
  "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_fixed_rep4_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/fixed_reports4_${SLURM_JOB_ID}"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" TMPDIR="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if "$PYTHON" "$PROGRAM" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$OUTPUT" --run-id "fixed_reports4_${SLURM_JOB_ID}" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=completed_same_image_report_control_unvalidated\n'
else
  printf 'status=failed reason=see_protected_cost_journal\n'
  exit 1
fi

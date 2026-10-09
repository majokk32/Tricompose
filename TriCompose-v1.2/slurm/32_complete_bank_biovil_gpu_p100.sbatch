#!/bin/bash
# PREPARED ONLY: full script/resource review and explicit approval before sbatch.
# Score previously unscored synthetic pairs; no generation, selection or training.
#SBATCH --job-name=tri_v12_bv_complete
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
REQUEST="$WORKSPACE/artifacts/protected/tricompose_v1_2/score_coverage_diagnostics/full_bank_endpoint_request_12621834_001"
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/complete_bank_endpoints"
RUN_ID="complete_bank_biovil_${SLURM_JOB_ID}"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
MODEL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/complete_bank_endpoint.py"
TASK_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/complete_bank_biovil_${SLURM_JOB_ID}"

chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_bv_complete_${SLURM_JOB_ID}.out"
chmod 0660 "$WORKSPACE/artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_bv_complete_${SLURM_JOB_ID}.err"
for required in "$REQUEST/manifest.json" "$REQUEST/request.json" "$PYTHON" \
  "$MODEL/biovil_t_image_model_proj_size_128.pt" "$MODEL/pytorch_model.bin" "$VENDOR" "$PROGRAM"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT/$RUN_ID" || test -e "$TASK_ROOT"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi
mkdir "$TASK_ROOT"
chmod 2770 "$TASK_ROOT"
mkdir "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
chmod 2770 "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
export PYTHONPATH="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$VENDOR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg"
export HF_HOME="$TASK_ROOT/cache/huggingface" TORCH_HOME="$TASK_ROOT/cache/torch"
export TMPDIR="$TASK_ROOT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" "$PROGRAM" score \
  --request-run "$REQUEST" --model-path "$MODEL" \
  --output-root "$OUTPUT" --run-id "$RUN_ID"

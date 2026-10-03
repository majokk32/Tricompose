#!/bin/bash
# Prepared only. Requires explicit user approval before sbatch.
# Frozen BioViL-T blind scoring of synthetic intervention pairs.
#SBATCH --job-name=tri_v12_bvint80
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_1"
BANK="$WORKSPACE/artifacts/protected/tricompose_v1_2/benchmarks/pool80_label_targeted_stratified_20261001_001"
CXR_SANA="$BASE/cxr_candidates/pool80_chexgenbench_sana_s1_20260813_d0ce"
CXR_PIXART="$BASE/cxr_candidates/pool80_chexgenbench_pixart_s1_20260813_18b7"
REPORT="$BASE/report_candidates/pool160_chexagent2_fp32_20260813_51f8"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
MODEL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/score_intervention_biovil.py"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2/benchmarks"
OUTPUT_RUN_ID=pool80_targeted_biovil_20261001_001
TASK_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/biovil_int_${SLURM_JOB_ID}"

for required in "$BANK/manifest.json" "$BANK/blind_items.jsonl" "$BANK/resolver.jsonl" \
  "$CXR_SANA/manifest.json" "$CXR_PIXART/manifest.json" "$REPORT/manifest.json" \
  "$PYTHON" "$MODEL/biovil_t_image_model_proj_size_128.pt" \
  "$MODEL/pytorch_model.bin" "$VENDOR" "$PROGRAM"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

mkdir "$TASK_ROOT"
chmod 2770 "$TASK_ROOT"
mkdir "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
chmod 2770 "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/benchmarks:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$VENDOR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export TORCH_HOME="$TASK_ROOT/cache/torch"
export TMPDIR="$TASK_ROOT/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$PROGRAM" \
  --bank-run "$BANK" \
  --cxr-run "$CXR_SANA" \
  --cxr-run "$CXR_PIXART" \
  --report-run "$REPORT" \
  --model-path "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$OUTPUT_RUN_ID"

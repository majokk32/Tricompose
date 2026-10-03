#!/bin/bash
# Prepared only. Complete script/resource request must be approved before sbatch.
# Frozen BioViL-T endpoint readout; no generation, training or routing changes.
#SBATCH --job-name=tri_v12_auto_bv80
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
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
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_1"
REQUEST="$WORKSPACE/artifacts/protected/tricompose_v1_2/automatic_replays/automatic_biovil_request_12576792_001"
OUTPUT="$WORKSPACE/artifacts/protected/tricompose_v1_2/automatic_replays"
RUN_ID="automatic_biovil_pool80_${SLURM_JOB_ID}"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
MODEL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
PROGRAM="$WORKSPACE/TriCompose-v1.2/benchmarks/score_automatic_replay_biovil.py"
TASK_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2/job_runtime/automatic_biovil_${SLURM_JOB_ID}"

for required in "$REQUEST/manifest.json" "$REQUEST/request.json" "$PYTHON" \
  "$MODEL/biovil_t_image_model_proj_size_128.pt" "$MODEL/pytorch_model.bin" "$VENDOR" "$PROGRAM"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT/$RUN_ID"; then
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
  --request-run "$REQUEST" \
  --cxr-run "$BASE/cxr_candidates/pool80_chexgenbench_sana_s1_20260813_d0ce" \
  --cxr-run "$BASE/cxr_candidates/pool80_chexgenbench_pixart_s1_20260813_18b7" \
  --cxr-run "$BASE/cxr_candidates/pool80_roentgen_v2_fp16_s1_20260813_0bd4" \
  --report-run "$BASE/report_candidates/pool160_maira2_20260813_2f6a" \
  --report-run "$BASE/report_candidates/pool160_cxrmate_single_20260813_74d1" \
  --report-run "$BASE/report_candidates/pool160_llavarad_20260813_aa39" \
  --report-run "$BASE/report_candidates/pool160_chexagent2_fp32_20260813_51f8" \
  --report-run "$BASE/report_candidates/roentgen80_maira2_20260813_a31e" \
  --report-run "$BASE/report_candidates/roentgen80_cxrmate_single_20260813_d6a4" \
  --report-run "$BASE/report_candidates/roentgen80_llavarad_20260813_b472" \
  --report-run "$BASE/report_candidates/roentgen80_chexagent2_fp32_20260813_c593" \
  --model-path "$MODEL" --output-root "$OUTPUT" --run-id "$RUN_ID"

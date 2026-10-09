#!/bin/bash
# NOT sbatch. Requires separate approval for the 43 linked real MIMIC CXRs.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12714150/' /proc/self/cgroup
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORKSPACE/.cache/radeval_biovil_cpu/huggingface"
export XDG_CACHE_HOME="$WORKSPACE/.cache/radeval_biovil_cpu/xdg"
export TMPDIR="$WORKSPACE/.tmp/radeval_biovil_cpu"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME"
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 600s \
  CheXGenBench/metrics_venv/bin/python \
  TriCompose-v1.2/tools/score_radeval_biovil.py run \
  --run-id biovil_cpu_12714150_001 \
  --plan-root artifacts/protected/tricompose_v1_2/radeval_image_benchmark_plans/biovil_cpu_12714150_001 \
  --allow-mimic-images

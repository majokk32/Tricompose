#!/bin/bash
# NOT sbatch: existing approved job 12766754, 4 CPU / 32 GB / no GPU.
# Exactly 100 fixed released manual reports; frozen CheXbert, cached RadGraph.
# Two model threads, max 100 encoder examples, no retries or truncation.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12766754/' /proc/self/cgroup
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export TMPDIR="$WORKSPACE/artifacts/protected/tricompose_v1_2/manual_literal_reader_runs/manual100_12766754_001/tmp"
export XDG_CACHE_HOME="$WORKSPACE/artifacts/protected/tricompose_v1_2/manual_literal_reader_runs/manual100_12766754_001/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 600s \
  cxrmate/venv/bin/python \
  TriCompose-v1.2/tools/benchmark_manual_literal_readers.py

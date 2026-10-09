#!/bin/bash
# NOT sbatch: approved existing job 12766754, 4 CPUs / 32 GB / no GPU.
# All 56 existing invented reports; frozen offline XL; 10-minute execution cap.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12766754/' /proc/self/cgroup
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools:$WORKSPACE/TriCompose-v1.2/benchmarks:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export TMPDIR="$WORKSPACE/artifacts/protected/tricompose_v1_2/radgraph_assertion_runs/authored56_12766754_001/tmp"
export XDG_CACHE_HOME="$WORKSPACE/artifacts/protected/tricompose_v1_2/radgraph_assertion_runs/authored56_12766754_001/cache"
export HF_HOME="$WORKSPACE/.cache/radgraph_v12/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 600s \
  runtime/venvs/radgraph-xl-v12-12714150-001/bin/python \
  TriCompose-v1.2/tools/benchmark_radgraph_assertions.py

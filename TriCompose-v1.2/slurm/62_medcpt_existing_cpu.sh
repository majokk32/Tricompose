#!/bin/bash
# Not sbatch. Approved download + authored CPU smoke in existing job 12714150.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12714150/' /proc/self/cgroup
cd "$WORKSPACE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export NETRC=/dev/null
export HF_HOME="$WORKSPACE/.cache/medcpt_query_12714150_001/huggingface"
export XDG_CACHE_HOME="$WORKSPACE/.cache/medcpt_query_12714150_001/xdg"
export TMPDIR="$WORKSPACE/.tmp/medcpt_query_12714150_001"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME"
timeout --signal=TERM --kill-after=10s 1800s \
  runtime/venvs/radgraph-xl-v12-12714150-001/bin/python \
  TriCompose-v1.2/tools/run_medcpt_authored_smoke.py \
  --approved-download-and-authored-cpu-smoke

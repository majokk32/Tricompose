#!/bin/bash
# NOT sbatch. Existing approved CPU allocation; authorized private references.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12714150/' /proc/self/cgroup
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools"
export TMPDIR="$WORKSPACE/.tmp/radgraph_deploy_12714150_001"
export XDG_CACHE_HOME="$WORKSPACE/.cache/radgraph_v12/xdg"
export HF_HOME="$WORKSPACE/.cache/radgraph_v12/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 1800s \
  runtime/venvs/radgraph-xl-v12-12714150-001/bin/python \
  TriCompose-v1.2/tools/score_radeval_expert_radgraph.py --run-id expert_radgraph_12714150_001

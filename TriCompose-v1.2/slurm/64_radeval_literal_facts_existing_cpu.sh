#!/bin/bash
# Not sbatch. Reuse actual existing CPU allocation; cached graphs only.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12766754/' /proc/self/cgroup
test "${SLURM_JOB_ID:-}" = 12766754
cd "$WORKSPACE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/tools"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export NETRC=/dev/null
export HF_HOME="$WORKSPACE/.cache/radeval_literal_facts_12766754_001/huggingface"
export XDG_CACHE_HOME="$WORKSPACE/.cache/radeval_literal_facts_12766754_001/xdg"
export TMPDIR="$WORKSPACE/.tmp/radeval_literal_facts_12766754_001"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME"
timeout --signal=TERM --kill-after=10s 600s \
  runtime/venvs/report-context-v12-12576792/bin/python \
  TriCompose-v1.2/tools/diagnose_radeval_literal_facts.py run

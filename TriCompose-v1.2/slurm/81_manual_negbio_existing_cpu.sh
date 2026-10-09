#!/bin/bash
# NOT sbatch: reuse approved job 12766754, 4 CPU / 32 GB / no GPU.
# All 100 existing manual reports, official frozen parser, 200 passes incl replay.
# Real report strings stay inside private worker; no labels/scores enter requests.
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
rg --quiet '/job_12766754/' /proc/self/cgroup
module load openjdk/11.0.20.1_1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH
cd "$WORKSPACE"
timeout --signal=TERM --kill-after=10s 600s \
  runtime/venvs/chexpert-negbio-py36-12682821-v1/bin/python \
  TriCompose-v1.2/interfaces/chexpert_negbio_manual100_v1.py

#!/bin/bash
# PROPOSED ONLY: a resource swap requires fresh explicit user approval.
# Reuse the previously displayed, hash-locked worker body; no nested sbatch.
# The loader uses hardware-dependent dtype; this is not FP16/BF16 equivalence.
#SBATCH --job-name=tri_guard_img2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
TRI_BODY=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.2/agent/slurm/10_guarded_image_probe2_v100.sbatch
TRI_BODY_SHA=7e85cb1d160b9153fd401a2ec4adf73acbedf5a36768585a0b0563a271a5eee4
printf '%s  %s\n' "$TRI_BODY_SHA" "$TRI_BODY" | sha256sum --check --status
# Bash ignores the body's SBATCH comments: this stays in the single A40 job.
exec /bin/bash "$TRI_BODY"

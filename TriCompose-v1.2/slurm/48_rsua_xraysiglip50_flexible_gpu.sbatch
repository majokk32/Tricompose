#!/bin/bash
# Submit only after this complete script/resources are shown and explicitly approved.
# Frozen public-cohort diagnostic: 50 existing images + 2 fixed technical repeats.
# No generation, training, threshold fitting or candidate/winner changes.
#SBATCH --job-name=tri_v12_rs_siglip
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug,gpu
#SBATCH --constraint="p100|v100|a40|a100|l40s"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/runtime/venvs/chexagent2-srrg-cu121-v1/bin/python"
EVIDENCE_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PLAN="$EVIDENCE_ROOT/real_validation/rsua_siglip_plans/plan50_12645021_001"
PLAN_SHA=ca2fb6e62237b0c66cff988425eeaf3d51e3d09b029e5ddf002f34af0a90037f
TASK="$EVIDENCE_ROOT/task_runtime/rsua_siglip_${SLURM_JOB_ID}"
RUN_ID="rsua_siglip50_${SLURM_JOB_ID}"
test -x "$PYTHON"
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
for directory in "$EVIDENCE_ROOT" "$EVIDENCE_ROOT/slurm_logs" "$EVIDENCE_ROOT/task_runtime" "$PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' "$PLAN_SHA" "$PLAN/manifest.json" | sha256sum --check --status
test ! -e "$TASK"
test ! -e "$EVIDENCE_ROOT/real_validation/rsua_siglip_pilots/$RUN_ID"
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
chmod 0660 "$EVIDENCE_ROOT/slurm_logs/tri_v12_rs_siglip_${SLURM_JOB_ID}.out" \
  "$EVIDENCE_ROOT/slurm_logs/tri_v12_rs_siglip_${SLURM_JOB_ID}.err"
finish_permissions() {
  find "$TASK" -type d -exec chmod 2770 {} +
  find "$TASK" -type f -exec chmod 0660 {} +
}
trap finish_permissions EXIT
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" TORCH_EXTENSIONS_DIR="$TASK/cache/extensions"
export MPLCONFIGDIR="$TASK/cache/matplotlib" TMPDIR="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/real_validation/rsua_xraysiglip.py" run \
  --plan-run "$PLAN" --plan-manifest-sha256 "$PLAN_SHA" \
  --output-root "$EVIDENCE_ROOT/real_validation/rsua_siglip_pilots" --run-id "$RUN_ID" \
  --allow-rsua-siglip-probe >"$TASK/run.log" 2>&1
printf 'status=rsua_xraysiglip_worker_finished_no_selection_change\n'

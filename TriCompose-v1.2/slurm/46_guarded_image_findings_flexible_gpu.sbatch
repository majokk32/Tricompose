#!/bin/bash
# Complete script/resources and subsequent explicit approval before submission.
# Prospective mechanical guard check; no generation or clinical repair claim.
#SBATCH --job-name=tri_v12_guardhook
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
EVIDENCE_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PLAN="$EVIDENCE_ROOT/guarded_image_plans/guardhook_scope2_12645021_001"
TASK="$EVIDENCE_ROOT/task_runtime/guardhook_${SLURM_JOB_ID}"
RUN_ID="guardhook_scope2_${SLURM_JOB_ID}"
test -x "$PYTHON"
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
for directory in "$EVIDENCE_ROOT" "$EVIDENCE_ROOT/slurm_logs" "$EVIDENCE_ROOT/task_runtime" "$PLAN"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  2864941d29f60e890cdd51e88a7126986027b495345d4cf8a0ff23bdc2b77959 "$PLAN/manifest.json" \
  | sha256sum --check --status
mkdir -p -m 2770 "$EVIDENCE_ROOT/guarded_image_runs"
test "$(stat -c %a "$EVIDENCE_ROOT/guarded_image_runs")" = 2770
project_gid=$(stat -c %g "$EVIDENCE_ROOT/guarded_image_runs")
test "$project_gid" = 96293 || test "$project_gid" = 65534
test ! -e "$TASK"
test ! -e "$EVIDENCE_ROOT/guarded_image_runs/$RUN_ID"
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
chmod 0660 "$EVIDENCE_ROOT/slurm_logs/tri_v12_guardhook_${SLURM_JOB_ID}.out" \
  "$EVIDENCE_ROOT/slurm_logs/tri_v12_guardhook_${SLURM_JOB_ID}.err"
finish_permissions() {
  find "$TASK" -type d -exec chmod 2770 {} +
  find "$TASK" -type f -exec chmod 0660 {} +
}
trap finish_permissions EXIT
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/benchmarks:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
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
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/tools/verify_guarded_image_findings.py" run \
  --plan-run "$PLAN" --output-root "$EVIDENCE_ROOT/guarded_image_runs" --run-id "$RUN_ID" \
  --allow-guarded-image-verification >"$TASK/run.log" 2>&1
printf 'status=guarded_image_verification_completed_no_selection_change\n'

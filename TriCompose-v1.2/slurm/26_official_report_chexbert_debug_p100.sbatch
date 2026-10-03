#!/bin/bash
# Prepared only: show the complete script/resources and obtain approval first.
# Real report text is consumed internally; no report/EHR/image/key export.
#SBATCH --job-name=tri_v12_cbgold
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
CODE="$WORKSPACE/TriCompose-v1.2/real_validation"
DATASET=/project2/ruishanl_1185/datasets
AUDIT="$ROOT/real_validation/official_report_gold_coverage_12576792_source1"
TASK="$ROOT/task_runtime/official_report_benchmark_${SLURM_JOB_ID}"
RUN="official_chexbert_reports_${SLURM_JOB_ID}"
CB="$WORKSPACE/.cache/tricompose_report_eval/models/chexbert/chexbert.pth"
BERT="$WORKSPACE/.cache/tricompose_report_eval/models/bert-base-uncased"
test "${ALLOW_REAL_REPORT_BENCHMARK:-0}" = 1
test -n "${SLURM_JOB_ID:-}"
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
test ! -e "$TASK"
test ! -e "$ROOT/real_validation/$RUN"
test -x "$WORKSPACE/cxrmate/venv/bin/python"
chmod 0660 "$ROOT/slurm_logs/tri_v12_cbgold_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_cbgold_${SLURM_JOB_ID}.err"
printf '%s  %s\n' \
  7e0255de51deb8c28b1578fc35cb62b9641e141d3e7bb1412aac929bc6c01d2f "$CODE/run_official_report_benchmark.py" \
  ff41f73874ed172d92309089a48d9c86dd991fdbed9b5b5acf57c057430c8ea0 "$CODE/official_report_benchmark.py" \
  397ba6b707f34c4c0d4ce13e996b77eaa9cceadd3507742e99b8facbb62d7eb6 "$CODE/audit_official_report_gold.py" \
  bf0b5064c5a477bc673196080e2c16b2dc7dc15672f6cf589161761ddfe77fc4 "$AUDIT/manifest.json" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$WORKSPACE/TriCompose-v1.0/eval/report_v1_1/contracts.py" \
  00459f4c8c9faf009d60b56529991a690a53ac2b782e9ea1c21604c2789779cd "$WORKSPACE/TriCompose-v1.2/src/tricompose_v12/report_assertions.py" \
  7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18 "$WORKSPACE/TriCompose-v1.2/benchmarks/repair_cached_report_evidence.py" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
trap 'find "$TASK" -type d -exec chmod 2770 {} +; find "$TASK" -type f -exec chmod 0660 {} +' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" TMPDIR="$TASK/tmp"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" TORCH_EXTENSIONS_DIR="$TASK/cache/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WORKSPACE/cxrmate:$CODE:$WORKSPACE/TriCompose-v1.2/src:$WORKSPACE/TriCompose-v1.2/benchmarks:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
"$WORKSPACE/cxrmate/venv/bin/python" "$CODE/run_official_report_benchmark.py" \
  --gold-labels "$DATASET/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.1.0-test-set-labeled.csv" \
  --linkage-manifest "$DATASET/three_modalities/v2_labs_vitals/manifest.csv" \
  --metadata-audit-run "$AUDIT" --checkpoint "$CB" --bert-path "$BERT" \
  --batch-size 8 --output-root "$ROOT/real_validation" --run-id "$RUN" \
  --allow-real-report-benchmark >"$TASK/run.log" 2>&1
printf 'status=real_report_label_benchmark_finished_no_training_or_selection_change\n'

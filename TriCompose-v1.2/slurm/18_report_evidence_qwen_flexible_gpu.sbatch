#!/bin/bash
# Show this entire script/resources and obtain explicit approval before sbatch.
# Secondary synthetic-development report review only; no generation or repair.
#SBATCH --job-name=tri_v12_qwen_evid
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --constraint="v100|a40|a100|l40s"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
MODEL=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
BANK="$ROOT/benchmarks/report_polarity_dev48_20261002_001"
CONTEXT="$ROOT/verification_runs/chexbert_context_dev48_12561407"
TOOLS="$WORKSPACE/TriCompose-v1.2/benchmarks"
RUNS="$ROOT/verification_runs"
TASKS="$ROOT/task_runtime"
TASK="$TASKS/qwen_report_evidence_${SLURM_JOB_ID}"
RUN_ID="qwen_report_evidence_${SLURM_JOB_ID}"

test -x "$PYTHON"
test -r "$MODEL/model.safetensors"
for directory in "$ROOT" "$ROOT/slurm_logs" "$RUNS" "$TASKS"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  f22fa8d0071edbfffe9beb377f7edaf32c3eff2384e7da0019bb294e34990ad7 "$BANK/manifest.json" \
  9f92be41b94ce2928ce25cd5a7632615cdefa4a48adca3d4ce140f93543a1121 "$BANK/resolver.jsonl" \
  4c75018be2950953b96e71736968415359be60f7fed5833b28202a0e5a3b0825 "$BANK/intervention_key.jsonl" \
  dd5dbbb48819e8d9ac1f03ffdc1f51e9a42650bed516ad5428639b23682da426 "$CONTEXT/diagnostic.json" \
  6bf0a2e7359bdeb016911fa80eb351ab2995dd3bbe7f2661cc731e1a4b554bc0 "$TOOLS/verify_report_evidence_qwen.py" \
  7d3b1251da77a8cfc3037abbc12f78e17a4d6f230b38fa5e30be5356f5755457 "$TOOLS/analyze_report_evidence_qwen.py" \
  2bf3be71fbde99dce3d2a49ebdfef58d1c3f4dd96f9977d263c73482f3aca17b "$TOOLS/diagnose_chexbert_polarity.py" \
  3272194c82a7a6bac6cb8d3c9638866d51a0dbd19a406d566130b6f2e90965c1 "$TOOLS/verify_candidate_findings_qwen.py" \
  bd2749330be65a1445ac9fe383c244ee79b8e23e109de7332ee6a2602a67e402 "$TOOLS/prepare_expanded_polarity.py" \
  25c2a480c52394e9ebf76a36cd86fff05fce2e6a9036059fe11c0930a443ce64 "$WORKSPACE/src/tricompose/verifiers/qwenvl_cxr_report.py" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$WORKSPACE/TriCompose-v1.0/eval/report_v1_1/contracts.py" \
  | sha256sum --check --status
test ! -e "$TASK"
test ! -e "$RUNS/$RUN_ID"
test ! -e "$RUNS/${RUN_ID}_analysis"
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
chmod 0660 "$ROOT/slurm_logs/tri_v12_qwen_evid_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_qwen_evid_${SLURM_JOB_ID}.err"
finish_permissions() {
  find "$TASK" -type d -exec chmod 2770 {} +
  find "$TASK" -type f -exec chmod 0660 {} +
}
trap finish_permissions EXIT

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$TOOLS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" MPLCONFIGDIR="$TASK/cache/matplotlib"
export TMPDIR="$TASK/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

"$PYTHON" "$TOOLS/verify_report_evidence_qwen.py" \
  --bank-run "$BANK" --model-path "$MODEL" --max-new-tokens 512 \
  --output-root "$RUNS" --run-id "$RUN_ID" >"$TASK/logs/verify.log" 2>&1
printf 'status=secondary_report_evidence_review_completed\n'
"$PYTHON" "$TOOLS/analyze_report_evidence_qwen.py" \
  --bank-run "$BANK" --review-run "$RUNS/$RUN_ID" --context-run "$CONTEXT" \
  --output-root "$RUNS" --run-id "${RUN_ID}_analysis" >"$TASK/logs/analysis.log" 2>&1
printf 'status=secondary_evidence_alignment_completed_no_selection_change\n'

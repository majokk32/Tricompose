#!/bin/bash
# Score 12 fully synthetic CXRs and 48 generated reports; no real targets.
# This script must be shown in full and explicitly approved before sbatch.
#SBATCH --job-name=tri_v12_score48
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --constraint=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/job_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/job_logs/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EVAL_ROOT="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_1"
PILOT="$BASE/evaluation/v12_explicit_two_12538545_001"
OUT="$PILOT/scoring_${SLURM_JOB_ID}"
REGISTRY="$PILOT/registry"
STAGING="$BASE/staging/v12_explicit_ehr_two_20261001_001"
XRV_PY=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
CB_PY="$WORKSPACE/cxrmate/venv/bin/python"
BV_PY="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
XRV_CACHE="$WORKSPACE/.cache/tricompose_v1/scoring/job_10871232/xrv"
XRV_WEIGHT=nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
CHEXBERT="$WORKSPACE/.cache/tricompose_report_eval/models/chexbert/chexbert.pth"
BERT="$WORKSPACE/.cache/tricompose_report_eval/models/bert-base-uncased"
BIOVIL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"

for required in "$XRV_PY" "$CB_PY" "$BV_PY" "$XRV_CACHE/$XRV_WEIGHT" \
  "$CHEXBERT" "$BERT/config.json" "$BERT/vocab.txt" \
  "$BIOVIL/pytorch_model.bin" "$BIOVIL/biovil_t_image_model_proj_size_128.pt" \
  "$VENDOR" "$REGISTRY/manifest.json"; do
  test -r "$required"
done
printf '%s  %s\n' \
  6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1 \
  "$CHEXBERT" | sha256sum --check --status
printf '%s  %s\n' \
  e44d7fa93a52eed2a0a1d1fa6470a0a5dfdab8282b550886eae74bb94a02a19c \
  "$REGISTRY/manifest.json" | sha256sum --check --status
test ! -e "$OUT"
mkdir -p "$OUT/cache/xrv" "$OUT/tmp" "$OUT/logs"
chmod 2770 "$OUT" "$OUT/cache" "$OUT/cache/xrv" "$OUT/tmp" "$OUT/logs"
install -m 660 "$XRV_CACHE/$XRV_WEIGHT" "$OUT/cache/xrv/$XRV_WEIGHT"

CXR_ARGS=()
for model in roentgen_v2 chexgenbench_sana chexgenbench_pixart; do
  CXR_ARGS+=(--cxr-run "$BASE/cxr_candidates/v12_explicit_two_${model}_s2_12533465")
done
REPORT_ARGS=()
for model in maira2 cxrmate_single llavarad chexagent2; do
  REPORT_ARGS+=(--report-run "$BASE/report_candidates/v12_explicit_two_${model}_12538545")
done

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$OUT/cache/xdg" HF_HOME="$OUT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$OUT/cache/torch" MPLCONFIGDIR="$OUT/cache/matplotlib"
export CUDA_CACHE_PATH="$OUT/cache/cuda" TMPDIR="$OUT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Keep all evaluator output in the protected run; Slurm stdout has status only.
export PYTHONPATH="$EVAL_ROOT"
"$XRV_PY" "$EVAL_ROOT/extract_cxr_labels_xrv.py" "${CXR_ARGS[@]}" \
  --cache-dir "$OUT/cache/xrv" --weight-filename "$XRV_WEIGHT" \
  --output-root "$OUT" --run-id xrv >"$OUT/logs/xrv.log" 2>&1
printf 'status=xrv_completed\n'

export PYTHONPATH="$WORKSPACE/cxrmate:$EVAL_ROOT"
"$CB_PY" "$EVAL_ROOT/extract_report_labels_chexbert.py" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" \
  --checkpoint "$CHEXBERT" --bert-path "$BERT" --batch-size 16 \
  --output-root "$OUT" --run-id chexbert >"$OUT/logs/chexbert.log" 2>&1
printf 'status=chexbert_completed\n'

export PYTHONPATH="$EVAL_ROOT:$VENDOR"
"$BV_PY" "$EVAL_ROOT/score_report_cxr_biovil.py" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" --model-path "$BIOVIL" \
  --output-root "$OUT" --run-id biovil >"$OUT/logs/biovil.log" 2>&1
printf 'status=biovil_completed\n'

export PYTHONPATH="$EVAL_ROOT"
"$CB_PY" "$EVAL_ROOT/aggregate_crossmodal.py" --staging-run "$STAGING" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" \
  --cxr-labels "$OUT/xrv/cxr_finding_labels.json" \
  --report-labels "$OUT/chexbert/report_finding_labels.json" \
  --biovil-scores "$OUT/biovil/biovil_scores.json" \
  --output-root "$OUT" --run-id crossmodal >"$OUT/logs/crossmodal.log" 2>&1
"$CB_PY" "$EVAL_ROOT/aggregate_ehr_edges.py" --staging-run "$STAGING" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" \
  --cxr-labels "$OUT/xrv/cxr_finding_labels.json" \
  --report-labels "$OUT/chexbert/report_finding_labels.json" \
  --output-root "$OUT" --run-id ehr_edges >"$OUT/logs/ehr_edges.log" 2>&1
"$CB_PY" "$EVAL_ROOT/finalize_unified_score_table.py" \
  --score-table-run "$REGISTRY" --crossmodal-run "$OUT/crossmodal" \
  --policy-config "$EVAL_ROOT/static_score_policy_v1_1.json" \
  --output-root "$OUT" --run-id scored >"$OUT/logs/scored.log" 2>&1
"$CB_PY" "$EVAL_ROOT/select_edge_specific_candidates.py" \
  --score-table-run "$OUT/scored" --ehr-edge-run "$OUT/ehr_edges" \
  --policy-config "$EVAL_ROOT/lexicographic_selection_policy_smoke48.json" \
  --output-root "$OUT" --run-id selection >"$OUT/logs/selection.log" 2>&1
printf 'status=score_and_selection_completed\n'

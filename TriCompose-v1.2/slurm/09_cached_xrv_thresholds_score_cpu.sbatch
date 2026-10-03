#!/bin/bash
# Cached weak-reference threshold fitting and 48-candidate re-evaluation.
# Show this complete script and resource request before submitting.
#SBATCH --job-name=tri_v12_xrvcal
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
EVAL_ROOT="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_1"
PILOT="$BASE/evaluation/v12_explicit_two_12538545_001"
OLD="$PILOT/scoring_12538856"
OUT="$PILOT/weak_thresholds_${SLURM_JOB_ID}"
STAGING="$BASE/staging/v12_explicit_ehr_two_20261001_001"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
REAL="$WORKSPACE/artifacts/protected/tricompose_v1_2/real_validation/real_xrv_weak_valtest128_20261001_001/scores.json"
DATASET_MANIFEST=/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals/manifest.csv

for required in "$PYTHON" "$REAL" "$DATASET_MANIFEST" \
  "$OLD/xrv/cxr_finding_labels.json" "$OLD/chexbert/report_finding_labels.json" \
  "$OLD/biovil/biovil_scores.json" "$PILOT/registry/manifest.json"; do
  test -r "$required"
done
printf '%s  %s\n' \
  e44d7fa93a52eed2a0a1d1fa6470a0a5dfdab8282b550886eae74bb94a02a19c \
  "$PILOT/registry/manifest.json" | sha256sum --check --status
test ! -e "$OUT"
mkdir -m 2770 "$OUT"
mkdir -m 2770 "$OUT/logs" "$OUT/cache" "$OUT/tmp"

export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$EVAL_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$OUT/cache/xdg" HF_HOME="$OUT/cache/huggingface"
export TORCH_HOME="$OUT/cache/torch" TMPDIR="$OUT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Uses cached scores and source split metadata; no real image/report read.
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/real_validation/calibrate_cached_xrv.py" \
  --cached-real-scores "$REAL" \
  --cached-synthetic-labels "$OLD/xrv/cxr_finding_labels.json" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --min-per-class 20 --output-root "$OUT" --run-id calibration \
  >"$OUT/logs/calibration.log" 2>&1
printf 'status=threshold_check_completed\n'

if ! "$PYTHON" -c \
  'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m["findings_enabled"] else 1)' \
  "$OUT/calibration/manifest.json"; then
  printf 'status=reranking_skipped reason=no_supported_calibration_finding\n'
  exit 0
fi

CXR_ARGS=()
for model in roentgen_v2 chexgenbench_sana chexgenbench_pixart; do
  CXR_ARGS+=(--cxr-run "$BASE/cxr_candidates/v12_explicit_two_${model}_s2_12533465")
done
REPORT_ARGS=()
for model in maira2 cxrmate_single llavarad chexagent2; do
  REPORT_ARGS+=(--report-run "$BASE/report_candidates/v12_explicit_two_${model}_12538545")
done

"$PYTHON" "$EVAL_ROOT/aggregate_crossmodal.py" --staging-run "$STAGING" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" \
  --cxr-labels "$OUT/calibration/cxr_finding_labels.json" \
  --report-labels "$OLD/chexbert/report_finding_labels.json" \
  --biovil-scores "$OLD/biovil/biovil_scores.json" \
  --output-root "$OUT" --run-id crossmodal >"$OUT/logs/crossmodal.log" 2>&1
"$PYTHON" "$EVAL_ROOT/aggregate_ehr_edges.py" --staging-run "$STAGING" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" \
  --cxr-labels "$OUT/calibration/cxr_finding_labels.json" \
  --report-labels "$OLD/chexbert/report_finding_labels.json" \
  --output-root "$OUT" --run-id ehr_edges >"$OUT/logs/ehr_edges.log" 2>&1
"$PYTHON" "$EVAL_ROOT/finalize_unified_score_table.py" \
  --score-table-run "$PILOT/registry" --crossmodal-run "$OUT/crossmodal" \
  --policy-config "$EVAL_ROOT/static_score_policy_v1_1.json" \
  --output-root "$OUT" --run-id scored >"$OUT/logs/scored.log" 2>&1
"$PYTHON" "$EVAL_ROOT/select_edge_specific_candidates.py" \
  --score-table-run "$OUT/scored" --ehr-edge-run "$OUT/ehr_edges" \
  --policy-config "$EVAL_ROOT/lexicographic_selection_policy_smoke48.json" \
  --output-root "$OUT" --run-id selection >"$OUT/logs/selection.log" 2>&1
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/benchmarks/analyze_candidate_scorers.py" \
  --selection-run "$OUT/selection" \
  --crossmodal-details "$OUT/crossmodal/crossmodal_details.json" \
  --report-structure "$PILOT/structure/report_unimodal_details.json" \
  --output-root "$OUT" --run-id audit >"$OUT/logs/audit.log" 2>&1
printf 'status=cached_score_and_selection_completed\n'

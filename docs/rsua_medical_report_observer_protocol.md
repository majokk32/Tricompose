# Same RSUA50: official SRRG + CheXbert diagnostic

Prepared after the Qwen image observer's RSUA50 result was inspected. This is
an explicitly development-time follow-up, not an untouched test or a scorer
qualification protocol. No new cohort, data acquisition, checkpoint download,
threshold fitting, training or original-winner update.

## Scope and scientific role

Reuse every original image and opaque source order from the existing 25
published pneumonia / 25 paper-described normal-cohort proxy RSUA pilot.
The official release's Non-Covid label does not become independently
adjudicated per-image normality. Patient grouping, age domain, training overlap
and transfer to synthetic images remain unverified. Only pneumonia has a
reference; the other findings are unevaluated, not assumed absent.

The local `StanfordAIMI/CheXagent-2-3b-srrg-findings` checkpoint is specialized
for structured findings generation. Preserve its exact deployed official
prompt, `Structured Radiology Report Generation for Findings Section`, seed
42, greedy decoding and 512-token cap. Reuse the already deployed float32
runtime, revision `9f7225fc382ddd1297ade1aa796da660237940bc`, and full local
XraySigLIP dependency. Do not force this checkpoint into a new classification
or JSON prompt. The exact existing loader remains frozen and unmodified.

Generate one protected report per source image, then use the exact existing
frozen CheXbert implementation/checkpoint/class mapping to extract 14 named
states. A report-derived finding is a measurement of this two-model pipeline,
NOT an independent image classifier or an adjudicated truth. CheXbert is also
the candidate-label scorer, and CheXagent shares XraySigLIP's visual encoder:
do not call their agreement independent validation or a gold majority vote.

This experiment is separate from the Qwen numeric scheduling role. It tests
the medical report-to-label observer, not LLM planner superiority, a targeted
repair policy, or saved generation cost.

## Execution contract

- CPU preparation uses hash-bound manifests, derived image inventory, schemas,
  code and model-file stats only. No source pixels or reference rows are parsed.
- Approved GPU execution checks complete frozen model asset hashes first.
  All 50 source BMP/PNG lineage hashes and order are retained; no replacement
  or easy-case filtering. Existing source PNGs are used without new conversion.
- A mechanical PNG guard precedes each report request; it detects basic
  invalidity, not clinical anatomy. The official decoder uses a local opaque
  image locator; byte hashes are checked before/after its call. No EHR, source
  report, reference class or previous scorer output is supplied to either model.
- Report and labeler run sequentially in their separate existing environments.
  The report process exits before the labeler loads; models are not co-resident.
- Reserve/fsync every load/request before execution. At most 50 report requests,
  50 single-report label requests and one load for each model, zero retries.
  Actual generate/forward counts are distinct from reserved request counts;
  failed loads and reserved failed requests remain charged, not free.
- A scoped observer records generated token lengths without altering any
  official generation arguments, and restores the original method even on
  exception. Empty reports or output reaching the 512-token cap are unavailable.
- Inspect the exact CheXbert-normalized token length, including special tokens,
  before its unchanged forward. Overlength reports are unavailable, not silently
  truncated. Labeler unknown/uncertain remain four-state predictions, distinct
  from failed, blocked and unattempted stages. Runtime failure stops subsequent
  calls of that stage; preserve every original denominator and completed prefix.
- Fsync full predictions before reference evaluation. Real reference labels and
  old XRV per-image results are consumed only internally in the approved Slurm
  worker, never during preparation or public logging.
- Atomic new protected run; no resume, automatic resubmission or overwrite.
  Failure retains fsynced charge journals and available protected reports.
  Public logs contain sanitized status/hashes only. Project-group access and
  2770/0660 permissions apply to outputs, reports, caches and private diagnostics.

## Predeclared readouts

Use the same pneumonia-only full-denominator four-state summaries as the
previous Qwen diagnostic: class-specific explicit support, explicit coverage,
conditional support, explicit-only TP/FN/TN/FP, unknown/uncertain/unavailable
counts and missingness bounds. Bounds are not confidence intervals. Qwen's
already observed result is contextual, not a new confirmatory paired outcome.

Also show the cached XRV default 0.5 and unchanged transported 0.55457607
profiles on the same sources. No new threshold, best-scorer selection, template
search, new clinical total or probability calibration. The pipeline gives
categorical states; AUROC/AP/Brier/ECE are NA rather than invented probabilities.
Record real request/generate/forward/load counts, failure stage, runtime and
allocated VRAM. Report generation is additional measured cost, not a free
classifier call. Original synthetic EHRs/images/reports/selections remain fixed.

Keep clinical acceptance, independent clinical accuracy, established fault
location, successful repair and measured saved model calls unavailable/false.
Even a good proxy result cannot grant automatic clinical repair authority.

## Files and approval boundary

Worker: `TriCompose-v1.2/real_validation/rsua_medical_report_observer.py`.
Tests: `TriCompose-v1.2/tests/test_rsua_medical_report_observer.py`.
Reuse the authenticated local full-asset receipt from
`llm_fresh_report_plans/fresh_report2_fullassets_12799642_001` and original
RSUA50 input plan. Neither consumed source plan/program is edited.

Proposed resources: one debug A40, two CPUs, 32 GiB host RAM, ten-minute
allocation cap. Parent timeout 540s; report subprocess 420s; label subprocess
60s. These caps do not promise runtime or queue start. The exact deployed
float32 report worker retains a 24-GiB minimum planning class, so do not
silently switch to 16-GiB cards, quantization or a different dtype.

The complete new script/resource request must be displayed and explicitly
approved before sbatch. This protocol, existing GPU approvals and CPU-only
preparation do not authorize the new submission.

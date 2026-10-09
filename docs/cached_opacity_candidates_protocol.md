# Frozen exact-opacity synthetic-candidate diagnostic

## Scope, before new predictions

This is a DEVELOPMENT diagnostic of the unchanged historical **80 synthetic EHR
anchors × 3 existing CXR models × 4 existing report experts = 960 triple slots**.
All 240 synthetic images are included, without score-selected cases or new
generation. No original EHR, facts, prompts, images, reports, thresholds, scores,
choices or consumed source file is changed. It is not a new router/selector.

The RICORD-1C 50-image result is used as a finding-specific frozen-scorer
diagnostic. Only its sealed aggregate summary is read, not patient records,
source keys or pixels. Its selected, unanimous, single-image reference and
clinical-display/training-overlap limitations still apply. Neither AUROC nor
accuracy is transferred to a synthetic candidate. It does **not** validate
pneumonia, EHR fidelity, report-label extraction or a faulty-modality verdict.

## Frozen score definitions

- Primary readout: **exact XRV `lung_opacity` head**, operating-point-normalized
  score, predeclared threshold **0.5**, not calibrated disease probability.
- Secondary same-call diagnostic: `max(lung_opacity, infiltration)` at **0.5**.
  This isolates aggregation changes with the same model call/preprocessing.
- Old `finding_probabilities.lung_opacity` cache stores only that maximum;
  the two original head scores cannot be recovered from it. New exact scores
  require an explicitly approved GPU job. No subtraction/label renaming.
- No refitting, tuning, uncertain band, best-threshold choice, calibration or
  model/template selection from this cohort. No Brier/ECE/clinical AUROC.
- The existing checkpoint is unchanged and hash-matches the RICORD run and
  original candidate XRV cache. Old cache has no runtime preprocessing source
  fingerprint, so old-vs-new cache differences are diagnostic, not guaranteed
  floating-point reproducibility or controlled head-only superiority.

## Cached evidence and denominator contract

Read only hash-bound synthetic metadata, cached four-state vectors and CSV
cells. The old synthetic-only schema/import adapter checks the 80 × 3 × 4
grid, artifact lineage, fixed EHR states and cached source categories.
Metadata manifests prove known synthetic staging origin; no EHR/report/prompt
bodies are read. Preparation only stats existing PNGs; evaluation verifies
their hashes and opens them inside the separately approved GPU allocation.

Each row retains cached EHR and report opacity states. Only positive/negative
on both sides is comparable; unknown/uncertain remains **not comparable**.
Do not infer opacity from pneumonia, CHF, medications, a global No Finding
statement, or absence of a diagnosis. In this cached extraction inventory,
all 80 EHR opacity states are unknown. This is not an EHR medical re-review.

Three edge readouts are unverified `proxy_support`, `proxy_opposition` or
`not_comparable`, **not** clinical contradictions or error localization.
Per-model/expert agreement divides support by comparable pairs and separately
reports comparable coverage over all rows. No comparable pair means NA, not
zero, perfect agreement or success. EHR–CXR/EHR–Report opacity comparisons
are expected unavailable, and will not be manufactured to create a triple
score. Report-only CheXbert proposals are not qualified image truth.

Each image is scored once, its score reused for four report slots. The 960
rows are not independent patients; image counts use 240, EHR counts use 80,
report-artifact duplicates are separately counted. No independent expert-vote
majority is assumed. No ranked winners or clinical-quality improvements are
reported from these diagnostics.

## Execution, provenance and failure handling

Tool: `TriCompose-v1.2/tools/score_cached_opacity_candidates.py`.
Tests: `TriCompose-v1.2/tests/test_cached_opacity_candidates.py`, invented
vectors/scores/hashes only, no real image/model fixtures.

Preparation requires an existing CPU Slurm allocation, fixes source/test/
protocol/legacy-helper/import-closure and environment hashes and creates a
new atomic protected plan. Both plan and original source hashes are rechecked
before and after evaluation. The GPU guard precedes all model/image access.
Offline only; no checkpoint downloads, network calls, credentials or training.
Caches/temp/bytecode policy stays within the workspace.

Every original candidate CSV cell and row order is preserved with `opacity_`
fields appended in a **new** run. Protected directories/files are 2770/0660,
project-group only. Existing run IDs are refused. Per-image failures retain
unknown new scores and all original 960 rows/240 image denominators, without
replacement or best-subset reporting. Model calls count attempts, not just
successes. Public logs show sanitized status/counts/hash only. Model/native
output stays private. New inference requires the complete script/resources
shown and explicit approval after that display.

This protocol and its consumed tool/tests become immutable once the plan is
written. Follow-up changes require a new version and a new plan/run ID.

# TriCompose scorer-validation progress

This is a non-sensitive code handoff. Detailed scores and all patient-derived
or synthetic outputs remain below `artifacts/protected/`, excluded by Git.

## Completed

- The local official manual report-label source passed a metadata coverage
  audit. Source Airspace Opacity is preserved, not relabeled as Lung Opacity.
- Frozen CheXbert text-state extraction completed as job 12594397. It retains
  unavailable records, four states, per-head confusion and scope coverage.
- Frozen BioViL-T single-fact positive/negative probes completed as job
  12597637 on the existing real cohort. All three authored templates were
  predeclared, scored label-blind and retained; reference unknown/uncertain
  does not become negative. Both tasks completed and derived metrics/asset
  hashes/permissions were verified.
- Existing synthetic candidate evidence received a derived diagnostic overlay,
  not a new score or selection weight. Original tables and winners are unchanged.

These steps test scorer components; they do not prove complete triple accuracy,
clinical error localization or targeted regeneration. Global similarity is
secondary evidence, not the sole repair judge. No training, fitting, downloads,
external patient-data API call or test-guided rule/weight/template tuning occurred.

## Publication scope

Publish the benchmark implementations, required helper dependency closure,
invented-fixture tests, approved-script sources and this non-sensitive handoff.
Include only the existing BioViL helper's two-line Slurm guard; preserve the
other uncommitted V1.0/V1.1 changes. Do not force-add protected artifacts,
environments, model repositories, checkpoints, credentials or exports.
The source dependency chain imports without a model load, the six included
fixture-test groups pass **151 tests**, and the file allowlist passes source/
credential/private-path screening. No protected result is staged for publication.

Historical scripts encode source/asset bindings and completed run provenance;
every new submission still requires full-script review and explicit approval.
A fresh code clone does not include the local assets needed for inference.

## Current independent-reference gap

The metadata-only check of the known `/project2/ruishanl_1185/datasets`
directories found MIMIC/linkage resources, but did not confirm an independent
image-level annotation source. No raw source rows, reports or image pixels were
opened for this check. Existing protected human-review readiness remains
`pending_human_annotation`; no returned clinical image gold was found in the
inspected review-run inventory.

The source report labels, automatic classifiers and four reports derived from
one CXR cannot provide independent image truth. The diagnosis is therefore a
missing reference resource, not an identified generator failure. Do not claim
global CARC absence based on this limited directory search.

## Next decision

1. Use authorized independent image/report adjudication, or identify an image-
   annotated dataset with appropriate access and licensing. Any new dataset
   acquisition/model job needs separate authorization.
2. Freeze cohort, reference scope and development/final evaluation rules before
   the new clinical test. Existing val/test pilots are not an untouched final set.
3. Only after that evidence gate, validate controlled error localization and
   compare equal-budget targeted repair against fixed/static composition.

Engineering fixtures may continue in parallel, but a controlled mechanical
intervention diagnostic must not be reported as clinical localization accuracy.
No trained router or new generator is needed.

Entry points and CARC-only asset/result paths:
[scorer-validation README](../TriCompose-v1.2/real_validation/README.md).

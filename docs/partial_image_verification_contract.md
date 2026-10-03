# CXR-before-report verification / 报告生成前的图像阶段验证

This is a separately frozen engineering extension, not a changed ranking policy
or a clinically validated controller. The original full-triple verifier and its
archived source hashes remain unchanged. No models, datasets or GPU jobs are
executed by this contract.

## Phases and missingness

1. Keep one immutable `EHRAnchor`, with the original EHR/facts hashes, four-state
   vector and cached categories. Never add a diagnosis/device to make it richer.
2. `CachedImageEvidence` binds an existing CXR's identity, model/seed, EHR and
   image hashes, cached XRV state vector, source-cache hash and image-validity
   metadata. It represents a classifier result that already exists, not an
   image waiting for classification. Unknown/uncertain states remain missing
   comparisons; an unavailable head is not invented. Missing image-validity
   metadata produces an unresolved status, not a passing gate.
3. `verify_image_phase(image, anchor)` returns a report-blind partial receipt.
   EHR–CXR support/opposition/coverage may be read out. Both report edges, report
   identity/hash/state vector and all-three support are `null`, and the report
   lifecycle is `not_generated`. Do not use an empty report, dummy hash or all
   unknown labels to stand in for a modality that does not yet exist.
4. `bind_completed_report(partial, completed, anchor)` links a later completed
   receipt without modifying the partial receipt. It requires the same fixed
   EHR, image ID/hash, XRV states and EHR–CXR readout. Changed report evidence
   cannot retrospectively change image verification. Several report models
   can link to the same partial receipt; they are not independent image votes.

```python
# Existing cache evidence only; no GPU side effects.
image = image_evidence_from_legacy_candidate(candidate, state_cache_sha256)
partial = verify_image_phase(image, fixed_anchor)
full = verify_candidate(completed_candidate, fixed_anchor)
link = bind_completed_report(partial, full, fixed_anchor)
```

The image projection reads only image lineage, image validity and XRV states;
it works when every report field, report label, triple gate, winner flag, scalar
score and cost field is absent. The cohort loader checks the complete historical
cache separately. Image receipts do not rank or select candidates, schedule a
report, reject a case, certify a repair or authorize model execution.

## Cache reconstruction is not prospective inference

This extension explicitly supports only the historical uncalibrated fourteen
finding profile. It does not silently reinterpret that cache as the calibrated
eight-head scoped preview. Its source-category provenance remains limited to
the existing cache; no new raw EHR evidence review is claimed. A new calibration
or prospective scorer requires its own contract/provenance adapter.

Inside the EXISTING Slurm CPU allocation, load the previously completed whole
synthetic cohort and full verification receipts with bounded size and manifest
hash checks. Reconstruct one partial receipt per existing image, independently
of which report is linked, and bind every original completed triple. Preserve
all EHRs, selections, actions and costs. Do not inspect patient source inputs,
report bodies or image pixels. Do not submit a job or call a model/API.

All links are marked `retrospective_phase_reconstruction_not_actual_execution_order`.
The cache test cannot prove the original job checked CXR before generating its
report, report calls were skipped, latency fell, or quality improved. Unresolved
and no-direct-EHR cases stay in the cohort with unavailable rates. Raw cached
proxy agreement is not clinical truth, and all clinical acceptance/repair flags
are false. This is previously inspected DEVELOPMENT evidence, not a new
independent confirmation experiment.

Write a new atomic protected run with config, anchors, partial receipts, report
bindings, summary, bilingual report and all source/result hashes. Do not overwrite
the full verification run. Prospective generation hooks, the execution policy
and generator/classifier failure/retry costs remain subsequent work; any new
GPU job needs its complete script/resources and explicit approval.

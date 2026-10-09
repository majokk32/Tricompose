# EHR–CXR reliability: cached diagnostic audit

This follows the completed two-case bounded retry and its recovered BioViL
endpoint. It changes no generator, EHR, prompt, threshold, gate or selection.
The next requirement is evaluator evidence, not another blind generation run.

## What was checked

Inside existing CPU allocation 12625457, the audit authenticated the completed
trial/plan, frozen checkpoint and preprocessing/mapping source fingerprints,
threshold bundle, saved labels and unchanged selection. It recomputed all
3,388 named state entries across 240 historical images and two new images,
plus the original two decisions. No discrepancy was found. This is an
arithmetic/lineage check, not a clinical accuracy check.

Only source code, synthetic score/anchor metadata and real-validation aggregate
statistics were consumed. No real patient record, report, image, per-row real
reference, synthetic report body or image pixels were opened. No factory,
inference, training, refit, API, download or new Slurm submission occurred.
Seven threshold and fifteen receipt fixture tests pass.

## Reliability limits found

1. Both retry triggers concern cached diagnosis-category pneumonia positives.
   An unchanged synthetic diagnosis is the requested generation intent, not
   independently adjudicated current-image ground truth. The cached extraction
   itself was not re-adjudicated by this audit.
2. The fixed XRV threshold is 0.55457607 in operating-point-normalized score
   space, not disease probability. All four old/new scores are below both this
   threshold and the historical default 0.5. This is not a near-boundary flip.
   Brier/ECE remain inapplicable; no lower threshold was selected post hoc.
3. Thresholds were fit on 103 groups using report-extracted weak labels. The
   diagnostic test has 105 groups, but finding-specific comparable denominators
   differ. It is label-stratified and previously descriptively inspected, not a
   natural-prevalence or untouched paper-final test.
4. Pneumonia's fitted test sensitivity against those weak labels is **9/22
   (0.4091)**, specificity **16/21 (0.7619)**, balanced accuracy **0.5855**.
   Its historical default-0.5 balanced accuracy is **0.6245**. These are not
   adjudicated clinical false-negative rates or a reason to change the threshold
   on synthetic results. A single negative XRV output cannot establish which
   modality is clinically wrong.
5. Of 80 fixed EHRs, four have explicit pneumonia facts, one pleural-effusion
   fact and three support-device facts. Only **5/80** have constraints covered
   by enabled XRV heads; the device head is unavailable. The other EHRs remain
   present with missing comparison, not negative labels or perfect agreement.
6. This frozen bundle's uncertainty margin is zero. It has no nonzero
   abstention band for an enabled finite score. No band was added or fitted.

BioViL evaluates generated-image/generated-report compatibility, whereas this
EHR edge compares a finding-specific image proxy with fixed EHR intent. Higher
BioViL can coexist with low EHR proxy support. Neither observation resolves
clinical correctness without independent evidence. The original 0/2 retry gate
and both historical choices remain unchanged.

## Immutable output

Protected directory:
`artifacts/protected/tricompose_v1_2/ehr_cxr_reliability_audits/retry2_reliability_12625457_001/`.

- `RESULTS_CN_EN.md`: bilingual findings, limits and ordered next steps.
- `synthetic_score_comparison.csv`: exact old/new finding scores and margins.
- `weak_reference_head_metrics.csv`: all eight enabled heads' aggregate test
  confusion counts, sensitivity, specificity and balanced accuracy.
- `audit.json` and `manifest.json`: provenance, invariants and hashes.

Runtime: 3.410 seconds. Manifest SHA256:
`583fbae662f9ff98c4b9bd43de192127748aa11f29b810ce905018d902f9ca15`.
Project-private permissions and unchanged source hashes were checked.

## Next benchmark boundary

First audit availability and provenance of an image-annotated reference cohort
and compatible existing frozen classifier. This audit did not search new
datasets and does not establish that such a resource exists locally. Existing
report-label pilots must not be relabeled as independent image truth.

Before looking at new model scores, seal the reference kind, finding mapping,
preprocessing/checkpoint revisions, explicit case inventory, patient-level
calibration/confirmation boundaries, exclusions and uncertainty/coverage
readouts. Existing radiologist-annotated benchmark labels can avoid requiring
the user to perform a new manual review; access, storage and provenance still
need checking. A second frozen model or correlated report votes are not a
substitute for reference labels.

Evaluate per-finding discrimination, sensitivity/specificity and coverage
first, then controlled wrong-modality localization, then bounded regeneration
against same-budget baselines. Keep this inspected synthetic pool developmental.
Any new download or GPU execution needs its own authorization; show the exact
complete batch script/resources before asking to submit. No new experiment is
authorized by this diagnostic report.

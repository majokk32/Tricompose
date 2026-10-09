# RICORD-1C frozen XRV opacity diagnostic / 50 例评分结果

Completed 2026-10-05, job **12667524**, V100 node `d13-04`.
Slurm COMPLETED, exit 0:0, elapsed **00:02:21**. Worker display + inference +
statistics runtime **127.811592 seconds**. Peak host RSS 1,489,380K; peak
PyTorch allocated CUDA tensors **0.043 GiB**, not total device/context memory.
The ten-minute request was a safety cap, not actual computation time.

## Completion and provenance

- Exactly **50/50 images scored**, 25 derived opacity-positive and 25 negative,
  50 unique patients; zero failures/substitutions; coverage 1.0.
- 50 decoded displays / 50 classifier calls / one frozen model load;
  **zero training, generation or threshold-fitting calls**.
- Source cohort, images, checkpoint, old threshold bundle and historical
  candidate choices are unchanged. No raw MIMIC input was used by this job.
- Artifact/source/decoder dependency hashes, input plan lineage and private
  permissions pass; PNGs have 50 distinct content hashes. Pixel contents were
  not opened into chat or manually inspected.
- Aggregate AUROC/AP/confusion statistics and fixed-seed 2,000-replicate
  bootstrap were independently recomputed from the protected score records and
  matched the saved summary. Individual records were not printed.
- Protected output/job-runtime directories use 2770, files 0660, project group
  exposed as 65534 by the NFS server. No public patient identifiers or images.

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_xrv_pilots/ricord_xrv50_12667524/
  displays/case_000.png ... case_049.png
  scores.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Output manifest SHA256:
`dc14a2ba1aed3823f0752a73dda297e874f638b31644d2992227eb3f1cf697a9`.

Frozen evaluation plan manifest SHA256:
`3732ed1592bed8dea7aa479f278bd14ddfb3d8a4304356039ba84080651bd742`.

## Main diagnostic: exact Lung Opacity head

| Metric | Result |
| --- | ---: |
| AUROC | **0.9840** |
| AUROC class-stratified patient-bootstrap 95% interval | 0.9520–1.0000 |
| Average precision (AP; not trapezoidal PR area) | **0.98577534** |
| AP bootstrap 95% interval | 0.95999754–1.0000 |
| Predeclared threshold | 0.5 |
| Sensitivity | **25/25 = 1.0000** |
| Sensitivity Wilson 95% interval | 0.86680775–1.0000 |
| Specificity | **19/25 = 0.7600** |
| Specificity Wilson 95% interval | 0.56570317–0.88503686 |
| Balanced accuracy | **0.8800** |
| TP / FN / TN / FP | 25 / 0 / 19 / 6 |

This supports strong frozen-score discrimination of the selected reference
classes. No missed positive in 25 cases does NOT establish perfect population
sensitivity; the uncertainty interval and selection biases remain important.
The six false positives also show why default-threshold positives alone cannot
be treated as certain clinical facts.

## Separate secondary diagnostic: historical max-score proxy

The old adapter's `lung_opacity` field takes **max(Lung Opacity, Infiltration)**,
not the exact head. Do not mix these definitions or retrospectively pick the
best configuration as a new primary endpoint.

| Secondary legacy proxy metric | Default 0.5 | Pre-existing weak-reference 0.642907085 |
| --- | ---: | ---: |
| AUROC | 0.9792 | 0.9792 |
| AP | 0.98292904 | 0.98292904 |
| Sensitivity | 25/25 = 1.00 | 24/25 = 0.96 |
| Specificity | 9/25 = 0.36 | 23/25 = 0.92 |
| Balanced accuracy | 0.68 | 0.94 |
| TP / FN / TN / FP | 25 / 0 / 9 / 16 | 24 / 1 / 23 / 2 |

The 0.642907085 threshold and its separate secondary role were fixed BEFORE
these predictions; it was not fitted on RICORD. The conversion from DICOM is
new preprocessing, so this is explicitly **unvalidated threshold transport**,
not a new calibration/probability claim. Applying this threshold to the exact
head would also change the tested definition and is not reported.

At the same default 0.5, the historical max proxy produced 16 false positives
versus 6 for the exact head on the same negative cases. This demonstrates that
aggregation and threshold choices matter. It does not prove why a past
synthetic case failed or that all legacy scoring is invalid. The two different
thresholded configurations must not be presented as a controlled head-only
comparison or as independent statistical superiority.

## Scientific limits and next step

Reference is **derived strict unanimity of three comprehensive annotation
groups**, not verified reader identity or reproduced official adjudication.
It tests **lung opacity**, not general pneumonia and not all 14 findings.
The balanced, unanimous, single-image, original-order-selected 50 cases favor
less-disputed examples and do not estimate population prevalence/PPV.
Bootstrap/Wilson intervals cover sampling variation conditional on this chosen
cohort, not selection bias or reference-label uncertainty.

The scorer is frozen operating-point-normalized XRV, not calibrated disease
probabilities. No Brier/ECE is claimed. Checkpoint training overlap remains
unresolved; RICORD absence from declared source names is not proof of zero
overlap. Clinical/anatomical adequacy of the rendered displays was not manually
adjudicated. Engineering transform/codec tests and binding checks passed, but
clinical display verification and paper-primary eligibility remain false.

This advances scorer-reliability evidence, not the TriCompose generator's
clinical accuracy, error-localization accuracy or repair success. A safe next
step is a separately frozen opacity-only sidecar analysis of existing synthetic
candidates, keeping report evidence, EHR unknowns and the exact/legacy score
definitions distinct. Do not relabel pneumonia from this opacity benchmark,
rewrite old scores/choices, run more generators or deploy a new router based
on this result alone. Any new GPU work still needs its full script/approval.

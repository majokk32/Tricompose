# RICORD annotation group diagnostic (no inference)

The acquired raw export contains six groups, not one ready-made binary gold
column. Its classification tags have STUDY scope. Three non-adjudication groups
have near-complete coverage; two have sparse coverage and one has an
adjudication metadata marker with sparse coverage. Sparse coverage does not
establish calibration or official role. Do not rename it as such.

[The dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7993245/) describes
three radiologist reads and majority/adjudication. It does NOT establish how
the six export groups reproduce the final reference algorithm. That algorithm
and participant-to-group identity remain unverified. Pooling all tags creates
apparent conflicts that need not be clinical annotation failures.

## Frozen derived diagnostic

`audit_ricord_reader_groups.py` verifies all acquisition artifact/source hashes,
then re-reads only the fixed public catalog to exclude the two withdrawn
studies. It does not redownload annotations or acquire images.

- Include all active exported studies for counting, without model scores.
- Select the non-adjudication groups with annotation coverage on at least 95%
  of the active studies, including uninterpretable reads. Coverage is not the
  number of diagnoses that can be interpreted; excluding a complete reader
  for abstaining more often would bias this audit. Require exactly three
  such groups; otherwise stop rather than invent reader roles.
- Require null-valued STUDY tags in this schema. Missing and within-group
  conflicting classifications are not negative and do not vote.
- Count strict three-group unanimity separately from disagreement/missingness.
  This is **derived label-group unanimity**, not verified independent reader
  identity or reproduction of official majority/adjudication.
- For a proposed lung-opacity diagnostic only, unanimous typical or
  indeterminate appearance provides a positive reference; unanimous negative
  for pneumonia provides a no-lung-opacities reference according to the
  [catalog schema](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281).
  Atypical, disagreement and missingness remain unknown/excluded. This does
  not validate clinical pneumonia or turn COVID positivity into a finding.
- Check only whether annotation counts could support 25 positive + 25 negative
  studies. Select no cohort, download no image and fit no threshold.
- Unanimity produces selection bias toward clearer/less-disputed examples.
  It can be a small engineering diagnostic, not a representative paper-final
  performance estimate or a proof of synthetic-image clinical correctness.
- Patient grouping, image membership, display transforms, classifier training
  overlap and participant-to-group roles remain unresolved. A study count is
  not an independent patient or image count.

Outputs are aggregate-only `audit.json`, `RESULTS_CN_EN.md` and a binding
manifest under a fresh protected run. Original annotations, acquisition
receipts, consumed code/protocols and old synthetic selections stay unchanged.
Public output has no participant/study identifiers, individual annotation
rows, per-patient hashes, free text or pixels. No new model, GPU task, job
submission, training or human review is performed.

Before any successful audit was saved or scorer run, an initial readiness
attempt failed closed because it used interpretable classification count as
reader availability. All three comprehensive groups actually annotate every
active study. This metadata-coverage error was corrected in this unsealed
auditor; no annotation, model threshold, selected case or prior receipt changed.

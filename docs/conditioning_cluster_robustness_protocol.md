# Exact-conditioning cluster robustness on the frozen development bank

Date: 2026-10-06. Exploratory post-hoc sensitivity, not a new clinical test.

All80 fixed EHRs,240 image slots,960 report slots, scores and winners remain
unchanged. The delivery already distinguishes49 prompts per CXR model from80
case slots. Exact duplicated conditioning can produce dependent image/report
outputs. Do not count960 triples or random seed settings as independent patients.

## Freeze grouping without using outcomes

Read the sealed first-version delivery manifest and allowlisted metadata CSVs:

```text
artifacts/protected/tricompose_v1_2/deliverables/first_version_12714150_001/
manifest SHA256: a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc
candidate_index.csv
baseline_case_means.csv
```

For each case, require the exact3CXR-model ×4report-expert ×seed0 grid and fixed
EHR/fact hashes. Its group signature is the ordered tuple of
`(cxr_model_id,prompt_sha256)` for all three models. Assign opaque group IDs by
first occurrence in sorted opaque case order. Do not inspect prompt text, EHR,
reports or images. Do not use labels, scores, disease, winner, quality or cost
to form groups. All cases stay in the analysis; grouping is not dropping or
regenerating EHRs. Exact prompt identity is not proven equivalence of EHRs or
clinical truth, and different prompts need not be independent.

The candidate-index CSV contains cached outcome columns, but those columns
are unused during grouping. Freeze the worker/tests/protocol, input metadata
hashes and case→group mapping in an atomic plan before analyzing case means.
Investigators have already seen historical results; do not claim prospective
blinding or preregistration of a new final test.

## Preserve existing methods, seeds and budgets

Use all2000 previously audited case/method/cap mean rows. Random acquisition
and final-choice seeds were already averaged within each EHR. Preserve five
methods (`fixed`, `random`, `static_rerank`, `targeted_heuristic`,
`random_acquisition_score_free_final`) and all caps4/8/12/20/30.
`random` means random acquisition followed by scored final selection; only
its contrast with score-free final choice shares acquisition/expenditure.

Report all11 quantities, not only favorable ones: raw BioViL cosine, each
edge's support/known, proxy-opposition/known and comparable/known coverage,
plus simulated calls. Keep missing EHR edges null. Zero comparable facts is
not perfect consistency. Cached XRV/CheXbert relations are uncalibrated proxies,
and BioViL remains a secondary unqualified endpoint.

Five signed paired contrasts at every cap:

```text
static_rerank - fixed
static_rerank - random_acquisition_score_free_final
random - random_acquisition_score_free_final
targeted_heuristic - static_rerank
targeted_heuristic - random_acquisition_score_free_final
```

## Two estimands and group bootstrap

For each quantity, include only same-EHR pairs where both values are available;
report all-attempted EHR count, paired available EHR count and available group
count. Missing values are not zeros. Case-weighted means preserve original
EHR multiplicities. Equal-conditioning means average within each exact-prompt
group, then give each available group equal weight. Report both; neither is
declared the best interpretation after seeing its value. The equal-conditioning
analysis changes the estimand, not the underlying cohort or selected artifacts.

For each contrast/quantity/cap, take1000 bootstrap draws (seed0), resampling
whole available conditioning groups with replacement. Carry all paired cases
inside a selected group together. Calculate both estimands for the same draws;
report percentile95% intervals. Fewer than3 available groups yields null CI.
Constant effects may legitimately yield degenerate intervals, but these are
not evidence of calibrated clinical correctness. Group IDs are sorted for
determinism. Intervals are descriptive, not multiplicity-adjusted hypothesis
tests, and do not account for scorer bias or unseen clinical distributions.

Also report group-size histogram, repeated case slots, exact prompt signature
counts, groups with identical3-image hash vectors, and per-model unique image
and prompt hashes. Hash grouping verifies byte identity only; no pixel or
clinical interpretation is performed.

## Isolation and outputs

Existing actual CPU Slurm only. No model/GPU/API call, new sbatch, training,
downloads, threshold fitting, changed selection, generated artifacts or raw
MIMIC reads. Bound every input by manifest/hash/schema; recheck before commit.
Refuse existing run IDs. Fresh protected plan/result directories use2770/0660
and project-group boundary. Emit case-group metadata, method means, all275
contrast rows, bilingual readable tables, summary and a sealed manifest.
Qualification, selection-change and regeneration flags remain false. Keep
existing first-version archives and prior benchmark runs immutable.

Source follow-up: [ReXVal's official metadata](https://physionet.org/content/rexval-dataset/1.0.0/)
describes explicit error counts but requires credentialing and its own DUA.
The two expected ReXVal CSV filenames were not found in the inspected workspace
artifact/runtime/experiment roots. No login, credential read, restricted data
download or raw report inspection was attempted. This is not a global CARC
storage claim. That access gap does not prevent this bounded cache robustness
analysis, but this analysis does not replace independent scorer qualification.

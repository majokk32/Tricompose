# Lossless candidate scorer reliability sidecar

Status: completed in existing CPU allocation **12632006**, with no new model
calls, Slurm submissions, training, API calls or source-body/pixel reads.
This is reliability/evidence metadata, **not rescoring, a clinical verdict,
fault localization, a new selection policy or an automatic repair controller**.

## Completed output

```text
artifacts/protected/tricompose_v1_2/candidate_reliability_overlays/reliability_pool960_12632006_001/
  candidate_score_table.csv
  fact_reliability.jsonl
  report_dependency_groups.jsonl
  scorer_reliability_profile.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result-manifest SHA256:
`d057efecd06a4db11c6fec27e79f2ecf085b1c7afe0fa7174fce554c8e85b673`.
Protected directories/files retain project-group access and modes 2770/0660.
The original bank, score table, thresholds and selected results remain unchanged.
Every future GPU submission still needs the complete script/resources shown
and explicit approval.

## Inventory and denominator boundaries

The completed historical bank has **80 fixed synthetic EHRs, 240 CXR slots and
960 triple/report slots**: three CXR models and four report experts per EHR,
with the historical seed inventory unchanged. These are not 960 independent
patients. There are 428 distinct report-artifact hashes; duplicated report
artifacts are counted once within each image/finding dependency group.

The original CSV's 50 columns and every cell/row order are preserved, with
18 prefixed `reliability_` columns appended. The sidecar contains 13,440
four-state fact rows (960 × 14) and 3,360 image/finding groups (240 × 14).
Repeated findings and same-image reports are correlated evidence, not
independent votes.

Under the **existing cached direct-EHR-fact rules**, 8/80 fixed EHRs have
explicit positive/negative reference findings; 72/80 lack those findings.
That is a statement about this comparison inventory and cached extraction
scope, not a claim that their EHRs contain no diagnoses or useful information.
No EHR was altered, relabeled, replaced or dropped to improve coverage.

## What is new and what stays unchanged

| Item | Treatment |
|---|---|
| Original raw support, opposition, coverage and BioViL cosine | Retained byte-for-byte as CSV cell values |
| Original row order, primary-prefix fields and historical winner artifacts | Unchanged; no selector invoked |
| Missing comparable evidence | Explicit availability reason, never zero or perfect consistency |
| Unknown / uncertain states | Unchanged; neither becomes a negative finding |
| Same-image report-expert disagreement | Measured from cached CheXbert proposals, deduplicating artifact hashes |
| Same-finding disagreement between image scorers on these synthetic images | Not measured; NA with explicit status |
| Clinical accuracy, confirmed erroneous modality, automatic regeneration | Unavailable / not authorized by this diagnostic |

There are **206/3,360 image/finding groups** where report proposals include
both an explicit positive and an explicit negative. These groups affect
544/960 candidate rows. This is not 206 verified clinical contradictions,
544 faulty reports, or an independent expert majority. Unknown/uncertain
report states do not create positive–negative disagreement.

## Evidence availability, not a new score

The following counts use **candidate-row denominators**. EHR–CXR repeats the
same image evidence for its four report slots; use image/EHR grouping rather
than treating those repeated rows as independent samples.

| Cached evidence status | EHR–CXR | EHR–Report | CXR–Report |
|---|---:|---:|---:|
| No explicit reference facts | 864 | 864 | 0 |
| Explicit reference but no comparable proxy facts | 36 | 64 | 138 |
| At least one proxy opposition, clinically unverified | 24 | 4 | 624 |
| Proxy support with missing evidence | 0 | 0 | 198 |
| Proxy support only, clinically unverified | 36 | 28 | 0 |

The left endpoint is the reference for each historical directed readout:
EHR for EHR–CXR/EHR–Report, XRV proposals for CXR–Report. The annotation checks
the original directed counts; it does not turn an XRV proposal into clinical
truth or claim symmetric complete agreement.

`reliability_*_clinical_score` stays NA because neither candidate clinical
truth nor a qualified clinical score has been established. **The existing raw
score columns are still populated wherever they were populated before.**
Unavailable clinical qualification must not be confused with losing measured
proxy scores. In particular, zero proxy opposition with zero comparable
evidence is not a successful consistency result.

## How the real benchmark is used

The [RSUA XRV diagnostic](rsua_pneumonia_pilot.md) and
[BioViL-T polarity diagnostic](rsua_biovil_followup.md) inform interpretation
only. Their sealed aggregate summaries/manifests are loaded, not patient rows,
images or reference labels. Published pneumonia/normal-cohort classes are
proxy references; segmentation-mask validation is not disease adjudication.

The candidate bank keeps its
`legacy_fourteen_raw_uncalibrated_xrv_chexbert_states` profile. The newer
eight-enabled-head/transported-threshold operating profile is **not** applied
to these historical labels. The old XRV checkpoint match remains explicitly
unverified rather than assumed from an installed model's identity.

The recorded BioViL image/text checkpoint hashes match the benchmark hashes,
but the tasks differ: a whole-report retrieval cosine is not a pneumonia
presence/absence probe, a finding label or EHR fidelity. The sidecar therefore
does not transfer RSUA AUROC, reliability weights, thresholds or per-image
labels to synthetic candidates. RSUA's scorer disagreement is not the
unmeasured scorer disagreement on this synthetic bank.

## Reproducible CPU-only build

Implementation:
`TriCompose-v1.2/benchmarks/build_scorer_reliability_sidecar.py`, with the pure
contract in `TriCompose-v1.2/src/tricompose_v12/scorer_reliability.py`.

The wrapper requires an existing CPU Slurm allocation, pinned manifests,
hash-bound cached sources, exactly 80 × 3 × 4 candidate slots and the old
synthetic-only profile. It recomputes the cached edge counts, checks all
original columns and fails closed on missing lineage, mutated states, mixed
profiles or sources changed during construction. Outputs commit atomically;
an existing run ID is refused. No model factory, scoring worker, selector or
regeneration worker is invoked.

Run from an already authorized CPU allocation; use a **fresh opaque run ID**:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  TriCompose-v1.2/benchmarks/build_scorer_reliability_sidecar.py \
  --candidate-run artifacts/protected/tricompose_v1_2/complete_bank_endpoints/complete_bank_secondary_12624822_001 \
  --xrv-run artifacts/protected/tricompose_v1_2/real_validation/rsua_xrv_pilots/rsua_xrv50_12636566 \
  --biovil-run artifacts/protected/tricompose_v1_2/real_validation/rsua_biovil_pilots/rsua_biovil50_12637081 \
  --output-root artifacts/protected/tricompose_v1_2/candidate_reliability_overlays \
  --run-id reliability_review_new_001
```

The recorded construction/validation phase took 0.580222 seconds; this timing
excludes final serialization/atomic output commit and is not model runtime.

## Verification

Twenty new invented-fixture tests passed. The full V1.2 regression suite
passed **1,047 tests in 10.527 seconds**. These tests verify contracts and
missing-data handling; they are not clinical validation.

A separate read-only CPU check verified all 48,000 original CSV cell values,
960 rows/order, 13,440 fact relations and directed count readouts, report-hash
deduplication, explicit positive/negative disagreement and unavailable clinical
fields. All six output artifact hashes, 13 source hashes and protected
permissions passed. Original states, thresholds, weights and selections were
not changed.

## Next boundary

This table can support an **evidence-aware decision preview**: distinguish
missing evidence, scorer limitations and correlated report disagreement before
deciding what additional evidence is needed. It does not yet qualify a score
for clinical fault localization or justify automatic modality regeneration.
Any next policy or inference experiment requires its own frozen protocol;
the present overlay must not silently change historical selection behavior.

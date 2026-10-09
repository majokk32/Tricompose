# RadEvalX annotation encoding: exploratory sensitivity, not a new gold standard

Date: 2026-10-06. Frozen before running this sensitivity analysis.

The [official release](https://physionet.org/content/rad-eval-x/1.0.0/)
describes eight consensus error-count categories but does not explicitly state
what a blank cell means. A bounded search of the release, original paper and
author-linked source information did not establish an official decoding rule.
Do not equate failure to find documentation with proof that no code exists.
The later `jbdel/RadEval` framework is a different work, not the original
RadEvalX authors' annotation-decoding implementation.

The completed conservative run remains immutable:
`artifacts/protected/tricompose_v1_2/report_metric_alignment_runs/radevalx_published_v2_12714150_001/`.
Manifest SHA256:
`1cd67427ef20e817103697b1883f208cded273d22725f2b54bfaad06cce4e4f5`.
Decode only its hash-bound derived numeric predictions and opaque references;
do not parse the source CSV, report bodies, source identifiers or candidate
EHR/images/reports. Original source bytes may be rehashed to verify the inherited
provenance pins, without accessing CSV columns. All100 annotated pairs remain included. Unannotated score
rows are not zero-error cases. No new model calls, download or Slurm submission.

## Two interpretations, explicitly separated

1. `blank_is_unresolved`: the existing interpretation. A severity total needs
   all eight category counts; unavailable totals stay null.
2. `hypothetical_blank_is_zero`: in this exploratory calculation only, replace
   null count cells with zero in memory. Preserve observed counts and every
   published score, row, group and orientation. This is an **assumption**, not
   an officially verified convention or newly adjudicated reference. Do not
   write a replacement reference file or change the original null run.

For each of the six published metrics, report full-cohort Spearman and Kendall
tau-b against negative significant and all-error totals. Orient quality upward,
including the frozen lower-is-better RadCliQ orientation. Significant-total
intervals use the existing source-group bootstrap: 1,000 draws, seed0, 95%
percentiles, same usability rules. No interval interpretation removes the
encoding uncertainty or selected-cohort limitation.

## Independent, previously published reproduction targets

Original [paper, Table4 and Section7](https://arxiv.org/html/2311.16764v1#S7):

| Cohort | Published metric | Quality vs negative total errors | Quality vs negative significant errors |
| --- | --- | ---: | ---: |
| All100 | RadGraph F1 | 0.2844 | 0.1633 |
| All100 | RadCliQ | 0.3349 | 0.1929 |
| Total errors >3, reported30 | RadGraph F1 | 0.1910 | 0.0091 |
| Total errors >3, reported30 | RadCliQ | 0.3380 | 0.0169 |

Compare these **signed** values at the paper's rounding precision (absolute
deviation at most0.00005). Do not flip directions, try alternative encodings,
fit thresholds, or choose a better-matching subset after seeing results. BLEU
is not a reproduction target because Table4 does not identify which released
BLEU column it means. The >3 subset is explicitly outcome-defined exploratory
reproduction, not an independently selected or primary test cohort.

A reproduction match is evidence of numerical compatibility with the authors'
published calculation. It is **not proof of blank-cell semantics**. A mismatch
does not prove the release, metrics or source annotations are wrong. Retain all
matches and mismatches, including the noisy-subset size.

## Execution and acceptance

Freeze worker, invented-fixture tests, this protocol and original run/source
pins in a fresh protected plan before computation. Require the existing actual
CPU Slurm allocation, explicit `--allow-exploratory-zero-assumption`, protected
atomic non-overwriting outputs, sanitized stdout and original-bank hash check.
Save aggregate JSON, one comparison CSV, a CN/EN explanation and manifest.
All `clinical_qualified`, `local_implementation_qualified`, `encoding_verified`,
`selection_changed`, `primary_metric_eligible` and `regeneration_authorized`
flags stay false. No current scoring, winners, gates or first-version package
changes. These published reference-based scores do not validate reference-free
synthetic selection, local checkpoints, EHR grounding or image factuality.

# Independent report-metric benchmark: result and limits

Date: 2026-10-06. Status: numeric execution and independent audit passed;
**primary expert-error alignment remains unavailable, not clinically validated**.

## What this experiment actually tests

The open [RadEvalX 1.0.0 release](https://physionet.org/content/rad-eval-x/1.0.0/)
contains 100 IU-Xray reference/generated-report pairs with consensus error
annotations from two radiologists and released metric scores. The annotators
compared reports without images. The cohort has abnormality/RadCliQ-related
selection, so it is not a representative TriCompose evaluation cohort.

This experiment uses **published numeric scores**, not freshly computed local
CheXbert, RadGraph or other model outputs. Its intended primary test is whether
higher metric quality aligns with fewer clinically significant expert errors.
It cannot establish image-report consistency, EHR correctness, within-case
best-of-N selection, or the reliability of our local scorer checkpoints.

All 100 annotated pairs were included by annotation-key membership, not by
score, diagnosis or error count. The metric release has 590 rows; the other
490 have no expert annotation here and were not treated as zero-error cases.

## Actual result / 当前结论

Six published score columns are complete for all 100 pairs. The reference
error-count tables, however, contain sparse entries:

| Reference availability | Count |
| --- | ---: |
| Attempted annotated pairs | 100 |
| Error-count cells: 100 pairs × 8 categories × 2 significance levels | 1,600 |
| Explicitly filled numeric count cells | 177 |
| Blank/unresolved count cells | 1,423 |
| Pairs with a complete significant-error total | 0 |
| Pairs with a complete insignificant-error total | 0 |

**The meaning of blank count cells has not been verified from an official
encoding convention.** They may represent unrecorded zero counts, but this
cannot be assumed from sparseness alone. The frozen protocol retains them as
null; it requires all eight category counts to compute a total. This is an
interpretation limitation, not evidence that the dataset or metrics are bad.

| Released score | Available scores | Paired significant-error totals | Primary Spearman | Kendall tau-b | 95% interval |
| --- | ---: | ---: | --- | --- | --- |
| BLEU-4 | 100/100 | 0/100 | NA | NA | NA |
| BLEU-2 | 100/100 | 0/100 | NA | NA | NA |
| BERTScore | 100/100 | 0/100 | NA | NA | NA |
| CheXbert column | 100/100 | 0/100 | NA | NA | NA |
| RadGraph F1 column | 100/100 | 0/100 | NA | NA | NA |
| RadCliQ | 100/100 | 0/100 | NA | NA | NA |

NA is not zero correlation, scorer failure, or perfect agreement. Correlation
cannot be computed for the primary outcome under this reference interpretation.
The configured 1,000 bootstrap draws were **not executed** for that outcome:
there were zero eligible source groups and zero usable draws.

Per-category statistics on explicitly filled cells are preserved in
`summary.json`. Their denominators vary, and absent cells are excluded rather
than imputed. They describe only nonmissing-count subsets; they must not be
used to rank the six metrics over the complete cohort or qualify an evaluator.

中文结论：接口跑通了，100 对的六列发布分数都齐全；但错误计数空白的含义还未
确认，当前不能给出有效的总错误相关性，更不能宣称某个评分器可靠或不可靠。
先解决标注编码，再谈验证通过和接入选优。

## Execution, privacy and preserved state

- Executed in the existing CPU Slurm allocation `12714150`, node `b05-06`.
  No new Slurm submission, GPU call, local scorer inference, training or
  external clinical API call. Only the small public release was downloaded;
  no model weights were downloaded.
- Source files match the official release SHA256 manifest. This verifies
  release-byte consistency, not an independent authenticity attestation.
- CSV bytes were read internally. Source-key fields were used only for an
  internal exact join; report-text fields and secondary identifiers were not
  accessed, inspected, displayed, used for scoring or exported. Derived records
  use opaque pair/group IDs. No MIMIC patient input or candidate body was opened.
- Raw source CSVs and derived artifacts remain under `artifacts/protected/`,
  with project-group access, directories `2770`, files `0660`, and Git exclusion.
  This document contains only protocol metadata and aggregate counts.
- Original bank manifest hash remains
  `ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`.
  Fixed EHRs, images, reports, scores, selected triples and regeneration gates
  were not changed. All qualification/promotion/repair flags remain false.
- Worker numeric phase: `0.027529` seconds; CLI execution approximately
  `0.084` seconds. This is cached numeric analysis, **not model inference time**.

## Repairs and reproducibility

Acquisition V1 rejected the checksum document's single-space separator before
CSV acquisition. V2 accepts the release's whitespace syntax while retaining
exact filenames, hash validation and path/duplicate guards. V1 is preserved.

Benchmark V1 incorrectly bounded the entire score file at 100 rows and failed
before reference-error decoding. A header/key-only inventory audit established
590 score rows and 100 annotated pairs. The V2 cohort clarification was frozen
before numeric evaluation; all 100 annotated pairs were retained. No error
counts, metric values, thresholds or clinical references were tuned to repair it.

There are 57 new tests for the statistical contract, acquisition, adapters and
worker guards. The full V1.2 suite passed **2,254 tests in 14.059 seconds**.
An independent derived-only audit recomputed all **114 metric/outcome entries**
(six metrics × 19 outcomes), checked 23 unique source pins and 17 protected
filesystem entries, and verified the unchanged original bank. This checks
implementation consistency; it does not create missing clinical references.

Audit script:
`.tmp/audit_radevalx_published_12714150_001.py`

Audit SHA256:
`f9e4b4f9669306d128206b713b37b2e2211a1990663032205c43529127d84808`

Paths below are relative to `/project2/ruishanl_1185/inference_3mod`:

```text
Source:
artifacts/protected/tricompose_v1_2/report_metric_sources/radevalx_source_12714150_002/
Source manifest SHA256:
7073624ae36b71b1581dc5e288902659513761b4051f0cd5b571821aaf6467ee

Frozen V2 plan:
artifacts/protected/tricompose_v1_2/report_metric_alignment_plans/radevalx_published_v2_12714150_001/
Plan manifest SHA256:
122ea634333d4afe859e0175eb5d942be2fe38f42a7bcc41f8bf304be12da239

Actual V2 result:
artifacts/protected/tricompose_v1_2/report_metric_alignment_runs/radevalx_published_v2_12714150_001/
Run manifest SHA256:
1cd67427ef20e817103697b1883f208cded273d22725f2b54bfaad06cce4e4f5
```

Result files: `summary.json`, `RESULTS_CN_EN.md`, opaque
`published_predictions.json`, separate nullable `references.json`,
`prediction_freeze_receipt.json`, and `manifest.json`.
The receipt discloses annotation-key access before prediction freezing and
error-count decoding afterward. Published scores and data were available to
the investigator; no clinical blinding is claimed.

## Next decision

1. Verify blank-count semantics from the authors' documentation or evaluation
   code. If an official blank-to-zero convention is established, freeze that
   evidence and a new protocol/run without editing this run or source data.
   Otherwise use an expert benchmark with an explicitly specified encoding.
2. Only then assess expert-error alignment and uncertainty. Keep reference-based
   report quality separate from image-grounded and EHR-grounded consistency.
3. RadGraph is a potential richer frozen report extractor, not a qualified
   replacement yet. The [official weight listing](https://huggingface.co/StanfordAIMI/RRG_scorers/tree/main)
   lists RadGraph-XL at about 416 MB. No download/deployment was performed;
   large-weight acquisition needs approval, and inference needs a complete
   reviewed Slurm request. Its reference/hypothesis F1 alone is not a
   reference-free CXR-report score.

Do not enable autonomous regeneration or replace the existing score table
on the strength of this diagnostic.

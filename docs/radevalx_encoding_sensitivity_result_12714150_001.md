# Report benchmark: encoding sensitivity completed

Date: 2026-10-06. Status: numeric calculation and independent audit passed;
**annotation encoding is not officially verified, and no scorer is promoted.**

## What changed

Added a separate exploratory check over the previous run's immutable opaque
numeric contracts. The existing conservative run still retains blank counts
as null. Neither source annotations nor current synthetic-bank scores, winners,
repair gates or first-version packages were changed. No source CSV rows or
report bodies were decoded in this new check; inherited source-byte hashes
were verified for provenance. No new model inference, training, weight download,
external clinical API call or Slurm submission occurred.

The [official RadEvalX release](https://physionet.org/content/rad-eval-x/1.0.0/)
describes consensus counts but does not explicitly explain blank-cell encoding.
A bounded search did not establish an official decoding convention. This does
not establish that no additional author code or documentation exists. The
later `jbdel/RadEval` framework is not the original RadEvalX author implementation.

## Conditional numeric results, not qualified scores

All100 annotated pairs were retained. Of1,600 count cells,177 were observed and
1,423 unresolved. The table below **assumes each unresolved count is zero**.
Intervals also condition on that assumption. They do not include uncertainty
about what the original blank cells mean.

| Published column | Spearman quality vs negative significant errors | Conditional 95% interval | Spearman quality vs negative all errors |
| --- | ---: | --- | ---: |
| BERTScore | 0.1953 | [-0.0259,0.3770] | 0.3491 |
| BLEU-2 | 0.1617 | [-0.0611,0.3729] | 0.2916 |
| BLEU-4 | 0.1381 | [-0.0718,0.3507] | 0.1744 |
| CheXbert column | 0.4132 | [0.2350,0.5805] | 0.4915 |
| RadCliQ | 0.1882 | [-0.0234,0.3864] | 0.3347 |
| RadGraph F1 column | 0.1591 | [-0.0510,0.3571] | 0.2848 |

These are correlations, not accuracy, probability, our960-candidate scores,
or evaluation of local checkpoints. The released CheXbert column is not
automatically our CheXbert14-label F1. Higher observed correlation here does
not establish a statistically superior or transferable selection criterion.
Readers evaluated report pairs without images; the source cohort also had
abnormality/RadCliQ-related selection. Neither CXR truth nor EHR grounding is
established by these reference-based metrics.

Under the conservative interpretation, all100 significant/all-error totals
remain unavailable and correlations stay NA. NA is not zero correlation or
evidence of scorer failure.

## Published result reproduction: retain the discrepancies

Compared the unambiguous RadGraph F1 and RadCliQ entries in the original
[paper, Table4/Section7](https://arxiv.org/html/2311.16764v1#S7), with directions
and rounded target values fixed before calculation. BLEU was not guessed from
an unspecified table heading. No weight, direction or subset was tuned.

- The hypothetical total-error >3 subset contains30 pairs, matching the
  reported subset size. This is outcome-defined reproduction, not a new
  independently selected test cohort.
- **0/8 correlations reproduce at the paper's rounding precision.** Full100
  RadCliQ total correlation is0.3347 vs published0.3349; significant correlation
  is0.1882 vs0.1929. Full100 RadGraph F1 total is0.2848 vs0.2844; significant
  is0.1591 vs0.1633. All four subset comparisons are also retained, including
  sign differences near zero, in `paper_reproduction.csv`.
- The values are numerically close in several comparisons, but an approximate
  match and subset-size agreement do not prove blank-cell semantics. The
  source of the discrepancies is **not determined**. Do not label this strict
  reproduction or use it to change current clinical references.

## Checks and exact outputs

Executed in existing CPU Slurm12714150. Numeric phase0.535526seconds is cached
statistical analysis, **not model inference speed**. Eight invented-fixture
tests passed; full V1.2 suite2,284tests passed in14.735seconds. Independent
stdlib code, without importing the worker/adapter/statistical implementation,
verified36outcomes,8paper comparisons, six bootstrap intervals,33source pins,
and protected permissions/Git exclusion. Original delivery archive hashes
remain unchanged. All qualification/selection/regeneration flags remain false.

Protected result:

```text
artifacts/protected/tricompose_v1_2/report_metric_encoding_runs/radevalx_encoding_12714150_001/
  summary.json
  paper_reproduction.csv
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`8f7f444b92953b0d8c992219ce14d47a66c197bb63adce1f0372cb5a39ce0b99`.
Frozen plan manifest:
`4838540ec8a450a507f5bb30ce9f58a932ffb2716407a5a0f56b94188e5fb561`.
Worker SHA256:
`c01d9b67bd0535df330433752cf34359d2e4fe2f65cff46bdc0b9fa6cd2dc449`.
Independent audit SHA256:
`cbf0f13601335f03250c212e9ca28d7e56b3e230bf1d3e59377538ca8294b641`.

Source/fixture/audit paths:

```text
TriCompose-v1.2/tools/check_radevalx_encoding_sensitivity.py
TriCompose-v1.2/tests/test_radevalx_encoding_sensitivity.py
TriCompose-v1.2/audits/audit_radevalx_encoding_sensitivity.py
docs/radevalx_encoding_sensitivity_protocol.md
```

## Next scientific gate

This check supplies actual conditional numbers instead of silently imputing
zero or endlessly producing NA-only dashboards. It does **not** close the
independent-verifier gate. Keep the first-version engineering demo separate
from claims of clinically reliable error localization and targeted repair.

Before changing selection, establish the annotation convention from an
authoritative implementation/document, or use another expert benchmark with
an explicit complete encoding. Qualification must evaluate the **actual local
frozen implementation**, not only a same-named published column, and separate
reference-report agreement from image-report/EHR consistency. RadGraph may
provide richer frozen entity extraction, but its absent weights require explicit
download approval and inference requires a complete reviewed Slurm request.

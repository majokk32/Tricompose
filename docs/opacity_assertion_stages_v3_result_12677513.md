# Opacity stages V3: format recovered, language validity still limited

## 完成与输出 / Completion and output

Approved job **12677513** completed on debug A40 `b11-09`, exit `0:0`, Slurm
elapsed **2 minutes 21 seconds**. It waited 24 seconds before starting. Worker
elapsed before serialization **104.847063 seconds**; peak allocated CUDA tensor
memory including model load **15.571 GiB**, actual dtype BF16. No patient or
synthetic candidate-bank EHR/report/image was read. No generation, training,
model download/API, selector, old-score replacement or repair.

```text
artifacts/protected/tricompose_v1_2/opacity_assertion_stages_runs/authored48_stages_v3_12677513/
  predictions.json
  raw_responses.json
  replay_checks.json
  prediction_freeze_receipt.json
  scored_checks.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result manifest SHA256:
`73dd5b1a26a0043caac1e4822ff901a496167947d873b12201e6d7c19e4bacce`.
Source/response bodies remain private; only aggregate readouts are reproduced.

## Paired results on all 48 authored texts

These are **same-investigator development fixtures**, not independent clinical
gold or an untouched test. V3 corrected the locator's contradictory output
instructions prospectively; it did not rescue or overwrite V2 failures. The
unchanged V1 baseline's 48 raw responses were byte-identical to job 12677127.
Input/reference files, polarity prompt, state policy, model/loader and all
consumed previous files/runs remain unchanged.

| Metric, fixed all-attempted denominator | V1 quote baseline | V3 two stages |
| --- | ---: | ---: |
| Interface completion | 43/48 | 48/48 |
| Unavailable predictions | 5/48 | 0/48 |
| Authored four-state matches | 23/48 = 47.92% | 38/48 = 79.17% |
| Macro F1 | 0.50036 | 0.79808 |
| Positive/negative flips | 0 | 0 |
| Determinate output on uncertain/unknown expectation | 19/30 | 9/30 |
| Six known controls matched | 3/6 | 5/6 |
| Primary model calls | 48 | 86 |

Authored-state match increased **15/48 = 31.25 percentage points**, including
availability recovery and linguistic differences. This is not an image/report
generation-quality improvement. The two-stage path uses more primary calls
(**48 locators + 38 polarities = 86**) than the 48-call baseline; no compute
savings are claimed. Failure-aware F1 retains unavailable references in class
support, rather than scoring only completed or favorable examples.

Actual total **138 attempts / 138 responses**: 48 baseline, 48 primary locators,
38 primary polarities and four calls for the two staged replays. Ten completed
empty primary locator selections skipped polarity; they remain unknown proposals,
not negative findings. Both fixed replays matched complete state, selected IDs,
assertions and exact stage-response hashes. No retries or replacement.

## State breakdown and remaining errors

| Authored expectation | Reference checks | V3 exact matches | Other V3 states |
| --- | ---: | ---: | --- |
| positive | 9 | 9 | none |
| negative | 9 | 9 | none |
| uncertain | 16 | 9 | 6 negative, 1 positive |
| unknown | 14 | 11 | 1 negative, 1 positive, 1 uncertain |

All 18 simple determinate references match, but seven uncertain references are
promoted to certainty and two unknown references become determinate. The ninth
unknown mismatch becomes uncertain. **Ten errors remain**, six despite exact
authored locator IDs: correct source addressing did not guarantee correct
polarity interpretation. Relevant source-ID sets match **42/48**, not 48/48.

The predeclared narrow language gate therefore **failed** (38/48 states and
42/48 authored ID sets, despite stable replays). Do not lower the gate after
seeing this result, call the 48/48 interface rate clinical accuracy, or deploy
the outputs as automatic fault/repair verdicts.

## All families retained

| Authored family, four checks each | V1 matches | V3 matches |
| --- | ---: | ---: |
| Explicit presence | 4 | 4 |
| Explicit absence | 4 | 4 |
| Possible finding | 0 | 4 |
| Unmentioned finding | 1 | 4 |
| Qualified absence | 0 | 0 |
| Opposing assertions | 4 | 3 |
| Negation of another finding | 4 | 4 |
| Other disease only | 2 | 2 |
| Generic summary | 0 | 3 |
| Change-only/unchanged language | 2 | 2 |
| Multisegment context | 2 | 4 |
| Repeated assertion | 0 | 4 |

Qualified absence remains **0/4** under both methods. Opposing-assertion match
declined from 4/4 to 3/4. These results cannot be omitted in favor of improved
possible/unknown/repeated-statement families. The authored conventions (such
as qualified absence -> uncertain) are the disclosed task policy, not universally
adjudicated radiological labels or image truth.

## Independent audit and preservation

Independent standard-library audit ran inside existing CPU Slurm **12666569**.
It reparsed **all 138** responses; checked all source IDs, baseline verbatim
quotes, availability/null handling, state reductions, 96 paired check records,
full class/family/control confusion and F1, call/skip caps and both replays.
Verified **30 source pins**, seven result artifacts, closed prediction receipt,
same input/reference bytes as V2, unchanged baseline/polarity/runtime bindings,
source/result SHA256 and project-group modes **2770/0660**. No model or patient
input was used by the audit. Script:
`.tmp/audit_authored_opacity_stages48_v3_12677513.py`.

Post-run full regression: **2,015 tests passed in 11.654 seconds**.
`git diff --check` passed and protected results are Git-ignored. The complete
bank manifest remains `ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`;
the dependency registry remains `87674da73de24b840f4baf08a1cd91fcd98e6d5f0d687026ab47ed3ec6b0600d`.
The original V2 result/plan hashes are unchanged.

## 当前可用范围 / What this permits

V3 is a functioning **secondary assertion-evidence interface**, with observed
limitations, not a qualified clinical scorer. Its improvement does not update
old candidate scores/winners, establish CXR-report correctness, create EHR
opacity facts, confirm a faulty modality or authorize regeneration. All fixed
EHR anchors remain untouched; missing EHR evidence is still not a negative.

Next development should prioritize **reliability/abstention and independent
assertion-policy evaluation**, rather than silently reinterpreting unavailable
outputs or repeatedly tuning on these 48 known answers. Until that is qualified,
uncertain/qualified/change-only and unsupported disease-inference evidence must
not become hard repair triggers. Any new gate/threshold/clinical mapping must
be versioned and disclosed prospectively, retain coverage and false-trigger
denominators, and never claim clinical accuracy from this authored benchmark.
Further model execution or candidate checking needs a new explicit scope/plan
and full script/resource approval.

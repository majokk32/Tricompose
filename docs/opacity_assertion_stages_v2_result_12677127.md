# Authored opacity stages V2: completed; locator format contract failed

Approved job **12677127** completed on debug A40 `b11-09`, exit `0:0`, Slurm
elapsed **2 minutes**. Worker elapsed before serialization **77.348584 seconds**,
peak allocated CUDA tensor memory **15.569 GiB**, actual dtype BF16. Actual
**99** attempts: 48 unchanged V1 baseline, 48 primary locators, one primary
polarity, and two replay locators. All 99 raw responses are retained. No retries,
source replacement, new generation/training, score selection or repair.

Protected result:
`artifacts/protected/tricompose_v1_2/opacity_assertion_stages_runs/authored48_stages_v2_12677127/`.
Manifest SHA256:
`25cb254abfcc5871c57c678922381cf96f1a8940c16db1c267756ec604580891`.

## Paired authored-language outcomes

| Same 48 authored inputs | V1 quote baseline | V2 staged interface |
| --- | ---: | ---: |
| Complete responses | 43/48 | 1/48 |
| Unavailable | 5/48 | 47/48 |
| Authored state matches, all attempted denominator | 23/48 | 1/48 |
| Macro F1, disclosed four-state policy | 0.50036 | 0.05000 |
| Determinate output on uncertain/unknown expectation | 19 | 0 |
| Six known controls matched | 3/6 | 0/6 |

The zero staged determinate-error count comes from **47 unavailable outputs**,
not safe clinical decisions. Both predeclared staged replays are unavailable;
their raw locator response hashes repeat, but stable failed responses are not
successful state replays. The narrow language gate failed. All results concern
investigator-written development language, not clinical report/image accuracy.

Five baseline failures are ambiguously repeated quotes. All 47 staged failures
are `object_response_required` at the locator phase, not polarity-classification
failures. The V2 diagnostic cannot meaningfully assess its unexecuted polarity
stage or clinical error localization.

## Concrete implementation issue, not a repaired score

The frozen locator prompt simultaneously says **return a bare empty array**
when nothing is mentioned and **return an object with segment_ids**. The
parser accepts only the object. This is a contradictory output instruction
introduced in the new interface, not proof of a bad CXR/EHR/report generator.

Structural inspection of all 50 locator responses (including two replays):
**49** are bare integer arrays, **one** is the specified object. Bare arrays
contain 14 empty and 35 nonempty lists. Of the 47 primary bare arrays, **43**
coincidentally equal their authored expected segment-ID lists. This is a
post-hoc **format diagnosis only**: do not rescue these outputs, treat them as
47 successful predictions, infer unrun polarity states or recompute a favorable
V2 score. Keep the original unavailable denominators and failed run immutable.
The prompt inconsistency is a concrete defect; no controlled experiment yet
establishes it as the sole cause of the model's output-format behavior.

## Independent audit and preservation

Independent CPU audit inside Slurm **12666569** reparsed all **99** raw
responses, reconstructed phase attempts, source-ID validation, baseline exact
quotes, reductions, 96 paired check records, full four-state confusion/F1,
all 12 families, six known controls, frozen predictions and both replays.
Verified **28** source pins, seven result artifacts, SHA256 and group-only
permissions **2770/0660**. No patient or candidate report/image was opened.
Audit script: `.tmp/audit_authored_opacity_stages48_12677127.py`.
Post-run full V1.2 suite: **2,003 tests passed in 11.794 seconds**.

## Next version scope

Create a separately versioned **format-only** prompt correction: consistently
require an object for both empty and nonempty selections, explicitly disallow
bare arrays. Preserve original strict decoder, segmenter, reduction policy,
polarity prompt, model/loader, all 48 authored texts and original V2 outputs.
Disclose that the correction follows observed V2 failures, not a held-out
improvement or blinded clinical qualification. A fresh sealed plan/job/result
is required; do not patch the consumed V2 code or silently re-score its failures.
New GPU execution requires full script/resource display and explicit approval.

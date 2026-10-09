# Cached image verification availability: completed

The [metadata-only protocol](image_verification_availability_protocol.md) was
executed in existing CPU Slurm allocation **12645021**. No new model/GPU/API
call, submission, source-body access, training or selection occurred.

## What is now connected

The full pool remains **80 fixed EHRs / 240 CXR slots / 960 triples**, with
13,440 finding rows and 2,072 original logical requests. The completed six-image
Qwen run is attached by exact CXR identity/hash; retained report relations also
require the exact triple/finding/report identity and hash. Text hashes alone
never extend report coverage. Every original score and history cell is retained.

| Availability | Count |
|---|---:|
| Checked image slots / all image slots | 6 / 240 |
| Candidate slots with image evidence / all triples | 24 / 960 |
| Candidate slots not checked | 936 |
| Unique supported image/finding checks | 48 |
| XRV/Qwen explicit agreements | 25 |
| XRV/Qwen explicit oppositions | 21 |
| Uncertain comparisons | 2 |
| Unique image/finding rows outside eight-head scope | 36 |
| Retained report occurrences / distinct image-finding pairs | 17 / 10 |

The 48 checks are not 192 independent observations after four reports per
image. A scorer opposition is not evidence of which modality is faulty.

## Existing request handoff

| Kind | Existing logical requests | With image-side metadata | Breakdown |
|---|---:|---:|---|
| Image finding | 220 | 4 | 1 agreement, 2 oppositions, 1 outside scope |
| Image/report relation | 1,172 | 20 | 10 agreements, 7 assertions not retained, 3 outside scope |
| Report assertion | 606 | 0 | Existing reportgate fields preserved; image readout is not a text-scope check |
| EHR observability | 72 | 0 | Not applicable to image-only evidence |
| EHR conditioning/fact scope | 2 | 0 | Not applicable to image-only evidence |

Thus **24 requests have image-side metadata**, not 24 resolved requests or
24 model calls. Among image-applicable requests, 216 image-finding and 1,152
image/report requests remain unchecked. All execution histories stay unchanged
and **zero requests are clinically resolved**. Unknown/uncertain/unavailable,
unsupported heads and withheld report assertions remain separate.

Same-Qwen image/text agreement is not independent clinical truth. No original
clinical score, winner, EHR anchor, threshold or regeneration authorization
changes; this is a previously inspected developmental diagnostic, not a held-out
accuracy or validated self-correction result.

## Outputs and checks

Protected run relative to `artifacts/protected/tricompose_v1_2/`:
`image_verification_availability/imagecheck_pool960_12645021_001/`.

Start with `candidate_score_table.csv`: original scores plus appended
`imageverify_` fields. Unchecked decision counts are null, not zero errors.
Detailed artifacts: `fact_verification_availability.jsonl`,
`evidence_request_availability.jsonl`, `unique_image_finding_table.jsonl`,
`summary.json`, `RESULTS_CN_EN.md` and `manifest.json`.

Manifest SHA256:
`d933d9ba37181e2eed31f3f92fd8831d514a313fc225c6a6bb6f4260ad08f1bf`.

Independent standard-library checks passed: **23 consumed source hashes / six
artifact hashes**, all 76,800 original annotated candidate cells (including
65,280 original score-table cells), all 13,440 fact rows and 2,072 request rows,
exact image/report dependencies and relations, 84 unique image/finding rows,
request coverage denominators, unchanged parent manifests and group-only modes.
Full suite: **1,306 passing tests**, including 27 new invented metadata tests.

This provides the joined operational table for later controlled error tests.
It does not authorize immediate regeneration on an unqualified scorer vote.

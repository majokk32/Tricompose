# Conservative full-bank evidence-request preview

Status: completed in existing CPU Slurm allocation **12632006**. No new model
call, Slurm submission, source-body/pixel read, API, training, ranking, EHR
change, clinical acceptance, rejection or regeneration occurred.

This extends the [lossless scorer reliability sidecar](candidate_scorer_reliability_sidecar.md)
with explicit requests for missing/conflicting evidence. It is **not a
validated error-localization or repair method**. The older
[two-EHR/eight-finding decision preview](decision_preview_protocol.md), its
scoped finding definitions and all historical selections remain unchanged.

## Completed files

```text
artifacts/protected/tricompose_v1_2/reliability_evidence_previews/evidence_pool960_12632006_001/
  decisions.jsonl
  decision_preview.csv
  evidence_requests.jsonl
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result-manifest SHA256:
`b38227d1e42bd36e90718558f837a5f442d4cf84bde2795f43b13050db629a9f`.
Directories/files are project-private, modes 2770/0660. The pinned source
sidecar manifest remains
`d057efecd06a4db11c6fec27e79f2ecf085b1c7afe0fa7174fce554c8e85b673`.

`decision_preview.csv` is the readable candidate table: action, availability,
reason codes, request IDs and proxy-pattern counts. `decisions.jsonl` also
retains the fixed EHR/CXR/report hashes and triggering evidence IDs.
`evidence_requests.jsonl` deduplicates dependencies and lists every consuming
candidate. Original score-table cells, rankings and winner artifacts are not
written or changed by this program.

## Frozen interpretation

The input is the historical fourteen-finding uncalibrated profile, not the
newer eight-enabled-head calibrated/scoped profile. Only authenticated cached
states, provenance categories and aggregate reliability metadata are read.
The absence of a legacy report syntax guard or same-finding second image
scorer is retained as missing evidence, not silently repaired by this preview.

| Explicit cached pattern | Evidence request, not fault confirmation |
|---|---|
| EHR = image proxy, report proxy differs | Verify the report assertion and image/report relation |
| EHR = report proxy, image proxy differs | Verify the image finding and image/report relation; the report is image-conditioned, not independent truth |
| Both downstream proxies oppose EHR | Verify image/report proposals and conditioning/EHR-fact scope; keep the EHR fixed |
| Explicit EHR fact but image/report state unknown or uncertain | Verify the missing image/report evidence; do not call it a negative or require an invented report statement |
| Same-image report experts give explicit opposite proposals | Verify the image finding and explicit report assertions, without majority voting |
| All three proxies agree | Retain unresolved agreement; no clinical acceptance or automatic stopping-success claim |
| EHR has no explicit comparison findings | Assess radiographic observability/extraction scope once per fixed EHR; do not add disease/device facts or replace the EHR |

Missing comparisons, disagreements and report omissions are requests to
**check evidence**, not prescriptions to alter an artifact. A report need not
restate every diagnosis, medication or lab, and a clinical diagnosis need not
require a visible radiographic finding at this time. The source categories
alone do not resolve timing, observability or clinical truth.

The preview does not read BioViL cosine to choose requests. RSUA benchmark
metrics never become synthetic labels, probabilities, thresholds or weights.
No controlled-error answer key or original winner/rank is used for decisions.

## Measured coverage: narrow, not generation failure

All **80 fixed EHRs, 240 CXR slots and 960 candidate triples** are retained.
Under the cached direct-fact definition, 8 EHRs have explicit comparison facts
and 72 do not. The historical 15/65 prompt-conditioning split is a different
definition and must not be substituted for this 8/72 direct-fact split.

There are 13,440 candidate/finding rows (960 × 14). Only 96 carry explicit
EHR reference facts, and **15/96** have explicit states for all three modalities:

| Three-way comparable proxy pattern | Candidate/finding occurrences |
|---|---:|
| All three agree, unverified | 9 |
| Image differs from EHR/report proxies | 2 |
| Report differs from EHR/image proxies | 1 |
| Both downstream proxies oppose EHR | 3 |
| Total three-way comparable | 15 |

These are repeated **candidate/finding occurrences**, not 15 patients or 15
independent annotations. The other 13,425 rows are not three-way comparable;
that denominator includes 13,344 rows with no explicit EHR reference and 81
explicit-reference rows lacking a comparable downstream state. Unknown and
uncertain are not negatives. High missingness does not mean 13,425 failures.

The candidate actions are 959 `request_verification_preview` and one
`retain_unresolved_preview`. **959 requests do not mean 959 faulty triples.**
The rules separately report evidence gaps and proxy opposition. No case is
actually accepted or rejected, including the candidate with no triggered
verification request. No triggered request is not proof of complete coverage.

## Dependency deduplication and cost boundary

| Logical request kind | Requests | Dependency within the fixed case |
|---|---:|---|
| Assess EHR radiographic observability | 72 | EHR + facts hashes |
| Verify conditioning/EHR-fact scope | 2 | EHR + facts + image hashes, finding |
| Verify image finding | 220 | Image hash, finding |
| Verify report assertion | 606 | Report hash, finding |
| Verify image/report relation | 1,172 | Image + report hashes, finding |
| Total | 2,072 | |

3,526 candidate/request links become 2,072 logical requests. Shared image
evidence is not requested four times merely because it has four report
experts; identical report artifacts are also deduplicated within their fixed
case and dependency. Requests remain correlated, and finding-level requests
are **not** model invocations. A future verifier may batch or reuse many of
them; no batching or cost estimate is invented here. This is neither 2,072
new model calls nor evidence of measured GPU savings.

No additional budget is supplied in the completed run. Limits, affordability
and prospective runtime remain unconfigured/unavailable, never zero-cost or
implicitly affordable. If both optional caps are supplied, the existing
budget contract counts failed/cancelled/timeout attempts and retains unknown
runtime. An exhausted declared budget yields `stop_unresolved_preview`, not
an executed rejection. Budget affordability cannot authorize a model call.

## Reproduce without inference

Implementation:

- `TriCompose-v1.2/src/tricompose_v12/reliability_preview.py`: pure metadata contract.
- `TriCompose-v1.2/benchmarks/preview_reliability_evidence.py`: protected CPU wrapper.
- `TriCompose-v1.2/tests/test_reliability_preview.py`: 28 invented-fixture tests.

The wrapper requires an existing authorized CPU Slurm allocation, the pinned
sidecar manifest/artifact hashes and the complete 80 × 3 × 4 inventory. It
validates all four-state facts, shared-artifact consistency, provenance,
directed edge counts, availability reasons and report-group deduplication.
Sources are hash-checked before and after processing. Output commits are
atomic and an existing run ID is refused.

From an existing CPU allocation, use a **fresh opaque output ID**:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  TriCompose-v1.2/benchmarks/preview_reliability_evidence.py \
  --reliability-run artifacts/protected/tricompose_v1_2/candidate_reliability_overlays/reliability_pool960_12632006_001 \
  --output-root artifacts/protected/tricompose_v1_2/reliability_evidence_previews \
  --run-id evidence_review_new_001
```

Optional `--max-additional-calls` and `--max-additional-gpu-seconds` must be
supplied together. They declare hypothetical additional budgets only, not
authorization for submission, inference or regeneration.

## Verification and remaining work

All **1,075 V1.2 regression tests** passed in 10.581 seconds, including the
28 new metadata-only tests. Construction/validation took 0.622416 seconds;
that timing excludes output serialization/commit and is not model runtime.

An independent read-only check verified 960 candidate identities/order and
fixed hashes, all source patterns and trace states, 2,072 dependency identities
and consumer/evidence inventories, 3,526 request links, CSV reasons/NA/false
fields, five output artifact hashes, ten source hashes and project-private
permissions. These are engineering checks, not clinical validation.

Next align **existing automatic verifiers** to a small, predeclared request
subset before considering regeneration. Report assertion/scope checks and
same-finding image evidence should remain separate; unsupported heads retain
NA. Keep all EHRs and report coverage separately from proxy quality. Human
feedback is not required for that exploratory automatic experiment, but
clinical accuracy/fault claims still need independent evidence. Any GPU work
requires its own frozen protocol, complete Slurm script/resources and explicit
approval. Do not silently turn these requests into an executable controller.

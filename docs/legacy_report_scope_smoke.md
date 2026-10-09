# Frozen report-scope check on the historical synthetic bank

Status: completed in existing CPU allocation **12632006**, with zero new model
calls, GPU requests, Slurm submissions or regenerated EHR/CXR/report artifacts. This is an
automatic report-text scope diagnostic, not clinical validation, image
factuality, error localization or successful targeted repair.

## Fixed cohort before text inspection

The [full-bank evidence preview](reliability_evidence_preview_protocol.md)
identified gaps and proxy disagreements. This step reuses an **existing frozen
literal scope gate**, not a newly trained scorer or new clinical rules.
The 80-case historical bank keeps its fourteen-finding uncalibrated profile;
the older eight-finding scoped pilot is not overwritten or reinterpreted.

Before opening any report body, a metadata-only plan fixed the first two
sorted opaque EHR case IDs and **all** their candidates: two EHRs × three CXR
paths × four report experts = **24 reports / six image slots**. Case selection
does not use scores, winner flags, disease prevalence or guard coverage. This
is a development engineering subset, not a representative or untouched test.

Only those synthetic report texts were read internally in the CPU allocation.
No raw patient EHR row, real report/target, EHR body, image pixels or model
weights were opened. No text or source quote is emitted to public logs, Git
or chat. New receipts store exact Unicode offsets and SHA256 without copying
the report quote. Original synthetic texts remain unchanged.

## Frozen checker and boundary

The existing gate is
`TriCompose-v1.2/src/tricompose_v12/report_assertions.py`, calling the unchanged
`scope_check()` in
`TriCompose-v1.2/benchmarks/repair_cached_report_evidence.py`.
Checker SHA256 remains
`7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18`.
Its four literal heads are cardiomegaly, consolidation, pleural effusion and
pneumothorax. This is **not** an official NegEx/NegBio implementation, a complete
clinical synonym extractor or an independently qualified clinical scorer.
Earlier authored development diagnostics and post-hoc gate design do not
establish held-out clinical accuracy.

The gate can retain the **same** frozen CheXbert proposal if its literal scope
is covered, or withdraw it to unknown. It cannot flip a state, recover a missing
model assertion, add disease/device facts, or resolve image truth. Unknown and
uncertain remain separate from negative. Pneumonia, edema, support devices and
the other unsupported heads keep explicit unavailable status.

## Completed outputs

```text
artifacts/protected/tricompose_v1_2/legacy_report_scope_plans/scope2_plan_12632006_001/
  plan.json
  manifest.json

artifacts/protected/tricompose_v1_2/legacy_report_scope_runs/scope2_checked_12632006_001/
  scope_fact_table.jsonl
  candidate_scope_table.csv
  scope_dependency_groups.jsonl
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Plan-manifest SHA256:
`5856c43c78eaa4b8c8959b153a6ee188ef97cfe3d617cce2ce14556f06e4d69c`.
Completed result-manifest SHA256:
`64b46bd52292462274e4a9ac2f16aade397189f945081060a906875032733e3b`.
Directories/files retain project-group access and modes 2770/0660. Existing
run IDs are refused **before text access**. No old score, winner, EHR, report,
threshold, checkpoint or dataset was edited.

## Results: evidence was filtered, artifacts did not improve

All 24 report slots and **336 finding rows** (24 × 14) remain. There are
22 distinct report-artifact hashes; all 24 source file slots were byte-hash
verified. UTF-8 is decoded exactly without stripping, silent truncation,
replacement characters or CRLF normalization. No text artifact was unavailable.

The four supported heads give 96 candidate/finding checks:

| Scope result | Candidate/finding checks |
|---|---:|
| Retain original proposal with checked literal scope | 16 |
| Abstain: proposal lacks complete checked scope or is vetoed | 26 |
| Original extractor had no assertion | 54 |
| Supported-head total | 96 |
| Other ten heads, explicitly unavailable | 240 |
| Full fourteen-finding inventory | 336 |

**All 16 retained proposals are negative.** The gate retains 8 effusion and
8 pneumothorax proposals, but no cardiomegaly or consolidation proposal in this
subset. This is observed retention, not an assessment that the 26 withdrawn
proposals are wrong. Reasons include uncovered literal terms/scopes and
qualified absence. The narrow checker cannot currently provide balanced
positive/negative scope coverage on this generated-report subset and should
not serve as the sole ranking or regeneration gate. No synonym, threshold or
rule was added after inspecting these results.

The image/report proxies on **only the same four supported heads** change as
follows:

| Four-head CXR–Report readout | Raw | Scope-filtered |
|---|---:|---:|
| Inventory | 96 | 96 |
| Comparable proxy facts | 42 | 16 |
| Support | 25 | 7 |
| Opposition | 17 | 9 |
| Unknown | 54 | 54 |
| Not comparable | 0 | 26 |

Eight opposition signals and eighteen support signals were withdrawn by
abstention. Across the **full fourteen-head** table, comparable facts fall
68→16, support 48→7 and opposition 20→9; the additional removed comparisons
are unsupported-head masking. Do not mix these denominators or claim that
falling opposition means better generated CXR/report quality. Conditional
opposition actually rises on the retained, negative-only subset; that is a
different comparison set, not a clinical failure rate.

Same-image report-proposal disagreement groups fall 4→0 out of 84
image/finding groups. That is withdrawal of evidence, **not** four resolved
report contradictions or successful repairs. Shared-image reports are
correlated. The two fixed EHRs have no explicit cached comparison facts under
the historical extraction scope, so EHR-edge reference-conditioned rates
remain NA rather than zero or perfect agreement. This does not imply that
the source EHRs contain no clinical information.

## Reproducible existing-CPU execution

Implementation:

- `TriCompose-v1.2/src/tricompose_v12/legacy_report_scope.py`: pure scoped-receipt builder.
- `TriCompose-v1.2/benchmarks/check_legacy_report_scope.py`: metadata prepare and bounded text execution.
- `TriCompose-v1.2/tests/test_legacy_report_scope.py`: 21 invented-fixture tests.

The wrapper requires an existing CPU Slurm allocation. Preparation authenticates
the full bank and reliability manifests, all eight report-run manifests and
the selected 24 candidate metadata files without reading text. Execution
revalidates the entire source selection and locator against those immutable
manifests before opening only synthetic text. Every raw state and source
evidence ID is retained in a separate diagnostic. Image loaders, model
factories and GPU workers are not invoked.

From an existing authorized CPU allocation, use **fresh opaque run IDs**:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  TriCompose-v1.2/benchmarks/check_legacy_report_scope.py prepare \
  --reliability-run artifacts/protected/tricompose_v1_2/candidate_reliability_overlays/reliability_pool960_12632006_001 \
  --candidate-run artifacts/protected/tricompose_v1_2/complete_bank_endpoints/complete_bank_secondary_12624822_001 \
  --output-root artifacts/protected/tricompose_v1_2/legacy_report_scope_plans \
  --run-id scope_review_plan_new_001

PYTHONDONTWRITEBYTECODE=1 python \
  TriCompose-v1.2/benchmarks/check_legacy_report_scope.py run \
  --plan-run artifacts/protected/tricompose_v1_2/legacy_report_scope_plans/scope_review_plan_new_001 \
  --output-root artifacts/protected/tricompose_v1_2/legacy_report_scope_runs \
  --run-id scope_review_checked_new_001
```

## Verification and next experiment

All **1,096 V1.2 tests** passed in 10.453 seconds, including 21 new invented
source-text/metadata tests. Execution construction/validation took 0.656001
seconds, excluding output serialization/commit; this is not inference runtime
or GPU savings. Independent checks verified all 336 raw states/hashes,
candidate order, scoped transitions, quote-free offset receipts, EHR/image
invariance, group deduplication, five output hashes, 71 source hashes and
private permissions. Contract tests do not establish clinical accuracy.

The next useful automatic experiment is a **frozen semantic report-assertion
verifier on this same sealed cohort**, retaining exact source evidence and
separately reporting positive/negative/unknown coverage. Reuse an existing
frozen local verifier; do not expand literal rules to fit these outcomes or
simply scale an inadequate gate to every case. Semantic model outputs remain
proposals, not clinical truth, and cannot silently replace scores or winners.
Human feedback is not required for that exploratory experiment, but strong
clinical error/localization claims remain unvalidated. Any GPU execution
requires the complete script/resource request and separate explicit approval.

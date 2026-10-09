# Report verifier progress: metadata-only development diagnostic

## Purpose and scope

Join the completed span-V2 report-extraction diagnostics to the existing
reliability evidence-request preview. This is not another Agent, new clinical
score, fitted policy, adjudication or permission to regenerate. The classification
rules were developed after the source tables had been observed; no claim of
preregistration or held-out qualification is made.

Worker: `TriCompose-v1.2/tools/diagnose_report_verifier_progress.py`.
Invented-fixture tests: `TriCompose-v1.2/tests/test_report_verifier_progress.py`.
Only standard-library metadata aggregation runs; no model imports or requests.

## Inputs and boundaries

The worker pins three existing parent manifest hashes and reads only:

1. `real_validation/official_span_v2_12645961/summary.json`: aggregate four-state
   reference/prediction matrices, not original annotations or real predictions.
2. `verification_runs/span_v2_scope2_12645404_analysis/assertion_comparison.jsonl`:
   quote-free synthetic candidate/finding rows, not reports, prompts, images,
   evidence quotes or raw model responses.
3. `reliability_evidence_previews/evidence_pool960_12632006_001/evidence_requests.jsonl`:
   existing opaque requests and dependency hashes.

Ancestor `source_paths` are not traversed. Every consumed artifact is hash-checked
before loading; eight direct input/program bindings are rechecked before commit.
Fresh atomic runs only; files 0660, directories 2770, project access group only.
Nothing is copied into public logs or Git except code and aggregate documentation.

## Meaning of the statuses

The manual-reference table describes directional extraction differences:
positive/negative flips, determinate→uncertain, uncertain→determinate,
annotated→unmentioned, unmentioned→determinate/uncertain, and exact matches.
Unmentioned→determinate is not automatically a hallucination; annotation-policy
equivalence and selected-span correctness have not been independently established.

The synthetic table describes symmetric CheXbert/Qwen disagreements. Neither is
used as truth. Explicit agreement, explicit polarity opposition, uncertainty,
single-extractor assertions, both-unmentioned, unavailable verifier and outside
scope remain separate. Unknown is not negative; uncertain is not explicit
agreement or a hard polarity conflict. Failed availability is not a successful
unknown prediction. Source cells, clinical scores and selection flags are kept.

Request matching requires exact `(consumer candidate ID, finding, report hash)`.
An identical text hash in another candidate does not expand the fixed cohort.
Report-only checks cannot satisfy image findings or image/report relation requests.
The original `execution_status=not_executed` is preserved; `reportcheck_*` fields
record later sidecar progress without rewriting the old execution history.
Technical review completion does not mean clinical resolution.

## Completed run

CPU metadata aggregation used existing allocation 12645021; no new submission,
GPU call, external API, training, thresholds/weights or primary ranking changes.

Protected output, relative to `artifacts/protected/tricompose_v1_2/`:

```text
report_verifier_progress/progress_scope2_12645021_001/
  manual_reference_transitions.csv
  report_verifier_fact_table.jsonl
  report_verifier_candidate_table.csv
  evidence_request_progress.jsonl
  evidence_request_progress.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manual-reference Qwen checks: 60 total, 26 annotated. Twenty annotated states
match, two reverse polarity and four become uncertain. Of 34 unmentioned
references, 27 remain unmentioned, five become determinate and two uncertain.
This is the same previously used small report diagnostic, not a candidate
confidence calibration or independent image ground truth.

The fixed synthetic subset remains 24 slots / 22 report hashes / 336 finding rows.
Within four heads (96 checks): 41 explicit agreements, one polarity opposition,
16 single-extractor assertions and 38 both-unmentioned. The remaining 240 rows
are outside this verifier's scope. No supported-head uncertainty/failure was
observed in this subset, but these states have explicit handling and tests.

All 2,072 original logical requests remain intact: eight now have technically
completed, clinically unqualified report reviews; two match the subset but are
outside the four heads; 2,062 are not covered by this check. Clinically resolved
requests: zero. These are request records, not new model calls.

Independent metadata audit reproduced the 128 aggregate matrix cells, all 336
original fact rows/cells, all 2,072 original request rows/cells, dependency joins,
candidate counts, eight source hashes, seven artifact hashes and protected modes.
Historical full-bank and reliability overlay manifest hashes remain unchanged.
Full V1.2 suite: **1,201 passing tests** (8.860 seconds), including 19 new invented
fixtures. Output manifest SHA256:
`b8999d053b6f6adb82127915e8a9194cc4fa3d65ea503ca239cce21bbd861837`.

## What this enables next

The candidate and request tables show which assertion needs more evidence without
claiming which modality is faulty. A subsequent verification experiment must
declare its independent reference, fixed cohort, endpoint and call budget before
execution. Repeatedly improving the same scorer's own output is not independent
evidence of clinical repair. Any new GPU submission requires its complete script,
resource request and separate explicit approval.

# V1.2 decision-interface preview

Status: engineering interface only. It does not execute or validate clinical
error localization, repair, or an agent. All generators/scorers remain frozen.

## Purpose

Connect the existing per-finding evidence contract to a deterministic next-step
request before an executable loop is considered. Preserve the fixed EHR,
unknown/uncertain states, every finding denominator, and hash-bound lineage.
Do not fit weights or rules to the diagnostic test results.

Inputs are the immutable synthetic `fact_scope_table.jsonl`, with eight cached
findings and the unchanged four-finding report syntax guard. No report body,
image pixels, raw EHR, real target, model, external API or downloaded asset is
read by this preview.

## Pattern interpretation

| Cached pattern on an explicit comparable fact | Preview interpretation |
|---|---|
| EHR = classifier; scoped report opposite | Provisional report verification target |
| EHR = scoped report; classifier opposite | Provisional CXR verification target, NOT image fault confirmation |
| Classifier = scoped report; both oppose EHR | Ambiguous downstream/EHR-conditioning discrepancy |
| Opposite provisional targets across findings | Ambiguous locus; verify rather than pick one |
| Unknown, uncertain or unsupported report head | Retain missing comparison and direct-EHR coverage gap |
| Reports disagree on the same CXR | Correlated disagreement, not independent majority voting |
| All cached states agree | Agreement only, not proof of clinical correctness |

All proposals currently resolve to `verify_more`, or a **proposed** `reject`
when the configured additional budget is exhausted. Execution authorization,
clinical acceptance and targeted-repair approval are always false. A provisional
target is NOT a ground-truth label; there is no reported localization accuracy.
This preview neither selects a new winner nor marks an existing case rejected.

## Cost contract

The caller declares maximum **additional** model calls and GPU seconds after
the existing bank. These are engineering caps, not calibrated medical stopping
thresholds. Every attempted additional call, including failed/retried/timeout/
cancelled calls, counts. Missing elapsed time or a missing verification-cost
estimate remains unavailable, never zero. Budget affordability cannot grant
model execution. The already-spent bank generation/verification cost is separate;
offline preview runtime is not prospective compute savings.

## Files and execution

- Pure interface: `TriCompose-v1.2/src/tricompose_v12/decision_preview.py`.
- Protected CLI: `TriCompose-v1.2/benchmarks/preview_candidate_actions.py`.
- Invented fixtures: `TriCompose-v1.2/tests/test_decision_preview.py`.

Inside an existing authorized Slurm allocation, the derived-only CLI produces a
new opaque protected run containing `decisions.jsonl`, `action_preview.csv`,
`summary.json`, `README_CN_EN.md`, and hashes in `manifest.json`. It requires the
full existing two-EHR/48-candidate scope inventory, verifies source hashes,
refuses an existing output run, and retains all underconditioned cases.
Public output contains status, runtime and manifest hash only.

Further scientific progress still needs independent evidence and frozen
calibration/evaluation boundaries. To activate a repair policy, review the
independent clinical gate, controlled intervention/adjudication benchmark,
per-finding error/abstention trade-off, and actual equal-budget repair comparison.
Invented-fixture tests or this cache preview cannot satisfy those requirements.
Every new Slurm submission still needs complete-script review and user approval.

## Completed local engineering run

The full existing two-EHR/48-candidate cache was previewed inside CPU allocation
`12576792`, with zero model calls and no new Slurm submission. The output is
`artifacts/protected/tricompose_v1_2/decision_previews/decision_preview_12576792_001/`.
All proposals request verification; none changes the existing selection or
authorizes repair. Source/artifact hashes, inventory, protected modes and old
winner/table hashes passed checks. Thirty-three new invented-fixture tests pass;
the complete local V1.2 suite now has 473 passing tests. Clinical acceptance,
localization accuracy and repair success remain unavailable.

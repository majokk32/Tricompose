# Official ConText opacity: interface stable, scope reliability insufficient

Completed in existing CPU Slurm **12666569**, without a new submission, GPU,
trained NLP component, model download, installation, API, patient input or
generation. The already installed pinned environment was read-only.

The mechanism is official medspaCy 1.3.1 default ConText, blank English
tokenization and PyRuSH sentence splitting. Official context rules were not
changed. The only new target inventory matches literal `opacity`/`opacities`;
this custom inventory is **not** an official lung-finding ontology. It does
not infer opacity from pneumonia/edema or normal summaries, and does not
disambiguate nonpulmonary opacity. See the
[official implementation](https://github.com/medspacy/medspacy/tree/1.3.1).

## Frozen plan and outputs

```text
artifacts/protected/tricompose_v1_2/opacity_context_plans/context112_12666569_001/
  inputs.json
  references_new64.json
  plan.json
  manifest.json

artifacts/protected/tricompose_v1_2/opacity_context_runs/context112_12666569_001/
  predictions.json
  replays.json
  prediction_freeze_receipt.json
  scored_checks.json
  staged_veto_checks.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Plan manifest SHA256:
`380f452755911a9d8d0393a649e956dbb4af7831d9e6e8467c898c7c08f8f7e0`.
Result manifest SHA256:
`8fa6426963272366944b5a61fffc2508b37e558f069f7cdb4e7b35bb031dd402`.

All 112 texts completed, with **224 primary/replay parser passes**, no changed
replay and no unavailable outcome. Parsing phase elapsed **0.385574 seconds**,
excluding initialization/loading and final serialization. Model calls: zero.
Evidence keeps exact Unicode offsets/hashes and official cue/scope metadata,
not source quotes. Results remain protected and Git-ignored, modes 2770/0660.

## Separate authored-development results

The old48 set and prior Qwen results were already known. New64 texts were
frozen before this parser execution but authored by the investigator, **not**
independent clinician gold or a held-out clinical evaluation. The qualifier,
change-only, historical and resolved-mention conventions are declared task
definitions, not universal radiological adjudication. These scores must not
become synthetic candidate labels, reliability weights or clinical accuracy.

| Metric, all-attempted denominator | Existing authored48 | New authored64 |
| --- | ---: | ---: |
| Official parser state matches | 35/48 = 72.92% | 32/64 = 50.00% |
| Macro F1 | 0.71146 | 0.44798 |
| Positive/negative flips | 1 | 3 |
| Determinate on uncertain/unknown reference | 11/30 | 27/46 |
| Texts with literal target noun | 34/48 | 56/64 |
| Unavailable outputs | 0 | 0 |

On the **same old48** texts, unchanged Qwen V3 has 38/48 matches and macro F1
0.79808. No Qwen prediction on new64 was made, so there is no new64 paired
Qwen comparison. The two sets are not pooled into a misleading headline.

All old/new families, confusion matrices and state-support denominators are
in the protected summary/report. Important new64 results, four texts each:

| Family | Matches / 4 |
| --- | ---: |
| Morphology presence | 4 |
| Explicit negation | 3 |
| Uncertainty | 1 |
| Historical only | 2 |
| Family only | 4 |
| Hypothetical only | 1 |
| Resolved prior finding only | 1 |
| History then current statement | 1 |
| Opposing current assertions | 1 |
| Qualified absence | 0 |
| Change-only language | 0 |
| Other-entity negation | 3 |
| Generic summary | 4 |
| Other disease only | 4 |
| Sentence boundaries | 3 |
| Nonpulmonary opacity | 0 |

Interface/replay stability does not establish semantic correctness. The
default positive interpretation of an unmodified noun remains explicitly
unverified. In particular, matching `opacity` outside the lung demonstrates
the target inventory's anatomy limitation, not a defect in ConText's intended
context-tagging task.

## Veto unchanged V3 determinate proposals

This sidecar may only withhold a signed proposal; no proposal is corrected or
promoted. Unknown/uncertain/unavailable stay distinct; an abstention is not
credited as a correct unknown. Both mechanisms read the same source report,
so different algorithms still do **not** supply independent clinical votes.

| Cached old48 signed-assertion readout | Retained / 48 | Correct retained | Incorrect retained | Conditional match |
| --- | ---: | ---: | ---: | ---: |
| Original V3, no veto | 27 | 18 | 9 | 66.67% |
| Prior same-model cross-format agreement | 23 | 15 | 8 | 65.22% |
| Official ConText agreement sidecar | 23 | 16 | 7 | 69.57% |

The new sidecar withholds **two wrong and two correct** determinate assertions.
Coverage is 23/48 = 47.92%, not complete evaluation. Seven retained errors
remain: all four qualified-absence cases, two change-language cases and one
opposing-assertion case. The two errors withheld concern generic-summary and
other-disease-only proposals; both correct losses concern explicit absence or
resolved/current language. This is not sufficient reliability for automatic
fault localization or regeneration. Hard-action eligibility stays **0/48**.

Original four-state matches stay 38/48 and the V3 language gate stays failed.
No old label, prompt, score-table cell, candidate, EHR, winner or generation
artifact was altered. New rules were not fitted after these outcomes.

## Verification and next boundary

- 37 new invented metadata/contract tests passed; full V1.2 regression passed
  **2,086 tests in 12.011 seconds**.
- Independent standard-library audit verified all 224 evidence records,
  target offsets/hashes, five-flag mappings, state reductions, replays, all
  class/family metrics, 48 veto rows, no label flips/promotions, 77 result-source
  pins, seven result artifacts, frozen receipt and protected permissions.
- `git diff --check` and Git-ignore checks passed. New plan/result IDs remain
  non-overwriting; previously consumed code and artifacts remain unchanged.

Audit: `.tmp/audit_opacity_context112_12666569_001.py`.
Implementation: `TriCompose-v1.2/tools/benchmark_opacity_context.py`,
`TriCompose-v1.2/src/tricompose_v12/opacity_context.py`.
Protocol: `docs/opacity_context_parser_protocol.md`.

**Conclusion / 结论:** default ConText is an available different parsing
mechanism, but this disclosed setup does not solve the task's qualifier,
temporal or anatomy scope. Do not promote it to a primary score or automatic
repair gate. The reusable contribution from this run is a source-bound,
failure-aware parser adapter and an immutable wider diagnostic set, not a
validated clinical algorithm. A further published radiology-specific parser
or conservative scope policy needs a separate prospective protocol and
evaluation; any GPU work still needs explicit script/resource approval.

# Cached opacity abstention: shared errors survive agreement

Completed inside the existing **CPU Slurm 12666569** allocation. No new Slurm
submission, GPU/model call, training, download, API, clinical candidate-body
read, ranking, EHR change, or regeneration. The gate's parsing/construction
phase took 0.015671 seconds, excluding initial loading and final serialization.

## Output

```text
artifacts/protected/tricompose_v1_2/opacity_assertion_abstention_runs/opacity_abstention48_12666569_001/
  policy.json
  gated_predictions.json
  freeze_receipt.json
  assertion_eligibility_table.csv
  scored_checks.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`958e7be74e0800b663b7b45776e60fecbe6fb5312b95b0669d4ed80c9812d936`.
The readable CSV keeps all 48 proposals, statuses, cross-format states and
eligibility decisions. Evidence/source/response bodies are not copied into it.
All artifacts remain protected, project-group modes 2770/0660 and Git-ignored.

## Question and result

Can agreement between the unchanged quote-format and staged-format readings
of the **same frozen model and same authored source** veto unreliable signed
assertions? Only complete positive/negative staged proposals with homogeneous
selected assertion states and same-state complete quote-format proposals are
retained as correlated soft evidence. Unknown, uncertain and failures stay
separate. No state correction or independent expert vote is claimed.

**This agreement gate did not improve reliability on these development
fixtures.** It withheld one wrong determinate assertion but also three correct
ones, while retaining eight shared mistakes.

| Signed assertion diagnostic | Original V3 | Agreement sidecar |
| --- | ---: | ---: |
| Determinate proposals retained / all 48 | 27/48 | 23/48 |
| Correct determinate proposals | 18 | 15 |
| Incorrect determinate proposals | 9 | 8 |
| Conditional authored match among retained | 18/27 = 66.67% | 15/23 = 65.22% |
| Coverage over all texts | 56.25% | 47.92% |

This signed-only diagnostic is **not** the previous four-state accuracy.
Original V3 four-state matches stay **38/48 = 79.17%**, unchanged; no gated
overall accuracy is invented. Abstention/null is never credited as a correct
unknown. All **48** rows remain present; only comparison eligibility changes.

Decision counts: 23 correlated soft-only agreements, 11 unknown/noncomparable,
10 uncertain/noncomparable, three unavailable cross-format checks and one
cross-format disagreement. There were no mixed selected-state determinate
proposals in this particular cached run. Hard-action eligibility is **0/48**.

## Shared error and coverage boundaries

All four qualified-absence fixtures pass the agreement check but remain wrong
under the disclosed authored convention. Two change-language errors, one
generic-summary error and one other-disease-only error also survive. The gate
filters the one determinate opposing-assertion error. Correct assertions lost:
two repeated-assertion fixtures and one multisegment fixture, because the older
quote reader was unavailable. Every family and its denominator remains in the
protected report/summary; none is removed from the diagnostic.

This is a same-investigator **known development set**, not a held-out test or
clinical gold. The policy was designed after seeing V3 results. The worker
freezes its decisions before parsing the reference key, which prevents
answer-assisted execution but does not erase investigator knowledge.

The existing frozen literal scope gate has no lung-opacity head. Agreement
therefore cannot certify negation scope, qualifiers, current status, or image
truth. Different prompts do not make the model independent of itself.

## Costs and preservation

Zero new calls were made for this CPU overlay. The cached paths already cost
**134 primary calls + four replays = 138**. Prospective use would need both
paths if they are not cached; no inference-cost saving is demonstrated.
Original raw predictions, prompts, scores, old winners and failed V3 language
gate remain unchanged. The 80-EHR/240-image/960-triple candidate bank was not
opened or rescored; its original manifest SHA remains
`ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`.

## Verification

- 34 new invented-fixture tests passed; the full V1.2 regression passed
  **2,049 tests in 12.285 seconds**.
- Independent standard-library audit verified all 48 decisions/joins, null
  and unknown handling, no label flips, preserved source-span/hash metadata,
  all readouts/families, CSV cells, 24 source pins, seven artifacts, freeze
  receipt and protected permissions.
- The earlier independent parent audit was rerun: all **138 cached responses**,
  30 source pins, seven artifacts and original 38/48 state matches still pass
  their engineering checks; the language gate remains failed.
- `git diff --check` passed; old run overwrite is refused.

Independent audit:
`.tmp/audit_opacity_abstention48_12666569_001.py`.
Policy/protocol and implementation:
`docs/opacity_assertion_abstention_protocol.md`,
`TriCompose-v1.2/src/tricompose_v12/assertion_abstention.py`,
`TriCompose-v1.2/tools/audit_cached_opacity_abstention.py`.

## Next boundary

Do not deploy same-model agreement as a high-confidence automatic repair rule.
The next useful experiment is a genuinely different **scope/negation parsing
mechanism**, with a prospectively frozen broader fixture set and explicit
coverage/failure readouts. Keep original EHRs and generation artifacts fixed.
Any later GPU/model experiment requires its own full script/resources and
explicit approval; this CPU result authorizes no such execution.

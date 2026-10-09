# Cached opacity assertion eligibility and abstention

CPU-only, veto-only diagnostic on the existing 48 wholly authored V3 texts.
This policy is declared **after the investigator observed V3 development
results**. Freezing decisions before reading the key in this worker prevents
answer-assisted execution, not investigator knowledge or post-hoc design.
It is not a held-out benchmark, calibration, or clinical validation.

## Purpose and frozen policy

Keep every raw prediction and every input. Add a separate eligibility sidecar:

1. Failed extraction -> unavailable, not an unknown/negative prediction.
2. Complete unknown -> no signed comparison; preserve unknown.
3. Complete uncertain -> no signed comparison; preserve uncertainty.
4. Determinate positive/negative requires mechanically checked source hashes,
   spans and decoder output, homogeneous selected segment states, and a
   complete same-state quote-format proposal from the earlier path.
5. A passing row becomes **correlated soft evidence only**. The two paths
   use the same source and same frozen Qwen model, not independent experts.
6. All hard-action eligibility is false. The existing frozen literal scope
   checker supports four other findings, not lung opacity. Do not invent a
   new qualified scope head, clinical confidence, or correctness guarantee.

No lexical disease/negation rules, new synonyms, thresholds, learned scorer,
prompt changes, fallback label flips, retries, inference or external API.
Agreement is a veto-only engineering check, not a sufficient clinical rule.
Two prompts can share the same mistake. Do not tune the policy after seeing
this diagnostic's coverage/error readout.

## Fixed inputs and ordering

Parent run: `opacity_assertion_stages_runs/authored48_stages_v3_12677513`;
manifest SHA256:
`73dd5b1a26a0043caac1e4822ff901a496167947d873b12201e6d7c19e4bacce`.

Parent plan: `opacity_assertion_stages_plans/authored48_stages_v3_12666569_001`;
manifest SHA256:
`238a8e44133c2edc9c69d460a49fcbfbb5e62ef38ed64672699a808eaa088551`.

Reparse the existing primary/replay responses with unchanged pinned decoders;
require exact equality to cached outcomes, source IDs, quoted spans and hashes.
Use only mechanically validated proposals for decisions. Save and fsync
`policy.json` and `gated_predictions.json`, then create a hash freeze receipt.
Only afterward parse authored reference states/families for diagnostic counts.
Never read clinical candidate bodies or real inputs/targets. Hash the existing
candidate-bank manifest only to confirm it remains unchanged.

All source/prediction files and old results remain immutable. Create a fresh
protected run atomically; refuse overwrite; directories 2770, files 0660,
project group ruishanl_1185 (NFS may expose nobody).

## Readouts and costs

Report all 48 rows, decision counts, original four-state matches, original
determinate assertions, retained correct/incorrect assertions, coverage,
incorrect determinate assertions withheld, and correct determinate assertions
lost. Preserve per-family denominators. An abstention/null is never counted
as a correct unknown. Conditional match is not overall accuracy; no gated
four-state accuracy or clinical score is invented.

This cached CPU gate has zero new model calls. The two proposal paths already
cost 134 primary calls (48 baseline + 86 staged), plus four replays. Prospective
use would require both proposal paths when they are not cached; agreement is
not free additional verification or evidence of lower inference cost.

Keep the previously failed V3 language gate failed. No ranking, case removal,
fixed-EHR change, artifact regeneration, winner replacement, or new clinical
fault conclusion. This run does not qualify the full 960-candidate pool.
If shared errors survive, report them instead of loosening the gate or
calling agreement reliable. A further semantic-scope experiment needs a
separate prospective protocol and independent evaluation.

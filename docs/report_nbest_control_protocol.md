# Fixed-image native n-best engineering control

Status: the two-case engineering run completed as approved job 12629940 in
56 seconds. All six report/scorer/secondary interfaces passed metadata audits;
zero alternatives passed the evidence-improvement gate, so both baselines were
retained unresolved. No clinical repair or efficacy is demonstrated. Any new
submission requires complete script/resource review and explicit user approval.
No new training, checkpoint, source EHR, or real target.

## Why this replaces the initial seed-only suggestion

The installed MAIRA-2, LLaVA-Rad and CheXagent-2 adapters use deterministic
decoding. CXRMate-single uses deterministic four-beam search. Changing a seed
alone is not a valid new-candidate/diversity experiment. The existing local
CXRMate checkpoint inherits native Transformers generation and supports
`num_return_sequences <= num_beams` by source inspection. The approved two-case
P100 run subsequently verified that this local interface returns three sequences.

Use CXRMate-single (CXR-only, NOT CXRMate-ED) with its existing checkpoint,
float32 precision, official resize/crop/normalization/tokenizer, four beams and
maximum length 256. The only extension is returning the top THREE sequences
instead of keeping only top one; explicitly `do_sample=False`. No extra text,
EHR facts, labels, previous report or evaluator answer is supplied to generation.
This is NOT three independent experts or stochastic seeds.

## Frozen scope

- The authenticated sorted original 80-EHR bank supplies opaque ordinals 0/1.
- Fix original Sana seed-0 images by their original IDs and byte hashes; do
  not regenerate images, edit prompts, enrich EHR or choose favorable failures.
- The staging plan makes a two-row METADATA copy, leaving original image
  paths and candidate JSON values unchanged. It does not fabricate missing
  historical tokenizer traces or claim historical tracing is certified.
- Each image receives ONE native `generate` call returning THREE reports.
  Fresh rank zero is the baseline; ranks one/two are alternatives. Historical
  top-one SHA equality is recorded, not presumed across hardware/decoding.
- Keep all six outputs, including duplicate reports; exact and normalized
  duplicate counts get no diversity credit.
- Fresh XRV and CheXbert score every baseline and alternative using one pinned
  profile. The XRV threshold bundle enables eight heads; disabled heads stay
  unknown. Operating-point-normalized scores are NOT probabilities. The
  historical uncalibrated 14-head scores are not reused or mixed.

## Exploratory gate and independent measurement

For each alternative, preserve supported image-positive and direct EHR fact
IDs, all previously comparable image/report and EHR/report fact IDs, and no
new explicit proxy opposition. At least one support/coverage set must improve
or an opposition must be removed WITHOUT silencing it into unknown/uncertain.

Reuse existing report-structure functions without changing them. Require a
nonempty report and the CXRMate findings/impression contract; completeness
cannot fall, genericity/unsupported temporal flags cannot worsen, sentence and
four-gram repetition cannot rise. These are structural proxies, not proof of
factuality. Measurement mentions alone are not automatically errors.

Select the lowest native beam rank among gate-passing nonduplicates. If none
passes, retain rank zero as unresolved. Selection is sealed before BioViL-T;
BioViL never breaks ties or changes the gate. Report the independent raw cosine
for ALL six pairs, plus paired baseline/selected changes even if negative.
Full text is used; over-context inputs are NA with reasons, not truncated/zero.
All-unknown EHR edges remain NA, not perfect consistency or exclusion.

## Cost, failure and claims

Two native report-generation invocations return six sequences. Charge the
shared call to rank zero only; other ranks have zero ADDITIONAL invocations,
not free independent generation. Persist beam width, model loading time,
generation-only time, stage startup/I/O wall time and actual peak VRAM.
XRV scores two samples; CheXbert scores six in a batch of six. Legacy scorer
`model_calls` fields count samples and must not be mislabeled forward batches.
BioViL records actual image/text encoder calls and NA counts separately.

Use fsynced stage reservations before spawning owned bounded processes and a
native per-image call journal before generation. Retain private failed work;
no automatic retries/resume or existing-run overwrite. This batch journal is
NOT the adaptive single-case ledger, which cannot yet represent one multi-output
native call without an explicit extension.

This is same-image candidate diversification and gate verification, NOT online
error-triggered targeted regeneration, validated fault attribution or a clinical
repair result. It runs every predeclared beam candidate, so no adaptive savings
can be claimed. The two reused development cases cannot establish efficacy.
Original winners remain untouched. Untouched cohorts, equal-cost controls,
independent evidence, quality/diversity and actual time-budget curves are still
needed before paper-grade targeted-repair claims.

## Files

- `TriCompose-v1.2/src/tricompose_v12/report_nbest.py`: pure policy/gate/measurement.
- `TriCompose-v1.2/benchmarks/prepare_report_nbest.py`: CPU-only authenticated plan.
- `TriCompose-v1.2/benchmarks/run_report_nbest.py`: approved GPU workers/controller.
- `TriCompose-v1.2/tests/test_report_nbest.py`: invented-fixture safety tests.
- Plans/results stay under `artifacts/protected/tricompose_v1_2/` with project
  group access only. Source pinning happens AFTER implementation/testing; the
  approved plan fails closed if source, assets, inputs or thresholds change.

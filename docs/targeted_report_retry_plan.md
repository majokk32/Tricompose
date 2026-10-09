# Next prospective report-retry experiment / 下一步同图报告重试

Status: full targeted retry remains **PLAN ONLY — not implemented or authorized**.
A narrower native n-best engineering precursor completed as approved job
12629940, with no gate-passing improvements and both baselines retained;
see [its protocol](report_nbest_control_protocol.md). This does not authorize
or implement the broader targeted-retry plan.
This follows the completed full-bank endpoint, same-image report-path and
label-preserving headroom controls. See
[the diagnostic contract](report_repair_headroom_protocol.md).
The existing source selector/winners and historical generated artifacts stay
immutable. All models remain frozen; no training or endpoint-weight fitting.

## Question

Does a bounded report retry provide better report evidence/quality per actual
computation than a fixed report or the same-budget non-targeted retry?
The answer must not be "the score increased" or "the agent chose it".
The historical static winner already has no strict same-label-preserving
alternative in the four-report pool; this is partly a consequence of its
existing objective, not independent proof of clinical correctness.

The opportunity diagnostic found label-preserving swaps from fixed baselines,
but some lose secondary BioViL. Thus changing weights to manufacture a better
cached winner is not the next experiment. New candidates can test whether
online retry is useful, while keeping the claim exploratory until independent
evidence is available. Local decoder inspection found deterministic adapters:
seed-only retries cannot promise diversity. The precursor instead returns three
sequences from CXRMate's existing four-beam search. It is NOT an online retry,
independent-expert ensemble, or validated implementation of this broader plan.

## Smallest engineering trial

1. Fix two opaque case ordinals (0 and 1 in the authenticated sorted cohort),
   without selecting easy diseases, favorable endpoints or failures. Fix one
   predeclared CXR-generator path and existing image per case, independently
   of report scores. Retain EHR and image hashes unchanged.
2. Freeze the baseline report expert and valid native sampling settings before
   new results. Do not pick a per-case expert using future endpoint values.
   Preserve official image preprocessing, text input and checkpoint revision.
3. Generate at most TWO additional report candidates per image (one declared
   alternative model, one declared alternate stochastic seed if its official
   interface actually supports stochastic inference). If a model is strictly
   deterministic, do not count a duplicate deterministic call as diversity.
4. Use one compatible frozen scorer profile for old and new reports on each
   fixed image. Re-score the baseline if needed; legacy 14-field cache and
   fresh eight-enabled-head workers are NOT interchangeable. Do not mix their
   probabilities, normalized operating points or state thresholds. Persist
   checkpoint, preprocessing, thresholds, section policy and label inventory.
5. Attach source-bound before/after receipts with preserved positive and direct
   EHR evidence, comparable-fact coverage, explicit opposition and unavailable
   states. Use the existing bounded call/failure ledger. Record actual time,
   successful calls, failed attempts, retry counts and peak memory.
6. Keep new candidates separate; do not overwrite or replace old winners.
   A retry can remain unresolved/rejected. No report opportunity alone may
   authorize CXR regeneration or natural clinical fault attribution.

This two-case trial is interface/cost verification, not statistical efficacy
or held-out validation. Do not expand to all 80 cases until image identity,
sampling diversity, scorer-profile compatibility and receipts pass checks.

## Proposed decision boundary, to freeze separately

Use a strict label-preservation gate as an EXPLORATORY acceptance constraint,
not clinical acceptance: known supported fact IDs cannot disappear; existing
comparable conflict cannot disappear solely by becoming unknown; no new
explicit proxy opposition; metadata/structure checks cannot worsen. At least
one relevant evidence set must improve. Unknown remains unknown. If no such
candidate is produced, retain the original and mark `unresolved`, or spend a
predeclared verification budget; do not weaken the gate after seeing results.

Gate passage does not guarantee clinical fidelity, and a secondary endpoint
fall must be recorded rather than hidden. A report asserting a finding with
an unknown image reference needs additional evidence, not automatic false/true
classification. Report models share the same image and are correlated.

Do NOT inject classifier labels or a candidate's final report into the report
generator as a supposedly true answer. Diagnostic finding IDs can decide WHICH
action to request, but generation must remain grounded in the unchanged image
and its documented input contract. A prompt that simply tells the generator
the evaluator's answer would create self-confirming agreement.

## Comparison and evaluation

For the subsequent fixed larger cohort, predeclare fixed baseline, equal-call
random/non-targeted retry, same-budget static reranking and targeted retry.
All cases remain, including sparse-EHR and unresolved outcomes. Existing EHR
generation/image costs are shared sunk costs in this report-only experiment;
report generator/scorer/evaluator load and execution, failures and verification
costs are not free. Do not claim end-to-end GPU savings from this report-only
control or simulated cache budgets.

Selection evidence and endpoint evaluation remain separate. Report known and
comparable counts, positive/negative support separately, contradictions,
omissions, structural/temporal hallucination checks, alternative evaluator
disagreement, abstention and quality/cost curves. BioViL is secondary and not
an error-location gold standard. Missing real-reference report metrics remain
NA; they cannot be fabricated for fully synthetic unpaired triples.

An efficacy claim requires an untouched synthetic cohort and appropriately
independent clinical evidence; the current 80-EHR bank is already inspected
development data. Controlled, clearly labeled intervention tests may validate
mechanical decision behavior, not substitute injected-modality IDs for true
clinical mistakes.

## Before execution

Implement/audit the request manifest, native sampling support, source lineage,
scorer-profile adapter and failed-call accounting first. Then check current GPU
availability and show the complete exact Slurm script/resource request for
explicit approval. Do not submit a job, change old thresholds, install/download
models or consume raw real targets merely because this plan exists.

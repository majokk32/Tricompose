# Native finding + frozen ConText: prospective veto-only scope probe

Build a generic source-bound eligibility sidecar over all fourteen official
CheXpert categories. Keep the native labels unchanged. Reuse the frozen
medspaCy 1.3.1 English ConText rules (102 rules); do not invent/tune new temporal,
negation, anatomy or qualifier rules from earlier failures. This is a cheap
training-free applicability diagnostic, not a new clinical scorer or Agent.

## Evidence contract

Run the unchanged official CheXpert Classifier through the existing failure-aware
worker. Reconstruct the exact official cleaned full-report passage for ConText
and require original/cleaned SHA256 equality. Supply the exact native annotation
inventory, not a second independently invented disease matcher. Use the official
ConText `input_span_type=group` API to preserve overlapping phrases/categories;
deduplicate only identical (start, end, category) processing targets, then link
every native annotation ID back to its target. Strict character/token alignment
is mandatory; no span expansion or dropped mentions to improve availability.

Record five official flags, cue/scope offsets/hashes and per-mention attribution:
historical/hypothetical/family **flagged** or **not flagged**. Not flagged does
not mean verified current patient pathology. Anatomy and qualified/change-only
semantics remain explicitly unverified. No original-offset alignment claim.
Only hashes, offsets, flags and structured states are serialized, not bodies.

Native unknown/uncertain is noncomparable; No Finding is aggregate-only. Any
native/source/context failure remains unavailable/null, not unknown. A signed
proposal is retained as **soft** only when its native mention-conflict view,
every matched ConText mention and native sign agree, with no history/hypothesis/
family/uncertainty flag. Mixed current/history evidence conservatively vetoes
the whole finding; measure lost correct evidence, do not call it a correction.
Neither parser promotes or flips the native state. Correlated same-report
agreement is not independent clinical truth or permission for regeneration.

## Prospective fixed controls

Freeze `native_context_scope_controls_v1.py` before execution: 80 invented texts
from eight finding terms x ten scope families, plus eight nonpulmonary-opacity
probes = **88**. Findings: lung opacity, pneumonia, edema, cardiomegaly, pleural
effusion, pneumothorax, atelectasis, consolidation. Include presence, absence,
uncertainty, history, family, hypothesis, resolved, qualified absence, change
and opposing assertions. Nonpulmonary tests must expose the anatomy limitation.
No favourable case selection, real source, patient text or copied report.

These are investigator-known, templated development task definitions, not
clinician labels, held-out natural reports or clinical accuracy. Conservative
qualified/change-only reference conventions are retained as task definitions,
not asserted universal radiology truth. Do not tune to this run or add phrase-
specific fixes after seeing outcomes. All 88 attempts remain in denominators.

The native worker sees only IDs, text and hashes; not target finding, family or
reference state. ConText/gate runs all native categories without target labels.
Run each native parse and context/gate twice. Freeze/fsync native outputs,
context evidence and gates before decoding reference JSON for evaluation.
Reference semantics are not passed to inference. Hashing fixture/source bytes
does not make the developer blind or count as decoding reference semantics.

## Diagnostics, permissions and immutable results

Report raw four-state alignment/F1 separately from veto-only retained precision
and coverage, incorrect retained, incorrect withheld and correct lost. An
abstention is not credited as a correct unknown, not a corrected report or an
increase in overall accuracy. Show every family and every finding, parse/context
availability, replay changes and remaining nonpulmonary/qualifier/change errors.
Fail closed on resource/source/runtime drift. All attempts, including failures,
are retained; no selective retries, rule fitting or hidden qualifier whitelist.

Reuse existing CPU Slurm only. No GPU/training/model installation/download/API.
Any new sbatch would need full-script/resource display and explicit approval.
External environments/checkpoints are read-only. Temp/cache/bytecode stay in
workspace; new atomic protected plan/run IDs, dirs 2770/files 0660, project
group only. Do not overwrite consumed code, old outputs, score tables, EHRs,
selected triples or qualification gates. Do not label the 960-slot clinical
candidate bank or authorize automatic repair with this diagnostic.

Official sources:
[medspaCy 1.3.1 ConText](https://github.com/medspacy/medspacy/blob/1.3.1/medspacy/context/context.py),
[CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler/tree/44ddeb363149aa657296237f18b5472a73c1756f).

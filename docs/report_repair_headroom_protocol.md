# Same-image report repair headroom / 同图报告可修复空间

## Role and frozen scope

This is a new descriptive, cached DEVELOPMENT diagnostic, not a new selection
or repair policy, clinical benchmark, or hidden change to existing winners.
It uses the complete existing 80 synthetic EHR / 240 CXR / 960 report inventory
and all four report experts. Freeze the predicate below before reading endpoint
values in this analysis. The cohort and previous aggregate endpoints were
already inspected; neither this predicate nor this cohort is held-out evidence.
No fitting, universal scalar weights, new thresholds, majority-vote truth,
training, downloads, API use, model calls or Slurm submissions.

Run only within an existing CPU Slurm allocation. Read cached synthetic labels,
metadata and hashes; do not open report bodies, image pixels, raw patient
inputs, source identifiers or real targets. New outputs are atomic and refuse
overwrite under `artifacts/protected/tricompose_v1_2/repair_headroom/` with
project-group access and 2770 directories / 0660 files.

## Unit and predicate

For each fixed CXR, enumerate all 4×3 directed baseline→alternative report
pairs: 2880 pairs across 240 images. Also project these opportunities onto
the four fixed report baselines and unchanged source-key baseline, giving
1200 baseline path slots. Alternatives are NEVER ranked or chosen by BioViL.
Require identical case, EHR and fact hashes/states/source categories, CXR
identity/hash/model/seed and classifier states. Keep four-state semantics;
only positive/negative are explicit. Weak priors are not hard constraints.

A strict label-preserving opportunity exists only if:

1. Both original artifact gates pass, and both image-validity checks pass.
2. Existing report-structure scores are available and the alternative is not
   lower. This is a cached engineering score, not guaranteed report quality.
3. Every IMAGE-POSITIVE finding already explicitly supported remains supported.
4. Every DIRECT EHR finding already supported remains supported.
5. Every previously comparable reference finding remains comparable, separately
   for EHR–report and CXR–report. Dropping a conflicting statement into unknown
   or uncertain is NOT a correction.
6. No NEW explicit opposition appears on either edge, at the fact-ID level.
   Lower total opposition cannot compensate for a new different contradiction.
7. At least one strictly better fact-set readout exists: additional supported
   image-positive/direct-EHR facts, additional comparable facts, or removal of
   an old opposition while preserving its comparison. Merely higher structure
   score, lower cost, fewer sentences or greater cosine is insufficient.

These are set-inclusion checks, not count-only or weighted objectives. Record
lost/gained fact IDs, added/removed oppositions, silence-caused removals and
quality/availability failures. Unknown classifier references remain unknown;
a report's positive assertion for an unknown image finding is tracked but is
not automatically a contradiction or proof of truth. Consequently passing is
NOT a safe clinical repair verdict. Preserve historical global No-Finding
adjustments only in the unchanged source selector; the diagnostic uses raw
four-state relations and does not expand global-normal language into invented
negative states. Use the historical 14-field profile, never mix it with the
fresh eight-enabled-head smoke.

## Endpoint measurement, not selection

Seal the complete label-only plan before attaching all authenticated 960
BioViL records. Keep directed pairs, baseline choices and admissible sets
identical after attachment. Output every directed pair, including failures,
plus per-image not-dominated sets under THIS diagnostic (not clinical Pareto
optimality) and all five baseline opportunity tables. Do not export a new best
triple or claim an alternative was generated on demand.

For predicate-passing alternatives report positive/zero/negative BioViL gaps
and a clearly CONDITIONAL mean over all such alternatives: average alternative
gaps within image, then opportunity-bearing images within EHR, then EHRs.
This is NOT a selected-output effect or a full-80-EHR treatment effect. Retain
all 80 cases in opportunity rates. Show opportunity-case/image/pair counts,
separate image generators and direct/no-direct EHR strata. A missing required
endpoint keeps that image/EHR conditional mean NA; partial available-case
means are labeled separately. Zero opportunities gives NA, not zero benefit.
The 2880 directed pairs are repeated observations, not independent patients.
No confidence/significance or actual GPU-savings claim; building the candidate
bank already incurred model cost. A positive cosine gap is not clinical truth.

## Interpretation and next gate

This asks whether EXISTING report alternatives contain conservative frozen-
label improvements without silencing evidence. It does not test whether new
seeds, prompts or a targeted repair call would succeed, identify a naturally
faulty modality, or prove clinical quality. No opportunity under this strict
diagnostic is not global optimality or impossibility of repair.

If label-preserving alternatives frequently lose secondary quality, record the
conflict instead of tuning to BioViL or silently weakening the predicate.
Keep original winners and rules fixed. A future online policy needs a separate
frozen protocol, independent confirmation and actual call/failure/GPU-time
accounting; any new model execution still requires the complete Slurm script,
resource request and explicit user approval.

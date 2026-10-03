# Automatic, no-human-feedback development track

The user cannot supply human clinical checks or feedback. That is **not** a
blocker for an exploratory automatic-policy experiment. It limits the claims:
optimize frozen proxy evidence, report independent/alternate automatic readouts,
and do not claim radiologist-verified clinical correctness or fault attribution.
Historical review packets and no-repair decision previews remain immutable.

## First experiment: equal-budget frozen-bank replay

Freeze `TriCompose-v1.2/configs/automatic_replay_v1.json` before observing replay
outcomes. Reuse the current complete six-CXR/four-report grid per fixed EHR.
Every synthetic EHR stays fixed; there is no replace-EHR action, new training,
threshold fitting, label enrichment or patient selection by score.

Compare four methods at 4/8/12/20/40/60 model-call caps:

1. Fixed Sana seed 0 + MAIRA-2, the historical predeclared path.
2. Random candidate order with all five predeclared seeds reported, not the best.
3. Static prefix best-of-K with the same registered model/seed order.
4. Targeted heuristic: direct EHR–classifier opposition requests another CXR;
   report opposition or missing known facts requests another report model;
   report exhaustion tries another image. Conflicts are **heuristic signals**,
   not confirmed modality errors. Shared-image report majority is never used.

Candidate ranking retains the existing V1.1 lexicographic order on observed
records, not their full-bank ranks: artifact gates, raw proxy opposition counts,
direct EHR support, report–classifier support, structure, known candidate runtime,
deterministic ID. There is no invented pooled metric weight.
The source V1.1 EHR–Report metric includes its pre-existing explicit global
No-Finding adjustment. It stays a labeled legacy proxy metric; replay does not
change cached unknown finding states to negative or call that adjustment truth.

Raw XRV/CheXbert states are explicitly unverified operational proxies. In
particular, a raw pneumonia label does not acquire a missing syntax-guard head.
Guarded comparisons retain that coverage gap in the separate end-point readout.
Unknown/uncertain never become negative, and low contradiction counts with low
coverage never mean clinically correct.

The targeted policy stops on the current candidate only when its artifact gates
pass, all known direct EHR facts agree with image/report proxies, all classifier
positive facts are explicitly represented, and there is some image/report
comparison evidence. This is **proxy stop satisfaction**, not clinical acceptance.
At budget/exploration exhaustion return the best eligible observed candidate with
its unresolved status; do not drop cases or fabricate successful repair.

## Cost and information boundaries

Per new image slot charge one generator plus one XRV invocation. Per new report
slot charge one generator plus one CheXbert invocation. Reuse the image within
one replay; do not recharge it for each report. These simulate single-case
invocations, not actual historical batching or measured GPU seconds. Shared fixed
EHR generation cost is the same sunk cost for every method. Failure/retry and
secondary end-point evaluation cost must be measured separately prospectively;
never report them as free or infer actual savings from CPU replay.

Model/seed slot inventories are visible to the controller. Other slots' scores,
old winner flags/ranks, intervention answer keys, BioViL-T and guarded end-point
readouts are not used to pick the next slot. BioViL-T is a different automatic
readout, **not** independent clinical truth; test overlap/correlation remains.
No method retuning after these pilot outcomes.

## Outputs and next gate

The protected run contains `method_comparison.csv`, `replay_outcomes.jsonl`,
`RESULTS_CN_EN.md`, `frozen_policy.json`, `summary.json`, and hash provenance.
The current two-EHR bank is a development smoke, not a population experiment;
random replicates do not increase the independent patient/case count.

Then test transfer on a separately frozen larger cohort and an alternative
automatic evaluator. If this heuristic helps only its own proxy, do not claim a
general repair advantage. Actual targeted generation requires a bounded execution
adapter, complete prospective cost/failure accounting, and a separately approved
Slurm script. Human feedback is not required to run those automatic experiments;
clinical claims must remain explicitly unvalidated without independent evidence.

## Completed development run

`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_replay_12576792_001/`
contains the first complete comparison: two fixed EHRs, 48 source candidates,
six budgets and five random seeds, yielding 96 replay trials. It ran within the
existing CPU allocation without new Slurm submissions, model execution or
source artifact text/pixels. Exact replay, source/artifact hashes, protected
permissions and original winner/table hashes passed. The 26 new invented-grid
tests and full 499-test local V1.2 suite pass. Detailed scores remain protected.

Two initial preparation attempts failed before output creation because of
legacy cache-contract differences. The adapter now preserves the source
EHR–Report global No-Finding metric without flipping cached unknown states, and
reads BioViL's status/calibration/raw-cosine object rather than assuming a
scalar. These were schema fixes, not post-result policy fitting. The policy
order, budgets, stopping logic and metrics were not tuned to the completed result.

## Larger historical cohort replay (completed)

The separate legacy adapter retains the full fixed 80-EHR cohort and its
960 triples (three CXR models × one seed × four report models per EHR).
Its explicit profile is historical **uncalibrated fourteen-head** XRV/CheXbert,
not the newer calibrated eight-head two-EHR profile. The routing logic/model
priority is reused, but numerical scores from the two profiles are not pooled
or presented as an identical-evaluator transfer result.

At call caps 4/8/12/20/30, all four methods and five random seeds yielded
3,200 replay trials. The exhaustive maximum is 30 simulated calls per EHR,
including image/report generation and XRV/CheXbert. Fixed EHRs, incomplete EHR
constraints, missing readouts and unresolved candidates are retained.

Protected output:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_replay_pool80_12576792_001/`.
The table is `method_comparison.csv`; EHR-evidence subgroups are separated in
`subgroup_comparison.csv`. Full replay, arithmetic, immutable source/winner
hashes and private permission checks passed. A separately versioned private
review is in `automatic_replay_pool80_review_12576792_001/RESULTS_CN_EN.md`.
This run made zero new model calls and no new Slurm submissions.

Prompt conditioning tiers and directly comparable cached finding states are
different definitions: the historical 15/65 conditioning split is not the
strict cached-label 8/72 split. The adapter preserves the latter as measured;
it does not add diagnoses/devices, reinterpret raw EHRs, or assume missing EHR
constraints are negative. Both definitions and denominators are retained in
the private review. No-direct-EHR edge rates remain NA. Uncomputed report
scope and BioViL readouts remain NA rather than zero.

The maximum-budget result is a proxy cost/quality tradeoff, not proof of a
general repair advantage. Full budget curves and subgroup readouts must be
reported; endpoint directions alone do not describe every budget. Historical
uncalibrated proxies are not clinical truth.

### Alternate endpoint evaluator (completed after approval)

`benchmarks/score_automatic_replay_biovil.py prepare` froze the deduplicated
union of all methods' selected candidates, without opening text/image bodies
or models. The private request is `automatic_biovil_request_12576792_001/`.
All candidate metadata hashes, parent/case/EHR lineage, frozen-run inventory
and private modes passed preflight against three CXR/eight report runs.

`slurm/28_automatic_pool80_biovil_debug_p100.sbatch` requests one P100,
two CPUs, 8 GiB RAM and ten minutes on debug. The complete script was shown,
explicitly approved, and completed as **12607645**, exit `0:0`, on `e23-02`.
Allocation elapsed 36 seconds; program runtime 33.322983 seconds; peak
allocated GPU memory 0.61 GiB; maximum resident host memory 1,388,880 KiB.
No checkpoint download is needed. The scorer pins the existing frozen weights,
uses full reports, and marks empty/overlength/unsupported-special-token reports
unavailable rather than silently truncating. Report bodies/pixels stay private;
stdout is status, runtime, memory and hashes only.

BioViL-T is excluded from routing and reranking. Compare it on paired available
EHR cases with availability/coverage retained; do not select the best BioViL
outcome, tune the policy afterward, or call cosine independent clinical truth.
The full local suite now has **537 passing invented-fixture tests**. Actual
prospective generation and complete failure/retry/verifier cost accounting
remain subsequent, separately approved experiments.

### Endpoint comparison and counterevidence

`src/tricompose_v12/automatic_secondary.py` and
`benchmarks/merge_automatic_secondary.py` attach the immutable endpoint scores
without changing selected IDs, action traces, fixed EHRs or proxy metrics.
They require exact request/record/hash inventories, complete predeclared random
seeds and paired available-case coverage. Missing context remains unavailable;
random endpoints average all five seeds within one EHR, never best-of-five.

Completed private tables/readable report:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_secondary_pool80_12607645_001/`.
Its `method_comparison.csv`, `subgroup_comparison.csv`,
`paired_case_comparison.csv`, `endpoint_outcomes.jsonl` and `RESULTS_CN_EN.md`
retain the full 80-EHR/3,200-trial comparison. All source/result hashes,
exact endpoint arithmetic, original winner/action/fixed-EHR invariants and
private modes passed. Slurm-created log files were restricted to mode 0660;
parent/output directories remain 2770 within the project boundary.

At the maximum budget the targeted heuristic's paired BioViL mean is lower
than fixed, random and exhaustive static selection. `random` is random-order
candidate acquisition plus the same score-based reranking, not a uniformly
random final choice. At the full-bank budget all five random seeds select the
static winner for every case. Those two method IDs are not independent endpoint
comparators at that budget. This is counterevidence
to a general improvement claim: improvements in the optimized label proxy do
not transfer to this alternate readout on the historical cohort. No thresholds,
weights, model order, cases or old winners were changed in response. Conversely,
global cosine is not clinical truth, so this does not alone establish a natural
clinical error or identify its modality.

Next, diagnose the discrepancy using frozen evidence/coverage and a separately
predeclared controlled fault experiment before promising targeted regeneration
quality gains. Keep these cohorts explicitly developmental; do not retune a
policy on this same endpoint and call it untouched evaluation. New GPU work
still requires its own complete script/resource presentation and approval.

### Completed frozen discrepancy diagnostic

`src/tricompose_v12/automatic_discrepancy.py` and
`benchmarks/diagnose_automatic_discrepancy.py` bind cached finding states and
source counters to the unchanged secondary outcomes. They inspect neither raw
patient inputs nor synthetic text/pixels and execute no model. The completed
protected result is
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_discrepancy_pool80_12605930_001/`.
It took 0.607697 seconds in the existing CPU allocation `12605930`, without a
new submission. Exact derived arithmetic, lineage, source/result hashes and
protected permissions passed; **552 invented-fixture tests pass**.

The 800 targeted-versus-fixed/static contrasts preserve all 80 EHRs and five
budgets. Positive/negative support and opposition are counted separately;
unknown/uncertain never become negative. Legacy No-Finding adjustments remain
named source counters, not edits to individual finding states. Ratios with
no evidence remain NA. Endpoint means require paired availability; raw support
sums retain their separately stated denominators.

At maximum budget versus fixed, 30 cases retain both artifacts, 12 change only
the report, and 38 change both image and report. Mean BioViL change is positive
in the report-only group and negative in the joint-change group; support gains
are larger for negative than positive agreements. Neither aggregate direction
shows clinical truth. In particular, the joint-change branch changes the
classifier reference as well as the report: it cannot establish that the image
alone was wrong or that the replacement repaired a fixed clinical target.
Negative agreement itself is valid and is not a reason to inject pathology.

The 532 observed common-image report pairs are from the endpoint-scored
selected union, not the entire 960-candidate pool. They cover 210 case/image
groups and 80 EHR cases. Report pairs share images/EHRs and are correlated;
case-level means, not pair counts as independent patients, are the readout.
The first differing source-ranking component is described without changing
the winner. Runtime-decided ties sometimes contain semantically different
reports, motivating a control rather than causal model-performance claims.

### Preregistered next controls (not executed or authorized by this diagnostic)

1. **Fixed-image report-only selection versus joint search.** Retain all EHRs,
   fix the image before either policy sees report scores, predeclare budgets,
   missingness and subgroup readouts, and keep the alternate evaluator out of
   selection. Current selected report-only cases are observational, not a
   randomized control or a general improvement result.
2. **Invariant evidence across image switches.** Track unchanged direct EHR
   constraints separately from the new image's own classifier labels. A new
   easier reference or an unavailable EHR edge must not certify repair. Use
   existing controlled fault diagnostics as prior evidence, not as proof of
   natural error localization or a newly independent benchmark.
3. **Runtime tie-break control.** Freeze a new comparison without the runtime
   tie-break and report quality/cost separately. Do not fit endpoint weights or
   disease priorities on these already inspected 80 cases.
4. **Score-free random final-choice control.** Distinguish this prospective
   baseline from the frozen random-acquisition-plus-reranking method. Preserve
   the old method IDs, tables and full-budget equality rather than redefining
   them after seeing results.

For confirmatory claims, freeze the policy and a separate evaluation cohort
before running those controls. Record shared sunk cost, actual calls, verifier
overhead, retries/failures and endpoint evaluation cost separately. No clinical
acceptance, physical regeneration, rule fitting, cohort filtering or new GPU
execution is enabled by this read-only diagnostic.

### Fixed-image report control (completed exploratory comparison)

The first control above has now been executed under the subsequent user request.
Its image slot, report order, caps, ranking and missingness contract were frozen
in `TriCompose-v1.2/configs/fixed_image_pool80_v1.json` before this control ran;
the cohort was already inspected, so this is NOT untouched validation or a
retrospective preregistration. The distinct protocol is
[fixed_image_control_protocol.md](fixed_image_control_protocol.md).

Protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/fixed_image_control_pool80_12605930_001/`.
All 80 EHRs retain their original Sana seed 0 image for new static/targeted
report-only controls. Their 800 trials plus 1,200 reused fixed/joint trials
completed in 0.909379 seconds in existing CPU allocation `12605930`, with zero
new model calls, submissions, patient reads or text/image opens. The original
rank order, original actions/winners and EHRs are unchanged. All new selected
endpoints already exist in the frozen scored union; pending endpoint count is
zero. Hash/permission/exact-replay/old-source checks passed; **571 tests pass**.

At the maximum cap the same-image mean BioViL readout is higher, but seven
cases improve versus fixed, nine decline and 64 tie. Report the full paired
counts and subgroup/budget curves, not only a favorable mean. Global cosine
does not certify clinical truth, significance, actual repair or causality.

Report-only targeted and static select the exact same final outputs in all
80 cases. Targeted blocks an image-change request in one case and exhausts
four reports in 79; no report-only case reaches proxy-stop satisfaction.
Its tiny simulated invocation reduction therefore is not evidence of an
effective adaptive stop or a quality gain over static report selection.
Four report generations/CheXbert plus one image generation/XRV cost ten
simulated calls, so equal budget caps must not be presented as equal actual
search effort or actual GPU savings. Existing generation costs remain sunk.

Continue with invariant image-switch evidence and runtime tie-break controls
as separate, frozen development experiments. Do not revise old winners,
scorer thresholds or model priorities to inflate these readouts. Confirmatory
evaluation still needs a separately frozen cohort/policy; actual new GPU/model
execution requires its own complete script and approval.

### Ranking and invariant-switch controls (completed development diagnostics)

The subsequent user request authorized these CPU-only controls. Their separate
configuration/protocol was frozen before this run; the cohort remains already
inspected DEVELOPMENT, not independently preregistered confirmation. See
[ranking_switch_control_protocol.md](ranking_switch_control_protocol.md).

Protected output:
`artifacts/protected/tricompose_v1_2/automatic_replays/ranking_switch_controls_pool80_12605930_001/`.
All 80 cases, 2,000 runtime-ablation trials and 800 joint-versus-fixed invariant
contrasts completed in 1.787128 seconds inside existing allocation `12605930`.
No patient inputs, report bodies/pixels, new model, GPU, API or Slurm submission
were accessed. Exact regenerated rows, original action/cost/stop histories,
source/result hashes, original artifacts and protected modes passed;
**591 invented-fixture tests pass**.

Runtime removal preserves the exact earlier five-component ranking prefix,
candidate-ID tie-break and original observed candidate history. It retains
original proxy-stop choices rather than executing new actions. At maximum cap
it changes one report-only winner and ten joint winners (including nine image
changes), with negative mean BioViL differences. Do not claim a general runtime
benefit or remove this component to chase a favorable endpoint. The comparison
tests final ranking only, not runtime preference's effect on future routing.
All maximum-cap pairs are scored; one unique lower-cap choice lacks a frozen
endpoint and stays NA. Its pending metadata is not execution authorization.

The targeted policy changes 38 images at maximum cap. Thirty-three of these
cases lack an explicit comparable cached EHR finding; five have such constraints.
Only one of those five gains fixed-EHR image-label support, with no gain in the
all-three support count. Image/report support gains mostly occur outside the
unchanged direct EHR constraints. This diagnoses missing evidence for attributing
repair, not clinical image failure or absence of all EHR information. The earlier
15/65 prompt tiers and 8/72 direct-label coverage remain different definitions.
No context is promoted to a hard finding and no underconditioned EHR is removed.

The next image-switch interface should retain an explicit invariant reference,
evidence/coverage status and unresolved outcome rather than accepting a new
image merely because its own classifier agrees with its generated report.
This is a design implication, not an implemented/tuned new clinical gate.
Freeze any revision separately, keep this inspected cohort developmental, and
use separate confirmatory cases before an improvement claim. Actual generation
or missing-endpoint GPU scoring still needs complete-script approval.

### Invariant verification hook (cache integration completed)

The subsequent request now has a reusable per-generation evidence interface,
not only an aggregate image-switch statistic. An immutable `EHRAnchor` binds
the fixed EHR hashes, four-state vector and cached source categories.
`verify_candidate` returns a content-hash-bound receipt with three edge
readouts/coverage; `verify_transition` records specific fixed-EHR support and
opposition changes while retaining missingness and clinical ambiguity.
No scalar score, new reference, correlated report vote or source No-Finding
adjustment can manufacture raw support or repair success. See
[invariant_verification_interface.md](invariant_verification_interface.md).

The adapter streams ORIGINAL requested slots from the full registered
case/method/budget/seed inventory; no action, selected triple, call ledger or
terminal reason is changed. Protected output is
`artifacts/protected/tricompose_v1_2/automatic_replays/invariant_verification_pool80_12605930_001/`.
All 80 anchors, 960 candidate receipts, 3,200 original trials, 15,001 observation
events and 11,801 transitions completed in 6.452556 seconds in the existing CPU
allocation, with zero new model, GPU, patient-source read, text/image open or
submission. Exact regeneration, source/result hashes, old artifacts, all
winner/action/cost/stop invariants and private modes passed. **616 tests pass**.

This is integration with cached histories, not new real-time generator
execution or a revised policy. The legacy adapter accepts completed triples;
image-only partial receipts and hooks in prospective inference adapters are
not implemented here. The eight-head scoped preview remains unchanged and
is not pooled with this explicitly uncalibrated fourteen-head profile.
All clinical acceptance, confirmed fault and clinical repair-success flags
remain false/unavailable. No-direct-EHR rates remain NA, not failure or normal.

Next, freeze a separate prospective execution policy, include partial-result,
failure/retry/verifier costs, and connect the receipt hook to that approved
bounded run. No human feedback is required for exploratory automatic execution,
but the interface's successful integrity tests do not supply clinical truth or
independent evaluation. Every new GPU/model job still needs its complete script,
resource request and explicit approval.

### CXR-before-report receipts (cache integration completed)

The partial interface now supports the previously missing image/scorer phase
without requiring any report fields. See
[partial_image_verification_contract.md](partial_image_verification_contract.md).
It binds the same fixed EHR anchor, image hashes and cached XRV states, and
leaves report edges/identity/hash/labels and all-three support `null`, with an
explicit `not_generated` lifecycle. A generated report with unknown labels is
a different state. Unknown/uncertain and no-direct-EHR comparisons remain NA.

Each completed report links to an unchanged partial receipt; image identity,
hash, classifier vector and EHR–CXR evidence cannot change during this link.
Report labels, scores, structure, triple gates, votes and runtime cannot affect
the report-blind image projection. All clinical acceptance, fault, repair and
model-authorization flags remain false/unavailable. Canonical JSON integrity
checks distinguish false/zero and integer/float representations.

Protected validated output:
`artifacts/protected/tricompose_v1_2/automatic_replays/partial_image_verification_pool80_12605930_002/`.
All 80 immutable EHRs, 240 partial image receipts and 960 completed bindings
completed in 1.658559 seconds in the existing CPU allocation. Exact repeated
reconstruction, source/result hashes, original full-run and underlying EHR
anchors, private modes and no-overwrite checks pass; **637 tests pass**.
No model, GPU, API, new submission, raw source, report text or image pixels
were accessed. The earlier full verification run was not modified.

This remains DEVELOPMENT cache reconstruction, explicitly not actual execution
order, automatic rejection, skipped report calls or GPU savings. The interface
supports only the separately named legacy fourteen-head cache; the existing
eight-head scoped preview is not changed or pooled. Next is a separately frozen
prospective execution/cost ledger and adapter integration; new GPU inference
still requires complete-script/resources and explicit approval.

### Bounded execution layer (invented smoke completed; model backend pending)

`execution_ledger.py` now accounts for individual generator/scorer attempts,
failure/retry budget, committed-result reuse, phase dependencies and fixed
artifact/EHR lineage. `runtime_dispatch.py` guards dispatch behind Slurm,
requires matching backend audit/mode declarations, journals before invocation,
and captures Python output in protected logs. Journals are hash-chained and
fsynced, restore against an unchanged budget/retry/mode contract, and preserve
in-flight charges without blind reruns. See
[bounded_execution_ledger_contract.md](bounded_execution_ledger_contract.md).

Protected test:
`artifacts/protected/tricompose_v1_2/execution_smokes/bounded_execution_fixture_12605930_001/`.
TWO invented IDs exercise the partial/full verification hooks, one scorer
timeout/retry, image/scorer reuse and a budget-limited unscored second report.
Ten fixture reservations correspond to ZERO actual model calls or generation
outputs. Exact durable journal restoration, source/result hashes, original
full/partial run hashes, protected modes and no-overwrite checks pass;
**676 tests pass**. Runtime is 0.047964 seconds within the existing CPU allocation.

This is not rerunning the 80 cases or clinical/generation evaluation. Only the
invented backend is installed; authentic deployed-worker/provenance adapters
remain next. No raw patient source, actual report body/image pixels, model,
checkpoint, GPU, API, download, training or new Slurm job was accessed. The
ledger grants neither clinical acceptance nor model-execution approval. A real
worker must separately verify authentic hashes/freezing/scorer scope, capture
subprocess/native output and retain failed/in-flight journals for reconciliation.
Every actual inference submission still needs the full script/resources and
explicit approval; no human clinical feedback is needed for the engineering
tests and none is fabricated.

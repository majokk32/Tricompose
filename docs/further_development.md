# TriCompose: further development

Status: research plan, not a validated result. This document refines the V1.2
protocol in [v1_2_localization_protocol.md](v1_2_localization_protocol.md).
The method is training-free. All generators and evaluators remain frozen;
synthetic EHR is fixed within a case. CXR generation currently uses
EHR-derived radiology text, **not** a direct structured-EHR-to-CXR checkpoint.

## Research question and claim boundary

Repeatedly generating until a scalar score rises is best-of-N search. The
research question is whether **scope-aware evidence and calibrated uncertainty
can localize an erroneous modality, choose a targeted repair, and abstain when
the evidence is insufficient**, improving quality per actual GPU-second over
static reranking. If it cannot, the defensible result is static composition,
not an "Agent" claim.

## Two-level score table

The protected fact table has one row per case, candidate triple, and clinical
finding. It records `ehr_state`, `ehr_evidence_id`, direct-vs-weak provenance,
`cxr_state` with scorer version and calibration status, each report's state,
`observable_in_cxr`, pairwise support/contradiction/unknown/not-comparable,
clinical severity, and dependency lineage. Unknown is never negative. An
EHR diagnosis, medication or lab not visible on CXR is not an image finding.
Multiple reports derived from one CXR are correlated, not independent votes.
XRV has no device head, so missing device evidence cannot be scored negative.

The protected candidate/action table records validity gates, direct EHR
support, hard contradictions, CXR-report factual support, comparable-fact
coverage, evaluator disagreement, single-modality quality/diversity, full
generation and verification cost, action, and triggering evidence IDs. Report
each edge separately. The present eight-case table has too little direct EHR
evidence to justify one pooled clinical score as a paper result.

Selection is lexicographic until thresholds are independently calibrated:
artifact validity; explicit hard contradictions; support of directly evidenced
facts; image-report factuality; coverage and uncertainty; quality; then cost.
Do not invent a universal weight for incompatible score scales. Selection
evidence and final independent evaluation must be separated.

## Deterministic action policy to test, not assume

Allowed actions are `select`, `switch_report_model`, `regenerate_report`,
`regenerate_cxr`, `verify_more`, and `reject`. There is no `replace_ehr` action.
An invalid image calls for image regeneration. A supported image fact that one
report explicitly denies calls for a report alternative. Two reports on the
same image disagreeing calls for more evidence, not immediate image repair.
Direct EHR-image conflict can motivate image repair only after calibrated,
sufficiently independent verification. Sparse or conflicting evidence calls
for verification or abstention. Every action is bounded by predeclared model
calls and actual GPU time. Development-set action-success statistics may guide
expected error reduction per unit cost without training a router.

## Required first benchmark: real matched-data scorer validation

Before interpreting synthetic-candidate scores or claiming localization, test
the frozen evaluators on a protected, patient-disjoint real matched cohort.
This is a prerequisite, not a later optional analysis. Use only EHR information
available **before** the CXR study; exclude previous/final radiology reports,
target image labels, and post-CXR diagnoses from the EHR-side input. Keep the
correctly matched image and report as a positive pair. Construct cross-patient
random negatives and independently checked same-disease hard negatives;
random swaps can remain clinically compatible and must not be assumed wrong.

Validate each edge separately:

1. Real CXR--report: does the frozen metric rank a matched pair above its
   controlled mismatches? Report AUROC/AUPRC, within-case pairwise ranking,
   sensitivity by finding, contradiction detection, coverage and uncertainty.
2. Pre-CXR EHR--CXR and EHR--report: compare only directly evidenced,
   radiographically observable facts. Record positive, explicit negative,
   uncertain and unknown separately; never treat missing EHR findings as
   negative. Report comparable-case coverage as well as discrimination.
3. Freeze operating points on a calibration split and evaluate once on a
   separate final split. Do not call XRV operating-point-normalized values
   probabilities or report Brier/ECE for them. Inspect clinically hard errors
   through an approved protected review process, not raw-data chat/log output.

No reviewed matched-real calibration bundle was found in the current protected
audit. The eight-case synthetic smoke below therefore checks plumbing and
cached-label sensitivity only. It does not clear this real-data gate. Any job
that reads real patient-level inputs must run through an approved Slurm script
with sanitized public output.

## Following benchmark: controlled synthetic intervention and adjudication

The first CPU-only artifact stage uses a fixed, predeclared path in the
accessible eight-case synthetic candidate bank. For each case it creates an
unaltered control, a cross-case report swap with the CXR held fixed, and a
cross-case CXR swap with the report held fixed. It retains immutable candidate
IDs and hashes; it neither copies images/reports nor edits existing runs.
Policy-facing item IDs are separated from a protected resolver and intervention
key. A swap's target modality is a **mechanical intervention label**, not proof
of a clinical mismatch: accidental matches and already-wrong controls require
independent adjudication. Only adjudicated items may contribute to clinical
localization accuracy. This eight-case stage is a plumbing smoke, not a CVPR
result. Same-disease, minimal negation, laterality, severity, device and
temporal corruptions follow only where independent evidence supports them.

After the real-data scorer gate and V1.1 calibration review, run frozen
evaluators on the rewired
artifact pairs through approved Slurm jobs; do not merely flip cached label
vectors and call it a clinical benchmark. Build patient-disjoint development,
threshold-calibration, and final-test splits before scaling.

## Primary comparisons and success criteria

At equal actual computation budgets compare operational fixed path, random
retry, all fixed paths, best-of-K/static reranking, and targeted repair.
Exhaustive reranking is a quality upper bound at its full cost. Offline replay
must report simulated cost separately from GPU time already spent building
the bank. Measure per-class localization, false repair on clean controls,
abstention/coverage, repair success, contradictions on all three comparable
edges, image/report quality, diversity, rejection, failed calls and GPU time.
Final quality needs a disjoint frozen evaluator and blinded clinical review;
otherwise the controller may simply optimize its own scorer. Ablate
localization, uncertainty, abstention, targeted repair, and cost awareness.

The proposed novelty is fault localization and targeted repair with evidence
scope, dependency awareness, and honest abstention—not a generic LLM Agent or
an unconditional generate-score-repeat loop. It remains a hypothesis until
the controlled benchmark and equal-budget prospective test succeed.

## Current report-extraction validation handoff

The completed authored-language diagnostics and the new generated-report
human-review contract are documented in
[report assertion status](report_assertion_status_20261002.md). The protected
48-report pilot packet hides model/score/winner metadata and retains four-state
evidence plus missing-review denominators. It has no completed human labels,
and protected anonymous text copies completed after explicit user approval.
This two-EHR development review
is neither an untouched final cohort nor a validated localization benchmark.
Do not substitute parser agreement or assistant-authored labels for independent
human evidence, or activate targeted repair merely because the packet exists.

The review packet now also has a protected anonymous reading book, two blank
192-row CSV forms, and a quote-bound importer. Blank imports/readiness and
deterministic replay are verified; **356 invented-fixture tests pass**. No human
labels have returned, so human-reference accuracy/kappa remain unavailable.
See the handoff's CSV workflow for protected copy/import/audit commands.
The next evidence-producing step requires independent people to annotate;
additional interface checks or model-generated labels cannot clear that gate.

A separate candidate-level **scope-availability diagnostic** now retains all
384 facts (48 candidates × eight findings), and compares raw versus source-
bound report proposals without changing selection. See the handoff's candidate
scope table section. Only four findings have a frozen syntax-checking head;
the two cases' direct pneumonia facts therefore yield no scope-usable
EHR–Report comparisons. This is a checker coverage limitation, not a proven
generator failure. Current four-finding human forms cannot establish accuracy
for that missing edge. Expanding independent annotation coverage needs a
separately frozen protocol; do not retune rules on this bank or mistake lost
comparison coverage for improved consistency. **379 tests pass**, but human
labels, clinical fault attribution and regeneration authorization remain absent.

### Next available real-report benchmark source

The approved metadata audit of the local 14-category manual MIMIC-CXR-JPG
report-label source completed and replayed exactly inside the existing CPU
allocation. Protected artifact hashes, four-state denominators, study dedup,
patient-disjoint linkage and permissions passed. Sanitized readiness is
`linked_test_gold_metadata_available`, not verified report-file availability.
See the [official-label audit handoff](report_assertion_status_20261002.md#official-human-label-source-metadata-audit-completed--官方标签元数据审核完成).
The source uses **Airspace Opacity**; it is retained without relabeling it to
the common Lung Opacity head. Pneumonia remains in the source inventory.
**398 V1.2 tests pass**, including 19 invented-CSV tests. This audit did not
consume report text or images, run a model, fit thresholds or measure extraction
accuracy. Checkpoint training overlap and annotation-policy equivalence remain
unverified; do not call it an independent final test, fit a policy to official
test labels, or substitute it for image adjudication/generated-report human
review. The next real-report/model benchmark requires separately approved
execution. No clinical selection or repair is enabled by the metadata audit.

The report-label diagnostic completed as **job 12594397**, after the complete
`slurm/26_official_report_chexbert_debug_p100.sbatch` was shown and separately
approved: one P100, two CPUs, 16 GiB RAM and a 10-minute cap. Slurm exit was
`0:0`, allocation elapsed 15 seconds; program runtime was 12.025 seconds and
peak allocated VRAM 1.227 GiB. Nonempty predictions, artifact hashes, inventory
denominators and protected permissions passed the post-run check. It froze conservative
Impression-only input selection, every unavailable denominator, raw four-state
metrics, source-name/non-four-state exclusions, and the unchanged four-finding
scope diagnostic. See the [predeclared benchmark protocol and completion](report_assertion_status_20261002.md#real-report-chexbert-benchmark-completed--真实报告评测完成).
Results remain below `artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_reports_12594397/`.
The protected `official_chexbert_review_12594397_001/RESULTS_CN_EN.md` in the
same parent presents coverage, per-finding/state metrics and interpretation;
its receipt verifies confusion-derived statistics and unchanged old winners.
No training or scorer/rule tuning occurred. **421 tests pass**. This is a
completed real-report text-label diagnostic, not independent image factuality,
EHR-edge validation or a targeted-repair authorization. Training overlap and
section-policy equivalence remain unverified; no old score table or winner changed.
The derived synthetic candidate overlay is available in
`artifacts/protected/tricompose_v1_2/diagnostics/candidate_report_diagnostic_12594397_001/`.
It joins 384 finding rows across 48 candidates/two EHR cases to raw-head
diagnostics and scope-availability notes, not selection weights. No new model
calls, clinical labels, ranking or repair claims are introduced. Candidate
checkpoint/section equivalence and diagnostic transfer remain unverified.

### Following diagnostic: single-fact CXR/text polarity

The existing matched/random and shared-finding hard-negative BioViL-T retrieval
pilots are complete, so the next test does not repeat them. The prepared
`real_validation/biovil_fact_polarity.py` holds the same real CXR fixed and
scores three predeclared authored presence/absence pairs for each of eight
findings. It measures polarity and balanced polarity wins separately from
margin AUROC; a high retrieval score cannot prove negation sensitivity.
See the [frozen probe protocol](biovil_fact_polarity_protocol.md).

It retains all fixed cases and unknown/uncertain reference denominators, reads
no new raw report/EHR text, and uses cached report-derived labels only after
scoring. No independent image truth, clinical threshold or repair gate is
claimed. **440 invented-fixture tests pass**, including 19 probe tests.
The complete P100 debug script `slurm/27_real_biovil_fact_polarity_debug_p100.sbatch`
was shown and explicitly approved, then completed as **12597637** on `e23-02`,
exit `0:0`, allocation 49 seconds; program 45.827 seconds, peak allocated VRAM
0.576 GiB including loading. Derived statistics replay exactly and source/
artifact/permission/old-winner checks passed. Protected interpretation:
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_polarity_review_12597637_001/RESULTS_CN_EN.md`.
Finding/template-sensitive polarity performance supports keeping global cosine
secondary, not making it the sole clinical repair judge. No existing winner,
threshold, template choice, scorer weight or repair authorization changed;
independent image truth remains unavailable.

## No-human automatic development track (2026-10-02)

The user cannot provide human checks/feedback. Exploratory automatic policy
development therefore continues under an explicitly limited claim, rather than
making every candidate wait for review. The clinical evidence gap remains an
evidence gap; it is not filled by invented people, consensus or perfect scores.
See [the automatic no-human protocol](automatic_no_human_protocol.md).

The fixed two-EHR/48-candidate bank now has an equal-call-budget cache replay:
fixed, five-seed random, static prefix reranking and deterministic targeted
heuristic. Routing uses existing raw proxy evidence and observed candidates
only; alternate automatic readouts are excluded from routing. Generator and
XRV/CheXbert calls are charged, shared image work is reused, and prospective
GPU time remains unavailable. All 96 development trials replay exactly; original
tables/winners stay immutable. **499 invented-fixture tests pass.** Private
results are in `artifacts/protected/tricompose_v1_2/automatic_replays/automatic_replay_12576792_001/`.

The next automatic experiment should test a separately frozen larger cohort
and alternative automatic evidence, then actual budget-bounded generation and
verification through explicitly approved Slurm jobs. Human feedback is not
required for those exploratory experiments. Without independent evidence,
do not claim clinical localization/repair accuracy, and do not turn replay
invocation savings into actual GPU savings.

### Larger automatic comparison now completed

The complete historical 80-EHR/960-triple pool now has a separate, explicitly
uncalibrated fourteen-head replay profile and 3,200 budget/seed trials.
All EHRs remain fixed; original winners remain unchanged. Source artifacts,
exact actions, budget/summary arithmetic and protected modes passed validation.
Private tables/review are under `artifacts/protected/tricompose_v1_2/automatic_replays/`.
The strict cached EHR-label coverage and earlier prompt-tier coverage are kept
as different definitions; no missing fact, scorer or guard is invented.

Next, a frozen deduplicated selected-union request allows a BioViL-T endpoint
readout that never drives routing/reranking. After complete script/resource
presentation and explicit approval, the P100 debug batch completed as
**12607645**, exit `0:0`, in 36 seconds (program 33.322983 seconds, peak GPU
allocation 0.61 GiB). Preserve missing full-text context scores
as unavailable. Only after this alternate readout should the no-human track
advance to actual bounded regeneration with prospective cost accounting.

The completed immutable overlay and paired per-EHR tables are in
`automatic_secondary_pool80_12607645_001/` under the same protected root.
The maximum-budget targeted policy has a lower paired BioViL mean than
fixed/random/static selection, despite its optimized-proxy improvement.
This exposes the need for discrepancy diagnostics and a predeclared controlled
fault benchmark before a general repair claim. No policy fitting, case changes
or winner changes were made on this result. The local suite now has **537
passing invented-fixture tests**. Global cosine still cannot certify clinical
truth; new GPU experiments remain subject to separate complete-script approval.

### Frozen discrepancy diagnostic completed

The new CPU-only diagnostic reuses synthetic cached states, hashes and the
unchanged selected endpoints. Protected report/tables:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_discrepancy_pool80_12605930_001/`.
It produced 800 contrasts and 532 observed common-image report pairs in
0.607697 seconds within the existing CPU allocation, with no new model, GPU,
patient-source read, text/image open or submission. Source/result hashes,
exact arithmetic, fixed-EHR/old-winner invariants and private modes passed.
The local suite now has **552 passing invented-fixture tests**.

At maximum budget versus fixed, 30 EHR cases keep both artifacts, 12 change
only reports and 38 change both image/report. Report-only changes have a
positive mean BioViL difference; the aggregate decline is concentrated in joint
changes. Negative support gains exceed positive gains, but both exist. This
does not make normal agreement invalid or authorize adding diseases/removing
cases. Switching the image also switches classifier reference labels, so
proxy gains cannot establish a repaired invariant target or identify a faulty
modality. Common-image pairs are correlated and selected-union observations,
not independent clinical ground truth.

Baseline clarification: the frozen `random` method randomizes acquisition and
then applies the SAME score-based reranking. At the full-bank budget every
seed/case selects the static winner. It is not score-free random final selection;
its endpoint equality is expected and not an independent replication.

The next separately frozen controls should compare report-only with joint
selection, preserve invariant EHR evidence across image switches, ablate the
runtime tie-break without endpoint weight fitting, and distinguish a pure
random final-choice baseline. See the
[preregistered control boundaries](automatic_no_human_protocol.md#preregistered-next-controls-not-executed-or-authorized-by-this-diagnostic).
Do not retune on this developmental cohort and present it as untouched
evaluation. Existing fault diagnostics remain prior diagnostics; no new
physical repair or clinically validated improvement has been demonstrated.

### Fixed-image report control now completed

The original Sana seed 0 image was fixed independently of report/endpoint
scores for all 80 EHRs. Two new report-only controls retain the original ranking
and heuristic signals; any requested image change is explicitly blocked.
The already inspected cohort remains DEVELOPMENT. Control configuration and
[protocol](fixed_image_control_protocol.md) were frozen before execution;
this is not new independent validation or a revised scoring policy.

Protected output:
`artifacts/protected/tricompose_v1_2/automatic_replays/fixed_image_control_pool80_12605930_001/`.
Its 800 new report-only trials and 1,200 reused deterministic baselines completed
in 0.909379 seconds in the existing CPU allocation, with no model/GPU/submission
or report/image opens. All choices have existing BioViL endpoints, so no scoring
job is pending. Exact replay, source/result hashes, old artifacts and project
permissions passed; **571 invented-fixture tests pass**.

The maximum-cap same-image mean BioViL is higher than fixed/joint means, but
versus fixed the paired counts are seven improved, nine declined and 64 tied.
Do not claim general casewise or clinical improvement. Report-only targeted
and static select the same outputs for all 80 EHRs; targeted exhausts four
reports in 79 cases and blocks an image change in one. There is no demonstrated
dynamic quality gain or successful proxy stopping in this control. Its costs
are simulated invocations, not prospective GPU savings.

The next development controls concern invariant evidence for image switches
and runtime ties; no endpoint weight fitting, EHR enrichment or filtering is
justified by these results. Freeze any revised policy separately before its
own comparison and reserve a new cohort for confirmatory claims. New GPU
execution remains subject to complete-script/resource approval.

### Runtime ranking and invariant image-switch controls completed

The separate frozen development controls retain original observed candidates,
actions/costs/stop reasons and fixed EHRs. Runtime removal changes final tie
resolution only; original proxy-stop winners stay fixed. The invariant image
diagnostic separates support on unchanged explicit EHR constraints from
image/report agreement with no such constraint. Neither control mutates old
rules, generates an artifact or confers clinical acceptance.

Protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/ranking_switch_controls_pool80_12605930_001/`.
Its 2,000 ranking trials and 800 invariant contrasts complete in 1.787128
seconds in the existing CPU allocation, with zero new model/GPU/submission or
raw/text/image opens. Exact replay, source/result hashes, original histories/
artifacts and private permissions passed; **591 tests pass**. One lower-cap
new choice has no cached endpoint and remains NA; all maximum-cap comparisons
retain the full 80 paired cases.

At maximum cap, removing runtime changes one report-only winner and ten joint
winners, with negative mean BioViL differences; it does not resolve the earlier
discrepancy. Of 38 targeted image changes, 33 lack an explicit comparable EHR
finding and five have one; only one gains fixed-EHR image support, while the
all-three support count does not increase. Extra image/report agreement largely
comes from findings outside direct EHR constraints. This is missing evidence,
not a natural clinical-error verdict or justification to enrich/drop EHRs.

Next, design invariant-evidence-aware image-switch verification separately
from report selection. Preserve four-state semantics, coverage and unresolved
outcomes; a new classifier reference alone cannot establish repaired EHR
fidelity. Do not fit that revision on these inspected endpoint outcomes and
present it as untouched evaluation. Independent confirmation and actual
regeneration cost remain future, separately approved work.

### Reusable invariant verification hook completed

The new `invariant_verification.py` connects a fixed cached-EHR anchor to
per-candidate receipts and before/after generation transitions. It retains
four-state coverage and specific support/opposition changes, rather than
inferring clinical repair from a weighted scalar. Source/provenance hash changes,
tampered counters, invented scope and forged acceptance are refused.
See [the interface contract](invariant_verification_interface.md).

Completed protected engineering integration:
`artifacts/protected/tricompose_v1_2/automatic_replays/invariant_verification_pool80_12605930_001/`.
All 80 fixed anchors, 960 completed-candidate receipts, 3,200 original trials,
15,001 observations and 11,801 transitions are bound without changing old
requests, winners, costs or stop reasons. Runtime is 6.452556 seconds inside the
existing CPU allocation, with no new model/GPU/submission or raw/text/image
opens. Exact replay, source/result/old-artifact hashes and protected modes pass;
**616 invented-fixture tests pass**.

The hooks are exercised on cached histories, not installed into newly executed
generation jobs. This legacy completed-triple adapter does not yet provide an
image-only partial receipt. Neither raw label agreement nor a receipt grants
clinical acceptance, model execution or repaired-generation accuracy. Keep the
existing eight-head scoped preview and fourteen-head legacy adapter distinct.

Next integrate partial/prospective results with a separately frozen bounded
execution policy and complete call/failure/time accounting. Preserve the fixed
EHR and all unresolved cases, keep alternate endpoints out of routing, and
freeze confirmation boundaries before inference. New GPU/model work requires
complete-script/resource approval; this engineering success alone cannot justify
clinical or independent quality-improvement claims.

### Partial image-phase verification completed; prospective integration next

The new `partial_image_verification.py` adds report-blind image/scorer receipts
to the earlier immutable full-triple interface. A report that has not been
generated has null edges, identity/hash/labels and all-three support, rather
than an all-unknown or negative label vector. Completed reports are linked to
unchanged image receipts with fixed EHR and image/classifier hash/state checks.
See [the phase contract](partial_image_verification_contract.md).

Validated cache output:
`artifacts/protected/tricompose_v1_2/automatic_replays/partial_image_verification_pool80_12605930_002/`.
All 80 EHR anchors, 240 image receipts and 960 report bindings completed in
1.658559 seconds within the existing CPU allocation. Exact reconstruction,
source/result hashes, unchanged original full run/underlying anchors, private
modes and no-overwrite checks pass; **637 invented-fixture tests pass**.
No new model/GPU/API/job or raw/report-body/image-pixel access occurred.

These are retrospective cache links, not newly executed generation hooks or
clinical verification. Nine image receipts agree on explicit EHR labels, six
oppose, nine lack image comparison, and 216 have no direct EHR constraint; none
of these counts certifies success/failure. All cases stay. Raw fourteen-head
legacy evidence remains uncalibrated; the separate eight-head preview stays
unchanged. No EHR enrichment, scalar-based success or report-vote truth is used.

Next implement a prospective, bounded call/failure/retry ledger and connect
the appropriate provenance adapter before inference. Preserve partial/unresolved
receipts, frozen EHR and models, and the original evaluation boundaries. A cache
receipt cannot authorize a GPU job or justify a clinical-improvement claim;
all actual new inference still needs full-script/resource approval.

### Bounded execution/cost layer completed; actual worker adapters next

The serial execution ledger and Slurm-guarded dispatcher now support separate
generator/scorer attempt charges, fsynced reservations before invocation,
failure/retry budgets, explicit successful-result reuse, fixed lineage/phase
dependencies and canonical journal restoration. Budget/retry/mode changes and
blind in-flight retries are refused. See
[the execution contract](bounded_execution_ledger_contract.md).

Protected engineering test:
`artifacts/protected/tricompose_v1_2/execution_smokes/bounded_execution_fixture_12605930_001/`.
Two INVENTED IDs, ten fixture attempts, one scorer timeout/retry, and one second
report left unscored at the budget exercise the partial/full verification hooks.
There are ZERO real model calls/generation outputs. Exact journals/costs,
source/result hashes, original full/partial runs, private modes and no-overwrite
checks pass; **676 tests pass**. Runtime is 0.047964 seconds inside the existing
CPU allocation. No patient source, actual model/checkpoint, report body/image
pixels, GPU, API, download, training or new submission is involved.

This is execution/accounting scaffolding, not a new scoring policy or a deployed
inference backend. Authentic already-installed model workers and fresh scorer
provenance still need adapters, native/subprocess output capture and durable
failure reconciliation. Do not pass fresh results off as the historical legacy
cache or infer clinical truth from a receipt/audit hash. The next small-cohort
inference script/resource request must be shown and explicitly approved; no job
is authorized by this fixture. Existing models, EHRs, rules and winners stay fixed.

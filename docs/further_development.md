# TriCompose: further development

Status: research plan, not a validated result. This document refines the V1.2
protocol in [v1_2_localization_protocol.md](v1_2_localization_protocol.md).
The method is training-free. All generators and evaluators remain frozen;
synthetic EHR is fixed within a case. CXR generation currently uses
EHR-derived radiology text, **not** a direct structured-EHR-to-CXR checkpoint.

## Deferred experiment: planner strength versus total generation cost (2026-10-08)

User-requested idea: a stronger API reasoning model may choose a more useful
repair or verification action, avoid unnecessary generation, and stop earlier
at the same final quality. **This is a hypothesis, not an established monotonic
relationship between model capability, decision quality and cost.** Current
execution stays on the frozen local Qwen2.5-VL-7B-Instruct planner; this note
does not authorize API calls, credential access, model downloads or a new job.

中文：以后比较“更强的决策模型，是否在同等输出质量下减少重生成调用和总成本”。
先继续本地 Qwen；不预设越大的模型一定越省，也不把少生成但质量更差算成胜利。

Hold fixed the synthetic EHRs, initial candidates, generator/checkpoint
revisions, action menu, seeds, available evidence, acceptance rules and
evaluation endpoint. Compare a deterministic policy, local Qwen, and later
separately approved API planners at multiple matched budgets. Each policy
sees only evidence acquired up to its current step, never the full future
candidate bank or the final evaluation endpoint. Report decision validity,
useful accepted actions, duplicate/useless requests, abstention, false repairs
when an independent reference exists, and stopping behavior. A second-stage
live run is needed: cached replay alone cannot prove actual GPU savings.

Primary test: quality-versus-cost Pareto curves and total cost required to reach
a predeclared quality target, including failed/unresolved cases in the
denominator. Track CXR generation, report generation, verification, planner
calls, input/output/cached tokens, failures, retries, model-load time and
end-to-end latency separately. Count planner/API overhead as well as avoided
generation; fewer generator calls alone do not prove lower total cost. Report
GPU-seconds and actual API charges separately; any combined dollar estimate
must disclose the GPU price and dated API pricing. Separate evaluation-only
scoring from policy-visible verification, while reporting both costs.

The API payload stays allowlisted: non-sensitive numeric consistency and
uncertainty, opaque candidate/model/action IDs, action history and costs. No
raw MIMIC records, real/synthetic patient-derived image/report content, source
paths, patient identifiers, credentials or private artifacts go to a provider.
An API planner proposes actions; the unchanged local guard validates them.
Keep planner and final evaluator separate so a model cannot grade its own
choices into apparent success. Stronger reasoning cannot recover missing or
incorrect evidence merely by choosing another API model.

OpenAI Docs recommends inspecting workflow traces for tool/routing failures
and using repeatable evaluation datasets for comparisons; its cost guidance
also considers request/token reduction and the accuracy/cost tradeoff. See
[agent evaluations](https://developers.openai.com/api/docs/guides/agent-evals)
and [cost optimization](https://developers.openai.com/api/docs/guides/cost-optimization).
These sources inform measurement practice, not proof of the TriCompose
hypothesis. Do not select a provider/model or quote current pricing until this
deferred experiment is explicitly activated.

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

### Expert-reference RadGraph benchmark completed (2026-10-06)

After explicit source/access approval, the official author RadEvalExpert CSV
was pinned and evaluated internally in existing CPU Slurm allocation 12714150.
All 624 reference/candidate pairs and 762 distinct native graphs completed in
92.315 seconds. Missing annotations remain null. Independent parsing, official
reward/statistical replay, source hashes and protected permissions passed.
No image/EHR read, training, external clinical API, new submission, regeneration
or old-winner change occurred.

See [the full result and boundaries](radeval_expert_radgraph_result_12714150_001.md).
The three predeclared significant-error correlations are only **0.0447,
0.0463, 0.0370**, with cluster intervals crossing zero. RadGraph is therefore
not promoted to a sole clinical judge, agent reward or repair gate.

A separately labeled **post-hoc, reference-based** three-candidate-per-anchor
diagnostic reduces mean significant errors from random **4.0837** to
**3.7552 / 3.8068 / 3.8116**. This modest within-case ranking evidence does
not negate poor pooled calibration or validate reference-free generated-report
consensus. Expert references available in the benchmark do not exist during
fully synthetic deployment. The 180 dependence clusters include 72 unverified
author source keys; one released reader per exact pair does not supply
inter-reader reliability, and checkpoint training overlap is unverified.

Next prioritize independently validated **image-grounded** verification and
scope-aware contradiction sensitivity, not additional scalar weight tuning.
First establish licensed local source-image linkage; actual new patient-input
model computation still needs a fixed protocol and applicable execution
approval. Keep frozen EHRs, old candidate choices and all unavailable cases.

The following [image-grounded benchmark preparation](radeval_biovil_preparation_12714150_001.md)
now binds 43 exact locally available MIMIC images to 132 author candidate
reports, 44 comparison anchors and 34 patient groups. Its source/code/model
pins, stat-only linkage and complete unavailable inventory passed independent
pre-run audit; 2,390 V1.2 tests pass. **No image pixels or model inference have
been consumed in this preparation.** BioViL-T candidate-image scores will be
compared to all three existing RadGraph components on the same available
cohort, with no fitting or old-winner changes. The CPU-only existing-allocation
script and input scope await explicit execution approval; no GPU queue or new
sbatch is needed while that allocation remains active. Dataset availability,
training overlap and reference-count-versus-image-truth limits remain visible.

### Image-grounded BioViL-T expert benchmark completed (2026-10-06)

After the complete CPU entry script and patient-input scope were shown and
explicitly approved, all 43 available images and 130 distinct candidate texts
were encoded in existing allocation 12714150. All 132 available pairs complete
in **42.579 seconds**, peak RSS **1.516 GiB**, with no GPU, new submission,
training, download, external API or original-winner change. All 624 original
rows remain; the 492 unavailable-image scores remain null. Independent source
hash/vector/score/statistical/permission replay passed, including all 624 rows
and 352 analytical choices. **2,390 V1.2 tests pass.** Frozen preparation and
protocol files retain their historical status; this paragraph records execution.

See [the result and same-cohort comparison](radeval_biovil_result_12714150_001.md).
On 132 pairs / 44 comparison anchors / 34 patient groups, BioViL-T significant-
error rho is **-0.1510**, interval **[-0.3198, 0.1098]**. Same-anchor score choice
reduces mean significant errors from random **4.8182** to **4.5682**, but its
difference interval **[-0.7647, 0.1062]** crosses zero. These do not qualify global
cosine as a primary clinical selector or repair trigger.

The three reused reference-based RadGraph components on the identical subset
have rho **0.1372 / 0.1555 / 0.2400**; full-relation has a positive pooled
association, but no same-anchor mean selection improvement. All four selection
difference intervals cross zero. Do not substitute these subset estimates for
the earlier full-pool benchmark, promote a favorable component, flip BioViL's
score direction or fit a combined reward on the inspected outcomes.

Keep matching/structural scores auxiliary. Next prioritize a separately frozen
finding-level contradiction/abstention benchmark with independently supported
references, rather than more generator tuning or an arbitrary scalar repair
loop. Released report-reference errors are not new image-radiologist truth;
training overlap and synthetic-domain transfer remain unverified. No EHR-edge,
fault-localization or targeted-repair claim is cleared by this experiment.

The subsequent [post-hoc error-type diagnostic](radeval_error_type_result_12714150_001.md)
uses derived numerical scores/counts only, with zero further model calls or
raw patient-source access. All seven categories and four frozen metrics remain
visible; 624 attempted pairs and all missing rows stay fixed. BioViL-T strict
same-anchor ranking is **68.29% for omission**, but **44.57% for false
prediction, 40.35% for location and 39.02% for severity**. The omission benefit
does not establish reliable contradiction/detail checks or a repair policy.
These exploratory categories are not independently adjudicated image labels;
do not choose favorable per-type weights or reverse scores on this evidence.
The 28 diagnostics, category-to-original-total additivity, 1,232 analytical
choices and independent cluster intervals passed replay. Runtime is 0.693
seconds. Fifteen new fixture tests pass; the full V1.2 suite now has **2,405
passing tests**. The clinical evidence boundary remains.

### Source-bound native observation evidence completed (2026-10-06)

The [new literal-evidence table](radgraph_literal_evidence_result_12714150_001.md)
reuses all 428 fully synthetic native graphs, all 1,440 same-image report pairs
and all 960 original candidate rows. It appends 15 fields while preserving all
60 old columns, original EHRs and winners. No new model/API/GPU/submission,
patient source, report-file or image read occurred. Construction took 0.546
CPU seconds; independent native-node/pair/CSV replay and protected modes pass.

There are **37** explicit same-literal-atom positive/negative opposition proposals,
touching 36 image slots / 31 unique images / 73 report slots. Current/patient/
temporal scope remains unverified, so these are not clinically confirmed
contradictions or modality faults. The 15,952 unmentioned-side atom occurrences
are preserved as missing comparisons, not negatives or verified omissions.
Location-context and modifier differences remain nonexclusive detail signals.
This avoids another universal score or independent-vote interpretation.

A separate post-hoc quote-free preflight links only **2/37** proposals to an
exact literal name among the existing eight image heads; 35 remain unmatched
without invented aliases. That is a linkage limitation, not proof of clinical
out-of-scope status or lack of image-verifier semantic ability. No request or
repair is executed. The next useful step is independently validated semantic
and scope linkage or a broader suitable frozen verifier—not claiming all 37
resolved by an eight-label job, tuning synonyms on these inspected cases or
regenerating solely to increase native graph agreement. Thirty-one new fixture
tests pass; the full V1.2 suite has **2,436 passing tests**. Clinical qualification
and historical selection stay unchanged.

### Published semantic alternative prepared, not executed (2026-10-06)

The [RaTEScore readiness review](ratescore_readiness.md) identifies an existing
EMNLP 2024 entity-aware metric with official NER and BioLORD synonym encoding.
Local checkpoint inventory did not find those assets. Small public source/model
metadata inspection fixed the code revision, both checkpoint revisions and
22 source/tokenizer/weight hashes; no model weights or patient data were fetched.

Prepared a CPU-only acquisition/environment script, separate approval/license
guards and a no-inference setup receipt. Weights total 1.17 GB; BioLORD's author
requires appropriate UMLS and SNOMED CT licensing. Both acquisition permission
and license confirmation were initially unconfirmed; the user subsequently
confirmed **no UMLS/SNOMED CT license**, so the default BioLORD/RaTEScore route
must not be deployed. No submission, dependency
installation, model call, clinical score, synonym retuning, winner change or
regeneration occurred. Source/checkpoint/cache trees stay excluded from Git.
Twenty new preparation fixtures pass; full V1.2 regression is **2,456 tests**.
Existing bank/literal/head-scope manifests and old selections are unchanged.

The next diagnostic keeps the existing expert-reference cohort and exact
image-accessible subset fixed, retaining failure/empty-entity coverage and all
reported matrices. RaTEScore is reference-based text similarity: neither its
deployment nor same-image report agreement validates current-patient scope,
image truth or a causal repair trigger. A new runtime/API smoke and expert
benchmark require separate execution protocols; setup alone authorizes neither.

A read-only alternative review found the official NCBI MedCPT Query Encoder
card/license marked public-domain. Its 437,951,328-byte weight is not locally
installed and has not been downloaded. Proposed use is **semantic candidate
retrieval**, with existing native RadGraph extraction retained, not an official
RaTEScore substitution or a proven negation/scope/clinical correctness judge.
No new scorer is trained, no equivalence threshold is fitted, and no historical
score/winner changes. Acquisition and actual evaluation require separate
approval; the [readiness note](ratescore_readiness.md) records the fixed metadata
and limitations.

### Approved MedCPT authored CPU diagnostic completed (2026-10-06)

After explicit acquisition/smoke approval, the fixed NCBI Query Encoder is now
locally deployed; the initial read-only proposal above is historical. See
[the result](medcpt_authored_smoke_result_12714150_001.md). All 11 asset hashes
and native parameter bindings pass. Existing CPU allocation only, no training,
new packages/environment/submission, patient inputs or old candidate rescoring.
Total worker time is 14.722 seconds, peak RSS 0.731 GiB; 38 texts/40 eligible
comparisons, one preserved empty/null control, exact two-pass replay.

The diagnostic explicitly rejects a high-cosine-equals-consistent policy:
authored paraphrases average 0.742, negations 0.987, history 0.914, hypothetical
0.939 and laterality 0.961. These are small investigator-authored contrasts,
not clinical gold/accuracy or a validated contradiction scorer. No thresholds,
semantic equivalence, modality fault, repair trigger or historical winner is
promoted. Keep semantic retrieval separate from polarity/anatomy/scope checks;
an independently frozen entity-matching/expert-reference evaluation is the next
qualification step and is **not authorized by this smoke**. No BioLORD/default
RaTEScore deployment is permitted without the absent licenses.

Independent stdlib recomputation/hash/protected-mode audit passes without model
calls. Thirty-one new tests pass; complete V1.2 regression has **2,487 passing
tests**. Worker/probe/protocol/assets are now consumed frozen artifacts; any
new evaluation must use new versioned files/run IDs, not retune these probes.

### MedCPT expert-reference benchmark prepared, not executed (2026-10-06)

The next [frozen diagnostic protocol](radeval_medcpt_protocol.md) and worker
compare whole-report MedCPT cosine against released expert error counts for
the same 624 attempted RadEvalExpert pairs. Reuse all three native cached
RadGraph components and the existing 132-pair BioViL-T image-accessible cohort;
no image encoding or source EHR/image loading. Every comparator shares its
declared available mask, incomplete anchors remain explicit, and the old
complete-cohort results/960 candidates/winners remain unchanged.

Use unchanged native CLS at the model's 512-position capacity, full report
inputs without truncation; oversize/empty/failed inputs stay unavailable. This
is an experimental report-reference representation, not the official 64-token
short-query example or official query/article retrieval metric. Predeclare a
first-eight-text replay only, 1,000 dependence-cluster bootstrap resamples,
seed 0, significant/all-error totals, all categories, and analytical within-
anchor ranking against uniform choice and the error-count oracle. It remains
reference-based, not untouched/blind testing or clinical repair qualification.

Metadata preparation completed at
`artifacts/protected/tricompose_v1_2/radeval_medcpt_plans/reportref_12714150_001/`.
Manifest SHA256 `671a1cd3adfafae3399a53d18a240dc3a51c70b636c67abb8a363607ac2b631b`.
Thirty-two pins, protected modes, shell syntax, independent metadata checks,
and 39 new wholly invented fixture tests pass. Full V1.2 regression: **2,526
tests passed in 14.376 seconds**. No author report CSV was opened and no new
model call occurred during preparation. Worker/runtime inference is untested.

The complete `slurm/63_radeval_medcpt_existing_cpu.sh` entry requires separate
approval for internal author-report computation in actual existing CPU job
12714150: 4 allocated CPUs/32GB/zero GPUs, two worker threads, 20-minute process
cap. No new submission is authorized; if that allocation expires, refuse it
and present a new batch/resource request for approval. No license bypass,
training, download, fit, promotion or selection change is authorized.

### MedCPT expert-reference benchmark completed with qualification withheld

The prepared protocol above was subsequently explicitly approved and executed
inside existing CPU job 12714150. See
[the result](radeval_medcpt_result_12714150_001.md). All 624 pairs and all 762
distinct texts complete with full inputs (maximum 421 native tokens), no
truncation, frozen native CLS, and exact predeclared eight-text replay. Runtime
92.556 seconds, peak RSS 1.087 GiB; no GPU, new submission/download/training,
source EHR/image read, bank rescoring or winner change.

Primary full-cohort MedCPT correlation with negative significant expert errors
is **−0.1054**, dependence-cluster CI [−0.2201, −0.0028]; all-error correlation
is −0.1274. Within 207 eligible fixed anchors, highest-cosine expected significant
errors are 3.8406 versus analytical uniform choice 4.0837, delta −0.2432 with
CI [−0.5070, 0.0323]. The shared 132-pair image subset has rho 0.0700 and
uncertain ranking benefit. Keep both cross-case correlation and within-anchor
results; no score sign flip, favorable-subset promotion, new threshold or
clinical consistency/repair qualification follows. This reference-based result
does not establish a reference-free fact verifier for synthetic triples.

Independent stdlib audit replays 624 cosines, 153 point correlations, 18
clustered correlation intervals, 2,092 eligible anchor/metric/target cells and
selection intervals; immutable hashes and protected modes pass. It reads no
raw clinical text and makes zero model calls. Actual tokenizer/model replay
is only the worker's eight-text replay, not independently repeated inference.
The next development should separate semantic retrieval from validated fact
states and scope, rather than add a cosine-weighted clinical score. No human
feedback, new trained scorer or unlicensed default RaTEScore is introduced.

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

### Source-bound fact comparison interface completed (2026-10-06)

The new `TriCompose-v1.2/src/tricompose_v12/fact_comparison_contract.py`
implements an engineering contract for upstream evidence-grounded proposals,
not a clinically qualified scorer. It separates concept identity, polarity,
temporality, experiencer, laterality, location and severity; every known typed
attribute requires source-bound evidence. Missing information stays unknown,
weak EHR context cannot become a hard constraint, and historical/hypothetical/
family statements are not treated as current patient assertions. Anatomical
differences can block an apparent polarity opposition; bilateral/side/location
differences are not automatically exclusive findings.

The native report adapter reuses the unchanged literal RadGraph module and
retains occurrence IDs, word spans, anatomy/modifier hashes and dependency
lineage. It does not invent semantic aliases or turn modifiers into a validated
severity/location extractor. Native severity, time and experiencer remain
unavailable; raw report-to-graph identity remains caller-supplied lineage, not
independently verified text binding. Agreement/opposition outputs are proposals,
not clinical truth, independently corroborated votes or faulty-modality labels.
No clinical score, repair authorization or selection change is produced.

Forty-three new wholly authored fixture tests pass. Complete V1.2 regression:
**2,569 tests passed in 14.672 seconds**. The model-free integration fixture
contains 12 typed comparisons and two invented native graph comparisons:
`artifacts/protected/tricompose_v1_2/fact_contract_smokes/authored_interface_12714150_001/`.
Runtime is 0.031 seconds. Manifest SHA256:
`e5cd970c9a6bb9c31c16c9296f35faf42da87f78084de15e84c7c2f39ad204c6`.
Independent metadata checks confirm all ten pinned source/historical-manifest
hashes, four artifact hashes, expected fixture relations and protected modes.
There are zero patient/synthetic cohort reads, model calls or new submissions.
These investigator-authored fixtures demonstrate interface behavior, not
clinical gold accuracy, actual attribute extraction or cohort evaluation.

Consumed module/test/worker files and the fixture run are frozen; any correction
requires a new version/run. Old scores, 960 candidate rows and winners remain
unchanged. The next qualification step is to bind an actual upstream attribute
extractor to independent released annotations and test its availability and
errors; passing this interface does not authorize automatic clinical scoring
or targeted regeneration. No new GPU/model job is authorized here.

### Cached literal facts versus expert errors completed (2026-10-06)

The next development diagnostic now exercises the source-bound native adapter
against the existing RadEvalExpert error-count release, not just invented
fixtures. It retains all **624 pairs**, the same **762 cached native graphs**,
all six literal difference features and all 17 available expert outcomes. This
is post-hoc development on a previously inspected report-reference benchmark,
not independently held-out entity-extraction gold or a reference-free verifier.
No disease alias, severity/scope parser, fitted threshold, score weight or new
model is introduced. Cached restricted graph text is consumed inside Slurm;
no original CSV, real EHR, image or target-report file is loaded.

Executed in the actual existing CPU allocation **12766754**, `a02-09`, `main`,
4 allocated CPUs/32 GB, zero GPUs and zero new submissions. The complete
`TriCompose-v1.2/slurm/64_radeval_literal_facts_existing_cpu.sh` entry was shown;
it is not an sbatch script. Runtime **10.601 seconds**, peak RSS **0.081 GiB**.
Every comparison is exactly replayed without neural calls; prediction bytes
are sealed before statistical evaluation, which does not imply blinding.
Five zero-entity native graphs stay explicit; missing measurements are not
zero-filled or converted into clinically correct unknown findings.

The primary comparison below uses significant-error totals: 623 valid pairs,
207 complete three-candidate anchors, and the same uniform-choice error mean
4.0837. More literal differences are hypothesized to correspond to more errors;
each separate diagnostic minimizes its count with uniform choice over exact
minimum ties. There is no combined score or actual bank winner selection.

| Separate literal signal | Spearman versus significant errors | Mean errors under diagnostic choice | Choice minus uniform, 95% cluster interval |
| --- | ---: | ---: | --- |
| Polarity-opposition proposals | 0.0620 | 4.0789 | -0.0048 [-0.0165, 0.0069] |
| Different anatomy contexts | -0.0099 | 4.1449 | +0.0612 [-0.0781, 0.2466] |
| Different modifier token sets | 0.1274 | 4.0266 | -0.0572 [-0.1298, 0.0028] |
| Hypothesis-only literal atoms | 0.2922 | 3.7279 | -0.3559 [-0.5515, -0.1577] |
| Reference-only literal atoms | 0.3001 | 3.9654 | -0.1184 [-0.2388, 0.0168] |
| Uncertain/mixed literal comparisons | 0.0853 | 4.0998 | +0.0161 [-0.0277, 0.0649] |

All category and insignificant/all-error results remain in the protected
evaluation, not only this significant-total table. Intervals use 1,000 whole-
dependence-cluster resamples, seed 0, and are not multiplicity-adjusted. Cluster
identity limitations and checkpoint training-overlap uncertainty remain those
of the earlier expert benchmark. No sign reversal or favorable-subset tuning.

Two declared category-specific probes clarify coverage:

- Polarity opposition occurs in only **9/624** pairs. Against the 337 reports
  with any expert significant false-prediction error, the fixed count>0 alarm
  finds six: recall **1.78%**, precision **66.67%**, count AUROC **0.5037**.
  This is NOT negation-specific accuracy; false prediction is a broader author
  category and the release does not identify which entity is wrong.
- Anatomy-context difference versus expert location error has count AUROC
  **0.5938**, fixed-alarm precision **27.51%**, recall **63.91%**, and rho 0.1496.
  Its location-error selection delta is -0.0377, CI [-0.0732, 0.0053]; total-
  error choice does not improve. Different anatomical mentions may coexist;
  they are not automatically contradictory or location errors.

The promising unmatched-atom ranking remains **reference-based and literal**.
Unmatched atoms are not verified hallucinations or omissions; synonyms and
report length can affect this result. Do not promote the best row into a
synthetic reference-free score, declare a faulty modality, or trigger repair.
Native time/experiencer/severity remain unavailable. The next safe diagnostic
should compare this ranking with shortest-report, reference-length-distance
and native-entity-count controls on identical anchors; new independent expert
entity/scope annotations are still needed for clinical qualification.

Official [RadGraph dataset documentation](https://physionet.org/content/radgraph/1.0.0/)
provides a two-reader entity/relation test set through credentialed DUA access,
but notes that comparison/history context is not captured. It cannot alone
qualify time/experiencer judgments. No gated gold data was downloaded. The local
official `RadGraph/tests/radgraph-xl_chexbert_test.py` explicitly treats its
stored JSON as expected model predictions for regression stability, NOT new
independent expert entity gold; those report JSON bodies were not opened.

Implementation:
`TriCompose-v1.2/src/tricompose_v12/radeval_literal_facts.py`,
`TriCompose-v1.2/tools/diagnose_radeval_literal_facts.py`,
`TriCompose-v1.2/tests/test_radeval_literal_facts.py`.
**29 new tests pass; complete V1.2 regression: 2,598 tests in 20.425 seconds.**

Protected output:
`artifacts/protected/tricompose_v1_2/radeval_literal_fact_runs/literal_expert_12766754_001/`.
Run manifest SHA256:
`0a05b720644c9c1ece2437a8cde183e5e9ff93f1d2c7309ed8f834effb22a782`.
Frozen plan:
`artifacts/protected/tricompose_v1_2/radeval_literal_fact_plans/literal_expert_12766754_001/`,
manifest `8fbe5bf2b661e3094059929c960d0b0f8dbfe01bc1818131764500aa298e4198`.

Independent stdlib numerical audit:
`TriCompose-v1.2/audits/audit_radeval_literal_facts.py`.
It verifies 3,744 count bindings, 102 correlations, 102 binary diagnostics,
28 clustered intervals and 21,180 eligible anchor/feature/target cells in
7.555 seconds. It hashes source bytes but does not decode raw reports/native
graphs or independently repeat native extraction; this is numerical/provenance
verification, not independent clinical adjudication. Private modes, old pins
and Git exclusion pass. Audit SHA256:
`481062efef6a4016c4cf0f3356927f576dce72e4916832d339740c4573c6b940`, at
`artifacts/protected/tricompose_v1_2/radeval_literal_fact_audits/numeric_12766754_001/audit.json`.
Consumed code/tests/entry and runs are frozen. Existing 960 rows, scores,
EHRs, generation outputs and historical winners remain unchanged.

### Length/entity controls completed; clinical superiority not established

The preceding literal-feature result has now been tested against the three
proposed metadata controls on the identical frozen expert cohort. All six old
literal signals are retained, not only the observed best one. The controls
minimize (1) full cached native tokenizer length, (2) absolute token-length
distance to the reference, or (3) all native RadGraph entity count. Token length
includes special tokens and is NOT a word or clinical-fact count. Entity count
includes anatomy/modifiers/measurements, not only findings; four hypothesis
pair slots have zero-entity graphs. These are explicit nuisance heuristics,
not new clinically qualified scores.

All 624 attempted pairs share available metadata for all nine selectors.
Significant-error totals retain 623 pairs and 207 complete three-candidate
anchors; all-error totals retain 622 pairs/206 anchors. Missing expert cells
stay unavailable. Minimum-cost ties are uniformly averaged, not resolved by
first slot. Every selector and all 17 expert outcomes are reported. The old
102 literal-feature/outcome cells replay exactly, including full anchor rows.

| Diagnostic choice | Mean significant expert errors, lower is better | Choice minus uniform |
| --- | ---: | ---: |
| Uniform random expectation | 4.0837 | 0 |
| Hypothesis-only literal atoms | 3.7279 | -0.3559 |
| Shortest native token length | 3.5338 | -0.5499 |
| Closest reference token length | 3.6787 | -0.4050 |
| Fewest native entities | 3.7440 | -0.3398 |

Compare the previously observed best literal feature with each control using
paired same-anchor differences, not separate confidence-interval overlap:

| Control | Literal choice minus control mean significant errors | 95% dependence-cluster interval |
| --- | ---: | --- |
| Shortest native token length | +0.1940 | [0.0591, 0.3660] |
| Closest reference token length | +0.0491 | [-0.1843, 0.3010] |
| Fewest native entities | -0.0161 | [-0.1298, 0.0951] |

This diagnostic does NOT show literal clinical-fact ranking beating simple
length/entity heuristics. Shortest length performs better on the observed total;
the literal feature has no clear benefit over the other two controls. These
post-hoc, previously inspected, multiple-comparison results are not an untouched
confirmatory experiment and cannot conclusively identify a causal length bias.
They do rule out presenting the earlier random-choice improvement alone as
evidence of superior clinical reasoning.

The error vector matters: both shortest length and hypothesis-only literal
ranking increase observed significant omissions relative to uniform choice.
Omission means are **1.9686 / 1.9573**, versus uniform **1.7681**, on 207 complete
anchors. False-prediction means are **0.7428 / 0.7821**, versus uniform **1.1763**,
on 208 category-complete anchors. Do not add means across these different
denominators or claim a category-specific significant effect from point
estimates alone; category intervals were not part of this control protocol.
Reference-length matching reduces observed omission to 1.6812 but itself
requires a reference report unavailable in fully synthetic generation.
No length-only policy is promoted and no after-the-fact balancing weights are
fitted to make a chosen metric look favorable. Keep false findings, omissions,
detail/scope errors and coverage separate; qualify actual fact extraction on
independent annotations before allowing clinical fault localization or repair.

Execution uses the existing CPU Slurm job **12766754**, `a02-09`, `main`, with
4 CPUs/32 GB already allocated, zero GPUs and zero new submissions. The shown
`TriCompose-v1.2/slurm/65_radeval_length_controls_existing_cpu.sh` is not sbatch.
Runtime **6.971 seconds**, peak RSS **0.070 GiB**. Only hashed/numeric metadata
is decoded: no source CSV, EHR, image, report/native-graph body, embedding
vector, model/tokenizer call, training, download or external clinical API.
There are 153 selector/outcome cells and 306 paired feature/control cells.

New implementation:
`TriCompose-v1.2/src/tricompose_v12/radeval_length_controls.py`,
`TriCompose-v1.2/tools/diagnose_radeval_length_controls.py`,
`TriCompose-v1.2/tests/test_radeval_length_controls.py`.
Twenty-five wholly invented metadata tests pass. Full V1.2 regression:
**2,623 tests in 18.041 seconds**. Clinical scores stay null; old candidate
rows, generation outputs, EHRs, model parameters and actual winners are fixed.

Protected run:
`artifacts/protected/tricompose_v1_2/radeval_length_control_runs/length_controls_12766754_001/`,
manifest SHA256 `7f927552158286e755daa98da81814621580e986607f14e98b75a3266e1c2bff`.
Its `metadata_table.json` and `evaluation.json` contain the complete numerical
cohort/control results, not report text. Frozen plan:
`artifacts/protected/tricompose_v1_2/radeval_length_control_plans/length_controls_12766754_001/`,
manifest `f8cd2805f36aff144340a9c47ffe217154e3c24e7a03631090fcac935c8d6d16`.

Independent audit `TriCompose-v1.2/audits/audit_radeval_length_controls.py`
recomputes **5,616 metadata costs, 153 correlations, 31,770 eligible
selector/target/anchor cells, 306 paired comparisons and 54 clustered intervals**
in 6.400 seconds, without production statistic imports or raw clinical text.
All source/result pins, protected modes and Git exclusions pass. Audit:
`artifacts/protected/tricompose_v1_2/radeval_length_control_audits/numeric_12766754_001/audit.json`,
SHA256 `c0d06267b1344dd10bc8e6ca8227f7fd374717b9e0719ccfe169b61ff5a8a72b`.
Consumed sources and runs are frozen. This is a negative-control diagnostic,
not a new synthetic scoring table, clinical accuracy estimate or repair result.

### Independent entity-gold availability audit (2026-10-06)

Next, a strictly read-only source/filesystem audit looked for human entity,
polarity and relation annotations. It did not decode patient artifacts or run
models. Within the bounded inspected roots, **no qualified local entity-gold
package was located**. This is not a claim that none exists anywhere on CARC:
the search had explicit depth/name limits and two project subdirectories were
inaccessible. RadEvalExpert supplies report-level error counts; RICORD supplies
image annotations. Neither supplies the report-span truth needed here. The
local RadGraph regression fixture remains model predictions, not human gold.

Official candidates, not acquired or evaluated:

- [RadGraph-v1](https://physionet.org/content/radgraph/1.0.0/) has 100 test
  reports with two independent radiologist annotations per report. It can test
  spans, observation polarity and native relations, not image/EHR truth or
  comprehensive temporal/experiencer scope. PhysioNet requires credentialing
  and the dataset-specific DUA.
- The [official Stanford CheXpert release](https://aimi.stanford.edu/datasets/radgraph-chexpert-results)
  supplies 50 human-annotated test reports. Its current official link leads to
  Redivis; access terms have not been verified. `CheXpert_graphs.json` is instead
  model-generated and must not be substituted for the human test file.
- [CXRGraph](https://physionet.org/content/cxrgraph/1.0.0/) reannotates the
  original RadGraph reports. Its human test contains 100 reports and adds
  normality, action and change attributes. It is useful for checking temporal
  language, but is not a new independent patient cohort. Its native schema
  differs, and change/action cannot silently become all our
  current/prior/hypothetical states. `inference.json` is not human gold.
  Its current access policy requires a registered account and the project DUA,
  not the full credentialed-access requirements of RadGraph-v1. The two
  annotators have clinical-data-processing/medical backgrounds; do not call
  them two confirmed radiologists. They did not consult images for annotation.

Two important source limitations emerged. The PhysioNet
[RadGraph-XL release](https://physionet.org/content/radgraph-xl/1.0.0/) contains
only its MIMIC subset. The [original paper, Table 2](https://aclanthology.org/2024.findings-acl.765/)
places those MIMIC reports in CT/MR, while its newly annotated CXR reports come
from Stanford. Therefore downloading this MIMIC subset is not a CXR entity-test
solution. Section 4 also combines old and new RadGraph data for training and
split selection. The old RadGraph-v1 test is **not automatically held out for
our pinned XL checkpoint**. Actual training overlap remains unresolved; a
human annotation may be independent of predictions without being checkpoint-
heldout. No clean generalization claim is allowed before this check.

The smallest next clinical experiment requires an authorized local human
annotation package, access confirmation, source hash/schema audit, fixed
cohort and checkpoint-overlap audit before prediction. Evaluate exact
span/type entity precision/recall/F1, polarity confusion on matched spans with
its coverage, and relations with exact typed endpoints. Keep released readers
separate instead of inventing consensus. Retain unknown/unsupported attributes
as unavailable; do not map missing entities to negatives or location mismatch
to a confirmed contradiction. CXRGraph requires a separately checked schema
adapter. This extraction experiment alone still cannot identify which CXR or
EHR modality is clinically wrong.

Machine-readable audit:
`artifacts/protected/tricompose_v1_2/gold_readiness_audits/gold_availability_12766754_001/audit.json`.
It records search bounds, official sources and unfulfilled gates, not clinical
results. No downloads, credential reads, new Slurm submissions, training,
scoring changes or regeneration occurred. Existing CPU job 12766754 was used
only for source/metadata checks. Existing banks, selected triples and consumed
benchmark workers remain frozen. Clinical qualification is still withheld;
the next missing input is authorized human-gold data, not another GPU queue.

### Exact-span gold-evaluation interface implemented, not clinically run

The follow-up adds one pure standard-library interface:
`TriCompose-v1.2/src/tricompose_v12/entity_gold_contract.py`, with 48 invented
tests in `TriCompose-v1.2/tests/test_entity_gold_contract.py`. No existing
extractor, metric worker, checkpoint, candidate or selection is changed.

`normalize_graph()` accepts only the documented RadGraph-v1 graph format or
the existing pinned native XL format. V1's four labels map to their exact XL
equivalents; all 11 XL labels remain preserved. It verifies inclusive native
word offsets against token text, converts to end-exclusive offsets, and emits
only opaque IDs, hashes, offsets and fixed ontology labels/relations. Entity
text, source report keys and arbitrary input fields never leave this interface.
Source-report hashes and human origins are declared, not authenticated here.
Hashing a declared annotation does not prove its access, provenance or truth.

`compare_annotations()` requires matching opaque report ID, declared source
hash and the exact token-sequence hash. No guessed alignment, text embedding,
disease synonym, threshold or learned scorer is introduced. Outputs are:

- exact span/type entity TP/FP/FN and precision/recall/F1;
- exact directed relations, including both endpoints' spans and labels;
- three-state polarity confusion **conditional on matched observation spans**,
  accompanied by gold-span coverage and unmatched observation counts;
- explicit predictions outside the gold annotation's supported label scope.

Those last predictions and their relations are unavailable for that label
scope, not silently projected into common findings. If a v1 observation is
instead predicted as an XL measurement, the supported gold entity is still
missed; the measurement prediction is reported separately. Conditional
polarity accuracy alone cannot hide missed spans. A genuinely produced
zero-entity graph is distinguishable from a missing/failed prediction, and
both-empty entity F1 is undefined, not perfect. Missing entities are never
negative findings. Observation spans include modifiers and are not
automatically clinical disease concepts.

`aggregate_reader()` sums micro counts for one reader against an explicit
fixed attempted-report inventory. Failed/missing comparisons remain visible
in availability, not favorable zeros. Duplicate cases, mixed readers, mixed
authored/human cohorts, changed receipts and inconsistent count partitions
are rejected. No automatic reader consensus or patient-level confidence
interval is invented. Reader disagreements do not become independent votes
about CXR correctness. These diagnostics are **not** the upstream F1RadGraph
report-reference reward and do not establish report, image or EHR factuality.

No CXRGraph parser or temporal/action conversion is claimed: its new schema
requires a separately verified adapter after authorized source intake. All
clinical qualification, source-access/heldout verification, scope/image/EHR
truth and regeneration flags remain false; clinical scores remain null.

`TriCompose-v1.2/tools/smoke_entity_gold_contract.py` accepts no clinical input
paths and runs only authored fixtures inside a genuine existing Slurm cgroup.
Its completed run used job 12766754 and took **0.032 seconds**, peak RSS
**0.022 GiB**, with zero models, downloads or new submissions. Eight eligible
fixtures are compared separately to two intentionally disagreeing authored
readers; a ninth binding-failure fixture stays unavailable for both. These
are not clinician annotations or independent clinical performance results.
Sixteen predeclared entity/relation count checks pass. A separate stdlib-only
recheck reconstructs 32 per-comparison count cells from normalized fixture
records, verifies reader totals, source/result hashes and protected modes.
Full V1.2 regression: **2,671 tests in 14.479 seconds**, all passing.

Protected smoke output:
`artifacts/protected/tricompose_v1_2/entity_gold_contract_smokes/authored_12766754_001/`.
Manifest SHA256:
`f9a863e9f37a81c18efa0e3155b102d02e4272864b392bf52b564425247da0e6`.
The new consumed worker/interface/tests and run are frozen. Existing bank and
length-control manifest pins remained unchanged, permissions are directories
2770/files 0660, and artifacts remain Git-excluded. This finishes an engineering
prerequisite only; authorized human-gold data and a checkpoint-overlap audit
are still required before the next actual clinical extraction benchmark.

### CXRGraph manual-schema adapter implemented, dataset test still unavailable

This supersedes only the earlier "no CXRGraph parser" implementation status;
it does not relax source-access, checkpoint-overlap or clinical-qualification
gates. No authorized local CXRGraph package was found in the inspected protected
source roots. No patient payload, credentials or model weights were acquired.

The author's old GitHub URL redirects to
[yxliao95/arrg_cxrgraph](https://github.com/yxliao95/arrg_cxrgraph).
Read-only public code inspection pinned revision
`4b0edaf75d18128cbccfaf90ff92549984600056`.
[config.py](https://github.com/yxliao95/arrg_cxrgraph/blob/4b0edaf75d18128cbccfaf90ff92549984600056/config.py)
defines five native entity classes, four forward relation types and three
attribute domains. The pinned entity inference/training source confirms
document-global token offsets, inclusive endpoints, attribute-slot ordering
and the `NA` marker. Source code was inspected, not imported or executed;
upstream training flags were never invoked.

New source:
`TriCompose-v1.2/src/tricompose_v12/cxrgraph_gold_adapter.py`.
`normalize_manual_document()` checks the exact manual five-field document
schema and sentence annotation inventory. It preserves native entities,
directed relations and explicitly assigned normality/action/change attributes
without retaining report words or source keys. Offsets remain document-global
and become end-exclusive. Duplicate/conflicting spans, unbound relations or
attributes, classifier null/inverse tags, guessed sentence-local alignment and
prediction keys presented as manual gold fail closed. Character/token limits
are checked before assembling document text. This is code/schema validation,
not verification of any actual dataset row or human provenance.

The common-label projection reuses the unchanged exact-span evaluation
interface. It maps only the four core entity names and retains only direct
common relations with supported endpoints. It does **not** turn `part_of` into
`modify`, flatten a location-attribute path, map positive/negative Change to
finding polarity, or impute unassigned normality. Native excluded entities,
relations and assigned attributes remain in a sidecar with explicit inventory.
The unassigned-slot count is a storage representation count, not a denominator
of clinically applicable attributes. CXRGraph's jointly annotated result is
not turned into two independent reader votes.

`compare_to_native_prediction()` can compare the common projected labels to
an already normalized native prediction with exact source/token binding.
It is **not** full official CXRGraph scoring: our current XL extractor does not
output CXRGraph's native normality/action/change, Location-Attribute or part_of.
Those extra dimension scores remain null. Annotation granularity can differ,
and this projection does not establish semantic equivalence, current-patient
scope, image truth, EHR consistency, error localization or a qualified repair
decision. No new extractor or semantic rules were deployed.

Tests:
`TriCompose-v1.2/tests/test_cxrgraph_gold_adapter.py` has **36 authored tests**.
`TriCompose-v1.2/tools/smoke_cxrgraph_gold_adapter.py` accepts no clinical-file
arguments and exercises six invented documents within existing CPU Slurm job
12766754. Thirty predeclared native/projected inventory checks pass, runtime
**0.034 seconds**, peak RSS **0.024 GiB**. A separate stdlib recheck reconstructs
the common projection and checks all 30 count cells, receipt hashes and modes.
Full V1.2 regression: **2,707 tests in 15.431 seconds**, all passing.

Protected run:
`artifacts/protected/tricompose_v1_2/cxrgraph_adapter_smokes/authored_12766754_001/`.
Manifest SHA256:
`0db65fd95936137a6a24c08c1d585ad932ae34313ba78137a1b65953b6efb9ea`.
The run and consumed new worker/module/tests are frozen. Existing entity-gold
smoke and bank manifest pins are unchanged; protected modes are directories
2770/files 0660 and artifacts remain Git-excluded. No models, patient inputs,
new submissions, training, revised clinical scores or selected triples.

Actual independent extraction accuracy remains unmeasured. The next actionable
input is an authorized manual annotation file/package and its private server
path, followed by access/schema/provenance and checkpoint-overlap intake in an
approved worker. Further authored smokes cannot substitute for that evidence.

### Image-available expert subset: metadata negative-control follow-up

The missing authorized manual entity annotation package does not block a
separate numerical control on the already acquired, independently audited
RadEvalExpert/BioViL-T results. This follow-up does not replace the planned
human entity-extraction benchmark or use any report/image body. It asks whether
the existing image-grounded ranking shows benefit beyond simple metadata
rules on the **same** image-available subset, rather than comparing different
cohorts. The earlier BioViL-T results have already been inspected: this is a
post-hoc development diagnostic, not a blinded or untouched final test.

Freeze seven selectors before the new calculation: original BioViL-T cosine,
all three original reference-based RadGraph scores, minimum cached native
token length, minimum absolute token-length difference from the reference,
and minimum native entity count. Token lengths come from the already cached
MedCPT tokenizer, include its special tokens, and are not BioViL-T lengths.
Entity counts include all native entities, not just diseases. Length/entity
count is not asserted to be clinical quality. Reference-length proximity and
RadGraph F1 require a reference report and cannot become fully synthetic
reference-free selection signals.

Retain all 624 attempted opaque rows and their missing scores. Require the
unchanged 132 image-available pairs / 44 complete three-candidate anchors for
every comparator; do not drop candidates or favorable cases. Analyze both
significant and all-error totals with uniform expected choice over exact ties.
Calculate paired selected-error differences versus each metadata control,
with 1,000 whole-source-group bootstrap resamples, seed 0. Positive difference
means the original metric selects more errors than that control. All 24
metric/control/target comparisons remain visible; no multiplicity adjustment,
threshold fitting, sign inversion or favorable comparator selection.

New paths are `TriCompose-v1.2/src/tricompose_v12/radeval_image_length_controls.py`,
`tools/diagnose_radeval_image_length_controls.py`,
`audits/audit_radeval_image_length_controls.py`, and 34 authored tests in
`tests/test_radeval_image_length_controls.py`. The worker uses existing CPU
allocation 12766754 and requires byte-pinned source manifests/audits. It must
exactly replay all eight original image-selector/target choice endpoints and
their point correlations before accepting results. No model call, new Slurm
submission, patient text/image/graph-body decoding, new training or changed
synthetic-bank winner. Clinical qualification and repair gates remain false.

The new cached-number run and separate audit have now completed. No clinical
source payload or generated patient content was opened. All 624 attempted
rows remain; all seven selectors share the unchanged 132 available pairs,
44 three-candidate source/section/reference anchors, and 34 source groups.
Forty-four anchors are not forty-four patients. The original eight
metric/target selection endpoints and point correlations replay exactly.

| Selector | Mean significant errors | Mean all errors |
| --- | ---: | ---: |
| Uniform expected choice | 4.8182 | 5.3258 |
| BioViL-T original cosine | 4.5682 | 5.0909 |
| RadGraph entity F1 | 4.6250 | 5.1932 |
| RadGraph relation-presence F1 | 4.7727 | 5.2500 |
| RadGraph full-relation F1 | 4.8182 | 5.3182 |
| Minimum native token length | 4.2386 | 4.7500 |
| Closest reference token length | 4.0227 | 4.5909 |
| Minimum native entity count | 4.3182 | 4.7727 |
| Expert minimum-error oracle | 3.2045 | 3.7273 |

Lower error count is better. Oracle uses expert outcomes and is only an
analytical lower bound, not an executable selector. These are released
report-reference expert error counts, not image adjudication or EHR truth.
They must not be renamed accuracy percentages or synthetic repair success.

BioViL-T's significant-error difference versus minimum length is **+0.3295**,
95% source-group bootstrap interval **[-0.4853, +0.8525]**; versus closest
reference length, **+0.5455 [-0.2210, +1.0861]**; versus minimum entity count,
**+0.2500 [-0.5006, +0.8183]**. All six BioViL-T/control intervals across the
two targets include zero. Its earlier difference versus uniform choice,
**-0.2500 [-0.7647, +0.1062]**, remains unchanged. This does not establish that
BioViL-T is better than metadata controls; nor do crossing-zero paired
intervals establish that it is definitively worse.

For significant errors, the three metadata-control differences versus
uniform choice are respectively **-0.5795 [-0.9053, -0.1275]**,
**-0.7955 [-1.1288, -0.3136]**, and **-0.5000 [-0.8824, -0.0684]**. These are
unadjusted exploratory intervals on an already inspected small development
subset. Do not promote minimum length as a clinical policy, infer causal
length confounding, or treat reference-length proximity as usable without
a real reference. Fewer entities are not necessarily a more faithful report;
omission/false-positive trade-offs still matter. All 24 paired comparisons,
not just the BioViL-T entries, are retained in the protected evaluation.

Development decision: keep BioViL-T as a secondary matching signal. Do not
enable automatic regeneration, invert a score, fit a combined reward or
change synthetic winners from this outcome. The next clinical-extraction
experiment still needs authorized human annotations and training-overlap
checking; this control is additional numerical evidence, not that benchmark.

Run:
`artifacts/protected/tricompose_v1_2/radeval_image_length_control_runs/image_length_12766754_001/`.
Plan is under `radeval_image_length_control_plans/image_length_12766754_001/`.
Run manifest SHA256:
`bfa77f2291c5a94c9f1d7a73a59a9666abfdac01d45385ae4bb7b79ff078654f`.
Plan manifest SHA256:
`37f224f03c25c30551d1d966a817513da7a6899734d9c0ddd970a9ad1d5a3ca8`.

Independent audit:
`artifacts/protected/tricompose_v1_2/radeval_image_length_control_audits/numeric_12766754_001/audit.json`.
Audit SHA256:
`f95a27f39ac368d991eed04ae34012abd17e36207d10adc78978062bbe3935e7`.
It reconstructs 4,368 cached score cells, 14 point-correlation cells, 616
eligible selector/target/anchor cells, all 24 paired comparisons, and all 38
bootstrap intervals without production statistics imports. It checks pinned
inputs and protected modes. This is independent numerical replay, not fresh
expert annotation or replay of model inference.

Execution used only existing CPU allocation 12766754: worker **0.552 seconds**,
peak RSS **0.028 GiB**, independent replay **0.453 seconds**, zero model calls
and new Slurm submissions. Full V1.2 regression **2,741 tests in 14.980 seconds**,
all passed. The new consumed worker/module/tests/auditor and run are frozen;
old manifests and the 80-EHR bank are unchanged. Protected directories/files
remain 2770/0660 and Git-excluded. No commit or push was performed.

### Human-gold benchmark start: checkpoint provenance preflight

After the user's instruction to begin, the two protected source/reference
roots were checked again by bounded names/metadata only. No matching manual
package directories or named manual test files/archives were found. This is
not a server-wide absence claim or a denial of the user's access approval.
No annotation payload, patient report/image/EHR or credentials were opened.
The actual extraction benchmark has not run; its accuracy remains unavailable.

The local XL archive was rehashed and still matches the previously pinned
official SHA256 `fadb5a3454e8996714b609e3105a07e447fa61aa88fffe966b650959475117b6`.
The 3,719-byte model config contains train/validation/test path declarations,
but the local model package does not supply the record-membership inventory.
Paths were hashed rather than printed; a path declaration is not held-out
membership evidence. No weights were decoded or model loaded.

The [RadGraph-XL paper, Sections 4.1–4.2](https://aclanthology.org/2024.findings-acl.765.pdf)
uses original RadGraph data along with its new annotations and selects a
cross-validation fold with 2,320/290/290 training/validation/test reports.
Importantly, Section 4.5 also reports a **separate** experiment excluding the
original RadGraph test set. This means we must not automatically assert either
contamination or a clean original-test holdout. The pinned checkpoint's
association with those training variants and exact membership is unverified.
Its [pinned Hugging Face card](https://huggingface.co/StanfordAIMI/RRG_scorers/blob/6646433b3ad83a10f6e141db76d0ece44312b236/README.md)
contains license metadata, not that association. The
[CXRGraph release](https://physionet.org/content/cxrgraph/1.0.0/) reannotates
original RadGraph reports; a new annotation scheme alone is not new report
content or proof of checkpoint-heldout generalization.

Protected preflight record:
`artifacts/protected/tricompose_v1_2/gold_readiness_audits/checkpoint_overlap_12766754_001/audit.json`.
It explicitly retains overlap count, benchmark accuracy and the local gold
path as null. Next needed: the authorized manual file's server path and
checkpoint training-variant/membership evidence. If overlap remains unresolved,
any later extraction result must be labeled development diagnostics, not an
independent generalization qualification. Full native CXRGraph attribute scores,
image truth, EHR truth and modality-error localization remain unavailable.
There were no model calls, downloads, new jobs, revised winners or changes to
consumed workers. This records a real prerequisite audit, not clinical progress
by substituting additional authored fixtures.

### User-terminal CXRGraph acquisition prepared

The user supplied the official authenticated wget command and approved using
it. `tools/download_cxrgraph_manual.sh` now wraps that workflow for the user's
own interactive terminal. The PhysioNet password must be entered at wget's
hidden terminal prompt; no password/token is requested in chat, put in argv,
saved by the script, or obtained from an existing credential file. The agent
has not initiated an authenticated download or claimed that access was verified.

Run from the workspace root:

```bash
bash tools/download_cxrgraph_manual.sh
```

The script creates a fresh opaque `cxrgraph_manual_XXXXXXXX` directory below
`artifacts/protected/tricompose_v1_2/reference_datasets/`, not over an existing
run. It retains manual annotations, the official checksum list and license,
while excluding the bulk machine-inferred corpus unnecessary for this test.
Recursive retrieval is restricted to the official HTTPS source subtree,
two levels and a 100 MB recursive quota; no redirected authentication or
inherited wget/netrc/cookie/HSTS/askpass credentials. Logs remain inside the
new protected directory. Success requires a nonempty `manual_data/test.json`
matching its unique official SHA256 entry. Checksum-list parsing does not
open clinical JSON or follow arbitrary checksum paths. Directories/files are
2770/0660; partial failures remain private and are not silently overwritten.

Shell syntax and the no-network dry run passed. Noninteractive execution
correctly refuses before creating a source directory or asking for credentials.
No source file has been downloaded, parsed or scored by these checks; no model,
new Slurm task, dataset license signature or stored-credential read. After the
user runs the command, its printed opaque test path and checksum are the next
inputs for protected provenance/schema intake. Training-overlap qualification
remains a separate unresolved gate. Share only the path/status, not the password,
private transfer log or report content.

### User-uploaded CXRGraph ZIP received and arranged (2026-10-07)

The ZIP uploaded at the workspace root has been moved, without replacing an
existing run, to the fresh protected intake:
`artifacts/protected/tricompose_v1_2/reference_datasets/cxrgraph_source_12766754_001/`.
The original bytes remain in `archive/cxrgraph_1.0.0.zip` (100,574,950 bytes;
SHA256 `d8cf3df0cc31e34f8607faf58ab42fba8dd58a3a074e6a4818e51c27eb0ec8a8`).
The root copy was relocated, not discarded. Other pre-existing download
directories and incomplete transfers were not changed.

`tools/intake_cxrgraph_zip.py` unpacked a fixed eight-file allowlist to `source/`:
the license, data dictionary, checksum list, and the five manual-data split
files. The manual test input is now at
`source/manual_data/test.json` relative to this intake. The selected output
totals 2,000,252 bytes. The bulk machine-inferred `inference.json`, example PDF
and README remain inside the ZIP; none was extracted or opened. No clinical
JSON was decoded or report content inspected. This is file availability and
integrity intake, not schema validation or extraction-accuracy evaluation.

All seven selected release files with embedded SHA256 entries matched; the
checksum list itself was separately hashed, not treated as self-authenticating.
ZIP CRC checks passed for all eight selected members. This verifies consistency
with the embedded checksum list, not an independently authenticated remote
release, nor the integrity of the three unextracted members. An independent
streaming hash/size/permission replay also passed. All intake directories/files
are 2770/0660, with project-group ownership exposed as NFS gid 65534, and the
archive, annotations and receipts are Git-excluded.

The intake manifest is `intake_manifest.json`, SHA256
`6b50154bf091e03bd2c6bb9a2d592dce31bb06a9156e3de6ca853626d36eece7`.
It records selected-file hashes, source-origin declaration, skipped members,
worker hash, zero model calls, unparsed annotation status and unknown checkpoint
training overlap. Ten authored ZIP/checksum safety tests passed. The worker
refuses an existing or interrupted intake instead of overwriting it. Work ran
within the existing CPU Slurm allocation 12766754; no new submission, model
download, inference, or changes to frozen benchmark outputs were performed.

Next, an explicitly scoped protected worker can validate the manual annotation
schema and bind an evaluation cohort. Any subsequent RadGraph-XL extraction
result must remain a development diagnostic until the previously documented
checkpoint-training overlap gate is resolved. Gold-file availability does not
by itself qualify the scorer or supply image/EHR truth.

### CXRGraph actual manual-test extraction diagnostic completed (2026-10-07)

This replaces the earlier **availability** blocker, not the checkpoint-overlap
or clinical qualification gates. The uploaded manual files were consumed only
inside the existing approved CPU Slurm allocation 12766754. No new job, GPU,
model acquisition, training, external API call, or candidate-bank change.
Source reports, identifiers, original images and real targets were not displayed.

`prepare_cxrgraph_manual_cohort.py` verified the JSON-lines schema on all 100
released test documents. The MIMIC and CheXpert subfiles each contain 50 entries,
form an exact disjoint partition, and match the corresponding complete test
documents. File order is fixed; all reports remain attempted. This is not a
claim of 100 independent patients or checkpoint-heldout membership. The joint
manual annotation is one released consensus, not two independent reader scores.
Contract receipt:
`artifacts/protected/tricompose_v1_2/cxrgraph_gold_contracts/manual_test_12766754_001/manifest.json`,
SHA256 `19f61d54c4cfda59212564e615f14518f9e3ce5cab3c71a59b9e1da9cdd56c15`.

The first frozen XL run, `cxrgraph_extraction_runs/manual_xl_12766754_001/`,
retains strict identical-word-token binding. It evaluated only 50/100 reports:
49 MIMIC and one CheXpert. The other 50 were unavailable because official XL
preprocessing changed token boundaries, not zero-error cases. Its entity F1
0.9624 and relation F1 0.7261 must not be presented as full-test results.
That run, code, script and predictions remain frozen.

The separate V2 diagnostic uses `entity_gold_character_contract.py` and
`score_cxrgraph_manual_xl_v2.py`. Prediction tokens are matched literally against
the released words joined with single spaces. This yields exact source-character
offsets, allowing punctuation splits while rejecting changed/deleted characters,
case changes, incomplete consumption and merges across original whitespace.
No fuzzy alignment, gold-boundary snapping, source rewriting or gold-label
correction. Partial entity spans remain false positives/negatives. Native model
input and predictions remain those of the pinned official frozen package.
The metric is a new exact-character extraction diagnostic, not the previous
strict identical-token metric or full official CXRGraph scoring.

All 100 V2 reports are eligible, with 100 native forward passes. Runtime was
30.502 seconds, peak RSS 2.206 GiB, CPU only (two model threads).

| Released domain | Attempted / eligible | Common entity micro-F1 | Direct common relation micro-F1 |
|---|---:|---:|---:|
| MIMIC | 50 / 50 | 0.9639 | 0.7200 |
| CheXpert | 50 / 50 | 0.9138 | 0.6565 |
| Overall | 100 / 100 | 0.9375 | 0.6857 |

Overall entity TP/FP/FN are 2,609/124/224; relation TP/FP/FN are
1,294/705/481. Three-state observation accuracy is 0.9833 **conditional on
1,499 exact matched observation spans**. Gold observation-span coverage is
1,499/1,641 = 0.9135; the 142 unmatched gold spans are not successful state
predictions. This conditional accuracy is not report factuality or complete
negation understanding. Current common gold scope contains 2,833 entities and
1,775 relations. The 29 native Location-Attribute entities, 305 part_of
relations, native normality/action/change attributes and other extra dimensions
are not scored by this extractor/projection. Seven XL measurement entities and
ten touching relations are explicitly outside the common gold-label scope.
No image truth, EHR truth, Comparison/History scope, clinical severity accuracy,
or modality-error localization is established here.

V2 output:
`artifacts/protected/tricompose_v1_2/cxrgraph_extraction_runs/manual_xl_12766754_002/`.
Its `evaluation.json` holds aggregate and per-domain metrics, and
`prediction_receipts.json` holds opaque availability/hash receipts. Do not
display `native_graphs.json`, private worker logs or patient-level annotations.
Run manifest SHA256:
`a56743b0a3a940184faebbc7a39b0947d227d7f9ed3ffadc00592e10bed7898c`.

An independent set/count auditor replayed 400 report-level entity/relation
metric cells across the overall and two domain summaries, per-label/per-relation
partitions, polarity confusion/coverage and source/artifact hashes. The first
run's 50 exact-token predictions and count metrics were unchanged. This is a
numeric replay on stored character coordinates; alignment semantics are tested
separately, not independently reconstructed from reports by this auditor.
Audit: `cxrgraph_extraction_audits/numeric_12766754_001/audit.json`, SHA256
`6339e5d705d56e23c10bffb151f1f2d6eb5a48626d4ee5db609c80b612be3ff5`.
Fifteen new cohort-worker tests and sixteen literal-character tests passed;
the complete V1.2 suite passed **2,782 tests** in 15.648 seconds. All new private
directories/files are 2770/0660, project-group owned and Git-excluded. The
historical 80-EHR bank manifest still matches its pinned original SHA256.

Interpretation: extraction plumbing and tokenization interoperability are now
tested on actual manual annotations. The relationship score remains materially
below entity accuracy, and checkpoint training overlap is unresolved. These
results therefore remain development diagnostics, not independent scorer
qualification or evidence that synthetic triples are clinically consistent.
Next: probe the specific polarity/uncertainty and relational decisions needed
for contradiction detection, preserve unsupported dimensions as unavailable,
and resolve held-out membership before using accuracy as paper-primary evidence.
No winner promotion, score inversion, fitted threshold or automatic repair was
triggered by this diagnostic.

### Assertion-specific reliability and fixed-reader masks (2026-10-07)

The next step reused the **unchanged existing** authored language challenge:
56 invented reports, 80 designated four-finding checks, and 224 full-vector
checks retained only as secondary diagnostics. No case edits, favorable-family
filtering, new reference judgments or threshold fitting. The four heads are
cardiomegaly, consolidation, pleural effusion and pneumothorax; this does not
qualify extraction for every disease or every TriCompose edge. Expected states
are the previously disclosed same-author policy, not clinician/image truth.

`radgraph_assertion_readout.py` implements a diagnostic-only readout of native
observation states covering literal finding heads. Pleural effusion requires an
explicit pleural phrase; generic normality is not expanded to negative findings.
Measurements/anatomy are not disease states. Missing head evidence is unknown;
uncertain or mixed native mentions remain uncertain. Exact character alignment
is reused without altering native predictions. No new negation/scope correction,
synonyms, fitting, trained scorer or clinical decision policy.

`benchmark_radgraph_assertions.py` ran the frozen official offline XL on all
56 invented inputs in existing CPU allocation 12766754: 56 calls, all complete,
8.957 seconds, peak RSS 1.600 GiB. Predictions were fsynced before opening the
authored reference key and previous baseline predictions. Readers did not
receive labels or baseline answers. The study itself is already inspected
development data, not a blinded or independently held-out study. CheXbert and
Qwen predictions were reused without rerunning either model or changing them.

| Same 80 designated language checks | RadGraph-XL literal readout | Frozen CheXbert | Frozen Qwen span V2 |
|---|---:|---:|---:|
| Four-state matches | 61/80 (76.25%) | 48/80 (60.00%) | 50/80 (62.50%) |
| Four-state macro F1 | 0.7750 | 0.6009 | 0.5930 |
| Positive/negative flips | 3 | 7 | 8 |
| Determinate output on uncertain/unknown reference | 12 | 13 | 18 |
| Uncertain-reference recovery | 15/20 | 8/20 | 2/20 |
| Unknown-reference recovery | 14/23 | 14/23 | 19/23 |

Paired designated checks: RadGraph vs CheXbert had 44 both correct, 17
RadGraph-only correct, four CheXbert-only correct and 15 both wrong. Against
Qwen: 41 both correct, 20 RadGraph-only, nine Qwen-only and ten both wrong.
These are conditional language-test outcomes, not general clinical superiority.
All family/finding results remain in the protected evaluation; the secondary
unknown-heavy 224-check score is not substituted for the primary 80 checks.

The cached CXRGraph manual result was also broken down by assertion state,
including unmatched spans in the denominator. End-to-end exact-span/state
recall is 992/1,116 = 88.89% for positive, 396/413 = 95.88% for negative,
and **86/112 = 76.79% for uncertain**, not the aggregate conditional 98.33%.
Unmatched gold spans are 118/17/7 respectively. Among exact matched spans there
are three positive-to-negative flips and 19 determinate outputs on uncertain
gold. Missing spans are not assumed negative or treated as clinical unknown
gold. No raw manual report was reopened for these aggregate computations.

Native assertion run, relative to `artifacts/protected/tricompose_v1_2/`:
`radgraph_assertion_runs/authored56_12766754_001/`.
`evaluation.json` contains all three-reader comparisons;
`manual_assertion_reliability.json` contains denominator-aware actual-manual
results. Manifest SHA256:
`317ba52eda0e258ec13bba5d64a1cb2e9763e194263cc7d2ebadeacbee20a993`.

A separate **post-hoc development** analysis evaluates seven fixed agreement
masks: each individual reader, all three pairwise agreements, and unanimity.
Only agreement on a positive or negative state authorizes a determinate
proposal. Unknown, uncertain, disagreement and unavailable readers withhold
that proposal; unknown agreement is not successful factual support. No mask
was picked as a best policy and none was deployed to the candidate bank.

| Fixed mask | Determinate proposals / 80 | Errors among proposals | Positive/negative flips | Determinate on uncertain/unknown |
|---|---:|---:|---:|---:|
| RadGraph | 47 | 15 | 3 | 12 |
| CheXbert | 46 | 20 | 7 | 13 |
| Qwen | 55 | 26 | 8 | 18 |
| RadGraph + CheXbert agree | 27 | 4 | 0 | 4 |
| RadGraph + Qwen agree | 25 | 0 | 0 | 0 |
| CheXbert + Qwen agree | 32 | 10 | 2 | 8 |
| All three agree | 19 | 0 | 0 | 0 |

Zero errors on 25 or 19 accepted, correlated authored checks does **not** mean
zero population/clinical error. Coverage falls to 31.25% or 23.75%; abstention
is not a correct prediction on the remaining checks. The 37 explicitly
determinate authored references remain the separate denominator for correct
determinate recall. Agreement among report readers is dependent evidence, not
independent image/EHR truth or proof of the faulty modality. No calibrated
confidence weight, decision threshold, score inversion or clinical gate is
derived from this small inspected challenge.

The analysis independently replayed 560 mask decisions/counts and 240 primary
state-matrix cells directly from frozen predictions/references. Output:
`assertion_agreement_runs/seven_masks_12766754_001/`, manifest SHA256
`4039886e151d2ee218c25b67f65ee34757e531e18012f93fcc172c0185b0d056`.
The complete suite passed **2,819 tests** in 15.564 seconds, including 24 new
readout/reliability tests and 13 agreement-mask tests. Output hashes, 2770/0660
permissions, project-group ownership, Git exclusion and the historical 80-EHR
bank hash were rechecked. No new Slurm submission, GPU inference, download,
external API, training, live score-table change, winner promotion or regeneration.

Consequence: native uncertainty handling is more promising on this language
challenge, but determinate errors remain, and actual-manual uncertain recall
is substantially below aggregate matched-span accuracy. Reader-agreement masks
show a measurable risk/coverage trade-off worth testing on independent real
annotation. They are diagnostic candidates, not qualified automatic-repair
rules. Next validation must address actual report scope and checkpoint-heldout
membership before these evidence proposals can drive clinical contradiction
penalties or error localization. Image and EHR verification remain separate.

### Reader agreement on cached real manual labels (2026-10-07)

The authored-mask result was followed by a **real-manual, cached count audit**,
not another authored test or a new model run. Reuse the immutable official-label
CheXbert diagnostic `official_chexbert_reports_12594397` and Qwen span V2
`official_span_v2_12645961`, with their unchanged strict Impression inputs.
All 687 annotation entries remain represented: 267 unlinked, 393 non-test,
12 parser rejections, and exactly 15 shared complete study reports. Four fixed
heads yield 60 checks: 26 explicit positive/negative references, 34 unknown,
**zero uncertain reference examples**. Neither 687 nor 60 is a patient count.

Three fixed masks are audited: each reader alone, and their positive/negative
agreement. Unknown/uncertain agreement withholds a determinate assertion; it is
not factual support. No most-favorable mask is chosen or deployed. This is an
already-inspected development resource; it does not test the RadGraph + Qwen
mask on real data, because no RadGraph prediction on these identical inputs
exists in the two cached runs.

| Fixed mask | Determinate proposals / all 60 checks | Correct known proposals / 26 | Known polarity flips | Determinate on unknown reference |
|---|---:|---:|---:|---:|
| CheXbert | 26/60 (43.33%) | 26/26 (100.00%) | 0 | 0 |
| Qwen span V2 | 27/60 (45.00%) | 20/26 (76.92%) | 2 | 5 |
| CheXbert + Qwen agree | 20/60 (33.33%) | 20/26 (76.92%) | 0 | 0 |

The Qwen mask commits on 22 known-reference checks, of which two are opposing
positive/negative predictions: conditional known error is **2/22**, not 2/27
or 2/60. Five other determinate proposals occur on unknown manual references:
these are annotation-policy promotions, not independently adjudicated clinical
hallucinations. The pair abstains on 34 unknown checks, four uncertain outputs
on determinate references, and two opposing-polarity checks; all sixty stay in
the coverage denominator.

**Negative method result:** the pair removes Qwen's two polarity flips, but
does not improve on the same-input CheXbert baseline, which already matches all
26 known labels here. It loses six known assertions. Thus “zero errors among
agreed outputs” does not establish added value from a second reader. The
authored seven-mask result does not justify making multi-reader unanimity the
default score, declaring it clinical truth, or paying for extra verification
calls everywhere. Conversely, this tiny, overlap-unresolved resource does not
establish CheXbert's general superiority or universal clinical qualification.

No original report, CXR, annotation/linkage row, source key, model weight or
native graph was reopened. The worker reads only pinned sanitized prediction
receipts and aggregate manual confusion tables. Exact source-report and
Impression hashes match for all fifteen paired inputs. Reference/source
metadata bindings, full paired-confusion equality and native prediction
marginals are checked. Pair reference counts are identified here only by the
special fact that CheXbert's existing four-state manual confusion is exactly
diagonal, checked against the entire cached two-reader joint matrix. No
per-record reference label is reconstructed or exported, and model predictions
are not installed as a new gold dataset. For a nonidentity baseline, marginal
confusions generally do not identify joint-mask accuracy; the new module
returns **null**, not guessed pair correctness. This shortcut is a count-proof
for this snapshot, not a general clinical ensemble rule.

New, diagnostic-only implementation:
`TriCompose-v1.2/src/tricompose_v12/cached_manual_agreement.py`;
worker `TriCompose-v1.2/tools/audit_cached_real_agreement.py`.
Twenty invented-metadata tests passed, including unknown/uncertain abstention,
failed/skipped inventory retention, hash alignment, reference/prediction margin
checks, nonidentity-baseline nulls, and a mismatched joint matrix with preserved
marginals. The full V1.2 suite passed **2,839 tests in 15.501 seconds**.

Completed output, relative to `artifacts/protected/tricompose_v1_2/`:
`real_agreement_audits/cached_manual15_12766754_001/`, containing
`frozen_plan.json`, `evaluation.json`, `summary.json`, `RESULTS_CN_EN.md`,
`manifest.json` and a private worker log. Manifest SHA256:
`d3d521c7213419a83adfa78a8c829c094a691b9f887fb5875792de97fff4f89a`.
The existing approved CPU job 12766754 completed the cached audit in 0.030
seconds, peak RSS 0.023 GiB, with 180 independently replayed mask checks.
Artifact/source pins, 2770/0660 project-group permissions, Git exclusion and
the original 80-EHR bank hash were rechecked. Zero new model calls, GPU work,
Slurm submissions, downloads, APIs, threshold fitting or changes to the old
score table/winners. Consumed implementations and prior runs remain immutable.

Consequence: reliability validation must measure both accepted-error risk and
lost factual coverage, and compare against the strongest fixed reader, not
only against a weaker reader or the authored examples. Missing uncertain gold,
report-section/temporal-scope equivalence, checkpoint training overlap and
independent image/EHR evidence remain unresolved. These results authorize no
automatic clinical contradiction penalty, fault localization or regeneration.

### Fixed 100-report human-span reader comparison (2026-10-07)

The next diagnostic runs a **second frozen reader on the same fixed CXRGraph
manual test reports**, rather than extending the small fifteen-report resource
or selecting favorable cases. All 100 released reports remain: 50 MIMIC and
50 CheXpert, in their original release order. Reuse the immutable cached native
RadGraph-XL graphs; run the existing unchanged CheXbert checkpoint/adapter on
those exact released token strings. No image, EHR, generator identity, gold
finding states or baseline predictions enter CheXbert requests. The original
JSON bundles text and annotations, so the job is not an operator-blinded study;
it extracts only sentence tokens for model requests, and serializes/fsyncs
CheXbert predictions before projecting manual reference states or analyzing
the cached RadGraph baseline.

Reference scope is deliberately narrow and must not be mislabeled: project
human observation spans that cover the **unchanged four literal finding
patterns** used by the existing RadGraph assertion readout. Labels come from
human spans, not model predictions. Generic normality, anatomy and measurements
do not imply findings; no synonyms, plural variants, negation correction or
temporal/person scope rules are added. Unknown means no annotated literal-head
observation in this projection, **not** a gold clinical absence/unmentioned
finding under the complete CheXpert report policy. Mixed native assertions
remain uncertain. The actual ten uncertain projected references all contain
native uncertain human spans; none comes solely from pooling opposite
determinate mentions. These are derived human-span references, not newly
obtained official fourteen-finding report labels or image truth.

All 100 outputs completed for both readers, giving 400 attempted four-head
checks: **21 positive, 125 negative, ten uncertain and 244 literal-unknown**.
Per-finding support: cardiomegaly 11 positive/89 unknown; consolidation two
positive/34 negative/six uncertain/58 unknown; pleural effusion six
positive/33 negative/two uncertain/59 unknown; pneumothorax two positive/58
negative/two uncertain/38 unknown. Unknown-heavy overall accuracy is not a
headline metric; 125/146 determinate references are negative, and positive
recovery must be shown separately. These checks are not independent patients.

| Frozen reader | Positive recovery | Negative recovery | Uncertain recovery | Known positive/negative flips | Determinate on uncertain |
|---|---:|---:|---:|---:|---:|
| RadGraph-XL literal readout | 20/21 | 125/125 | 8/10 | 1 | 2 |
| CheXbert | 21/21 | 121/125 | 4/10 | 1 | 6 |

The three fixed determinate-proposal masks were evaluated without choosing a
winner, changing thresholds or deploying a mask to the synthetic bank:

| Fixed mask | Determinate proposals / 400 | Correct known proposals / 146 | Known polarity flips | Determinate on uncertain | Determinate on literal-unknown |
|---|---:|---:|---:|---:|---:|
| RadGraph | 150/400 (37.50%) | 145/146 | 1 | 2 | 2 |
| CheXbert | 223/400 (55.75%) | 142/146 | 1 | 6 | 74 |
| RadGraph + CheXbert agree | 145/400 (36.25%) | 141/146 | 0 | 2 | 2 |

The pair's accepted known-reference error is 0/141, not 0/400. It filters the
two readers' separate known polarity errors, but retains **two determinate
outputs on genuinely uncertain human spans**. It also withholds four known
facts RadGraph got right, yielding 141/146 known recovery rather than
145/146. The 74 CheXbert determinate outputs on literal-unknown references
cannot be called 74 clinical hallucinations: the complete classifier can
recognize synonyms or global normality outside the restricted literal gold
projection. This asymmetric vocabulary scope favors the literal readout on
unknown handling; it prevents a general superiority claim from these numbers.
The uncertainty finding is a language-state diagnostic, not demonstrated image
hallucination or permission to blame/regenerate a modality.

Domain masks retain their own denominators. MIMIC has 84 determinate references:
RadGraph 83 correct/one flip, CheXbert 80 correct/one flip, pair 79 correct/zero
flips. CheXpert has 62: all three masks retain all 62 known states, but the pair
still commits on two uncertain and two literal-unknown references. No domain,
finding or source was dropped after observing an outcome. Full four-state
matrices, failures and head/domain results remain in the protected evaluation.

Implementation: `src/tricompose_v12/manual_literal_assertions.py`,
`tools/benchmark_manual_literal_readers.py`, and
`slurm/80_manual_literal_readers_existing_cpu.sh`, relative to `TriCompose-v1.2/`.
The latter is **not sbatch**: it reuses approved CPU allocation 12766754
(four CPUs/32 GiB, no GPU), limits model threads to two and execution to ten
minutes, and checks the actual Slurm cgroup. CPU CheXbert used the previously
pinned checkpoint/adapter, seed zero, eval/frozen weights, offline/blocked
network, batches of four, no retries, and no truncation. It made 100 encoder
examples in 25 batches: 23.481 seconds inference, 35.707 seconds total,
2.053 GiB peak RSS. RadGraph made **zero** new forward calls. No new Slurm
submission, GPU queue, download, training, API or synthetic generation.

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`manual_literal_reader_runs/manual100_12766754_001/`, with frozen plan,
CheXbert/RadGraph prediction receipts, human reference projection, full and
domain evaluations, mask details, summary, manifest and private runtime/logs.
Manifest SHA256:
`030ac27c56217db264bb09c93cba6a3ec2ae7b5322c17323a918fd89ded01a5f`.
The reference/provenance receipts contain opaque indices, states, hashes and
offsets, never original report text, patient/source keys or images. Do not
display the reused native graph cache or private worker logs.

Independent numeric audit: `manual_literal_reader_audits/numeric_12766754_001/`.
Its **`score_table.csv`** lists all three masks for all/MIMIC/CheXpert with
positive, negative, uncertain and unknown denominators, proposal coverage,
known recovery and flips. The auditor replays 1,600 reader matrix checks,
2,400 mask decisions across the full/domain partitions, and 177 human evidence
bindings against existing exact manual character spans. It opens no original
report/annotation row or native graph text; literal regex/source semantics are
tested with invented fixtures, not independently reread by this auditor.
Audit manifest SHA256:
`a726dc4b550ca37dcc660eda06b6ad9dc6bd2cfa24b2e7ee5349f831f1e9f9ea`.

Twenty-four projection/risk tests and six request-plumbing tests passed. The
complete V1.2 suite passed **2,869 tests in 15.284 seconds**. Artifact and
consumed-source hashes, 2770/0660 project-group permissions, Git exclusion,
and the historical 80-EHR bank manifest were verified. All consumed code,
models and earlier outputs remain unchanged; no old score/winner was promoted.

Consequence: reader agreement reduces some determinate polarity errors but
does **not** guarantee correct uncertainty. The paper should distinguish
unknown-safe coverage, uncertainty commitments, positive/negative recovery
and the vocabulary/scope of its references. This report-state test does not
establish cross-modal error localization or targeted-repair success. Training
membership, semantic/current-patient scope, broader disease coverage and
independent image/EHR evidence remain unresolved; clinical scores and repair
authorization stay unavailable. Next work must address those missing
independent evidence dimensions rather than interpret the narrow high known
recovery as general clinical calibration.

### Official third-reader diagnostic on the same 100 reports (2026-10-07)

Completed the next fixed benchmark with the **unchanged official CheXpert
rule-based classifier / NegBio parser**, not another invented clinical scorer.
The existing sealed Python 3.6.7 environment, native rules, models and resources
were reused. A new real-manual-specific wrapper calls the frozen generic parser
components; it does not modify or bypass the older authored-only CLI/guard.
All 100 released reports remain in the original order. Parser requests contain
only exact released sentence-token text, an opaque index and its SHA256: no
gold states, native annotations, images, EHR, generator identity or other
readers' predictions. Bundled source annotations are read internally with the
source JSON but are not used by the parser. Predictions and full repeat outputs
are closed/fsynced before reference or other-reader analysis. This is an
unblinded, reused development benchmark, not a new independent clinical test.

The native primary run attempted every report twice: **200 parser passes,
99 complete reports and one failed/unavailable**, with zero full-evidence
replay differences. The repeated failed parse is still failed, not reproducible
success. Its reason is `parse_tree_unavailable`; its projected reference checks
are three negative and one literal-unknown. All four remain in the denominators
as unavailable, with null label vectors, rather than becoming negatives,
unknown successes or dropped cases. There are no selective retries.

All seven predeclared masks are retained: three single readers, three pairs
and the all-three agreement. A mask proposes a determinate finding only when
all participating readers are available and agree on positive or negative;
unknown, uncertain and disagreement abstain. The reference remains the exact
same narrow four-literal-head human-span projection: **21 positive, 125
negative, ten uncertain and 244 literal-unknown**. It is not full clinical
fourteen-finding gold, patient-scope gold, image truth or EHR truth. The narrow
literal vocabulary makes unknown handling asymmetric against broad classifiers.
No mask, domain or readout was selected to improve the observed result.

Primary results, using the official native aggregated labels:

| Fixed mask | Correct positive / 21 | Correct negative / 125 | Known polarity flips | Determinate on uncertain / 10 | Determinate proposals / 400 | Determinate on literal-unknown / 244 |
|---|---:|---:|---:|---:|---:|---:|
| RadGraph-XL literal readout | 20 | 125 | 1 | 2 | 150 | 2 |
| CheXbert | 21 | 121 | 1 | 6 | 223 | 74 |
| Official CheXpert/NegBio | 21 | 112 | 7 | 7 | 212 | 65 |
| RadGraph + CheXbert agree | 20 | 121 | 0 | 2 | 145 | 2 |
| RadGraph + CheXpert/NegBio agree | 20 | 112 | 0 | 2 | 136 | 2 |
| CheXbert + CheXpert/NegBio agree | 21 | 112 | 1 | 5 | 199 | 60 |
| All three agree | 20 | 112 | 0 | 2 | 136 | 2 |

Consequently, adding the third reader to the original pair **loses nine correct
negative proposals**, reducing known recovery from 141/146 (96.58%) to 132/146
(90.41%). It removes neither of the pair's two determinate outputs on genuinely
uncertain human spans. Both pair and triple have zero accepted known polarity
flips, with denominators 141 and 132 accepted known proposals, respectively;
this does not imply zero uncertainty/scope/image errors. Total proposal
coverage is 145/400 (36.25%) versus 136/400 (34.00%). The lost proposals include
failed-reader abstention, not only differences in available labels. There is
no justification here for making the extra reader mandatory or promoting
the zero-flip masks to clinical contradiction/repair policies.

The two source domains retain all 50 reports each. MIMIC has 84 known states:
the original pair recovers 79, the all-three mask 72. CheXpert has 62: the pair
recovers all 62, the all-three mask 60. The same two uncertain commitments
remain in the CheXpert partition. Complete per-head matrices, all masks,
four-state support and availability counts remain in the protected outputs.
Determinate outputs on literal-unknown references are **not** automatically
clinical hallucinations; broad synonyms/global report normality are outside
the restricted human projection, as in the preceding two-reader comparison.

The already-defined `mention_conflict_view` is a **separate secondary readout
of the same native parse**, not a fourth reader or replacement official metric.
It recovers the same 133/146 known states and three/ten uncertain states as
the native parser. Native known flips fall from seven to one because six wrong
negative-reference positives become uncertain/abstained, not correctly
recovered negatives. Its seven determinate-on-uncertain outputs remain.
The all-three mask is unchanged at 132/146 with two uncertain commitments.
Both views and every mask are reported; no better-looking view was deployed.

Execution reused approved CPU allocation **12766754**, four CPUs/32 GiB,
no GPU. New runner `slurm/81_manual_negbio_existing_cpu.sh` is not sbatch and
checks the actual job cgroup, loads the existing Java module, limits the
unchanged parser to ten minutes, disables network and redirects logs/caches/
temporary files to the private run. Total native worker time was **276.927
seconds**, including 10.307 seconds startup/initialization and 264.449 seconds
parsing; peak worker RSS was 1.308 GiB. No new Slurm submission, download,
training, API, image generation or old winner/score change occurred. Cached
analysis used zero model calls and took 2.121 seconds.

Paths below are relative to `artifacts/protected/tricompose_v1_2/`:

- Native parse and repeat receipts:
  `manual_negbio_runs/manual100_12766754_001/`.
  Manifest SHA256:
  `97878f07cdd76853d983fd9c778a02e32d23cedef63a699397a17d3025a7ae82`.
- Full/domain matrices and **42-row `score_table.csv`**, covering two readouts
  x three domains x seven masks:
  `manual_three_reader_runs/seven_masks_12766754_001/`.
  Manifest SHA256:
  `de2439533772e30a68714a7eef6c18b3970c52afee03360aa8e50b45be2fef78`.
- Independent numeric/evidence audit:
  `manual_three_reader_audits/numeric_12766754_001/`.
  Manifest SHA256:
  `a5575d29cb9d996e321a2c4de0aca7bda242eeec45ca1d3c3def19d0c3f91ae5`.

Implementation, relative to `TriCompose-v1.2/`:
`interfaces/chexpert_negbio_manual100_v1.py`,
`src/tricompose_v12/manual_three_reader_agreement.py`,
`tools/score_manual_three_readers.py`, and
`audits/audit_manual_three_readers.py`.
The independent auditor does not import the production decision/statistics
functions: it replays **4,800 reader matrix checks, 11,200 mask decisions**,
100 full native repeat bindings and all 42 CSV rows; it confirms the nine lost
correct negative proposals and two retained uncertain commitments. Original
source bodies/annotation rows/native graph text are not decoded by the audit;
source pins are hashed as bytes only. It does not independently reconstruct
literal semantic gold. Source/artifact pins, project-group 2770/0660 modes,
Git exclusion and the unchanged original 80-EHR bank manifest were verified.
The 27 parser/mask tests and 12 independent arithmetic tests passed; the complete
V1.2 suite passed **2,908 tests in 15.719 seconds**. Consumed code, earlier
benchmarks and candidate-bank outputs remain immutable.

Next development implication: **reader count is not independent evidence**.
Agreement can preserve a shared uncertainty error while suppressing correct
facts. Keep uncertainty, unknown, availability, accepted-error risk and lost
positive/negative coverage separate in the proposed evidence table; do not
collapse these counts into an uncalibrated clinical score. A natural clinical
fault-localization claim still needs independent, scope-matched image/EHR
references, resolution of checkpoint membership, and an evaluation separate
from these reused development reports. This diagnostic neither qualifies an
automatic clinical penalty nor authorizes changing the existing winners or
regenerating a modality.

### Uncertainty-aware risk/coverage and clustered intervals (2026-10-07)

The next completed diagnostic makes the preceding masks' risk explicit rather
than interpreting **zero known polarity flips** as zero error. No new reader,
clinical threshold, reliability weight, inference or repair was introduced.
Only the sealed human-literal projection and seven-mask decisions were used;
raw reports, source annotation rows, original patient keys, images and native
graph text were not decoded. The original reference states, failed parse,
historical masks, candidate bank, raw scores and winners remain unchanged.

The new endpoint is **conditional literal assertion error**, not a calibrated
clinical score or probability:

```text
numerator   = known positive/negative flips
            + determinate proposals on uncertain human-literal references

denominator = determinate proposals on positive, negative or uncertain
              human-literal references
```

Literal-unknown proposals do not enter either term: this restricted vocabulary
does not adjudicate those proposals as clinical false findings. Their count
and reference support remain separate. Failed/unavailable outputs and
abstentions are not accepted errors or successes; they remain in full
coverage and known-reference recovery denominators. Report positive and
negative recovery separately. Risk cannot be improved meaningfully by
withholding everything; conditional risk becomes unavailable at zero accepted
adjudicable proposals, not zero. This is still a language-state endpoint, not
EHR fidelity, image-grounded hallucination or correct modality localization.

Using the same 100 distinct report artifacts, draw **2,000 report-clustered
bootstrap replicates, seed zero**, separately preserving each source's 50
reports. All four heads, seven masks and both native/secondary readouts share
the same paired draws. Bootstrap units are reports, **not verified patient
groups** or independent finding rows. The same report's heads and decisions
stay together. Canonical opaque index ordering and a recorded PCG64/Numpy
1.26.4 sampling hash make the draws reproducible. The ninety-five-percent
percentile intervals are exploratory, conditional on these reused development
reports and the literal reference projection. They do not resolve patient
dependence, checkpoint overlap, reference vocabulary/scope, dataset shift or
paper-grade independence. No multiplicity-adjusted significance claim is made.

Primary native-label results:

| Fixed mask | Literal errors / adjudicable proposals | Conditional literal error | Report-cluster percentile interval | Correct known / 146 |
|---|---:|---:|---:|---:|
| RadGraph-XL literal readout | 3/148 | 2.03% | 0.00–4.64% | 145 |
| CheXbert | 7/149 | 4.70% | 1.38–9.09% | 142 |
| Official CheXpert/NegBio | 14/147 | 9.52% | 5.09–15.13% | 133 |
| RadGraph + CheXbert agree | 2/143 | 1.40% | 0.00–3.55% | 141 |
| RadGraph + CheXpert/NegBio agree | 2/134 | 1.49% | 0.00–3.82% | 132 |
| CheXbert + CheXpert/NegBio agree | 6/139 | 4.32% | 1.43–8.15% | 133 |
| All three agree | 2/134 | 1.49% | 0.00–3.82% | 132 |

The two-reader proposal denominator 143 means 141 accepted known states plus
two determinate outputs on uncertain references, excluding two literal-unknown
outputs. The triple denominator 134 means 132 accepted known states plus the
same two uncertain commitments. The known polarity error alone remains zero
for these masks, but the uncertainty-inclusive literal error is **not zero**.
Neither risk is the probability that an entire synthetic triple is wrong.

Three fixed paired contrasts are reported for all domains and both readouts;
differences below are right minus left and expressed in **percentage points**:

| Native primary contrast | Literal-risk difference (percentile interval) | Known-recovery difference (percentile interval) |
|---|---:|---:|
| RadGraph to RadGraph + CheXbert | -0.63 (-2.12, +0.10) | -2.74 (-5.63, -0.67) |
| CheXbert to RadGraph + CheXbert | -3.30 (-7.13, -0.64) | -0.68 (-2.22, 0.00) |
| RadGraph + CheXbert to all three | +0.09 (0.00, +0.31) | -6.16 (-11.59, -2.05) |

Thus the pair's apparent risk reduction relative to the stronger single
RadGraph reader has an interval containing zero; it cannot be claimed as a
reliable general improvement. Adding the third reader retains the same
uncertain errors while losing correct known facts. Do not choose the
best-looking reader/mask/readout after observing these diagnostics or turn
their rates into synthetic-case weights. Perfect observed recovery can yield
a degenerate percentile interval (e.g. 21/21 positive recovery); that does
**not** establish perfect population performance or bound unseen errors.
Zero-denominator resamples are reported as missing with explicit valid/missing
draw counts; they never contribute fabricated zero-risk observations.

Implementation, relative to `TriCompose-v1.2/`:
`src/tricompose_v12/manual_reader_risk_coverage.py`,
`tools/analyze_manual_reader_risk_coverage.py`, and
`audits/audit_manual_reader_risk_coverage.py`.
Only the current existing CPU Slurm allocation **12784259** was used; the
worker checks the actual cgroup and makes no new submission. Cached analysis
and deterministic full replay completed in **0.452 seconds**, peak RSS
0.057 GiB. Those are arithmetic timings, not model inference or demonstrated
inference-time savings. No download, training, GPU, API or source-body read.

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`manual_reader_risk_coverage/clustered_12784259_001/`.
It contains `frozen_plan.json`, `report_cluster_counters.json`,
`evaluation.json`, **42-row `risk_coverage_table.csv`**, 378-row
`all_intervals.csv`, 162-row `paired_contrasts.csv`, summary and manifest.
The counters retain opaque report indices and hashes only, never original
source keys or text. Output is atomically committed and existing runs are
refused. Manifest SHA256:
`48da1182d159d57db0d826b10d630c19741baba58ca176f643dc4704d40033dc`.

Independent arithmetic audit:
`manual_reader_risk_coverage_audits/numeric_12784259_001/`.
Without importing production analysis functions, it reconstructs **5,600
finding decisions and 1,400 report-counter vectors**, replays all 2,000 shared
bootstrap draws, and checks all 378 scalar and 162 paired percentile intervals
with separate scalar interpolation. This is a numeric/source-binding audit,
not independent semantic gold or clinical adjudication. It took 0.994 seconds.
Audit manifest SHA256:
`f74185da6ec2bf5f7674e50f7fbea5bf8530e3bbee6772c878053d963d47928d`.

Thirty new risk/plumbing tests and six independent percentile tests passed.
The full V1.2 suite passed **2,944 tests in 14.118 seconds**. The new artifacts
remain Git-excluded with project-group directories 2770/files 0660, and the
source/artifact pins and unchanged original bank were verified. Consumed
implementations and previous experiments remain immutable. This closes the
arithmetic/uncertainty-accounting gap, not the independent clinical-evidence
gap; automatic fault localization, clinical score calibration and targeted
repair are not authorized by these results.

## Cached image-reader consensus risk/coverage (2026-10-07)

Continue the same reliability-first plan on the **already fixed RICORD-1C
50-case opacity diagnostic**, without a new image/model pass. The reference is
the previously recorded unanimous classification of three comprehensive
annotation groups, **not XRV/BioViL agreement** and not a reproduction of the
official majority/adjudication endpoint. Each case belongs to a different
patient according to the pinned acquisition receipt. The present cached
analysis compares recorded input hashes and does not reopen pixels, DICOM
headers, raw annotation rows, reports or original patient keys. It therefore
does **not independently reverify** the prior patient/source binding.

The fixed inventory remains 25 opacity-positive and 25 opacity-negative
references. Compare all four predeclared masks, not a selected winning mask:

1. Exact XRV **LungOpacity** output at the unchanged 0.5 operating point.
   Do not substitute the historical max(LungOpacity, Infiltration) proxy or
   transplant that proxy's separate threshold into this exact output.
2. BioViL-T mean of the three previously fixed positive-minus-negative
   cosine margins, with positive/negative decided at zero; an exact tie
   abstains. No favorable template subset is selected.
3. Accept only XRV and the fixed BioViL mean agreeing on a determinate state.
4. Additionally require all three BioViL template margins to have the same
   nonzero sign. These are one model's template-sensitivity checks, **not
   three independent experts**.

Failure/unavailable results never become negatives. Abstentions remain in
all-planned coverage and class-specific reference-recovery denominators.
Accepted error risk is `(false positive + false negative) / accepted` and
becomes null, not zero, if no judgment is accepted. Preserve the full 50-case
denominator and both reference classes when describing selective risk.

| Fixed mask | Errors / accepted | Accepted error risk (95% percentile interval) | Coverage / 50 | Positive recovered / 25 | Negative recovered / 25 |
|---|---:|---:|---:|---:|---:|
| XRV exact LungOpacity, 0.5 | 6/50 | 12.00% (4.00–20.00%) | 50 (100%) | 25 | 19 |
| BioViL fixed three-template mean | 6/50 | 12.00% (4.00–20.00%) | 50 (100%) | 25 | 19 |
| XRV and BioViL mean agree | 3/44 | 6.82% (0.00–14.64%) | 44 (88%) | 25 | 16 |
| Agreement plus all-template stability | 2/40 | 5.00% (0.00–11.90%) | 40 (80%) | 25 | 13 |

All observed errors here are false-positive opacity judgments; the perfect
25/25 positive recovery does not establish perfect population sensitivity.
Although the two single readers have identical aggregate counts, they do
not make all the same errors: they disagree on six cases and share three
false positives. The agreement mask drops three correct negative judgments
along with three errors relative to either single reader. Requiring template
stability drops another three correct negatives and one shared error. Thus
this is **selective risk with a coverage cost**, not uniformly better
classification or a reliable majority-vote clinical oracle.

Replay **2,000 class-stratified paired case bootstrap draws, seed zero**.
Preserve the 25/25 reference balance and use the same draws for all policies
and all seven metrics. Cases correspond to previously verified patient
groups; current raw patient grouping is not newly inspected. Numpy 1.26.4,
PCG64 sampling hash:
`b8dc015c5b55e9451fc0fb4224f1ed2a68cbb1ad4a4c0c55124148060e450f14`.
Fixed contrasts below are right minus left, in **percentage points**:

| Contrast | Accepted-risk difference (95% percentile interval) | Coverage difference |
|---|---:|---:|
| XRV to two-reader agreement | -5.18 (-11.81, +0.20) | -12 |
| BioViL mean to two-reader agreement | -5.18 (-11.68, +0.33) | -12 |
| Two-reader agreement to template-stable agreement | -1.82 (-6.74, +0.87) | -8 |

All three risk-difference intervals include zero. Do not claim a reliable
general improvement, select this dataset's best-looking mask, fit a threshold
on it or use these conditional rates as synthetic-triple correctness
probabilities. Intervals are unadjusted exploratory development intervals;
perfect observed recovery and degenerate bootstrap intervals do not bound
unseen errors. The class-balanced selection and unanimity reference exclude
many ambiguous cases, the cohort is small and previously inspected, model
training overlap is unresolved, and the display transform has not received
clinical adjudication. The readers share images and correlated evidence.
They also have different frozen preprocessing; this is not an isolated
architecture-only comparison. XRV's operating-point output and BioViL cosine
margins are not calibrated clinical probabilities.

Scope remains **exact opacity**, not pneumonia, a validated 14-finding
classifier, report factuality, EHR ground truth or natural-error modality
localization. Synthetic-domain transport is still unvalidated. Do not fill
missing EHR facts from this benchmark. Existing candidate banks, winner
files, thresholds and score tables are unchanged; no repair policy is
enabled. This result supplies one component's risk/coverage evidence alongside
the preceding report language-state evidence, not a full-triple clinical score.

Implementation, relative to `TriCompose-v1.2/`:
`src/tricompose_v12/image_reader_consensus.py`,
`tools/benchmark_cached_image_consensus_v2.py`, and
`audits/audit_cached_image_consensus.py`. The first launcher failed on a
nonexistent dependency path before creating a protected run or decoding cached
score rows. It is retained unchanged; V2 corrects the dependency inventory,
reusing the same fixed join/table helpers without changing references,
readers, thresholds, templates or endpoints. Actual arithmetic and
deterministic replay used existing CPU Slurm allocation **12784259**, taking
**0.145 seconds**, peak RSS 0.043 GiB. No new Slurm submission, GPU, model
call, training, download or external API. These timings are cached arithmetic,
not generation speed or demonstrated inference savings.

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`image_reader_consensus_runs/ricord50_12784259_002/`.
It contains the frozen plan, cached joined score metadata, 200 opaque-index
policy outcomes, aggregate evaluation, **four-row `risk_coverage_table.csv`**,
21-row `paired_contrasts.csv`, summary and manifest. Manifest SHA256:
`add0c915921d0e2fa0741d604608186ce67faf83d94ad81f370eddc3c5997c29`.

Independent arithmetic audit:
`image_reader_consensus_audits/numeric_12784259_001/`.
Without importing production consensus functions, it rechecks **50 cache
bindings, 200 decisions, 28 scalar intervals, 21 paired intervals and 25 CSV
rows**, using separate scalar counter reconstruction and percentile
interpolation. The same 2,000 draw hash is replayed. It verifies numbers and
source bindings, **not independent clinical semantics**. Runtime 0.555 seconds.
Audit manifest SHA256:
`4c4e1a29cf85dc3fbe2d8d24a731b8cfd077b2d69097892bb04d0753df357ab2`.

Fifty new invented-fixture tests passed. Full V1.2 suite: **2,994 tests in
14.418 seconds**, passed. Nine output artifacts and 26 source pins, protected
project-group directories 2770/files 0660, Git exclusion and the unchanged
original candidate-bank manifest were verified. Consumed code and results
remain immutable. 下一步仍要按 finding/domain 分开判断证据是否足够；模型一致
可以降低本组被接受判断的观察错误率，但不能被当作真值，也不能用拒判把
coverage 的损失隐藏掉。

## Report-reader finding/domain breakdown / 分项证据支持量 (2026-10-07)

Decompose the preceding pooled four-head report diagnostic using the **same
100 report artifacts, cached human-literal projections, three frozen readers,
seven fixed masks, two readouts and 2,000 paired report-cluster draws**. No new
model inference, source-report read, pixel read or reference annotation was
performed. This does not create an untouched test set or additional independent
cases. The primary readout remains `native_labels`; the mention-conflict view
remains secondary rather than whichever view looks best.

The pooled 146 known references are strongly uneven across heads. Unknown
here means **no annotated literal-head observation in this restricted
projection**, not clinical absence. These numbers are reference support,
not disease prevalence in the original population:

| Literal finding | Positive | Negative | Uncertain | Literal unknown | Known reference coverage / 100 |
|---|---:|---:|---:|---:|---:|
| Cardiomegaly | 11 | 0 | 0 | 89 | 11% |
| Consolidation | 2 | 34 | 6 | 58 | 36% |
| Pleural effusion | 6 | 33 | 2 | 59 | 39% |
| Pneumothorax | 2 | 58 | 2 | 38 | 60% |

Consequently, an observed perfect cardiomegaly result cannot demonstrate
negative-state specificity: **there are no negative references for that
literal head**. Consolidation and pneumothorax each have only two positive
references, so correct recovery of those two does not establish reliable
positive-state performance. In the MIMIC subset, consolidation has **zero
positive, 22 negative and three uncertain** literal references; the other
subset supplies both consolidation positives. Neither subset can borrow the
other's missing reference class. `reference_support.csv` retains all twelve
domain/head inventories, with count-based scope reasons and no invented
minimum-sample qualification threshold.

Primary native-label literal-error fractions on the combined development set
are below. Each fraction is `(known polarity flips + determinate commitments
on uncertain references) / determinate outputs on known or uncertain
references`. Literal-unknown outputs are not judged false clinical findings
and do not enter this risk denominator. Full proposal coverage, reference
recovery, uncertain commitments and unknown commitments remain separate.

| Literal finding | RadGraph-XL | CheXbert | Official CheXpert/NegBio | RadGraph + CheXbert agreement | Pair: correct known / all known |
|---|---:|---:|---:|---:|---:|
| Cardiomegaly | 1/11 | 0/11 | 0/11 | 0/10 | 10/11 |
| Consolidation | 0/36 | 3/37 | 4/36 | 0/34 | 34/36 |
| Pleural effusion | 0/39 | 2/40 | 4/40 | 0/38 | 38/39 |
| Pneumothorax | 2/62 | 2/61 | 6/60 | 2/61 | 59/60 |

Thus the pair's pooled **2/143** literal-error fraction is not one uniform
clinical reliability rate. Both surviving errors are determinate commitments
on uncertain pneumothorax references. By source, that head has pair risk
**0/34 with 34/35 known recovery in MIMIC**, versus **2/27 with 25/25 known
recovery in CheXpert**. This separates missed/withheld known judgments from
overconfident uncertain judgments; it does not prove a dataset-level cause or
justify a test-fitted rule. Adding the third reader retains these two errors
while reducing the pneumothorax adjudicable denominator to 56, not 61.

Every finding and domain receives all nine previously defined metrics and
the same three paired contrasts. Resampling stays at the **report artifact**
level, stratified 50/50 by source. Patient clusters remain unverified. All
four heads, masks and readouts share the original PCG64 draw hash:
`e46d905651f73b1d1ee852afed362210188cdca9f4aed6855cbf1f02e164ff31`.
This yields 168 risk-table rows, 1,512 scalar intervals and 648 paired metric
rows. Zero-denominator resamples and endpoints are missing, never fabricated
zeros. Rare reference classes can have missing resample denominators; valid
and missing draw counts remain explicit. Perfect observed recovery or a
degenerate zero-error percentile interval does not bound unseen errors.
These exploratory intervals are not multiplicity-adjusted significance
tests or clinical score calibration.

The head decomposition conserves **all 42 pooled point-counter rows and
all 1,400 per-report counter vectors** exactly. It does not change the
previous benchmark, reader states, thresholds, masks, candidate bank, score
table or selected triples. It does not apply reference-set error rates as
candidate weights, and it does not classify literal-unknown promotions as
hallucinations merely because the gold vocabulary is narrower.

Implementation, relative to `TriCompose-v1.2/`:
`src/tricompose_v12/manual_reader_head_risk.py`,
`tools/analyze_manual_reader_head_risk.py`,
`audits/audit_manual_reader_head_risk.py`.
Output, relative to `artifacts/protected/tricompose_v1_2/`:
`manual_reader_head_risk_coverage/finding_domains_12784259_001/`.
The readable outputs are **`reference_support.csv`** and
**`risk_coverage_table.csv`**; the complete intervals and contrasts are in
`all_intervals.csv`, `paired_contrasts.csv` and `evaluation.json`.
Opaque report/head counter receipts remain protected and contain no report
text, original patient keys or native graph bodies. Manifest SHA256:
`b6e0b853e0b0feaf72322fbef94e713dde7203b15ea5bb427a24ae211591d75c`.

Independent cached arithmetic audit:
`manual_reader_head_risk_audits/numeric_12784259_001/`.
It reconstructs 5,600 head/report vectors, verifies twelve reference-support
groups, replays all 2,000 draws, and independently checks **1,512 scalar
intervals, 648 paired intervals and 2,160 interval CSV rows**, without
importing production analysis functions. It reuses the prior independently
audited scalar percentile helper. This is not new semantic adjudication.
Audit manifest SHA256:
`bad337d6cc6e116af43cd8ea8e53af7d8da321c4e6c4c95ecbd9065404d9219e`.

Existing CPU Slurm job **12784259** only: analysis/replay 2.256 seconds,
peak RSS 0.090 GiB; independent arithmetic audit 3.530 seconds. No new job,
GPU, download, API, training or generation. Twenty-seven new invented-fixture
tests and the full **3,021-test V1.2 suite passed in 16.678 seconds**.
Ten output artifact hashes, thirty source pins, Git exclusion, unchanged
original bank and protected project modes 2770/0660 were verified. Consumed
code and output remain immutable.

Practical next boundary: image opacity evidence and these four report-literal
heads are **not automatically the same comparison vocabulary**. This report
reference has no LungOpacity head, and the current RICORD image reference
does not establish these four image findings. Do not combine their separate
error rates into a three-modal probability, call an agreement pattern correct
fault localization, or enable repair from pooled performance. Synthetic
transport, temporal/person scope, model-training overlap, image-grounded
truth and independent patient evaluation remain unresolved. 分项表的用途是让
每个分数的参考支持量与边界可见，而不是把缺失参考用平均数补齐。

## Fixed image masks on the complete candidate pool (2026-10-07)

Connect the four already fixed image-proposal masks to the **existing complete
80-EHR / 240-image-slot / 960-triple bank** using cached XRV and BioViL outputs.
This is a lossless candidate evidence overlay, not new generation, clinical
rescoring, candidate selection, acceptance/rejection or targeted repair.
It does not read the real-reference benchmark's risk values or labels and
does not use them as synthetic-case correctness probabilities or weights.
The sealed benchmark plan pins the same policies, exact XRV 0.5 head,
all-three BioViL mean and scope limitations only.

Keep every original **86-column cell and row order** from
`candidate_opacity_biovil_runs/biovil_opacity240_12670345/`.
Also verify all 66 older named columns against the preceding XRV cache.
Append eighteen columns: for each of four masks, an image proposal state,
status and relation to the **cached, unqualified report proposal**; six common
fields describe the narrow scope and deny clinical-primary eligibility,
reference-metric transfer, validated synthetic transport, selection and
regeneration. The result has **960 rows and 104 columns**. Unavailable mask
states are blank with explicit status, never negative or successful agreement.
Unknown and uncertain report states remain non-comparable.

| Fixed mask | Determinate image slots / 240 | Proxy support | Proxy opposition | Non-comparable / 960 | Comparable coverage | Proxy agreement among comparable |
|---|---:|---:|---:|---:|---:|---:|
| Exact XRV opacity, 0.5 | 240 | 173 | 95 | 692 | 27.92% | 64.55% |
| BioViL fixed three-template mean | 240 | 218 | 50 | 692 | 27.92% | 81.34% |
| XRV and BioViL mean agree | 184 | 159 | 36 | 765 | 20.31% | 81.54% |
| Agreement plus all-template stability | 178 | 150 | 33 | 777 | 19.06% | 81.97% |

These are proposal relations, **not confirmed clinical support, contradictions
or error counts**. No model output changed, so differences are scorer/mask
sensitivity, not image/report generation improvement. The two single-reader
rows reproduce the previous overlay's original point results. The pair
retains 88 positive and 96 negative image proposals; template stability
retains 82 positive and 96 negative. All 80 EHRs remain fixed and unknown
for this exact opacity comparison. Other historical EHR heads are not erased
or enriched by this single-finding overlay.

Relative to XRV alone, two-reader agreement withholds 56 image slots and
224 candidate slots: 14 proxy-support rows, 59 proxy-opposition rows and
151 already non-comparable rows. Relative to BioViL alone, the same withheld
slots instead comprise 59 proxy-support rows and 14 proxy-opposition rows,
plus 151 non-comparable. This illustrates the dependence of an agreement
claim on which unqualified image proposal is treated as the reference.
Adding template stability withholds six more image slots / 24 candidate
slots: **nine proxy-support, three proxy-opposition and twelve already
non-comparable rows**. It does not repair the three oppositions. Do not
credit lower opposition counts from abstention as successful regeneration.

All four report paths and their full 240-slot denominators remain in the
aggregate output; do not choose the highest conditional agreement while
ignoring different assertion coverage. The fixed bank contains **147 unique
image-byte hashes** across 240 image slots and **428 unique report-byte
hashes** across 960 report slots. All four mask readouts are consistent across
repeated image hashes in the cache. Duplicate artifacts, four reports per
image and reused EHRs are correlated evidence, not 960 independent patients
or 240 unique images. Counts are not a patient-bootstrap superiority result.

Implementation, relative to `TriCompose-v1.2/`:
`src/tricompose_v12/candidate_image_masks.py`,
`tools/annotate_candidate_image_masks.py`,
`audits/audit_candidate_image_masks.py`.
The cached wrapper checks original parent manifests, selected bounded
numeric/state artifacts and lineage joins. It **does not reopen source text,
image bytes, raw EHRs, patient keys or model assets**. Old source/model/pixel
provenance comes from the sealed receipts; current pixel/weight bytes are
not independently reverified. Outputs are atomic and refuse existing run IDs.

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`candidate_image_mask_overlays/pool960_12784259_001/`.
**`candidate_evidence_table.csv`** is the complete 104-column candidate table;
**`mask_comparison.csv`** is the four-row aggregate comparison. The plan,
image-mask records, per-report-model summary and manifest retain the audit
trail. Manifest SHA256:
`02dc768e868ee4a40ddc9170cb1cdef4a0e6e501c1f81eba5df88105b2b06aea`.

Independent arithmetic/lineage audit:
`candidate_image_mask_audits/numeric_12784259_001/`.
Without importing production mask functions, it verifies **82,560 original
CSV cells**, 960 image-slot/mask decisions, **3,840 candidate/mask relations**,
all fixed denominators, per-report-model counts and nested withholding counts.
It reuses the preceding independent image decision replay helper, not any
clinical reference labels. This verifies consistency of cached computations,
not clinical truth or correctness of the report assertions. Audit SHA256:
`6d2d48fb3aa246708968c86f50ddd3e6671d9fb5777ba9a1899d5403ee16942e`.

Existing CPU Slurm allocation **12784259** only: construction/replay 0.151
seconds, peak RSS 0.059 GiB; independent audit 0.087 seconds. No new job,
GPU, model call, download, API, training or selection. Thirty-one new
invented-fixture tests passed. Seven artifact hashes, 32 source pins,
Git exclusion, the unchanged original bank and project-group modes
2770/0660 were verified. Consumed code and historical output remain immutable.
The full V1.2 regression suite passed **3,052 tests in 16.582 seconds**.

This makes the fixed masks available as **candidate-specific diagnostic
evidence**, but does not clear the shared-head/image-grounded clinical gate.
The four-head human-literal report benchmark still has no opacity head;
cached report opacity proposals do not borrow its pooled reliability.
The RICORD reference does not supply synthetic EHR truth. No best mask,
new winner, clinical fault assignment or executable repair action is produced.
下一步需要验证相同 finding 的报告断言与图像证据，而不是把两份不同任务
的 benchmark 分数相乘，或通过减少可比较候选来宣称修复有效。

## Frozen baseline choices under all four image masks (2026-10-07)

Apply the already audited candidate-mask overlay to **existing choices**, not
to a newly optimized selector. Preserve all 80 synthetic EHRs, 13,200 historical
trials, five budgets (4/8/12/20/30) and five methods. The old random method means
random acquisition followed by scored final selection; the separate uniform
final-choice control shares its acquisition traces and expenditure. Average
all 1/5/25 seed replicates within each EHR before averaging all 80 EHRs.
The four masks yield 52,800 correlated diagnostic readouts, not new patients
or new calls. Missing choices, unknown/uncertain report proposals and abstained
image proposals remain non-comparable in the full denominator.

The table below uses budget 30 and the fixed **XRV + BioViL mean agreement**
mask to make one comparison readable. It is not selection of the best mask
or budget: all 100 method/budget/mask combinations are saved in CSV.

| Existing method | Mean historical simulated calls | Proxy support / all EHRs | Proxy opposition / all EHRs | Comparable coverage | Agreement among comparable |
|---|---:|---:|---:|---:|---:|
| Fixed Sana + MAIRA path | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Random acquisition + scored final selection | 30.000 | 11.25% | 5.00% | 16.25% | 69.23% |
| Static reranking | 30.000 | 11.25% | 5.00% | 16.25% | 69.23% |
| Targeted heuristic | 28.825 | 10.00% | 3.75% | 13.75% | 72.73% |
| Random acquisition + uniform final selection | 30.000 | 16.05% | 3.80% | 19.85% | 80.86% |

This is **not evidence that targeted repair outperforms ordinary choice**.
Fixed-path 100% agreement has only nine comparable EHRs out of 80, not 80
clinically correct triples. The targeted point results do not beat uniform
final choice on support or coverage; these proxies do not establish either
method's clinical quality. Random/scored and static coincide at cap 30; they
are not two independent validations. Historical calls are simulated costs,
not measured GPU time or prospective savings. Only scored-versus-uniform
final choice has identical acquisition/expenditure; equal budget alone does
not imply equal cost.

For the **same static choices** at cap 30, changing the evaluator gives:
XRV 52.38% conditional agreement / 26.25% coverage; BioViL mean 71.43% /
26.25%; pair agreement 69.23% / 16.25%; all-template agreement 66.67% /
15.00%. No selected artifact changed. Relative to XRV, pair masking withholds
2.5 percentage points of support and 7.5 points of opposition, with 10 points
of comparison coverage lost. Adding template stability then withholds another
1.25 points of support and **no opposition**. These are withheld comparisons,
not fixed clinical errors or successful regenerations.

The existing opacity report checks remain a qualification gap:
`docs/opacity_report_check_result_12675347.md` reports 3/6 authored controls
correct; `docs/opacity_assertion_stages_v3_result_12677513.md` reports 38/48
authored development states correct, including remaining uncertainty/scope
errors, and **failure of the frozen gate**. The V3 interface completion must
not promote the cached report opacity proposals to clinical truth. The
four-head manual report benchmark has no opacity head. All 80 EHRs lack an
explicit opacity reference. Therefore these tables remain a CXR/report
proposal sensitivity diagnostic, not a validated three-modal score, fault
localizer or repair policy.

Output below `artifacts/protected/tricompose_v1_2/`:
`frozen_selection_mask_comparisons/all_caps_12784259_001/`.
`method_comparison.csv` contains 100 aggregate rows; `case_means.csv` contains
8,000 case/method/budget/mask means; `paired_method_comparison.csv` contains
100 paired-method descriptive contrasts; `mask_withholding.csv` contains
75 nested-mask withholding comparisons. `trial_mask_readouts.jsonl` retains
all original trial hashes, chosen IDs, artifact hashes, seeds and costs.
Plan, summary and manifest complete the protected receipt. Manifest SHA256:
`ef24a684fd5e6903dd894f9d60caa0afc9f6ca0fd7056d168b43f21e899b3aae`.

New code: `TriCompose-v1.2/src/tricompose_v12/frozen_selection_image_masks.py`,
`TriCompose-v1.2/tools/compare_frozen_selection_masks.py` and
`TriCompose-v1.2/audits/audit_frozen_selection_masks.py`.
The independent audit imports neither production mask nor comparison functions;
it verifies 13,200 unchanged trial bindings, **369,600 mask-counter cells**,
all 8,000 case means, 100 method rows, 100 paired rows and 75 withholding rows.
Audit output: `frozen_selection_mask_audits/numeric_12784259_001/`.
Audit manifest SHA256:
`fb290da1c50d20e86db0e8eca3a2436f23d41c141840d931cf30b99c7de077eb`.

Existing CPU allocation 12784259 only: comparison 0.927 seconds / 0.197 GiB
peak RSS; independent audit 0.606 seconds. No GPU, model, API, download,
training, new Slurm submission, altered choice or actual regeneration. No raw
report/EHR body, patient key, image pixel or current weight file is reopened.
Source provenance comes from sealed receipts, not current pixel revalidation.
No confidence intervals or clinical superiority test are added: reuse of image
and report artifacts across synthetic EHRs has not been established as an
independent patient-cluster sample. Nineteen new invented-fixture tests pass;
the full V1.2 suite passes **3,071 tests in 16.951 seconds**. Consumed code and
outputs remain immutable; later changes require a new version and run ID.

Practical next step: independently qualify report assertions for the **same
finding and scope** before treating agreement as a repair trigger. Freeze its
reference projection, independent held-out inventory and uncertainty/negation
protocol before inference; any new Slurm request still needs complete script,
resource request and explicit approval. Do not tune or lower gates on the
already inspected 48 authored examples, select a favorable evaluator from
this table, or claim that abstention improved generated artifacts.

## Prepared opacity manual-reference inventory on the released dev file (2026-10-07)

**Prepared and unit-tested; not submitted, not a completed evaluation.** A
bounded read-only source/receipt audit found `source/manual_data/dev.json` in
the existing user-uploaded CXRGraph intake. Its sealed file size is 212,072
bytes and source SHA256 is
`4c2d80d73ebaf01c9b7ba786168ae5bab33739f630306f2182b14a5e5bd050e5`.
The checked local source/docs did not reference this dev input before this
addition. That does not prove it was unused by other people or excluded from
a model's training/development. No raw dev document or original source key has
been opened by this preparation. The already inspected 100-report test file
is not relabeled as an untouched benchmark.

This first task inventories **manual annotation support**, not scorer accuracy.
Use every dev document in release order, with a predeclared upper bound of
1,024 documents. Do not assume its case count or class balance before the
protected job reads it. Validate the unchanged native manual schema and exact
source hash. Select only literal `opacity` / `opacities` heads; human observation
spans provide positive/negative/uncertain states. No disease synonyms, generic
normality expansion, model-derived reference or answer-assisted case selection.
No annotated literal observation means unknown; failed validation is a separate
unavailable outcome, never an unknown or negative reference. Retain duplicates,
failures, all source positions and fixed denominators.

The output must not confuse **language annotation** with a global current lung
finding. A negative observation inside a qualified phrase is not proof of an
unqualified absence in the current image. Anatomy and temporal/person scope
are not established by a head-span polarity label. Therefore every projection
keeps `current_lung_opacity_reference: null` and disables clinical-primary use,
selection and regeneration. This reference inventory cannot by itself close
the shared-head clinical gate or qualify the existing Qwen/CheXbert proposal.

Save only opaque report indices, canonical source hashes, exact coordinates,
human native states, unavailable reasons and aggregate support counts below
`artifacts/protected/tricompose_v1_2/manual_opacity_reference_inventories/`.
Do not export original keys, report text, real images or native token strings.
Compare canonical report hashes against all 100 prior test hashes to expose
exact artifact overlap; zero hash overlap is **not patient disjointness**.
Preserve unresolved checkpoint overlap and dev/reference role explicitly.
No predictions exist for this newly scoped task yet; do not describe the
reference preparation as successful report-scoring validation.

Implementation:
`TriCompose-v1.2/src/tricompose_v12/manual_opacity_inventory.py`,
`TriCompose-v1.2/tools/inventory_manual_opacity_dev.py`, and fixed request
`TriCompose-v1.2/configs/manual_opacity_inventory_v1.json`.
Request SHA256:
`e1b32da80df69ef4e96dccb587f5df049f417649eba7b39c0cc10c69ce8c64f5`.
The wrapper explicitly refuses the current cache-only CPU allocation
12784259, a login node, GPUs, changed scope arguments and a missing batch
scope marker. A newly approved protected CPU task is required for the raw
annotation/report consumption. Existing source files and historical results
remain unchanged; atomic output refuses any existing run ID.

Proposed complete batch script:
`TriCompose-v1.2/slurm/82_manual_opacity_dev_cpu.sbatch`.
SHA256:
`fbd5f1fbaedeb33ee21e12258a03bac4b3fc8d04352ad8f114afdab5cd52a623`.
Resource request: **main, one node/task, one CPU, 2 GiB RAM, five-minute
wall-time limit, zero GPUs**. No inference, weight access, training, download,
external API or new environment. Logs/runtime/output stay protected and
workspace-local; stdout contains sanitized status/runtime/memory/hashes only.
The time limit is not a queue or runtime prediction. Show the entire script
and obtain explicit subsequent approval before `sbatch`.

Seventeen new invented-fixture tests pass, covering native schema conversion,
source hashing/coordinates, unknowns, uncertainty, qualified/anatomical/time
scope limitations, duplicates, failures, overlap and pre-source-read execution
guards. The full V1.2 suite passes **3,088 tests in 16.786 seconds**; batch syntax
and Git whitespace checks pass. Only source code, cache manifests and filesystem
metadata were inspected during preparation. After approval and completion,
audit the actual reference support before choosing a prospective frozen-reader
evaluation; do not fit rules to the known 48 authored development answers.

### Completed dev annotation-support check (job 12787886)

After the complete script/resources were shown and the user approved, submit
the **unchanged** `82_manual_opacity_dev_cpu.sbatch`. Slurm job **12787886**
completed on a01-02, exit 0, three seconds scheduler elapsed; one CPU / 2 GiB
requested, no GPU. The worker consumed only the authorized dev reports and
manual annotations internally, with **zero model calls**. Worker runtime
0.109 seconds, peak RSS 0.0245 GiB. This is reference support inventory,
**not a completed report-extractor benchmark or new model score**.

All **75 released documents** remain in order and denominator. Seventy-four
passed the current adapter; one remains unavailable with `ValueError`. Do not
call that source document clinically invalid or assign it a finding state:
the exact validation cause has not been established from the sanitized
receipt. No source document, adapter rule or prior run was changed to rescue
the failure.

| Reference outcome | Reports |
|---|---:|
| Literal opacity observation positive | 22 |
| Literal opacity observation negative | 2 |
| Literal opacity observation uncertain | 0 |
| No annotated literal opacity observation / unknown | 50 |
| Adapter unavailable | 1 |
| All attempted reports | 75 |

The 24 reports with literal heads contain **32 mentions and 32 human-span
bindings**; no unmatched literal mention was found among the successfully
projected reports. All 75 canonical source hashes are distinct. Exact text-hash
overlap with the earlier 100-report test inventory is zero. These are artifact
checks, not verified patient independence or absence from checkpoint training.
The projection still supplies no global current-patient lung-opacity truth.

Practical consequence: there is **no uncertain-reference support** and only
two negative-reference reports for this literal vocabulary. Do not claim
complete four-state qualification, estimate uncertainty recall from a zero
denominator, promote the 50 unknowns to negatives, or use a favorable positive
result to justify automatic repair. A later prediction pilot may assess narrow
text extraction with these per-state denominators shown; it cannot certify
the previously failed uncertainty/scope gate. Any further source or model
work needs its own predeclared scope, not post-hoc synonym expansion or
replacement of difficult cases.

Protected output:
`artifacts/protected/tricompose_v1_2/manual_opacity_reference_inventories/released_dev_12787886_001/`.
`summary.json` is the aggregate result; `reference_inventory.json` retains the
opaque-index/hash/coordinate reference records and failed slot. Receipt SHA256:
`c698dc6c047090142633c886219d98064ed049428fe8bd77d110b4eb191a7dd5`.

Independent cached count/binding audit:
`TriCompose-v1.2/audits/audit_manual_opacity_inventory.py` and protected output
`manual_opacity_reference_audits/numeric_12784259_001/` relative to the V1.2
protected root. Audit SHA256:
`047a095442423d66fe557c29122787cf5869c5418fd47816969c16d061040270`.
It imports neither production projection nor summary functions, replays all
75 outcomes and 32 head/span bindings, source-hash overlap/duplicates and full
denominators, and checks that scope/clinical/action eligibility remains false.
It runs inside the existing cache-only CPU allocation 12784259 in 0.025
seconds. It **does not reopen dev source bytes or annotations**, independently
reconstruct literal words, adjudicate the unavailable case or verify clinical
scope. Source provenance comes from the completed approved worker receipt.

Six new invented-metadata audit tests pass. The full V1.2 suite passes
**3,094 tests in 16.646 seconds**. Artifact hashes, project-group 2770/0660
permissions, Git exclusion and the unchanged original synthetic-bank manifest
are verified. No extra Slurm submission, training, model inference, download,
external API, new selected triple or actual regeneration was performed after
this approved inventory task. Consumed projection code, tests, request,
script and protected outputs now remain immutable.

### Prepared frozen CheXbert dev-opacity pilot — not yet submitted

The next step now performs actual frozen report-label inference rather than
another annotation inventory. Its scope is **all 75 previously inventoried
released dev reports**, in their existing order, including the single
reference-adapter failure. A reference failure must not suppress an otherwise
readable model input. No source cases, synonyms, annotation rules, thresholds,
existing winners or synthetic triples are changed. The existing cache-only
allocation 12784259 is explicitly forbidden to run this worker; no raw source
or checkpoint weights were read during preparation.

New versioned implementation:

- `TriCompose-v1.2/tools/benchmark_manual_opacity_dev_chexbert.py`
- `TriCompose-v1.2/src/tricompose_v12/manual_opacity_reader_diagnostic.py`
- `TriCompose-v1.2/configs/manual_opacity_chexbert_v1.json`
- `TriCompose-v1.2/tests/test_manual_opacity_reader_diagnostic.py`
- `TriCompose-v1.2/tests/test_manual_opacity_reader_worker.py`
- `TriCompose-v1.2/slurm/83_manual_opacity_dev_chexbert_cpu.sbatch`

Request SHA256:
`0ccf6d48968486613dcb9ecaf9cda3ae9d24a433a9a3c731e865c8b1727a443c`.
Complete script SHA256:
`b2025a682f75c3df26eb581522f287d3d0665177cc427c86046b828d4aa130e9`.
**Prepared resource request: main, one node/task, two CPUs, 8 GiB RAM,
15-minute wall-time limit, zero GPUs.** No new environment, download,
training or external API. Previous 100-report CPU CheXbert inference took
23.481 seconds; this is feasibility evidence, not a promise about the new
queue, runtime or outcome. The 15 minutes are a time limit, not an expected
inference duration. Show the entire exact script and obtain explicit subsequent
approval before submission. This section does not record an approved or
completed inference job.

The unchanged `cxrmate/tools/chexbert.py` loads the pinned existing checkpoint
and BERT tokenizer assets. Its checkpoint path is absolute; `ckpt_dir` points
to a new protected runtime cache so no model directory becomes a write target.
The model is in eval mode with gradients disabled, CPU seed 0, two threads,
batch size four, at most 75 encoder examples, zero retries. Preflight uses the
official wrapper's exact normalization with token truncation disabled.
Overlong reports are retained as unavailable and never sent for a truncated
forward pass. Full named 14-head output is retained privately, but only the
lung-opacity head is compared in this narrow diagnostic; the other 13 heads
have no references here.

Model inputs contain only the released words joined with single spaces, not
NER labels, reference states, source keys, EHR, images or previous predictions.
The release bundles text and annotations, so the JSON parser necessarily
decodes the annotation fields; the request builder ignores them. This is
**not an operator-blinded experiment**: source class counts are already known.
The completed reference-inventory states are separately decoded only after
prediction JSON is fsynced and hash-sealed. Source text hashes must bind the
same prediction/reference slots. Network socket connections are blocked during
model loading/inference, in addition to offline flags. Framework messages,
runtime files and results stay protected; stdout reports only sanitized
status/runtime/memory/hashes. Output directories are 2770, files 0660 under
the project-group boundary; fresh atomic run IDs refuse overwrite.

Proposed output after an approved successful run:
`artifacts/protected/tricompose_v1_2/manual_opacity_reader_runs/manual_dev_<job_id>_001/`.
It contains `frozen_plan.json`, `chexbert_predictions.json`,
`comparison_records.json`, `evaluation.json`, `summary.json`,
`manifest.json` and a protected framework log. **No model scores are available
from this prepared code yet.**

The evaluation preserves all attempted slots in a five-by-five state/status
table, exposes per-state supports and prediction failures, and separately
reports known-literal state matches and literal polarity flips. All-supported
and completed-only denominators are both shown; failures cannot improve the
former by disappearing. Zero-support uncertainty fractions remain null.
The reference counts remain 22 positive, 2 negative, 0 uncertain, 50 unknown,
1 unavailable. Small negative support and absent uncertain support do not
qualify those behaviors.

Most importantly, the human reference is **literal opacity/opacities span
polarity**, while CheXbert predicts the broader **lung-opacity finding**.
Vocabulary, current-patient, anatomical, qualifier and temporal equivalence
are unverified. Determinate predictions on literal-unknown reports are shown
as a vocabulary/scope diagnostic, not clinical hallucinations. No pooled
unknown-heavy accuracy/F1, image factuality score, probability calibration,
clinical qualification, fault localization, new selection or repair gate is
claimed. Patient disjointness and checkpoint training overlap remain unresolved.

Twenty-three new invented-fixture tests pass, including full-slot retention,
hash mismatch, unavailable references, missing predictions, polarity
denominators, official head decoding, token overflow, exact token limit,
failed/malformed batch accounting without retries, annotation-independent
requests and direct-call/login/current-allocation/GPU/scope guards. Full V1.2
suite: **3,117 tests in 16.937 seconds**, all passed. Batch syntax, Git whitespace,
protected-output Git exclusion, and unchanged reference/synthetic-bank
manifest hashes also pass. No inference or new Slurm submission occurred in
this preparation step.

### Completed frozen CheXbert dev-opacity pilot (job 12788366)

After the entire `83_manual_opacity_dev_chexbert_cpu.sbatch` and resource
request were shown, the user approved submission. The exact previously shown
script and request hashes remained unchanged. Slurm job **12788366** completed
with exit 0, **41 seconds scheduler elapsed**, two CPUs / 8 GiB requested,
zero GPUs. Frozen CPU model inference took **20.730 seconds** for all 75
encoder examples in 19 batches; worker runtime 37.535 seconds, initialization
2.290 seconds, peak RSS 2.044 GiB. No retries, token truncation, new training,
downloads, external API or new environment were used. These are this run's
measured timings, not a promise for future jobs.

**All 75 reports received a complete named 14-head prediction**, including
the previously reference-unavailable slot. There are 74 available literal
human references and one unavailable reference. The reference failure was
not repaired, dropped or converted to unknown; its model prediction remains
present but is not scored as a reference match. This pilot compares only
the lung-opacity head with the frozen literal opacity/opacities projection.

| Literal human reference | Support | Same-state CheXbert predictions | Fraction |
|---|---:|---:|---:|
| Positive | 22 | 19 | 86.36% |
| Negative | 2 | 2 | 100% (only two references) |
| Uncertain | 0 | 0 | N/A |
| Unknown / no annotated literal observation | 50 | 47 | 94% (vocabulary diagnostic only) |
| Reference unavailable | 1 | Not scored | N/A |
| Known literal positive/negative states combined | 24 | 21 | 87.5% |

Three positive literal references received unknown head predictions; none
received an opposite negative prediction. There were **zero known-literal
positive/negative polarity flips out of 24 explicit reference states**.
On the 50 literal-unknown references, the broader head predicted positive
twice and negative once, unknown 47 times. Those three determinate outputs
are **not adjudicated clinical hallucinations**: literal vocabulary absence
does not imply finding absence, and synonym/qualified/temporal/anatomic
equivalence has not been established. The unavailable reference slot
received an unknown prediction, without becoming an unknown gold label.

The 87.5% number is a **literal annotated state-match fraction**, not triple
clinical accuracy, image factuality, full-report lung-opacity accuracy or a
qualified automatic-repair score. Only two negative references and zero
uncertain references prevent broad negation/uncertainty claims. Patient
independence, checkpoint training overlap and reference/head scope equivalence
remain unresolved; this is dev data with known class support, not a blinded
untouched final test. No favorable unknown-heavy pooled metric is promoted
to the headline. All failure/unknown conventions from the prepared protocol
are unchanged.

Protected results:
`artifacts/protected/tricompose_v1_2/manual_opacity_reader_runs/manual_dev_12788366_001/`.
`evaluation.json` contains aggregate state/support statistics;
`chexbert_predictions.json` contains private opaque-index/hash predictions;
`comparison_records.json` retains all 75 outcome slots. Worker manifest SHA256:
`61622a2986dec980a22063a47da23c54eb327346c33f460c5a1875feadcfcf34`.
No raw report, original source key, image or native annotation was printed
or exported as a result data artifact. Framework diagnostics remain protected.

Independent cache arithmetic audit:
`TriCompose-v1.2/audits/audit_manual_opacity_reader.py`, with six invented
metadata tests. It imports no production evaluator, reproduces the entire
five-by-five confusion matrix and all per-state/known-state denominator
statistics, checks prediction/reference source-hash binding, all 75 comparison
slots, actual recorded model-call counts and false clinical/action eligibility.
It verifies consumed code/cached-artifact hashes and project permissions.
It **does not reopen dev reports, native annotations or heavy checkpoint
weights**; their provenance remains the approved worker receipt. It also
does not independently adjudicate clinical meaning, reproduce tokenizer
normalization from source text or establish vocabulary equivalence.

Audit completed in the existing cache-only CPU allocation 12784259 in
0.029 seconds, zero model calls. Protected audit:
`artifacts/protected/tricompose_v1_2/manual_opacity_reader_audits/numeric_12784259_001/`.
Audit manifest SHA256:
`9e2db467458454a9db559edae9fa9e8e23a6b5af1452ffca245d8864537d11bc`.
Full V1.2 suite passes **3,123 tests in 16.579 seconds**. Protected artifacts
remain project-group 2770/0660 and Git-excluded. The original synthetic-bank
manifest remains unchanged. Consumed inference/audit code, tests, configs,
batch script and results now remain immutable.

Next scientific requirement is uncertainty/qualified-scope evidence that
actually supports the intended operating behavior, not another pass over
these same known positive cases or relabeling the unknowns. Until that gap
is closed, CheXbert may provide a clearly marked secondary text-label
diagnostic, but this pilot does **not** unlock error localization, a new
selected triple or automatic regeneration.

### Next benchmark: prepared directional RadNLI interface, awaiting source access

The user requested continued work after the 75-report opacity pilot. The
read-only review found that the broader exact-observation uncertainty diagnostic
has **already completed**, so it was not recomputed or presented as new work:
the cached 100-report RadGraph-XL extraction includes 112 uncertain human
observation spans, with 86 exact-span/state matches, 19 determinate predictions
on exact uncertain spans and seven unmatched uncertain spans. Those existing
counts do not establish current-patient/history/qualified-absence scope.

The official CXRGraph description explicitly limits capture of clinical context,
including Comparison/History information. Consequently, another count over
the same native labels cannot fill that semantic-reference gap. Consult
[CXRGraph official description](https://physionet.org/content/cxrgraph/1.0.0/).
Field definitions are the useful source metadata; do not print identifiers,
source-path examples, report fragments or dataset examples that may occur in
data dictionaries or publication pages.

A complementary candidate benchmark is
[RadNLI official dataset](https://physionet.org/content/radnli-report-inference/1.0.0/),
version 1.0.0, associated with *Improving Factual Completeness and Consistency
of Image-to-text Radiology Report Generation*. It supplies expert-reviewed
directional radiology sentence-pair labels: entailment, neutral and
contradiction, with a published 480-pair dev split and 480-pair test split.
The construction annotates both premise/hypothesis directions. These are
published inventory claims to verify at intake, not locally executed results;
the actual expert class frequencies must not be assumed balanced. Paired
directions must not be counted as independent patients.

RadNLI is a text-inference reference, not an image/EHR factuality reference.
Its three classes do not replace the four finding states, provide a complete
temporal/anatomic/qualified-negation annotation taxonomy, or automatically
identify which generated modality is wrong. It can test whether a proposed
text-consistency reader actually distinguishes contradiction from absence of
support, and whether it respects inference direction. Current-patient,
anatomy, severity and temporal subgroup gold must not be claimed from the
four published source fields alone.

Prepared **pure in-memory** implementation:

- `TriCompose-v1.2/src/tricompose_v12/radnli_benchmark.py`
- `TriCompose-v1.2/tests/test_radnli_benchmark.py`
- `TriCompose-v1.2/configs/radnli_benchmark_v1.json`

Protocol SHA256:
`61fcff721c4ab216516fd613c320cbd6d5f4496aac01032591c1e9d48a276ab9`.
This is not a data-intake worker, model adapter, executed benchmark or new
score-table column. No source download, raw RadNLI read, model inference,
training, API call or Slurm submission was performed in this preparation.
The current user access confirmation, source intake hash and model checkpoint
hash remain null; download/submission approval flags remain false.

Implemented contract:

1. Keep exact sentence characters/order and preserve every attempted row.
   Input preparation ignores gold labels and original IDs, returns sentence
   pairs separately in memory, and exports only opaque indices and hashes.
   Invalid text/schema/gold labels remain unavailable, never a dummy pair or
   neutral reference. A failed gold row does not prevent a readable model input.
2. Bind predictions to the ordered premise/hypothesis pair hash. Maintain an
   additional unordered two-sentence hash to expose duplicate/reversed groups.
   Entailment is not forced to be symmetric; duplicate exact input with
   conflicting gold labels is flagged and retained, not silently filtered.
3. Provide a predeclared constant-neutral **no-model** baseline receiving only
   request hashes, not expert labels or fitted class prevalence. It is a
   benchmark sanity control, not a clinical decision policy.
4. Evaluate a three-class-plus-unavailable confusion matrix, per-class
   precision/recall/F1 and support, failure-aware accuracy, completed-only
   accuracy, and fully annotated bidirectional group correctness. Missing
   predictions remain false negatives and in the reference-available accuracy
   denominator. Zero-support recall/F1 stays null; macro F1 names its included
   reference-supported classes. Unknown gold labels are not recoded as neutral.
5. Provide exact ordered/unordered pair and sentence hash overlap checks across
   dev/test. These are artifact overlap diagnostics, not verified patient
   independence or checkpoint training exclusion. Do not drop overlapping or
   difficult source rows post-hoc to create a favorable test subset.
6. A future protected worker must seal predictions before decoding the
   reference projection, freeze reader/prompt/rules before test evaluation,
   and report any already-seen development data and unresolved training overlap.
   The pure helpers do not themselves enforce job approval, model freezing,
   licensing or that IO ordering; no real worker is claimed present.

Twenty-four tests pass using only invented, nonclinical strings. They cover
privacy-safe metadata, exact character preservation, label-independent inputs,
direction changes, missing references/predictions, invalid schemas, the fixed
baseline, three-class metrics, zero support, failed-prediction denominators,
duplicate/source-hash/split mismatch, conflicting gold, reversed dev/test
overlap and refusal to promote clinical/action eligibility. Full V1.2 suite:
**3,147 tests in 16.244 seconds**, all passed in the existing cache-only CPU
allocation. The latest actual opacity-run and original synthetic-bank hashes
remain unchanged. No new clinical scoring result is asserted from fixture tests.

Access prerequisite: the official resource requires credentialed access,
the resource DUA and CITI training under PhysioNet Credentialed Health Data
License/DUA 1.5.0. Existing MIMIC/CXRGraph access is not treated as an implicit
confirmation of this new resource's terms. No credential files were read,
and no authentication/download was attempted. RadNLI filenames are absent
from the checked V1.2 reference intake and configured local model paths;
this is not a claim about every possible location on the server.

A fresh, empty, project-group 2770 upload inbox is prepared:
`artifacts/protected/tricompose_v1_2/reference_datasets/radnli_upload_20261007_001/`.
Upload the authorized archive or the two official JSONL files there without
copying sentence contents into chat. The inbox is Git-excluded. After explicit
resource-access confirmation and file upload, first inventory filenames,
sizes and hashes without decoding reports; then present the complete protected
Slurm intake/benchmark script and resource request for separate approval.
No original generation, selector or repair trigger has changed.

### Route change: no new dataset authorization required (2026-10-07)

The user requested a simpler alternative to obtaining RadNLI access. RadNLI
is now **optional and deferred**, not a prerequisite for current development.
The prepared interface and empty upload inbox are retained, but no upload,
credential, new resource agreement or download is requested. The preceding
RadNLI protocol hash describes the original prepared configuration; the
configuration now explicitly records this deferral and remains unexecuted.
This does not relax access restrictions or authorize obtaining the resource
through a mirror.

The active development route uses only existing authorized local references
and already completed diagnostics:

1. **Real text-extraction reference:** reuse the sealed CXRGraph manual
   annotation benchmarks and their cached numeric comparisons. The 100-report
   RadGraph-XL benchmark already includes positive, negative and uncertain
   observation spans; the 75-report dev-opacity CheXbert pilot is a separate,
   narrower literal-state diagnostic. Do not rerun them or pool their different
   vocabularies, denominators and reference scopes into a new clinical score.
2. **Language stress checks:** reuse the completed wholly authored assertion
   and uncertainty controls. Their answers are author-specified behavioral
   expectations, not independent clinical gold. Existing failures remain
   visible; deferring RadNLI neither clears those failures nor permits tuning
   on their answers and reporting the same examples as an unseen benchmark.
3. **Synthetic-bank comparison:** continue the fixed 80-EHR candidate-bank
   development experiment using existing frozen predictions and explicit
   comparable-fact masks. Compare fixed-path and static-reranking outputs
   under the same evidence contract before adding any newly executed targeted
   regeneration experiment. Keep extraction reliability, proxy cross-modal
   agreement, coverage/abstention and model-call cost separate. Cached action
   replay is not an executed repair, and shared-reader agreement is not
   evidence of independent clinical correctness.

This route removes the new-access blocker, **not the scientific validation
gap**. Existing CXRGraph annotations do not establish full temporal/history
scope, image-grounded truth or which generated modality is wrong. Any novel
controlled corruption set must be predeclared, wholly authored or processed
from authorized inputs inside an approved protected job, and labeled a
development sensitivity diagnostic rather than an expert-labeled clinical
test. A future independent semantic reference may strengthen the paper, but
acquiring RadNLI is no longer the next required action.

Only the unexecuted optional-protocol status and this development plan changed.
No original source data, benchmark output, generated candidate, scoring rule,
selected triple or repair policy was modified. No model inference, download,
external API call or new Slurm submission was performed for this route change.

### First report delivery using existing results (2026-10-07)

The professor/teammate engineering report is now consolidated at
`artifacts/protected/tricompose_v1_2/deliverables/progress_report_12784259_001/`.
`REPORT_CN_EN.md` explains the frozen generation graph, actual versus unique
artifact counts, all three historical edge scopes, static/uniform/fixed
comparisons, the separate opacity-mask sensitivity readout, real-annotation
extraction results and the unsuccessful two-case prospective repair control.
This is a new report over existing evidence, not a new generation benchmark.

Its complete copied numeric tables preserve 75 historical comparison rows,
225 edge rows and 100 fixed-mask sensitivity rows. A nine-row extraction
summary retains the exact supports and marks zero-support uncertainty NA.
Metadata source hashes bind these results without reopening raw inputs,
targets, synthetic payloads, image pixels or model weights. The report stays
within protected project-group access; no source/body content is reproduced.

The first engineering report can be delivered today. Independent clinical
localization and a demonstrated equal-cost repair advantage remain research
milestones, not prerequisites for showing the existing engineering result.
No unsupported completion date is assigned to a paper-grade method. Keep
the optional RadNLI access path out of the immediate critical path.

### Four-day research initial-version sprint: Oct 7–11 (2026-10-07)

The user sets **October 11** as the initial-version delivery target. Reuse
the existing generation graph, real-reference extraction diagnostics, frozen
scorers and protected banks. Do not rebuild model deployments, train a router,
request new dataset access, or make an LLM wrapper a prerequisite. This is a
target for reproducible code, experiments and a research report, **not a
guarantee of a positive method result or CVPR readiness**.

| Date | Deliverable | Acceptance / stop rule |
| --- | --- | --- |
| Oct 7 | Observed-action controller, frozen-cache comparison, prepared paired-probe job | Preserve EHR and exact supported/comparable fact identities; report failures and ablation results honestly. |
| Oct 8 | Execute separately approved paired probes through unchanged official adapters; wire the validated receipts to actual decisions | A cache replay or a freshly collected bank is not online adaptive repair. Authenticate the exact model inputs and charge failures. |
| Oct 9 | Equal-budget fixed/retry/static-rerank/targeted comparisons plus action-feedback ablation | Compare cost curves and case-paired outcomes. Include all predeclared cases, low-evidence cases and duplicate-artifact counts; no cherry-picking. |
| Oct 10 | Analyze preserved facts, proxy conflicts, missing comparisons, quality risks, diversity and actual cost; consolidate report | Separate frozen-reader agreement from clinical truth. Report unchanged/worse endpoints, not just favorable ones. |
| Oct 11 | Deliver code, protected results, baseline tables, limitations and reproduction commands | Reserve as delivery/debug buffer. If targeted repair does not beat static reranking, state that result and narrow the claim. |

Day-1 implementation: `TriCompose-v1.2/src/tricompose_v12/probe_repair_v1.py`
receives one completed observation at a time. It has no full-bank ranking or
BioViL input. It starts with the cheapest eligible report probe; observed
failure can change the next action to an image probe **only if a direct
EHR–image proxy conflict or basic artifact failure already authorizes that
branch**. Same-image report agreement is not independent image truth. Failed
probes remain charged; exact comparable facts and supported positive findings
cannot be silently lost to obtain a better score. This is a deterministic
prototype, not a trained classifier, validated fault localizer or novelty claim.

The first new developer replay is sealed at
`artifacts/protected/tricompose_v1_2/probe_repair_runs/pool80_12784259_001/`.
It contains 800 new policy trials, the unchanged 13,200 historical baseline
replicates, 2,800 case/method/budget means and 35 aggregate rows. Random seeds
are averaged within EHR before averaging over cases. All 80 EHRs remain in
the tables; direct EHR-edge rates have only **8** available EHR denominators.
Unknown/uncertain remain unknown/uncertain (never automatically negative),
and the old global No-Finding expansion is not used for these raw-state
readouts. Choices are sealed before the secondary endpoint is read. No source
text, pixels, weights or API calls are used by this replay.

At budget cap 30, fixed-path BioViL raw cosine is 0.6186388 and the new
fact-preserving controller is 0.6371743, using 10.1 simulated diagnostic calls
on average. Raw CXR–report support/known rises from 0.0864583 to 0.1135417;
opposition/known stays 0.0041667 and comparison coverage rises from 0.090625
to 0.1177083. Seven of 80 cases have an accepted proxy replacement at this
cap; this is **not seven clinical repairs or new generations**. Static
reranking uses 30 simulated calls, reaches support/known 0.2364583 and zero
raw opposition, with BioViL 0.6122109. Different endpoints favor different
methods: there is no demonstrated equal-cost superiority of the new method.
The feedback/no-feedback variants have identical final choices at every cap;
only one case has a different action order at caps 12/20/30. Action-feedback
benefit is therefore **not established**. Do not retune on this table and
report the same bank as unseen confirmation.

The prepared prospective protocol is
`artifacts/protected/tricompose_v1_2/paired_probe_plans/paired2_12784259_001/`.
Worker: `TriCompose-v1.2/tools/collect_paired_probes_v1.py`;
batch: `TriCompose-v1.2/slurm/84_paired_probes2_v100.sbatch`.
Keep the same two previously EHR-only-stratified development anchors, final
prompts and frozen checkpoint pins. RoentGen-v2 seeds **3 and 4**, each followed
by **CXRMate-single and CheXagent-2**, yield at most four new images/eight reports
and 24 generation/verification attempts. Each image branch has its own stable
directory and six-attempt ledger, avoiding any overwrite of the old worker's
single-model prompt-copy filenames. This is a **paired-action collection**;
subsequent choices on that bank are replay, not executed online routing. Its
fresh eight-enabled-head classifier profile and binary official-section
quality flag remain separate from the historical uncalibrated reader profile.
Basic PNG validity is not anatomy or clinical realism. The script prepares a
sealed BioViL request but does not execute an unbudgeted endpoint worker.

Requested resources: one V100, four CPUs, 48 GiB RAM, 45-minute allocation
cap. Previous job 12657042 completed the same frozen generator/report-expert
stack on one V100 in 9m16s; that is feasibility evidence, not a runtime or
queue-wait guarantee for the larger collection. Runtime GPU memory must meet
the inherited 24-GiB planning requirement. This new script is **prepared,
not submitted**: display it in full and obtain explicit submission approval.

Verification for this sprint start: **3,202 tests pass** (55 new invented/mock
fixtures; no inference) in the existing cache-only CPU allocation. The batch
script passes `bash -n`. Replay outputs and source/choice hashes, prepared-plan
hash, all 246 inherited/new workspace code pins, protected 2770/0660 project
modes and Git exclusion pass. No source inputs, report bodies, image pixels,
checkpoint bytes, external API or new Slurm submission are used for these
checks. The new replay/plan and their consumed versions are now sealed.

Submission update (2026-10-07): after the complete script/resource request was
displayed and the user approved it, the unchanged `84_paired_probes2_v100.sbatch`
was submitted as **Slurm 12791443** (gpu partition, one V100, four CPUs,
48 GiB RAM, 45-minute runtime cap). Exact script SHA256:
`29caf843d9db6a9af8c00f5a429ca2b5c7c84c9e34db349676a41626f90d84a3`.
Plan manifest SHA256:
`ea4a1faf17b9d22a063bccd6d9482c4f598d017acb63920a59c2a0934ab0d37a`.
Initial scheduler state: `PENDING (Priority)`; submission does not imply a
completed generation. Planned output is
`artifacts/protected/tricompose_v1_2/paired_probe_runs/paired2_12791443/`.
No duplicate/replacement job or secondary endpoint job has been submitted.

The same job subsequently started on **d14-08** with the requested V100.
The protected start manifest and execution journal were created after the
frozen source/asset validation. This is a running collection, not a completed
result; do not infer clinical success from scheduler `RUNNING` status.

### Paired action collection completed; no observed repair advantage (2026-10-07)

Slurm **12791443 completed successfully in 14m53s** with exit code 0.
The frozen two-EHR/two-seed/two-expert grid produced **four images and eight
reports**, paying exactly **24** generation/verification attempts with **zero
failures**. Image hashes are four distinct values; report hashes are seven
distinct values, so eight report slots are not eight independent text outputs.
No original EHR, final prompt, checkpoint, old bank or old selected triple changed.

Generation output:
`artifacts/protected/tricompose_v1_2/paired_probe_runs/paired2_12791443/`.
Manifest SHA256:
`e5979cddf6b2075ea5a3e5ddef2c282a6278e89493144714b3afd1fd4dac0713`.
New metadata-only auditor `TriCompose-v1.2/tools/audit_paired_probes_v1.py`
recomputes the exact sealed choices/action credits, checks four six-attempt
branch ledgers against their original journals and immutable EHR anchors,
binds each score row to its completed model/scorer dependency chain, and
retains both declared EHRs even if a future branch is incomplete. It reads
neither report bodies, pixels nor checkpoint bytes. Eight invented-fixture
tests cover matrix completeness/duplicates, same-image scorer invariance,
parent binding, explicit unavailable endpoints and paid versus simulated costs.

Audited output:
`artifacts/protected/tricompose_v1_2/paired_probe_audits/paired2_12791443_001/`.
`candidate_scores.csv` has all eight candidates; `method_comparison.csv`
has both cases for fixed, observed-action replay and no-feedback replay.
`case_readouts.json` retains every directed paired-action credit/rejection.
Audit manifest SHA256:
`548ee53017c7a46fe836098e2be5bc00dc2cc8a1b2d7c8a70aa171ffb35fd195`.
Source/output/code hashes, protected project-group modes and Git exclusion pass.

**Negative result:** none of the eight directed report probes or eight
directed image probes satisfies the strict target-gain acceptance condition.
Every directed probe lacks strict target action gain; some additionally lose
comparable facts, introduce new proxy opposition or silence conflicts. Do not
claim that loosening only the preservation veto would demonstrate successful
repair. Both replay variants retain the fixed initial candidates; their
10-call replay costs are simulated policy costs, while the actual collection
pays 12 calls per EHR including unrequested candidates. No online adaptive
execution or computation saving is established by this experiment.

| Edge, summed over the two fixed selections | Known | Comparable | Supported | Proxy opposition |
| --- | --- | --- | --- | --- |
| EHR–CXR | 2 | 2 | 0 | 2 |
| EHR–Report | 2 | 1 | 0 | 1 |
| CXR–Report | 16 | 5 | 5 | 0 |

The selected CXR–report conditional agreement is 5/5, but coverage is only
5/16: it must not be called complete consistency or clinical correctness.
The two cached direct EHR constraints concern pneumonia; the image reader
disagrees across all four generated images. These are frozen-reader proxy
observations, not evidence proving pneumonia is absent or localizing a
clinically wrong modality. Next distinguish conditioning transfer and reader
reliability using independent evidence before spending on repeated seeds or
changing acceptance criteria. BioViL is **not executed** for these new outputs;
its CSV values stay explicitly unavailable, never borrowed from the old bank.
Secondary inference requires its own complete script/resource review and approval.

### Oct 7: paired-bank conditioning transfer and independent endpoint preparation

Read only the authenticated plan and generated-candidate metadata in existing
cache-only CPU allocation 12784259. Both fixed synthetic cases retain their
direct-conditioning tier; each request includes two direct fact IDs. RoentGen
records prompt lengths of **66 and 60 tokens** (its adapter limit is 77), and
all four generated candidates have runtime tokenizer observations with
`pipeline_changed_text=false`. Thus no evidence of extra-prefix insertion or
length-check truncation was found. This metadata does not establish correct
EHR-to-radiographic observability or prove an encoder-hook capture. The exact
recorded trace and finding-phrase correspondence will be checked privately in
the approved GPU endpoint job; no source EHR body was opened in this CPU check.

Prepared `TriCompose-v1.2/tools/score_paired_probes_v1.py` reuses the unchanged
frozen BioViL-T worker for the eight already generated, hash-bound pairs.
Its new 13 invented-fixture tests cover untouched selections, trace/length
checks, explicit missing phrases, NA retention, complete lineage, encoder
limits, checkpoint pins and GPU-before-read guards. The full 3,223-test suite
passes. It writes separate candidate, method, paired-action endpoint and
conditioning-transfer tables; it never changes policy acceptance based on
the secondary cosine and never reports a clinical accuracy label. Text slots
remain eight, with seven distinct source report hashes, not eight independent
patients or independent clinical annotations. The existing worker caches by
candidate ID, so maximum billed endpoint operations are four image encodings
and eight text encodings, not seven text encodings.

Prepared complete batch script:
`TriCompose-v1.2/slurm/85_paired_probes_biovil_v100.sbatch`.
Resource request: **gpu partition, one V100, four CPUs, 24 GiB host RAM,
15-minute wall-time cap**, one 300-second private worker timeout, zero retries,
zero new generators/training/downloads/external API calls. V100 availability
was checked with `noderes -f -g`; free GPUs are not a scheduling guarantee.
Source collection 12791443 already completed, so no pending dependency is
required. The new score run will use the exclusive protected directory
`artifacts/protected/tricompose_v1_2/paired_probe_endpoints/paired2_biovil_<jobid>/`.
**Prepared, not submitted**: complete-script review and subsequent explicit
user approval are still required. Existing runs, winners, adapters, checkpoint
directories and teammate files remain unchanged.

#### Approved independent endpoint submission: 12792810

After the complete script and resources were displayed, the user explicitly
approved with `go`. Submitted the unchanged script as Slurm **12792810**;
script SHA256 `cf0dbd0c8b348746620fa0e857d9107314f6f69c270efedb13b7fbad56da389e`.
The worker and test pins matched the reviewed script. The first scheduler
check showed **RUNNING on d14-08**, with a 15-minute wall-time cap. Protected
Slurm logs have project-compatible group ownership and mode 0660.

Destination:
`artifacts/protected/tricompose_v1_2/paired_probe_endpoints/paired2_biovil_12792810/`.
This submission measures existing synthetic artifacts only: no new generation,
training, online decisions or unreviewed resource substitution. Completion and
scientific readout remain to be verified; submission alone is not a metric result.

#### 12792810 partial outcome: valid independent scores, unresolved trace gate

Scheduler outcome is **FAILED, exit 1:0, elapsed 33 seconds**, not a successful
complete endpoint run. However, the frozen BioViL worker completed all **four
image encodings and eight text encodings**, with zero unavailable reports.
Its eight-pair lineage, request, model pins, numeric scores and exact encoder
accounting pass the existing independent endpoint validator. The subsequent
wrapper failed before writing final tables during its auxiliary source-image/
tokenizer checks. The precise failing trace predicate has **not yet** been
observed; do not invent a truncation, checkpoint or attention-mask diagnosis.

Native endpoint SHA256:
`7762615ed873d0d99ccde04c39371b66d975439fe6cea7ae37a0f02f2e03f648`.
Failure receipt SHA256:
`24514b0b88b912bf872db9edb8598237ad51ee8d5e0e5403c20a625d164b54f2`.
The consumed worker and failed run remain immutable. A new metadata-only
readout authenticates these receipts and the previously sealed choices,
retaining the failed source status rather than promoting it to successful.
It consumes no report/prompt/EHR bodies, pixels or weights and makes zero
model calls in the existing cache-only CPU allocation.

Validated numeric tables:
`artifacts/protected/tricompose_v1_2/paired_probe_endpoint_readouts/paired2_biovil_12792810_001/`.
Manifest SHA256:
`48b472d148ccffc94ce6e48fbb0d3ed34bac4430fed07a1a824218cba3753c52`.

| Readout | Slots | Distinct fixed EHRs | Mean raw BioViL cosine |
| --- | --- | --- | --- |
| CXRMate-single reports, both seeds | 4 | 2 | 0.2683514301 |
| CheXagent-2 reports, both seeds | 4 | 2 | 0.3652492994 |
| Fixed selections | 2 | 2 | 0.4515310228 |
| Observed-probe replay selections | 2 | 2 | 0.4515310228 |
| Without-feedback replay selections | 2 | 2 | 0.4515310228 |

These are uncalibrated secondary image-text similarities, **not clinical
accuracy percentages**. Expert averages include both image seeds; method
averages use only each fixed EHR's selected pair. Both replay methods retain
the fixed choices, so this new independent readout shows **no improvement**
over the fixed selections. Eight paired slots/four images/seven distinct
report hashes remain only two EHR development cases, not independent trials
of clinical efficacy. Do not use these cosines retrospectively to alter the
sealed policy or claim an oracle clinical choice.

Prepared a separate text-free trace diagnostic with explicit unavailable-mask
handling and old-gate predicate receipts, not a weakened clinical acceptance
rule: `TriCompose-v1.2/tools/check_paired_conditioning_v2.py`.
Its eight invented-fixture tests plus five cache-readout tests pass. Neither
the protected trace bodies nor image pixels were opened in the cache-only
allocation. To finish the input-check diagnosis, prepared complete script
`TriCompose-v1.2/slurm/86_paired_conditioning_cpu.sbatch`: **main partition,
one CPU, 2 GiB RAM, five-minute cap, no GPU/model/tokenizer execution**, reading
only the four already recorded synthetic traces privately. This is
**prepared, not submitted** and requires its own complete-script review and
subsequent explicit approval. The old CPU cache allocations are intentionally
blocked from running its trace-body reader.

#### Approved trace-only CPU submission: 12793549

The user approved the displayed complete CPU script with `go`. Submitted it
unchanged as Slurm **12793549**; script SHA256
`f4485c923a9cd35b061e81925083b210b6de08a80b7906c95b933005501396d2`.
Worker, test, cache-reader and original-score-worker pins all matched the
reviewed script. The first scheduler observation is **PENDING (Priority)**,
not a GPU resource wait. Request remains one CPU/2 GiB/five minutes in `main`,
with no GPU, no model/tokenizer invocation and no new generation.

Destination:
`artifacts/protected/tricompose_v1_2/paired_conditioning_diagnostics/paired_trace_12793549/`.
Submission is not a diagnostic result; completion and hash-bound text-free
trace readouts still need verification. No source run or recovered score
table was overwritten, and unrelated queued jobs were left unchanged.

#### 12793549 failure and deterministic legacy-role compatibility defect

Scheduler subsequently reports **FAILED, exit 1:0, elapsed two seconds** for
12793549. Its protected worker status reports `ValueError`; no completed
diagnostic manifest exists. Keep this failed job and the consumed V2 source
immutable. Read-only checks of all four generated candidate metadata records
confirm their exact EHR/prompt/seed/request binding and registered image
hashes, without opening a trace body or image pixels in the cache allocation.

The sealed requests use renderer
`roentgen_v2.ehr_context_prompt.v1_1_3`. Their saved
`included_direct_fact_ids` contain **pneumonia plus
congestive_heart_failure** for each of the four requests. The current
`v1_1_4` radiographic phrase inventory excludes congestive heart failure;
the current fact extractor explicitly records this concept as clinical
context, not a direct image finding. Both prior added trace checkers reject
that saved ID list with `recognized_unique_rendered_fact_ids_required`.
This is a deterministic metadata-version compatibility defect, reproduced
with an authored trace fixture and the same ID/version shape. Exact other
runtime trace predicates still require the approved private trace readout;
do not assert this is the only possible failing condition or infer checkpoint
failure, prompt truncation or absent attention masks from it.

Added V3 diagnostic with an explicit **readout-only** role adapter for saved
v1_1_3 metadata. It keeps pneumonia in the radiographic inventory and identifies
the saved CHF entry as legacy clinical context; it **does not** turn CHF into
edema/cardiomegaly, add an EHR reference label, edit a prompt or change the
already sealed score/selection table. Current v1_1_4 requests with a context
ID in their direct list still fail closed, as do unknown IDs/versions and
duplicate IDs. Length/hash/NA checks and the original guard failure are
reported separately. Private failure receipts now include a fixed execution
stage plus an allowlisted error code, never arbitrary exception text.

V3's eight invented-fixture tests pass, including reproducing the old failure,
no source modification, no context-to-image-finding promotion and nullable
attention masks. Metadata-only inspection of the actual four sealed requests
also passes, reconciling four radiographic ID occurrences and four legacy
context occurrences; it reads **no trace body** and invokes **no model**.

Prepared complete replacement diagnostic job script:
`TriCompose-v1.2/slurm/87_paired_conditioning_v3_cpu.sbatch`.
Resource request is unchanged in scale: **main, one CPU, 2 GiB, five-minute
cap, no GPU/inference**. It uses a new exclusive run ID and preserves both
failed jobs and all scored outputs. **Not submitted**; complete-script review
and subsequent explicit user approval remain required.

Final preparation check for the V3 trace job: **3,244 tests pass** in the
existing cache-only CPU allocation, batch syntax is valid, the consumed V2
script hash remains unchanged, and every artifact hash in the recovered
eight-pair score table still matches its original manifest. These checks do
not stand in for the as-yet-unexecuted protected V3 trace diagnostic.

#### Approved version-aware trace-only submission: 12798919

The user explicitly approved the fully displayed script with `go`. Submitted
the unchanged `TriCompose-v1.2/slurm/87_paired_conditioning_v3_cpu.sbatch`
as Slurm **12798919**; script SHA256
`3bf14a91c557b3b7b95c74303bbf907d912ea57d64fc970615bca0a43ef0df2d`.
All five worker/helper/test pins match the approved script. Initial scheduler
status is **PENDING (Priority)** in `main`, requesting one CPU, 2 GiB host RAM
and five minutes, with no GPU, inference, training, downloads or API calls.

Destination:
`artifacts/protected/tricompose_v1_2/paired_conditioning_diagnostics/paired_trace_v3_12798919/`.
This run diagnoses the four already saved synthetic tokenizer traces under
their original renderer version. It does not alter source EHRs, prompts,
scores, selections or the recorded failure status of the earlier jobs.
Submission alone does not establish successful input transfer or clinical
correctness; the completed text-free readout still requires verification.

#### 12798919 completed: input transfer verified, clinical fault unresolved

Slurm **12798919 completed, exit 0:0, elapsed two seconds** on CPU node
`a01-02`. Its four authenticated, text-free trace records show:

- prompt token counts 66/60 match the recorded attention counts for both seeds;
- padded context length is 77 for all four records;
- all four length/hash checks pass; unavailable attention counts: zero;
- four declared radiographic phrase occurrences are present (4/4);
- four legacy CHF context occurrences are separately identified and present;
- zero model calls and no EHR/prompt/score/selection changes.

Diagnostic manifest SHA256:
`e91bdd463a3c262da73c9f34a1f7f5c63c3dd06f6a3086e198dc48025dace254`.
The old combined guard still fails on all four records due to the documented
legacy metadata-role mismatch. Do not rewrite either failed predecessor as
completed. This resolves the recorded tokenizer-boundary checks without
claiming a text-encoder hook, correct pathology, or clinical error localization.
CHF remains context, never an inferred edema/cardiomegaly finding.

Added `TriCompose-v1.2/tools/join_paired_evidence_v1.py` and ten invented,
text-free fixture tests. The cache-only join binds the four completed trace
readouts to the exact eight sealed candidate IDs and six method selections;
every original score cell, NA, selection and source failure status is preserved.
It rejects duplicate/missing/cross-case or cross-seed bindings and never opens
a tokenizer trace body, EHR/report/prompt body, image or weight.

Completed new companion table:
`artifacts/protected/tricompose_v1_2/paired_evidence_tables/paired2_evidence_12784259_001/`.
Files are `candidate_score_table.csv`, `method_comparison.csv`, `summary.json`
and `manifest.json`. Manifest SHA256:
`7719688621274432ccbbe5fd09d26bb679d47f5dc52e1ae595590fd80b8dd7dc`.
Directories/files have the project-compatible group and modes 2770/0660.
No old table, run, consumed checker or professor handoff was overwritten.

**Research consequence:** missing/truncated model text is not supported as the
explanation for these four CXR proxy mismatches. It remains unresolved whether
the generated image violates an observable EHR constraint or the frozen reader
misses it. The independent BioViL endpoint and completed input check do not
identify a clinically wrong modality. Both repair replays still retain the
fixed selections and show no secondary-score improvement; their extra calls
cannot be presented as demonstrated compute savings or superior self-repair.
Proceed using the existing controlled-error/reliability evidence and equal-cost
comparisons, not by changing the fixed EHR, relaxing acceptance retrospectively
or spending on repeated seeds solely to obtain a favorable example.

#### Same-cap benchmark: current feedback loop does not beat static selection

Completed an additional **metadata-only** comparison in existing CPU allocation
12784259. No new Slurm submission or model call. New analysis:
`TriCompose-v1.2/tools/compare_probe_budgets_v1.py`; its eight authored scalar
fixture tests pass. The consumed controller, historical choices and all
generation/score outputs remain unchanged.

The comparison retains all 80 fixed EHRs, all seven methods, all five caps
(4/8/12/20/30), and all 2,800 within-EHR method/cap means. Acquisition/final-choice
replicates are already averaged inside each EHR, never counted as independent
patients. It reuses the previously frozen, outcome-free 49 exact-conditioning
groups and reports **both** case-weighted and equal-conditioning estimands.
Unknown EHR edges remain unavailable: only eight EHRs contribute to direct
EHR-edge metrics, not all 80. Positive CXR-report support is retained separately
from total positive-plus-negative support.

Output:
`artifacts/protected/tricompose_v1_2/probe_budget_comparisons/probe_caps_12784259_001/`.
Files: `method_means.csv`, `paired_contrasts.csv`, `summary.json`, `manifest.json`.
Manifest SHA256:
`f005281e562a570e15dc81b8dc68f970f46e5b91407778344a7d25a7d9af8f62`.

Selected descriptive rows below retain the unnormalized raw metrics. They are
cached development proxies, not clinical correctness or measured GPU savings.

| Cap | Method | Mean simulated calls | CXR-report support / known | Raw BioViL cosine |
| --- | --- | ---: | ---: | ---: |
| 8 | Observed-probe repair | 8.000 | 0.090625 | 0.6270305740 |
| 8 | Static reranking | 8.000 | 0.118750 | 0.6303674025 |
| 12 | Observed-probe repair | 10.025 | 0.113542 | 0.6371742886 |
| 12 | Static reranking | 10.000 | 0.137500 | 0.6388078213 |
| 20 | Observed-probe repair | 10.100 | 0.113542 | 0.6371742886 |
| 20 | Static reranking | 20.000 | 0.186458 | 0.6702482773 |
| 30 | Observed-probe repair | 10.100 | 0.113542 | 0.6371742886 |
| 30 | Static reranking | 30.000 | 0.236458 | 0.6122109429 |

Equal cap is not equal realized expenditure; the cap-8 rows do have identical
simulated charges, whereas the cap-20/30 rows plainly do not. At cap 8, repair
minus static support/known is **-0.028125**; its exploratory case-weighted
conditioning-group bootstrap interval is [-0.0651882, -0.0052529]. BioViL
difference is -0.00333683 with interval [-0.0160988, 0.00530459]. These are
unadjusted developmental proxy intervals, not significance claims about
clinical superiority.

At cap 30, the apparent case-weighted BioViL difference +0.0249633 reverses
under equal-conditioning weighting to **-0.0285017**; both intervals include
zero. Report-finding support is lower for repair under both estimands.
Feedback-versus-no-feedback evidence cannot establish an output-quality
advantage: their selected outcomes and quality metrics are identical across
these caps. Do not suppress this ablation or select cap 30 after seeing its
secondary score. The finite-cache stopping pattern and replay charges are not
an executed online controller or an actual GPU cost reduction.

An independent stdlib replay, without importing the worker/statistical
implementation, reproduces all **420 method means, 240 paired contrasts and
960 bootstrap interval endpoints**, verifies source/output hashes, and checks
project modes/Git exclusion. The eight-candidate joined table was separately
checked cell-for-cell against its original table: all old score and method
cells, missing values and choices are unchanged; 13 derived artifact hashes
and protected modes verify. At that stage the V1.2 suite passed **3,254 tests**
before the eight new budget-analysis tests were added.

**Decision:** do not scale this feedback loop or label it a successful adaptive
repair method yet. The engineering demo can show the full generation graph,
observable input-transfer checks and reproducible same-cap comparisons, while
the first paper-method claim remains unresolved. Next work should explain the
accepted/rejected action effects and design a separately frozen policy/benchmark
that tests a concrete improvement over static selection. It must keep EHRs
fixed, preserve unknowns and original baselines, and avoid tuning on secondary
endpoint outcomes. More blind seeds or changing the EHR are not a remedy for
the observed method gap.

Final verification after both companion tools: **3,262 V1.2 tests pass** in
17.693 seconds. `git diff --check` passes; the protected comparison and joined
score-table paths are excluded from Git. The completed diagnostic, consumed
source/test pins and original selected-score files remain unchanged.

#### Probe-gap diagnostic: acquisition versus rejected observations

Added `TriCompose-v1.2/tools/diagnose_probe_gap_v1.py` with ten authored
four-state tests. It replays all **800** sealed feedback/no-feedback trials
and **400** static trials exactly, then reconstructs every reserved attempt,
unchanged EHR, original action credit, accepted/rejected observation and final
choice. No secondary endpoint is consulted. Static choices that the controller
never observed are used only for post-hoc diagnosis, never as routing inputs.

Completed protected result:
`artifacts/protected/tricompose_v1_2/probe_gap_diagnostics/probe_gap_12784259_001/`.
It contains `case_gap_diagnostics.csv`, `summary.json`, `manifest.json`.
Manifest SHA256:
`628d022085eb39400f472415a671cc3a32f5ec903f34f4edd7301e7d7b73bc79`.
No old code/rules, scores, choices, EHR or protected run was changed.

| Cap | Same final choice | Static choice observed but rejected | Static choice not requested | Static-minus-probe CXR-report support: positive / negative |
| --- | ---: | ---: | ---: | --- |
| 4 | 80 | 0 | 0 | 0 / 0 |
| 8 | 71 | 9 | 0 | 3 / 24 |
| 12 | 71 | 9 | 0 | 2 / 21 |
| 20 | 45 | 7 | 28 | 15 / 55 |
| 30 | 37 | 7 | 36 | 20 / 98 |

At cap 8, all nine different static choices were observed. Eight recorded
rejections cite no strict target-action gain and one cites structure regression.
The net support gap is 27 finding-edge observations, of which 24 are negative
and three positive; these are not independent patients or clinical truth.
The original static selector rewards total classifier/report agreement,
whereas the repair guard requires positive CXR-report gain or resolved
opposition, plus exact fact/quality preservation. Consequently the comparison
mixes acquisition, stopping and acceptance objectives; it cannot alone isolate
the value of adaptive scheduling. Correct negative agreement may be useful,
but is neither a verified clinical gain nor a reason to remove preservation.

At cap 30, all 36 unrequested static alternatives use another image branch;
none has an active final image-probe trigger. With sparse direct EHR evidence,
the current policy deliberately does not claim a wrong image merely from
dependent report votes. Those unobserved candidates do not establish that
blind image regeneration would be clinically justified or cost-effective.

#### Prepared matched-guard fixed-image control, before numerical execution

To separate commit-rule differences from scheduling, add a **fixed-image
guarded report-prefix control**, not a revised adaptive method. It keeps the
initial Sana seed-0 CXR and all EHRs fixed, acquires reports in the unchanged
MAIRA-2/CXRMate-single/LLaVA-Rad/CheXagent-2 source order, and applies the exact
existing `probe_repair_v1.action_credit` to each completed observation.
Unknown is not negative; duplicate, new opposition, lost comparisons/facts,
unavailable or regressed quality still veto commitment. It does not stop merely
because the current proxy needs are satisfied and never acquires another image.

Frozen pre-execution configuration:
`TriCompose-v1.2/configs/matched_report_prefix_v1.json`.
Implementation: `src/tricompose_v12/matched_report_prefix_v1.py` and
`tools/benchmark_matched_report_prefix_v1.py` under V1.2. Thirteen authored
tests pass. Compare all 80 cases at all five caps, including every failure
charge. Initial cost is four simulated calls; each report probe costs two;
these are cache replay accounting, not actual GPU time.

Seal the 400 new control choices before reading the already saved independent
secondary endpoint. Keep old selections/rules unchanged. This is a post-hoc
developmental control to test the scheduling explanation, not preregistered
clinical evaluation, a trained router, a new scorer or prospective repair.

#### Matched-guard control completed: no observed adaptive scheduling benefit

Completed the declared fixed-image control in existing CPU allocation 12784259,
without any new generation or Slurm submission. New output:
`artifacts/protected/tricompose_v1_2/matched_guard_controls/matched_reports_12784259_001/`.
Manifest SHA256:
`8f4c1fd053f975cbbc229b843df1b97449880afa49da67c8792cbbda8c8d1b4f`.
The 400 control choices were sealed before endpoint readout, with SHA256
`e3e0516630af56aa019e1f72588caa236cd9eec4d28cb4639f63335e422a422c`.

**All 80 final candidate IDs match observed-probe repair at every cap: 400/400
case-cap comparisons.** Thus every selected quality/edge/secondary value is
identical, not merely its rounded mean. The fixed schedule accepted zero/one/
seven/seven/seven transitions at caps 4/8/12/20/30.

| Cap | Fixed-prefix simulated calls | Adaptive simulated calls | Same final choices | Shared CXR-report support / known | Shared raw BioViL cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 4.000 | 4.000 | 80/80 | 0.0864583333 | 0.6186388061 |
| 8 | 8.000 | 8.000 | 80/80 | 0.0906250000 | 0.6270305740 |
| 12 | 10.000 | 10.025 | 80/80 | 0.1135416667 | 0.6371742886 |
| 20 | 10.000 | 10.100 | 80/80 | 0.1135416667 | 0.6371742886 |
| 30 | 10.000 | 10.100 | 80/80 | 0.1135416667 | 0.6371742886 |

The control uses the **exact unchanged acceptance guard**. Fixed report
acquisition is marginally cheaper in simulated charges at caps 12/20/30;
neither measured GPU savings nor clinically successful repair follows.
Historical static reranking remains visible but has a different acceptance
objective. Do not claim adaptive superiority by comparing unlike guards.

Independent stdlib verification recomputes all 1,200 case means and 15 method
rows from the original derived four-state vectors, authenticates 960 secondary
lineages and 800 selected source bindings, checks all 400 diagnostic rows,
18 distinct source/code hashes, sealed choices, protected modes and Git
exclusion. **3,285 V1.2 tests pass** in 17.786 seconds. No raw patient input,
synthetic body/pixel, tokenizer or model weight was opened by this analysis.

#### Next cache-only diagnostic: unchanged-guard reachable headroom

Before changing policy, enumerate accepted report/image transitions among the
12 existing slots per fixed EHR, retaining the exact current action-credit
guard. Use initial cost four, report-transition cost two and image-transition
cost four. Compute cheapest accepted paths and reachable sets at all five
caps. Compare with the sealed fixed-prefix selections using exact supported,
comparable and opposition fact identities, positive CXR-report support and
non-regressing artifact/structure quality; do not introduce a scalar weight or
read BioViL to choose an oracle path.

This is a **full-information developmental upper-bound diagnostic**, not an
executable controller, independent clinical truth, prospective repair, or
measured cost. It knows the cached transition outcomes and avoids failed or
rejected attempts; those unavailable-to-router advantages must remain explicit.
Report both unreachable and reachable improvement possibilities, preserving
all 80 EHRs and all old selections. If no dominating path exists under the
unchanged contract, changing acquisition alone cannot yield that type of
improvement on this finite bank. That is a reason to redesign the evidence
benchmark separately, not to loosen guards retrospectively or replace EHRs.

#### Reachable-headroom diagnostic completed: a narrow low-budget opportunity

Added the frozen protocol `TriCompose-v1.2/configs/guard_headroom_v1.json`,
cache-only tool `tools/diagnose_guard_headroom_v1.py` and nine authored fixture
tests. Consumed implementations and all previous choices remain immutable.
Completed output:
`artifacts/protected/tricompose_v1_2/guard_headroom_diagnostics/headroom_12784259_001/`.
Manifest SHA256:
`c81239875dd1b66467eea9621d7cc2128d85c6d03d054c5f76b8a97dc1517a46`.
Files include `frozen_protocol.json`, `case_headroom.csv`, `summary.json` and
`manifest.json`. No secondary scores or synthetic payloads were read.

Across the 80 separate 12-slot graphs there are **119 accepted directed
transitions**. This is not 119 independent cases or clinically correct repairs.
The number of EHRs with a reachable candidate strictly dominating the sealed
fixed-prefix output under the exact protected-fact partial order is:

| Simulated cap | Cases with dominating reachable candidate |
| --- | ---: |
| 4 | 0/80 |
| 8 | 6/80 |
| 12 | 0/80 |
| 20 | 0/80 |
| 30 | 0/80 |

All six cap-8 dominating candidates use CheXagent-2. This is a post-hoc
mechanistic explanation of the source report order, **not** evidence that
CheXagent-2 is clinically best, that an adaptive policy predicts those cases,
or that a new policy improves independent evaluation. Other constraints,
missing EHR observability and the finite candidate inventory remain unchanged.
The oracle knows which transitions succeed and avoids rejected attempts; its
cheapest accepted-path cost is an information-advantaged bound, not executable
cost accounting. Real policy failures must still be charged.

Independent simple-path enumeration (not the tool's shortest-path algorithm)
reproduces all 400 headroom rows and 119 accepted edges, verifies each reported
path's budget, source/output hashes, protected modes and Git exclusion.
**3,294 V1.2 tests pass** in 17.967 seconds. No training, model download,
GPU inference, new Slurm submission, original EHR change or rule relaxation.

**Next discriminating control:** keep the initial MAIRA-2 path fixed and test
all six permutations of the remaining three report experts, at the same caps
and with the same commit guard. Enumerate these schedules in advance; do not
select one using BioViL or report the best developmental order as a held-out
adaptive result. If an inexpensive fixed order attains the low-budget headroom,
report-order choice explains the benefit and a feedback/Agent claim remains
unsupported. Only a separately frozen policy with held-out, outcome-independent
evaluation can establish additional adaptive value. More blind generation is
not the next evidence-producing step on this bank.

#### Prepared six-order report acquisition control (before execution)

Current continuation runs inside existing CPU Slurm allocation 12799642,
not on a login node and without a new submission. New configuration:
`TriCompose-v1.2/configs/report_order_control_v1.json`; implementation:
`TriCompose-v1.2/tools/benchmark_report_order_v1.py`. Keep the initial frozen
Sana seed-0 image and MAIRA-2 report unchanged. Enumerate all six permutations
of CXRMate-single, LLaVA-Rad and CheXagent-2, retaining all 80 EHRs and caps
4/8/12/20/30. This yields 2,400 new cache trials, not new generated triples.

Use the unchanged fixed-prefix acquisition and exact action-credit guard;
reserve two simulated calls for every report probe, including failures.
Require the original schedule to replay its 400 sealed histories, final
observations and charges exactly. Seal all new choices before reading old
outcomes, the full-information headroom table or the independent secondary
endpoint. Never feed those diagnostics or BioViL scores to the schedule.

Report all six schedules plus the existing observed-probe policy, with raw
edge counts, known/comparable coverage, supported positives, proxy oppositions,
artifact validity, raw secondary cosine and simulated charges. Retain the
previously frozen 49 exact-conditioning groups and show both case-weighted
and equal-conditioning means; missing EHR edges remain NA. Report per-case
strict dominance and attainable-oracle coverage separately from numerical
means. This is post-hoc development sensitivity, not independent clinical
evaluation, a policy learned on validation, prospective repair or actual GPU
savings. Do not hide unfavorable schedules or select a best order as if it
were evaluated on an untouched final test.

The first V1 worker invocation failed before choice creation: the frozen
conditioning receipt stores artifact hashes as scalars, whereas the reused
metadata helper expected an object. Preserve that source, its passing 17
fixture tests and sanitized failure receipt at
`artifacts/protected/tricompose_v1_2/report_order_controls/report_orders_12799642_001/failure_receipt.json`
(SHA256 `38fa452dd89e44919e1e8f9f3e74c63f6cd88ac0d43f6e30779135e87558d89f`).
No numerical result or model call was produced by that attempt.

Before re-execution, add companion `benchmark_report_order_v2.py`,
`report_order_control_v2.json` and 21 authored tests. The only executable
change is a bounded, authenticated reader of `case_groups.csv` accepting the
two existing artifact-hash receipt encodings. All six orders, caps, cohort,
guard, estimands, endpoint isolation and claim restrictions are unchanged.
Do not overwrite the failed version or report it as completed.

#### Six-order control completed: inexpensive fixed scheduling explains headroom

Completed V2 in existing CPU allocation 12799642, with no new Slurm
submission or model invocation. Protected output:
`artifacts/protected/tricompose_v1_2/report_order_controls/report_orders_12799642_002/`.
Manifest SHA256:
`72b647db9cd1731083ae6cf832e0fc2d2cad199cd5c9cc5c70a453df2ab190a7`.
All 2,400 new choices were sealed before old-comparison/headroom/secondary
readout; choice SHA256:
`9090a19e61a03effa030958dbafd0cc036ca1b5ba355620e3f7f1784b2a1fb41`.

Files include `order_outcomes.jsonl`, `case_means.csv`,
`method_comparison.csv`, `conditioning_weighted_metrics.csv`,
`case_order_contrasts.csv`, `order_comparison.csv`, `summary.json`,
`frozen_protocol.json` and `manifest.json`. These contain IDs, hashes,
four-state/score metadata and simulated histories, not report/prompt/EHR
bodies or images. All outputs remain group-restricted and excluded from Git.

The initial image/report is always Sana seed 0 + MAIRA-2. The following
table lists only the subsequent report acquisition order. At cap 8, two of
the three listed reports can be requested; the third is not secretly inspected
by the controller. Improvements below use the unchanged exact-fact guard,
not independent clinical adjudication.

| Schedule | Remaining report order | Same as original at cap 8 | Strictly dominates original | Original strictly dominates it | Reaches the six oracle opportunities |
| --- | --- | ---: | ---: | ---: | ---: |
| 001 | CXRMate-single, LLaVA-Rad, CheXagent-2 | 80/80 | 0 | 0 | 0/6 |
| 002 | CXRMate-single, CheXagent-2, LLaVA-Rad | 73/80 | 6 | 1 | 6/6 |
| 003 | LLaVA-Rad, CXRMate-single, CheXagent-2 | 80/80 | 0 | 0 | 0/6 |
| 004 | LLaVA-Rad, CheXagent-2, CXRMate-single | 74/80 | 6 | 0 | 6/6 |
| 005 | CheXagent-2, CXRMate-single, LLaVA-Rad | 73/80 | 6 | 1 | 6/6 |
| 006 | CheXagent-2, LLaVA-Rad, CXRMate-single | 74/80 | 6 | 0 | 6/6 |

Schedules 004 and 006 at cap 8 produce **exactly the same 80 final candidate
IDs as the original schedule at cap 12**, which actually spends ten simulated
calls per EHR. The fixed control therefore achieves those cached selections
using eight rather than ten simulated charges (20% lower in that diagnostic
accounting). This is **not measured wall time/GPU savings**, a held-out policy
result, prospective repair or an adaptive routing benefit. At caps 12/20/30,
all six fixed schedules select the same 80 candidates and spend ten charges.
The existing adaptive cap-12 policy spends 10.025 mean simulated calls and
reaches the same choices; cap-20/30 spends 10.1 with unchanged outcomes.

Descriptive cap-8 proxy endpoints for the original versus schedules 004/006:

| Metric | Original: case weighted | 004/006: case weighted | Original: equal conditioning | 004/006: equal conditioning |
| --- | ---: | ---: | ---: | ---: |
| CXR-report support / known | 0.0906250000 | 0.1135416667 | 0.0765306122 | 0.1054421769 |
| Mean supported positive finding count | 0.7875000000 | 0.8875000000 | 0.8163265306 | 0.9591836735 |
| CXR-report explicit proxy opposition / known | 0.0041666667 | 0.0041666667 | 0.0051020408 | 0.0051020408 |
| Raw secondary BioViL cosine | 0.6270305740 | 0.6371742886 | 0.5814397958 | 0.5980597316 |

Supported-positive count is a **count per EHR**, not accuracy, probability or
recall. Support/known is the mean of case-level finding fractions. BioViL is
uncalibrated, was not consulted to select a schedule, and is not clinical truth.
The six improved EHRs span five exact-conditioning groups; **none has a
directly comparable EHR radiographic fact**. Thus the observed change is
CXR-report proxy support, not demonstrated three-modal clinical improvement.
Direct EHR metrics remain available for only eight EHRs/seven groups and
unchanged; the other 72 are NA, never zero or perfect agreement.

An independent stdlib replay reconstructs all **2,400** trials, **5,280**
charged transitions and exact original guard credits, all **400** source-order
histories, **960** secondary artifact lineages, **2,800** case means, **35**
method means, **980** conditioning-weighted rows, **2,400** case contrasts
and **30** order summaries. It verifies 22 distinct source/code hashes,
sealed choices, group modes and Git exclusion. All **3,332 V1.2 tests pass**
in 27.060 seconds. Failed V1 metadata compatibility status remains preserved.

**Method decision:** inexpensive fixed report ordering attains the entire
observed cap-8 strict-dominance headroom on this finite bank. Do not label it
an Agent innovation or retune a new policy on these same cases and call that
independent validation. Retain 004/006 as development-selected low-budget
fixed comparators, alongside all other schedules and original static/probe
baselines. The next paper-method experiment must provide an independent
error/evidence test and demonstrate additional value over these inexpensive
comparators, not merely over an unfortunate source ordering. Any new inference
requires a separately shown and approved Slurm script; no larger generation
job is authorized or needed to establish the current ordering explanation.

#### Prepared localization-observability diagnostic, before execution

Reuse the immutable, artifact-rewired 80-EHR intervention bank
`benchmarks/pool80_label_targeted_stratified_20261001_001/` under the protected
V1.2 root. It contains 80 untouched controls, 80 CXR swaps and 76 report swaps;
four intended report swaps are unavailable and remain in missingness counts.
Construction used the same frozen XRV/CheXbert labels to select opposite-label
donors. Mechanical replacement targets therefore are not independent clinical
fault labels; untreated controls are not known clinically correct. The old
48/16/16 role partition has been inspected and cannot become an untouched test.

Freeze `configs/localization_observability_v1.json` and metadata-only tool
`tools/diagnose_localization_observability_v1.py` under V1.2. Bind the fixed
EHR and displayed artifacts to authenticated existing unimodal predictions;
do not alter artifacts or state vectors. The predictor sees only an opaque
item ID and complete named EHR/XRV/CheXbert states. Reuse the exact preexisting
`audit_localization_evidence.triad_pattern` rule, with unknown/uncertain
excluded from explicit comparisons and a tentative signal rather than clinical
fault confirmation. Seal predictions before opening the intervention key or
the existing independently executed BioViL swap scores.

For two declared clinical-label-evidence projections—edge counts and complete
named finding states—count identical inputs carrying different mechanical
targets. Report the empirical full-information target-reconstruction ceiling,
balanced ceiling, ambiguity and majority-target baseline for the whole bank
and each historical role. These ceilings apply only to these projections and
this inspected finite sample; they exclude identities/hash cues, pixels, raw
text, model identity, history and BioViL. They are not trained classifiers,
clinical localization accuracy or bounds for richer policies/new cases.
Keep abstention, fixed EHRs, all available items and all missing arms. Do not
enable regeneration or change a winner based on the diagnostic.

#### Localization observability completed: scalar summaries discard target information

Completed in existing CPU allocation 12799642, with **zero new model calls**,
no new Slurm submission and no generation or policy change. Protected output:
`artifacts/protected/tricompose_v1_2/localization_observability/observability_12799642_001/`.
Manifest SHA256:
`8813d183ba5d31135296c4b342511c2fefd63797b64027611b9cc8de4740acec`.
The 236 opaque-item predictions were sealed before the intervention key and
secondary readout were opened; prediction SHA256:
`f9d088812fd8d463451518dc426997ad6c7ad20fb2175f70a129826945e33376`.
The output contains `blind_predictions.jsonl`, `item_diagnostics.jsonl`,
`collision_summary.csv`, `signature_cells.csv`, `summary.json`, the frozen
protocol and manifest. No EHR/report/prompt bodies, pixels or weights were read.

The untouched 80 EHRs contribute 80 controls, 80 actual CXR swaps and 76 actual
report swaps. Four unavailable report swaps remain missing, not silently
excluded from the intended 240-item inventory. Replacements are mechanically
known operations, not independent annotations of clinically faulty modalities.

| Evidence retained | Distinct signatures | Mixed-target signatures | Items in mixed signatures | Unavoidable empirical target errors | Empirical target-reconstruction ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three edges' known/comparable/support/positive-support/opposition counts | 52 | 23 | 184 | 81 | 155/236 = 0.656780 |
| Named 14-finding four-state vectors for all three modalities | 142 | 5 | 29 | 5 | 231/236 = 0.978814 |

For each identical evidence signature, the ceiling assigns its most frequent
mechanical target (`none`, `cxr`, `report`) and sums the corresponding counts.
The balanced ceilings are 0.658114 and 0.979167; the majority-target baseline
is 80/236 = 0.338983. All historical role-specific results are retained.
These are **finite-sample, full-information reconstruction ceilings**, not
an implemented classifier's performance, a held-out generalization bound or
clinical localization accuracy. Neither row is a limit for policies with
pixels, text, history or additional independent evidence. Particularly, 97.9%
does **not** establish that the current system can localize 97.9% of errors.

The unchanged conservative triad rule produces:

| Output pattern | Items |
| --- | ---: |
| Abstain: no direct EHR finding | 212 |
| Abstain: no three-way comparable finding | 21 |
| Tentative CXR-target signal | 1 |
| Tentative report-target signal | 1 |
| No asymmetric direct contradiction | 1 |

Only two of the 156 mechanically intervened items receive tentative target
signals. Each matches its replacement target, but two selected signals are
not grounds for a clinical accuracy or successful-repair claim. Controls are
not clinically correct gold; the one non-asymmetric item is not a declaration
of clinical correctness. Clinical localization accuracy and false-repair rate
remain **NA**, no regeneration was authorized or executed, and old selections
are unchanged. Unknown/uncertain findings never became negative evidence.

An independent stdlib check, without importing the diagnostic predictor,
reconstructs all **236 artifact bindings**, **236 sealed predictions**,
**236 secondary-score lineages and values**, **eight reconstruction-bound
rows** and **444 signature cells**. It verifies 16 source/code pins, output
hashes and project-group modes. All **3,353 V1.2 tests pass** in 25.815 seconds.

**Method decision:** retain typed finding-level evidence and missingness in the
controller interface; aggregate scalar scores alone lose substantial target
information on this bank. Nevertheless, richer bookkeeping is insufficient
to claim clinical fault attribution: sparse direct EHR evidence and dependent
image-conditioned reports remain decisive limitations. Do not loosen the rule,
add disease/device facts to fixed EHRs, infer current edema from CHF, or tune
against the now-open intervention key. The next discriminating experiment
should test whether a separately frozen verification intervention supplies
new evidence beyond cheap fixed report scheduling. Keep detection, tentative
attribution, abstention and independently supported clinical repair separate;
compare against the inexpensive fixed controls at equal realized expenditure.

#### Prepared same-image report-probe control, before execution

Freeze `configs/localization_report_probe_v1.json`, metadata-only
`tools/benchmark_localization_report_probe_v1.py` and authored fixture tests
under V1.2. Keep the same 236 mechanically rewired items and all 80 fixed EHRs;
four unavailable report swaps remain in the intended inventory. This is an
already inspected development diagnostic, not new clinical adjudication.

The initial displayed report is always the fixed CheXagent-2 artifact. The
one declared probe is the previously generated MAIRA-2 report **for the
displayed image**, which may be the donor image after a CXR swap, not MAIRA-2
for the recipient's original image. Bind exact parent-image/hash/XRV states
before projecting only four-state findings into the verifier. The donor's EHR
must never replace the recipient's fixed EHR. No source text/pixels are opened.

Compare no probe, a uniform single-report probe, and a simple anchored probe
requested only where a fixed EHR has at least one explicit direct fact. This
is a cheap missingness control, **not an Agent innovation**. Charge each
requested report plus label extraction two simulated units, including missing
or duplicate results. Cache lookups are not actual new inference or measured
GPU savings. The anchored policy can match uniform corroboration by design
where no initial signal exists without direct EHR evidence; do not call that
equivalence a discovered adaptive quality advantage.

A probe may corroborate only an already proposed unique target, on **all the
same exact finding IDs** that supported the initial signal. Its explicit
states must match fixed EHR evidence; an opposite state challenges the signal,
unknown/uncertain leaves it unconfirmed, and duplicate report bytes provide
no new corroboration. Probe results cannot create an original-report target
from unmentioned findings, override conflicting loci, or become independent
clinical votes. Seal all method outcomes before decoding the mechanical key.
Do not read BioViL or other secondary endpoints, change old selections/rules,
train anything, authorize regeneration, or report clinical localization
accuracy/false repair. The hypothesis is whether this **one declared cheap
probe** adds corroborating proxy evidence, not whether any possible verifier
can establish truth. Preserve all unfavorable and unavailable outcomes.

#### Same-image report probe completed: one proxy corroboration, no localization-coverage gain

Completed in existing CPU allocation 12799642, with zero new inference,
training, Slurm submission or secondary-endpoint reads. New protected output:
`artifacts/protected/tricompose_v1_2/localization_report_probes/report_probe_12799642_001/`.
Manifest SHA256:
`4b7c8678377fecffe4aa9571ca5883cea32ecc382c727c3d748038b8fcee72c7`.
All **708 method/item outcomes** were sealed before decoding the mechanical
key; prediction SHA256:
`6c2e646d7df917b14a25f3c1d6c059544a765f528186c495a60139c41fc359c5`.
Files are `blind_probe_outcomes.jsonl`, `probe_requests.jsonl`,
`method_by_arm.csv`, `summary.json`, `frozen_protocol.json` and `manifest.json`.

| Method, all 236 available items retained | Requested cached report probes | Additional simulated charges | Initial tentative signals | Signals corroborated on all original finding IDs |
| --- | ---: | ---: | ---: | ---: |
| No report probe | 0 | 0 | 2 | 0 |
| Uniform MAIRA-2 probe | 236 | 472 | 2 | 1 |
| Probe only with explicit fixed-EHR evidence | 24 | 48 | 2 | 1 |

The **24 requests are intervention items, not 24 patients**: they arise from
the eight EHRs with direct finding evidence across their available arms. The
remaining 212 items have no direct EHR finding. Uniform probing observes 234
items without a unique initial target; anchored probing observes 22 such
items and skips the 212 unanchored items. Both preserve every missingness and
abstention denominator, including the four unavailable intended report swaps.

The one corroborated signal is a report-target proposal on the report-swap
arm, matching that mechanical replacement target. The CXR-target proposal
remains unconfirmed because the probe does not provide an explicit state on
its required finding. No observed unique signal was challenged, but that is
not a zero clinical false-repair rate. No duplicate-byte or unavailable probe
occurred in this finite cache; the interface still charges and preserves both
when present. Unknown/uncertain cannot corroborate an explicit target.

This **does not increase target coverage**, create an independent clinical
label, establish report factuality, authorize regeneration, or improve old
selected triples. A same-image second report is dependent evidence. Uniform
and anchored corroboration equality follows the declared direct-EHR guard;
it is a cheap missingness control, not an empirical Agent innovation. Report
request/label charges are simulated diagnostic units, not model invocations,
measured wall time or actual GPU savings. No inference was rerun here.

An independent stdlib replay, without importing the verifier or adapter,
reconstructs **all 708 outcomes**, **260 requested-probe artifact bindings**,
all **36 method/arm/historical-role rows**, and all method totals. It binds
MAIRA-2 to the actually displayed image, leaves recipient EHR states untouched
when the image comes from a donor, verifies 16 source/code pins, prediction
seal, output hashes, and project-group 2770/0660 permissions. The output is
Git-ignored; `git diff --check` passes. All **3,376 V1.2 tests pass** in
25.537 seconds, including 23 new invented-fixture tests.

**Next research boundary:** do not repeatedly acquire more image-conditioned
reports to manufacture three-modal certainty. Retain this cheap report-probe
control and the six fixed schedules in future comparisons. A useful new
verification action must add independently supported, scope-qualified evidence
or demonstrate an incremental action effect beyond those controls, with
uncertainty/coverage intact and actual execution costs recorded separately.
Image-finding reader disagreement is still not a generator-failure verdict;
clinical localization and repair accuracy remain NA. Any new model execution
requires its own fully displayed resource/script request and explicit approval.

#### Prepared image-attribution shortcut stress test, before execution

Do not repeat the completed 50-case RICORD image-reader consensus diagnostic.
Instead, freeze `configs/image_attribution_stress_v1.json`, metadata-only
`tools/benchmark_image_attribution_stress_v1.py` and 24 invented-fixture tests.
Reuse the unchanged `image_reader_consensus.decision` function and all four
previous masks: exact XRV opacity at 0.5, all-three-template BioViL mean at
zero, their agreement, and agreement plus all-template nonzero-sign stability.
The tested function's hash must match the old reference receipt; no new head,
threshold, prompt, weight, best-mask selection or model execution is permitted.

Bind the existing actual exact-opacity and BioViL generic-finding score caches
to all 240 image slots. Retain all 960 candidate slots, all 80 immutable EHRs,
and the 236 artifact-rewired intervention items. Preserve four unavailable
report swaps. For interventions, read the **displayed** CXR/report predictions,
not their original parent pair. The full four-state source join authenticates
fixed EHRs; in this exact-opacity projection every EHR remains unknown.
Do not substitute pneumonia, CHF, devices or legacy max-head opacity evidence.

For each fixed image mask, report image acceptance/abstention, report
comparability, explicit proxy support/opposition and coverage. Add a deliberately
unsafe comparator: label every detected image/report opposition as a proposed
`report` target. This is **not an authorized policy or actual clinical fault**.
After sealing all outcomes, count its agreement/disagreement with the mechanical
replacement target separately for control, CXR-swap and report-swap arms and
all historical roles. Flags on unchanged controls are not clinical false
positives; failure to flag is not a clinically clean judgment. Mechanical
targets were label-selected and are not independent clinical error labels.

Write a separate full-bank image/candidate verification table and evidence-
request reasons, without overwriting or reranking any old table/winner. Reuse
the prior four-row reference risk/coverage readout only after outcome sealing;
it does not supply synthetic labels or qualify synthetic transport, report
assertion extraction, other findings or triple accuracy. No raw annotation,
real/synthetic body, pixel, weight, external API or secondary full-report cosine
is read. Clinical attribution remains NA and regeneration remains unauthorized.
This test distinguishes **pairwise disagreement detection from ownership of
the error**, rather than assuming that a stronger image critic solves both.

#### Image-attribution stress test completed: conflict detection does not identify its source

Completed metadata-only inside existing CPU allocation 12799642. No new
Slurm submission, inference, training, downloaded assets or generated artifact.
Protected output:
`artifacts/protected/tricompose_v1_2/image_attribution_stress/image_attribution_12799642_001/`.
Manifest SHA256:
`7e82e30025e344a575bf3f480cc0e9c1255cd62d784cd8323c32dbf52d9c1404`.
All **944 intervention/mask outcomes** were sealed before key decoding and
the old reference readout; outcome SHA256:
`ac263e1a8c4a1dcb388c8328f7c9deac4651ba1e1a15e12a64c72d803a13a62c`.

The new **960-row `image_policy_table.csv`** covers four masks for each of
240 images. **3,840-row `candidate_verification_table.csv`** covers every
original 960 candidate slot under every mask, with exact artifact/EHR lineage,
report state, scoped image proposal, comparison eligibility and evidence-request
reason. It is a separate verification sidecar, not a replacement scoring or
winner table. Other files: `blind_intervention_outcomes.jsonl`, 48-row
`mechanical_arm_comparison.csv`, `reference_scope_record.json`, summary,
protocol and manifest. No synthetic or real clinical body/pixel/weight is read.

| Fixed image mask | Accepted image slots / 240 | Explicit image/report comparisons / 236 intervention items | Opposition flags on 80 controls | Opposition flags on 80 CXR swaps | Opposition flags on 76 report swaps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact XRV opacity at 0.5 | 240 | 111 | 9 | 18 | 34 |
| BioViL all-three-template mean at zero | 240 | 111 | 14 | 22 | 46 |
| XRV/BioViL mean agree | 184 | 90 | 9 | 18 | 34 |
| Agreement plus all-template sign stability | 178 | 86 | 9 | 18 | 34 |

The mean-agreement mask explicitly abstains on 56/240 image slots. The
template-stable mask accepts 178, with 51 reader-disagreement and 11 template-
sensitivity abstentions; prioritizing the template check makes these mutually
exclusive status counts. They are not 62 distinct failing patients. The
different 240-image and 236-intervention denominators must not be combined:
interventions reuse displayed images and contain only the fixed CheXagent-2
report path, whereas the full bank has four experts per image.

**Key counterexample:** the naive shortcut "image proposal and report oppose,
therefore the report is wrong" returns a report target for 18 CXR-swap items
under either consensus mask, even though the mechanically changed modality
is CXR. It also flags nine unchanged controls and 34 report-swap items. The
same 61 flags survive template-stability gating; additional abstention removes
four supported comparisons here, not any of those oppositions. Do not call
34/61 clinical precision: controls are not clinically correct gold, swap
targets are known operations rather than adjudicated faults, donor selection
used existing labelers, and opacity is not the targeted edema/effusion scope.
All four masks and historical-role results remain visible.

This result distinguishes detection from attribution; it does **not** prove
all flags wrong, establish synthetic transport of the 50-case RICORD risk
estimates, qualify cached CheXbert current-opacity assertions, or demonstrate
successful clinical repair. All 80 EHRs remain unknown in this opacity scope;
both EHR edges and clinical localization/false-repair metrics stay **NA**.
The unsafe shortcut is exported solely as a stress-test comparator. Every
actual action-authorization flag remains false; no old score, choice or model
rule changes and no modality is automatically regenerated.

Independent stdlib replay, without importing production readers/worker,
reconstructs **960 image-mask decisions**, **3,840 candidate rows**, all
**944 intervention outcomes**, **48 arm/role comparisons** and the unchanged
four-row reference attachment. It verifies 23 source/code pins, source/output
hashes, the sealed outcome file and project-group 2770/0660 modes. Results are
Git-excluded and `git diff --check` passes. **3,400 V1.2 tests pass** in
25.779 seconds, including 24 new invented-fixture tests.

**Method implication:** keep a mismatch detector and a modality localizer as
different interfaces. More confident same-image evidence does not supply
missing original conditioning intent, and an improved image classifier alone
does not identify which generation stage should be repaired. The new table
now supplies explicit scoped evidence/abstention/request reasons for every
candidate without manufacturing a scalar clinical total. A subsequent
experiment must preserve that distinction, compare against cheap fixed
schedules and image-only/report-only controls, and measure additional
information or action effect rather than rewarding classifier/report agreement.

### Next missing evidence: final conditioning prompt versus generated CXR

After the image-attribution counterexample, prepare a distinct diagnostic:
**does the generated image retain identifiable information about its original
fixed EHR-derived final prompt?** This measures text-conditioning transfer,
not native structured-EHR-to-CXR generation, clinical truth, error ownership,
or an authorized regeneration decision. It is not another report-voting test.

Implementation: `TriCompose-v1.2/tools/score_prompt_image_conditioning_v1.py`,
with fixed protocol `TriCompose-v1.2/configs/prompt_image_conditioning_v1.json`.
Metadata-only preparation authenticates the original professor-delivery
manifest and all 960 candidate-index rows. It preserves 80 fixed EHRs, all
240 CXR slots, all four reports per image, the three original generators and
seed zero. The existing frozen BioViL runtime receipt is reused; neither
clinical bodies, pixels, checkpoints nor tokenizer bodies are opened by
preparation. Existing scores and choices cannot be overwritten.

The separately approved GPU evaluation will encode only saved final synthetic
generator prompts and synthetic CXRs, **not generated reports, EHR bodies,
real targets or raw MIMIC**. No prefix or rewritten clinical statement is
added. The critic's official TextInput terminal punctuation preprocessing is
recorded; its actual unpadded token sequence is hashed without exporting
token IDs/text. Byte-distinct prompts that collide under that tokenizer are
one retrieval group, not false-negative competitors. No input truncation or
empty-prompt substitution is allowed. Another textual prompt is an
unadjudicated competing condition, not a clinical negative.

Compare images against every prompt group **within the same generator**;
cross-generator renderer style is not a discrimination signal. Return raw
matched cosine, matched-minus-other-group mean cosine, complete-inventory
rank intervals and expected recall at 1/5 under uniform tie breaking.
Exact ties must not create a perfect retrieval score. Average unique image
hashes within a prompt group, then average prompt groups equally; also retain
all case slots, alias counts, failures and conditional coverage. One surviving
prompt group is explicitly non-discriminating, not success. Token equality
does not resolve different-token but semantically equivalent prompts, so
retrieval ranks remain descriptive development diagnostics.

The authenticated index contains **147 different image hashes and 98
different final-prompt byte hashes globally**, with **49 prompt/image byte
groups per generator** and 80 retained case slots per generator. With
hash-cached encodings and fixed two-image/two-text replay controls, the
planned encoder accounting is **149 image calls and 100 text inputs**, not
960 independent examples. Controls cannot rehabilitate a failed primary
encoding. Keep image/text replay deltas and typed missingness in separate
receipts. All 960 old candidate rows gain a named secondary conditioning
sidecar; no primary clinical total, selected winner or repair policy changes.

GPU evaluation requires separate complete-script review and subsequent
explicit approval. Preparing an experiment, a passing software test or a
high similarity does not establish successful clinical localization or
clinical regeneration benefit. The previous finite inspected pool is not
an untouched final test. No training, new generator, API or download is needed.

#### Conditioning-transfer preparation completed, inference pending approval

Metadata staging completed inside existing CPU allocation **12799642**:
`artifacts/protected/tricompose_v1_2/prompt_image_conditioning_plans/conditioning_pool240_12799642_001/`.
Plan manifest SHA256:
`8566f1a78abc5aac0a63fb19c03ac01979bebac462ffb9f0a87dbcd342f1646a`.
Consumed worker SHA256:
`b89570362fd8d3f6a81a3d43ffebafed02315bd1f44e871d91ddcdddd56c0b27`.
The plan/config/worker/tests are now immutable; this is a prepared run,
**not a completed metric evaluation**.

An independent stdlib-only replay, without importing the production worker,
verified 13 source/code pins, exact equality of all 960 original index rows,
all 240 image/prompt/EHR bindings, 80 unchanged case anchors, 147 unique image
hashes, 98 unique prompt byte hashes, 49 prompt groups per generator,
149/100 planned image/text encoder accounting and protected 2770/0660 modes
with project-compatible group ownership. It read metadata/source only and
no clinical body, pixel or checkpoint. **3,433 V1.2 tests pass** in 26.133
seconds, including 33 new invented-fixture tests. Tests cover ties, collapsed
groups, tokenizer collisions, missing coverage, fixed lineage, no overwrite,
no score/choice changes and approval-before-read guards. Software tests do
not qualify clinical metrics or guarantee GPU completion.

Prepared batch file:
`TriCompose-v1.2/slurm/88_prompt_image_conditioning_v100.sbatch`.
Resources: **gpu partition, one V100, two CPUs, 8 GiB host RAM, 10-minute
wall-time cap**; private worker timeout 480 seconds, no retry, offline frozen
models, no new generation/training/download/API. `noderes -f -g` showed several
available V100 nodes; availability does not guarantee queue/start time.
Complete-script review and subsequent explicit approval are required.
Future exclusive result root:
`artifacts/protected/tricompose_v1_2/prompt_image_conditioning_runs/conditioning_pool240_<jobid>/`.
Expected outputs include all-image and all-960-candidate secondary conditioning
tables, every within-generator group cosine, group-level summaries, token-hash
and runtime/replay receipts. No Slurm submission has occurred for this job.

#### Approved prompt/image conditioning submission: 12805302

After the complete batch script and resources were displayed, the user
explicitly approved with `go`. Submitted the unchanged
`TriCompose-v1.2/slurm/88_prompt_image_conditioning_v100.sbatch` as Slurm
**12805302**. Script SHA256:
`5fc9392f11d9124708e28f6fcaf318b0384f64636d04d7fed82c3772df503e6d`.
The worker and plan manifest hashes match the reviewed frozen pins.
Scheduler receipt confirms **gpu partition, one V100, two CPUs, 8 GiB host
RAM, ten-minute wall-time cap**, no dependency or resource substitution.
Initial status: **PENDING (Priority)**, not yet an executed metric result.

Exclusive intended result directory:
`artifacts/protected/tricompose_v1_2/prompt_image_conditioning_runs/conditioning_pool240_12805302/`.
This scores only existing hash-bound synthetic prompt/CXR artifacts; no
generation, training, downloads, external API or historical selection changes.
Completion, output permissions, numerical consistency and encoder accounting
remain to be checked after execution. Submission alone is not an evaluation
result or evidence that clinical localization is valid.

#### Prompt/image conditioning result: 12805302 completed and independently replayed

Slurm **12805302 completed**, exit **0:0**, on **d14-10**, total allocation
elapsed **39 seconds**, batch MaxRSS **1,654,664 KiB**. Worker elapsed before
serialization is **22.130096 seconds**, peak allocated GPU tensors including
load **0.713 GiB**; this is not total device reservation. Protected logs and
result files have project-compatible group ownership and 0660 modes, with
2770 result/plan directories.

Output:
`artifacts/protected/tricompose_v1_2/prompt_image_conditioning_runs/conditioning_pool240_12805302/`.
Manifest SHA256:
`241ecf98b9be6dbe2da9c8bba2460cc3ad28dbc61f33827d393f034034bd65bb`.
All **240 image slots** have complete rankings, all **98 global distinct
final-prompt byte hashes** were encoded, and each generator retains **49
critic-token groups**, with no byte-to-token group collision in this pool.
Hash caching retains all aliases while executing **149 image encodings,
100 text inputs in seven batches**. Text replay maximum absolute embedding
delta is **5.960464477539063e-08**; both image replay deltas are zero.
No generator, training, download, external API, old winner or primary score
changes occurred. The job privately read only bound synthetic final prompts
and synthetic CXRs, no EHR/report body or real data.

| Generator | Prompt groups | Matched cosine, group mean | Matched minus other-group mean | Expected retrieval R@1 | Expected retrieval R@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PixArt | 49 | 0.635207 | 0.280820 | 5/49 = 10.20% | 18/49 = 36.73% |
| Sana | 49 | 0.516249 | 0.222866 | 3/49 = 6.12% | 14/49 = 28.57% |
| RoentGen-v2 | 49 | -0.013760 | 0.030398 | 0/49 = 0% | 7/49 = 14.29% |

These are descriptive **within-generator prompt-retrieval** results, not
clinical accuracy, sensitivity, factuality or modality-localization scores.
The uniform-random group baselines are **1/49 = 2.04%** for R@1 and
**5/49 = 10.20%** for R@5. No significance or held-out superiority is
established. Prompt-group equal weighting, unique-image weighting within
groups and tie expectations were frozen before this run. Identifiable text
conditioning is not perfect intent recovery: R@1 remains low even where
matched cosine or margin is large. Different-token prompts can still be
semantically/radiographically equivalent and are not clinically negative
ground truth.

**Do not infer that PixArt has the best clinical image quality or that
RoentGen-v2 is clinically failed from this table.** RoentGen uses a different
model-specific prompt renderer, and the frozen BioViL text embedding operates
in its own domain. Raw cosine and retrieval patterns cannot separate rendering
style, clinically observable information, image content and critic bias.
The actual saved synthetic prompts are deliberately not rewritten in this
diagnostic. Token-count receipts show 21–77 tokens for each distinct Sana/
PixArt prompt and 16–50 for RoentGen prompts; no text-encoding failures or
truncation substitution occurred. The source generator conditioning itself
was not rerun, so this does not establish a generator-input or checkpoint bug.

Independent stdlib replay, without importing the production worker, validates
**11,760 pair cosines**, recomputes all **240 retrieval rankings** with ties,
reconstructs **147 generator/token groups** and equal-weight summaries, and
checks all **960 unchanged original candidate rows** plus their appended
secondary readouts. It verifies 15 metadata/code pins, the six output hashes,
plan/runtime bindings, all **495 source pins by metadata lineage**, encoder
accounting and protected modes. Prompt/image body hashes were checked by the
approved GPU job; the CPU replay does **not** reopen them, model weights or
token IDs. The CPU audit makes zero model calls and does not claim independently
reconstructed clinical truth.

Files: `image_conditioning_table.csv`, `candidate_conditioning_table.csv`,
`within_generator_pair_cosines.csv`, `prompt_group_summary.csv`,
`text_encoding_receipts.json`, `summary.json`, `manifest.json`.
The conditioning sidecar is now completed rather than pending; all clinical
fault attribution, regeneration authorization and EHR-opacity-unknown
constraints remain unchanged. It adds original-intent evidence, not authority
to repair a modality based solely on a similarity rank.

### Conditioning plus mismatch: fixed mechanical attribution ablations

Next cached development question: does original conditioning evidence help
distinguish the two known artifact replacement operations, rather than always
blaming a report whenever an image/report proposal disagrees? The frozen
protocol is `TriCompose-v1.2/configs/conditioning_attribution_v1.json` and the
worker is `TriCompose-v1.2/tools/benchmark_conditioning_attribution_v1.py`.
It reuses the completed 12805302 numeric prompt/image matrix, unchanged exact
XRV and BioViL opacity scores, unchanged cached report opacity states and
all 236 historical intervention items. No body, pixel, checkpoint, external
API or new model call is needed; execution stays in the already approved
cache-only CPU allocation. All 80 EHRs remain fixed and opacity-unknown.

Always compare the **currently displayed image with the recipient's original
final prompt** within its generator. Do not borrow the displayed donor's own
cached prompt match: that would erase the conditioning intervention. A
metadata adapter validates exact artifact/EHR/token-group lineage, then strips
case/model/donor/hash identities from predictor inputs. The predictor only
receives an opaque item ID, numeric image-reader evidence, one report state
and scalar conditioning evidence/rank interval. Item IDs cannot affect a
decision; the known substitution key is opened only after predictions seal.
The pool/key has already been inspected in previous work; the seal is audit
discipline, not an untouched-test or independence claim.

Predeclare five methods, under **all four unchanged image masks**, with no
threshold fitting or best-mask/threshold selection after results:

1. Existing unsafe report-blame-on-opposition control, ignoring conditioning.
2. Conditioning-only control: negative matched-minus-other-group mean gives
   a tentative CXR hint; otherwise abstain, never declare a report fault.
3. Joint margin-zero rule: an explicit image/report opposition plus positive
   conditioning margin gives a report hint; negative margin gives a CXR hint;
   zero/missing margin or absent opposition abstains.
4. Joint rank-one rule using the same opposition gate: the intended prompt
   entirely inside the first-rank tie interval gives a report hint; entirely
   outside gives a CXR hint; boundary ties abstain.
5. Joint rank-five rule, retaining the same rank-interval/tie handling.

These are deliberately **nonclinical attribution heuristics**, not authorized
repair actions. A global prompt retrieval signal is not a finding state and
cannot fill the EHR's unknown opacity; lexical/renderer effects and equivalent
clinical prompts remain unresolved. Both BioViL usages share architecture and
the image, not independent clinical voters. Once conditioning cosine enters
the predictor, it cannot also serve as an independent success metric.

Retain all controls, swapped items, historical roles and four unavailable
report swaps. Count proposed CXR/report targets, abstentions, matches versus
known replacements and recovery over all available intervened items. An
abstention is not a correct/clean control; flags on unchanged controls are not
clinical false positives. Report all methods even if they worsen attribution.
Keep a new 960-candidate/four-mask evidence sidecar without altering any old
score or winner. Every clinical localization/false-repair metric stays NA,
regeneration authorization stays false, and no modality is regenerated.

#### Conditioning-attribution result: added intent similarity does not improve the joint rule

Completed inside existing cache-only CPU Slurm allocation **12799642**;
no new submission or model call. Protected output:
`artifacts/protected/tricompose_v1_2/conditioning_attribution/conditioning_attribution_12799642_001/`.
Manifest SHA256:
`476d3bbb3d4a39d8345c7a69f8be244c273149aee9260461fbda226167b7e13a`.
All **4,720 item/mask/method predictions** were sealed before key read;
sealed prediction SHA256:
`512ad9099a67d0029446fedaf6243b4a7e3dd536722adcda5b3ae7faa3212eb7`.
The consumed protocol/worker/tests and output directory are immutable.

The original 80 EHRs, 240 image slots, 960 report slots and four missing
report-swap arms are retained. **236 conditioning bindings** use the actual
displayed image and the recipient's original prompt. **3,840 candidate-mask
rows** provide a separate joint-evidence sidecar; **240 arm/role/method/mask
rows** retain every ablation and historical role. No old score, selected
triple, generator, prompt, source data or checkpoint changes.

For the fixed mean-agreement image mask, the mechanically changed source
is recovered as follows. Denominators retain all available intervened items,
including abstentions. Abstaining on an unchanged control is not a correct/
clean clinical judgment.

| Predeclared heuristic | CXR replacement target recovered / 80 | Report replacement target recovered / 76 | Tentative targets on 80 unchanged controls | Wrong target on intervened items |
| --- | ---: | ---: | ---: | ---: |
| Naive report-blame control | 0 | 34 | 9 | 18 |
| Conditioning-margin-only control | 38 | 0 | 26 | 26 |
| Joint margin-zero | 7 | 18 | 9 | 27 |
| Joint rank-one | 18 | 0 | 9 | 34 |
| Joint rank-five | 17 | 7 | 9 | 28 |

**Do not present joint margin-zero as a successful improvement.** It changes
seven previously report-blamed CXR swaps to the correct mechanical source,
but changes 16 previously correct report-swap targets to CXR. Total recovered
intervened targets falls **34 -> 25**, while wrong intervened targets rises
**18 -> 27**, with the same nine flagged unchanged controls. Rank-based rules
shift target bias toward CXR, not reliable two-modality localization. The
conditioning-only control recovers more CXR swaps but loses report recovery
and flags 26 unchanged controls; it is not a qualified policy improvement.
All other image-mask results remain in the 240-row table. No best method,
threshold or mask is chosen retrospectively.

The key remains an artifact substitution reference, **not independent clinical
fault adjudication**. Flags on unchanged controls are not clinical false
positives; a clinical flaw could already exist before substitution. No clinical
localization, false-repair rate, successful regeneration or actual compute
advantage is established. Conditioning cosine is now explicitly a predictor
input, so it cannot be reused as an independent success endpoint. Prompt
retrieval does not add missing evidence about a specific finding.

An additional authenticated metadata-only scope audit explains a central
observability gap, without changing the EHR bridge or replacing cases:

- All 80 cached EHR opacity and edema states are unknown.
- Only **8/80** EHRs have any explicit state in the strict cached 14-finding
  evaluation projection: four pneumonia, three support-device and one effusion
  positive, with no explicit negatives. This is the strict cached evaluation
  projection, **not** a claim that only eight original EHRs contain diseases;
  it is distinct from earlier broader bridge conditioning counts.
- CXR swaps target 38 edema and 42 effusion items; report swaps target one
  edema and 75 effusion items. In **154/156 available intervened items**, the
  recipient's cached EHR state for the construction-targeted finding is
  unknown. Only one CXR swap and one report swap have an explicit positive
  effusion anchor. Opacity, the image mask's current scope, is not the same
  as the edema/effusion scope used to construct these swaps.
- Every CXR swap changes the original critic-token prompt group, whereas all
  controls and report swaps keep it. That is a post-seal lineage diagnostic;
  donor/group identity itself is forbidden from the score-only predictor.
  Knowing how artifacts were swapped is not an inference-time clinical signal.

The scope audit consumes only the same authenticated primary numeric cache,
resolver/key metadata and named-state projection; it reads no EHR/report body,
image, weight or token sequence and performs no new extraction. Thus the
scientific next step is **scope-aligned finding evidence and conditioning
observability**, not threshold tuning or automatic regeneration driven by
whole-prompt similarity. Preserve the original 80-EHR cohort; do not invent
edema, device or negative findings to make attribution appear possible.

Independent stdlib replay, importing neither the new predictor nor image
reader, verifies **236 recipient/displayed-image bindings**, recomputes all
**4,720 predictions**, **240 arm/role summaries** and **3,840 original
candidate-mask rows**, with complete inventories, source lineage and NA
semantics. It authenticates 27 numeric/code pins, all output hashes, sealed
predictions and project-compatible 2770/0660 permissions. The replay reads no
bodies, pixels or weights and makes zero model calls. **3,459 V1.2 tests pass**
in 26.558 seconds, including 26 new wholly invented-fixture tests. Passing
tests validate the implementation and audit, not clinical efficacy.

Files: `mechanical_ablation_table.csv`, `candidate_joint_evidence_table.csv`,
`conditioning_bindings.csv`, `blind_predictions.jsonl`, `summary.json`,
`frozen_protocol.json`, `manifest.json`. This completes the combined score
diagnostic; it does not install a repair agent or authorize a modality change.

### Named-finding alignment: six-finding scoring prepared, not executed

The next bounded experiment addresses the mismatch between the opacity-only
image reader and edema/effusion artifact-replacement construction. It does
**not** repair the EHR or select easier cases. Preserve all original 80 EHRs,
240 image slots and 960 triples. The new worker is
`TriCompose-v1.2/tools/score_finding_matched_biovil_v1.py`; its fixed JSON
protocol is `TriCompose-v1.2/configs/finding_matched_biovil_v1.json`.

Prepare, completed inside the existing cache-only CPU Slurm allocation
**12799642**, reads authenticated numeric caches, source code and receipts
only. It opens no EHR, report, generation prompt, image, token sequence or
checkpoint body. The protected immutable plan is
`artifacts/protected/tricompose_v1_2/finding_matched_biovil_plans/finding6_pool240_12799642_001/`.
Manifest SHA256:
`63cd3e4ea1400782ee12f51c2b21a65951182c5b0752d3b94cb81c904d220923`.

Scoring is **pending separate GPU approval**. For each existing synthetic
image, the same frozen BioViL-T runtime will compare authored positive and
negative descriptions of edema, pleural effusion, cardiomegaly, pneumonia,
pneumothorax and support devices, using all three previously used generic
template families. The six-finding scope is a post-hoc development choice,
not an untouched validation protocol. No template or threshold is fitted.
The mean positive-minus-negative cosine is an uncalibrated preference, not
a disease probability. Three phrasings are correlated probes of one model,
not three independent experts. Support-device prompts are exploratory;
the missing XRV device head must remain unknown even if BioViL prefers
device-positive text.

Planned outputs preserve every failed image and every original fact:
`image_scores.json`, `image_finding_table.csv` (1,440 rows),
`candidate_finding_table.csv` (5,760 rows), `summary.json`,
`frozen_protocol.json` and `manifest.json`. Tables bind the **same finding**
to unchanged EHR, cached XRV and CheXbert states, with new BioViL numeric
readouts and explicit comparison coverage. No total clinical score,
winner, intervention-target prediction or regenerated modality is produced.
Reports remain unqualified regarding current/temporal assertion scope;
independent clinical references for these synthetic findings are absent.
All clinical accuracy, confirmed-fault and clinical-repair outputs remain NA.

An independent stdlib preparation audit, importing neither the new worker nor
the image reader, authenticates **94 source pins**, compares all **960 original
index rows** and reconstructs all **5,760 fact rows** from the original cached
edge vectors and source categories. It verifies full inventories, fixed EHRs,
unavailable device heads, 2770/0660 project permissions and encoder budgets.
Strict cached EHR projection remains unchanged: edema/cardiomegaly/
pneumothorax unknown in all 80, effusion positive in one, pneumonia positive
in four and devices positive in three; every other state is unknown. This
counts the existing narrow projection, not diseases in the EHR body.

Expected inference budget: **149 image calls** (147 unique image hashes plus
two replay checks), **38 text inputs** (36 authored probes plus two replay
checks), and three text batches. The prepared exact script is
`TriCompose-v1.2/slurm/89_finding_matched_biovil_v100.sbatch`: one V100,
two CPUs, 8 GiB host memory and a 10-minute walltime cap, with offline local
weights, workspace-only caches and private sanitized logs. No `sbatch` has
been submitted for this experiment. It requires the complete script and
resource request to be shown and explicitly approved first.

**39 new tests pass; all 3,498 V1.2 tests pass** in 28.024 seconds. The batch
script passes `bash -n`. Source, protocol, tests and the sealed plan are now
immutable; future changes require a new version/run. These checks establish
readiness and lineage, not clinical efficacy or a successful repair method.

#### Six-finding scoring completed; cache-only stress evaluation also completed

After the complete script/resource display, the user explicitly approved
submission. Unchanged script `89_finding_matched_biovil_v100.sbatch` was
submitted as GPU Slurm **12809821**, ran on **d13-05**, and completed in
**31 seconds** with exit code **0:0**. Worker time was **17.917 seconds**;
peak allocated GPU tensor memory was **657,864,704 bytes** (approximately
0.613 GiB). All **240 image slots** scored successfully, reusing **147
distinct image hashes**, with 149 image calls, 38 text inputs and three
text batches. Two image replay deltas are zero; text replay maximum absolute
differences are approximately 9.31e-8. These are execution/repeatability
checks, not clinical accuracy.

Completed protected output:
`artifacts/protected/tricompose_v1_2/finding_matched_biovil_runs/finding6_pool240_12809821/`.
Manifest SHA256:
`8fd0930e24ade4de7cd759056da07f880308f95fdaf26c6046f228aa922e9b0b`.
The full **1,440-row image-finding table** and **5,760-row candidate-finding
table** now contain actual frozen BioViL scores, not pending placeholders.
All original EHRs, cached XRV/CheXbert states, prompts and winners remain
unchanged. No raw MIMIC input, EHR/report body or external API was used.

| Finding | Explicit cached EHR cases / 80 | XRV/BioViL mean state agreement / 240 image slots | Explicit report proposals / 960 triples | Three-way comparable candidate-finding rows with agreed image state |
| --- | ---: | ---: | ---: | ---: |
| Edema | 0 | 120 | 174 | 0 |
| Pleural effusion | 1 | 104 | 596 | 8 |
| Cardiomegaly | 0 | 90 | 528 | 0 |
| Pneumonia | 4 | 137 | 84 | 2 |
| Pneumothorax | 0 | 96 | 577 | 0 |
| Support devices | 3 | NA: XRV head unavailable | 158 | 0 |

These are evidence-availability and **proxy agreement** counts. They are not
clinical correctness, and repeated triple rows/shared image hashes are not
independent cases. The ten three-way-comparable rows are **candidate-finding
rows**, not ten patients. A missing XRV device head is not filled by the new
text/image preference. Cached report proposals still lack qualified current/
temporal assertion scope. BioViL preference is not calibrated disease
presence/absence; substantial disagreement with XRV cannot decide which
reader is correct.

While the GPU job ran, the all-six-finding cache consumer was implemented:
`TriCompose-v1.2/tools/benchmark_finding_matched_stress_v1.py`, with a fixed
protocol `configs/finding_matched_stress_v1.json` and 30 invented-fixture
tests. Protocol/worker/test hashes were sealed before inspecting the new
score values. It consumes numeric receipts only in existing CPU Slurm
**12799642**, with no new submission/inference. Every prediction receives
all six named findings, four-state EHR/report/classifier evidence and the
corresponding numeric BioViL pairs; it receives no case/model/donor/hash
identity, source role or known intervention finding/target.

All four predeclared image policies are retained: cached XRV, BioViL mean,
XRV/BioViL mean agreement, and agreement with consistent signs across the
three BioViL template families. Same-finding opposition detection is kept
separate from tentative attribution. The unsafe comparator always blames
the report on opposition; the guarded comparator requires explicit
three-way EHR/image/report asymmetry, with incompatible target hints causing
abstention. Neither is a clinically qualified action policy.

Completed protected stress output:
`artifacts/protected/tricompose_v1_2/finding_matched_stress/finding6_stress_12799642_001/`.
Manifest SHA256:
`2d9ac39eec9b1a82e727ed02cdac9628c34bd51d92066a7f767a1a34ee8483ed`.
It seals **944 item-policy predictions**, containing **5,664 same-finding
diagnostics**, before opening the construction key, then writes **96
arm/role/policy/attribution-control summary rows**. All 236 available items,
80 fixed EHRs and four unavailable report-swap arms remain. The historical
pool/key has already been inspected, so sealing does not imply an untouched
clinical test. The construction-targeted finding is used only in explicitly
named post-seal evaluation columns, never to choose a predictor finding.

| Image policy | Opposition flags on 80 CXR swaps | Opposition flags on 76 report swaps | Opposition flags on 80 unchanged controls | EHR-asymmetry target recovered on CXR / report swaps | EHR-asymmetry targets on unchanged controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached XRV | 65 | 59 | 63 | 1 / 1 | 0 |
| BioViL mean | 52 | 73 | 41 | 4 / 1 | 3 |
| Mean agreement | 23 | 45 | 18 | 1 / 1 | 0 |
| Agreement + stable template signs | 19 | 42 | 15 | 1 / 1 | 0 |

**Do not present these as clinical detection/localization accuracy or repair
success.** Unchanged controls may already contain actual disagreements;
construction labels come from historical classifier proposals. The unsafe
report-blame comparator mechanically recovers report swaps by definition
but attributes every flagged CXR swap to the wrong modality. All-six-finding
flags are not directly comparable to the former opacity-only flags, because
scope and opportunities to flag differ. Agreement masks reduce flags partly
by withdrawing comparisons, not by correcting generation. BioViL-only
EHR attribution recovers four CXR substitution targets but also targets three
unchanged controls and the wrong modality in three report swaps. No best
mask or threshold is selected after these results.

The correct current conclusion is: same-finding numeric evidence is now
available and the test is no longer opacity-only, but reliable fault
attribution remains unestablished. Most EHRs lack an explicit finding anchor;
agreement between dependent/unqualified readers does not create clinical
truth. All clinical localization, false-repair and successful-regeneration
metrics remain NA, and automatic regeneration remains unauthorized. This
experiment does not establish an advantage over fixed generation/reranking.

Independent stdlib replay imports neither scorer nor judge. It authenticates
the GPU/plan/stress artifact hashes, recomputes all **1,440 image-finding
readouts**, **5,760 unchanged candidate-finding bindings**, **944 blinded
predictions** and **96 summary rows**, verifies missing/unknown semantics,
recipient EHR versus displayed donor binding, sealed prediction hash, encoder
accounting and 2770/0660 protected permissions. GPU image pins are compared
as metadata lineage without reopening images, prompts, EHRs, reports or
weights on the CPU. The audit makes zero model calls. **All 3,528 V1.2 tests
pass** in 27.350 seconds; tests validate implementation, not clinical efficacy.

The new worker/protocol/tests and completed output directories are immutable.
The scoring result's main numeric file is `candidate_finding_table.csv`;
the stress result's main comparison file is `mechanical_comparison_table.csv`.

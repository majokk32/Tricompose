# TriCompose V1.2: controlled-intervention preparation

Research protocol: [further development](../docs/further_development.md) and
[localization protocol](../docs/v1_2_localization_protocol.md).

`benchmarks/build_intervention_smoke.py` is a CPU-only, metadata-only builder.
It reads the protected synthetic candidate registry, not source EHR, image
pixels, report text, or real targets. It prepares a fixed-path control, a
different-case report swap, and a different-case CXR swap per case. The
output is a new immutable protected run containing:

- `blind_items.jsonl`: opaque item IDs for policy-facing evaluation;
- `resolver.jsonl`: hash-bound candidate references for a later protected
  artifact loader;
- `intervention_key.jsonl`: mechanical intervention labels and pending
  independent-adjudication status;
- `manifest.json`: source hash, fixed path, arm counts, and status.

The resolver is not itself a model input. A future scoring runner must load
the referenced *artifacts*, compute evidence for each rewired pair, and keep
the intervention key hidden from the policy. An injected swap is not
automatically a clinical error, and an untouched generated pair is not
automatically clinically correct. This stage does **not** report localization
accuracy or claim a validated method. Evaluator inference needs a separately
reviewed Slurm script and explicit submission approval.

`benchmarks/diagnose_cached_pair_sensitivity.py` can recombine existing frozen
XRV and CheXbert artifact labels for a CPU-only manipulation check. It measures
whether a swap changes label disagreement relative to its control. It does not
rerun image-text verification, distinguish image from report fault, calibrate
XRV, or supply clinical ground truth. Results remain protected and diagnostic.

## Initial accessible eight-case smoke (2026-09-30)

The protected intervention run is
`artifacts/protected/tricompose_v1_2/benchmarks/directfacts8_intervention_smoke_20260930_001/`.
It contains eight untouched controls, eight report swaps, and eight CXR swaps
from the predeclared Sana seed-0 / MAIRA-2 path. Artifact bytes and source
patient data were not opened; the builder used the hash-bound candidate table.
The original, group-incorrect duplicate was removed after a byte-identical
project-group copy was validated. The retained directories have mode `2770`
and files `0660`; on this NFS mount, the project group appears as `nobody`.

The cached-label manipulation check is in
`artifacts/protected/tricompose_v1_2/diagnostics/directfacts8_cached_pair_sensitivity_20260930_001/`.
Compared with each control, explicit CXR-report contradictions increased in
only 2/8 CXR swaps and 2/8 report swaps. The untouched controls already had
one contradiction across eight items. These are **not** clinical localization
accuracy, false-repair rate, or evidence that the other swaps were medically
compatible: XRV is uncalibrated here, report and image labelers can miss facts,
and no independent artifact adjudication has occurred. The immediate design
implication is that a single 14-label disagreement score cannot safely decide
which modality to regenerate.

## Eighty-case label-targeted intervention bank (2026-10-01)

`benchmarks/build_targeted_interventions.py` prepares a larger CPU-only,
metadata-only benchmark from the immutable 80-case candidate registry. It
chooses a different-case donor only when the frozen same-modality label state
for edema or pleural effusion is explicitly opposite; unknown and uncertain
never count as negative. It never reads image pixels, report text, or EHR
contents. Each EHR stays fixed, every case retains an untouched control, and
the source case is assigned once to development/calibration/final-test
(48/16/16). The blind item list contains only opaque item IDs; the intervention
answer key is a separate protected file and must not be given to a policy.

The first diagnostic path, Sana seed 0 + MAIRA-2, produced 80 controls and 80
CXR swaps but no qualifying report swaps. The frozen CheXbert output for this
fixed MAIRA-2 path had almost no explicit edema/effusion states; that is a
label-coverage observation, not proof that the reports or model are poor.
The immutable diagnostic run is
`artifacts/protected/tricompose_v1_2/benchmarks/pool80_label_targeted_20261001_001/`.

Two intermediate CheXagent-2 construction runs were retained for provenance:
`pool80_label_targeted_chexagent2_20261001_001` allowed donors to cross split
boundaries, and `pool80_label_targeted_splitlocked_20261001_001` restricted
donors but left the final-test split without a positive report donor. Neither
is the benchmark to score.

The retained benchmark is
`artifacts/protected/tricompose_v1_2/benchmarks/pool80_label_targeted_stratified_20261001_001/`.
Its Sana seed-0 + CheXagent-2 fixed path uses case-level 48/16/16 splits,
stratified solely for positive pleural-effusion donor availability. Donors
remain inside the same split. It has **80 untouched controls, 76 report swaps,
and 80 CXR swaps**; four report swaps are explicitly unavailable. The arm
counts by split are:

| Split | Control | Report swap | CXR swap |
|---|---:|---:|---:|
| Development | 48 | 46 | 48 |
| Calibration | 16 | 16 | 16 |
| Final test | 16 | 14 | 16 |

Artifact hashes, split isolation, and the one-modality-only contract were
revalidated; all 20 synthetic-only unit tests pass. This is **not** clinical
localization accuracy: donor selection uses the same frozen labels that would
make a label-only evaluation circular, and a swapped artifact may remain
clinically compatible. Future verification must use independent evidence and
compare against each case's untouched control; tuning must stay out of
final-test. The `final_test` name in this pilot means held out from subsequent
policy tuning only: the fixed report model was chosen after inspecting
aggregate label coverage over this 80-case pool. A genuinely untouched
paper-grade final cohort must be generated or reserved separately.

`benchmarks/score_intervention_biovil.py` and
`slurm/04_synthetic_interventions_biovil_p100.sbatch` scored the 236 displayed
pairs with frozen BioViL-T as approved job `12530046` (32 seconds). The scorer
read the blinded resolver, not the intervention answer key, and embedded each
of 80 unique synthetic CXRs and reports once. The protected score run is
`artifacts/protected/tricompose_v1_2/benchmarks/pool80_targeted_biovil_20261001_001/`.
The separate CPU-only post-hoc analysis, which read the key only after scoring,
is `pool80_targeted_biovil_analysis_20261001_001/` under the same protected
benchmarks directory. All 22 synthetic-only unit tests pass.

| Arm | Pairs | Fraction with lower BioViL cosine than untouched control | Mean control-minus-swap cosine |
|---|---:|---:|---:|
| Report swap | 76 | 100.0% | 0.431 |
| CXR swap | 80 | 93.75% | 0.377 |

These are strong **cross-case swap sensitivity** results, not error-localization
accuracy or clinical correctness. The score is a single pairwise similarity
number and cannot distinguish an erroneous CXR from an erroneous report.
The report-swap donor pool has few explicit positive examples, so repeated
donors and case/style differences could make this benchmark easier than
natural generation errors. No score threshold was fitted, and the pilot's
`final_test` split is not paper-grade independent because the fixed report
model was chosen after inspecting aggregate label coverage over the pool.

## Direct EHR evidence coverage gate (2026-10-01)

`benchmarks/audit_localization_evidence.py` joined the same blinded benchmark
to the existing V1.1 EHR-edge records without new inference. Its protected
output is `artifacts/protected/tricompose_v1_2/benchmarks/pool80_localization_coverage_20261001_001/`.
The audit counts only a finding that is explicit in the *fixed EHR*, the
displayed CXR label, and the displayed report label. An unknown/uncertain
state is never treated as negative.

Across 80 untouched controls, **72 have no direct radiographic EHR fact**, seven
have a direct EHR fact but no three-way comparable finding, and only **one**
has a three-way comparable finding. The same one-case ceiling appears in both
swap arms (80 CXR swaps and 76 report swaps); the diagnostic state pattern
points to the injected modality for that one case in each arm. This is not
an accuracy estimate: the XRV/CheXbert labels used here also informed donor
selection, so they are not independent evaluation evidence. All 25
synthetic-only unit tests pass.

**Go/no-go:** the current 80-case cohort supports testing pair-mismatch
detection and static selection, but it cannot support a broad claim that the
system identifies whether the CXR or report is clinically wrong. A conservative
policy must abstain on most cases. Before presenting targeted regeneration
as V1.2's contribution, obtain a separately fixed cohort with substantially
more explicit, evidence-grounded radiographic EHR facts and an independent
evaluator, or narrow the claim to mismatch-aware candidate selection.

The initial eight-case smoke was prepared before the required **real
matched-data scorer validation**. It does not clear that gate. The corrected order is documented in
[`docs/further_development.md`](../docs/further_development.md): validate frozen
metrics on real matched and independently adjudicated mismatched pairs first,
then interpret or extend the synthetic intervention benchmark.

## Separate explicit-EHR feasibility cohort (2026-10-01)

`benchmarks/generate_unconditional_ehr_pool.py` uses the already-pinned frozen
SynEHRgy runtime and sampling parameters but writes the new pool with project-
group protected modes. It does not change the older generator or its outputs.
`benchmarks/select_explicit_ehr_cohort.py` predeclares a CPU-only screen for a
**new** 100-case SynEHRgy Qwen2-40bins pool. It checks the immutable source
manifest and every case hash, canonicalizes each fully synthetic EHR, and uses
the V1.1 fact extractor. Eligibility requires a positive, evidence-backed
finding in the latest visit's diagnosis field for cardiomegaly, pleural
effusion, pulmonary edema, pneumonia, pneumothorax, or atelectasis. Clinical
context, uncertain/unknown findings, earlier-visit findings, and inferred
devices do not qualify. The first two eligible cases in source-manifest order
are selected; the complete 100-case denominator and all rejected reasons are
retained in a protected, non-overwriting screen run. If fewer than two qualify,
the pipeline stops without silently generating more or substituting cases.

This is **unconditional EHR generation followed by predeclared eligibility
selection**, not disease-prompted EHR generation or an unbiased prevalence
sample. The checkpoint's `mimic4_hf_bins40` provenance is not established as
the MIMIC-IV-ED schema. `slurm/05_new_explicit_ehr_pool100_p100.sbatch` was
shown in full with its resource request and then explicitly approved. Job
`12531476` completed in 7m28s on one P100, with no CXR/report inference.
Its protected, immutable runs are `ehr_pools/v12_explicit_ehr_pool100_20261001_001`,
`ehr_cohorts/v12_explicit_ehr_screen_20261001_001`, and matching V1/V1.1
staging runs under `artifacts/protected/`. Of 100 generated cases, 74 were
structurally valid and five had an explicit eligible direct finding; the first
two eligible cases were staged. Both selected cases have an explicit pneumonia
finding, so this is a pipeline smoke, not a disease-diverse localization test.
V1 and V1.1 staging validation both passed, and each case has a distinct
clinical intent and distinct prompt for each active CXR model.

The older V1 staging helper initially assigned its new run to the owner's
personal group. The 18-file V1 run was copied byte-for-byte into a project-
group-inheriting directory, rechecked with an identical tree SHA256, and
restored at its original path. The redundant owner-group copy was removed
after verification. All new run directories have mode `2770`, files `0660`;
on this NFS mount project-group ownership displays as `nobody`. The existing
80-case bank remains unchanged.

The next protected CPU-only request run,
`artifacts/protected/tricompose_v1_1/cxr_requests/v12_explicit_two_all3_s2_20261001_001`,
is hash-bound to the two staged EHRs. It contains 12 requests: two cases ×
RoentGen-v2/Sana/PixArt × seeds 0/1. The full frozen CXR inference script
`slurm/06_explicit_two_all_cxr_p100.sbatch` and resource request were shown
and explicitly approved before submitting array job `12533465`. All three
P100 tasks completed without error: four candidates per model, 12 total.
RoentGen-v2 produced 512×512 images (peak 3.283 GiB, 1m37s), Sana 1024×1024
(7.767 GiB, 1m32s), and PixArt 512×512 (12.92 GiB, 3m49s). The three runs
share the exact request-manifest hash; all candidates declare frozen weights,
two cases, seeds 0/1, no added adapter prefix, and runtime-observed tokenizer
input. Output directories/files have modes `2770`/`0660` and inherited the
project mount group. This validates execution and lineage, **not image quality
or clinical agreement**; image contents have not yet been inspected. Both
selected EHRs carry explicit pneumonia, so comparison remains a narrow smoke
test. Report quality and cross-modal scoring require separate evaluation.

A protected, hash-bound 48-request CXR-only report run was prepared at
`artifacts/protected/tricompose_v1_1/report_requests/v12_explicit_two_four_experts_12533465_001`:
12 generated CXRs × MAIRA-2/CXRMate-single/LLaVA-Rad/CheXagent-2. These four
experts receive one current synthetic CXR each; `cxrmate_single` is **not**
CXRMate-ED and receives no structured EHR. The A40 array script
`slurm/07_explicit_two_four_reports_a40.sbatch` and its resource request were
shown and explicitly approved before submitting array job `12538545`. All four
tasks completed with exit code 0: 12 reports per model, 48 total, with one
report per model for each of the same 12 synthetic CXRs. The protected output
runs are `report_candidates/v12_explicit_two_{maira2,cxrmate_single,llavarad,chexagent2}_12538545`
under `artifacts/protected/tricompose_v1_1/`. Runtime and peak VRAM were
MAIRA-2 1m57s/27.359 GiB, CXRMate-single 20s/0.660 GiB, LLaVA-Rad
1m20s/14.334 GiB, and CheXagent-2 1m15s/13.362 GiB. All output manifests
bind to the same request-manifest SHA256, declare frozen weights and a single
current synthetic CXR input, and declare no structured EHR or real target
supplied to the models. Every candidate file exists; output directories/files
have modes `2770`/`0660` with inherited project mount group (displayed as
`nobody` on this NFS mount). These checks establish generation completeness
and provenance, **not clinical quality**; report text has not been manually
reviewed.

For this same October cohort, a bounded CPU-only integrity/structure pass is
complete under
`artifacts/protected/tricompose_v1_1/evaluation/v12_explicit_two_12538545_001/`.
All 12 generated PNGs were readable and nonconstant; all 48 reports were
nonempty and passed their model-specific section contract. The conservative
unsupported-prior/no-change language flag fired on 15/48 reports; this is a
review signal, not a clinical adjudication. The explicit 48-lineage registry
was complete but remained `selection_ready: false` until frozen cross-modal
evidence was computed. The full `slurm/08_explicit_two_score_select_a40.sbatch`
script and resource request were shown and explicitly approved. Job `12538856`
completed in 42 seconds with exit code 0 on one A40. Its protected output is
`evaluation/v12_explicit_two_12538545_001/scoring_12538856/`: frozen XRV
labels for 12 images, frozen CheXbert labels and secondary BioViL-T cosine for
48 image-report pairs, all three cross-modal edge summaries, the 48-row scored
table and two static selections. The selected cases had no hard contradiction
signal under the present diagnostic label rules, but the exhaustive selector
only raised mean support/coverage from 0.2857 to 0.3214 versus the
predeclared fixed Sana seed-0 + MAIRA-2 path, at 6 CXR and 24 report calls per
case rather than 1 + 1. This is same-scorer optimization, not independent
clinical validation. More concerning, the selected `case_035` image-report
pair has BioViL-T cosine -0.020, compared with 0.838 for its fixed-path pair;
the secondary metric was deliberately excluded from selection. This discordance
requires review before any selected output is called clinically better. XRV
scores are not calibrated probabilities, and two cases cannot support a
clinical superiority claim.

The synthetic-only `case_035` selected/fixed/same-image-alternative spot-check
is kept in protected `manual_review_12538856_001/` under that evaluation run.
It found that extra report labels can be rewarded despite less certain
image-grounded detail and an unsupported comparative phrase; it is a
non-radiologist review, not clinical adjudication. A separate, CPU-only
descriptive cross-scorer audit at
`selection_audits/v12_explicit_two_12538856_001/` flags one of the two selected
cases for review (selected BioViL-T rank 21/24, with a large same-image report
gap). This post-hoc flag **does not change selection** and must not be used as
an independently validated threshold or a paper-primary metric.

`benchmarks/analyze_candidate_scorers.py` extends the cached audit to all 48
triples while holding the CXR fixed for each report comparison. The protected
review bundle is
`selection_audits/full48_scorer_audit_12538856_002/` under the same October
evaluation run. It validates candidate/image/report hashes, separates positive
and negative label support, and exports model, image, case and pair tables.
Only two EHR cases underlie the 12 images and 72 report-pair comparisons.
The existing selector and secondary BioViL-T pick the same top report on
2/12 images. Among 52 report pairs with different label support and BioViL
scores, 25 have opposite rankings; 8/13 such pairs with equal total hard
contradiction counts are also discordant. These are descriptive disagreement
counts, not clinical adjudications.

XRV assigns positive states to 43/48 known heads across four PixArt images,
40/48 across four Sana images and 15/48 across four RoentGen-v2 images.
Three images have all 12 mapped, available heads positive. The preprocessing
source uses XRV normalization, center crop and 224-pixel resizing, but the
cause of the positive saturation is not established. This flags a need to
validate operating points and image-domain behavior before treating these
labels as strong selection evidence. On this matched image inventory, MAIRA-2
has the fewest XRV/CheXbert explicit conflict signals (5), while LLaVA-Rad
has the highest mean BioViL-T cosine (0.739). Neither establishes the clinically
best report model. The bundle also retains the previous 80-case swap pilot
as separate context; swap sensitivity is not evidence of within-case report
selection quality. Five CPU-only tests cover unknown-safe counts, image-fixed
comparisons, ties, case denominators and lineage/hash rejection. No new model
inference or GPU job was needed for this audit.

`real_validation/calibrate_cached_xrv.py` with
`slurm/09_cached_xrv_thresholds_score_cpu.sbatch` completed as explicitly
approved Slurm job **12544371** (1 CPU, 4 GiB, no GPU; 3 seconds, exit 0).
The immutable protected result is
`evaluation/v12_explicit_two_12538545_001/weak_thresholds_12544371/`
under `artifacts/protected/tricompose_v1_1/`. It reuses the fixed real
128-validation/128-test XRV cache and reads source patient split metadata only
within Slurm, never real image pixels or report text. Thresholds fit validation
only, with at least 20 explicit positives and 20 explicit negatives per finding,
and are evaluated once on the cached test cohort. Missing reference heads or
insufficient support disable a finding rather than defaulting it to negative.
The same enabled-head subset is used for before/after synthetic positive-call
counts, separating threshold changes from reduced label coverage. The job also
recomputed the October 48-candidate scores and diagnostic selection with cached
CheXbert/BioViL evidence.

Only **pleural effusion** met the predeclared validation support requirement
(40 explicit positives / 24 negatives). Its validation-fitted threshold is
0.745114535. On the held-out-from-fitting test subset (47 positives / 19
negatives), balanced accuracy changed from 0.6473 to 0.7088, sensitivity from
0.9787 to 0.6809, and specificity from 0.3158 to 0.7368. The test negative
count still falls below the predeclared 20-per-class support flag; these are
descriptive weak-reference results, not a validated clinical operating point.
Pneumonia had only 18 / 11 explicit validation references and is disabled;
all other insufficient or unavailable heads are also disabled, never negative.

On the **same one-head subset** across the 12 cached synthetic images, positive
calls changed from 7 to 1. Masking the other heads does not establish that the
earlier multi-head saturation or synthetic-image fidelity has been repaired.
Both fixed EHR cases have pneumonia, so their EHR-CXR edge now has **zero
comparable direct facts**. The reused selector exports zero support and zero
coverage for that edge; this means unavailable image evidence, not a negative
pneumonia finding or clinical inconsistency. The exported selector's inherited
`diagnostic_uncalibrated_cxr_labels` status remains conservative; consult the
calibration provenance and protected `STATUS_AND_INTERPRETATION.md` for the
actual one-head weak-reference threshold scope.

The re-ranked diagnostic outputs are RoentGen-v2 seed 1 + LLaVA-Rad for
`case_035`, and PixArt seed 1 + CheXagent-2 for `case_045`. Their displayed
balance score is 83.33, but it uses a different, much narrower available-label
scope from the original 48-row audit. **Do not compare that number with the
old balance score as an improvement, designate these outputs clinically best,
or use this run to authorize EHR-CXR repair.** The original outputs and choices
remain unchanged. The immediate next gate is a separately fixed real validation
cohort with adequate explicit positive/negative evidence for the relevant
findings, followed by independent clinical evaluation; do not lower the support
requirement post hoc to force every head on.

References remain report-derived weak labels, the test pilot was previously
inspected descriptively, and the historical real adapter lacks a recorded
adapter-file hash. Its protocol is reconstructed from recorded XRV library
fingerprints and the inspected implementation, checked against the synthetic
scorer fingerprint, with that limitation retained in provenance. No model
weights are fitted; this is operating-point selection, not probability or
clinical calibration. No new image/report inference was performed. Seven
synthetic-fixture tests cover this cached runner, and seven existing calibration-
contract tests pass. Every subsequent batch submission still requires its full
script/resource request and explicit approval.

### Completed new-patient reference coverage audit

`real_validation/audit_reference_coverage.py` and
`slurm/10_real_reference_coverage_cpu.sbatch` completed as explicitly approved
job **12548196** (1 CPU, 8 GiB, no GPU; 5 seconds, exit 0).
This CPU-only step reads linkage metadata and official report-derived finding
labels inside Slurm, not images, report text, or EHR clinical fields. It verifies
the earlier real-XRV source hashes, excludes every previously scored patient
(including that patient's other studies), verifies the original train/val/test
patient boundaries, and selects one AP/PA study per remaining patient by a
label-independent deterministic hash. It then selects a bounded, deterministic
label-stratified cohort within each existing split, targeting at least 20
explicit positives and 20 explicit negatives per supported finding. Unknown
and uncertain labels remain separate. At most 512 patients per split are
allowed, and the new image-side pilot is gated on pneumonia quotas in **both**
validation and test. Unsupported findings are reported, not silently filled.

The retained output is a new protected
`real_validation/real_reference_coverage_12548196/` run with aggregate
`summary.json/md`, a hash manifest, and protected internal `cohort.json` row
references. No raw patient key or source artifact path is serialized into the
cohort. A successful quota check only authorizes planning another separately
approved scorer job, not automatic GPU submission or targeted regeneration.
The cohort is label-stratified, not a prevalence sample, and references remain
weak; no paper-primary or independently adjudicated final cohort is claimed.
Nine synthetic-fixture tests cover deterministic selection, patient and split
exclusion, binary quotas and budget failure, missing/uncertain references,
label-independent study choice, and the Slurm-before-real-read guard.

The frozen cohort contains **103 validation patients and 105 test patients**,
all disjoint from the prior 256-patient scorer pilot. Every one of the eight
available reference findings has at least 20 explicit positive and 20 negative
labels in both selected splits. Pneumonia has 24 / 22 validation and 22 / 21
test references; consolidation reaches exactly 20 / 20 in both. The cohort
SHA256 is `7c097915fda822876135b6e4a43b8db485bbab9f3f2c1d92fd33fa4ea4cc9ba0`.
This clears the **reference quantity** gate only, not classifier performance or
independent image-ground-truth validation. No CXR pixels were opened by this job.

`real_validation/score_fixed_reference_cohort.py` and
`slurm/11_fixed_reference_xrv_p100.sbatch` were shown in full with the resource
request and explicitly approved, then submitted as job **12548713**.
The job requests one P100, 2 CPUs, 8 GiB RAM and a 15-minute upper limit.
At submission it was pending scheduler priority; completion and test performance
remain to be verified. The reviewed batch-script SHA256 is
`abaae3e3f068103932f0f116363d92f876b6871316992fc17eaa7e0aacff60df`.
It reconstructs and hash-validates the fixed selection before image reads,
rejects prior/duplicate images, applies the same frozen XRV checkpoint to the
208 real CXRs inside Slurm, and saves protected individual scores plus an
aggregate summary. Unlike the earlier pilot, recorded peak VRAM includes
model loading and the adapter/protocol fingerprints are explicit. This is a
label-stratified weak-reference pilot; possible scorer-pretraining overlap is
not ruled out. No real report text or EHR is sent to the scorer or an API.

The cached-threshold runner now accepts an optional, hash-bound `--fixed-cohort`
for this new 103/105 split. The historical default still requires exactly
128/128. Fit uses validation only; test is held out from fitting, although its
label quotas have already been audited. The job fitted per-finding operating
points and relabel the existing 12 synthetic XRV caches without regenerating
anything; it does **not** select new winners or start a repair loop. Review
test sensitivity/specificity and weak-reference discrimination before another
reranking run. Nine synthetic-fixture tests cover the new fixed-cohort runner
and calibrated protocol/count compatibility; previous tests remain passing.

At the next resource check, normal `gpu` P100 slots were no longer free and
job `12548713` remained pending priority, with a scheduler estimate around
19:43 PDT (an estimate, not a guarantee). `debug` still showed two idle P100s.
`slurm/12_fixed_reference_xrv_debug_p100.sbatch` is a prepared alternative
with the **same** GPU/CPU/RAM/time request and computation, changing only the
partition to `debug`; its SHA256 is
`159b1a264df949270ab35e7c98b95840011ca7cb9b9b2c99db0c87503ff612f0`.
The complete replacement script was shown and explicitly approved. The old
job **12548713** was confirmed pending, canceled with `scancel --state=PENDING`,
and confirmed canceled at zero elapsed compute time. Replacement job
**12549079** was submitted and started immediately on `debug / e23-02`,
requesting one P100, 2 CPUs, 8 GiB and the unchanged 15-minute limit.
There was no duplicate running inference. Job **12549079 completed in 1m34s**
with exit code 0. The fixed 208-image XRV pass took 87.286 seconds; cached
threshold fitting completed afterward. The retained protected runs are
`real_validation/real_xrv_reference_12549079/` (individual scores and aggregate
summary) and `real_validation/real_xrv_thresholds_12549079/` (thresholds,
aggregate held-out-from-fitting results and relabeled synthetic caches).
All new output directories/files have mode `2770`/`0660` with inherited
project-group ownership; the previous generation/selection runs are unchanged.

| Finding | Test weak-reference AUROC | Test BA, default 0.5 | Test BA, validation-fitted |
|---|---:|---:|---:|
| Atelectasis | 0.754 | 0.510 | 0.707 |
| Cardiomegaly | 0.788 | 0.688 | 0.766 |
| Consolidation | 0.850 | 0.625 | 0.725 |
| Edema | 0.775 | 0.585 | 0.658 |
| Lung opacity | 0.741 | 0.559 | 0.716 |
| Pleural effusion | 0.899 | 0.682 | 0.810 |
| Pneumonia | 0.643 | 0.624 | 0.585 |
| Pneumothorax | 0.673 | 0.525 | 0.561 |

Eight findings now have sufficient explicit reference quantity to fit an
operating point; that does not mean all eight are reliable clinical verifiers.
Pneumonia sensitivity fell from 0.7727 to 0.4091 while specificity rose from
0.4762 to 0.7619. Pneumothorax discrimination also remains weak. **Do not tune
thresholds again on these test results, automatically flip back to 0.5, or
authorize targeted repair from these heads.** The test macro AUROC is 0.7653
over eight findings, a descriptive weak-reference score, not clinical accuracy.
All thresholds remain `primary_metric_eligible: false` pending independent
image evidence and clinical review. No test-informed reliability mask or new
selection rule has been fitted.

On the identical eight-head subset of the 12 synthetic images, positive calls
changed from 64/96 to 40/96. This is a verifier decision change, not improved
generated image quality or proof that all saturation is fixed. Six unavailable
reference heads remain unknown. The cached labels were subsequently used for
the separately approved CPU-only **diagnostic** 48-candidate re-score below.

`slurm/13_cached_eight_head_rescore_cpu.sbatch`
(1 CPU, 4 GiB, 5-minute upper limit, no GPU) was shown in full and explicitly
approved, then completed as job **12550488** in **2 seconds**, exit code 0.
It verifies
the pinned registry, relabeled XRV cache and aggregate validation hashes,
reuses the immutable CheXbert/BioViL evidence, recomputes the three edges, and
applies the same diagnostic lexicographic policy to the 48-row/two-case bank.
The retained output is `eight_head_rescore_12550488/` under the October V1.1
evaluation root, not a replacement of old runs or the 80-case bank. There are
48 unique candidate triples, 12 unique synthetic CXRs and two selected triples;
each fixed EHR has 24 candidate triples. No image/report model was called again.

`benchmarks/annotate_scoring_scope.py` produces a companion `scoped/` candidate
CSV with unchanged score values/winner flags and explicit
`primary_metric_eligible=false`, `targeted_repair_approved=false`, and pending
independent-clinical-evaluation fields. Its protected verifier profile attaches
the aggregate per-finding test AUROC, operating-point performance and scope.
It does not fit a reliability mask, change thresholds, alter a selection or
declare the selected pair clinically best. Five invented-fixture tests verify
unchanged scores, rejection of patient-record inputs/forged primary claims,
cohort-count consistency and absence of test-informed mask fitting.

| Method, identical eight-head scope | Diagnostic balance | Support / known facts | Hard contradictions / known facts | Coverage |
|---|---:|---:|---:|---:|
| Predeclared Sana seed 0 + MAIRA-2 | 52.5 | 0.20 | 0.15 | 0.35 |
| Exhaustive lexicographic selection | 70.0 | 0.40 | 0.00 | 0.40 |

The displayed balance is `50 * (1 + (supported - contradictions) / known)`;
it is not the selection objective or a correctness probability. Selection
still prioritizes hard gates, contradiction counts, direct EHR support,
CXR-report support, structure, runtime and deterministic ID. Generation costs
remain 6 CXR + 24 report calls per case versus 1 + 1 for the fixed baseline;
the two-second cache replay is not the cost of generating those candidates.

**The ranking conflict remains unresolved.** One selected pair still has
secondary BioViL-T cosine -0.020 versus 0.838 on its fixed path, and its
EHR-report edge has zero comparable facts. Unknown is not a contradiction,
but zero contradictions is not full evidence of correctness either. Across
the 12 images, the label-based selector and BioViL-T select the same top report
on only 3/12 images. Both fixed EHRs contain pneumonia, whose weak-reference
test AUROC is 0.643 and fitted sensitivity 0.409. This result therefore shows
that the cached scoring/selection contract runs, **not** that clinical quality
has improved or that error localization/targeted repair is now safe.

The reusable artifacts are `scoped/candidate_score_table.csv`,
`scoped/verifier_validation_profile.json`, and
`selection/selected_triples.jsonl` inside the new run. All 23 manifest-listed
artifact hashes were checked, and all 47 run entries passed protected
directory/file mode and inherited project-group checks. The inherited V1.1
selection summary still says uncalibrated labels; the actual scope is
validation-fitted operating points against weak report-derived references,
**not** probability or independent clinical calibration. The scoped companion
profile carries that provenance without changing the original summary.
Independent within-case evidence and clinical adjudication are the next gate;
no test-driven threshold/mask changes, agent or repair loop were implemented.

### Completed modality-separated secondary verifier

`benchmarks/verify_candidate_findings_qwen.py` reuses the existing local frozen
Qwen2.5-VL checkpoint and compatibility loader without downloads or changes to
the external model/environment. Its approved-allocation guard precedes artifact
loading and GPU-framework imports. The bounded run performed **12
image-only and 48 report-only calls**: no prompt contains both modalities, the
EHR, candidate/model IDs, old scores or winner flags. Every image is assessed
once, then reused for its four report comparisons. The extraction scope is the
same eight named findings as the current weak-reference pilot, not a complete
clinical ontology; devices, location, severity and temporal facts are outside
this extraction contract.

Responses must contain all eight named keys with
positive/negative/uncertain/unknown states. Missing keys, duplicate keys,
positional vectors, extra fields or token-limit exhaustion make evidence
unavailable, never negative or an invented perfect score. There is no scalar
LLM score or response-based automatic retry. Model-file hashes, input artifact
hashes, adapter/prompt versions, runtime and peak allocated VRAM including
model loading are recorded only in new protected outputs.

`benchmarks/analyze_qwen_evidence.py` joins these secondary states to the
unchanged 48-candidate selection and existing grounded EHR states. It exports
`candidate_evidence.csv` with the original scores/winner flags preserved,
separate support/contradiction/coverage for all three edges, and XRV-image /
CheXbert-report state disagreements. Fixed-versus-selected comparisons remain
descriptive. It does **not** treat Qwen as clinical ground truth, establish
pretraining/data independence, re-rank candidates, fit thresholds/masks or
authorize a repair action.

`slurm/14_separated_qwen_review_a40.sbatch` was shown in full and explicitly
approved, then submitted as job **12553939**: one A40, two CPUs, 32 GiB host RAM,
20-minute upper limit on `gpu`. The submitted script SHA256 is
`b4a9c88f78dc90b2d3db5c536c445e7c74c38e5ee89b7c5f4ff39384446af510`.
The initial scheduler check reported pending priority; the job subsequently
ran on `a03-06` from 21:31:37 to 21:35:13 PDT and **completed in 3m36s**, exit 0.
The main verification program took 210.548 seconds, including checkpoint
fingerprinting; model loading/inference took 182.185 seconds. Peak PyTorch
allocated VRAM including model loading was **15.642 GiB**, not total driver
memory or reserved-memory usage. The queued request was not canceled, changed
or duplicated because completion was established before the proposed GPU switch.
The existing unquantized weight file is approximately 16.6 GB, so this request
leaves substantially more activation-memory margin than a 16-GiB P100.
The latest resource snapshot showed planned/mixed A40 slots, not guaranteed
immediate allocation. Synthetic input lineage/hash checks passed without
inference; 13 new invented-fixture tests and all **80 V1.2 tests** passed after
including the existing bridge/SynEHRgy import paths below. Output is protected
under `verification_runs/qwen_separated_12553939/` and its `_analysis` companion
inside `artifacts/protected/tricompose_v1_2/`. All 12 image responses and 47/48
report responses met the strict named-finding contract. One report-side Qwen
response failed parsing without reaching its token limit; its eight states
remain unknown/unavailable, and the original generated report is retained.
All 48 candidate rows remain represented. The five manifest-listed output
hashes, protected modes/group and preservation of all old score/winner fields
were checked.

| Original method | Qwen CXR-report known / comparable | Supported | Contradiction signals | Coverage |
|---|---:|---:|---:|---:|
| Fixed Sana seed 0 + MAIRA-2 | 14 / 4 | 3 | 1 | 0.286 |
| Original static selections | 13 / 8 | 6 | 2 | 0.615 |

The selected pairs gain Qwen support/coverage but also have more contradiction
signals. On comparable facts the signal rate is 1/4 versus 2/8 (both 0.25);
different coverage prevents declaring either method clinically superior.
Both selected EHR-CXR and EHR-report edges have zero comparable direct facts
under the separate Qwen states; uncertain is not negative, support or success.
Across all twelve images, Qwen and XRV have opposite explicit states in 28/80
comparable image-finding pairs (35%). That measures evaluator disagreement,
not the error rate of either evaluator. The original zero-contradiction static
choices therefore remain **diagnostic, not clinically verified**. No winner,
threshold, mask or candidate artifact was changed, and no repair loop was run.

### Completed cached per-finding evidence contract

`benchmarks/build_finding_review.py` is a bounded **metadata-only** follow-up;
it does not run an evaluator, open image pixels/report text, compute a new
clinical score, change a winner or execute an action. It verifies the preceding
Qwen analysis, source CSV/JSONL, cached finding vectors and staging-fact hashes.
It reuses the existing `DIRECT_EHR_TO_CHEXPERT` mapping rather than interpreting
raw EHR again or introducing new clinical rules. Each non-unknown direct EHR
fact must retain evidence IDs and source-field pointers; known states without
provenance are refused. Weak CHF/medication context remains excluded.

The immutable protected result is
`artifacts/protected/tricompose_v1_2/diagnostics/qwen_finding_review_12553939_001/`:

- `fact_evidence.jsonl/csv`: **384 rows = 48 triples × 8 findings**. Each row
  retains five states (EHR, XRV, CheXbert, image-only Qwen, report-only Qwen),
  support/opposition/unknown/not-comparable relations, missing-evidence reasons,
  fixed-EHR and artifact hashes, staged EHR evidence IDs/source fields, and a
  deterministic finding evidence ID.
- `review_items.jsonl`: 48 candidate-level review records with triggering
  evidence IDs; no confirmed faulty modality or regeneration authorization.
- `adjudication_template.jsonl`: 48 pending review placeholders, **not gold
  labels or an independently adjudicated benchmark**.
- `summary.json/md`: descriptive counts. Image-only edges use 96 unique
  image-finding pairs rather than 384 repeated report-linked rows.

All 48 triples have at least one image-evaluator opposition signal somewhere
in their shared image. This is **not 48 clinically failed samples**: XRV and
Qwen may disagree, and four reports reuse each of twelve images. Report-label
extractors explicitly disagree on only two candidate-finding rows; their much
larger unknown/uncertain region must not be counted as agreement. The two
original winners each retain one Qwen image/report opposition in the protected
summary, and neither is automatically attributed to an image or report error.
Clinical severity and confirmed faulty modality remain unset. A direct EHR
finding is an existing structured-clinical proxy, not independent image truth.

The run took under one second of lightweight cached-metadata work and needed
no Slurm/GPU submission. All six output artifact hashes, bound source hashes,
group ownership and modes passed checks. Ten new invented-fixture tests cover
deterministic evidence IDs, EHR provenance, immutable state/hash bindings,
unknown/uncertain/unavailable semantics, image-side deduplication and refusal
to assign fault from correlated or contradictory evaluators. All **90 V1.2
tests** pass. This completes a traceable fact/review table, not a trained router,
clinical localization validation, blinded review, or targeted repair.

### Completed selected-disagreement engineering audit

`benchmarks/audit_selected_disagreements.py` follows the cached finding table
without inference or interpretation of report text/image pixels. It reproduces
the previous named-state join, checks generation manifests and actual synthetic
artifact byte hashes, replays all frozen XRV finding decisions, and checks
recorded Qwen adapter/prompt fingerprints and response token metadata.
It neither recalibrates a threshold nor changes historical scores/winners.

The protected result is
`artifacts/protected/tricompose_v1_2/diagnostics/selected_disagreement_audit_12553939_001/`.
The two original selected-pair Qwen opposition rows passed these engineering
checks; their responses were complete and not token-limited. `selected_review.json`
retains evidence IDs, artifact pointers/hashes, finding states and exact frozen
operating points. `peer_report_states.csv` contains eight related report rows
on the same two images; these are correlated descriptions, not independent votes.
`summary.json/md` records the completed checks and unresolved clinical boundary.

No clinical fault has been assigned. An XRV score's distance from its cutoff
is not confidence or a calibrated error probability. Qwen raw response text was
not retained, so this follow-up cannot independently reparse it or audit its
semantic reasoning. Missing EHR evidence and CheXbert unknown remain unavailable,
not negatives. The next clinical gate still requires independent artifact
review and controlled artifact-level interventions, not majority voting or
automatic regeneration. Nine new invented-fixture tests pass; all **99 V1.2
tests** now pass. No new Slurm submission was needed for this bounded audit.

### Completed minimal report-polarity diagnostic

`benchmarks/build_report_polarity_benchmark.py` preserves the two selected
synthetic images/EHRs and creates separate protected report copies. It accepts
only one isolated, explicitly affirmative finding sentence, refuses mixed
findings/negation/uncertainty/repeated mentions, and records a reversible edit
with unchanged prefix/suffix. Opacity is not promoted to consolidation to
manufacture an eligible edit. This whitelist is a narrow construction contract,
not a new clinical extractor or an assertion about the image.

The immutable bank is
`artifacts/protected/tricompose_v1_2/benchmarks/report_polarity_smoke2_12553939_001/`:
two byte-identical original controls, two token-preserving whitespace controls,
one explicit-positive-to-negative report edit, and one unavailable edit with
its rejection reason. All five report copies, the resolver, intervention key,
summary and source hashes passed protected permission/hash checks. The original
EHRs, CXR/report files, scores and winners are unchanged.

`benchmarks/score_report_polarity.py` runs frozen CheXbert and
BioViL-T in separate existing environments. Its inference guard precedes model
and data access. Evaluators read only a hash-checked allowlisted resolver, not
the intervention key; the answer key is consumed only by the subsequent cached
analysis. It measures target polarity extraction, non-target label changes,
whitespace invariance and raw cosine differences. An unchanged original is not
clinical gold, and a cosine decrease is not assumed to be the correct direction.
This is a one-edit diagnostic, not clinical localization accuracy or an
independent validation of the original winner/image. No thresholds, ranking,
or repair actions are changed.

The approved `slurm/15_report_polarity_debug_p100.sbatch` completed as job
`12558784` on debug node `e23-02` (P100) in **49 seconds**, exit 0. Its request
was one GPU, two CPUs, 16 GiB host memory and a ten-minute wall-time cap;
the cap was not expected runtime. Outputs are under
`artifacts/protected/tricompose_v1_2/verification_runs/report_polarity_12558784_{chexbert,biovil,analysis}/`.

| Mechanical check | Result | Scope |
|---|---|---|
| Explicit target positive-to-negative report edit | CheXbert positive -> negative; no other head changed | One eligible polarity pair, not clinical detection accuracy |
| Whitespace-only controls | Identical CheXbert vectors in both cases; BioViL cosine deltas 0 | Token-preserving formatting invariance, not clinical agreement |
| BioViL response to the polarity edit | Cosine delta -0.02515189 | Sensitivity only; the decrease is not proof that the edited pair is wrong |
| Second planned target edit | Unavailable; no explicit consolidation phrase | Kept in the denominator; no invented target statement |

CheXbert processed five unique reports in 17.128 seconds and BioViL-T processed
two images/five reports in 21.094 seconds, including loading and output-related
work. Their peak PyTorch allocated VRAM **including loading** was 1.227/0.570
GiB; these are not total driver-reserved GPU memory measurements. All three
stages completed, source/output hashes and protected modes/group passed checks,
and original reports/images, source-score files and winner flags are unchanged.
No Qwen inference, EHR/CXR generation, real-target access or model download was
included. Twelve new invented-fixture tests and all **111 V1.2 tests** pass;
the batch script passes `bash -n`. Clinical artifact review remains pending,
and neither a repair policy nor clinical localization has been validated.

### Completed development-only expanded polarity diagnostic

`benchmarks/prepare_expanded_polarity.py` first froze a metadata-only case plan
before opening report text. It inherits the historical 48/16/16 development,
calibration and final-test roles, fixes Sana seed 0 -> CheXagent-2, and reads
only the **48 development reports** semantically. No scores, winner flags or
construction availability choose cases. The other 32 reports are not opened
as text, and there is no new EHR generation or threshold fitting.

The case plan and staged bank are respectively
`artifacts/protected/tricompose_v1_2/benchmarks/report_polarity_dev48_plan_20261002_001/`
and `report_polarity_dev48_20261002_001/` under the same parent. The constructor
tries all eight named findings in every fixed case: **384 attempts**. It accepts
only an isolated explicit finding assertion, changes its polarity in either
direction, preserves unrelated bytes/header text, and retains all rejections.
Qualified absence such as "no large effusion" is not global absence. Missing,
uncertain, mixed or repeated statements are not rewritten into invented facts.

The bank contains **193 report copies**: 48 byte-identical original controls,
48 token-preserving whitespace controls and 97 eligible edits. Eligible targets
are cardiomegaly (17), consolidation (1), pleural effusion (40) and pneumothorax
(39); the other four heads have no constructible edit. All **287 unavailable
attempts** retain reasons. They describe construction coverage, not model errors
or absence of disease. There are only **30 distinct image hashes, 26 original
report hashes and 100 report-copy hashes**; neither 48 case records nor 97 edits
should be called independent clinical examples.

`benchmarks/score_expanded_polarity.py` reuses the frozen CheXbert mapping and
BioViL-T loader, with bounded classifier batches of eight. Scorers see only a
hash-bound allowlisted resolver, not intervention labels. The subsequent cached
analysis reports case-weighted and deduplicated finding/text-pair fractions,
formatting invariance, non-target state changes and raw cosine differences.
There is no new clinical score, calibrated threshold, winner or repair action.
This historical development pool was already used for static evaluation; it is
not independent final-test evidence or clinically adjudicated gold.

The explicitly approved `slurm/16_expanded_polarity_debug_p100.sbatch` completed
as job **12560507**, exit 0, on debug node `e23-02` (P100) in **54 seconds**.
Its request was one GPU, two CPUs, 16 GiB host memory and a ten-minute cap;
the cap was not expected runtime. Outputs are under
`artifacts/protected/tricompose_v1_2/verification_runs/report_polarity_dev48_12560507_{chexbert,biovil,analysis}/`.

The table measures whether CheXbert extracted **both** the original and edited
explicit text assertions, not whether either report correctly describes its
image. Deduplication uses the original/edited text-hash pair within each finding.

| Finding | Eligible case-linked pairs | Both states extracted | Case-linked fraction | Unique text pairs | Successful unique pairs | Deduplicated fraction |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 17 | 16 | 94.1% | 6 | 5 | 83.3% |
| Consolidation | 1 | 1 | 100.0% | 1 | 1 | 100.0% |
| Pleural effusion | 40 | 15 | 37.5% | 21 | 12 | 57.1% |
| Pneumothorax | 39 | 7 | 17.9% | 20 | 6 | 30.0% |

The other four planned findings remain **unavailable**, not failed evaluations.
Overall, 39/97 case-linked pairs and 24/48 unique finding/text pairs met the
two-state extraction check; the direction mix is imbalanced (17 affirmative-to-
negative versus 80 negative-to-affirmative edits). These totals are exploratory
development diagnostics, not independent clinical accuracy estimates. The
single consolidation pair cannot establish general reliability.

All **48 whitespace controls** had identical CheXbert state vectors and zero
BioViL cosine differences at the saved precision. Original assertion extraction
matched in 96/97 eligible pairs. For pleural effusion, 24/40 edited reports still
received a negative label; for pneumothorax, 32/39 did. Changes in non-target
heads occurred in 41/97 pairs, including support-device state changes in 36
pairs. These are extraction/context-sensitivity findings, not evidence of 41
clinical errors or changes to the original images. Neither score adapters nor
the text construction protocol were changed after inspecting these results.

Raw BioViL cosine changes are recorded in `analysis/per_intervention.csv` and
JSON. Their direction is **not** scored as clinical correctness: the original
image/report pair lacks independent clinical adjudication. This diagnostic does
not establish CheXbert as a standalone trigger for automatic localization or
regeneration; extraction failures must be investigated before relying on that
decision rule. Existing winners, thresholds and repair permissions are unchanged.

CheXbert processed 100 distinct report texts in 13 batches (18.565 seconds,
peak PyTorch allocated VRAM including loading 1.227 GiB). BioViL-T processed
30 distinct images and 100 texts (23.453 seconds, peak allocated VRAM including
loading 0.603 GiB). Peak figures are not driver-level/reserved-memory measurements.
All seven output artifact hashes and run file/directory project permissions
passed validation. Nineteen new invented-fixture tests and all **130 V1.2 tests**
pass. Original source artifacts and the completed two-case diagnostic remain
unchanged. No generation, Qwen inference, training, threshold fitting, real-data
access, selection or repair was part of this job.

### Completed read-only polarity audit and context/batch diagnosis

The follow-up audit reconstructed all **97 edits** from the immutable controls;
unrelated bytes were identical. The loaded head order and four-state indices
agree with [Stanford's constants](https://github.com/stanfordmlgroup/CheXbert/blob/master/src/constants.py)
and [CSV class conversion](https://github.com/stanfordmlgroup/CheXbert/blob/master/src/label.py).
Tokenizer-only checking of the 100 distinct synthetic texts found **zero token
ID differences** between the current wrapper preprocessing and the official
[CSV tokenizer path](https://github.com/stanfordmlgroup/CheXbert/blob/master/src/bert_tokenizer.py).
Lengths were 55--141 tokens, not near the 512-token limit. No model initialization
or inference ran on the login node. These checks exclude the observed mapping,
edit-reconstruction and token/truncation explanations, but do not independently
authenticate the checkpoint origin or prove the remaining root cause.

`benchmarks/diagnose_chexbert_polarity.py` is a **diagnostic**, not an
evaluator replacement. It uses all 100 frozen report-copy texts, all 97 eligible
case-linked edits and their 15 distinct original/edited target sentences. It
does not choose only failures, open the reserved 32 cases, or modify any sources.
Under an approved Slurm GPU allocation it will:

- Reproduce the existing wrapper labels and compare its argmax with the
  instrumented frozen-head logit path.
- Compare full-report batch sizes eight and one, and check exact token-ID parity
  with official CSV preprocessing.
- Contrast full-report and isolated-target-sentence states, retaining every
  successful, failed and unknown extraction and both duplicate-aware denominators.
- Store raw logits/margins as diagnostics, not calibrated clinical confidence.

This context ablation intentionally reads the protected construction key to
obtain target sentences, but never passes expected labels, findings, case IDs or
prior scores into the model. It is not a blinded or independent final-test
benchmark. Sentence-only scores must not replace the current report score, and
neither context sensitivity nor batch stability establishes image truth or
clinical error localization. No selection, thresholds or repair rules change.

The explicitly approved `slurm/17_chexbert_context_debug_p100.sbatch` completed
as job **12561407**, exit 0, on debug node `e23-02` (P100) in **19 seconds**.
The request was one GPU, two CPUs, 16 GiB host memory and a five-minute cap,
not a five-minute runtime estimate. Outputs are under
`artifacts/protected/tricompose_v1_2/verification_runs/chexbert_context_dev48_12561407/`.

All 193 cached item labels reproduced exactly; none of the 100 distinct report
texts changed labels between batch sizes eight and one. The instrumented
logit-path argmax matched the existing wrapper, with zero official CSV token-ID
mismatches. Thus the observed misses are reproducible and are not explained by
these checked batching, interface, cache or preprocessing differences.

| Finding | Case-linked pairs | Full-report both-state extraction | Isolated-sentence both-state extraction | Unique report text pairs | Full-report deduplicated successes | Isolated deduplicated successes |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 17 | 16 | 17 | 6 | 5 | 6 |
| Consolidation | 1 | 1 | 1 | 1 | 1 | 1 |
| Pleural effusion | 40 | 15 | 40 | 21 | 12 | 21 |
| Pneumothorax | 39 | 7 | 39 | 20 | 6 | 20 |

Overall full-report versus isolated-sentence extraction was **39/97 versus
97/97** case-linked pairs and **24/48 versus 48/48** unique finding/report-text
pairs. There are only **15 distinct isolated sentences**; the 97 intervention
pairs are not independent clinical examples. Every pair retained its original
denominator and direction. There were 57 edited assertions that matched only
when isolated; one additional pair failed only the full-report original
assertion check. No edited assertion failed in both contexts.

This controlled ablation supports input-context/representation sensitivity as
an important factor: the model can extract these simple assertions in isolation,
but full-report outputs can differ. It does **not** establish which surrounding
sentence, header, position or semantic conflict causes each difference. It also
does not rule out conflicting or synonymous assertions elsewhere in an edited
report, nor prove that the isolated output is clinically correct. Do not turn
the 100% toy-sentence result into a clinical accuracy claim, replace full-report
scores with isolated ones, tune thresholds, or automatically regenerate a CXR.
The next gate is to inspect report-level semantic conflicts and validate
evidence-scoped extraction independently before changing a selection rule.

The program took 13.594 seconds including loading/output work; peak PyTorch
allocated VRAM including loading was 1.227 GiB. Its 315 encoder examples include
the diagnostic/wrapper double pass, batch-size replay and sentence ablations;
they are not 315 unique reports or clinical cases. Output/source hashes,
protected project modes/group and in-memory cached-analysis replay passed.
Twelve new invented-fixture tests bring the lightweight V1.2 suite to **142
passing tests**. Original artifacts, primary scores, winners and repair
permissions remain unchanged; no additional GPU task was submitted.

### Completed evidence-scoped full-report follow-up (secondary diagnostic)

The follow-up uses the existing frozen local Qwen2.5-VL-7B checkpoint; it does
not download RadGraph or a new model. The local dependency audit does not show
an available RadGraph checkpoint. A coarse read-only text-structure check found
remaining negation cues in all 97 edited full reports, but negation about a
different disease does not constitute a same-finding contradiction. Counting
words or repeated exact disease names cannot establish report-level semantics.

`benchmarks/verify_report_evidence_qwen.py` is a secondary, report-only
development diagnostic with exactly four previously testable findings:
cardiomegaly, consolidation, pleural effusion and pneumothorax. It reads the
immutable blinded resolver, deduplicates the 193 report copies to **100 texts**,
and makes one greedy, frozen call per distinct text. No image, EHR, intervention
key, expected state, previous score or winner is passed to the model. The 32
reserved cases remain unopened. This is not an independent final evaluation.

Each positive/negative/uncertain assertion must have a short, exact, contiguous
source quote and unambiguous Unicode-character offsets. Missing assertions stay
unknown; quoted positive and negative assertions are retained as a conflict
signal rather than majority-voted away. Invalid JSON, duplicate keys, invented
quotes, ambiguous offsets, incomplete output or the token limit make the whole
response unavailable, with all four states unknown and the failure denominator
retained. Protected raw responses are saved for parser replay. Each polarity is
capped at two quotes: neither an empty array nor exact quote alignment proves
semantic correctness or exhaustive absence of another assertion.

Only the separate CPU-only `benchmarks/analyze_report_evidence_qwen.py` reads
the construction key after inference. It verifies raw-response hashes and exact
parser replay, reconstructs every edit, and aligns quotes to the known target
span versus other report positions. It compares those evidence patterns with
the existing full-report CheXbert diagnostic while retaining all 97 pairs and
duplicate-aware counts. Expected-polarity quotes in the target span and
opposite-polarity quotes elsewhere are **model-attributed evidence**, not
independently adjudicated clinical labels or an established explanation for
CheXbert's behavior. Absence of a recorded quote does not establish absence of
a conflicting assertion. No current primary score, threshold, selection,
generator or repair rule is changed.

`slurm/18_report_evidence_qwen_flexible_gpu.sbatch` was shown in full with its
resource request and explicitly approved before submission as **job 12562445**
on 2026-10-02: one GPU from V100/A40/A100/L40S, two CPUs, 32 GiB host memory, twenty-minute
wall-time cap. The runtime enforces a conservative 24 GiB visible-VRAM guard
(not a measured minimum requirement) and records
actual GPU/dtype; V100 uses FP16 whereas compatible newer devices can use
BF16, so bitwise cross-device equivalence is not claimed. The existing PyTorch
build supports V100's architecture. The last resource check showed only
drained A40/A100 nodes and free P100s; a compatible request can still queue.
The cap is not a runtime or queue-time promise. The initial scheduler check
reported **PENDING / Priority**, with a provisional 06:31:34 local start time;
at that check, no diagnostic results existed. Full
script/resource disclosure and explicit user approval remain required before
any additional submission. The submitted script SHA256 is
`d7a666711ed798c3250263d3294b8786f8cf1c9e2bde4e29ccf9ed71ec6a08c6`;
all eleven pinned input/code checks and the shell syntax check passed.

After the user explicitly approved an in-place queue switch, the existing
pending job was updated at approximately **03:08 PDT** using
`scontrol update JobId=12562445 Partition=debug Features=a40`. The live
allocation request was changed to **debug / A40**, still one GPU, two CPUs, 32 GiB
host memory and a twenty-minute cap. No job was canceled or resubmitted, and
the submitted script body, model, prompts, code hashes and output IDs are
unchanged. The last observed pre-run scheduler estimate was **03:46:56 PDT**,
with state **PENDING / Resources**. The earlier test-only estimate of 03:31
did not create a second job. Neither estimate was an actual start time:
accounting now confirms execution from **03:22:11 to 03:30:10 PDT** on
`b11-09`, **7m59s**, **COMPLETED / exit 0**. No second submission was needed.

P100 is not a drop-in replacement for the current Qwen runtime: the installed
PyTorch `2.9.1+cu128` build reports compiled architectures beginning at
`sm_70`, without P100's `sm_60`. This was a lightweight build-metadata check
with CUDA uninitialized, not login-node inference. The earlier A40 run's
15.642 GiB allocated peak also excludes driver/reserved-memory overhead and
does not establish safe operation on a 16-GiB P100. A P100 migration would
need a separately validated compatible environment and memory plan; neither
was implemented or submitted during this queue switch.

The completed, non-overwriting protected outputs are:

```text
artifacts/protected/tricompose_v1_2/verification_runs/
  qwen_report_evidence_12562445/
    evidence.json, raw_responses.json, summary.json, manifest.json
  qwen_report_evidence_12562445_analysis/
    pair_alignment.json, summary.json, summary.md, manifest.json
```

Twenty-three additional invented-fixture evidence/alignment tests bring the
lightweight V1.2 suite to **165 passing tests**. This only validates contracts
and bookkeeping; it does not establish semantic accuracy or a working repair
method. No model inference ran on the login node. Submission alone does not
establish successful execution, semantic accuracy or completed results.

The verifier made **100 frozen, greedy text-only calls** using A40 BF16.
Its recorded program runtime was 473.942 seconds including fingerprinting,
loading and artifact writing, with peak PyTorch allocated VRAM including
loading **15.597 GiB**. This is not total driver/reserved-memory use. The
separate alignment program made zero model calls. Six manifest-listed artifacts
passed SHA256 checks; both completed output trees passed project-group modes
`2770`/`0660`. Raw-response parser replay and the entire cached analysis replay
were identical, and the 165 lightweight tests still passed.

**Contract result:** 69/100 distinct texts had complete, exact-source evidence;
31/100 were unavailable because at least one returned quote was not an exact
substring. All failed records remain present with four unknown states, not
negative states. By construction arm, failures were 3/26 distinct unchanged
texts, **26/26 whitespace-only texts**, and 2/48 distinct minimally edited
texts. Thus all 48 case-linked whitespace controls had an unavailable member;
their zero recorded state changes do **not** establish formatting invariance.

A subsequent bounded, read-only cached audit found 118 nonmatching quotes:
105 matched after whitespace folding. In **23/31 failed texts**, every
nonmatching quote was whitespace-equivalent to a source substring; the other
eight texts had at least one mismatch not explained by this check. These
counts diagnose alignment brittleness, not validated normalization, semantics
or paraphrase acceptance. No failed record was silently repaired or promoted.

| Finding | Case-linked edit pairs | Qwen target-polarity quotes captured in both reports | Earlier full-report CheXbert both text states |
|---|---:|---:|---:|
| Cardiomegaly | 17 | 16 | 16 |
| Consolidation | 1 | 0 | 1 |
| Pleural effusion | 40 | 12 | 15 |
| Pneumothorax | 39 | 12 | 7 |

Qwen captured both target quotes in **40/97** case-linked pairs and **23/48**
unique finding/report-text pairs. Three case-linked pairs had an unavailable
report, and 54 complete pairs did not capture the expected target polarity in
both reports. These are text-assertion diagnostics with repeated templates,
not independently adjudicated clinical accuracy or a like-for-like clinical
comparison with CheXbert. There is no clear overall extraction gain: the
earlier CheXbert counts were 39/97 and 24/48, respectively.

**Traceability is not polarity correctness:** the minimally injected positive
effusion and pneumothorax assertions were captured in 40/40 and 39/39 edited
reports. However, in the corresponding source-negative checks, Qwen placed a
quote inside the mechanically negative target span in the **positive** array
for 27 case-linked effusion checks and 27 pneumothorax checks. Each such quote
still contained a negation marker. These are not 54 independent patients or
confirmed clinical errors: each finding's cases share 11 distinct source
reports, with only three and two distinct negative target sentences,
respectively. The parser faithfully retained the model's assigned polarity;
verbatim substring validation does not validate its semantic interpretation.
Consequently the lack of recorded opposite assertions elsewhere cannot clear
the report-conflict gate or establish the cause of CheXbert's behavior.

**Decision:** retain Qwen as secondary, unvalidated evidence and keep all old
scores, thresholds, selections and repair permissions unchanged. The next
small step is a separately versioned, cache-only quote-alignment diagnostic
with explicit original-character mapping, plus checks of negation scope and
multi-finding sentences. Preserve this strict baseline; do not accept arbitrary
paraphrases, replace primary scores, or regenerate an EHR/CXR/report because
of a verifier's text-extraction failure. No additional GPU job, training,
download or external API call was performed during the cached follow-up.

### Completed cache-only evidence interface repair (secondary, not new scores)

`benchmarks/repair_cached_report_evidence.py` implements the approved small
follow-up using the existing 100 frozen Qwen responses. It imports no model
runtime, makes **zero model calls**, and changes no report, CXR, EHR, primary
score, threshold or selected triple. The repair pass sees only the immutable
synthetic source text and cached response; it completes before the separate
post-hoc analysis reads the construction key. Reserved cases remain unopened.

Alignment collapses Unicode whitespace **only to locate** a unique source
substring. Stored evidence is restored to the original, byte-unchanged report
with Unicode-character offsets and both original/returned quote hashes.
Case, punctuation, numbers, units, qualifiers and negation are not normalized;
paraphrases, repeated locations, duplicate normalized spans, malformed JSON,
partial inventories and token-limit failures are rejected. Whole-record
failures retain four unknown states and all denominators.

The separately versioned four-finding literal scope guard can only **veto**
model-attributed evidence. It never moves a quote into another polarity or
manufactures a new negative. Qualified absence and explicit uncertainty cannot
become global absence. Source punctuation, contrast and explicit independent
clauses bound negation; original line structure distinguishes a new presence
clause from a wrapped negation/list. A remaining veto cannot resolve a conflict
into a determinate opposite state. Uncovered synonyms/grammar are explicitly
unchecked, and retained quotes are still not clinically verified.

Scope and pseudo-trigger considerations have precedents in
[NegEx](https://pubmed.ncbi.nlm.nih.gov/12123149/) and
[ConText](https://pubmed.ncbi.nlm.nih.gov/19435614/).
This narrow hand-written veto is **not** either official implementation and
is not the dependency-based [NegBio](https://arxiv.org/abs/1712.05898).
It does not reuse the benchmark constructor's vocabulary/answer key as a
scorer and does not inherit any published clinical performance claims.

The inspected completed diagnostic is:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  qwen_cached_evidence_repair_12562445_002/
    summary.json, summary.md, repaired_evidence.json,
    pair_alignment.json, manifest.json
```

The earlier `_001` cache diagnostic is retained, not overwritten. Its first
scope version over-abstained when whitespace folding erased separate original
lines; `_002` uses `literal-four-finding-veto-v2-original-line-scope` with new
invented regression fixtures. Neither run was used to select or regenerate
data. Refining a guard on this development diagnostic is **not** held-out
validation; independent false-veto/clinical extraction evaluation is pending.

| Stage | Complete source-quote contracts / 100 | Unavailable / 100 | Non-unknown finding states / 400 | Target quotes in both / 97 | Unique target pairs / 48 |
|---|---:|---:|---:|---:|---:|
| Original strict baseline | 69 | 31 | 273 | 40 | 23 |
| Unique whitespace alignment | 91 | 9 | 360 | 40 | 23 |
| Alignment + conservative scope veto | 91 | 9 | 230 | 40 | 23 |

Twenty-two failed contracts were recovered and no previously complete contract
was lost. Eight remaining responses contain non-whitespace mismatches; one
contains duplicate/conflicting normalized source spans. Accepted aligned
evidence includes 275 exact and 89 whitespace-equivalent quotes. The current
guard vetoes 130 quotes: 52 explicit-positive/negation mismatches and 78
uncertainty/qualified-absence mismatches. Its 130 state transitions are all to
unknown, with **zero unknown promotions or polarity flips**. Coverage loss is
visible rather than counted as success. Quote-contract recovery is **not 91%
clinical accuracy**, and target-quote extraction has not improved.

A post-hoc audit confirms that the earlier 27 effusion and 27 pneumothorax
case-linked negative-target quotes assigned positive were all vetoed. These
54 checks involve only 22 unique finding/source-text pairs and repeated
mechanical assertions, not 54 independent clinical mistakes or a validated
false-repair rate.

Whitespace controls are now comparable in **44/48** case-linked checks, with
four unavailable. **23/44** comparable pairs still have different state
vectors in both the alignment-only and guarded stages. This exposes sensitivity
in the cached model outputs; fixing quote provenance cannot fix those frozen
responses or prove that generated images/reports are clinically wrong.
Qwen remains secondary diagnostic evidence, unsuitable as the sole selector
or regeneration trigger. Input whitespace canonicalization with offset mapping
is a possible next controlled model diagnostic, not an implemented cure; any
new inference requires complete Slurm disclosure and explicit approval.

Thirty-nine added invented-fixture tests bring the lightweight V1.2 suite to
**204 passing tests**. They include original-line scope, cropped quotes,
negated lists, independent clauses, qualifiers, ambiguous/duplicate locations,
key-read ordering and non-overwrite refusal. Artifact/program/source hashes,
exact cached replay, protected modes `2770`/`0660` and project-group boundary
all passed. Historical verifier/parser/script and primary-score/winner hashes
remain unchanged. No real patient data, GPU, training, download or external
API was used in this cached repair.

### Completed line-preserving input/cache interface (engineering, not a new clinical benchmark)

`benchmarks/canonical_report_evidence_interface.py` follows the cached repair
with an explicit request-normalization contract. It collapses horizontal
spaces/tabs only, preserves original line breaks, punctuation, case, numbers,
negation and ordering, and never strips or flattens the report. Every input
retains a canonical-to-original Unicode offset map. Quotes are projected to
unchanged source text and the inverse coordinates are checked before the
unchanged conservative scope guard runs.

A read-only feasibility audit found that all-whitespace folding would produce
74 inputs but **zero exact old-input cache hits**, while also deleting original
line boundaries. That is not the chosen interface. Line-preserving horizontal
normalization instead produces **74 exact old-input cache hits** from the 100
source texts: 26 groups of two formatting variants and 48 singleton edited
texts. Every input was already processed in the completed frozen Qwen run, so
this interface experiment required **zero new model calls, no GPU allocation
and no Slurm submission**.

Cache identity binds the completed run's frozen checkpoint/asset fingerprints,
producer revision, fixed request prompt, sampling settings, execution metadata,
exact canonical text and request hashes. A missing/nonmatching canonical input
refuses execution rather than approximately reusing a response or invoking a
model on the login node. A future cache miss needs a separately disclosed and
approved inference job. No score, expected finding, construction key or winner
is supplied to the normalization/projection pass; the key is read only in the
subsequent diagnostic evaluation. All original-format cached responses remain
available as the measured instability baseline.

The completed, replay-checked output is:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  canonical_report_input_cache_12562445_002/
    canonical_inputs.json, source_mappings.json, evidence.json,
    pair_alignment.json, summary.json, summary.md, manifest.json
```

The earlier `_001` output is preserved. `_002` makes group-size metadata keys
explicit JSON strings so both in-memory and persisted summary replay agree;
this bookkeeping correction changes no evidence or clinical conclusion.

| Interface stage | Complete original-source projections / 100 | Complete distinct canonical inputs / 74 | Target quotes in both / 97 | Comparable formatting controls / 48 | Formatting state differences |
|---|---:|---:|---:|---:|---:|
| Exact canonical-input cache + quote projection | 92 | 69 | 40 | 45 | 0 (shared input/cache) |
| Same interface + unchanged scope veto | 92 | 69 | 40 | 45 | 0 (shared input/cache) |

Five distinct canonical inputs still have unavailable evidence contracts; their
eight original-text projections remain present and unknown. Three case-linked
formatting controls are unavailable. **The zero differences are by construction:**
identical canonical inputs share the same old response. They do not independently
measure repeated-inference stability, model invariance, clinical accuracy or
an improvement over the previously observed 23/44 different state vectors.
The mechanical target-quote checks remain 40/97 and 23/48 unique finding/text
pairs. There is no clinical extraction gain, better selected triple or approved
automatic-regeneration trigger. This step establishes reproducible input/cache
plumbing only; independent clinical/scope validation remains necessary.

Twenty-three additional invented-fixture tests bring the lightweight V1.2
suite to **227 passing tests**, covering line preservation, Unicode inverse
maps, exact cache binding, producer pinning, cache-miss refusal, unknown-safe
failures, key-read ordering, JSON-stable metadata and non-overwrite refusal.
All six output artifacts, program/source fingerprints, complete in-memory
replay and protected project modes/group passed. Original verifier/parser,
cached repair, primary score table and selected-triple hashes remain unchanged.
No reserved cases, real patient inputs, external API, download or training were
used. The interface is not wired into the main selector or generation pipeline.

### Completed authored assertion challenge, conditional veto audit and frozen CheXbert diagnostic

`benchmarks/report_assertion_challenge.py` fixes **56 wholly invented short
reports**, independently of the earlier mechanical polarity-edit constructor.
No patients, real targets, previous generated reports, or reserved cases are
used. The four checked heads are cardiomegaly, consolidation, pleural effusion
and pneumothorax. There are **80 designated finding checks** (20 per head),
with 15 authored positives, 22 negatives, 20 uncertain states and 23 unknowns;
the complete 224-entry vectors are secondary diagnostics. Template families
are correlated and **the same project author assigned the expected states**:
this is a language challenge, not independent radiologist annotation or
held-out clinical validation. In particular, qualified absence is conservatively
marked uncertain at the broader finding level, unmentioned/unassessed findings
stay unknown, and opposed assertions stay uncertain. These are disclosed
annotation choices, not universal CheXbert gold labels.

The immutable protected runs are:

```text
artifacts/protected/tricompose_v1_2/benchmarks/
  authored_assertions_20261002_001/
    reports/, resolver.jsonl, references.jsonl, summary.json, manifest.json
artifacts/protected/tricompose_v1_2/diagnostics/
  authored_assertion_veto_20261002_001/
    details.json, summary.json, manifest.json
```

The CPU-only audit freezes the **pre-challenge** literal scope guard at SHA256
`7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18`.
For each designated check it supplies the entire invented report as an oracle
quote and probes positive, negative and uncertain attribution: 240 probes in
total. Full quotes may contain multiple assertions. It tests only conditional
veto behavior, **not quote extraction, actual model error, clinical accuracy
or a calibrated false-repair rate**. The rule code is not changed to fit results.

| Conditional rule probe | Count |
|---|---:|
| Exact authored polarity supplied | 57 |
| Exact-polarity probes retained | 56 |
| Incorrect determinate polarity deliberately supplied | 123 |
| Incorrect determinate probes vetoed | 65 |
| Incorrect determinate probes not vetoed | 58 |
| All probes without a checked literal mention | 42 |

This broader diagnostic exposes incomplete scope coverage: the existing veto
must not be promoted into an automatic regeneration trigger. No score,
threshold, candidate or selected triple is replaced. **中文结论：当前规则在
人为构造的否定/不确定性压力测试中仍有漏拦，尚不能安全地决定自动返工；
以上数字不是模型真实错误率，也不是临床验证。**

`benchmarks/score_report_assertion_challenge.py` prepares a separate **frozen
CheXbert** measurement using the existing checkpoint and unmodified CXRMate
wrapper. It loads only invented text, hashes and opaque IDs, never the authored
reference key. It preserves the audited internal mapping (0 unknown, 1 positive,
2 negative, 3 uncertain), which differs from the CSV export convention documented
by the [official CheXbert repository](https://github.com/stanfordmlgroup/CheXbert).
Batch-8 inference is replayed at batch 1 (112 encoder examples, 63 forward
batches); inputs are checked to prevent silent 512-token truncation. Predictions
are committed before separate CPU analysis reads the reference key. Analysis
reports the four-state confusion matrix, per-state F1, macro F1, per-family
and per-head results, hard positive/negative flips and unsafe commitments on
unknown/uncertain states. Unavailable predictions remain in the denominator
and never earn unknown-state credit.

The complete `slurm/19_authored_assertions_chexbert_debug_p100.sbatch` script
and resource request were shown, then explicitly approved. Job **12580901**
completed with exit code 0 on `e23-02`: one debug P100, two CPUs, 16 GiB host
memory and a five-minute wall-time limit. It briefly queued for priority, then
ran for **13 seconds**; the prediction program recorded **10.914 seconds** and
**1.227 GiB peak allocated VRAM including loading**. The approved script SHA256
is `30c595fb9391c34dfa347627c1469e7abf62f84e4b37a052e77804bbcf4a91e7`.
All 56 prediction records completed, with no truncation (6–33 tokens), and
batch 8 versus batch 1 produced **zero changed four-head state vectors**.

The immutable protected outputs are:

```text
artifacts/protected/tricompose_v1_2/verification_runs/
  authored_assertions_chexbert_12580901/
    predictions.json, manifest.json
  authored_assertions_chexbert_12580901_analysis/
    summary.json, details.json, manifest.json
```

| Authored expected state | Designated checks | Exact matches | State F1 |
|---|---:|---:|---:|
| Positive | 15 | 15 | 0.6522 |
| Negative | 22 | 11 | 0.5946 |
| Uncertain | 20 | 8 | 0.4000 |
| Unknown | 23 | 14 | 0.7568 |

Across the **80 designated checks**, 48 matched the predeclared authored state
(60.0%) and four-state macro F1 was 0.6009. There were seven positive/negative
flips (all authored negative → model positive) and 13 determinate commitments
on authored uncertain/unknown states. Literal positive controls and the
no-evidence family each matched 4/4, while postposed absence matched 0/4
(all four became positive); cannot-be-excluded matched 0/4. These small,
correlated language families are not independent clinical samples. Qualified
absence and conflicting-assertion results also depend on the disclosed author
annotation policy. The secondary full-vector result, 188/224 = 83.93%, includes
**159 authored unknown entries** and must not replace the harder designated
checks or be presented as patient-level clinical accuracy.

**中文结论：冻结 CheXbert 能提取简单阳性和部分常见否定，但在这组虚构语言
挑战中仍存在否定及不确定性错误；batch 大小和输入截断不能解释这些错误。
本轮是在检查报告标签提取器，不是在证明生成图像或报告的临床质量。
现有规则与 CheXbert 标签都不足以单独授权自动定位错误模态和返工。**

The 250 lightweight synthetic-fixture tests, source/artifact fingerprints,
protected project-group permissions, deterministic CPU audit replay and
complete post-hoc analysis replay passed. All predictions were committed before
reference-key analysis; the scorer did not read the key or receive expected
states. Existing primary score table, edge details and selected-triple hashes
remain unchanged. No checkpoint, guard, threshold or annotation was changed to
fit these results; no regeneration trigger was enabled. Training, downloads,
external API calls and patient-data access were not performed.

### Completed source-bound scope gate, official ConText and source comparison

The reusable interface now lives in
`src/tricompose_v12/report_assertions.py`, rather than hiding clinical state
decisions inside another model adapter. Each finding retains the original
proposal, gated state, source report SHA256, exact Unicode offsets/quote hash,
evidence ID, scope-check reasons, coverage, and a review action. **Scope verified
means only the limited literal rules covered the proposal; it is not independent
clinical verification or evidence that the report describes its CXR.**

`benchmarks/gate_report_assertion_predictions.py` applies this gate blindly to
the completed frozen CheXbert predictions before separate authored-key analysis.
The underlying pre-challenge scope guard is unchanged. A proposal can be
retained only when every matched literal scope is covered and agrees with its
existing state. A non-veto is no longer sufficient: uncovered vocabulary or
scope now abstains. The gate never changes positive to negative, promotes an
unknown proposal, creates a finding, or authorizes regeneration. Its outputs
are `scope_commit`, `abstain`, and `no_model_assertion`; the latter two are not
counted as successful decisions, and unknown is never treated as negative.

The completed protected runs are:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  report_scope_gate_12580901_001/
    evidence.json, decision_table.csv, manifest.json
  report_scope_gate_analysis_12580901_001/
    summary.json, details.json, manifest.json
```

The decision table has all **224 report/finding rows**, not just successful
rows. Risk/coverage analysis keeps the full **80 designated-check denominator**:

| Development diagnostic | Count |
|---|---:|
| Raw CheXbert exact authored-state matches | 48/80 |
| Raw non-unknown proposals | 66/80 |
| Scope commits | 30/80 (37.5% coverage) |
| Exact authored-state matches among commits | 30/30 |
| Abstained non-unknown proposals | 36/80 |
| No model assertion | 14/80 |
| Raw errors not committed | 32/32 |
| Raw matching states not committed | 18 |

Among all 224 rows there are 34 scope commits, 40 abstentions and 150 missing
model assertions. **This gate was designed after the authored challenge results
were observed.** Its 30/30 conditional match is a post-hoc development result,
not 100% overall accuracy, independent validation, clinical specificity, or a
false-repair estimate. It explicitly sacrifices coverage and also drops 18
matching states (14 unknowns and four non-unknown assertions). It cannot replace
the original score table or establish an improvement in generated triples.

**中文进展：报告事实接口已经输出完整的原状态、证据、规则覆盖和弃权表；
不是把分数强行提高。当前门控保留 30/80 项，但没有对其余 50 项作出明确
判断。下一步需要独立对照与未用于开发的临床标注，而非立即触发返工。**

No NegBio/ConText package was found in the inspected existing environments.
`benchmarks/score_report_assertions_context.py` and
`slurm/20_report_context_cpu.sbatch` implement an additional **official medspaCy
ConText** comparison, following its [official API](https://github.com/medspacy/medspacy/tree/1.3.1).
The job uses a new isolated environment, blank English tokenization, official
PyRuSH sentence rules and unmodified official English ConText rules; no learned
NLP model or CXR checkpoint is loaded. The existing limited four-finding literal
inventory is shared and disclosed, so this is a separate context algorithm,
**not independent clinical votes or a full clinical synonym extractor**.
ConText cue/scope spans and historical/hypothetical/family/negated/uncertain
flags retain source hashes and character offsets. An unmodified mention's
conventional positive baseline is explicitly marked unverified.

The script installs pinned medspaCy 1.3.1, spaCy 3.7.5, PyRuSH 1.0.12
and NumPy 1.26.4 into a new per-job environment. The official 244,637-byte
medspaCy source distribution is SHA256-pinned; installed transitive versions,
rule/code hashes and installation metadata are retained. No original model
environment is changed. The job compares ConText with the authored references
only after blind predictions are committed, then measures a second-parser
veto on the original CheXbert proposals. Parser disagreement can only reduce
coverage, not flip a state or promote an unsupported finding. There is no
fitted threshold, custom trigger, training, new router or original-score change.

The complete standalone script and request were shown: **main, two CPUs,
8 GiB host memory, 15-minute cap including first installation, no GPU**. Instead
of submitting another job, the exact script was executed with `bash` inside
the current session's existing CPU Slurm allocation **12576792**, on `b05-04`.
Its allocation and cgroup were verified before execution: four CPUs and
32 GiB allocated, worker threads limited to two and command timeout 15 minutes.
`#SBATCH` directives do not allocate resources when a script is run with `bash`.
**No new `sbatch`, GPU request or standalone benchmark job was submitted.**
The command completed with exit code zero; the private execution receipt is
`task_runtime/report_context_12576792/execution_receipt.json` below the protected
V1.2 root. The earlier separate-submission approval request is no longer needed
for this completed run; it must not cause a duplicate submission.

Official ConText processed the 56 invented reports twice in **2.363 seconds**
(parser-program elapsed time, excluding dependency installation); no replay
state/evidence changed. It loaded 102 unchanged official rules and no trained
NLP component. None of those default rules restrict `allowed_types` or
`excluded_types`, so the finding-label entity names were not excluded by such
restrictions. This does not establish that the default rule set covers all
clinical assertions or that another ConText setup would have the same result.

Completed official-parser runs below `artifacts/protected/tricompose_v1_2/`:

```text
verification_runs/
  authored_assertions_context_12576792/
  authored_assertions_context_12576792_analysis/
  authored_assertions_context_12576792_gate/
  authored_assertions_context_12576792_gate_analysis/
diagnostics/
  report_extraction_sources_12576792_001/
    comparison.json, source_table.csv, manifest.json
  report_extraction_sources_analysis_12576792_001/
    summary.json, details.json, manifest.json
```

`benchmarks/compare_report_assertion_sources.py` adds a **separate literal
readout**, using the same unchanged scope checker frozen before this challenge.
It proposes its own source state only when all literal scopes are covered;
otherwise it returns unknown. It does not change the CheXbert/ConText labels,
invent a new trigger, or use authored answers during extraction. The comparison
wrapper and this development analysis were added after observing the challenge,
so these are not held-out results even though the underlying rules are older.

| Source extraction diagnostic, 80 designated checks | Authored-state matches | Four-state macro F1 | Positive/negative flips | Determinate output on unknown/uncertain reference |
|---|---:|---:|---:|---:|
| Frozen CheXbert | 48/80 | 0.6009 | 7 | 13 |
| Official default medspaCy ConText | 42/80 | 0.5224 | 14 | 24 |
| Frozen limited literal readout, separate proposal | 69/80 | 0.8680 | 0 | 0 |

The literal readout covers only **46/80** checks. Its remaining 34 outputs are
unknown: 23 match authored unknown states, and 11 miss non-unknown states.
Thus 69/80 is not broad clinical coverage or independent clinical accuracy.
Among the 46 covered checks, 16 literal proposals differ from CheXbert. There
are 25/80 CheXbert/ConText disagreements; ConText corrects three CheXbert
authored-state mismatches but disagrees with nine correct CheXbert states.
These are **extraction disagreements, not established report or CXR errors**.
Every comparison row explicitly keeps `report_content_error_established`,
`image_error_established` and `regeneration_authorized` false.

As a second-parser veto, ConText reduces the CheXbert scope gate from
**30/80 retained (37.5%)** to **21/80 (26.25%)**. Both have zero incorrect
retained states on this authored development set; the additional veto drops
nine more matching states and shows no added benefit here. It is not promoted
into primary selection. Agreement over the same text remains correlated
evidence, not a majority of independent clinical observers.

**中文结论：这一轮已实际跑完，不是在等待新作业。官方 ConText 在当前
语言挑战上没有改善结果；旧字面规则能解释部分提取分歧，但覆盖有限。
现在有完整证据表与弃权接口，下一步应做独立标注验证，不能把解析器
分歧直接变成生成报告／胸片返工。**

The reusable interface, official serializer contracts, absence/uncertainty
handling, source binding, non-overwrite, blind-key ordering, separate literal
readout and abstention denominators are covered by **298 passing lightweight
tests**. All six new runs' 11 artifacts, producer/source hashes, official parser
replay, analyses and exact CSV bytes were rechecked. Protected modes/groups
pass; original primary selection hashes remain unchanged. No real patient
inputs or generated-cohort reports were opened for this diagnostic.

## Metadata-only blinded review of generated reports

The report assertion validation follow-up is implemented in
`src/tricompose_v12/report_review.py` and
`benchmarks/prepare_blinded_report_review.py`, with its return/evidence audit in
`benchmarks/audit_blinded_report_review.py`. It does not use another model to
invent independent human gold. The unchanged two-case pilot's **48 reports
(12 images × four experts)** are assigned opaque review IDs, with no winner,
score, model, case ID or source path in the reviewer files. All four current
scope findings remain represented: **192 annotations per human reader**.
Identical report texts would share one review item without dropping candidate
lineages; this pilot has 48 distinct text hashes.

Protected output:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_20261002_001/
    reviewer/       # blind items, two blank templates and instructions
    investigator/   # source resolver, frozen caches and method/code hashes
    summary.json, manifest.json
  report_blind48_readiness_20261002_001/
    summary.json, manifest.json
```

The metadata preparation and readiness audit completed with **zero model
calls, zero opened report texts/pixels/EHR records and zero gold annotations**.
Actual human annotation progress is 0/192 jointly reviewed rows. Pending is
null, not a reviewed unknown or negative. Extractor comparison and reader
kappa therefore stay unavailable. **328 lightweight tests** pass; the two runs'
nine artifacts and exact metadata replay were checked with a guard that
refuses report-text/pixel opens. Source and original winner hashes are unchanged.

Once qualified separate humans return annotations, evidence checking requires
verified source text in Slurm. Reviewed assertions need exact original Unicode
quotes/offsets/hashes. Unknown, unassessable, incomplete and conflicting reviews
are distinct; their coverage denominators are preserved. The audit can compare
unchanged CheXbert/Qwen and the frozen limited literal readout against matching
reader states and report the separate gate's risk/coverage. **Those matching
states are provisional, not automatically adjudicated clinical gold.** No
parser resolves reader conflicts or authorizes a regeneration action.

This is a previously inspected two-EHR **development review**, not independent
final-test data or 48 independent patients. Report-only annotation can validate
what the text asserts; it cannot establish whether a CXR is correct. Authorized
members retain access to investigator files, and report style may expose model
identity, so model blinding is procedural, not an access-control guarantee.

The prepared `slurm/21_blinded_report_copies_existing_cpu.sh` was shown in full.
After explicit user approval with `go`, the unchanged script **completed with
exit zero** in the existing CPU allocation `12576792` on `b05-04`. The allocation
has four CPUs and 32 GiB; copying was limited to one worker thread and a
120-second timeout. Command wall time was approximately **0.53 seconds**.
No new `sbatch`, GPU request, inference or generation was performed.

The guarded `benchmarks/materialize_blinded_report_review.py` wrote a separate
immutable reviewer-readable run:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_text_12576792/
    reports/report_0000.txt ... reports/report_0047.txt
    items.json, reviewer_a_template.json, reviewer_b_template.json
    INSTRUCTIONS.md, summary.json, manifest.json
  report_blind48_text_readiness_12576792_001/
    summary.json, manifest.json
```

All **48/48** synthetic report copies match their original hashes, and all
53 manifest-listed artifacts passed checks. Protected modes/groups are correct.
Source EHR/images/real reports, raw patient inputs, external APIs, new checkpoints
and training remain out of scope. No synthetic text was placed in this README,
public logs or Git. The original metadata packet is unchanged, including its
historical unmaterialized status. The copied-template audit retains 192 pending
rows per reader, zero completed human labels and unavailable human-reference
metrics. Original score and winner hashes remain unchanged.
The flag `--allow-synthetic-report-copy` remains only a program guard; user
approval, not the flag or a Slurm job ID alone, authorized this completed copy.

See [report-fact interface handoff](../docs/report_assertion_status_20261002.md)
for the reviewed methods, annotation boundary and completed/pending status.

### Easier human review: protected reading book and CSV returns

The approved synthetic copies now have a deterministic reading book and two
blank CSV forms, under:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_sheets_20261002_001/
    reports_for_review.md
    reader_a.csv, reader_b.csv
    reader_a_identity.json, reader_b_identity.json
    HOW_TO_REVIEW.md, summary.json, manifest.json
  report_blind48_sheet_roundtrip_a_20261002_001/
  report_blind48_sheet_roundtrip_b_20261002_001/
  report_blind48_sheet_readiness_20261002_001/
  report_blind48_sheets_verification_20261002_001/
```

`benchmarks/report_review_sheets.py --mode export|import` shares the unchanged
annotation contract. The importer preserves human status/state/reason and
only computes exact Unicode evidence offsets/hashes. Ambiguous repeated
quotes require a human-selected zero-based occurrence; paraphrases and changed
source hashes are rejected. CSV evidence retains original CRLF characters.
Missing review is not reviewed unknown, and neither means negative.

The book contains **48 reports from two EHR cases**, not 48 patients. Two blank
192-row imports and the readiness audit completed in the existing CPU Slurm
allocation, with **0 completed human labels**. The book replay, source bindings,
12 manifest-listed artifacts, protected permissions and old winner hashes
passed checks; overwriting the existing export was refused. **356 tests pass**,
using invented fixtures for the human-filled cases. A final CRLF-only importer
fix followed the blank imports; exact current-core replay matches those pending
artifacts, and the current CLI newline behavior is tested on invented text.

No new Slurm submission, model/API call, independent clinical gold, selection
update or regeneration occurred. Human-reference accuracy/kappa remain
unavailable. Separate humans must copy the forms to new protected return
directories before filling them independently; do not edit immutable runs.
The [handoff](../docs/report_assertion_status_20261002.md#csv-review-workflow--csv-审核流程)
contains copy/import/audit commands and the scope limits.

## Candidate-level report-scope availability (2026-10-02)

`benchmarks/build_report_scope_table.py` now propagates the **unchanged frozen
four-finding report syntax gate** into a separate diagnostic table for the
existing 48 candidates. It reads only approved copied synthetic reports and
hash-bound cached metadata. No source EHR, image pixels, real targets, human
annotation returns, model/API, new checkpoint, primary score or selector is
used. Raw CheXbert states and all eight cached findings remain intact.

The immutable output is:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  report_candidate_scope_12576792_001/
    fact_scope_table.jsonl
    candidate_scope_table.csv
    candidate_scope_table.json
    report_scope_assertions.json
    cross_path_groups.json
    summary.json, summary.md, manifest.json
  report_candidate_scope_verified_12576792_001/
    summary.json, manifest.json
```

| Edge | Inventory denominator | Raw comparable facts | Scope-usable comparable facts |
|---|---:|---:|---:|
| EHR–CXR | 96 unique image/finding pairs | 12 | 12 (report gate does not affect this edge) |
| EHR–Report | 384 candidate/finding pairs | 11 | 0 |
| CXR–Report | 384 candidate/finding pairs | 146 | 34 |

The 384 rows mean **48 candidates × eight findings, from two fixed EHRs**.
Image-side counts deduplicate by CXR byte hash, not report count or candidate
alias. Four scope-supported findings give 192 report assertions: 34 syntax
commits, 68 abstentions and 90 absent model assertions. The other 192 rows
explicitly remain `outside_scope_inventory`; they are not negative findings.

Both fixed EHRs' comparable direct finding is pneumonia, which the frozen
four-finding syntax gate does not cover. Therefore EHR–Report has **no
scope-usable comparison**, not a score of zero or a proven generation failure.
Unsupported findings must not be silently dropped from the denominator or
assigned an invented checking head. Raw EHR–CXR signals remain unvalidated
clinical evidence despite their unchanged availability.

CXR–Report raw/scoped support signals are 93/21 and opposition signals 53/13.
Withdrawing 112 comparable facts is **lost evidence coverage, not better
clinical consistency or repaired reports**. Conditional support actually moves
from 93/146 to 21/34; those fractions are diagnostic arithmetic, not accuracy.
The 12 raw positive/negative report-disagreement groups become zero under the
limited scope mask, which likewise does not mean cross-path errors disappeared.
Reports share an image, so no independent votes or fault attribution result.

The gate never flips a polarity, fills an unknown, changes the EHR or selects a
new winner. Candidate-level clinical selection scores stay null and automatic
repair eligibility stays false. The run and seven artifacts replay exactly;
source/code hashes, protected permissions and original winner hashes passed.
**379 lightweight tests pass.** Execution used the existing CPU Slurm allocation
`12576792`; no new submission or GPU inference occurred. An existing-run
overwrite was refused.

The next scientific requirement is still independent human evidence and
proper coverage of the intended finding inventory before clinical selection or
targeted regeneration. The existing four-finding human-review packet remains
unchanged and has zero returned labels; it cannot by itself validate the
uncovered pneumonia EHR–Report edge. Do not tune the frozen guard to make this
development table look better.

## Real-data scorer validation pilots

`real_validation/biovil_matched_pairs.py` and
`slurm/01_real_biovil_valtest128_p100.sbatch` prepare the first real-pair
scorer check. The Slurm-only program reads the existing read-only
`three_modalities/v2_labs_vitals` linkage manifest internally, verifies
patient-disjoint splits, chooses one frontal study for each of 128 validation
and 128 test patients, and compares frozen BioViL-T cosine for the matched
report versus a different patient's report. No raw image, report, subject ID,
study ID, source path, or report excerpt enters stdout or Git. Individual
scores, opaque row indices and hashes are written only below
`artifacts/protected/tricompose_v1_2/real_validation/`.

The pilot reports AUROC, average precision and within-case matched-score win
rate separately for validation and untouched test. Cross-patient reports are
*mechanical* mismatches and may remain clinically compatible; this tests basic
pair retrieval, not fine-grained clinical correctness or a selection threshold.
BioViL-T may share MIMIC-CXR training data and is not an independent final
evaluator. Hard negatives with clinical adjudication, XRV/CheXbert calibration,
and image/report review remain required. EHR validation is deferred until the
dataset's admission-level diagnosis/procedure features are proven to predate
the image; current source documentation explicitly warns of temporal leakage.

The first pilot completed as Slurm job `12492909` in 68 seconds. Its protected
aggregate results were val/test AUROC 0.934/0.940 for matched reports versus
random cross-patient reports. These are retrieval results, not clinical
contradiction validation. The complete script and resource request for **each
new** job must be shown to the user for explicit approval before `sbatch`.

`real_validation/hard_negative_biovil.py` reuses the fixed 128+128 real cohort and
previous scores. Within each split, it uses official report-derived CheXpert
labels to choose a different-patient, same-view report that shares at least
one positive finding yet has an explicit opposite finding. It scores those
reports with the same frozen BioViL-T checkpoint, then compares matched-vs-
hard and matched-vs-random on the **same available subset**. Unknown and
uncertain labels are not negatives. Cases without a qualifying donor remain
unavailable and are counted. The source labels are not independent image truth
and the challenge remains unadjudicated. The approved batch script
`slurm/02_real_biovil_hardneg_p100.sbatch` completed as job `12508001` in
60 seconds. The protected result is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_hardneg_valtest128_20260930_001/`.

| Split | Qualifying pairs | Unavailable | Matched vs hard AUROC | Matched vs random AUROC on same subset | Matched > hard |
|---|---:|---:|---:|---:|---:|
| Validation | 78 | 50 | 0.931 | 0.920 | 96.2% |
| Test | 74 | 54 | 0.886 | 0.926 | 90.5% |

The lower test AUROC against hard negatives than random negatives demonstrates
a harder retrieval challenge, but **not** independently verified clinical
contradiction detection. Donors were preselected by labels extracted from
their own reports; matched pairs can contain reporting errors, and the actual
CXR finding has not been independently adjudicated. No threshold or
probability calibration was performed. The next gate is independent image-side
evidence and case-level adjudication before using BioViL-T as a clinical
selection score.

The next check is
`real_validation/xrv_real_findings.py` with
`slurm/03_real_xrv_weak_findings_p100.sbatch`. It applies the frozen XRV
image classifier to the same fixed 128+128 real images and reports per-finding
AUROC only where both report-derived classes have at least ten examples.
Unknown/uncertain reference labels are excluded, and classes with insufficient
support remain unavailable. Its reference is still report-derived, not an
independent radiologist image label; XRV may also have trained on MIMIC-CXR.
This check diagnoses classifier coverage and failure modes but cannot alone
clear the clinical-adjudication gate or justify a selection threshold.

The approved script completed as job `12529380` in 1 minute 48 seconds. Its
protected result is
`artifacts/protected/tricompose_v1_2/real_validation/real_xrv_weak_valtest128_20261001_001/`.
The table shows AUROC only where each class had at least ten explicit labels;
`NA` means insufficient support, not a score of zero.

| Finding | Val AUROC | Test AUROC | Test positive / negative |
|---|---:|---:|---:|
| Cardiomegaly | 0.935 | NA | 31 / 6 |
| Edema | 0.828 | 0.904 | 18 / 18 |
| Pleural effusion | 0.864 | 0.809 | 47 / 19 |
| Pneumonia | 0.576 | 0.790 | 10 / 10 |
| Pneumothorax | 0.684 | 0.690 | 16 / 25 |
| Atelectasis, consolidation, lung opacity | NA | NA | insufficient explicit negatives |

Only five findings on validation and four on test met this minimum support.
The variation for pneumonia and consistently weak pneumothorax separation
argue against treating raw XRV values as authoritative image truth. These are
operating-point-normalized scores, not calibrated probabilities. No threshold
was fitted. The run's recorded `peak_vram_gib` excludes model loading because
CUDA peak statistics were reset after initialization; do not use that field as
total GPU memory cost.

The approved official human-report-label **metadata audit** implemented in
`real_validation/audit_official_report_gold.py` completed through
`slurm/25_official_report_gold_source_schema_existing_cpu.sh`: existing CPU
allocation `12576792`, four CPUs / 32 GiB allocated, one worker, 120-second
timeout, no new `sbatch` or GPU. It ran in 0.745 seconds and replayed exactly.
Output: `artifacts/protected/tricompose_v1_2/real_validation/official_report_gold_coverage_12576792_source1/`.
Patient-disjoint linkage, four-state denominators, deduplication, artifact/source
hashes and protected permissions passed; original score/winner hashes stayed
unchanged. Status: `completed_verified_metadata_only_not_model_validation`.
Test report-path **metadata** is available; existence/content remains unchecked.

The initial exact-column audit rejected **Airspace Opacity** versus the expected
**Lung Opacity**. The final audit preserves the original source label name and
leaves the common Lung Opacity mapping null, without altering label values or
frozen report rules. Historical wrappers `22`–`24` preserve old pins, not
current rerun commands. See [the handoff](../docs/report_assertion_status_20261002.md)
for failure provenance and the protected execution/verification receipt.
This audit consumed only approved label/linkage metadata, not report files,
separate EHR files or image pixels. It produced no model accuracy, fitted
threshold, fault attribution, selection update or regeneration. Official report
labels are not image truth; CheXbert training overlap/annotation alignment
remain unverified. **398 invented-fixture tests pass**, including 19 audit
tests. A real-report/model benchmark needs separate execution approval.

The real-report label diagnostic is implemented in
`real_validation/official_report_benchmark.py` and
`real_validation/run_official_report_benchmark.py`, with
`slurm/26_official_report_chexbert_debug_p100.sbatch`: debug, one P100, two CPUs,
16 GiB RAM, ten minutes maximum. **Approved and completed as job 12594397**,
exit `0:0` on `e23-02`, allocation elapsed 15 seconds. Program runtime was
12.025 seconds and peak allocated VRAM was 1.227 GiB. It uses only a
declared conservative Impression section, retains missingness, compares frozen
categorical states to the original human labels, excludes unaligned opacity/
binary heads from four-state metrics, and diagnoses the unchanged four-finding
scope guard without changing selection. Source text/keys/paths/images/reference
rows are not exported. [Protocol and limitations](../docs/report_assertion_status_20261002.md)
are frozen before real results. **421 invented-fixture tests pass**, including
23 benchmark tests. Separate approval of the complete batch script preceded
real-report/model consumption; prior metadata approval was not a submission approval.
Nonempty predictions, artifact hashes, inventory denominators and protected
permissions passed post-run checks. The output is
`artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_reports_12594397/`;
the manifest SHA256 is
`aeea0df50e73c4b57af35d3d686ee5d3677f1663174cf28a570205f5248deec6`.
Per-finding results remain protected. This does not clear an independent
clinical/image/EHR validation gate, tune any rule, change a score/winner or
authorize targeted regeneration.
The readable bilingual diagnostic is
`artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_review_12594397_001/RESULTS_CN_EN.md`,
with a hash-bound receipt and manifest. It separates full inventory coverage,
four-state F1, non-unknown matches, omissions, explicit polarity flips and
guard commit coverage. Statistics were recomputed from derived confusion
matrices, not by reopening source reports or rerunning a model.
The existing synthetic pool also has a derived diagnostic overlay at
`artifacts/protected/tricompose_v1_2/diagnostics/candidate_report_diagnostic_12594397_001/`:
48 candidate triples, two EHR cases, 384 finding rows. Its CSV connects cached
report states and scope availability to per-head real-report diagnostics,
without new scores, ranks, model calls or raw-text/image reads. This does not
establish checkpoint/section equivalence, distribution transfer or clinical
truth, and must not be used to fit weights on the official test source.

## Next fact-conflict diagnostic: BioViL-T polarity / 单事实正负极性

`real_validation/biovil_fact_polarity.py` and
`slurm/27_real_biovil_fact_polarity_debug_p100.sbatch` were separately approved
and completed as **job 12597637** on `e23-02`, exit `0:0` in 49 seconds.
They reuse the existing fixed 128+128 real CXR cohort, test eight findings with
three authored presence/absence template pairs, and keep images fixed. All
prompts are scored label-blind; report-derived cached references enter only
the summary, with unknown/uncertain denominators retained. Raw source report
text and EHR fields are not read, and reference rows are not exported.

The diagnostic separates positive/negative polarity wins, balanced wins,
margin AUROC where support permits, fixed template-family results and the
predeclared mean. A scorer can have strong retrieval or margin AUROC while
always preferring positive text; balanced polarity wins expose that failure.
No threshold/template/rule fitting, changed winner or repair claim is enabled.
References remain weak report labels, not independently adjudicated image truth;
the reused splits and possible MIMIC training overlap preclude an independent
final-test claim. See [the frozen protocol](../docs/biovil_fact_polarity_protocol.md).

Request: debug P100 × 1, CPU × 2, host RAM 16 GiB, ten-minute resource cap.
The complete script received separate explicit approval before submission.
**440 fixture tests pass**, including 19 new polarity tests; no real image/model
was loaded during preparation.
The submitted output is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_fact_polarity_12597637/`.
Program runtime was 45.827 seconds and allocated peak VRAM including model
loading 0.576 GiB. Derived metrics recomputed exactly; fixed source/reference
bindings, all templates, protected permissions and old winner/table hashes
passed verification. The readable result is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_polarity_review_12597637_001/RESULTS_CN_EN.md`.
Polarity performance depends on finding and authored sentence family; retain
BioViL-T as secondary evidence, not a sole clinical repair judge. No independent
image adjudication, calibrated threshold, score/winner update or regeneration
authorization is claimed by this weak-reference result.

Unit tests use only invented synthetic metadata:

```bash
PYTHONPATH="src:TriCompose-v1.0/src:TriCompose-v1.1/src:experiments/synehrgy_v2/src:TriCompose-v1.2/real_validation:TriCompose-v1.0/eval/report_v1_1" \
  PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s TriCompose-v1.2/tests -p 'test_*.py' -v
```

## Decision-interface preview / 决策接口预览 (2026-10-02)

The pure `src/tricompose_v12/decision_preview.py` connects existing cached
finding states to a deterministic **verification request**, not an executable
clinical repair policy. See [the interface protocol](../docs/decision_preview_protocol.md).
It retains all eight findings, fixed EHR hashes and source evidence IDs; direct
EHR coverage gaps and correlated report/scorer disagreement remain explicit.
Raw/scoped edge arithmetic is independently checked before any decision hint.
Agreement cannot grant clinical acceptance. A provisional report/CXR verification
target is not a confirmed faulty modality, and every model-execution flag is false.

`benchmarks/preview_candidate_actions.py` completed a derived-only run inside the
existing CPU allocation `12576792`; no new `sbatch`, model, GPU, raw source
patient input, synthetic report body or image pixels were accessed. It consumed
the full prior scope inventory: **48 candidate triples, two fixed EHRs, 384 facts**.
Caller-specified caps of four additional model calls and 120 additional GPU
seconds are engineering limits only; no verification-cost estimate was invented.
Missing runtime/estimates do not become zero, and failed/retried calls count in
the tested ledger. Old bank generation cost is separate from any future calls.

Protected output:
`artifacts/protected/tricompose_v1_2/decision_previews/decision_preview_12576792_001/`
contains `decisions.jsonl`, `action_preview.csv`, `summary.json`,
`README_CN_EN.md`, and a hash-bound `manifest.json`. The original source and
winner/table hashes, output inventories and project-group modes were verified.
All proposed next actions request more verification; none executes a repair or
changes acceptance/rejection of any existing case. This demonstrates the guarded
interface, **not** clinical localization accuracy or improved generation.

**473 V1.2 invented-fixture tests pass**, including 33 new tests covering
determinism, unknown/uncertain semantics, unsupported pneumonia scope, unchanged
EHR, scope withdrawal, forged eligibility, correlated reports, budgets and
refusal-before-protected-read outside Slurm. Independent adjudication and the
clinical/calibration gate remain pending; a successful cache preview cannot
enable targeted regeneration or substitute for that evidence.

## No-human automatic policy track / 无人工自动策略路线 (2026-10-02)

The user cannot provide human checks or feedback. That no longer blocks
**exploratory automatic policy experiments**. It does not create independent
clinical truth or authorize radiologist-validated claims. Historical pending
review packets and decision previews are unchanged. The new explicit track is
defined in [the no-human protocol](../docs/automatic_no_human_protocol.md).

Implemented:

- `configs/automatic_replay_v1.json`: frozen model order, six call budgets,
  five random seeds, existing V1.1 candidate objective and invocation accounting.
- `src/tricompose_v12/automatic_replay.py`: fixed, random, prefix static rerank,
  and targeted heuristic, observing only requested candidate slots.
- `benchmarks/run_automatic_proxy_replay.py`: hash-checked protected cache
  comparison, with alternate automatic readouts and per-step action/cost traces.
- `tests/test_automatic_replay.py`: 26 invented-grid tests, including no-peeking,
  fixed EHR, image reuse, budget bounds, no secondary-score routing, missingness,
  raw-versus-guarded scope, legacy normal-statement metrics and source integrity.

Completed protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_replay_12576792_001/`.
The run uses all 48 candidates from two fixed EHRs, not 48 independent patients.
It has 96 replay trials: two EHRs × six budgets × (three deterministic methods
+ five random replicates). The table is `method_comparison.csv`; per-step
records are `replay_outcomes.jsonl`; readable interpretation is `RESULTS_CN_EN.md`.
The program took 0.030806 seconds within the existing CPU allocation, with zero
new model calls, GPU execution, patient source inputs, text/image opens or Slurm
submissions. Exact replay, source/result hashes, protected permissions and old
winner/table hashes passed. The full local V1.2 suite has **499 passing tests**.

This is no longer an all-`verify_more` interface: heuristic actions explore
existing report/model/seed alternatives and stop at a proxy constraint or call
budget. It still does **not** physically regenerate an image/report. New
prospective execution requires its own reviewed Slurm script and approval.
The private table deliberately separates optimization proxies from BioViL-T
and guarded readouts; raw label gains are not evidence of clinical correctness.
No calibration, mask, stopping rule or model priority was retuned on the replay
outcomes. The next no-human step is transfer to a separately frozen larger
cohort/alternative automatic evaluator, followed by bounded prospective calls
with full actual generation/verification/failure cost accounting.

## Fixed 80-EHR automatic replay / 固定 80 例自动对比 (2026-10-02)

Implemented `src/tricompose_v12/legacy_replay_adapter.py`,
`configs/automatic_replay_pool80_v1.json`,
`benchmarks/run_legacy_automatic_replay.py` and a private verification reader.
They reuse cached synthetic states/lineage only, not image pixels/report bodies.
All 80 fixed EHRs and 960 source triples were retained; 3,200 budget/seed trials
completed inside the existing CPU allocation with no new models or submissions.
The source bank is the explicitly named historical uncalibrated fourteen-head
profile, not the new calibrated eight-head profile. Do not pool their scores.

Protected outputs below
`artifacts/protected/tricompose_v1_2/automatic_replays/`:

- `automatic_replay_pool80_12576792_001/method_comparison.csv`: full budget curves.
- `automatic_replay_pool80_12576792_001/subgroup_comparison.csv`: direct-EHR
  and no-direct-EHR groups, with missing edge rates preserved as NA.
- `automatic_replay_pool80_review_12576792_001/RESULTS_CN_EN.md`: readable
  result directions, coverage-definition distinction and verification status.

Exact replay, budget/summary arithmetic, artifact/source/winner hashes and
private permissions passed. Historical prompt-conditioning tiers (15/65)
must not be confused with the directly comparable cached EHR labels (8/72).
No facts were inserted and no cases were discarded. The maximum-budget
outcome remains a proxy cost/quality tradeoff, not independently validated
clinical repair or an established advantage over exhaustive static selection.

The distinct BioViL-T endpoint request is frozen in
`automatic_biovil_request_12576792_001/`. All metadata lineage/preflight checks
passed without opening report/image bodies. `benchmarks/score_automatic_replay_biovil.py`
and `slurm/28_automatic_pool80_biovil_debug_p100.sbatch` use one
P100/two CPUs/8 GiB/ten minutes. The complete script was displayed and approved;
**GPU scoring completed as job 12607645**, exit `0:0`, on `e23-02` in 36 seconds
(program 33.322983 seconds, peak GPU allocation 0.61 GiB). BioViL is excluded
from policy selection; full-report context failures stay NA, not silently
truncated scores. No training, downloads, original-winner changes or GitHub
actions were performed. **537 invented-fixture tests pass**.

## BioViL endpoint comparison / 补充评分对比 (2026-10-02)

Implemented `src/tricompose_v12/automatic_secondary.py`,
`benchmarks/merge_automatic_secondary.py` and thirteen new invented-cache tests.
The immutable secondary scores are overlaid in a NEW protected run without
changing any selected candidate, action trace, original score or fixed EHR.
All five random replicates are averaged within a case; paired comparisons
require both endpoints and retain missing-case counts, rather than treating
repeated seeds as patients or silently filtering incomplete replicates.

Read the current completed comparison under
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_secondary_pool80_12607645_001/`:

- `RESULTS_CN_EN.md`: readable method/budget table and interpretation.
- `method_comparison.csv`: proxy metrics, simulated calls and BioViL availability.
- `subgroup_comparison.csv`: direct-EHR versus no-direct-EHR readouts.
- `paired_case_comparison.csv`: paired BioViL differences and full-cohort call
  differences for every budget against fixed/random/static baselines.

The maximum-budget BioViL paired mean is lower for targeted search than all
three baseline method IDs. Here `random` means random-order acquisition followed
by the same score-based reranking, NOT a random final triple. At the complete-bank
budget it selects exactly the static winner in all five seeds for all 80 cases;
random/static are therefore not two independent endpoint confirmations.
Thus current proxy improvements do NOT demonstrate a general
consistency/repair advantage. Preserve the negative result; do not change the
cohort, weights, priorities or existing winners to improve this endpoint.
BioViL remains an alternate uncalibrated readout, not independent clinical
truth or confirmed fault localization. Exact arithmetic, source/result hashes,
unchanged actions/winners/EHRs and protected permissions passed verification.
No generation/training/repair/API/GitHub action was executed. Actual endpoint
GPU scoring cost is reported separately from simulated policy invocations.

## Frozen discrepancy diagnostic / 冻结分歧诊断 (2026-10-02)

Implemented `src/tricompose_v12/automatic_discrepancy.py`,
`benchmarks/diagnose_automatic_discrepancy.py` and fifteen invented-cache tests.
The diagnostic reuses cached synthetic states, hashes, choices and endpoint
scores only: no report bodies, image pixels, raw patient inputs or models.
It decomposes explicit positive/negative support, retains unknown/uncertain
and missingness, and distinguishes legacy source adjustments from raw states.
Fixed EHRs, source counters, model order, action traces and winners are unchanged.

Completed inside the existing CPU allocation `12605930` in 0.607697 seconds,
with zero new model calls, GPU work or Slurm submissions. Protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_discrepancy_pool80_12605930_001/`.

- `RESULTS_CN_EN.md`: readable decomposition and interpretation.
- `case_contrasts.jsonl` / `action_slices.csv`: 800 contrasts, grouped by budget,
  EHR evidence coverage and hash-based image/report changes.
- `same_image_report_pairs.jsonl` / `same_image_controls.csv`: 532 observed
  report-pair controls on 210 case/image groups; within-case means preserve
  correlation and the selected-union sampling limitation.
- `model_selection_frequencies.csv`: selection frequencies, not model accuracy.
- `summary.json` / `manifest.json`: counters, limitations and hash provenance.

At maximum budget versus fixed, 30 cases keep the same artifacts, 12 change
only the report, and 38 change both image and report. The report-only group has
a positive mean BioViL difference; the overall decline is concentrated in the
joint-change group. Support gains include both polarities, but the negative
increment is larger. Normal agreement is legitimate: do not add diseases or
discard healthy/underconditioned EHRs to alter this diagnostic. Joint changes
also change the classifier reference, so an improved proxy does not identify
which modality was repaired. These are descriptive associations, not causal
clinical localization or a significance result.

Common-image comparisons also reveal semantically different reports tied on
the earlier coarse ranking criteria and decided by runtime. This motivates a
separately preregistered fixed-image/cost-tie control, not retrospective policy
tuning or a claim that faster models are worse. Existing controlled fault
diagnostics are not relabeled as a newly completed repair benchmark.
Exact regeneration, all source/result hashes, old artifacts and protected modes
passed checks; the full V1.2 suite now has **552 passing invented-fixture tests**.
Next controls and evidence boundaries are recorded in the no-human protocol.

## Fixed-image report control / 固定图像报告选择对照 (2026-10-02)

Implemented `src/tricompose_v12/fixed_image_control.py`,
`benchmarks/run_fixed_image_control.py`, a separate frozen control configuration,
and nineteen invented-cache tests. The original Sana seed 0 image is fixed for
every EHR, without consulting report/endpoint scores. Static report-only search
visits the four registered reports; targeted report-only search retains the
old report-switch/stop rules and explicitly blocks any image-change request.
Source ranking, fixed EHRs, unknown/uncertain semantics and old outcomes remain
unchanged. BioViL scores are attached only AFTER choices; absent scores stay NA,
never a reason to substitute a previously scored candidate.

The new control is exploratory on the already inspected development cohort,
not retrospectively preregistered independent validation. See
[the frozen control protocol](../docs/fixed_image_control_protocol.md).
The completed protected run is
`artifacts/protected/tricompose_v1_2/automatic_replays/fixed_image_control_pool80_12605930_001/`.
It contains `RESULTS_CN_EN.md`, `method_comparison.csv`,
`subgroup_comparison.csv`, `paired_case_comparison.csv`, `control_outcomes.jsonl`,
`case_contrasts.jsonl`, `terminal_reasons.csv`, `frozen_control.json`, and hashes.

All 80 cases were retained, with 800 new report-only replay trials and 1,200
reused deterministic baseline trials. The program took 0.909379 seconds inside
the existing CPU allocation, with zero new model calls/submissions or text/image
opens. All control winners already had cached endpoint scores; zero pairs need
GPU scoring. Exact regeneration, source/result hashes, old selection artifacts
and protected permissions passed; **571 invented-fixture tests pass**.

At maximum cap, report-only selection has a higher mean BioViL readout than the
fixed/joint methods, but versus fixed only seven cases improve, nine decline
and 64 remain identical. A positive mean is not uniform improvement or clinical
accuracy. Report-only static and targeted select exactly the same final outputs
for all 80 cases: current dynamic rules have no additional endpoint-quality
gain there. Targeted saves only one blocked case's remaining report calls; 79
cases exhaust all four reports, none achieves the proxy-stop condition. Do not
describe that exhaustion as successful repair.

Cost is simulated, not measured regeneration GPU savings: one image plus four
reports costs ten generator/scorer invocations, while the joint inventory can
cost thirty. Equal caps do not equate search spaces/expenditure. This supports
separating report selection from image-switch verification in the next control,
not claiming that this old proxy already provides a working repair algorithm.

## Ranking and invariant-switch controls / 排序与固定证据对照 (2026-10-02)

Implemented `src/tricompose_v12/ranking_switch_controls.py`,
`benchmarks/run_ranking_switch_controls.py`, a separate frozen configuration and
twenty invented-history tests. See the
[control protocol](../docs/ranking_switch_control_protocol.md).
The runtime ablation reselects ONLY from each trial's original observed slots;
all earlier quality-key components, action traces, costs and stop reasons stay
fixed. Original proxy-stop choices are retained. It is final-ranking ablation,
not a new routed execution or removal of cost accounting. BioViL/availability
cannot affect a choice. Unscored choices remain NA with a pending hash inventory.

The invariant diagnostic holds explicit cached EHR states/hashes fixed, separates
image/report agreement within versus outside those constraints, and preserves
unknown/uncertain and no-direct-EHR cases. It never promotes weak medication/lab
context or diagnoses into newly asserted image truth. Agreement counts remain
unverified automatic proxies, not an error-localization/repair gate.

Protected run:
`artifacts/protected/tricompose_v1_2/automatic_replays/ranking_switch_controls_pool80_12605930_001/`.
It contains `RESULTS_CN_EN.md`, `ranking_comparison.csv`,
`invariant_switch_summary.csv`, protected per-case histories/contrasts,
`pending_endpoint_pairs.json`, frozen configuration and source/result hashes.
All 80 EHRs, 2,000 ranking trials and 800 invariant contrasts completed in
1.787128 seconds inside the existing CPU allocation, with zero new models,
GPU work, text/image opens or submissions. Exact replay, original histories,
all source/result hashes, old artifacts and private modes passed.
**591 invented-fixture tests pass**.

At maximum cap, removing runtime changes one report-only winner and ten joint
winners (nine change images); the mean BioViL differences are negative. This
does NOT show that runtime always improves clinical quality, but the simple
remove-runtime hypothesis is unsupported here. All maximum-cap endpoint pairs
are available. One new pair outside that scored union remains NA at a lower cap;
its pending metadata is not GPU-job authorization.

Of the original targeted policy's 38 image changes, 33 have no comparable
explicit cached EHR finding and five do. In those five, image support for the
fixed EHR increases in one case, while all-three supported-fact count is unchanged.
Most extra image/report agreement occurs outside explicit EHR constraints.
This is a lack of verification evidence, NOT proof those images are clinically
wrong or that those EHRs have no useful context. Normal agreement remains valid.
Future image-switch verification must distinguish invariant evidence from a
new classifier reference; do not fit a revised policy on this cohort and call
it independent evaluation. Existing winners/rules remain unchanged.

## Invariant verification hook / 固定 EHR 验证接口 (2026-10-02)

Implemented `src/tricompose_v12/invariant_verification.py`,
`benchmarks/bind_invariant_verification.py`, a frozen interface configuration
and twenty-five invented-cache tests. See
[the interface contract](../docs/invariant_verification_interface.md).

The public Python hooks are `anchor_from_cached_candidate`, `verify_candidate`
and `verify_transition`. An immutable EHR anchor binds states, cached source
categories and EHR/facts hashes; its evidence IDs never depend on the selected
image/report. Candidate receipts bind artifact/evidence lineage and all three
raw edge readouts with separate positive/negative support, opposition and
missingness. Transitions track specific fixed-EHR support gained/lost and
opposition added/removed, not a score-derived repair-success flag.

Integration currently binds these hooks to original **cached generation
histories**, not to a newly executed GPU pipeline. It does not edit the V1.0/
V1.1 generators or the existing scoped eight-head decision preview. This
version accepts validated completed legacy triples; partial image-only
verification and prospective inference-adapter hooks remain separate work.
The raw fourteen-head profile is explicitly unverified and uncalibrated.

Completed protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/invariant_verification_pool80_12605930_001/`.
It contains `ehr_anchors.jsonl`, `candidate_receipts.jsonl`,
`verification_events.jsonl`, `trial_bindings.jsonl`, status CSVs,
`RESULTS_CN_EN.md`, frozen configuration and source/result hashes.
All 80 anchors, 960 unique candidate receipts, 3,200 original replay trials,
15,001 requested observations and 11,801 transitions were retained. The run
took 6.452556 seconds in the existing CPU allocation, with zero new model,
GPU, patient input, text/image open or submission. Exact replay, all source/
result hashes, every original winner/action/cost/stop reason, old artifacts and
protected modes passed verification. **616 invented-fixture tests pass**.

No-direct-EHR comparisons stay unavailable; unknown/uncertain are not negative.
Legacy global No-Finding source adjustments cannot edit receipt states.
Unchanged artifacts cannot acquire changed classifier/report vectors, and
changed receipt hashes/arithmetic/scope/acceptance claims are refused.
Scalar BioViL/runtime/proxy scores never alter the reference or receipt.
All agreement/improvement remains proxy evidence, not clinical truth. A
report-only change cannot claim image repair. Receipts grant neither clinical
acceptance nor model-execution approval and do not rerank or drop cases.

Next is prospective hook integration with an independently frozen execution
policy/cost ledger and separately approved bounded inference. This interface
test demonstrates integrity/coverage tracking, not improved generation or a
new clinically validated controller.

## CXR-before-report partial verification / 报告前阶段验证 (2026-10-02)

Implemented `src/tricompose_v12/partial_image_verification.py`,
`benchmarks/bind_partial_image_verification.py`, a separate frozen configuration
and twenty-one invented-state tests. See
[the partial-result contract](../docs/partial_image_verification_contract.md).
The original completed-triple verifier and archived full run are unchanged.

`CachedImageEvidence` and `verify_image_phase` support a CXR with an existing
cached XRV result before a report is available. The image projection works with
all report fields absent; report labels, structure, triple gates, scalar scores,
winner flags and costs do not affect this receipt. Its EHR–CXR readout retains
explicit support, opposition and missingness. Both report edges, report
identity/hash/states and all-three support remain `null / not_generated`.
This is not an all-unknown report vector or a perfect-consistency score.

`bind_completed_report` links a completed receipt without overwriting the image
receipt. It requires the same fixed EHR, image identity/hash, XRV vector and
EHR–CXR counts. Receipts use canonical serialized-content validation, including
refusal of Python-equal but hash-different false/zero or integer/float changes.
No clinical acceptance, repair success, confirmed fault or execution permission
is inferred. The eight-head scoped preview remains separate and unchanged.

Validated protected output:
`artifacts/protected/tricompose_v1_2/automatic_replays/partial_image_verification_pool80_12605930_002/`.
It contains `ehr_anchors.jsonl`, `image_partial_receipts.jsonl`,
`completed_report_bindings.jsonl`, `summary.json`, `RESULTS_CN_EN.md`, frozen
configuration and source/result hashes. All 80 fixed EHRs, 240 image partial
receipts and 960 completed-report bindings completed in 1.658559 seconds in
the existing CPU allocation, with zero new model/GPU/API/submission or raw,
report-body/image-pixel access. Exact reconstruction, source/result hashes,
unchanged full-run/underlying EHR anchors and private modes passed; existing-run
overwrite is refused. **637 invented-fixture tests pass**.

Image-phase statuses: nine explicit proxy agreements, six explicit proxy
oppositions, nine missing image comparisons, and 216 image receipts with no
direct comparable EHR constraint. These are 240 image-level cache readouts, not
240 independent patients or counts of clinical successes/failures. The 216
reflect the original 72 no-direct-EHR cases times three image candidates; they
remain included with NA EHR support/coverage. No disease/device was inserted.

This is explicitly **retrospective phase reconstruction**, NOT proof that the
original job verified images before producing reports, saved report calls,
improved generation or ran a new real-time pipeline. Prospective adapter hooks,
a frozen bounded execution policy and measured failure/retry/verifier costs
remain the next step. Every new GPU job needs full-script/resource approval.

## Bounded execution accounting / 有预算的调用执行层 (2026-10-02)

Implemented `src/tricompose_v12/execution_ledger.py`,
`src/tricompose_v12/runtime_dispatch.py`, `benchmarks/smoke_bounded_execution.py`,
a frozen invented-fixture configuration and thirty-nine new tests. See
[the execution contract](../docs/bounded_execution_ledger_contract.md).
The existing generation/scoring policies, protected full/partial verification
runs and V1.0/V1.1 teammate code are unchanged.

The ledger separates CXR generation, XRV, report generation and CheXbert as
individual single-case invocation **attempts**. It fsyncs a private reservation
before backend invocation, retains failed/in-flight charges, limits retries,
and permits explicit zero-new-cost reuse of committed image/scorer results.
Phase dependencies and fixed EHR/image/report/model-audit/seed hashes cannot
change silently. Restart validates the canonical event chain and frozen budget/
retry/mode contract; an in-flight worker is not automatically rerun or refunded.
Python output is redirected to private operation logs, not public stdout.

Validated protected smoke:
`artifacts/protected/tricompose_v1_2/execution_smokes/bounded_execution_fixture_12605930_001/`.
It contains `RESULTS_CN_EN.md`, `summary.json`, per-case durable journals,
ledger snapshots, invented partial/full verification receipts, frozen config,
private fixture logs and source/result hashes. **676 invented-fixture tests
pass**. Exact journal restoration, event/cost arithmetic, protected modes,
all source/result hashes, original full/partial run hashes and no-overwrite
checks pass. Runtime is 0.047964 seconds in the existing CPU allocation.

This uses TWO explicitly INVENTED IDs, not the historical 80-case cohort.
There are ten reserved fixture attempts, one invented timeout, and zero actual
model calls or generation outputs. One fixture reuses its image/scorer and
reaches the five-attempt budget with a second report unscored; the other charges
the image-scorer retry. Unscored/unresolved output is not a negative finding,
clinical failure or acceptance. Partial/full hooks preserve the fixed anchor.

**At this archived fixture stage, only the invented backend was supplied.** The dispatcher/ledger primitives
are implemented, but authentic generator/scorer-provenance subprocess adapters
are not connected yet. A recorded frozen audit hash or receipt ID is not itself
proof of freezing, clinical truth or approval. A deployed adapter must validate
real artifact hashes and checkpoint/scorer provenance, capture child/native
output, retain failed/in-flight journals and enforce Slurm resources. Do not
relabel fresh inference as the historical fourteen-head cache or silently mix
it with the eight-head scoped interface. No GPU/API/download/training/job was
started, and no script has been submitted or implicitly approved.

Next connect those already deployed frozen workers to this bounded execution
layer with a separate small-cohort policy and complete script/resource review.
This is accounting/dispatch readiness, not a completed prospective pipeline,
measured GPU saving or a quality-improvement experiment.

## Real frozen-worker adapters / 真实后端接入 (2026-10-02)

Implemented `live_workers.py`, `live_plan.py`, `live_execution.py`, separate
fresh `live_receipts.py`, CPU preflight and GPU-only runner CLIs. Existing
V1.0/V1.1/model adapters and the archived verification/ledger runs are unchanged.
See [the live-worker contract](../docs/live_frozen_workers_contract.md).

The protected plan is
`artifacts/protected/tricompose_v1_2/live_worker_plans/live_workers2_12605930_001/`.
The original inventory indices 0 and 1, EHR/facts/prompt hashes, real checkpoint
bytes, model audits, scorer sources and frozen threshold bundle passed CPU
preflight in 24.726 seconds. **706 synthetic-only tests pass**. All checkpoint/
source/run hash checks and plan-private modes passed; **zero model calls during preflight**.

Three CXR generators and four CXR-only report experts are registered. The
prepared first P100 subset is RoentGen-v2 + Sana → CXRMate-single, with fresh
XRV and CheXbert verification between phases: four images/four reports, sixteen
normal attempts, maximum ten attempts per fixed EHR including failures/retries.
MAIRA-2's current float32 adapter is not squeezed onto a 16-GB P100. Registered
excluded experts are not newly preflighted/inference-certified by this plan.

The source EHR/prompts are never rewritten/enriched; underconditioned cases
remain with NA direct-EHR support/coverage. Fresh XRV uses eight enabled frozen
threshold heads, **not probabilities** or the old fourteen-head cache. Receipts
record raw positive/negative support, opposition and missingness per edge;
clinical acceptance/repair/primary eligibility stay false/NA. No new scorer,
agent/router training, adaptive quality policy or best-triple selection is added.

After full-script/resource review and explicit user approval,
`slurm/29_live_workers_smoke2_debug_p100.sbatch` was submitted as **job 12615231**:
one P100, two CPU cores, 24 GB host RAM, twenty minutes. It completed on e23-02
in 00:09:10 with exit 0:0. Its stable private output directory contains model
outputs, receipts, durable charged journals and `score_table.csv`; interrupted
runs are retained, not overwritten.
Native subprocess output/caches are private and timeout cleanup targets only
the owned worker group. CPU allocation cannot launch the real GPU workers.

The retained run is `live_worker_runs/live_workers2_12615231/`: **four CXRs,
four reports, sixteen validated calls, zero failures/retries**. The separate
`live_worker_audits/live_workers2_audit_12615231_001/` audit rechecked checkpoint/
source/input/output hashes, actual artifact/tokenizer lineage, exact durable
ledger restoration and all recomputed receipts/score CSV. Worker maximum
allocated VRAM was 7.169 GiB. Framework temporary/cache permissions and the
two Slurm logs were normalized only within this new run to the protected modes;
original source/model/checkpoint files remain unchanged.

Both fixed EHRs lack direct radiographic facts, so **all four completed receipts
remain unverified with NA EHR-related support/coverage**. The successful live
chain is not three-modal clinical correctness, adaptive repair, improved image/
report quality, best-triple selection or measured compute savings. The audit
did not review report bodies/image pixels. Next separately preflight/approve
additional report experts and independent endpoint scoring, then freeze an
adaptive policy/evaluation without fabricating facts or excluding these cases.

## Same-image report choice / 固定胸片的报告对照 (2026-10-03)

Prepared a separate offline control reusing the exact four fresh CXRs and
four CXRMate-single reports. MAIRA-2 will add four reports using the unchanged
official image-only adapter. The explicit frozen proxy policy chooses one
report per fixed image before BioViL-T is invoked; opposition, missing positive
image evidence, positive support and NA EHR coverage remain separate readouts.
Negative agreement is not a reward. No EHR/image is regenerated or enriched.
See [the complete contract](../docs/fixed_image_report_expansion_contract.md).

`prepare_fixed_image_reports.py` completed CPU preflight in 43.230 seconds,
strongly pinning local checkpoint bytes, synthetic artifacts, report requests
and source code, with zero model/runtime calls. **722 synthetic-only tests pass**.
The plan is `fixed_image_report_plans/fixed_reports4_12605930_001/` under the
protected V1.2 root. The A40 debug script is **prepared, not submitted** and
requires separate full-script/resource approval. No new report/endpoint scores
are claimed until that job actually runs.

固定两份 synthetic EHR 和四张 CXR，补 MAIRA-2 报告得到八个图文候选；先按
明确、不可变的标签规则择优，再独立测 BioViL-T。规则把阳性标签缺失也计入
代价，避免只靠省略获得高分；unknown/uncertain 不当阴性，两份报告不当独立
临床投票。当前两份 EHR 的直接影像条件为零，因此 EHR 相关比率仍为 NA。
这一轮是同图报告选择的工程对照，不能宣称完整三模态正确性、错误定位成功、
自适应修复或论文级质量提升。

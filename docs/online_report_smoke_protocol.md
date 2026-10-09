# Online report escalation: prepared prospective engineering test

Two fixed, already-inspected synthetic DEVELOPMENT EHR anchors from the previous
EHR-only-stratified retry plan. No EHR/prompt rewriting, difficulty-based case
replacement, new patient selection, training or scorer threshold fitting.
Generate one new RoentGen-v2 CXR at predeclared seed 2 per EHR; generators consume
EHR-derived radiology text, not native structured EHR. A new seed is not an
optimization selected by endpoint. There is no CXR repair branch in this test.

## Online method and controls

Initial prefix: CXR generator, XRV, CXRMate-single, CheXbert: four charged
single-case attempts. All models frozen; CXRMate-single is not CXRMate-ED.
Use fresh eight-enabled-head receipts with six disabled unknowns and fixed EHR
anchor provenance. All report experts here remain CXR dependent.

If direct EHR/image proxy opposition exists, stop unresolved: report-only
escalation cannot certify or repair image fidelity. Otherwise request the second
report expert (CheXagent-2) for section failure, common generic/temporal/repetition
risk, direct EHR/report or image/report opposition, missing direct EHR/report
comparisons or missing image-positive comparisons. Missing negatives are not
automatically report errors. With none of these signals, stop unverified.
These triggers are heuristics, not confirmed clinical faulty-modality attribution.

Escalation adds report generation plus CheXbert, at most two additional charges.
Use the frozen fresh output veto after completion, fall back to the initial
section-eligible reference, or null if ineligible. Operational failure stays
charged/unresolved, never becomes a clinical contradiction. Six primary attempts
per EHR maximum, zero operational retries. Actual journal reservations precede
backends; new continuation journal links the exact initial prefix without
overwriting it. No blind resume. Backend wall timeout 180 seconds per process.

Controls: fixed initial path (four attempts normally), and always generate the
second expert plus the SAME strict output veto (six normally). Both online and
static are under the same six-attempt cap, but fixed has less expenditure.
The static branch shares actual generated candidates with online. If online
stops early, seal/fsync its choice BEFORE acquiring the shadow static report.
That extra report cannot enter online observations or revise its sealed choice.
Online stops are real chronological decisions, unlike a full-bank replay.

The shadow control is actual paid work. Record each method's executed prefix
separately from total collection work and wall time. At most twelve primary
attempts total (2 images, 4 reports, 2 XRV operations, 4 CheXbert operations).
No actual GPU saving may be claimed because the collection still runs controls;
any potential prefix saving is an engineering accounting result. If the initial
chain fails, retain the case, null output and charges; do not manufacture a
baseline or exclude it from summaries.

Seal all online/static/fixed choices before full-report BioViL endpoint scoring:
at most 2 image and 4 text encodings, charged separately, no silent truncation.
Use the existing local hi-ml vendor bootstrap. Endpoint failure is preserved
with NA, costs and private logs; it does not alter decisions or trigger generation.
No BLEU/reference metrics without a reference report. Raw label and structure
gains are scorer-dependent proxies, not independent clinical factuality.

## Implementation and preflight

New tool `TriCompose-v1.2/tools/online_report_smoke.py`, tests
`TriCompose-v1.2/tests/test_online_report_smoke.py`. Do not edit consumed
historical workers/modules/plans/protocols/output. Reuse their frozen official
generation, scoring, dependency-validation and ledger primitives unchanged.
CPU prepare authenticates source plan/case/input/model/checkpoint/configuration
hashes, preflights the second report worker without constructing a model factory
and seals the complete plan. The pure route excludes secondary scores and old
winners. No inference, raw patient data, report/EHR body or image-pixel read during
preparation. Large checkpoint hash scans require the existing CPU Slurm allocation.

GPU run enforces Slurm allocation/visibility and planning VRAM >=24 GiB, then
performs actual frozen generation. All source-sensitive bodies, generated
reports/images, child logs, journals, caches and outputs remain project-private
under artifacts/protected, 2770 directories/0660 files. Exclusive stable output
path, no overwrite. Normalize only new run files; never external assets.

Tests must cover four-state rules, unknown and weak context, direct-image
abstention, report-only trigger reasons, no endpoint leakage, frozen policy,
exact prefix/continuation/failed cost arithmetic, online-before-shadow seal,
no online observation of controls, missing candidates/nulls, endpoint NA,
plan/input/model pinning and GPU/CPU guards before data access.

Prepared resources: GPU partition, one V100 (CARC lists 32GB VRAM), four CPUs,
48GB host RAM, thirty-minute wall upper bound, no array or training. This is
not a runtime estimate or CUDA-compatibility certification. Inspect current
noderes before submission. Display the full concrete sbatch script/resources
and obtain explicit approval afterwards. PREPARED ONLY: no submission is
authorized by a protocol or generic continuation before that display.

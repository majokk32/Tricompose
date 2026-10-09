# Frozen span V2 versus manual report labels — completed, limited agreement

This follows the completed same-cohort span V2 interface pilot. JSON/source-ID
availability reached 22/22; it did not establish semantic correctness. This new
diagnostic uses the existing manual-label resource instead of declaring model
agreement clinical truth. No old scores, winners or protected runs are changed.

## Reference and scope

The [official MIMIC-CXR-JPG 2.1.0 documentation](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
describes radiologist-authored report labels for fourteen categories. They are
report-level labels, not gold evidence spans, image adjudication or a guarantee
of agreement with our stricter temporal/qualified-absence policy. Local bytes
have passed schema/coverage auditing; official byte authenticity and checkpoint
training overlap are still unverified.

Reuse the immutable metadata audit and CheXbert diagnostic. Retain all 687
annotation entries, including 267 unlinked and 393 linked non-test entries;
only all 27 metadata-linked test reports may supply text. Previously, the
unchanged strict Impression parser accepted 15 and rejected 12. These are
coverage observations, not eligibility chosen by score, finding or model result.
Do not seek easier cases, use train/val entries, add a findings/full-report
fallback or repair the parser after seeing this diagnostic.

Use only the four frozen V2 heads: cardiomegaly, consolidation, pleural effusion,
pneumothorax. Preserve source label names and four states. Other labels are not
in scope. No EHR, CXR, generator identity, old prediction, manual label or score
enters a model request. These real reports differ from the 24 synthetic pilot
slots, but this already-used MIMIC resource is NOT an untouched final test.

## Fixed execution and metrics

Same V2 prompt/segmentation/decoder, checkpoint/loader, greedy inference,
seed 0, 512 output-token cap, no retries. Maximum 27 calls; the old parser
coverage suggests about 15 usable reports, not a promised completion count.
No checkpoint acquisition, training, threshold fitting, API, new generation,
reranking or modality repair.

Report all inventory and failure reasons; do not encode a failed response as a
negative or even as a successful all-unknown prediction. Report:

- Per-head four-state confusion, class precision/recall/F1 and reference counts.
- Annotated-state match, omissions, hard positive/negative flips and determinate
  promotions on unknown/uncertain references; unknown/unknown matches cannot
  inflate the separately reported annotated-state metric.
- Failure-aware annotated recovery over all request-ready annotated checks,
  including contract failures in its denominator.
- Complete all-unknown outputs, input/contract coverage, calls, runtime/memory.
- Paired CheXbert/V2 metrics on exact same Impression hashes; no policy weights
  or model priority are fitted to either outcome.

Correct report-level labels cannot certify the selected span or its temporal
scope. No new label accuracy may authorize primary ranking, natural-error
localization or repair. The small number of studies is not an independent
patient count; no significance claim or patient-cluster CI is produced.

## Privacy and preparation status

Original annotation/linkage rows and real reports are consumed only inside a
separately approved GPU Slurm process with actual job cgroup/CUDA guards. No raw
text, keys, per-record source paths, reference rows or model responses are
exported. Derived opaque predictions and evidence offsets/hashes stay protected;
aggregate tables contain no clinical text. All external sources remain read-only.

Worker: `TriCompose-v1.2/tools/verify_official_report_spans.py`.
Sixteen invented/mocked tests passed; full V1.2 suite: **1,182 tests, 8.692s**.
Metadata-only preparation in existing CPU allocation 12645021 made zero new
model/API/Slurm calls and opened no original source rows or reports.

Protected plan: `official_span_plans/official_span_v2_12645021_001/`, relative
to `artifacts/protected/tricompose_v1_2/`. Manifest SHA256:
`c27454cf8b0eaf36222fef7617862ccd3774bdb4f92b6fe3f2dbf660bb3adc13`.

Approved/submitted script: `TriCompose-v1.2/slurm/42_official_report_spans_flexible_gpu.sbatch`.
One A40/A100/L40S GPU (runtime VRAM guard >=24 GiB), two CPUs, 32G host RAM,
20-minute upper limit. This is not an inference/queue prediction. No drained
node is forced and 16-GiB P100 is deliberately ineligible.

After complete script/resource display and explicit user approval, job
**12645961** completed on A100 in **1m09s**, exit 0:0, at
**2026-10-04 12:14:35–12:15:44 server time**. It made exactly 15 model calls,
zero retries; BF16, PyTorch 2.9.1+cu128, peak allocated VRAM **15.724 GiB**.

Completed output: `real_validation/official_span_v2_12645961/`, containing
`summary.json`, `predictions.json`, `RESULTS_CN_EN.md`, `manifest.json`.
All run directories are fresh/atomic and refuse overwriting. Every further
submission requires a separate complete script/resource display and approval.

## Completed result and independent output audit

All 687 annotation entries remain represented: 267 unlinked, 393 linked non-test,
12 without an explicit Impression, **15 complete**. All 15 ready requests pass
the response contract; one complete response has all four heads unknown.
No response hit the token cap. These are 15 study reports, not 687 evaluated
reports or 60 independent patients.

| Finding | Annotated reference checks | V2 exact matches | Hard positive/negative flips | Paired CheXbert exact matches |
|---|---:|---:|---:|---:|
| Cardiomegaly | 5 | 4 | 1 | 5 |
| Consolidation | 2 | 2 | 0 | 2 |
| Pleural effusion | 11 | 8 | 1 | 11 |
| Pneumothorax | 8 | 6 | 0 | 8 |
| Total | 26 | 20 | 2 | 26 |

Failure-aware annotated-state recovery is **20/26 = 0.7692**, not the JSON pass
rate. The six nonmatching annotated checks consist of two hard polarity flips
and four determinate references predicted uncertain; there are zero omitted
annotated assertions. Separately, **five** unknown-reference checks receive a
determinate prediction. Unknown is not a negative, so these are promotions
relative to the annotation policy, not five independently adjudicated clinical
hallucinations. Retain the full four-state confusion tables and denominators.

The paired CheXbert readout is 26/26 on these annotated checks. This tiny,
previously used resource, lack of uncertain reference examples in the evaluated
four-head sample and unresolved training/policy overlap do NOT qualify either
extractor as a universal clinical evaluator. Manual finding states are not gold
span/temporal-context annotations. In particular, syntactically valid V2 evidence
IDs cannot alone decide which modality to regenerate.

Result manifest SHA256:
`b9274424e49301b8e41e147716a0e751cf3743fc5f1a032f91a4dc1b7baf4b5b`.
Independent audit checked all 128 source bindings, result hashes, 687 opaque
prediction records, 15-call accounting, evidence-field allowlists, source offsets,
four-state derivation, aggregate confusion/metric algebra, exact paired historical
CheXbert statistics, readable-report reproduction and protected group/modes.
The audit opened neither original reference rows nor real report text. The old
V2 receipts and full-bank endpoint manifests remained unchanged.

Current decision: keep V2 secondary and unqualified for primary clinical ranking
or targeted-repair authorization. Do not modify this prompt, reference policy,
thresholds or old winners to improve these observed results. Next diagnostic
must separate missingness, uncertainty and opposing polarity without inventing
human adjudication or reporting image/report generation quality from this test.

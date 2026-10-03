# Live frozen-worker integration / 真实冻结模型执行接口

Status updated 2026-10-03: implemented and CPU-preflighted; the complete script and
resources received explicit user approval and were submitted as **job 12615231**.
The job completed on e23-02 with one P100 (Slurm elapsed 00:09:10, exit 0:0).
Post-run artifact/journal checks passed. This is an operational test, not an improved clinical
generation result, trained agent, validated localization algorithm or repair.

## Immutable inputs / 固定输入

The plan is
`artifacts/protected/tricompose_v1_2/live_worker_plans/live_workers2_12605930_001/`.
Its manifest SHA256 is
`1e18eb7664b406551100ffad296dbbf4c75f8876fe349b062f1d1db8089cc55e`.
CPU preflight completed in 24.726 seconds with zero model calls. File hashes,
private modes and the archived full/partial/ledger-smoke manifests were checked.

Exactly opaque inventory indices 0 and 1 of the original fully synthetic
80-case request bank are used, independent of disease/conditioning difficulty.
There is no new EHR generation, enrichment, disease/device insertion, prompt
rewriting, exclusion of underconditioned cases, raw MIMIC input or real target.
Original single-case request JSONs, canonical EHR/fact hashes, original model
prompt references and cached EHR evidence-category provenance remain fixed.
The EHR evidence anchor is explicitly **cached direct-state/category evidence**,
not a new independent clinical review of EHR contents.

The plan pins source files, dependency-related scorer source fingerprints,
checkpoint bytes and model audits. A source/asset change refuses execution;
rebuild/review a new plan rather than bypassing the hash or overwriting a run.
Environment executables/assets exist; preflight does not import Torch/load a
model or prove CUDA/runtime compatibility. That is the purpose of the live test.

## Registered workers versus first live subset / 注册与本轮实际调用

| Worker | Actual input edge | First P100 smoke |
|---|---|---|
| RoentGen-v2 | EHR-derived radiology text → CXR | Active, existing float16 adapter |
| CheXGenBench Sana | EHR-derived radiology text → CXR | Active, unchanged existing adapter |
| CheXGenBench PixArt | EHR-derived radiology text → CXR | Registered, not preflighted by this plan |
| CXRMate-single | One current synthetic CXR → report | Active, existing float32 adapter |
| MAIRA-2 | One current synthetic CXR → report | Registered; current float32 adapter needs a larger card |
| LLaVA-Rad | One current synthetic CXR → report | Registered, not preflighted by this plan |
| CheXagent-2 | One current synthetic CXR → report | Registered, not preflighted by this plan |
| XRV DenseNet | One synthetic CXR → frozen finding scores/states | Active |
| CheXbert | One generated report → frozen finding states | Active |

Registration is not a new deployment or inference-success certification. All
generators remain frozen; no fine-tuning/LoRA/adapter is trained. LLaVA-Rad's
existing pretrained adapter/base composition is not altered. These text-to-CXR
models are **not direct structured-EHR→CXR models**. CXRMate-single is **not
CXRMate-ED**; it never receives structured EHR, a source/target report or prior.
The first smoke cannot establish multi-report-expert complementarity because
it uses only CXRMate-single. Other registered experts need their own preflight
and separately reviewed compatible GPU request.

## Execution and accounting / 实际执行与成本

```text
fixed EHR + unchanged model-specific prompt
  → one CXR generator
  → one XRV extraction + image-only receipt
  → one CXRMate-single report
  → one CheXbert extraction + completed receipt
```

Repeat the fixed path for two image models and two fixed EHRs: four new images,
four reports, four XRV operations and four CheXbert operations. Normal operation
uses **16 single-case attempts**. Budget is ten charged attempts per EHR, twenty
total maximum; at most one identical-operation timeout retry. Failures and
in-flight reservations are charged and do not count as clinical contradictions.
No quality-driven model/seed changes, reranking, rejection of difficult EHRs,
adaptive stopping, Report→CXR cycle or best-output selection is performed here.

The existing dispatcher fsyncs a reservation BEFORE backend invocation. The
new backend requires Slurm **and explicit GPU-allocation/visibility metadata**
before subprocess launch. It uses argv lists, no shell, local/offline assets,
private per-operation native stdout/stderr, and workspace-only caches/temp.
API/cloud token environment variables are removed; no credential file is read.
Model result manifests must prove one case, one call, frozen audit, original
request, output artifact hash and parent lineage. PNG headers/dimensions are
checked without an anatomy-quality claim. Runtime tokenizer traces are
validated: the saved prompt is the pipeline argument, not automatically the
literal post-formatting tokenizer text. Native transformations are recorded
by the existing trace adapter, not silently called unchanged tokenizer input.

Each worker **process** has a 180-second wall timeout. Only its owned process
group is terminated/reaped before retry. Checkpoint hashing and parent validation
are extra startup/I/O costs; 180 seconds is not a whole-operation deadline.
Slurm's 20-minute wall limit bounds the full job. Reported duration includes
startup/validation/I/O; it is not CUDA kernel time, GPU seconds or compute saved.

The outer run directory is exclusively created at its final stable path,
not renamed afterward: existing adapters embed absolute image/report/tokenizer
paths. Child model/request runs remain individually atomic. Interrupted runs
retain durable journals and private artifacts; no blind resume/refund/overwrite.

## Fresh scoring scope / 新评分结果的语义

Fresh receipts use `fresh_frozen_xrv_chexbert_four_state_diagnostic_v1`, separate
from the historical uncalibrated fourteen-head cache and earlier scoped-preview
profile. XRV uses the already frozen real weak-reference threshold bundle
`real_xrv_thresholds_12549079/thresholds.json`. Checkpoint, preprocessing,
mapping, score space, threshold content and disabled-head mask must match.
Only eight heads are enabled. Disabled/unavailable findings stay unknown.
The legacy field `finding_probabilities` actually stores **operating-point-
normalized XRV scores, not calibrated probabilities**.

The image receipt records only EHR–CXR; both report edges are `null/not_generated`.
The completed receipt binds the exact same fixed EHR/image/XRV receipt to its
report/CheXbert result, then records all three raw edges separately:

- known reference facts and comparable facts;
- explicit positive and negative support;
- explicit proxy opposition and missing comparisons;
- support/coverage denominators, retaining NA when there is no known EHR fact.

Unknown/uncertain never become negative. A globally normal report does not
rewrite an unknown finding. Weak medication/CHF/lab context is not promoted
to a hard image fact. A disabled-head/missing EHR constraint cannot receive
credit from report-image agreement. No normalization or weighted total is
invented. No model is declared best from these raw counts. CheXbert labels
remain unverified, without assertion-grounding or new factuality validation.
The threshold bundle awaits independent review; clinical-accuracy, acceptance,
repair-success and primary-clinical-metric eligibility remain false/NA.

This first job does **not** add new BioViL/Qwen/RadGraph evaluation or demonstrate
fault localization. Independent endpoint scoring/costs and any adaptive policy
require a separately frozen test. The old rules, selections and archived outputs
remain unchanged. This is previously inspected development data, not a new
paper-grade held-out cohort.

## Output and approval / 输出与审批

Prepared batch script: `TriCompose-v1.2/slurm/29_live_workers_smoke2_debug_p100.sbatch`.
Resources: debug, one P100, one task, two CPU cores, **24 GB host RAM**, twenty
minutes, no array. Host RAM is not GPU VRAM. A resource snapshot showed P100
availability and drained A40/A100 nodes; queue placement/time is not guaranteed.
Every submission requires the full script/resource request and explicit approval.

After an approved successful job, stable output is:

```text
artifacts/protected/tricompose_v1_2/live_worker_runs/live_workers2_<job_id>/
├── start_manifest.json
├── cases/<opaque_case>/
│   ├── inputs/{synthetic_ehr.json,ehr_facts.json,cxr_prompts/*.txt}
│   ├── ehr_anchor.json
│   ├── execution.journal.jsonl
│   ├── ledger_snapshot.json
│   ├── case_manifest.json
│   └── operations/<operation_attempt>/...
├── completed_triplets.jsonl
├── score_table.csv
├── execution_summary.json
└── manifest.json
```

All artifacts/copies, generated reports/images and logs remain project-private,
directories 2770/files 0660, group ruishanl_1185 (possibly NFS-mapped nobody).
Controller/native/dispatch logs are private. Public job output is sanitized
status only. CPU preparation produced no synthetic image/report; job 12615231
is the separately approved prospective generation/verification execution.

Software validation: **706 synthetic-only tests pass**, including fixed two-slot
phase wiring, identical timeout retry/cost restoration, failed-parent dependency
skipping, no runtime-factory construction in CPU preflight, private native logging,
GPU guards, owned-process-group cleanup, disabled-head/score-state consistency,
immutable EHR lineage, full/partial receipt binding and NA preservation. Mocked
callbacks are not real generated data or proof of model quality.

## Completed live run and audit / 实际运行完成与核验 (2026-10-03)

Stable private output:
`artifacts/protected/tricompose_v1_2/live_worker_runs/live_workers2_12615231/`.
Both fixed EHRs completed both predeclared image-model paths, yielding **four
CXRs, four reports and four completed verification receipts**. All sixteen
single-case model/scorer operations completed, eight per EHR, with zero failures,
retries or in-flight reservations. No EHR, prompt, old winner or model was changed.
Controller duration including startup/IO/validation was 547.005 seconds; Slurm
elapsed was 550 seconds. Maximum worker-reported allocated VRAM was 7.169 GiB,
not total device memory or a utilization measurement. Slurm batch MaxRSS was
7952104K. No GPU-time saving or adaptive stopping was measured.

The metadata-only post-run audit is
`artifacts/protected/tricompose_v1_2/live_worker_audits/live_workers2_audit_12615231_001/`.
Its manifest SHA256 is
`7485bc033f82cfb4696984588d0588291eb02b220143f54fdb636c5eb5589d15`.
`audits/audit_live_smoke.py` rehashed the frozen checkpoint/source/input/output
files, restored both durable ledgers exactly, revalidated actual model/request
audits and tokenizer traces, and recomputed all partial/completed receipts from
the same fixed proxy evidence. The published score CSV exactly matches those
raw receipt edges. This audit performed zero model calls, no report-body/image-
pixel review and no new independent clinical evaluation.

The framework created sixteen temporary directories with mode 2700 and four
cache files with mode 0664; Slurm created its two logs with mode 0664. After job
completion, permissions were normalized **only for this new run and its two
Slurm logs** to 2770/0660. Project-group ownership was already correct. Protected
ancestor directories remained the access boundary; no external model, checkpoint,
dataset or original source permissions/content were changed. The audit confirmed
the final private modes and groups.

Both predeclared EHRs have **zero cached direct radiographic constraints**. All
four receipts therefore remain `unverified_no_direct_ehr_constraints` and the
EHR-related support/coverage rates remain NA. CXR–report raw classifier/label
comparisons are recorded, but cannot certify three-modal consistency. Do not
turn this engineering success into an EHR-fidelity, clinical quality, best-output,
error-localization or regeneration-success claim. This fixed-path smoke did not
inspect the appearance of images or the clinical correctness of report bodies.

Next: separately preflight/approve additional report experts and independent
endpoint scoring, then freeze the adaptive policy/cost evaluation. Keep these
underconditioned cases rather than enriching EHRs or quietly dropping them.

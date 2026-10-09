# Online report escalation / 在线报告切换：已准备，未提交

Update: the user subsequently approved the displayed full script with `go`.
Submitted job **12657042**; see
`docs/online_report_smoke_submission_12657042.md`. The preparation record below
describes the earlier pre-submission state, not a GPU completion result.

This is an unsealed progress note, not a generation result or submission
authorization. Frozen execution protocol: `docs/online_report_smoke_protocol.md`.

## 本轮完成 / Completed preparation

New implementation: `TriCompose-v1.2/tools/online_report_smoke.py`.
New tests: `TriCompose-v1.2/tests/test_online_report_smoke.py`.
New batch script: `TriCompose-v1.2/slurm/49_online_reports2_v100.sbatch`.
Historical workers, consumed protocols, EHRs, prompts, scorer thresholds,
candidate outputs and previous selections are unchanged.

The full V1.2 suite passed **1,707 tests in 10.507 seconds**, including **24 new
invented/mock-worker tests**. They exercise prospective call ordering, initial
and continuation ledger binding, failure charges, null case retention,
unknown-safe routing, the separate fresh-output gate, online-before-shadow
sealing, static-only observation isolation and post-selection endpoint scoring.
These software tests do not establish clinical accuracy, runtime GPU
compatibility or generation quality. No model inference was run.

CPU preparation in existing Slurm allocation **12654973** authenticated the
old EHR-only-stratified two-case development cohort, unchanged source requests
and frozen local assets. The new report expert was preflighted without
instantiating a model factory. Checkpoint byte hashing was performed in this
allocation, not a login node. No raw source patient rows/reports/image pixels
or generated clinical bodies were parsed during preparation.

Sealed plan:

```text
artifacts/protected/tricompose_v1_2/online_report_plans/online2_12654973_001/
  plan.json
  manifest.json
```

Manifest SHA256:
`fca888e160ce692eaf100aa5cddf3aabd504c283a1b17e35ce6bf64c628dafac`.
Controller SHA256:
`5f9f2d1f7d90d8e686973e8afe160a989ec83d8d9024564dcb9674c214389cfc`.
Do not edit plan-consumed implementation/tests/protocol after sealing. A change
requires a new plan and run, never replacement of this plan.

A second CPU-only load using the actual controller Python environment verified
all **243 pinned source files**, input/checkpoint asset pins, exact policy,
the private plan modes and unchanged historical retry plan/run/audit manifest
hashes. Controller/tests/protocol hashes still match the sealed plan. The new
batch script passed `bash -n`; Python AST and whitespace checks passed.

## 将运行什么 / Planned execution

Two fixed synthetic DEVELOPMENT EHRs, not an untouched test cohort:

```text
fixed EHR-derived prompt -> RoentGen-v2, seed 2 -> XRV
                                             -> CXRMate-single -> CheXbert
online policy: stop OR request CheXagent-2 -> CheXbert -> fresh-output veto
static control: always request second expert -> same fresh-output veto
seal choices -> frozen full-report BioViL-T secondary measurement
```

RoentGen consumes EHR-derived radiology text, not native structured EHR.
CXRMate-single is not CXRMate-ED; both report experts consume the generated
image. All existing generators and verifiers remain frozen. No training,
finetuning, new scorer, new router, external API or download.

Maximum primary acquisition: **2 images, 4 reports, 2 XRV operations, 4 CheXbert
operations = 12 charged attempts** total. At most six per EHR, zero operation
retries. There is no image-regeneration branch. Direct EHR/image proxy conflict
stops unresolved rather than being declared a clinically localized error.
Operational failure remains charged and retains its EHR with null/fallback
output; it is not converted into clinical disagreement or dropped.

Fixed-path normally consumes four attempts/EHR. Online consumes four or six.
Always-second static consumes six. All share the same six-attempt maximum,
not necessarily the same expenditure. If online stops, its decision is sealed
before the paid shadow static report is generated. The latter cannot alter
the online decision. Total collection still pays for both controls: **actual
GPU savings are not demonstrated** by this test. At most two additional image
and four text encoder calls are reserved separately for BioViL; failures and
overlength reports retain NA, without silent truncation or extra generation.

## GPU 申请与输出 / Resources and outputs

Prepared request: account `ruishanl_1185`, partition `gpu`, one node/task,
one `v100`, four CPUs, **48G host RAM**, **00:30:00 maximum allocation time**,
no array/node pin. Frozen worker planning minimum is 24 GiB GPU VRAM; CARC lists
V100 at 32GB. This is not a measured peak requirement or compatibility
certification. October 4 resource snapshots show several V100 slots while
several A40/A100 nodes are drained; availability can change before submission.

After approval, the script creates new job-keyed, exclusive directories:

```text
artifacts/protected/tricompose_v1_2/online_report_runs/online2_<job_id>/
  cases/<opaque_case>/inputs/              # fixed canonical EHR/facts/prompts
  cases/<opaque_case>/operations/          # new CXR/report/verifier artifacts
  cases/<opaque_case>/online_selection.json
  selection.json
  score_rows.json
  completed_triplets.json
  execution_summary.json
  endpoint_sidecar.json
  method_comparison.json
  method_comparison.csv
  summary.json
  controller.journal.jsonl
  manifest.json
artifacts/protected/tricompose_v1_2/online_report_audits/online2_<job_id>/
artifacts/protected/tricompose_v1_2/job_runtime/online2_<job_id>/
```

Three-edge readouts keep explicit fact denominators and missingness; no
unjustified unified clinical score. The method CSV compares fixed, online and
static choices, charged attempts, raw-edge readouts and secondary cosine.
An in-job metadata audit reconstructs receipts/decisions/costs without opening
report bodies or image pixels. Audit success is software/lineage validation,
not clinical acceptance or repair success.

Protected directories/files use project-group modes 2770/0660. Framework
logs/caches/temp remain private inside the workspace. The script refuses
existing runs, enforces offline model use and normalizes only its new runtime.
Public stdout is sanitized status only. Before submission, show the complete
actual script/resource request and obtain **subsequent explicit approval**.
No new GPU job has been submitted during this preparation.

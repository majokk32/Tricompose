# Authored opacity stages V2: prepared, not submitted

Following the unqualified 3/6 authored-control result of job 12675347, created
new versioned files only. The consumed V1 prompt/worker/tests/protocol/plan/run,
synthetic EHR/image/report artifacts, old scores/winners and teammate edits
remain unchanged. No new model inference, API, download or sbatch performed.

## Prepared implementation

- `TriCompose-v1.2/interfaces/opacity_assertion_stages_v2.py`: mechanical source
  segments, separate evidence-locator and polarity messages/strict decoders,
  deterministic four-state reducer and quote-free source references.
- `TriCompose-v1.2/benchmarks/opacity_assertion_controls_v2.py`: **48** distinct
  authored development texts, **12** families of four, including all six known
  V1 controls verbatim. Expected states: **9 positive / 9 negative / 16 uncertain
  / 14 unknown**. These are not clinical gold or an untouched evaluation set.
- `TriCompose-v1.2/tools/benchmark_opacity_assertion_stages_v2.py`: metadata/CPU
  preparation and separately approved paired frozen-model diagnostic. Compare
  unchanged V1 on the same 48 inputs; never read the synthetic candidate bank.
  Fsync/hash predictions before parsing authored reference semantics.
- `TriCompose-v1.2/tests/test_opacity_assertion_stages_v2.py`: **45** new CPU
  tests. Injected oracle answers test contracts, not model accuracy.
- Protocol: `docs/opacity_assertion_stages_v2_protocol.md`.

Completed inside existing CPU Slurm **12666569**. Full V1.2 regression:
**2,003 tests passed in 11.819 seconds**; `git diff --check` and batch syntax
check passed. Independent preparation audit verified **24 source pins**, three
plan artifacts, all 48 source hashes, independently rebuilt mechanical Unicode
offsets, separate keys, complete family/state inventory, exact six-control
preservation, cost cap and protected project-group **2770/0660** modes.

## Sealed protected plan

```text
artifacts/protected/tricompose_v1_2/opacity_assertion_stages_plans/authored48_stages_v2_12666569_001/
  inputs.json
  references.json
  plan.json
  manifest.json
```

Plan manifest SHA256:
`2e2f06e28ba7dc86245409618080ef78b7495061c4660b913174413cf90506c9`.
Worker SHA256:
`5277ab34e57070582c2f64293645ae8cf9fe9736c01a05c8a920d28787189e16`.
Complete batch SHA256:
`fe994540de1329033acd578eb120d95f17e161d89d7a7887da9aefd35735263d`.

These consumed code/fixtures/tests/protocol and plan are now immutable. The
plan is Git-ignored. Audit helper:
`.tmp/audit_authored_opacity_stages48_prepare_12666569.py`.

## Pending GPU approval

`noderes -f -g` showed one free debug A40 on `b11-09`; read-only scheduler
inspection confirmed two A40s configured, one allocated, 13 free CPUs and
approximately 34G schedulable host memory. Debug partition time limit is one
hour. This snapshot does not reserve a slot or promise a queue start.

Proposed script:
`TriCompose-v1.2/slurm/55_authored_opacity_stages48_a40_debug.sbatch`.
Request: account `ruishanl_1185`, **debug / one A40 / two CPUs / 32G host RAM /
15-minute upper cap**, no forced node. Choosing an A40 does not diagnose the
old V100 result; the paired methods use one identical frozen runtime in this
new job. Preserve the loader's dtype policy and report actual device/dtype.

Maximum **148 attempts**: 48 baseline, 48 primary locators, up to 48 primary
polarities and up to four calls for two fixed staged replays. Actual counts
must reflect skipped/failed stages. Empty completed locator -> unknown;
failed stage -> null unavailable, never negative or a successful unknown.

The narrow authored-language gate requires 48/48 staged states and selected
IDs plus two stable complete replays. Even a pass is not a primary clinical
score or authorization for selection/error localization/repair. A failure
does not justify modifying the sealed tests/prompt to report an improvement.

Future new protected output:
`artifacts/protected/tricompose_v1_2/opacity_assertion_stages_runs/authored48_stages_v2_<job_id>/`.
Show the **entire** script/resources and obtain explicit subsequent approval
before submitting. This preparation has not submitted it.

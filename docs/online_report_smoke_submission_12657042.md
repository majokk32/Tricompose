# Online report smoke: approved submission 12657042

Submitted on 2026-10-04 after the complete concrete batch script/resources
were shown and the user explicitly replied `go`. This approves only this
two-EHR report-escalation smoke and its static/fixed controls, not expansion,
new jobs, training or clinical acceptance.

Script: `TriCompose-v1.2/slurm/49_online_reports2_v100.sbatch`.
Script SHA256:
`f6b0bcb311ccd182596ce1d28569ac1d8b973d247c77ca11f89dd7d1a3d07fd4`.
Plan manifest SHA256:
`fca888e160ce692eaf100aa5cddf3aabd504c283a1b17e35ce6bf64c628dafac`.
Controller SHA256:
`5f9f2d1f7d90d8e686973e8afe160a989ec83d8d9024564dcb9674c214389cfc`.

Command: `sbatch --parsable TriCompose-v1.2/slurm/49_online_reports2_v100.sbatch`.
Returned Slurm job ID: **12657042**.
Immediately verified scheduler state: **PENDING / Priority**. Start time was
unknown; no inference-completion, queue-duration or quality claim is made.

Follow-up scheduler check: **RUNNING on d14-15**, elapsed 00:01:07 at that
snapshot. Scheduler allocated the approved V100/four-CPU/48G resources; the
new private job runtime directory had project/NFS group and mode 2770.
Generation/audit completion had not yet been observed at that snapshot.

Final status: **COMPLETED, exit 0:0, elapsed 00:09:16**. The in-job metadata
audit passed for four completed receipts. Two images/four reports were
generated; all twelve primary attempts completed. Both online cases stopped
unresolved before shadow controls; all methods retained the same two initial
outputs. No clinical repair or method advantage is demonstrated. Complete
aggregate evidence: `docs/online_report_smoke_result_12657042.md`.

Scheduler confirmed account `ruishanl_1185`, partition `gpu`, one node/task,
one V100, four CPUs, 48G host RAM and a 00:30:00 allocation limit, no
dependency/array/node pin. Existing unrelated jobs were not changed.
Pre-submission resource snapshot listed mixed V100 nodes with free slots and
several drained A40/A100 nodes; a free slot does not guarantee immediate start.
Syntax/script/plan/controller hashes were checked without running GPU work.

Expected exclusive outputs after execution starts:

```text
artifacts/protected/tricompose_v1_2/online_report_runs/online2_12657042/
artifacts/protected/tricompose_v1_2/online_report_audits/online2_12657042/
artifacts/protected/tricompose_v1_2/job_runtime/online2_12657042/
```

No raw or synthetic clinical content is included in this submission note.
See `docs/online_report_smoke_protocol.md` for the frozen experiment and
`docs/online_report_smoke_preparation.md` for preflight/testing evidence.

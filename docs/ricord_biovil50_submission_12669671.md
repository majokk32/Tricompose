# Approved RICORD BioViL-T opacity diagnostic / 已提交

On 2026-10-05, the full batch script and **one V100 / two CPUs / 8G host RAM /
five-minute upper limit** were displayed to the user. The subsequent explicit
`gogogo` approved this exact submission. Preparation alone did not authorize it.

Submitted once:

```bash
sbatch TriCompose-v1.2/slurm/52_ricord_biovil50_v100.sbatch
```

Slurm job **12669671**, account `ruishanl_1185`, partition `gpu`.
First check: **RUNNING**, V100 node **d13-04**, elapsed three seconds. This is
an observed initial state, not a completion result or queue-time guarantee.
Do not resubmit this historical script to the same run.

Exact submitted script SHA256:
`2347a238242054b933b78d0e72231948f0754f7a374b8c7ac36254dc575c10cd`.
Worker SHA256:
`e1b86f80412a528681d1455241063d3fb37d3eff188b4c40fc482ac42edca808`.
Plan manifest SHA256:
`91560c8407d638979d665e3fc4717dff80e68289e58d9036ce1258a8f96687e1`.

The script/worker/plan hashes and batch syntax were checked immediately before
submission. `noderes -f -g` showed multiple available V100s; the request did not
force a node or change resources. Initial sanitized log metadata showed
project-protected mode 0660; NFS exposed the project group as `nobody`.

Inputs are exactly the 50 existing hash-bound RICORD PNGs and six generic
opacity polarity probes. No raw MIMIC input, EHR/report text, training,
generation, new download, external API, threshold fitting, template selection
or historical winner change. Reference labels join only after encoding.
This remains a post-hoc DEVELOPMENT component diagnostic, not clinical repair.

Expected protected result:

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_biovil_pilots/ricord_biovil50_12669671/
  scores.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Job-local cache/log/temp:
`artifacts/protected/tricompose_v1_2/job_runtime/ricord_biovil50_12669671/`.
Sanitized Slurm logs:
`artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_ricord_bv50_12669671.out`
and its `.err` sibling. Native worker logs remain protected; do not copy them
wholesale or expose individual reference/prediction records.

After completion, audit Slurm exit/status, the complete 50-case denominator,
failures/ties, fixed-mean and class-specific metric algebra, all source/artifact
hashes, actual encoding/replay costs and project permissions before interpreting
results. Do not change the sealed worker/tests/protocol/plan to improve scores.
Any further submission needs another full-script/resource display and approval.

## Completion / 已完成

Job **12669671** completed on **d13-04**, V100, exit **0:0**, Slurm elapsed
**00:00:38**. Worker elapsed before serialization **23.220128 seconds**;
peak allocated CUDA tensor memory including load **0.608 GiB**, not total GPU
context/reserved memory. The five-minute request was only an upper limit.

All **50/50** fixed images scored, zero failures/replacements. Actual encoding
counts: **52 image calls** (50 primary plus the two predeclared replays) and
one **eight-input text batch**. Zero generation/training/new selector calls.
Duplicate text embeddings and the first-two image score replays have zero
maximum absolute delta; reproducibility is not clinical validation.

Independent post-run checks passed for all reference/display joins, all four
predeclared profiles, class-specific wins/recovery/coverage, tied-score AUROC
and AP using a separate implementation, mean-margin/XRV disagreement, actual
calls/replays, source/model/vendor/output hashes and protected permissions.
The checks reopened no source pixels, MIMIC rows, reports or EHR contents.

Completed result manifest SHA256:
`d50e5594e43f7979f385f0e14745d66103714b16693e7656ddcf7812e85480ef`.
Detailed derived metrics remain in the protected `RESULTS_CN_EN.md` above.
See `docs/ricord_biovil50_result_12669671.md` for scope/interpretation.
Historical choices and original runs are unchanged. Do not select the best
template or retroactively treat this post-hoc diagnostic as a final endpoint.

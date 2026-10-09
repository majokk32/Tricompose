# Opacity conflict registry: completed

Completed metadata-only inside existing CPU Slurm allocation **12666569**.
No model execution, GPU job, new submission, clinical-text/pixel inspection,
training, threshold/rule tuning, candidate filtering or old-winner modification.

## Output

```text
artifacts/protected/tricompose_v1_2/candidate_conflict_registries/opacity_conflicts_12666569_001/
  candidate_registry.csv
  image_registry.json
  report_registry.json
  case_registry.json
  verification_requests.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`87674da73de24b840f4baf08a1cd91fcd98e6d5f0d687026ab47ed3ec6b0600d`.
Worker SHA256:
`301df76ba5d52c9a14e75fc6d1920d21f481a4139b4b55448f23787c9f658a28`.
The worker, 31 new fixture tests and protocol are now source-bound and consumed;
do not edit them or existing protected results.

## What the interface now provides

All **80 fixed EHR / 240 image slots / 960 candidate slots** retained. The
projected registry binds each source row digest and original named lineage;
the completed 86-column evidence table remains untouched and is referenced
by hash. Image-model IDs come from bound outcomes, not parsed filenames/IDs.

| Dependency pattern | Candidate slots | EHRs in this pattern | Image slots | Unique report texts |
| --- | ---: | ---: | ---: | ---: |
| Image evidence unavailable | 0 | 0 | 0 | 0 |
| Image scorers disagree | 224 | 41 | 56 | 143 |
| Image scorers agree, report assertion unavailable | 541 | 76 | 184 | 196 |
| Three proxy proposals agree | 159 | 62 | 93 | 94 |
| Report proposal opposes both agreeing image scorers | 36 | 20 | 29 | 16 |

The candidate-slot column sums to 960; unique image/report/EHR columns do not:
their sets overlap across patterns. None is an adjudicated fault label or a
clean clinical control. The whole bank contains **428** distinct report-byte
hashes, with **130** reused across image slots and **21** used across different
dependency patterns. Exact duplicates can be legitimate repeated wording;
reuse alone does not prove collapse, a factual error or poor report quality.

The 36 report-opposition slots become **16 text-only assertion-check requests**
retaining all 36 source contexts. Cached report proposals there are 30 positive
and six negative; these are proposals, not verified text states. Image scorer
disagreement becomes **56 image-slot evidence-check requests**, not 224 image
calls. Eleven images retain a separate mixed-template-sign flag; it is not a
calibrated confidence threshold or a repair trigger.

Requests have opaque IDs/hashes and source-context joins only. They expose no
cached states/scores/models as verifier input, contain no clinical bodies or
ready execution paths, and are marked **metadata_only_unmaterialized** with
zero completed/model calls. Investigator registries do retain model/proxy
metadata; this is input separation, not access-isolated clinical blinding.
No new selection, acceptance, clinical accuracy or repair-success score exists.

## Tests and independent audit

- **31** new invented-fixture tests pass.
- Full V1.2 regression: **1,928 tests pass in 11.834 seconds**.
- Independent standard-library audit verified **567 source pins**, seven
  result artifacts, all 960 source-row digests/joins, all 240 image/428 report/
  80 EHR inventories, exact content-group membership, multiplicities, per-model
  denominators, request dedup/context reattachment, all patterns and flags,
  immutable old sources and protected group modes **2770/0660**.
- Protected outputs are Git-ignored; `git diff --check` clean.

## Next automatic step / 下一步

先自动核验这 16 份文本到底明确表达 positive、negative、uncertain 还是
unknown，检查模型标签提取是否把否定或不确定表达读错；同一文本的结果应
复用到所有原图像上下文。此步骤判断文字说了什么，不判断图像或患者事实。
随后再独立核验 56 张评分器意见不同的图像，保留未知与模型相关性。

These are developmental investigation subsets, not representative clinical
accuracy samples or a new untouched benchmark. Do not tune a repair policy to
these observed patterns then claim independent validation. Human feedback is
not a prerequisite for the exploratory automatic track, but model consensus
does not replace independent clinical evidence. New frozen-model inference
requires its own complete Slurm script/resources and explicit approval.

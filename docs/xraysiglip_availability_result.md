# Full-bank alternate readout handoff / 全候选辅助证据表

The [append-only protocol](xraysiglip_availability_protocol.md) is implemented
by `TriCompose-v1.2/tools/attach_xraysiglip_availability.py`. The completed
XraySigLIP job 12654030 is now available beside all original candidate scores,
without replacing any score or changing an EHR, winner, priority or action.
Execution used existing CPU Slurm allocation **12645021**. There were zero new
model calls, GPU jobs, Slurm submissions, downloads, training or API requests.
Only named bounded metadata was read, never source EHR/report/image bodies,
real targets, weights or recursively followed parent source paths.

## Where to look

```text
artifacts/protected/tricompose_v1_2/xraysiglip_availability/siglip_pool960_12645021_001/
  candidate_score_table.csv
  fact_verification_availability.jsonl
  unique_image_finding_table.jsonl
  supplemental_image_evidence_requests.jsonl
  evidence_request_frontier.jsonl
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`764a114166a817c06c9a36d4b50265db35764da7d324b2f68407e5bf04112a0e`.

`candidate_score_table.csv` has **960 candidates / 80 fixed EHRs / 240 image
slots**, all **162 original columns** plus **14 `siglip_` availability columns**.
The finding JSONL contains each raw three-template cosine margin, its mean and
range, preference, exact guard/source hashes and unqualified Qwen/XRV direction
comparisons. The image-only readout does not validate its report or an EHR edge.

| Scope | Count |
|---|---:|
| Checked unique images | 6 |
| Original report consumers of those images | 24 |
| Candidates still not checked | 936 |
| Unique image/finding inventory | 84 |
| Supported unique image/finding readouts | 48 |
| Unique template-sensitive readouts | 25 |
| Unique consistent presence-text preferences | 6 |
| Unique consistent absence-text preferences | 17 |
| Supplemental logical requests | 21 |
| Supplemental candidate/finding links | 84 |

The full 13,440-row finding inventory remains intact: 100 template-sensitive
occurrences, 92 stable-text-preference occurrences, 144 outside-scope
occurrences and 13,104 not-checked occurrences. The first two counts repeat
image readouts across four report consumers; they are **not** independent votes.
Across the 21 disputed requests, 14 are template-sensitive and seven prefer
absence text; none are clinically resolved. Original execution status remains
`not_executed` because independent clinical evidence was requested, not merely
another automatic similarity score.

**中文：新表能查看已有评分和这次补充读数，但它不是新的临床总分或择优结果。
未检查的数值保持空值，unknown/uncertain 不变阴性。临床状态、概率及临床
择优分数仍为空；没有开启自动定位或返工。SigLIP 与 CheXagent-2 共用视觉
编码器，不能作为该报告模型的独立最终评估。**

## Verification and preservation

- **1,494 tests pass**, including 20 new invented-metadata attachment tests.
- Rechecked **33 consumed source/metadata hashes** and **seven artifact hashes**.
- Preserved **155,520 original candidate cells**, **752,640 original finding
  fields**, **672 original supplemental request fields**, row order and scopes.
- The entire **2,072-request original history is byte-identical**.
- Independent post-run metadata replay produces byte-identical candidate,
  finding, unique-image/finding and supplemental-request tables.
- Exact candidate/case/EHR/facts/image identities and guard receipts were
  checked; equal image hashes did not propagate evidence to unchecked IDs.
- Six historical attempts/forwards and four historical guard blocks are kept
  separate from zero new calls. Clinical resolution and regeneration remain
  false; old scores and missingness are unchanged.
- Output directories are 2770/files 0660 with project GID 65534 on this NFS
  mount. No raw synthetic report text or images appear in this aggregate note.

This is an evidence-availability deliverable, not a new clinical benchmark,
accuracy result, successful self-correction or improvement claim. The sealed
run, worker, tests and protocol must not be edited to fit a desired outcome.

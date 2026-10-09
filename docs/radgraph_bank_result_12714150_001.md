# Full synthetic-bank RadGraph extraction and cross-path agreement

## Result / 结果

Actual frozen RadGraph-XL inference completed for **428/428 distinct synthetic
report texts**, covering **960/960 candidate slots** from the historical
80-EHR bank. All **1,440 same-image report pairs** have official graph-reward
values. This adds measured diagnostic columns, not empty placeholders, while
preserving all old score values, candidate ordering and historical winners.

全池已运行：428 份不同报告全部完成提取，960 个候选都有新列，1,440 对同图
报告完成图谱 agreement。旧评分和 80 个历史选择均保留。新指标表示不同报告
描述的一致程度，不是 CXR–Report 临床准确率，也不触发自动修复。

See `radgraph_bank_protocol.md` for the frozen scope. No real EHR/report/image
input, real target, new generation, training, API or new Slurm submission was
used. The actual CPU allocation was 12714150. Both graphs and scoring outputs
remain protected and Git-ignored.

## Execution

| Item | Actual value |
|---|---:|
| Fixed synthetic EHR cases | 80 |
| CXR candidate slots | 240 |
| Report candidate slots | 960 |
| Unique report forward passes | 428 |
| Graph extraction failures / empty inputs | 0 / 0 |
| Same-image pairs | 1,440 |
| Identical report byte pairs within the same image | 0 |
| CPU model initialization | 4.020 seconds |
| Report inference | 55.728 seconds |
| Total including source/hash checks and exports | 67.401 seconds |
| Process peak RSS | 1.660 GiB |

Repeated report byte hashes reuse a native graph without dropping their
candidate slots or lineage. An extraction marked complete is an engineering
status, not proof that all clinical facts were extracted correctly. Graph F1
uses the unchanged official reward implementation, not a new learned scorer.

## Same-image agreement: descriptive, not an accuracy ranking

Each row below averages the 240 original image **slots**, retaining all cases.
The source has only 147 distinct image hashes and 49 exact three-model prompt
groups; these 240 slots must not be treated as 240 independent test patients.
This table makes no confidence-interval or clinical-superiority claim.

| Report-model pair | Entity F1 | Relation-presence F1 | Full-relation F1 |
|---|---:|---:|---:|
| CheXagent-2 / CXRMate-single | 0.1839 | 0.1463 | 0.1108 |
| CheXagent-2 / LLaVA-Rad | 0.2551 | 0.2308 | 0.1743 |
| CheXagent-2 / MAIRA-2 | 0.0798 | 0.0632 | 0.0479 |
| CXRMate-single / LLaVA-Rad | 0.2392 | 0.2128 | 0.1721 |
| CXRMate-single / MAIRA-2 | 0.0379 | 0.0364 | 0.0269 |
| LLaVA-Rad / MAIRA-2 | 0.0679 | 0.0577 | 0.0243 |

These relatively low lexical graph agreement values are not evidence that a
particular model is clinically wrong. Exact-token graph matching is sensitive
to description length, entity wording and extraction mistakes. Shared image
dependence, incomplete entity extraction, scope and synonymous findings remain
unresolved. A model disagreeing with the others can be the correct one; no
majority or higher-F1 report is promoted to clinical truth or a new winner.

The next meaningful validation is a documented independent clinical benchmark,
not maximizing these peer-agreement values. The earlier official manual-report
pilot and the unresolved RadEvalX blank-annotation sensitivity retain their
original, limited roles. This run does not solve EHR-edge missingness, acquire
new radiographic EHR facts, or validate modality error localization.

## Where to find the table

```text
artifacts/protected/tricompose_v1_2/radgraph_bank_runs/
  native_pool960_12714150_001/
    candidate_score_table.csv     # 960 old rows/columns + appended RadGraph columns
    same_image_agreement.csv      # 1,440 model-pair comparisons
    report_graph_index.csv        # 428 distinct report graph statuses/counts
    graph_receipts.json           # metadata, label counts and hashes
    native_graphs.json            # sensitive synthetic graph text; protected only
    frozen_plan.json
    summary.json
    manifest.json

artifacts/protected/tricompose_v1_2/radgraph_bank_audits/
  native_pool960_12714150_001/audit.json
```

The three per-candidate `radgraph_peer_mean_*` columns average the other three
same-image reports. `radgraph_available_peers` supplies their denominator.
Entity/relation counts are extraction diagnostics, not clinical quality.
Empty/unavailable inputs would remain null with explicit status; valid native
zero reward is retained. All old EHR–CXR, EHR–Report, CXR–Report and BioViL
cells are copied unchanged. No synthetic report or graph text belongs in Git,
public README, public logs or chat.

## Integrity evidence

Run manifest SHA256:
`ebc747392d791595901997af38169c0c95944e5b79786876d6c8db9beac295b6`.
Independent audit SHA256:
`d425a731cfbd0f0e633b7f5ff5f273b4e81291aa414be2559d2bb97b86adfca6`.

The separate audit passed:

- 977 source/code/output hash checks;
- all 960 original rows and column values preserved;
- all 1,440 pair identities reconstructed as within-image expert combinations;
- 2,140 native graph metadata checks;
- 4,320 reward components recomputed using unchanged official reward code;
- 2,880 per-candidate peer means independently recomputed;
- 11 protected run entries with 2770 directory / 0660 file permissions.

V1.2 fixture suite: **2,325 tests passed**, 14.868 seconds. Fixtures are invented
and do not establish clinical accuracy. The official checkout remains clean;
no downloaded model source/checkpoint is added to project Git. No old first-
version handoff, baseline or protected run was overwritten.

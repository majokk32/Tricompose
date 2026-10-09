# RadEvalExpert / RadGraph result — 2026-10-06

## Main conclusion / 主要结论

The local frozen RadGraph implementation ran successfully, but its pooled
correlation with expert clinical-error burden is weak. It must **not** become
the sole clinical judge, a calibrated repair trigger, or an agent reward on
this evidence.

官方冻结 RadGraph 已实际跑完并通过独立复算，但整体分数与专家临床错误
数量的相关性接近零。它可作为结构化报告相似度的辅助信号，不能据此
决定 CXR 或 Report 谁错了，也不能直接驱动自动修复。

A clearly labeled **post-hoc** same-anchor selection diagnostic finds a
modest reduction in expert error counts against uniform random choice. That
is reference-based report reranking, not proof of reference-free TriCompose
selection or successful targeted regeneration. Both findings are retained.

## Execution and coverage

Official source: [RadEvalExpertDataset](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset),
linked by [RadEval](https://github.com/jbdel/RadEval) and associated with the
[EMNLP 2025 demonstration paper](https://aclanthology.org/2025.emnlp-demos.40/).
This is not the older RadEvalX cached-score sensitivity experiment.

- Fixed dataset revision `b4bd9d6ee75fcd155b6de466f1b7b7bd774408be`;
  691,815 bytes; exact author Git blob verified.
- 208 source rows, 624 reference/candidate pairs, 762 distinct report texts.
- **624/624** metric pairs completed; **762/762** native graphs completed.
- 623/624 significant-error totals available; 622/624 all-error totals.
  Two malformed source count cells remain unavailable, never zero-filled.
- Two reader identities in the release, but one released rating per exact
  pair. Inter-reader agreement cannot be calculated from this as if every
  pair had two ratings.
- 180 dependence clusters: 77 MIMIC patient-folder keys, 31 CheXpert
  patient-folder keys, 72 unrecognized author source keys. Those 72 are not
  verified patient identities; full patient independence is unproven.
- Fixed official RadGraph 0.1.18 / XL, CPU, two threads, frozen parameters,
  offline tokenizer, blocked sockets during scoring. No model download,
  training, external clinical API, new sbatch, image/EHR read or generation.
- CPU runtime **92.315 seconds**, including 80.191 seconds of neural inference;
  peak process RSS **2.636 GiB**. Five nonempty inputs yielded zero-entity
  graphs; the official zero reward is retained, not mislabeled model failure.

The existing 960-candidate synthetic table and historical choices are unchanged.

## Predeclared primary endpoint

Correlation is between higher-is-better metric score and **negative** expert
error count: positive would mean higher score corresponds to fewer errors.
All three official components are reported, without metric-weight fitting.
Intervals use 1,000 whole-cluster bootstrap resamples, seed 0.

| Official component | Significant-error Spearman rho | 95% cluster interval | All-error rho |
| --- | ---: | --- | ---: |
| Entity F1 | 0.0447 | [-0.0685, 0.1470] | 0.0306 |
| Relation-presence F1 | 0.0463 | [-0.0637, 0.1478] | 0.0289 |
| Full-relation F1 | 0.0370 | [-0.0709, 0.1323] | 0.0150 |

All three primary intervals include zero. Section-specific descriptive
correlations are also in the protected evaluation, not used to pick a favorable
scope or redefine the primary endpoint. Weak aggregate performance does not
establish that every individual graph or generated report is incorrect.

## Post-hoc within-anchor choice diagnostic

This diagnostic was added **after** seeing the primary result. It is not a
replacement primary endpoint or independent confirmatory test.

Each released source/section/reference has exactly three candidate slots.
Compare analytical uniform random choice, each fixed source slot, score-max
choice, and minimum-expert-error oracle. Maximum-score ties are uniformly
averaged, not resolved by favoring the first candidate. All three error counts
must be available for an anchor; no poor candidate is selectively dropped.

207/208 anchors (179 dependence clusters) have complete significant counts.
Uniform choice has mean **4.0837** significant errors; fixed slots 1/2/3 have
**4.1111 / 4.0870 / 4.0531**; the expert oracle has **2.7826**.

| Component used to choose | Mean significant errors | Difference vs uniform random | 95% cluster interval | Strict pairwise accuracy* |
| --- | ---: | ---: | --- | ---: |
| Entity F1 | 3.7552 | -0.3285 | [-0.5067, -0.1128] | 55.19% |
| Relation-presence F1 | 3.8068 | -0.2770 | [-0.4610, -0.0510] | 53.35% |
| Full-relation F1 | 3.8116 | -0.2721 | [-0.4343, -0.0968] | 53.14% |

*462 candidate comparisons have unequal expert error counts. Metric-score
ties contribute expected 0.5 correctness and are explicitly counted, not
discarded. Oracle-hit probability uses all tied minimum-error candidates;
its random baseline is about 50.08%, **not automatically 1/3**.

The secondary all-error diagnostic retains 206 anchors: random mean 4.8236,
score-chosen means 4.4612 / 4.5097 / 4.5728. All components and confidence
intervals remain in the protected summary. No component is promoted because
one number happens to look best.

Interpretation: modest **reference-based** within-case ranking usefulness can
coexist with poor absolute cross-case calibration. Ground-truth reference
reports are available here; fully synthetic inference does not have them.
Replacing the reference with other generated reports changes the problem and
requires independent validation. This result does not qualify report consensus
as image truth or establish fault localization.

## Artifacts and audits

All paths below are workspace-relative and protected:

```text
artifacts/protected/tricompose_v1_2/
  report_metric_sources/radeval_expert_source_12714150_001/
  radeval_expert_radgraph_runs/expert_radgraph_12714150_001/
    score_table.csv
    evaluation.json
    summary.json
    manifest.json
  radeval_expert_radgraph_audits/expert_radgraph_12714150_001/audit.json
  radeval_expert_selection_runs/within_anchor_12714150_001/
    replay.json
    summary.json
    manifest.json
  radeval_expert_selection_audits/within_anchor_12714150_001/audit.json
```

Text-bearing native graphs and the original CSV are protected internal artifacts;
do not print them into chat, Git, README, or public logs.

- Main receipt SHA256:
  `21e1724a5b364eee97b56cdb8f66c0d56194baadc98c333b5c473f7efb37038d`.
- Main audit passed: 17,472 independent source/aggregate category checks,
  1,872 official reward replays, 3,048 graph metadata checks, all 624 CSV rows,
  six correlation/cluster-CI replays, source hashes and protected permissions.
- Main audit SHA256:
  `b37e28de6cbad22a615e2ee03a88c5ac913ce5b74572a608fd46b77c609f7712`.
- Within-anchor audit passed: all 1,248 diagnostic records, six summaries and
  independently reimplemented cluster intervals, hashes and permissions.
- Within-anchor receipt SHA256:
  `28dbe76229529054efae50438feedb346eea741c38ce5ba92597c9d7cc38850e`.
- Within-anchor audit SHA256:
  `420b70f24e5abb591163bc67625557703e1bb12c0da28670d303d90b99ad6063`.
- Full V1.2 suite: **2,365 tests passed** in 15.690 seconds, including 40 new
  invented-schema/selection tests. Git whitespace check passed; raw CSV and
  native graphs are ignored by the project Git wrapper. Test success is an
  engineering check, not additional clinical accuracy evidence.

## Next evidence-producing step

Keep RadGraph as an auxiliary structural signal. Next assess an already frozen
**image-grounded** verifier against independently annotated errors, with a
separate fixed protocol and approved patient-input computation. First establish
image/source linkage and licensed local availability; do not silently fetch
real images or call external clinical APIs. Report within-case ranking,
contradiction sensitivity, abstention and coverage before integrating repair.
No fitted scorer, new generator, EHR replacement or favorable-case filtering
is justified by this benchmark. Checkpoint training overlap remains unverified.

# Score-free final-choice benchmark / 真实随机选择对照

Completed inside existing CPU Slurm allocation **12654973**, following the
[separate frozen protocol](score_free_random_control_protocol.md). This is an
already inspected DEVELOPMENT bank, not an untouched confirmatory cohort.
No model/GPU/API/download/Slurm submission, training, source EHR/report/image
body access, new generation or old winner change.

## What was actually compared

All **80 fixed EHRs, 240 image slots and 960 cached triples** stay unchanged.
The old `random` policy acquired candidates in a randomized order but selected
the final candidate with the original score rule. The new
`random_acquisition_score_free_final` retains each old randomized acquisition
trace and its exact cost, then chooses uniformly among observed candidates
passing the same artifact gate. Its selector accepts **only IDs and gate
booleans**, not clinical scores, model names, runtime or alternate endpoints.

Five acquisition seeds × five independent final-choice seed settings × five
caps × 80 EHRs produce **10,000 cached control trials**. These are not new
patients, generated images or actual calls. Average all settings within each
EHR first. The control retains scorer costs even though its final selector
does not use the scores: this isolates **final selection only**, not an
optimized cheap random-generation pipeline or actual GPU savings.

BioViL-T endpoints and raw three-edge count readouts are joined after the IDs
are chosen. No score availability affects selection. The existing fixed,
random-acquisition/scored-final, static and targeted endpoints are retained.
All methods are evaluated through the same immutable full-bank cache, without
changing their historical ranking or selecting a best new seed.

## Full-cap overview

All rows below use cap 30, but their actual **simulated expenditure differs**.
BioViL is a secondary global retrieval readout, not clinical correctness.
Raw edge results are fourteen-head XRV/report-label proxies, not image truth.
Coverage and opposition denominators are explicitly known proxy findings.

| Method | Mean simulated calls | BioViL mean, 80/80 EHRs | CXR–Report support / known | Opposition / known | Comparable coverage / known |
|---|---:|---:|---:|---:|---:|
| Fixed Sana + MAIRA-2 | 4.000 | 0.6186 | 0.0865 | 0.0042 | 0.0906 |
| Random acquisition + score-free final choice | 30.000 | 0.5082 | 0.1551 | 0.1035 | 0.2586 |
| Old random acquisition + scored final | 30.000 | 0.6122 | 0.2365 | 0.0000 | 0.2365 |
| Static reranking | 30.000 | 0.6122 | 0.2365 | 0.0000 | 0.2365 |
| Targeted heuristic | 28.825 | 0.6075 | 0.2271 | 0.0000 | 0.2271 |

The old random/scored and static choices are identical on all 80 EHRs at full
inventory; those are not independent replications. The selector-only contrast
has exactly equal acquisition/expenditure: scored final minus score-free
final mean BioViL **+0.1041**, with **63 EHRs higher, 17 lower, zero ties**.
These paired counts average all seed settings per EHR and are descriptive,
not a significance test. Zero proxy opposition has only **23.65% comparable
coverage**, not complete consistency or 100% clinical accuracy.

**中文结论：满预算下，原分数择优比真正随机选最终候选更有用；但还没有证明
定向策略优于静态择优，固定路径的辅助图文均值也并不差。现在的证据支持
“筛选有作用”，不是“定位错误并修复已成功”。**

## Do not hide lower-cap results

The same-acquisition selector-only comparison is not uniformly favorable:

| Cap | Scored-final BioViL | Score-free-final BioViL | Mean simulated calls in both |
|---|---:|---:|---:|
| 4 | 0.5028 | 0.5028 | 4.000 |
| 8 | 0.5114 | 0.5312 | 7.570 |
| 12 | 0.5107 | 0.5155 | 11.380 |
| 20 | 0.5854 | 0.5093 | 19.955 |
| 30 | 0.6122 | 0.5082 | 30.000 |

At cap 4 there is only one acquired candidate, so equality is expected. The
two lower nontrivial caps favor score-free choice on this alternate readout.
Static's mean is 0.6702 at cap 20 and 0.6122 at cap 30; more inventory/optimized
proxy search does not guarantee a better alternate endpoint. Do not tune a
budget or threshold on these outcomes and label it independent confirmation.

## Three-edge interpretation

`method_comparison.csv` retains all 75 method/cap/scope rows, each raw edge's
known/comparable/support/opposition and positive/negative support counts,
rates and available-EHR denominators. There are 2,000 case/method/cap means
and 60 paired method/cap/scope contrasts. All cases remain.

The strict cached direct-EHR coverage is **8/80**, while **72/80** have no
comparable cached direct radiographic finding. Both EHR-edge readouts therefore
have only eight available EHR denominators; unavailable edges stay NA. This
is not a statement that those EHRs have no disease or useful clinical context,
and is distinct from the earlier 15/65 prompt-conditioning tier. No facts,
pathology, devices or negatives were added, and no difficult case was removed.

Reports generated from one image and repeated model/seed trials are correlated.
XRV/CheXbert, global BioViL and their agreement do not become independent
clinical adjudication. These metrics measure proxy compatibility and missingness,
not confirmed natural error localization, clinical repair success or the
reality of a synthetic patient.

## What follows

Freeze any new dependency/uncertainty-aware stopping or repair policy separately;
compare it with fixed, true random, static and current targeted choices at
explicit budgets, with support **and** coverage retained. Preserve the fixed
EHR, partial/unresolved outcomes and actual-call/failure ledger. Do not silently
replace the old results or optimize an alternate evaluator on this inspected
bank. Clinical fault claims still need qualified independent evidence.

## Output and verification

```text
artifacts/protected/tricompose_v1_2/automatic_replays/score_free_random_12654973_001/
  control_outcomes.jsonl
  case_means.csv
  method_comparison.csv
  paired_case_comparison.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124`.
Analysis before serialization took **1.078405 CPU seconds**, not model/GPU
runtime. Private directories/files are 2770/0660 in the CARC project boundary.

The post-run check validated **ten source hashes, six artifacts, all 10,000
domain-separated random draws, all 10,000 unchanged acquisition cost receipts,
fixed-EHR/observed-only lineage and exact output/table/report replay**. All
2,000 case means, 75 comparison rows, 60 paired rows and full-budget old
random/static endpoint equality passed. Old sources/winners remain unchanged.

**19 new invented-fixture tests passed; the full V1.2 suite passed 1,563 tests
in 10.103 seconds.** The sealed worker/tests/protocol are not changed after
execution. Tests and cached replay checks are not clinical validation.

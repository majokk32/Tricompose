# Fixed-bank opacity evidence overlay: completed

This is a **post-hoc development diagnostic**, not an independent final clinical
endpoint, new total-triple score, fault verdict or reranked winner table.

## Output and provenance

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_biovil_runs/biovil_opacity240_12670345/
  image_scores.json
  candidate_evidence_table.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result manifest SHA256:
`6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828`.
The immutable worker/protocol/plan are unchanged. Inputs remain the complete
**80 EHR / 240 synthetic image slots / 960 correlated candidate rows**.
All 66 old named fields and **63,360** old cell values are preserved; **20**
finding-evidence fields appended, **86** columns total. Field order may differ
because frozen JSON dictionaries were serialized with sorted keys: consumers
must join by names, not assume positional columns.

User-approved job **12670345** completed on V100 `d13-05`, exit `0:0`, Slurm
elapsed **41 seconds**. Worker elapsed before serialization **23.180682 seconds**;
peak allocated CUDA tensor memory **0.608 GiB**, not total device memory.
240/240 primary images scored; zero failed/replaced. 242 image encodings
(including two predeclared replays), one eight-input text batch. Replay and
duplicate-text deltas zero. No training/generation/new XRV/selector calls.

Independent standard-library audit verified **559 source pins**, four result
artifacts, original values/row order, all 960 lineage joins, all six cosine
values and three margins/image, fixed means/states/relations, full denominators,
replays/cost counters, hashes and group modes **2770/0660**. No clinical-text
body or image pixel was read for this audit. Protected output is Git-ignored.
Post-run full V1.2 regression: **1,897 tests passed in 11.926 seconds**.

## What was measured

Frozen BioViL-T compares each image with six generic lung-opacity assertions,
forming three positive-minus-negative cosine margins. Use the unchanged mean
of all three: positive/negative signs indicate preference, zero stays unknown.
No favorable-template selection, calibration, threshold fitting or probability
interpretation. No generated report text is encoded. Cached CheXbert report
states are separate, still unqualified assertion proposals.

Two image sources agreeing does not establish truth: both consume the same
image. Report-label extraction errors and classifier/preprocessing/domain bias
remain possible. The 50-case bounded RICORD development result is not validated
transport to this synthetic pool or a qualification of all findings.

## Proxy agreement, always with coverage

| Image proposal vs cached report proposal | Support | Opposition | Not comparable | Comparable coverage | Agreement on comparable rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact XRV opacity at fixed 0.5 | 173 | 95 | 692 | 268/960 = 27.92% | 173/268 = 64.55% |
| BioViL-T fixed three-template mean | 218 | 50 | 692 | 268/960 = 27.92% | 218/268 = 81.34% |

The numerical difference is scorer-dependent proxy agreement, **not an image
or report quality improvement**: the generated artifacts are unchanged.
Unknown/uncertain proposals do not become negative, agreement or contradiction.

| Report expert | Support | Opposition | Not comparable | Coverage | Conditional proxy agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| CheXagent-2 | 96 | 16 | 128 | 112/240 = 46.67% | 85.71% |
| CXRMate-single | 41 | 13 | 186 | 54/240 = 22.50% | 75.93% |
| LLaVA-Rad | 51 | 21 | 168 | 72/240 = 30.00% | 70.83% |
| MAIRA-2 | 30 | 0 | 210 | 30/240 = 12.50% | 100.00% |

Do not rank experts by conditional agreement alone. MAIRA-2's 100% concerns
only 30 comparable proposals, not all 240 reports or clinically complete text.
CXRMate-single here is **CXR-only**, not CXRMate-ED. Repeated/shared report
artifacts and four rows/image are correlated, not independent patients.

## Dependency patterns, not fault labels

At image-slot level, BioViL-T has 120 positive and 120 negative preferences:
that observed split was not enforced. XRV/BioViL agree on **184/240** images
(88 both positive, 96 both negative), disagree on **56/240** (32 XRV-negative /
BioViL-positive; 24 XRV-positive / BioViL-negative). The three BioViL templates
have 114 all-positive, 115 all-negative and **11 mixed/tied** image patterns.
Template dependence is not a calibrated uncertainty probability.

At correlated candidate-row level the mutually exclusive patterns are:

- Image sources disagree: **224** (56 images x four report slots).
- Image sources agree, report opacity evidence unavailable: **541**.
- Three proxy proposals agree: **159**.
- Report proposal opposes both agreeing image sources: **36**.

These sum to **960**. The 36 rows are a **review/localization hypothesis pool**,
not 36 confirmed report errors; no automatic action or faulty modality is set.
All 80 EHR anchors remain **unknown for this exact opacity finding**, not free
of disease. The two EHR-related clinical edges remain unavailable in this scope.

## 下一步的界限

现在可以观察评分器之间是否分歧，以及候选报告的提取标签是否与图像证据
相反；不能据此声称已经定位错误或提升生成质量。下一阶段应先限定可比较
的 finding、审计报告标签的否定/不确定提取及图像证据可靠性，再冻结修复
触发规则与独立评价端点。不能把缺失证据造成的低矛盾率当成修复成功，
也不能将同一评分器的选优分数作为独立证明。新的选优/生成/修复需另行
明确任务与批准；本轮没有修改任何既有选中 triple。

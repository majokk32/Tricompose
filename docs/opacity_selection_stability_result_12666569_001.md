# Frozen-selection opacity audit / 已冻结选优的 opacity 稳定性审计

Completed 2026-10-05 in existing CPU Slurm allocation **12666569**.

## 结论 / Outcome

历史选择不变时，满预算静态择优的 CXR–Report opacity 代理一致率，在
`max(Lung Opacity, Infiltration)` 定义下为 **100%**，在单独 `Lung Opacity`
输出下为 **52.38%**；两者可比较覆盖率均为 **26.25%**。因此，既有“高一致率”
结论对评分定义敏感。生成图像和报告没有改变，这不是生成效果突然变差。

本次结果是 **post-hoc DEVELOPMENT diagnostic**，不是临床准确率、独立确认性
测试或完整 triple 总分。现有 targeted heuristic 尚未证明优于静态择优或随机
对照，也没有从这些 readouts 得到错误定位或重新生成成功的证据。

No model, threshold, choice, prompt, generator, report extractor or historical
cost was changed. The two readouts share a checkpoint and cached report labels;
they are not independent clinical evaluators.

## 审计范围与复核 / Scope and checks

- 保留全部 **80 个固定 synthetic EHR / 240 个现有 CXR slots / 960 个候选
  triple slots**，没有替换“难例”。没有打开 EHR/report/prompt 正文或图像像素。
- 连接 **3,200 条历史 trials + 10,000 条已冻结 uniform-final-control trials**；
  每条选择读取两个 head，共 **26,400 readouts**。这些是同一批 80 个 EHR 的
  重复选择，不是新增患者或生成次数。
- 全部旧预算 **4、8、12、20、30** 和全部旧 seeds 均保留，未挑选最佳预算、
  seed、head 或新 winner；未再次运行 selector 或抽取随机最终选择。
- fixed/static/targeted 每例 1 个设置；旧 random 每例 5 个 acquisition seeds；
  uniform final 每例 5 × 5 = 25 个设置。先在每个 EHR 内平均，再对全部 80 个
  EHR 平均，避免把更多随机设置当成更多独立病例。
- 独立复核通过：全部 lineage/冻结选择/费用、**4,000 个 case means**、
  **50 个方法汇总**、**50 个配对对比**、**313 个 source pins**、7 个输出
  artifact hashes，以及 protected 的项目组权限。
- 本轮 **0 new model calls / 0 selector invocations / 0 new Slurm submissions**。
  Worker 约 **0.971 CPU 秒**，不含最终序列化，不是 GPU 生成耗时。
- 新增 24 个 synthetic-fixture tests；V1.2 回归共 **1,841 tests passed**。
  消费过的代码、协议、源输入和本次输出继续冻结；未修改队友的已有文件。

The preceding 240-image sidecar used 240 actual frozen XRV calls in approved
job 12668204. That extra evaluation work is not counted as historical GPU
savings. This audit only consumes its completed metadata.

## 指标与分母 / Definitions

只比较双方明确 positive/negative 的 **Lung Opacity** 断言。unknown/uncertain
不等于 negative；没有可比较证据不等于成功、矛盾为零或临床一致。

```text
Support / all EHRs    = mean of within-EHR support fractions over all 80 EHRs
Opposition / all EHRs = mean of within-EHR opposition fractions over all 80 EHRs
Coverage             = Support + Opposition
Conditional agreement = Support / Coverage; NA when Coverage = 0
```

这里的 opposition 是冻结 XRV/CheXbert 输出之间的代理冲突，不是经临床核实的
hard contradiction。XRV 输出是 operating-point-normalized score，不是校准疾病
概率。本轮阈值固定为 **0.5**，没有拟合。仅使用一个 finding，不推广到全部
CheXpert 14 labels。

对于这个 opacity endpoint，80 个 EHR 的现有 cached evidence 都是 unknown；
因此 EHR–CXR 与 EHR–Report 的 opacity 临床评分保持 **NA**。这不代表 EHR
没有诊断，也没有把 CHF、药物或 pneumonia 提升为直接 opacity 真值。

## 满预算汇总 / Cap 30 summary

旧 `random` 是 **随机获取候选 + 打分选最终结果**，并非完全随机选择。
`uniform final` 是已冻结的随机获取 + 无分数均匀最终选择。

| 方法 | 平均历史模拟调用 | Support / 全部 EHR | Opposition / 全部 EHR | 可比较覆盖率 | 可比较部分代理一致率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed: Sana + MAIRA-2 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Random acquisition + scored final | 30.000 | 13.75% | 12.50% | 26.25% | 52.38% |
| Random acquisition + uniform final | 30.000 | 17.70% | 9.40% | 27.10% | 65.31% |
| Static rerank | 30.000 | 13.75% | 12.50% | 26.25% | 52.38% |
| Targeted heuristic | 28.825 | 12.50% | 11.25% | 23.75% | 52.63% |

Fixed 的 100% 仅对应 **9/80** 例有可比较证据：9 support、0 opposition、
71 not comparable，不能据此宣称 fixed 最好。Static 是 11 support、10 opposition、
59 not comparable；targeted 是 10、9、61。Random/uniform 的比例来自 seed
平均，不要转换成单次最终输出的整数病例数。

Uniform 的 “79 EHRs with any comparison” 指至少一个重复选择具有比较证据，
不是一次运行输出了 79 个可比较 triples；它的 case-weighted coverage 仍是
**27.10%**。满预算的旧 random 与 static 选择相同，不构成独立重复验证。

### 同一选择的 secondary max-head 对照

| 方法 | Same-call max 代理一致率 | Exact-head 代理一致率 | 同一可比较覆盖率 |
| --- | ---: | ---: | ---: |
| Fixed | 100.00% | 100.00% | 11.25% |
| Random + scored final / Static | 100.00% | 52.38% | 26.25% |
| Random + uniform final | 92.07% | 65.31% | 27.10% |
| Targeted heuristic | 100.00% | 52.63% | 23.75% |

此前 50-case RICORD reference 的结果只支持那个受限真实参考集上的 opacity
区分能力；不能直接把此处 exact/max 的差异解释为 synthetic 临床错误数。
Synthetic 域迁移、报告断言提取和真实临床参考仍未充分验证。

## 全预算 exact-head 结果 / All frozen caps

下表不是事后挑选的最佳预算。Calls 是历史 **simulated** calls，不是实测 GPU
时间；相同预算上限不保证相同实际花费。

| 方法 | Cap | Calls | Support | Opposition | Coverage | Conditional agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 4 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Fixed | 8 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Fixed | 12 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Fixed | 20 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Fixed | 30 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Random + scored final | 4 | 4.000 | 20.50% | 9.75% | 30.25% | 67.77% |
| Random + scored final | 8 | 7.570 | 18.50% | 8.25% | 26.75% | 69.16% |
| Random + scored final | 12 | 11.380 | 15.00% | 8.25% | 23.25% | 64.52% |
| Random + scored final | 20 | 19.955 | 12.25% | 8.00% | 20.25% | 60.49% |
| Random + scored final | 30 | 30.000 | 13.75% | 12.50% | 26.25% | 52.38% |
| Random + uniform final | 4 | 4.000 | 20.50% | 9.75% | 30.25% | 67.77% |
| Random + uniform final | 8 | 7.570 | 19.20% | 9.75% | 28.95% | 66.32% |
| Random + uniform final | 12 | 11.380 | 20.85% | 8.60% | 29.45% | 70.80% |
| Random + uniform final | 20 | 19.955 | 18.30% | 8.95% | 27.25% | 67.16% |
| Random + uniform final | 30 | 30.000 | 17.70% | 9.40% | 27.10% | 65.31% |
| Static rerank | 4 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Static rerank | 8 | 8.000 | 12.50% | 1.25% | 13.75% | 90.91% |
| Static rerank | 12 | 10.000 | 15.00% | 5.00% | 20.00% | 75.00% |
| Static rerank | 20 | 20.000 | 12.50% | 5.00% | 17.50% | 71.43% |
| Static rerank | 30 | 30.000 | 13.75% | 12.50% | 26.25% | 52.38% |
| Targeted heuristic | 4 | 4.000 | 11.25% | 0.00% | 11.25% | 100.00% |
| Targeted heuristic | 8 | 8.000 | 12.50% | 1.25% | 13.75% | 90.91% |
| Targeted heuristic | 12 | 10.025 | 15.00% | 5.00% | 20.00% | 75.00% |
| Targeted heuristic | 20 | 19.775 | 11.25% | 3.75% | 15.00% | 75.00% |
| Targeted heuristic | 30 | 28.825 | 12.50% | 11.25% | 23.75% | 52.63% |

所有 50 个 cap/method/head 组合，以及 50 个配对对比，都保存在结果 CSV；
没有计算 clinical significance、训练 router 或用这些结果修改 policy。

## 配对解读 / Matched comparisons

仅旧 scored-final random 与 uniform-final control 共享完全相同的 acquisition
trace 和模拟花费。Cap 30 下，scored 相对 uniform：support **−3.95 pp**、
opposition **+3.10 pp**、coverage **−0.85 pp**，calls 差 **0**。
这是这个单一代理 endpoint 上的描述性对比，不证明随机选择临床上更优。

该配对有 **24/80** 个 EHR 出现：scored-final 的平均 proxy opposition 更低，
但它已没有任何可比较证据，而 uniform-final 的平均比较证据仍大于零。
这里比较的是每例 seed 平均（uniform 有 25 个设置），不是同一报告的临床
复判。这个计数标记 **lost comparison evidence**，不是 24 个成功修复。

Cap 30 的 targeted 相对 fixed：support **+1.25 pp**、opposition **+11.25 pp**、
coverage **+12.50 pp**，平均模拟调用 **+24.825**。Targeted 比 static 少
1.175 个平均模拟调用，同时 coverage 也低 2.50 pp；不能把这解释为已验证的
临床收益或实测 GPU 节省。

## 输出与不可变来源 / Artifacts and provenance

```text
artifacts/protected/tricompose_v1_2/opacity_selection_audits/selection_opacity_12666569_001/
  audit_plan.json
  trial_readouts.jsonl
  case_means.csv
  method_comparison.csv
  paired_case_comparison.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result manifest SHA256:
`94889888e3af839b7ab2367bcd413d0819ce5d4ae990c926fddaf199f2c6ea62`.

Frozen source manifest SHA256s:

- Historical replay:
  `af82764ac90f9af5b701c577fb2f9bfe6ecf4052310299e8e7cdcb415d2aed5f`.
- Complete-bank endpoint:
  `ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`.
- Existing uniform-final control:
  `771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124`.
- Completed opacity sidecar:
  `778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94`.

Protocol: `docs/opacity_selection_stability_protocol.md`.
Worker: `TriCompose-v1.2/tools/audit_opacity_selection_stability.py`.
Tests: `TriCompose-v1.2/tests/test_opacity_selection_stability.py`.

The protocol was written after the full-pool sidecar diagnostic was visible;
this is not an untouched confirmatory test. Historical choices nevertheless
preceded the new exact-head predictions and were never changed by this audit.

## 下一阶段边界 / Next boundary

本轮完成的是“冻结选择对评分定义有多敏感”的审计，不是重新优化生成。
下一阶段需要分开验证 image finding 与 report assertion 的可靠性，再用独立、
保留比较证据的 endpoint 评价定位和修复。不能靠切换 head、提高阈值、删除
难例或偏好少写事实的报告来制造提升。本轮不授权新 GPU 任务或新选择策略；
任何新 `sbatch` 仍需完整脚本、资源展示和用户明确批准。

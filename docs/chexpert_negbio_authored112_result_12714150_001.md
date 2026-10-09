# CheXpert/NegBio authored112：工程通过，current-finding 语义未通过

## 实际结果 / Actual outcome

在既有 CPU Slurm **12714150**（4 CPU、32 GB RAM）中，冻结官方
CheXpert/NegBio 解析 **112 条已知开发用人工文本**，每条重复一次。
112/112 两次均 complete，0 unavailable，0 full-evidence replay difference。
初始化 **7.747249 秒**；输入加载与 224 次解析阶段共 **6.290423 秒**，
不包括后续哈希复核/指标计算。没有新 GPU/Slurm 提交、下载、训练或 API。

| 文本集 | 输出视图 | 符合人工任务参考 / 全部尝试 | Macro F1 | 不当确定判断 | 正负翻转 | 不可用 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| old48 | 官方 native | 39/48（81.25%） | 0.80237 | 9 | 0 | 0 |
| old48 | TriCompose mention-conflict | 43/48（89.58%） | 0.89509 | 5 | 0 | 0 |
| new64 | 官方 native | 26/64（40.63%） | 0.40537 | 34 | 3 | 0 |
| new64 | TriCompose mention-conflict | 29/64（45.31%） | 0.45930 | 30 | 2 | 0 |

“不当确定判断”指：人工任务参考为 uncertain/unknown，但输出为
positive/negative。它使用这份测试约定，不等于临床误诊率。
Macro F1 基于四状态混淆矩阵；失败仍计入尝试分母，不作为正确 unknown。

这些数字衡量的是与既有 **literal lung-opacity/current-finding 任务**的
对齐，不是官方 CheXpert 全 ontology 的临床准确率，不是 80 个 triples
的质量提升，也不是 held-out gold benchmark。官方 Lung Opacity 词表更宽。
作者已知这些开发文本与之前结果；没有调整参考标签、模型规则或阈值。

## 哪些可用、哪些仍有问题 / Scope limitations

old48 的显式阳性、显式阴性、不确定性、未提及、其他 finding 否定等
基础组均为 4/4。官方 native 的正负冲突组为 0/4，而另外保存的
mention-conflict view 为 4/4；这是披露过的 aggregation policy 差异，
不是另一份独立临床判断或新训练的 scorer。

new64 的显式存在、不确定性和句子边界组均为 4/4；显式否定为 3/4。
但下面各组，两种视图均为 **0/4**：

- historical-only、family-only、hypothetical-only；
- resolved-only、change-only、qualified-absence；
- nonpulmonary mentions。

history-then-current 两种视图均只有 1/4；other-entity-negation 为 2/4。
opposed-current native 为 0/4，conflict view 为 3/4。
完整逐组混淆与 literal-present slice 保存在 protected summary 中。

结论是 **冻结官方解析器已能实际运行、重复结果稳定，但不能直接当作
当前肺部 finding 的独立临床真值或自动修复裁判**。原生否定/不确定性
标签没有解决所有 anatomy、experiencer、temporal scope 和 qualified
absence 语义。仅用冲突聚合不足以解决这些问题；不能据此触发候选拒绝、
替换 winner 或生成修复。

## 实现与复核 / Implementation and audit

```text
TriCompose-v1.2/interfaces/chexpert_negbio_legacy_worker_v1.py
TriCompose-v1.2/tools/benchmark_chexpert_negbio_authored112.py
TriCompose-v1.2/tests/test_chexpert_negbio_legacy_worker.py
docs/chexpert_negbio_authored112_protocol.md
```

上游 Classifier.classify 源码和调用顺序未修改。观测 hooks 在 native
删除句子前检查 syntax availability/mention-node coverage，并捕获可能被
官方 converter/detector 吞掉的 ERROR。错误是 unavailable/null，不能转成
unknown 或无标记 positive。原生十四类标签和 conflict view 独立保存；
No Finding 不扩展为十四项阴性。Offsets/hashes 对齐官方 cleaned passage，
没有序列化报告正文或临床输入。

18 项新增 failure-injection/contract 测试通过；V1.2 全量 **2,156 项测试
通过（13.651 秒）**。这些是代码测试，不是临床正确率。
独立 standard-library audit 验证 **2,127 个 source pins**、7 个结果
artifact、固定输入/参考 join、所有指标/逐组分母、evidence spans、全部
replays、protected 权限和原候选库 manifest。两个官方 root repo 仍干净。

## 参考隔离的精确含义 / Reference phase disclosure

worker 没有解析 reference JSON、没有把参考 state/family 传给解析器；
预测/replay 先 fsync/hash，评估器之后才解码 reference semantics。
预执行 provenance 校验会为 manifest 中的 reference source 文件读取字节
计算 SHA256。因此原 receipt 字段 `references_read_before_prediction_fsync=false`
指 **未解码/使用参考语义**，不能理解为参考文件绝对零字节读取。
这一命名边界在此明确披露，未覆盖已封存 receipt；哈希读取不是临床盲测，
也没有使作者已知开发文本变成 held-out 数据。

## 输出 / Output

```text
artifacts/protected/tricompose_v1_2/chexpert_negbio_authored_plans/authored112_12714150_001/
artifacts/protected/tricompose_v1_2/chexpert_negbio_authored_runs/authored112_12714150_001/
  predictions.json
  replays.json
  prediction_freeze_receipt.json
  runtime.json
  scored_checks.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Plan manifest SHA256：

```text
6680a508f33c055d22c61c350d0d0bda53c794233f12dc4e9df97573789c6bfc
```

Run manifest SHA256：

```text
d76721820fc756f789873482fee76e9aefd23f7f999c383a3af686813e65af0a
```

独立审计：`.tmp/audit_chexpert_negbio_authored112_12714150_001.py`。
目录 2770、文件 0660，project-group boundary；NFS group 可显示为 nobody。
原 complete-bank manifest 保持
`ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`。

## 下一步 / Next boundary

先制定 scope-aware evidence/abstention 的独立协议：当前/历史、患者/家属、
肺部/非肺部、qualified/change-only 信息应显式分开，且不把不同解析器
对同一文本的 agreement 当作独立临床真值。先封存规则/接口与新测试，再
实际验证；不根据这 112 条逐条添加修补规则，不训练新模型。

本轮不重评已有 960 个报告槽位、不更新旧 score table、EHR facts 或
selected triples、不运行 targeted regeneration。自动择优/修复仍需要
独立可靠性证据；当前 `clinical_qualified=false` 保持不变。

# MedCPT 专家报告评分检查结果

2026-10-06。MedCPT 在已固定的 RadEvalExpert 队列上完成编码和独立数值复核，
但结果**不支持把 full-report cosine 直接用于临床质量评分或再生成决策**。
全量分数与负向临床显著错误数的 Spearman 为 −0.105，按最高分选报告相对
随机选择的错误数减少尚不确定。旧 960 候选的评分、选优和生成内容保持不变。

## 完成范围

本轮在现有 Slurm CPU 作业内完成，无新提交、训练、下载、依赖安装或外部 API。
只在受保护的 worker 内输入已授权的参考及候选报告，没有读取 EHR 或图像，
没有向聊天、Git 或公开日志输出报告原文及患者标识。

624/624 对报告有分数，762/762 条不同文本成功编码；native token 数为 6–421，
低于模型 512-position 容量，全部完整输入，未截断。参数冻结、CPU FP32、native
CLS 768 维，共 97 个 forward calls；预先固定的前 8 条输入重放完全一致。
这不是对全队列的第二遍重放，也不是官方 query/article 检索指标或 64-token
短 query 示例。checkpoint、输入合同和代码在运行前固定，没有事后调参。

总耗时 92.556 秒，其中编码及小重放 82.451 秒、统计 6.168 秒，峰值 RSS
1.087 GiB。分配 4 CPU / 32GB / 0 GPU，worker 使用 2 个线程。

## 与医生错误计数的相关性

比较方向固定为：**分数越高，应对应越少错误**，即与负向专家错误计数相关。
下列为临床显著错误总数；95% CI 来自 1,000 次依赖组 bootstrap，seed 0。
分数不是概率，相关性也不是准确率。

全量共同队列：624 对有评分，623 对显著错误参考可用，622 对总错误参考可用。
180 个依赖组中有 72 个未确认是患者级分组，不能称为 624 个独立患者。

| 指标 | 有效显著错误比较数 | Spearman | 95% CI |
|---|---:|---:|---|
| MedCPT full-report cosine | 623 | −0.1054 | [−0.2201, −0.0028] |
| RadGraph entity F1 | 623 | 0.0447 | [−0.0685, 0.1470] |
| RadGraph relation-presence F1 | 623 | 0.0463 | [−0.0637, 0.1478] |
| RadGraph full-relation F1 | 623 | 0.0370 | [−0.0709, 0.1323] |

MedCPT 与负向总错误数的相关性也为负：−0.1274，95% CI [−0.2430, −0.0300]。
这是该队列上的相关现象，不能推断模型导致错误或据此翻转分数方向。

图像评分共同子集：132 对、34 个已识别 MIMIC 患者组。所有指标使用完全相同的
子集；其余 492 对保留为图像证据不可用，没有换图或删除困难病例。

| 指标 | 有效显著错误比较数 | Spearman | 95% CI |
|---|---:|---:|---|
| MedCPT full-report cosine | 132 | 0.0700 | [−0.1488, 0.2339] |
| BioViL-T image-report cosine | 132 | −0.1510 | [−0.3198, 0.1098] |
| RadGraph entity F1 | 132 | 0.1372 | [−0.0885, 0.2958] |
| RadGraph relation-presence F1 | 132 | 0.1555 | [−0.0687, 0.3224] |
| RadGraph full-relation F1 | 132 | 0.2400 | [0.0509, 0.3992] |

子集 full-relation 相关性为正，不代表可以把 reference-based F1 当作 synthetic
inference 中的 reference-free verifier；该子集中的选择收益仍不确定。
完整 evaluation 保留了全部显著及不显著错误类别，没有挑选最好的类别作为主指标。

## 同一参考下选择三份报告

固定 anchor 指同一个 source、section 和参考报告，各有三份候选；不是三个
独立病例。最高分平局使用均匀期望选择，随机 baseline 直接计算三份候选的平均
错误数。缺少任一评分或错误计数的 anchor 不参与该 endpoint，但保留不可用记录。

| 队列及 MedCPT endpoint | 完整 anchors | 随机期望错误数 | 最高分期望错误数 | 差值及 95% CI |
|---|---:|---:|---:|---|
| 全量显著错误 | 207/208 | 4.0837 | 3.8406 | −0.2432 [−0.5070, 0.0323] |
| 全量总错误 | 206/208 | 4.8236 | 4.6068 | −0.2168 [−0.4943, 0.0862] |
| 图像子集显著错误 | 44/208 | 4.8182 | 4.5682 | −0.2500 [−0.7659, 0.5196] |
| 图像子集总错误 | 44/208 | 5.3258 | 5.1136 | −0.2121 [−0.7501, 0.5787] |

差值为选择后错误数减随机错误数，负值较好，但四个区间都跨 0。
全量显著错误的候选间排序正确率为 0.5487，是参考计数有严格差异时的排序诊断，
不是疾病诊断准确率。整体跨病例相关性与同 anchor 内排序回答不同问题，不能
只保留较有利的一个。

## 对 TriCompose 的决定

保留 MedCPT 部署作为语义检索的实验组件，**不将当前 cosine 升级为 clinical
consistency score、不拟合阈值或新权重、不触发修复、不改历史赢家**。
手写检查中否定、历史及左右侧差异也可能有高 cosine；本轮专家参考结果并未
解决这些问题。下一步应验证事实匹配及 polarity、anatomy、current-patient/
temporal scope，而不是继续给一个全局相似度加权。

本队列的专家计数和其他指标此前已分析，不能称为 untouched 或 blinded test。
训练重叠未验证，每对仅一份已发布 reader cell，部分依赖分组身份未确认。
参考报告错误计数不等于新的图像医生审阅，因此本轮不验证图像真值、EHR 一致性、
三模态错误定位或修复成功。没有使用未获许可的 BioLORD/default RaTEScore。

## 输出与独立复核

```text
artifacts/protected/tricompose_v1_2/radeval_medcpt_runs/reportref_12714150_001/
  scores.json
  token_receipts.json
  private_embeddings.json
  prediction_receipt.json
  score_table.json
  evaluation.json
  summary.json
  manifest.json
artifacts/protected/tricompose_v1_2/radeval_medcpt_audits/numeric_12714150_001/audit.json
```

独立 stdlib 实现重算全部 624 个 cosine、153 个点相关性、18 个相关性 bootstrap
区间、2,092 个可用 anchor/指标/endpoint 单元及选择差值区间；源码和历史 hash、
受保护目录 2770 和文件 0660 均通过。耗时 6.438 秒，0 model calls，未读取报告
原文。独立复核没有重新执行 tokenizer 或模型，不能作为第二次模型推理的证据。
预测在统计整合前独立封存。机械复核通过不等于临床资格通过。

- Worker manifest SHA256：
  `7005c06b9882317fc8b4148cb06e525c78c07623370e191e5bde2ce88cc8d754`。
- Audit SHA256：
  `4ffe9ae03d2a0155f728f225e0a6bece8998bbde117879270cacf3ed557ca66f`。

模型与输入定义见冻结协议 `docs/radeval_medcpt_protocol.md`；该协议保留运行前状态，
不事后改写。旧 source、模型、生成输出、评分与选择保持不变。

# MedCPT 部署与手写文本检查结果 / Deployment and authored-text diagnostic

2026-10-06。用户明确批准公开模型下载及小规模手写文本 CPU 测试后完成。
本轮没有读取真实或已生成病例、训练、安装依赖、新建环境、新提交 Slurm
作业、重算旧候选池、改变选优或执行再生成。BioLORD/default RaTEScore 未部署：
用户没有所需的 UMLS/SNOMED CT 许可。

## 实际完成了什么

使用 NCBI 官方 `MedCPT-Query-Encoder`，固定 revision
`d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`。全部 11 个公开模型、tokenizer
及文档文件的字节数和 SHA256 均验证通过；437,951,328-byte 权重 SHA256：
`19d78c0d5eaee2f81e6c47c5425bbadcc0c6af016cbb5da4a000d64e59d6e342`。
模型参数完整绑定，没有 missing/unexpected/mismatched keys。

在现有 Slurm CPU allocation 12714150 内运行：分配 4 核 / 32GB，实际使用
2 个线程、FP32、0 GPU。沿用已存在环境，加载后冻结参数、eval/inference mode，
禁止网络访问。原生 CLS 输出 768 维，不增加 prefix，不改 tokenizer/pooling。

- 下载：8.226 秒；模型初始化：1.151 秒。
- 两遍编码：2.182 秒；整个 worker：14.722 秒。
- 峰值 RSS：0.731 GiB。
- 41 组预先固定的手写比较，40 组非空完成；1 组空输入保持 null/不可比较。
- 38 条不同非空文本，10 个 batch forward calls（两遍，各 5 个 batch）。
- 全部输入未截断；两遍向量及评分完全相同。

这些是 investigator-authored language probes，**不是医生标注、独立临床 gold、
代表性疾病测试集或全量病例评估**。尤其手写 paraphrase 不构成经验证的医学等价词表。

## 原始 cosine 结果：相关不等于一致

| 手写比较类型 | n | 平均 query-query cosine |
|---|---:|---:|
| 完全相同的文本 | 6 | 1.000000 |
| 手写同义改写 | 6 | 0.742246 |
| 不同 finding | 6 | 0.645474 |
| 肯定 vs 否定 | 6 | 0.987439 |
| 当前 vs 既往描述 | 6 | 0.913962 |
| 当前 vs 假设描述 | 6 | 0.938872 |
| 左右侧差异 | 1 | 0.961050 |
| 解剖位置差异 | 1 | 0.935143 |
| 严重程度差异 | 1 | 0.971574 |
| 设备位置差异 | 1 | 0.814865 |

同义改写的 cosine 高于不同 finding：4/6 组；但低于同 finding 的否定、历史、
假设描述：每类均 6/6 组。这里报告的是固定手写组内的排序现象，**不是临床准确率**。
不拟合阈值，不翻转评分方向，不删除表现不好的组，不修订已消费的 probes。

结论：**部署和编码接口成功；仅凭 cosine 不能判定 clinical consistency。**
本轮不把它添加为主要矛盾分数，不接入 regeneration trigger，也不把 null 变为 0。
若后续使用，合理的待验证角色是实体候选语义检索；polarity、anatomy、severity、
current-patient/temporal scope 必须另外判定。MedCPT 不是官方 RaTEScore，也不是
image-grounded factuality verifier。

## 目录与复核

```text
MedCPT/
  README.md / LICENSE  # 官方文档快照，不是完整代码 clone
runtime/models/medcpt-query-v12-12714150-001/
  model/ / source/ / asset_manifest.json
TriCompose-v1.2/tools/run_medcpt_authored_smoke.py
TriCompose-v1.2/tools/audit_medcpt_authored_smoke.py
artifacts/protected/tricompose_v1_2/medcpt_authored_runs/
  query_12714150_001/
    frozen_plan.json / score_table.json / embeddings.json
    token_receipts.json / summary.json / manifest.json / worker.log
  query_12714150_001_audit/audit.json
```

新增目录 2770、文件 0660，项目组权限验证通过；原有 runtime 父目录权限保持不变。
模型、文档快照、cache 和 protected 输出由 Git 排除。

- Worker manifest SHA256：
  `1f5c66f573a0f6a36037a23a7b9d5fc8072f9e375c89b5d398694d5e65cea992`。
- 独立 audit SHA256：
  `1c60483ad354f098294e93a3f2cc793ac4a574d1edd1cf90a0b531541d113247`。
- 独立 stdlib audit 重算全部 40 个 cosine、所有分组均值及排序；验证所有 asset、
  source、历史 manifest hash、输出完整性和保护权限。耗时 0.350 秒、0 model calls。
- 两遍 inference replay 由 worker 实际执行并验证；独立 audit 核验其 receipt，
  **没有再次运行模型，也不声称独立重跑了向量生成**。
- 新增 31 项 guard/fixture tests；全 V1.2 回归 **2,487 项通过，14.515 秒**。

测试和 hash 验证的是实现、可复现性及溯源，不验证临床准确性。
旧 960-candidate table、EHR、CXR、reports 和 winners 未修改。

## 下一步（尚未执行）

先冻结一个独立的实体匹配/事实状态验证协议，再决定是否测试官方专家参考报告。
保持身份匹配 baseline；把语义相近候选与事实等价、否定冲突、位置冲突分开。
如使用此前已分析的 RadEvalExpert 624 对，应明确它不是 untouched test，且严格
保持共同分母、缺失覆盖和 patient/source clustering；不因观察到 cosine 而挑选病例。

本轮批准仅覆盖下载和手写文本测试，不自动授权新的患者输入范围、专家报告模型
计算、960 候选重算或任何新 `sbatch`。新作业仍须展示完整脚本和资源并取得批准。

官方依据：[模型卡](https://huggingface.co/ncbi/MedCPT-Query-Encoder)、
[许可证](https://huggingface.co/ncbi/MedCPT-Query-Encoder/blob/main/LICENSE)、
[官方仓库](https://github.com/ncbi/MedCPT)、
[Bioinformatics 2023 论文](https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842)。
本次 query-query cosine 是相关度诊断，不是 query/article 原生 inner-product 检索指标。

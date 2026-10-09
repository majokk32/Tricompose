# 新病例 prompt 顺序对照的提交准备

2026-10-09。获批作业 `12884193` 已在 A40 上完成，耗时 1m44s：24 张图与
24 次 XRV 评分全部完成，0 失败、0 重试，只读 postflight 通过。
完整配对数字与解释边界见 `docs/new_case_prompt_order_result.md`。
本记录用于追溯已消费的计划，不应重交同一 run 或修改 sealed code、tests、
protocol 和 plan。CPU preparation 固定既有 100-case
synthetic pool 中全部剩余三例 eligible EHR：2 pneumonia、1 pleural effusion。
每例两种冻结模型、seeds 0/1、两种句子顺序，共 24 张图与最多 24 次 XRV 评分。
这是不同病例及当前 renderer 的开发扩展，不是临床独立测试或已完成修复。

## 已完成检查

历史 screen 与源 manifest 的 SHA256、100-row linkage、3 个源 JSON hashes、
structural validity 与现有 fact extractor 的 eligibility 全部通过。源 pool 的完整
分母仍是 100：74 valid、5 eligible、2 已用于先前 smoke、3 此次全部保留。
这三例 canonical hashes 与上一轮 order diagnostic 的两个 anchors 不同。
新旧 pool 可能使用相同 opaque ID，因此没有把 ID 不同当成内容独立的证明。

3 个 canonical EHR、facts 和 source manifests、12 个 final prompt files 已写入
新的 protected run。6 组原顺序/前置文本的词 multiset 一致；每个病例的各位置共用
固定 EHR、evidence、clinical intent、checkpoint 和预声明 seeds。只有本轮每组内部是
纯顺序对照：当前 V1.1.4 renderer 与上一轮 legacy renderer 不同，不把跨轮差异
全部归因于新病例。CHF 不再成为直接 image assertion，PA 仍是固定生成协议。

源码与输入 pins、28 个 prepared artifacts、模型资产 size/mtime 与既有 receipts、
文件 0660、目录 2770、CARC NFS GID65534 均通过。上述 CPU preparation 未使用
模型 factory/load、GPU forward 或大型权重 bytes。整个实验未读取真实患者
row/report/image/target，未调用 API、report generation 或 planner。没有改变源
EHR、旧 prompt/scorer/threshold、historical outputs 或 winner。

28 项新增 authored-fixture tests 通过。全量初跑因遗漏已有 SynEHRgy source 路径
和 DICOM dependency 路径而出现环境错误；把已有路径加入该测试命令的 PYTHONPATH
后，**4,058 项 V1.2 tests 全部通过，32.731s**。没有改测试语义、旧源码、环境
安装或依赖。CPU staging、tests 和 metadata audit 均在实际 allocation `12879534`。

## 封存入口与计划

Entry：`TriCompose-v1.2/agent/run_new_case_prompt_order.py`。
SHA256：`b67ea183bd04bde02885cfbb4485727678ecd77ecaa11e79acd99c6907294ed8`。

Protected plan：
`artifacts/protected/tricompose_v1_2/new_case_prompt_order_plans/new_order24_12879534_001/`。
Manifest SHA256：
`d54380e9640179d629d341fd20d89ed8ac3ba3703c6eb180cc71fb4b8eea06e8`。

Protocol：`docs/new_case_prompt_order_protocol.md`。
Source code、tests、protocol 和 prepared plan 均保持封存不变。

## 已批准资源与结果范围

完整 batch：`TriCompose-v1.2/agent/slurm/22_new_case_prompt_order24_gpu_flexible.sbatch`。
SHA256：`113a57be2cecd60e79e46b57e01a37fa659f1633a126878fa66aeb06367bd4c2`。
Batch shell syntax 检查通过。
申请 `debug,gpu` 分区、`v100|a40` node feature、1 GPU、2 CPU、32G RAM、
15 分钟上限，禁止 requeue。三个串行 workers 最多 3 次 model loads 和 48 次
charged inference attempts，不重试；失败/缺失保留。资源查询时 debug A40 有
1 张空闲，不能据此保证启动或完成时间。前一轮 16 张图的 1m35s 只作历史参考。

按每个病例的条件分别报告 pneumonia 和 effusion 的 raw XRV score 与差值。
不混合两种 finding 的尺度，不改阈值，不把上升解释为 clinical correctness。
全部 12 配对行保留，不选择最佳 seeds，也不替换旧的最优 triple。
完整脚本与请求展示并获得用户新的明确批准后，已执行：

```bash
sbatch TriCompose-v1.2/agent/slurm/22_new_case_prompt_order24_gpu_flexible.sbatch
```

已完成输出：
`artifacts/protected/tricompose_v1_2/new_case_prompt_order_runs/new_order_12884193/`。
实际获得 debug A40、1 GPU、2 CPU、32G；Slurm exit `0:0`。
全部 12 配对行已保存，但 clinical acceptance 仍为 false，accuracy 为 NA。

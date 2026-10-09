# 固定 synthetic EHR 的 prompt 顺序对照结果

2026-10-09。同一份封存 prompt 只移动 finding block 后，RoentGen-v2 的肺炎
XRV 分数在 4/4 组中上升，Sana 只有 1/4 组上升。全部 16 张图和 16 次评分
完成，耗时 1 分 35 秒，0 失败。这个结果不支持统一采用 finding 前置，也不等于
临床质量提升或自动修复成功。原有 EHR、facts、prompt、评分规则和最优组合未变。

## 固定输入与实验设置

固定历史 synthetic pool 中的两个 EHR，opaque case IDs 为 `case_009`、
`case_018`；它们是已有 development anchors，不是新 held-out 病例。每个模型
使用 seeds 0 和 1，分别重新生成 `original` 与 `findings_first` 两个分支。
共 2 EHR × 2 模型 × 2 seeds × 2 分支 = 16 张图、8 组配对。没有复用旧图评分。

每组内部固定 checkpoint、seed 和官方推理参数。RoentGen-v2 使用 75 steps、
CFG 3、512×512、float16；Sana 使用 20 steps、CFG 4.5、1024×1024、float16。
Sana 的原生前缀与文本规范化保留。所有模块冻结，不生成 report、不调用 planner
或外部 API，不训练、不下载、不改阈值、不重试。

两条分支使用完全相同的历史文字和标点，仅移动完整 finding block；不添加疾病、
阴性、设备、位置、严重度或新的 EHR 信息。历史 renderer 与当前 renderer 不同，
因此本轮复制 SHA256 绑定的旧 prompt，而不是用当前版本重建。两条分支均保留
历史 CHF-to-CXR prior，且明确标为未经验证：CHF 临床诊断不能独立证明某项
胸片征象。本实验隔离文字顺序变化，不验证这个旧 prior 的临床合理性。

这仍是 EHR-derived text → CXR，不是 direct structured-EHR-conditioned CXR。

## 全部配对分数

数值来自原有冻结 XRV，使用 unchanged checkpoint 与预处理的
operating-point-normalized 分数，不是校准后的疾病概率。差值为前置减原顺序。
肺炎是本轮直接 finding 的主要 readout；实变是延续上一轮的次要 readout，
不是新增的独立 EHR 事实。CSV 保留精确数值，下表四舍五入到四位小数。

| Case | 模型 | Seed | 肺炎原顺序 | 肺炎前置 | 肺炎差值 | 实变原顺序 | 实变前置 | 实变差值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 009 | RoentGen-v2 | 0 | 0.1849 | 0.2410 | +0.0560 | 0.5058 | 0.5157 | +0.0099 |
| 009 | RoentGen-v2 | 1 | 0.1089 | 0.3492 | +0.2404 | 0.1515 | 0.4246 | +0.2731 |
| 009 | Sana | 0 | 0.5005 | 0.4336 | −0.0669 | 0.5397 | 0.5336 | −0.0061 |
| 009 | Sana | 1 | 0.5041 | 0.5242 | +0.0202 | 0.5474 | 0.5345 | −0.0129 |
| 018 | RoentGen-v2 | 0 | 0.0635 | 0.1114 | +0.0479 | 0.5031 | 0.5010 | −0.0021 |
| 018 | RoentGen-v2 | 1 | 0.1997 | 0.2041 | +0.0044 | 0.2356 | 0.2772 | +0.0416 |
| 018 | Sana | 0 | 0.5119 | 0.2561 | −0.2558 | 0.5290 | 0.5055 | −0.0235 |
| 018 | Sana | 1 | 0.5536 | 0.5002 | −0.0534 | 0.6282 | 0.5370 | −0.0912 |

| 模型 | 配对数 | 肺炎均值原顺序 | 肺炎均值前置 | 肺炎平均差值 | 肺炎上升组数 | 实变平均差值 | 实变上升组数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoentGen-v2 | 4 | 0.1393 | 0.2264 | +0.0872 | 4/4 | +0.0806 | 3/4 |
| Sana | 4 | 0.5175 | 0.4285 | −0.0890 | 1/4 | −0.0334 | 0/4 |

这只是描述统计。8 组配对共享两个 EHR、两个 seeds 与两个模型，不能作为
8 个独立患者进行显著性推断，也不足以进行模型临床排名。全部 seed 保留，
没有筛掉下降分支或选择最佳 seed 来汇报。

## 输入边界与结果解释

8/8 组的 tokenizer input-ID 哈希不同，8/8 组的图像字节哈希不同，证明句子
顺序变化抵达 tokenizer 边界并伴随输出变化。哈希不同不能证明图像解剖、疾病
或临床事实正确。Sana 的 8 个位置均记录原生文本转换，RoentGen 的 8 个位置
均未记录文本转换；本轮未改官方生成参数。

finding 前置对两个生成器的观测影响不同，因此不替换全局默认 renderer。
可以把顺序变换作为一个模型特定的候选干预，但必须先用不参与该选择的病例
与独立观察器验证，再比较固定路径、静态 reranking 和动态修复。
本轮没有 report 边、临床 gold standard、故障模态定位或 LLM 决策，因此没有
三模态修复成功、成本节约或 Agent 优势结论。

## 执行记录与完整性

用户在完整脚本及资源请求展示后明确批准。作业 `12883369` 在 `debug` 分区的
A40 节点 `b11-09` 完成；Slurm 状态 `COMPLETED`，exit `0:0`，耗时 1m35s。
批准的请求为 `debug,gpu`、`v100|a40`、1 GPU、2 CPU、32G 主机内存、
15 分钟上限、`--no-requeue`。本次实际获得 A40，无排队等待。

实际计入 32 次 inference attempts，其中 16 次 CXR generation 与 16 次 XRV
评分；3 次 model-load attempts、3 次 serial worker processes。失败数、未完成
reservation 数、重试数均为 0。历史 EHR 生成与其他实验成本不包含在这 32 次中。
记录的最高 PyTorch peak allocated VRAM 为 RoentGen 3.02 GiB、Sana 6.98 GiB；
这不是整个进程或 CUDA reserved memory 的上限。

CPU allocation `12879534` 的只读 postflight 通过：复核输出 manifest 的全部
48 个 artifact hashes，重算所有 8 个配对行，重核 16 个固定位置、image-score
绑定、逐模型 reservations、dispatch、summary 算术、scorer checkpoint 及预处理
标识、原输入和源代码 pins。文件 0660、目录 2770、CARC project NFS GID65534
检查通过。检查不解码任何图像，不读取原始患者输入或报告，也不读取大型权重。
执行前的 26 项新增 fixture tests 与全部 4,030 项 V1.2 tests 已通过。

输出目录：
`artifacts/protected/tricompose_v1_2/fixed_ehr_prompt_order_runs/prompt_order_12883369/`。
`paired_readout.csv` 保存精确分数；`summary.json` 保存调用计数；PNG、tokenizer
traces、native logs 和 journals 仅保留在 protected 目录。

输出 manifest SHA256：
`f1a22389b267257f36c104b9ebdaf0b633f14005114e1b87e7546469ddf712e2`。
封存 plan manifest SHA256：
`f45718f3a2917ec24fd06c4b8551fd84b958528e9790511d4262c924d3bc06c7`。
协议：`docs/fixed_ehr_prompt_order_protocol.md`。
Consumed code、tests、protocol、plan 以及旧 outputs 均未修改。

## English summary

All 16 images and 16 unchanged XRV observations completed on one A40 in 1m35s,
with 32 charged inference attempts, three model loads and no failure or retry.
Moving the exact sealed legacy finding block changed all eight paired tokenizer
and image hashes. Pneumonia readouts increased in all four RoentGen pairs
(mean delta +0.0872), but only one of four Sana pairs (mean delta −0.0890).
This argues against globally replacing the renderer. There are only two reused
synthetic development EHRs, and legacy CHF image priors remain unvalidated.
The scores are operating-point-normalized observations, not calibrated
probabilities, clinical accuracy or evidence of repaired triples or LLM advantage.
Original EHRs, scoring rules and selected outputs remain unchanged.

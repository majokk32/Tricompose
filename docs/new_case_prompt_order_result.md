# 新 synthetic 病例的 prompt 顺序对照结果

2026-10-09。全部 24 张图和 24 次冻结 XRV 评分完成，耗时 1 分 44 秒，
0 失败或重试。新病例没有支持 RoentGen 的 finding 前置作为通用修复：肺炎
4 组配对中只有 2 组分数上升，平均差值 −0.1514；积液两组均上升。Sana 的
肺炎也是 2/4 上升，积液则 0/2 上升。顺序变换应保留为待验证的候选干预，
不能据前一轮的 4/4 肺炎上升直接替换默认 prompt。

这些是同一未改变分类器的响应读数，不是临床准确率或已接受的三模态修复。
没有修改 EHR、旧 prompts、评分器、阈值、策略或最优输出。

## 固定病例与实验设置

源是既有 unconditional SynEHRgy Qwen2-40bins 的 100-case synthetic pool。
原筛查的完整分母为 100：74 structurally valid、5 eligible，其中先前已使用
前两例。此次固定全部剩余三例：case_079 为 pleural effusion，case_080 和
case_082 为 pneumonia。没有根据本次图像或分数换病例、追加病例或筛 seed。
它们不是新的临床 gold test，也不是无条件生成分布的随机样本。

每例使用 RoentGen-v2 与 Sana，各 seeds 0、1，分别重新生成 original 和
findings_first 两条分支；共 3 × 2 × 2 × 2 = 24 张图、12 组配对。
各组内部保持 EHR、evidence-grounded facts、clinical intent、checkpoint、seed
和官方推理参数，仅移动完整 finding block，保留所有词与标点。原顺序分支也
重新生成，不使用旧图充当 control。

两条分支均使用当前 V1.1.4 renderer，CHF 只作为临床背景，不成为自动影像
断言；PA 来自固定生成协议，不是 EHR 中的 view evidence。上一轮采用 legacy
renderer 与未验证 CHF image prior，因此跨轮同时存在病例来源和 renderer
版本差异，不能把跨轮差异全部归因于新病例。只有本轮每组内部是纯顺序对照。
pneumonia-to-opacity 仍是 renderer 的弱影像 prior，不是独立图像真值。

RoentGen-v2 保持 75 steps、CFG 3、512×512、float16；Sana 保持 20 steps、
CFG 4.5、1024×1024、float16，原生 prefix/lowercase 保留。所有模型冻结。
没有报告生成、planner、外部 API、训练、下载、重试、阈值拟合或 winner 更新。
该路径是 EHR-derived text → CXR，不是直接读取 structured EHR 的 CXR 模型。

## 全部配对读数

XRV checkpoint 与 grayscale/crop/224px 预处理不变，使用既有 pneumonia head
及 effusion head。每例仅以自己预声明的 condition 作主读数，积液不以肺炎分数
评价；未知事实不变成阴性或零。以下 raw operating-point-normalized scores
不是校准后的疾病概率。差值为前置减原顺序；精确数字保存在 protected CSV。

| Case | Finding | 模型 | Seed | 原顺序 | Finding 前置 | 差值 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 079 | Pleural effusion | RoentGen-v2 | 0 | 0.0463 | 0.1676 | +0.1213 |
| 079 | Pleural effusion | RoentGen-v2 | 1 | 0.0354 | 0.0693 | +0.0340 |
| 079 | Pleural effusion | Sana | 0 | 0.9716 | 0.9672 | −0.0044 |
| 079 | Pleural effusion | Sana | 1 | 0.8108 | 0.7989 | −0.0119 |
| 080 | Pneumonia | RoentGen-v2 | 0 | 0.1567 | 0.1240 | −0.0327 |
| 080 | Pneumonia | RoentGen-v2 | 1 | 0.0963 | 0.1469 | +0.0506 |
| 080 | Pneumonia | Sana | 0 | 0.5415 | 0.6334 | +0.0919 |
| 080 | Pneumonia | Sana | 1 | 0.4937 | 0.5188 | +0.0251 |
| 082 | Pneumonia | RoentGen-v2 | 0 | 0.8774 | 0.1651 | −0.7123 |
| 082 | Pneumonia | RoentGen-v2 | 1 | 0.0744 | 0.1631 | +0.0887 |
| 082 | Pneumonia | Sana | 0 | 0.6280 | 0.6128 | −0.0152 |
| 082 | Pneumonia | Sana | 1 | 0.5509 | 0.5260 | −0.0249 |

| Finding | 模型 | EHR 数 | 配对数 | 原顺序均值 | 前置均值 | 平均差值 | 上升组数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pleural effusion | RoentGen-v2 | 1 | 2 | 0.0408 | 0.1185 | +0.0776 | 2/2 |
| Pleural effusion | Sana | 1 | 2 | 0.8912 | 0.8830 | −0.0082 | 0/2 |
| Pneumonia | RoentGen-v2 | 2 | 4 | 0.3012 | 0.1497 | −0.1514 | 2/4 |
| Pneumonia | Sana | 2 | 4 | 0.5535 | 0.5728 | +0.0192 | 2/4 |

RoentGen 的肺炎平均下降主要受 case_082 seed 0 的大幅变化影响；该位置没有
被当作异常值排除。同一病例另一个 seed 的差值为正，不能把均值解释为所有
RoentGen 图像都变差。Sana 的两例肺炎各呈同向但彼此相反的变化，也不支持
按模型名称固定决定是否前置。不同 finding 的分数不合并，不进行模型临床排名。

只有三个 EHR，12 组配对不等于 12 个独立患者；没有患者级显著性或临床
confidence interval。诊断是生成约束，不是这张 synthetic CXR 的 ground truth。
分数变化还可能反映观察器偏差，因此既不宣称 repair success，也不通过改 scorer
或 threshold 将下降结果转换成改善。

## 输入边界与执行完整性

12/12 配对的 tokenizer input-ID hashes 不同，12/12 配对的 image-byte hashes
不同；这是输入顺序与输出变化的工程证据，不能证明疾病、解剖或图像质量正确。
RoentGen 的 native prompt token counts 为 17/27/32，Sana 为 20/34/38，均在
原生预算内。Sana 的 12 个位置记录原生文本转换，RoentGen 的 12 个位置未记录
文本转换；没有改官方生成参数。

用户在完整脚本与资源申请展示后明确批准提交。作业 `12884193` 在 debug 分区
A40 节点 b11-09 完成，Slurm `COMPLETED`、exit `0:0`、elapsed 1m44s。
请求为 debug,gpu 分区、v100|a40 feature、1 GPU、2 CPU、32G RAM、15 分钟
上限，禁止 requeue。实际计入 48 次 inference attempts：24 次 generation、
24 次 XRV verification；3 次 model loads、3 个 serial workers，0 失败、
0 未完成 reservation、0 重试。历史 EHR 生成成本不计入这 48 次。

记录的最高 PyTorch peak allocated VRAM 为 RoentGen 3.02 GiB、Sana 6.98 GiB，
不是整个进程或 CUDA reserved memory 上限。CPU allocation `12879534` 的
只读 postflight 通过：重核 64 个 artifact hashes，重算全部 12 个配对行，
复核 24 个位置及 image-score 绑定、原 source/input pins、checkpoint 与预处理
标识、逐模型 reservations、dispatch 和 summary 算术。文件 0660、目录 2770、
CARC NFS project GID65534 检查通过。检查不解码像素或读取 EHR/report bodies，
不读取大型权重。执行前的 28 项新增 fixtures 与全部 4,058 项 V1.2 tests 通过。

输出：
`artifacts/protected/tricompose_v1_2/new_case_prompt_order_runs/new_order_12884193/`。
`paired_readout.csv` 保存精确分数，`summary.json` 保存状态与费用计数；PNG、
tokenizer traces、private logs 和 journals 仅留在 protected。
Manifest SHA256：`fd8487995ae84d1a2ea3a6ecefcf742fbbc6f7e11ab41040db45b82109a39134`。
Plan manifest SHA256：`d54380e9640179d629d341fd20d89ed8ac3ba3703c6eb180cc71fb4b8eea06e8`。
Protocol：`docs/new_case_prompt_order_protocol.md`。Consumed code、tests、protocol、
plan 和旧 outputs 保持不变。

## 下一步边界

保留 original 与 findings_first 两种候选，不按本轮观测写死 model-specific
默认策略。下一项三模态评测应固定这些新图，生成 hash-bound 报告并分别检验
EHR 条件支持、image-report support/contradiction、报告结构与验证分歧，再比较
固定路径、静态选择与受限动态动作。不能用更多相关 report votes 充当独立图像
真值，也不能仅靠 XRV 得分证明错误模态定位或 Agent 优势。
任何新报告、观察器或 GPU 作业都需另行冻结设置、展示完整脚本并获明确批准。

## English summary

All 24 images and 24 unchanged XRV observations completed on one A40 in 1m44s,
with 48 charged inference attempts, three model loads and no failure or retry.
Across three fixed remaining synthetic EHRs, findings-first increased RoentGen
effusion readouts in both pairs, but increased pneumonia in only two of four
pairs (mean delta −0.1514). Sana pneumonia also increased in two of four pairs
(mean delta +0.0192), while effusion decreased in both. All paired tokenizer
and image hashes differed. This does not support a universal or automatically
model-specific ordering policy. The current renderer and source pool differ
from the earlier legacy experiment. These are uncalibrated observer readouts,
not clinical truth, accepted triple repairs or evidence of LLM advantage.

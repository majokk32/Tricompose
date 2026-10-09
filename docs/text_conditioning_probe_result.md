# 冻结生成模型的文本条件响应诊断

2026-10-09。本次手写文本对照在 RoentGen-v2 和 Sana 上完成了全部 8 张图与
8 次 XRV 评分。4 组配对的 tokenizer 输入 ID 哈希、图像字节哈希均不同；
肺炎与实变的原始 XRV 分数均随阳性描述提高。不过 RoentGen 的 seed 1
差值较小，两个 seed 不足以证明稳定响应或给模型排名。这是条件输入与响应的
工程诊断，不是临床准确率、EHR 忠实度或自动修复成功的证据。

## 固定实验设置

两个完全手写的 radiology-style 文本分别描述有或无肺炎相关实变，不来自
任何患者 EHR、真实报告或目标图像。两者都作为正常的 main prompt 输入，
阴性描述不是 Diffusers 的 `negative_prompt` 参数。

每个模型使用 seeds 0 和 1，每个 seed 运行两个文本。每组内部固定 checkpoint、
seed、precision 与原生推理设置，只改变 main prompt。RoentGen-v2 使用
75 steps、CFG 3、512×512；Sana 使用 20 steps、CFG 4.5、1024×1024。
原生 Sana 前缀与文本规范化保留，所有模型冻结、分别加载。未生成报告，
未运行 LLM 调度、训练、下载、API 调用、重试或阈值优化。

## 配对数值结果

下表为原有 XRV 的 operating-point-normalized 分数，不是经过校准的疾病
概率。差值为阳性文本分数减去阴性文本分数；保留全部 seed，不挑最佳结果。

| 模型 | Seed | 肺炎阳性文本 | 肺炎阴性文本 | 肺炎差值 | 实变阳性文本 | 实变阴性文本 | 实变差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoentGen-v2 | 0 | 0.8870 | 0.5191 | +0.3680 | 0.5531 | 0.2313 | +0.3218 |
| RoentGen-v2 | 1 | 0.3818 | 0.3724 | +0.0094 | 0.3616 | 0.3156 | +0.0460 |
| Sana | 0 | 0.5292 | 0.0486 | +0.4805 | 0.7550 | 0.1035 | +0.6515 |
| Sana | 1 | 0.5727 | 0.0187 | +0.5541 | 0.6355 | 0.1096 | +0.5259 |

输入 ID 的变化证明文本变化抵达 tokenizer 边界，不能证明模型注意力正确
使用了该事实。图像字节变化证明输出发生变化，不能证明解剖或疾病正确。
XRV 的方向性响应还可能包含分类器偏差。该手写对照不能替代真实匹配数据验证，
也不能算作已修复的 EHR-conditioned triple。

## 执行与完整性

完整 batch 脚本和资源申请展示后，用户明确批准提交。作业 `12864717` 在
`gpu` 分区的 V100 节点 `d11-03` 完成，耗时 **3 分 35 秒**，Slurm exit
`0:0`。申请为 1 GPU、2 CPU、32G 主机内存、15 分钟上限。

8 个生成位置、8 个 classifier 位置全部完成；实际计入 16 次 inference
attempt 和 3 次 model-load attempt，0 失败、0 未完成 reservation、0 重试。
报告、planner 与原始病例采集成本不在这个独立诊断的调用数中。

CPU allocation `12851223` 的 metadata-only 检查通过：重核输出 manifest
所有 artifact 哈希，重算 4 个配对行，复核逐模型 reservation 的唯一性、
次数上限与 summary 算术。文件 0660、目录 2770 检查通过。该检查只读数值
metadata 和字节哈希，不解码图像，也不读取原始患者数据、生成报告或模型权重。
执行前 16 个新增 fixture tests 与完整 4,004 项 V1.2 测试通过。

输出位置：
`artifacts/protected/tricompose_v1_2/text_conditioning_probe_runs/text_probe_12864717/`。
其中 `paired_readout.csv` 保存精确配对数值，`summary.json` 保存状态与调用计数；
PNG、tokenizer traces、native logs 和 journals 保留在 protected 目录。
输出 manifest SHA256：
`aa3c981b79eb4c5a17ba6e05e1238f58fa1a89c1dd9c62e9e2db1828123c79d9`。

原始 EHR、facts、prompts、评分器、阈值、selected triples 与 consumed
代码及协议均未更改。可复现协议：`docs/text_conditioning_probe_protocol.md`。

## 下一步的边界

下一项可检验的假设是：原 EHR 文本表达与 hand-authored radiology-style
表达的差异，可能影响条件响应。先固定原来的 synthetic EHR 与 evidence-grounded
facts，只准备可追溯、临床含义不变的另一种表达，与原 prompt 做同模型、同 seed
比较；不得添加疾病、阴性、设备、位置、严重度或为了过关而改变 EHR。
下一次 GPU 对照仍须展示完整脚本与资源申请，并获得新的明确批准。

## English summary

All eight images and eight unchanged XRV observations completed on one V100
in 3m35s, with 16 charged inference attempts, three model loads and no retry.
All four paired tokenizer/image hashes differed, and both raw finding scores
increased under the invented positive main prompt. One RoentGen seed had a
small difference. These are interface-response readouts, not calibrated
probabilities, clinical truth, repaired EHR triples or evidence of LLM advantage.
Original EHRs, facts, selected triples and scoring rules remain unchanged.

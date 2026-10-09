# 固定 EHR prompt 顺序对照的提交准备

2026-10-09。CPU staging 已完成，GPU 尚未提交。固定原两个 synthetic EHR、
两种已部署的冻结 CXR 模型、seeds 0 和 1，比较封存原文字句与 finding block
前置后的文字句，共 16 个生成位置、8 组配对，最多 16 次 XRV 评分。
本轮没有新 report、planner、API、训练、下载或 winner 替换。

## 固定内容与历史版本限制

准备检查发现旧 prompt 与当前 renderer 版本不同，并包含旧 CHF-to-CXR prior。
因此不重建或替换旧 prompt，而是从 SHA256 绑定的原文复制完整 finding block，
只移动它的位置。8 个 final prompt 文件的 hash、4 个原文与前置文本的词 multiset、
原 request 与 source metadata、源 EHR/facts hash、evidence refs 均复核通过。
旧 CHF image prior 在两条分支保持不变，但明确标为未经验证；临床背景不证明
影像征象，本实验不能宣称所有历史 prompt 断言都由 EHR 直接支持。
当前 renderer 仅记录历史差异，不用于本次 generation。

前两个未完成的 staging 尝试因 renderer mismatch 拒绝并清理各自私有临时
目录；未改 EHR、facts、源 prompt、scorer、threshold 或旧 outputs，也未换病例。
成功计划采用新的 opaque run，不覆盖已有 run。

CPU allocation `12879534` 内执行：26 个新增 authored-fixture tests 通过，
完整 **4,030 项 V1.2 测试通过**，32.729s。准备及复核没有模型 factory/load、
大型权重字节读取、真实 EHR/报告/图像/target 读取、API 或 sbatch。
文件 0660、目录 2770 与 CARC NFS GID65534 访问边界通过。

## 封存文件

Entry：`TriCompose-v1.2/agent/run_fixed_ehr_prompt_order.py`。
SHA256：`52263a8840de0169cdd96d8c222a8338f8812db26af099474473fa6784003cb6`。

Protected plan：
`artifacts/protected/tricompose_v1_2/fixed_ehr_prompt_order_plans/prompt_order16_12879534_003/`。
Manifest SHA256：
`f45718f3a2917ec24fd06c4b8551fd84b958528e9790511d4262c924d3bc06c7`。

完整脚本：
`TriCompose-v1.2/agent/slurm/21_fixed_ehr_prompt_order16_gpu_flexible.sbatch`。
SHA256：`65eb5221bffd83ee8471e90bd99e157e0d2f267240828257f64332e5a6f840ee`。
Batch shell syntax 检查通过。

## 资源申请修订与批准

初始 protocol 提议 debug A40；准备期间再次查询发现该卡已被占用。
此处仅修订 scheduling scope，不改已封存的 clinical/text protocol、模型参数、
case/seed/arm 或 source pins。新脚本申请 `debug,gpu` 分区、1 GPU、
`v100|a40` node-feature constraint、2 CPU、32G 主机内存、15 分钟上限。
两种模型原生 worker 已在 V100/A40 上完成过推理；本次环境/kernel 运行仍须在
批准的 GPU allocation 内验证。`--no-requeue` 禁止自动重新排队重跑。

CARC `sinfo` 确认 node features 和 GRES 对应 V100/A40，分区均允许该请求的
时长与资源形式。Slurm 支持多个 partition 与 OR node-feature constraints，
可扩大候选节点范围；这不保证立即启动或一定优于某个单分区队列。
语法依据：[Slurm 官方 sbatch 文档](https://slurm.schedmd.com/sbatch.html)。

只有完整展示新脚本和资源申请、获得新的明确批准之后才能执行：

```bash
sbatch TriCompose-v1.2/agent/slurm/21_fixed_ehr_prompt_order16_gpu_flexible.sbatch
```

新输出将位于
`artifacts/protected/tricompose_v1_2/fixed_ehr_prompt_order_runs/prompt_order_<job>/`。
保留全部生成位置、8 个配对行、raw XRV 分数、tokenizer traces、images 与
charged journals；clinical accuracy/acceptance 不成立，缺失值保留 NA。
解释边界与固定实验设置见 `docs/fixed_ehr_prompt_order_protocol.md`。

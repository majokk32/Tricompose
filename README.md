# TriCompose

TriCompose 组合冻结的现有模型，生成并验证 structured EHR、胸部 X-ray 和
radiology report 三模态候选。

## Current research status — September 2026

The current main line is the fully synthetic V1.0/V1.1 pipeline, not the early
Phase-0 real-anchor smoke test documented later on this page.

| Version | Current role | Status |
|---|---|---|
| V1.0 | Frozen generation graph and static best-of-N baseline | Implemented engineering baseline; evaluators are not clinically calibrated |
| V1.1 | Prompt de-collapse plus explainable three-edge evaluation | Substantially implemented; calibration and paper-primary validation remain |
| V1.2 | Error localization and targeted regeneration | Planned, not yet validated |
| V1.3 | Paper-grade scale-up, independent evaluation, and human review | Planned |

Current documentation:

- [`docs/version_roadmap_zh.md`](docs/version_roadmap_zh.md): 中文主逻辑；如果只看
  一份文件，优先看这个；
- [`docs/version_roadmap.md`](docs/version_roadmap.md): accurate version
  boundaries, completion gates, CARC execution order, and server-agent prompt;
- [`TriCompose-v1.0/README.md`](TriCompose-v1.0/README.md): immutable V1.0
  generation graph and static baseline;
- [`TriCompose-v1.1/README.md`](TriCompose-v1.1/README.md): repaired
  EHR-to-CXR conditioning and V1.1 execution contracts;
- [`TriCompose-v1.0/eval/report_v1_1/README.md`](TriCompose-v1.0/eval/report_v1_1/README.md):
  unknown-safe EHR-CXR, EHR-report, and report-CXR evaluation;
- [`tricompose_evaluation_research_20260812.md`](tricompose_evaluation_research_20260812.md):
  literature-backed metric design.

All deployed report generators in the aligned V1/V1.1 pool are
CXR-conditioned. `CXRMate-single` is not CXRMate-ED, so an independent-path
error-localization claim is reserved for V1.2 and requires either a true
EHR+CXR report path or controlled-injection validation.

Package version `0.0.1` and research protocol versions V1.0/V1.1 are separate
version axes.

## Historical Phase-0 context

早期 Phase-0 首先实现了以下 real-anchor longitudinal smoke path：

```text
真实来源的既往 CXR + 既往报告哈希特征 + 区间 EHR 哈希特征
    -> EHRXDiff
    -> synthetic follow-up CXR
    -> UniDisc
    -> synthetic radiology report
```

这不是严格的纯 `EHR + CXR -> CXR`，也不是 fully synthetic generation。
当前准备好的 EHRXDiff 条件包含既往报告，并使用本地 hashing placeholder
生成 1536 维表示；因此本阶段只验证接口、资源和隐私边界，不能用于临床结论。

## EHR-to-prompt-to-CXR baselines — Aug 4, 2026

### RoentGen-v2：已成功跑通

RoentGen-v2 已使用官方冻结 checkpoint 和官方 Diffusers pipeline 完成部署，
并跑通以下无训练路径：

```text
protected structured EHR facts
  -> deterministic radiology-style prompt
  -> frozen RoentGen-v2
  -> synthetic CXR
```

当前实现保持官方 demo 的主要推理设置：BF16、CFG 3、75 inference steps 和
`512 x 512` 输出。现有结果已经达到作为 Phase-0 text-conditioned CXR
candidate generator 和 baseline 的要求，因此不再继续进行 prompt、CFG、seed
或 selector 优化实验。

这条路径是显式的 `EHR -> text prompt -> CXR`，不是 direct structured
`EHR -> CXR`。生成阶段不读取 matched real CXR 或 source radiology report，
也没有训练、fine-tuning、LoRA 或 checkpoint 修改。代码和运行说明位于
`experiments/roentgen_v2/`，受保护输出统一保存在
`artifacts/protected/roentgen_v2/`。

### MeDiM：链路跑通，但仅作参考

MeDiM 的以下冻结推理链路也已完成工程验证：

```text
protected structured EHR facts
  -> deterministic report-like prompt
  -> frozen MeDiM
  -> synthetic CXR
```

该实验对照了仓库中的官方 text-to-image / mask-image-only 接口，但在当前
EHR-derived prompt 条件下，生成质量和稳定性没有达到主 baseline 的要求。
因此 MeDiM 不作为当前成功的 EHR-to-prompt-to-CXR 方案，只保留为模型接口、
prompt adapter、Slurm 运行和 evaluation 实现的参考。相关代码位于
`experiments/medim/`，受保护产物保留在
`artifacts/protected/medim/runs/`。

### UniDisc 与 Liquid：独立模型实验

UniDisc 和 Liquid 的模型专用 runtime、adapter、README 和 Slurm 分别位于
`experiments/unidisc/` 与
`experiments/liquid/`；公共的确定性 EHR 文本桥位于
`src/tricompose/ehr_prompt_cxr/`。其受保护输出分别保存在
`artifacts/protected/unidisc/` 和 `artifacts/protected/liquid/`。

以上两个实验均不能被表述为 direct structured-EHR-conditioned CXR
generation；它们都是 deterministic EHR-to-text adapter 加冻结文本条件图像
生成器。

## Layout

```text
src/tricompose/       安全 I/O 与模型适配器
experiments/<model>/  按模型分类的 adapter、runtime、tests 与 Slurm
slurm/                完整但不会自动提交的 Slurm 作业
docs/                 实验契约和运行说明
logs/slurm/<model>/   按模型分类的 sanitized Slurm 日志
artifacts/protected/<model>/  按模型分类的患者衍生产物，目录权限 2770
runtime/              运行时缓存，禁止写入外部模型目录
.cache/<model>/       按模型分类的框架与 checkpoint 缓存
.tmp/<model>/         按模型分类的作业临时文件
```

完整运行说明见 [`docs/first_path.md`](docs/first_path.md)。
统一的模型分类存储结构见 [`docs/storage_layout.md`](docs/storage_layout.md)。
统一 report metrics、Qwen-VL verifier 和 agent score contract 见
[`docs/scoring.md`](docs/scoring.md)。
快速连接 OpenAI-compatible LLM policy 见
[`docs/llm_agent.md`](docs/llm_agent.md)。

## Weekly report — Jul 30, 2026

### 本周目标

先跑通一个最短的 frozen-model composition baseline。TriCompose 的重点是生成
多个候选并选择当前最好的 output；模态循环不是必需步骤，
`Report -> CXR` cycle 只作为后续可选 verifier。

### 已跑通的短路线

```text
real longitudinal anchor
  (previous CXR + interval EHR condition)
    -> EHRXDiff
    -> synthetic follow-up CXR
         |-> UniDisc   -> report candidate A
         `-> LLaVA-Rad -> report candidate B
```

UniDisc 和 LLaVA-Rad 使用的是 EHRXDiff 在同一次 run 中生成的同一张 CXR，
不是真实 target CXR。Phase-0 没有加载真实 target CXR 或真实 target report。

### Sanitized run result

Run ID：`real_smoke_003`

| Stage | Slurm job | State | Runtime | Peak GPU memory | Protected output |
|---|---:|---|---:|---:|---|
| EHRXDiff | `10731081` | `COMPLETED` | 51 s | 5.904 GiB | `artifacts/protected/real_smoke_003/stage1_ehrxdiff/generated_cxr.png` |
| UniDisc | `10731429` | `COMPLETED` | 31 s | 5.916 GiB | `artifacts/protected/real_smoke_003/stage2_unidisc/generated_report.txt` |
| LLaVA-Rad | `10732055` | `COMPLETED` | 46 s | 14.342 GiB | `artifacts/protected/real_smoke_003/stage2_llavarad/generated_report.txt` |
| Qwen2.5-VL verifier | `10733757` | `COMPLETED` | 17 s | 15.565 GiB | `artifacts/protected/real_smoke_003/verification/qwen25vl_v1/qwen_vl_scores.json` |

### Outputs for CARC review

包含 synthetic CXR、两份完整英文输出、中文翻译和初步候选比较的
中英双语 protected weekly report：

`artifacts/protected/weekly_reports/jul30_2026_bilingual/weekly_report.md`

包含两个候选各自的文本指标、Qwen 分数和 Agent 决策的评分版：

`artifacts/protected/weekly_reports/jul30_2026_scored_bilingual/weekly_report.md`

面向周会汇报、聚焦已完成工作与下一步计划的中英双语进展版：

`artifacts/protected/weekly_reports/jul30_2026_progress_bilingual/weekly_report.md`

原始英文版保留在：

`artifacts/protected/weekly_reports/jul30_2026/weekly_report.md`

该文件只能在 CARC 工作区内由获授权用户打开。患者衍生的图像和报告正文
不复制到普通 README，也不写入公开 Slurm 日志或 Git。

生成 CXR 的固定尺寸为 `256 x 256`，文件大小为 `80,577 bytes`，
SHA-256 为
`67751bc9fcb7461c1b42a8f4d06dacf18a12b58d7e6e50ea323469551c702c3a`。

生成 report 文件大小为 `1,891 bytes`，SHA-256 为
`d601895cd8ecd8ea3c52a90b89bed799a14e07f4986f9e8cb70bd26e5f73df37`。

LLaVA-Rad report candidate 文件大小为 `485 bytes`，SHA-256 为
`67c3da58d5516569510ffb8280e943d4cb2e0e9ac62386618783261343bc09e6`。

所有患者衍生产物均位于 `artifacts/protected/`。该历史 Phase-0 run 创建时采用
owner-only `0700/0600`；当前协作策略采用 CARC 项目组 `2770/0660`，并可通过
`tools/set_collaboration_permissions.sh` 迁移。README 和 Slurm 日志不包含图像、报告正文、患者标识符
或源数据键。需要人工查看时，只能由获授权用户在 CARC 内从上述 protected
路径直接打开，不得复制到公开日志或聊天。

### 当前结论与限制

- 三个冻结模型的端到端接口、Slurm 资源、离线 checkpoint 和 protected I/O
  已经连通。
- 同一张 synthetic CXR 已经生成两个 report candidates；Qwen2.5-VL
  分别给 UniDisc 和 LLaVA-Rad `0.0`，因此当前没有合格 winner。
- Phase-0 selector 返回 `verify_more`，没有在分数并列且低于阈值时强行选取。
- BLEU/ROUGE-L/METEOR 当前比较两个生成候选，只表示 peer agreement，
  不是无真实 reference 情况下的单候选准确率。
- 当前 EHRXDiff 适配使用 1536 维 hashing placeholder；本次结果只能证明
  工程链路可运行，不能据此判断 EHR conditioning 或临床一致性。
- Qwen judge 与 selector 阈值尚未校准，以上结论只用于 Phase-0 诊断。

### 下一步

保持同一个 real anchor，不额外生成 EHR：

1. 为 EHRXDiff 运行多个 seed，得到多个 synthetic CXR candidates。
2. 将每张候选 CXR 分别交给 UniDisc 和 LLaVA-Rad，得到 report candidates。
3. 增加独立的 CXR–Report fact verifier，或生成第三个 report candidate，
   处理当前 `verify_more` 动作。
4. 在 matched/shuffled/hard-negative pairs 上校准 Qwen judge 与 selector
   阈值，再评估 static selection 和 cost-aware dynamic router。

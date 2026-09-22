# TriCompose 中文版本路线与研究逻辑

更新时间：2026-09-21
[English version](version_roadmap.md)

这份文件是当前项目的**唯一主逻辑入口**。其他 README 主要用于记录具体代码、
服务器运行方式和历史实验，不需要同时全部阅读。

特别注意：

- 仓库里存在代码或 Slurm 脚本，只表示“已经实现或准备好接口”；
- 只有在 CARC 上实际生成了完整 protected artifacts，并通过校准和独立评价，
  才能说对应实验已经完成；
- 下文会明确区分“已实现”“已运行”和“尚未验证”。

## 1. 项目最终要解决什么问题

TriCompose 使用冻结的现成模型，生成一个完整的 synthetic patient：

```text
Synthetic structured EHR
    ↓
EHR 中有证据支持的 radiology facts 和 model-specific prompt
    ↓
多个 Synthetic CXR candidates
    ↓
多个 Synthetic Report candidates
    ↓
跨模态检查、选择或定向修复
    ↓
一个最终 (EHR, CXR, Report) triple
```

真正的研究问题不是“能不能把模型串起来”，而是：

> 在不训练新生成模型的情况下，能否通过 inference-time candidate
> composition 和跨模态验证，生成比固定单一路径更一致的
> Synthetic EHR–CXR–Report triple，并控制额外计算成本？

因此：

- 所有生成模型保持冻结；
- 第一阶段不训练 router；
- 不需要模糊的 LLM Agent；
- 主要方法应当是确定性的、可复现的、可以追踪每个决定依据的。

## 2. 所有版本都必须遵守的规则

1. **Synthetic EHR 一旦固定，就不能因为下游结果不好而换掉。**
2. 必须保存完整 lineage：EHR、facts、prompt、CXR、report、模型版本、seed、
   parent IDs、SHA256、runtime 和 cost。
3. 所有 finding 使用四种状态：

   ```text
   positive / negative / uncertain / unknown
   ```

4. `unknown` 不等于 negative，也不能算 agreement。
5. EHR evidence 必须区分：

   ```text
   direct / compatible / risk_only
   ```

6. 只有明确的 `direct positive/negative` 证据才能构成强矛盾。例如 CHF、
   furosemide 或低氧只是 compatible/risk evidence，不能直接推出当前一定有 edema。
7. 必须区分：

   - **Real-anchor evaluation**：有真实 CXR/report，只在评价时使用；
   - **Fully synthetic evaluation**：没有唯一真实答案，评价内部一致性、分布、
     utility、diversity 和 privacy。

8. GREEN、标准 RadGraph-F1、RadCliQ、RadFact、VLScore 等需要真实 reference
   的指标，不能拿来做 fully synthetic inference-time selection。
9. FID、KID、PRDC 是 cohort-level 指标，不能用于挑选单张 CXR。
10. selection scorer 与最终 held-out evaluator 应尽量分开，防止对 evaluator
    过拟合。
11. 旧 candidate bank 和实验结果不能覆盖。每次运行使用新的 run ID 和 manifest。

## 3. 一眼看懂版本划分

| 版本 | 主要目的 | 当前状态 |
|---|---|---|
| 历史 Phase-0 | 验证 real-anchor、模型接口、Slurm 和隐私边界 | 仅作历史记录，不是当前主论文逻辑 |
| V1.0 | 多模型生成候选并做 static best-of-N | 工程流程已完成；scorer 尚未临床校准 |
| V1.1 | 修复 prompt collapse，建立可解释 evaluation | 大部分代码已实现；还缺校准和论文级验证 |
| V1.2 | 定位错误模态并定向重新生成 | 计划阶段，尚未形成验证过的方法 |
| V1.3 | 论文级扩规模、独立评价和人工评价 | 计划阶段 |
| V2.0（可选） | 更一般的 cost-aware/learned composition policy | 只有 V1.2 证明有必要才做 |

之前的讨论曾把“错误定位 + 定向修复”整体叫 V1.1。但是仓库现在已经使用
`TriCompose-v1.1` 表示 prompt repair 和 evaluation foundation，所以从现在开始：

```text
V1.1 = prompt repair + explainable static evaluation
V1.2 = error localization + targeted regeneration
```

不要重命名旧 run；只需要在新文档和新实验中遵守这个定义。

## 4. V1.0：静态组合基线

### 4.1 V1.0 已经实现什么

```text
Synthetic EHR candidates
    ↓
deterministic EHR-to-radiology text
    ↓
RoentGen-v2 / Sana / PixArt
    ↓
MAIRA-2 / CXRMate-single / LLaVA-Rad / CheXagent-2
    ↓
XRV / BioViL-T / Qwen2.5-VL / deterministic rules
    ↓
static best-of-N selection
```

已经实现：

- SynEHRgy synthetic EHR candidates；
- EHR→radiology facts/prompt bridge；
- 三个 CXR generation branches；
- 四个 Report generation branches；
- hash-bound candidate graph；
- fixed path 和 exhaustive static reranking；
- atomic triple export；
- XRV、BioViL-T、Qwen 和 rule-based engineering evidence。

准确的模型描述是：

- 当前 CXR 模型读取的是 EHR-derived radiology text，不是原生 structured EHR；
- `CXRMate-single` 只读取 CXR，不是 `CXRMate-ED`；
- CheXGenBench 是框架/benchmark，Sana 和 PixArt 才是具体 generator 配置。

### 4.2 V1.0 证明了什么

V1.0 证明：

- 三模态 candidate graph 可以真实跑通；
- lineage、hash、模型、seed 和父子关系能够保存；
- fixed baseline 和 static selection 能够输出完整 triple；
- scoring 可以改变最终选择的模型路径。

### 4.3 V1.0 没有证明什么

- 两个不同 EHR 被压缩成相同 CHF-only prompt，因此下游图像和报告重复；
- Qwen positional 14-finding vector 不完整，只保留了 scalar；
- XRV、BioViL-T、Qwen 和 lexical scores 尚未临床校准；
- selected report 仍出现错误的多视图描述；
- 一个 unique prompt 不能证明 cohort-level superiority 或 diversity。

所以 V1.0 的准确结论是：

> 已跑通 frozen-model generation、lineage 和 static selection 的工程流程，
> 但还没有证明临床最优或大规模有效。

V1.0 应保留为不可修改的历史 baseline。新 evaluator 可以重新评价旧 bank，
但必须写入新的 evaluation run，不能覆盖旧结果。

## 5. V1.1：Prompt 修复与 Evaluation Foundation

### 5.1 为什么不能直接开始 targeted repair

在自动判断“CXR 错了还是 Report 错了”之前，系统必须先可靠回答：

1. 每个模态到底表达了哪些 finding？
2. 两个模态之间是 support、contradiction，还是信息不足？
3. 一个分数是单病例 selection metric，还是 cohort evaluation metric？

如果这些问题没有解决，所谓“自动修复”只是根据不可靠的 scalar 反复生成。
因此 V1.1 先建立 measurement/evaluation layer，而不是做动态 Agent。

### 5.2 V1.1 Prompt repair 已经做到什么

`TriCompose-v1.1/` 已经包含：

- 新的 EHR facts 和 prompt manifest schema；
- direct radiographic facts 与 longitudinal clinical context 分离；
- RoentGen-v2、Sana、PixArt 的 model-specific prompts；
- prompt token limit 检查；
- prompt/fact fingerprints 和 duplicate 检查；
- hash-bound CXR/report request 与 candidate contract；
- non-overwriting execution adapter；
- 相应的 tests 和 Slurm scripts。

已经记录的 80-case CPU bridge run 显示：

- 80/80 canonical EHR hashes 保持不变；
- 80/80 staging 成功；
- V1.0 每个模型只有 7 种 prompt；
- V1.1 增加到 49 种 grounded clinical intents；
- 20 个 EHR 没有足够 radiology evidence，因此诚实地保留 neutral prompt，
  没有用 case ID 或随机文字制造假 diversity。

这说明 prompt collapse 明显改善，但仍然只证明“conditioning text 更丰富”。
它还没有证明 CXR 的临床内容真的随 EHR 变化。

### 5.3 V1.1 Evaluation 已经实现什么

虽然目录仍叫：

```text
TriCompose-v1.0/eval/report_v1_1/
```

但其中实际是 V1.1 evaluation 代码。目录暂时不移动，是为了避免破坏已有脚本和
protected run path。

现有实现包括：

- Report structural quality；
- CheXpert-style 14 named findings；
- `positive/negative/uncertain/unknown` 语义；
- XRV CXR label extraction；
- CheXbert Report label extraction 入口；
- EHR–CXR consistency；
- EHR–Report consistency；
- Report–CXR consistency；
- 每条边的 support、explicit contradiction 和 coverage；
- CHF/diuretic weak evidence 与 direct evidence 分开；
- BioViL-T 和 Qwen 只保留为 secondary evidence；
- 80 cases × 3 CXR models × 4 Report models = 960 lineage rows 的 registry；
- fixed paths、deterministic random 和 exhaustive static reranking；
- validity/contradiction 优先的 lexicographic selector。

代码还区分：

```text
pending evidence = null
not applicable = no comparable EHR fact
unknown = source没有明确表达
```

这三种情况不能混为一个零分。

### 5.4 V1.1 还缺什么

1. 在服务器上核对 pool80/pool160 的哪些 CXR/Report jobs 已实际完成。
2. 确认 XRV、CheXbert、BioViL-T、Qwen 的确切 checkpoint/revision 和 coverage。
3. 在独立 real matched validation set 上校准每个 finding 的 CXR thresholds。
   默认 0.5 threshold 只能叫 diagnostic，不能作为论文主结果。
4. 建立：

   - random patient swaps；
   - same-disease hard negatives；
   - minimal finding perturbations。

5. 验证 prompt repair 是否真正改变 CXR clinical content 和 diversity，而不只是
   prompt hash 变多。
6. 在完全相同 cohort 上比较：

   - predetermined fixed path；
   - random selection；
   - 每个 fixed model path；
   - exhaustive static reranking。

7. 输出一个 protected evaluation report，包含三条边、coverage、quality、runtime、
   calls 和 reject/incomplete rate。

### 5.5 V1.1 完成标准

只有同时满足以下条件，才能说 V1.1 完成：

- 每个 prompt collapse 都被修复或明确标记；
- 三条边都输出可追踪的 per-finding evidence；
- unknown 和 weak evidence 不会产生强 negative；
- thresholds 在 final test 前冻结；
- fixed/random/static baselines 能公平比较；
- 实验包含多个 unique clinical intents，而不是重复病例。

V1.1 **不包括** targeted regeneration，也不包括 LLM router。

## 6. V1.2：错误定位与定向修复

### 6.1 V1.2 真正要证明什么

Static reranking 只能从已经生成的候选里选。V1.2 研究：

> 能否根据跨模态 disagreement 判断更可能是 CXR 还是 Report 出错，
> 然后只重新生成对应模态，以更少计算达到接近或超过 exhaustive selection
> 的一致性？

### 6.2 当前最大限制：Report 路径并不独立

当前四个 Report generator：

```text
MAIRA-2
CXRMate-single
LLaVA-Rad
CheXagent-2
```

都从同一张 CXR 生成报告。它们彼此一致，可能只是共同跟随了同一张错误 CXR，
不能独立证明“CXR 是错的”。

因此，在做强 error-localization claim 之前必须选择：

1. 接入真正的 `EHR + CXR → Report` 路径，例如 CXRMate-ED；或者
2. 明确把 localization 称为 heuristic，并通过 controlled error injection、
   held-out evaluator 和人工评价证明它的准确率。

绝对不能把 `CXRMate-single` 写成 `CXRMate-ED`。

### 6.3 V1.2 计划

首先建立 corruption benchmark：

- random patient swap；
- same-disease hard negative；
- wrong CXR replacement；
- wrong Report replacement；
- presence/absence、negation、laterality、severity、device、temporal flips。

然后实现确定性动作：

```text
select
regenerate_cxr
regenerate_report
switch_report_model
verify_more
reject
```

基本规则：

```text
EHR + independent reports agree, CXR disagrees
    → regenerate_cxr

EHR + CXR agree, one Report disagrees
    → regenerate_report / switch_report_model

Reports disagree
    → 选择同时更符合 EHR 和 CXR 的 Report；否则 verify_more

证据不足或预算耗尽
    → reject
```

每次 action 必须保存：哪个 finding、哪条 edge、什么矛盾触发了决定。

### 6.4 V1.2 实验对照

- fixed single path；
- random selection；
- exhaustive static reranking；
- generate-all oracle analysis；
- targeted repair。

主要指标：

- error-localization accuracy；
- abstention 和 false-repair rate；
- regeneration success rate；
- 三条边的 support/contradiction/coverage；
- any-strong-contradiction rate；
- minimum-edge consistency；
- CXR/Report quality 和 diversity；
- reject rate；
- 平均模型调用次数与 GPU time。

核心不是只看 consistency，而是比较 **quality–compute trade-off**。

如果 static reranking 在相同或更低成本下已经一样好，就不应该强行继续讲
targeted repair；论文可以退回到更可靠的 static composition contribution。

## 7. V1.3：论文级完整实验

V1.3 原则上不再频繁修改核心算法，而是完成独立、完整的实验验证。

### EHR自身

- schema/clinical validity；
- marginal、dependency 和 temporal fidelity；
- TSTR/TRTS 或选定 downstream utility；
- 如果声称可共享 synthetic cohort，再做 privacy/memorization。

### CXR自身

- RadDINO-FID/KID/PRDC；
- pathology-stratified results；
- artifact 和 duplicate rate；
- downstream utility；
- blinded radiologist review。

### Report自身

- Real-anchor：GREEN + RadGraph-F1 + CheXbert-F1；
- 预算允许且实现稳定时加入 RadFact、VLScore、RadCliQ 或 SPEC-CXR；
- Fully synthetic：结构质量 + 跨模态事实，不误用 reference-based metrics。

### 跨模态和统计

- 三条 pairwise edges；
- cross-path disagreement/uncertainty；
- abnormal 和 rare-disease subgroup；
- patient-level bootstrap 95% confidence intervals；
- independent held-out evaluator；
- blinded human/radiologist review；
- 完整 calls/GPU-time/Pareto curve。

数据必须按 patient-level 分为：

```text
development/debug
calibration/threshold selection
final held-out test
human-review subset
```

看过 final test 后不能再修改 threshold、weight 或 stopping rule。

## 8. V2.0：只有必要时再做 Learned Router

只有 V1.2/V1.3 同时证明以下事实才考虑 learned policy：

- 不同模型确实存在 case-level complementarity；
- consistency scores 与独立 fidelity/人工判断相关；
- adaptive calls 相比 exhaustive generation 能节省计算；
- deterministic policy 存在一个明确、可测量的不足。

不要从 RL 开始。优先考虑：

- exhaustive oracle action sequence；
- imitation learning；
- 简单 cost-aware stopping policy。

如果 deterministic V1.2 已经足够，就没有必要为了“像 Agent”而训练 router。

## 9. 现在服务器上应该按什么顺序做

1. 对 protected V1.1 runs 做严格只读审计，不打印真实 patient-derived 内容。
2. 核对哪些 80-case CXR 和 report jobs 已完成。
3. 核对 candidate 数量、unique case、model paths、hash 和 evidence coverage。
4. 确认 XRV、CheXbert、BioViL-T、Qwen checkpoints。
5. 完成 real matched/hard-negative threshold calibration。
6. 完成 V1.1 fixed/random/static comparison。
7. 人工检查一部分 per-finding evidence 是否合理。
8. 只有 V1.1 通过 review 后，才开始 corruption benchmark 和 V1.2。
9. 所有 GPU inference 必须使用 Slurm；提交前先展示完整脚本和资源请求。

## 10. 可以直接发给服务器 Agent 的话

```text
请完整阅读：
1. docs/version_roadmap_zh.md
2. TriCompose-v1.1/README.md
3. TriCompose-v1.0/eval/report_v1_1/README.md

先进行严格只读审计，不修改文件、不下载模型、不提交 Slurm job。

请报告：
1. 当前实际存在并完成的 V1.1 protected runs；
2. candidate 数量、unique case IDs、model paths、hash 和 evidence coverage；
3. 当前可用的 evaluator checkpoints 和 threshold bundles；
4. 当前 artifacts 与 V1.1 completion gate 之间的差距；
5. 最小、按顺序排列的实施/评价计划及准确路径；
6. 准备使用的 tests 和 Slurm commands，但不要提交。

不得打印 raw EHR rows、真实报告、真实图像、患者标识符或 protected synthetic
报告/图像正文。不得覆盖旧 run。unknown 必须保持 unknown；weak clinical
context 不能产生 hard contradiction。V1.1 未经检查批准前，不要开始 V1.2
targeted regeneration。
```

## 11. 是否进入下一版本的判断标准

### V1.1 → V1.2

- 三条 edge scorer 能否区分 matched pairs 和 hard negatives？
- per-finding evidence 是否与人工检查基本一致？
- prompt repair 是否真正影响 CXR clinical content？
- static reranking 是否在多个 unique cases 上优于 fixed/random baselines？

### V1.2 → V1.3

- 是否能在 controlled corruption 中区分错误 CXR 和错误 Report？
- targeted repair 是否在相近成本下优于 static reranking？
- 修复后是否没有牺牲图像/报告质量和 diversity？

### V1.3 → V2.0

- 模型互补性是否真实且稳定？
- 是否存在明确的质量–计算 Pareto frontier？
- learned policy 是否解决了真实问题，而不是增加复杂度？

如果某个 gate 没通过，就保留已经验证的较窄贡献，不要强行进入下一版本。

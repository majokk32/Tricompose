# TriCompose 三模态与跨模态评估调研

更新时间：2026-08-12

## 0. 最重要的结论

TriCompose 不应寻找一个模糊的“总一致性分数”，而应维护一个六维 evaluation vector：

1. Synthetic EHR quality
2. Synthetic CXR quality
3. Synthetic Report quality
4. EHR–CXR consistency
5. CXR–Report consistency
6. EHR–Report consistency

目前文献中没有一个被广泛接受、可以直接评估完整 EHR–CXR–Report triple 的单一指标。EHR–CXR 和 EHR–Report 尤其没有现成标准。因此，最稳妥、也最可能成为论文贡献的方案是：

> 单模态质量沿用各领域的成熟 benchmark；三条跨模态边使用同一个 provenance-aware clinical fact contract，并在真实 matched triples、hard negatives、人工事实扰动和专家评分上校准。

必须严格区分两类评价：

- **Real-anchor evaluation：**有真实 CXR 和真实 report，可以评价 generation fidelity、clinical accuracy 和 reference-based report quality。
- **Fully synthetic evaluation：**没有唯一正确的目标 CXR/report，只能评价分布质量、内部一致性、utility、privacy 和专家可接受性，不能把 reference-free consistency 写成“ground-truth accuracy”。

另外，GREEN、RadFact、RadGraph-F1、CheXbert-F1、RadCliQ、VLScore 等常见 report 指标大多需要真实参考报告。它们不能直接作为 fully synthetic 阶段的 reference-free CXR–Report 或 EHR–Report scorer。

## 1. 建议直接采用的 evaluation matrix

| 维度 | 论文主指标 | 辅助指标 | 是否需要真实 reference | 是否适合 inference-time selection |
|---|---|---|---|---|
| EHR 自身 | schema validity；clinical-rule violation；marginal fidelity；dependency/temporal fidelity；TSTR | PRDC、missingness、privacy attacks | 需要真实 EHR cohort，不需要一一配对 | validity/rule 可用；cohort 指标不可用于单病例选择 |
| CXR 自身 | RadDINO-FID/KID；RadDINO-PRDC；pathology-stratified fidelity；radiologist realism | artifact rate、nearest-neighbour privacy、downstream utility | 需要真实 CXR cohort，不需要逐例配对 | artifact gate 可用；FID/KID/PRDC 不可用于单图选择 |
| Report 自身 | Real-anchor：GREEN + RadGraph-F1 + CheXbert-F1；专家 error count | RadFact、VLScore、RadCliQ、SPEC-CXR、BLEU/ROUGE | 主流 clinical metrics 需要真实 report | 多数不适合 fully synthetic selection |
| EHR–CXR | masked known-fact support + weighted contradiction rate | BioViL-T prompt/image；matched-vs-hard-negative retrieval | 主分数可 reference-free；必须用真实 pair 校准 | 适合 |
| CXR–Report | calibrated image-label/report-label agreement + contradiction rate | BioViL-T image/report；location/device grounding；VLM judge | 主分数可 reference-free；VLScore 等需 reference | 适合 |
| EHR–Report | provenance-aware entailment/support + explicit contradiction rate | RadNLI-style NLI；RadGraph entity/relation matching | 可 reference-free；必须人工校准 | 适合 |

论文中不要只公布一个加权总分。至少同时报告六个维度，并给出 triple 的：

- `mean edge consistency`
- `minimum edge consistency`（最弱边，推荐作为安全性指标）
- `any-strong-contradiction rate`
- `coverage`（有多少 fact 真正可判定）
- 模型调用次数、GPU 时间和 reject rate

用于 candidate selection 时，建议先设置 hard gates，再在通过 gate 的候选中排序；不要让一个很高的 BioViL cosine 抵消明确的 pneumothorax contradiction。

## 2. Synthetic EHR 自身怎么评

### 2.1 你当前使用的 SynEHRgy 原论文做了什么

[SynEHRgy（NeurIPS 2024 GenAI for Health workshop）](https://arxiv.org/abs/2411.13428)把评价分成 fidelity、utility 和 privacy：

- ICD code：single-visit unigram、bigram、trigram，以及跨 visit sequential-bigram probability 与真实数据的 Pearson correlation。
- 时间序列：基于前 48 小时统计量的 Precision、Recall、Density、Coverage（PRDC）。
- 时间序列依赖：真实与 synthetic correlation matrix 的 MSE，以及 missingness co-occurrence。
- Utility：in-hospital mortality 和 phenotype prediction；报告 TSTR 以及 real + synthetic augmentation 的 AUROC。
- Privacy：nearest-neighbour membership inference，使用 WD、JSD 和 attack AUROC。

这是与你当前 EHR generator 最直接对应的一套指标，但需要注意它是 workshop paper，而且原实验是 MIMIC-III。TriCompose 使用 MIMIC-IV-ED schema 时必须重新定义变量、时间窗和 downstream tasks，不能照搬其数值。

### 2.2 PromptEHR、HALO 和系统 benchmark 提供了什么

[PromptEHR（EMNLP 2022）](https://aclanthology.org/2022.emnlp-main.185/)提出：

- longitudinal imputation perplexity（lpl）：衡量纵向 visit coherence；
- cross-modality imputation perplexity（mpl）：衡量 diagnosis、drug、procedure、lab 等 EHR 内部模态的条件一致性；
- membership inference 与 attribute inference；
- next-event prediction recall@10/20，用于 synthetic-only 和 real+synthetic utility。

lpl/mpl 需要一个能够给 EHR sequence 计算条件概率的模型，因此更适合作为模型论文复现指标或辅助指标，不建议把生成器自己的 likelihood 当唯一质量分数。

[HALO（Nature Communications 2023）](https://www.nature.com/articles/s41467-023-41093-0)重点比较：

- code prevalence；
- 同一次 visit 内的 code co-occurrence；
- 相邻 visit 的 conditional probability；
- visit/record length 和 inter-visit temporal statistics；
- downstream phenotyping AUROC；
- privacy。

[A Multifaceted Benchmarking of Synthetic EHR Generation Models（Nature Communications 2022）](https://www.nature.com/articles/s41467-022-35295-1)给出了比较通用的结构化 EHR benchmark：

- categorical/binary：average absolute prevalence difference（APD）；
- continuous：average Wasserstein distance（AWD）；
- dependency：真实与 synthetic correlation matrices 的 cell-wise absolute difference；
- joint distribution：latent cluster deviation；
- record-level：clinical knowledge violation 和 medical concept abundance；
- utility：TSTR、TRTS 和重要特征 overlap；
- privacy：attribute inference、membership inference 等。

### 2.3 TriCompose 推荐的 EHR panel

主论文最小集合：

1. `schema_valid_rate`：字段、类型、code vocabulary、时间顺序和单位是否合法。
2. `clinical_rule_violation_rate`：例如 SBP < DBP、不可能的人口学组合、药物/诊断/实验室的硬冲突。
3. `marginal_fidelity`：categorical 用 APD 或 total variation；continuous 用 Wasserstein。
4. `dependency_fidelity`：diagnosis–medication、diagnosis–lab、lab–vital correlation/co-occurrence 差异。
5. `temporal_fidelity`：visit 数、事件数、inter-event interval、transition/sequential-bigram correlation。
6. `TSTR utility`：至少一个 MIMIC-IV-ED 可复现任务，和 TRTR 上限比较。
7. 如果准备发布 synthetic cohort，再加入 MIA、attribute inference 和 nearest-neighbour duplication。

注意：EHR fidelity 大多是 **cohort-level**，不能给单个 candidate 一个可靠的“0.83 EHR quality”。单病例阶段只能用 schema、rule 和 provenance completeness 作为 gate。

## 3. Synthetic CXR 自身怎么评

### 3.1 当前最值得跟随的 benchmark：CheXGenBench

[CheXGenBench（TMLR 2026）](https://arxiv.org/abs/2505.10496)是目前最接近本项目的 synthetic CXR 统一 benchmark。它将 CXR 评价分为 fidelity、privacy 和 utility，并明确指出普通 ImageNet Inception-FID 或旧 DenseNet feature FID 对医学影像并不理想。

它推荐：

- **RadDINO-FID 与 RadDINO-KID：**真实与 synthetic CXR 的 distributional fidelity。
- **RadDINO-PRDC：**
  - Precision：生成样本局部真实性；
  - Recall：真实模式被覆盖多少；
  - Density：生成样本在真实 feature manifold 中的密度；
  - Coverage：长尾模式覆盖。
- **BioViL-T image–text alignment：**图像与 conditioning text 的 global cosine；这是条件遵循指标，不等于 CXR 自身真实性。
- **Pathology-conditional evaluation：**每种疾病单独计算 FID/KID/PRDC/alignment，防止 “No Finding” 大类掩盖 rare-disease failure。
- **Privacy：**最近训练图的 pixel distance、RadDINO latent distance，以及 patient re-identification Siamese score。
- **Utility：**在真实 test set 上评估 classification、segmentation 和 report generation；同时比较 real-only 与 real+synthetic，附录还做 synthetic-only utility。

### 3.2 EHRXDiff 能借鉴什么，不能借鉴什么

[EHRXDiff（CHIL 2025）](https://arxiv.org/abs/2409.07012)评价三类内容：

- clinical information preservation：Chest ImaGenome 与 CheXpert disease classifiers 的 macro AUROC；
- temporal preservation：把病例分成 `Same` 和 `Diff`，分别评价不变和发生变化的疾病；
- demographic preservation：age correlation、sex/race AUROC；
- visual realism：基于 XRV DenseNet feature 的 FID。

它的输入是 previous CXR + interval EHR events，目标是未来 CXR，因此 `Same/Diff` 指标只适用于 longitudinal route。TriCompose 的 cold-start `synthetic EHR -> text -> CXR` 不能直接声称采用了 EHRXDiff 的 temporal evaluation。

而且 CheXGenBench 2026 对 image feature 做了更新：最终论文应优先用 RadDINO-FID/KID/PRDC，而不是只报告 XRV-DenseNet FID。

### 3.3 TriCompose 推荐的 CXR panel

主论文：

1. RadDINO-FID、KID。
2. RadDINO-Precision、Recall、Density、Coverage。
3. 所有指标按 finding/pathology 分层，特别报告 abnormal 和 rare cases。
4. artifact/non-degeneracy rate：裁剪、倒置、非胸片、严重曝光异常、文字/身体部位伪影。
5. blinded radiologist study：realism、diagnostic usability、是否存在明显解剖错误；建议至少抽样 50–100 张。
6. downstream CXR utility：classifier 或 report generator 在 real test 上的表现。
7. 若有数据发布主张，再做 nearest-training-image 和 re-identification privacy。

不要把生成 CXR 与唯一真实 CXR 的 pixel MSE、PSNR 或 SSIM 当主要指标。相同 EHR 可以对应多张临床上合理但像素完全不同的胸片；逐例 paired similarity 只能作为 real-anchor 的辅助结果。

## 4. Synthetic Report 自身怎么评

### 4.1 Real-anchor：有真实 report 时

推荐主指标：

1. **GREEN**：给出 clinical error categories、error counts 和可解释说明；论文在 6 名专家 error count 和 2 名专家 preference 上验证。来源：[GREEN，Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.21/)。
2. **RadGraph-F1**：比较 clinical entities 和 relations；来源：[RadGraph，NeurIPS Datasets & Benchmarks 2021](https://physionet.org/content/radgraph/1.0.0/)，更大规模标注可参考 [RadGraph-XL，Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.765/)。
3. **CheXbert macro/micro F1-14**：finding-level label overlap；来源：[CheXbert，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.117/)。

强辅助指标：

- **RadFact logical precision/recall**：逐句双向 entailment，precision 对应 hallucination，recall 对应 omission；但它明确以 ground-truth report 为 premise/reference。来源：[MAIRA-2 / RadFact](https://arxiv.org/abs/2406.04449)。
- **VLScore**：输入 image、candidate report、reference report，做 image-aware report comparison；它在 NeurIPS 2024 提出，并在 radiologist error dataset 上验证。来源：[VLScore，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0fbbc5129cafcee8530223b8565561ac-Abstract-Conference.html)。
- **RadCliQ**：由多个指标组合、与 radiologist error score 对齐；来源：[Evaluating Progress in Automatic CXR Report Generation，Patterns 2023](https://doi.org/10.1016/j.patter.2023.100802)。
- **SPEC-CXR**：entity presence、location、severity、comparison 等细粒度错误；来源：[MICCAI 2025](https://papers.miccai.org/miccai-2025/0856-Paper3344.html)。
- **ReFINE**：AAAI 2026 的可解释细粒度 evaluator，可作为最新 metric ablation，但要先确认 checkpoint/code 能稳定复现。来源：[AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37680)。

BLEU、ROUGE-L、METEOR、BERTScore 只作可比性辅助，不能作为临床 factuality 主结论。[CXRMate-ED（ACL 2025）](https://aclanthology.org/2025.acl-long.9/)本身也是同时报告 RadGraph-F1、CheXbert-F1、CXR-BERT、GREEN、BERTScore、ROUGE-L、BLEU-4 和 repetition metric，而不是依赖单个 lexical score。

### 4.2 Fully synthetic：没有真实 report 时

“Report 自身”只能独立检查语言和结构质量：

- non-empty、section validity；
- repetition / absence of repeated n-grams；
- 长度和句子数分布；
- 非法 measurement、view、prior/comparison 和 unsupported temporal language；
- PHI 或模板泄漏；
- report diversity 和近重复率。

报告是否“临床正确”不能在完全不看 EHR/CXR 的情况下定义。clinical factuality 必须归到 CXR–Report 或 EHR–Report 跨模态边，或由 radiologist 直接看图审阅。

## 5. 三条跨模态边：建议的主指标

### 5.1 统一 evidence contract

首先定义共同 finding ontology。第一版可以采用 CheXpert-style findings，再单独加入 devices：

```json
{
  "finding": "pleural_effusion",
  "state": "positive | negative | uncertain | unknown",
  "evidence_strength": "direct | compatible | risk_only",
  "source": "diagnosis | medication | lab | vital | image_classifier | report_text",
  "provenance": ["original event/code/text span"],
  "time": "current | prior | unspecified",
  "laterality": "left | right | bilateral | unknown",
  "severity": "mild | moderate | severe | unknown"
}
```

核心语义：

- `unknown` 绝不等于 negative，也不能计为 agreement。
- EHR 中的 CHF、furosemide、低氧等多数只是 `compatible` 或 `risk_only`，并不逻辑蕴含 edema、cardiomegaly 或 effusion。
- 主 contradiction score 只使用 `direct`、明确 positive/negative 的事实。
- laterality、severity、location 只有在来源明确时才评分；缺失不能自动补全。

### 5.2 EHR–CXR：Masked Known-Fact Consistency

文献没有一个通用、原生支持 structured EHR–CXR cold-start 的现成 metric。最合理的主分数应由 EHR 中可影像化且明确的 facts，与冻结 CXR classifier 输出比较。

设 EHR finding state 为 `e_k`，CXR classifier probability 为 `p_k`，`m_k=1` 表示 EHR 对该 finding 有 direct、明确证据。定义：

```text
support(e_k, p_k) = p_k       if e_k = positive
                    1 - p_k   if e_k = negative

S_EC = sum_k m_k w_k support(e_k,p_k) / sum_k m_k w_k
```

用真实 validation set 为每个 finding 学到两个固定阈值 `t_neg,k` 和 `t_pos,k`，再定义强矛盾：

```text
positive EHR fact AND p_k < t_neg,k
negative EHR fact AND p_k > t_pos,k
```

报告：

- `EHR-CXR soft support`
- `EHR-CXR weighted contradiction rate`
- `known-fact coverage`
- dataset-level per-finding AUROC/AUPRC、macro-AUPRC
- matched-vs-hard-negative AUROC/AUPRC 与 pairwise ranking accuracy

辅助：

- BioViL-T(`EHR-derived radiology prompt`, CXR) cosine，并用 matched pairs 与 hard negatives 校准。
- 对 longitudinal EHRXDiff route，额外报告 Same/Diff temporal-change accuracy。
- 若加入 location/temporal scene graph，可采用 [Chest ImaGenome（NeurIPS Datasets & Benchmarks 2021）](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/17e62166fc8586dfa4d1bc0e1742c08b-Abstract-round2.html) ontology。

必须在论文中诚实表述：BioViL-T 接收的是 EHR-derived text，不是 structured EHR。因此它衡量 prompt–image alignment，而不是原生 EHR–CXR understanding。

### 5.3 CXR–Report：Finding Agreement + Image–Text Alignment

主指标：

- CXR：冻结 image classifier，输出每个 finding 的 calibrated probability。
- Report：CheXbert、RadGraph 或 validated named-finding extractor，输出 positive/negative/uncertain/unknown。
- 只在 report 明确陈述、且 image classifier 置信度足够时计算 agreement/contradiction。

建议报告：

- per-finding precision、recall、F1；
- macro-F1、macro-AUPRC；
- clinically weighted contradiction rate；
- positive-finding recall 与 negative-finding contradiction 分开；
- support-device accuracy 单独报告；
- report/cxr `unknown` 和 abstention coverage。

辅助：

- [BioViL-T（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.html) global image/report cosine，按 matched、random-negative、same-disease hard-negative 分布校准。
- RadGraph entity + Chest ImaGenome/BioViL phrase grounding，用于 laterality、location、device。
- VLM judge 只能作为第三路证据；必须用事实扰动和 radiologist subset 验证，不能直接把 0–1 scalar 当临床概率。

VLScore、GREEN、RadFact 和 RadGraph-F1 的标准用法是 candidate 与真实 report 比较。它们非常适合 real-anchor end-to-end report fidelity，但不是 direct reference-free CXR–Report edge 的替代品。

### 5.4 EHR–Report：Provenance-Aware Entailment and Contradiction

同样没有现成公认的 structured EHR–radiology report 单分数。推荐：

1. 从 EHR 生成带 provenance 的 controlled facts，例如：`Known current right pneumothorax is present.`
2. 从 report 用 CheXbert + RadGraph/RadGraph-XL + named finding extraction 得到事实。
3. 比较 direct facts 的 support、contradiction 和 coverage。
4. 可再使用 radiology NLI 模型判断 controlled EHR statements 与 report sentences 的 entailment/contradiction。

可参考 [RadNLI](https://physionet.org/content/radnli-report-inference/1.0.0/)，但 RadNLI 主要是 radiology sentence–sentence inference，不是 structured EHR–report。若使用，必须在手工标注的 EHR-fact/report pairs 上验证迁移效果。

建议报告：

- `EHR-known-fact support recall`
- `explicit contradiction rate`
- `severity-weighted contradiction rate`
- `temporal contradiction rate`
- `device contradiction rate`
- `unverifiable report fact rate`
- `known-fact coverage`

最后一项只能称为 `unverifiable`，不能直接称为 hallucination：EHR 没写某个 finding，并不代表图像或报告不能发现它。

## 6. Cross-path 和 triple-level 评价

每张 CXR 有 CXR-only report 和 EHR+CXR report 时，可额外计算：

- report–report CheXbert/RadGraph fact agreement；
- cross-path strong contradiction rate；
- 两条 report path 对同一 CXR finding 的 disagreement entropy；
- 哪条路径更符合 EHR 与 CXR 的 joint evidence。

它是 error localization 的重要证据，但不是三条 pairwise edge 的替代。

Triple-level 不建议直接把所有 raw scores 相加。推荐流程：

```text
1. Hard validity gates
2. Reject any high-severity explicit contradiction
3. 在剩余候选中最大化三条 edge 的 geometric mean
4. 用 minimum edge score 处理短板
5. 在质量接近时才用 cost 决胜
```

最终表格同时展示三条 edge，不只展示 aggregate。

## 7. 如何证明这些 scorer 真的可信

这是论文能否成立的关键，比再加一个生成模型更重要。

### 7.1 Calibration set

从 patient-level held-out real matched cohort 建立 calibration / test split。每个真实 triple 构造：

- true matched pair/triple；
- random patient swap（easy negative）；
- same-disease patient swap（hard negative）；
- 单事实 perturbation：presence/absence、negation、laterality、severity、device、temporal change；
- CXR replacement；
- report replacement。

### 7.2 Scorer meta-evaluation

每条边至少报告：

- matched-vs-negative AUROC、AUPRC；
- true pair 优于 corrupted pair 的 pairwise ranking accuracy；
- Spearman/Kendall correlation with expert score；
- 每类错误的 detection sensitivity；
- coverage/abstention；
- calibration error 或 reliability plot。

[ReEvalMed（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.598/)特别强调 metric 的 discrimination、robustness 和 monotonicity；TriCompose 的 perturbation benchmark 可以直接沿用这套 meta-evaluation 思路。

阈值只能在 validation/calibration set 选择，test set 固定。所有结果使用 patient-level bootstrap 95% CI，并报告 abnormal/rare subgroup。14-label accuracy 容易被大量阴性 finding 支配，因此主表优先 macro-AUPRC、macro-F1 和 contradiction rate。

### 7.3 防止 evaluator circularity

如果 XRV、BioViL-T 或 Qwen 被用于 candidate selection，最终论文不能只用同一个 scorer 宣布 selection 更好，否则存在 reward hacking / Goodhart 问题。

建议两套 evaluator：

- **Selection stack：**便宜、reference-free，例如 XRV + CheXbert/named facts + BioViL-T + rules。
- **Held-out evaluation stack：**GREEN/RadFact/VLScore、不同 image classifier/VLM、人工 radiologist review、hard-negative benchmark。

## 8. 对当前 V1 scoring 的直接判断

当前 V1 的 XRV、BioViL-T、Qwen scalar 和 lexical EHR–Report agreement 适合证明 pipeline 跑通，但还不能支持论文中的 clinical consistency claim：

1. **XRV EHR support：**如果把 CHF、用药或 risk signal 当成确定的影像 finding，会高估 EHR–CXR consistency。
2. **BioViL-T prompt/CXR：**只证明生成图与桥接 prompt 语义接近，不等于 structured EHR 与 CXR 一致；而且 raw cosine 未校准。
3. **Qwen scalar：**没有具名 fact、provenance、coverage 和 error type；无法可靠定位是哪个 finding 冲突。
4. **lexical EHR–Report agreement = 1.0：**词面重叠不能证明临床一致，也不能区分 unknown、compatible 和 entailed。
5. **单病例高分：**FID/KID/PRDC、EHR distributional fidelity 等本来就是 cohort metric，不能被单个 candidate score 代替。

因此下一步优先级应从“继续调 scalar 权重”改为：

1. named finding contract + provenance；
2. real matched/hard-negative calibration bank；
3. 三条 edge 各自的 support/contradiction/coverage；
4. independent held-out evaluator；
5. 最后再学习或手工设 composition policy。

## 9. 实现可行性：哪些现在就能接

下面的“成本”只表示相对推理与工程负担，不是精确显存估计。

| Evaluator | 输入 | 公开实现 / checkpoint | 相对成本 | TriCompose 中的建议角色 |
|---|---|---|---|---|
| XRV classifier | CXR | 当前 V1 已运行 | 低 | selection；必须先校准概率和阈值 |
| BioViL-T | CXR + text | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/html/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.html) | 低–中 | CXR–text 辅助分数，不单独作临床一致性结论 |
| CheXbert | report | [official code/checkpoint](https://github.com/stanfordmlgroup/CheXbert) | 低 | **现在就接**；report named findings |
| RadGraph / RadGraph-XL | report | [official package](https://github.com/Stanford-AIMI/radgraph) | 中 | **现在就接**；entity/relation 与 RadGraph-F1 |
| RadDINO + CheXGenBench | CXR cohort | [RadDINO checkpoint](https://huggingface.co/microsoft/rad-dino)、[benchmark code](https://github.com/Raman1121/CheXGenBench) | 单图低、全 cohort 中–高 | **现在就接离线评价**；FID/KID/PRDC，不用于单图选择 |
| GREEN | candidate + reference report | [public 7B model](https://huggingface.co/StanfordAIMI/GREEN-RadLlama2-7b) | 中–高 | real-anchor held-out evaluator；fully synthetic 无 reference 时不能直接用 |
| RadCliQ | candidate + reference report | [official code](https://github.com/rajpurkarlab/CXR-Report-Metric) | 中 | real-anchor composite evaluator |
| RadFact | candidate + reference report | [official code](https://github.com/microsoft/radfact/) | 高 | 预算允许时做 held-out factuality；不作为第一版必需项 |
| SPEC-CXR | candidate + reference report | [official code](https://github.com/lunit-io/spec-cxr) | 中–高 | location/severity/comparison 的细粒度补充 |
| VLScore | image + candidate + reference | [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0fbbc5129cafcee8530223b8565561ac-Abstract-Conference.html) | 中–高 / 有复现工作 | real-anchor image-aware 补充；先确认公开实现再排期 |
| TriCompose named-fact verifier | EHR facts + CXR labels + report facts | 本项目实现 | 低 | **核心 selection 与 error-localization contract** |

推荐服务器上的接入顺序：

1. 先统一 `positive / negative / uncertain / unknown` named-fact JSON；
2. 接 CheXbert 与 RadGraph-XL；
3. 用当前 XRV + CheXbert 完成三边 contradiction/support/coverage；
4. 离线接 RadDINO/CheXGenBench 做 CXR cohort 评价；
5. real-anchor 才运行 GREEN、RadCliQ，预算允许再加 RadFact 或 SPEC-CXR；
6. 最后建立人工审阅集和 hard-negative meta-evaluation。

这里不需要训练新的生成模型。TSTR/downstream utility 会训练一个独立预测器；如果“任何模型都不训练”是硬约束，可将它列为可选评价而不是 V1 必选项。

## 10. 明天可以选择的三个版本

### A. 最小可运行版（推荐先做）

- EHR：schema validity + rule violations + cohort marginals。
- CXR：artifact gate；XRV probabilities；小规模 RadDINO-FID/KID。
- Report：CheXbert named labels + repetition/temporal hallucination rules。
- EHR–CXR：direct known-fact support/contradiction/coverage。
- CXR–Report：XRV–CheXbert agreement + BioViL-T。
- EHR–Report：direct fact agreement + contradiction/coverage。
- 50 个 real matched cases：random swap、same-disease swap、单事实 flip。

这个版本不训练任何生成模型；只部署冻结 evaluator 和规则。

### B. WACV 论文完整版

在 A 上增加：

- EHR：dependency + temporal fidelity + TSTR + privacy。
- CXR：完整 RadDINO-FID/KID/PRDC、pathology-stratified、privacy、downstream utility。
- Report real-anchor：GREEN + RadGraph-F1 + CheXbert-F1 + VLScore 或 RadFact。
- location/device/temporal attribute evaluation。
- 50–100 样本 blinded radiologist review。
- selection scorer 与 held-out evaluation scorer 分离。
- patient-level bootstrap CI 和 rare/abnormal subgroup。

### C. 更强的方法论文版

在 B 上增加：

- 一个公开的 TriModal-Consistency perturbation benchmark；
- 三条 edge scorer 的 systematic meta-evaluation；
- budgeted selection 的 quality–compute Pareto curve；
- error localization accuracy 和 targeted regeneration success rate；
- scorer ensemble uncertainty 与 abstention calibration。

## 11. 最终推荐

如果明天只选一套，我建议：

> **单模态：SynEHRgy/Nature EHR benchmark + CheXGenBench + GREEN/RadGraph/CheXbert；跨模态：统一 named-fact contract，分别计算 masked support、explicit contradiction 和 coverage；BioViL-T 只做 CXR–text 辅助；用 real matched + same-disease hard negatives + fact perturbations 校准；最终评价与 selection scorer 分离。**

这套设计既能支持 V1 static reranking，也能自然支持 V1.1 error localization：agreement pattern 不再是四个无法解释的 scalar，而是能指出具体哪个 finding、哪一条边、哪一种错误。

## 12. 重点参考文献

### EHR generation/evaluation

- [SynEHRgy: Synthesizing Mixed-Type Structured EHRs using Decoder-Only Transformers, NeurIPS 2024 GenAI for Health workshop](https://arxiv.org/abs/2411.13428)
- [PromptEHR: Conditional EHR Generation with Prompt Learning, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.185/)
- [HALO: Synthesize High-dimensional Longitudinal EHRs, Nature Communications 2023](https://www.nature.com/articles/s41467-023-41093-0)
- [A Multifaceted Benchmarking of Synthetic EHR Generation Models, Nature Communications 2022](https://www.nature.com/articles/s41467-022-35295-1)

### CXR generation/evaluation

- [CheXGenBench, TMLR 2026](https://arxiv.org/abs/2505.10496)
- [RoentGen, Nature Biomedical Engineering](https://www.nature.com/articles/s41551-024-01246-y)
- [EHRXDiff, CHIL 2025](https://arxiv.org/abs/2409.07012)
- [RadDINO, Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00965-w)
- [BioViL-T, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.html)
- [Chest ImaGenome, NeurIPS Datasets & Benchmarks 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/17e62166fc8586dfa4d1bc0e1742c08b-Abstract-round2.html)

### Report generation/evaluation

- [CXRMate-ED / auxiliary patient data for report generation, ACL 2025](https://aclanthology.org/2025.acl-long.9/)
- [CheXbert, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.117/)
- [RadGraph, NeurIPS Datasets & Benchmarks 2021](https://physionet.org/content/radgraph/1.0.0/)
- [RadGraph-XL, Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.765/)
- [RadCliQ / Evaluating Progress in Automatic CXR Report Generation, Patterns 2023](https://doi.org/10.1016/j.patter.2023.100802)
- [GREEN, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.21/)
- [MAIRA-2 and RadFact](https://arxiv.org/abs/2406.04449)
- [VLScore: Image-aware Evaluation of Generated Medical Reports, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0fbbc5129cafcee8530223b8565561ac-Abstract-Conference.html)
- [SPEC-CXR, MICCAI 2025](https://papers.miccai.org/miccai-2025/0856-Paper3344.html)
- [ReEvalMed, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.598/)
- [ReFINE, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37680)
- [RadNLI, PhysioNet 2021](https://physionet.org/content/radnli-report-inference/1.0.0/)

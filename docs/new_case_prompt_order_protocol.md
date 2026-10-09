# 新 synthetic 病例的 prompt 顺序扩展对照

这项前瞻性开发诊断固定另一个已生成 synthetic pool 中剩余的三个 eligible EHR，
检验 finding block 前置的响应是否仍具有模型差异。它不调评分器，不训练，不调用
LLM，不替换病例或旧 winner，也不把分类器变化视为临床修复。

## 病例选择与分母

源是冻结 SynEHRgy Qwen2-40bins 的既有 unconditional 100-case pool。
原筛查保留完整 100 例分母：74 structurally valid，5 有明确 eligible diagnosis，
其中前两例早已用于 October smoke。此次一次性固定全部剩余三例，不根据新图
或分类器得分筛选、替换或追加病例。opaque IDs 为 case_079、case_080、case_082，
条件分别是 pleural effusion、pneumonia、pneumonia。原始 source JSON 不修改。

CPU preparation 在既有真实 CPU Slurm cgroup 中核对 source manifest、历史 screen、
100-row metadata linkage、三个源 case hashes 与 structural validity，再以已有
canonicalizer、fact extractor 和 renderer 私下转换。只允许已确认 fully synthetic
的 EHR；不读取真实患者 EHR、源/目标报告或图像。新 canonical EHR、facts、provenance、
最终 prompts 和 plan 全部原子落盘在新的 protected run，拒绝覆盖。

这是新病例的开发扩展，不是独立临床 gold test。它们以前经过 eligibility screening，
同样来自冻结 EHR generator；当前检查未发现它们属于该 100-case pool 的已准备
CXR 请求。canonical hashes 须与上一轮 order diagnostic 的两个 anchor 不同。
新病例 ID 与其他历史 pool 可能重名，不能仅凭 ID 推断复用或独立性。

## 对照与 renderer 版本

每例 RoentGen-v2 与 Sana，各 seeds 0、1，每个 seed 重新生成 original 和
findings_first。共 24 个生成位置、12 组配对、最多 24 次 XRV 评分。
所有病例与 seeds 保留，包括失败、缺失或下降结果；不挑最佳 seed。

两条分支都用当前 V1.1.4 renderer，只移动完整末尾 finding block。所有词、
标点、EHR、fact evidence、clinical intent、context surface budget 与 fixed PA
generation protocol 保持一致；PA 是固定生成协议，不是 EHR 中观测到的事实。
CHF 只作为 clinical context，不能产生 cardiomegaly/edema 断言。unknown、negative
或 uncertain finding 不被添加为阳性，不加入无证据的设备、左右侧或严重度。
pneumonia-to-opacity 是原 renderer 的弱影像 prior，不是独立图像真值。

上一轮使用了 legacy renderer 和未验证 CHF image prior，而这一轮用当前 renderer。
因此两轮间同时有病例来源和 renderer 版本差异，不能把跨轮差异全部归于病例；
只有本轮每组内部是纯顺序对照。任何 current rendering、单一目标 finding、source
或 hash 检查不通过就停止，不改 EHR 或换病例来继续。

RoentGen 保持 75 steps、CFG 3、512×512、float16；Sana 保持 20 steps、CFG 4.5、
1024×1024、float16。保留 Sana 官方 prefix/lowercase。观察真实 tokenizer input
并绑定 text/token-ID hashes；outer adapter 不另加 prefix。不换 checkpoint 或原生
推理参数。新图不能用旧图代替，实际生成 text 文件须是 sealed final model input。

## 评分与解释

XRV checkpoint、grayscale/crop/224px preprocessing 与 operating-point-normalized
score space 不变。使用现成 `pneumonia` 和 `effusion` heads，后者映射到已存在的
pleural_effusion 名称。每个病例只比较自己预声明 condition 的原始分数：不拿肺炎
分数评价积液，不混合两个 finding 的绝对尺度，不把 unknown 当 negative 或零。

保存全部 24 位置、12 配对行、原顺序分数、前置分数、差值、token/image hash 变化、
缺失状态、逐模型调用和耗时。按 model 和 finding 分开描述，pneumonia 每模型
4 组、effusion 每模型 2 组；它们分别只来自两个和一个 EHR，不作患者级显著性
或临床模型排名。没有新 score 权重、阈值、head fitting 或 acceptance gate。

诊断记录是生成约束，不是某张图的 ground truth。raw XRV 分数不是校准疾病概率；
分数上升、文本哈希变化或图像字节变化均不证明临床改善。现有其他自动观察器
尚未建立足够可靠的独立 clinical gold，本轮不会把它们的 agreement 冒充人工标签。
没有 report 边、fault localization、quality/diversity 的独立评价、repair acceptance
或 Agent/cost-saving 结论。下一轮三模态验证仍需单独预声明和批准。

## 执行与隐私

三个串行 worker 为 RoentGen、Sana、XRV，最多 3 次 model loads 与 48 次 charged
inference attempts；load/forward 前 fsync reservation。失败不退款、不重试，余下
位置保留 unavailable；缺失配对差值为 NA。run 不覆盖、不自动 resume/requeue。
源模型、环境、checkpoints 和旧代码只读。新 cache/tmp/logs 在 protected workspace。

只允许 GPU Slurm 加载模型或生成/评分。申请 1 V100 或 A40、2 CPU、32G RAM、
15 分钟上限，parent timeout 870s，workers 300/300/120s。资源空闲不保证排队或
耗时。完整新脚本和资源请求展示后，必须再次获明确批准；继续准备不是 sbatch
授权。public logs 只有 sanitized status/hash，病例文本与像素仅留在 protected。
所有 protected directories 2770、files 0660，project GID96293 或 NFS GID65534。

Entry：`TriCompose-v1.2/agent/run_new_case_prompt_order.py`。
新 plan/run parents：`artifacts/protected/tricompose_v1_2/new_case_prompt_order_plans/`
和 `new_case_prompt_order_runs/`。协议、测试和代码 staging 后保持封存不变。

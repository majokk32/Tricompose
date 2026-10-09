# 固定 synthetic EHR 的 prompt 语句顺序对照

该前瞻性开发诊断只检验原 prompt 的 finding 语句顺序是否影响冻结生成模型的
条件响应。它沿用同一批已使用的两个 synthetic EHR，不是新的测试集或完整
自动修复实验。之前手写正负文本的结果不能证明这两个 EHR 的原 prompt 错了。

## 唯一改动与固定输入

固定原有 case_009、case_018，保留 canonical EHR、facts、临床 intent、
封存的旧版 renderer 文本、included 与 omitted context、原生模型、参数和 seed。
两个模型为 RoentGen-v2 与 Sana；seeds 为 0、1。每个病例、模型、seed
重新生成 original 与 findings_first 两个分支，共 16 个生成位置、8 组配对。
不根据上次分数选 seed、病例或胜出模型，不把历史图像代替本次 original 分支。

findings_first 只能把原 prompt 最后的 EXACT finding block 移到最前面；
不改任何词、标点或临床语句，不删除 context、demographics 或原固定 PA
generation protocol。验证词的 multiset、原最后 finding block、原 renderer
metadata、源文本 SHA256 和 evidence references；任何检查失败就停止，不另挑病例。

准备阶段发现封存文本并非当前 renderer 的输出：RoentGen 使用 v1_1_3，Sana
使用 v1_1，而当前二者都是 v1_1_4。两者旧 metadata 都声明 pneumonia 与
congestive_heart_failure，包含旧 CHF-to-CXR prior。当前 renderer 不再把 CHF
当直接 image finding。为了只测试顺序，不用当前 renderer 重建或替换原文本；
逐项验证 sealed request 与源 prompt metadata，直接复制已 SHA256 绑定的
原文字句。新版 renderer 仅用于记录历史差异，绝不用于本次生成。

原始 facts 中的未知、阴性、uncertain finding 不因本次改写进入 prompt。
不复制上次手写的 focal consolidation，不加入设备、部位、严重度、时间变化或
新疾病。原始 pneumonia-to-opacity derived rule 只按旧 renderer 保留，不新增
推导规则，也不是独立的图像真值。旧 CHF-to-CXR prior 同样在两条分支保留为
未经验证的 legacy assumption，不能宣称所有原图像断言均由 EHR 直接支持。
本次不会把新的 clinical context 变成 radiographic assertion，也不偷偷删除
旧断言从而伪装纯顺序对照。两个模型原有 context budget 不相同，因此模型间差异
不是纯 architecture ablation；每组内部只变化原文字句的顺序。

## 数据与执行边界

CPU prepare 在已有实际 CPU Slurm cgroup 中读取已确认 fully synthetic 的 facts、
原 prompt 与其 legacy metadata，仅输出状态、次数和 hash。不解析 canonical EHR body，
不打开任何真实患者 row、源报告、真实图像、target artifact、生成 report 或图像
像素。只复用已 sealed 的 checkpoint receipts 与 asset size/mtime，不加载模型
或读取大型权重 bytes。旧病例、facts、prompt 文件与 winners 不写入或覆盖。

新 prompt、其 evidence proof 与 final-input hashes 原子写入新的 protected plan。
模型必须实际读取这些 final text 文件，adapter 不另加 prefix；Sana 官方原生
prefix 与 lowercase 处理保留，观察真实 tokenizer 边界。文本与 trace 只留在
protected，不能进入 Git、聊天或公开日志。

GPU 使用三个串行原生 worker：RoentGen、Sana、XRV，加载后显式检查 frozen
eval/gradient flags。RoentGen 保持 75 steps、CFG 3、512px；Sana 保持 20 steps、
CFG 4.5、1024px，float16。各自模型 settings 与 precision 不因本次对照改变。
XRV 使用旧 checkpoint、grayscale/crop/224px preprocessing 与原始 op_norm
分数。所有 load/forward 在调用前 fsync reservation；最多 3 个模型 load、
16 个 image forwards 与 16 个 XRV forwards，32 次计费 inference attempts。
失败不退款、不重试；该 worker 剩余位置保留 blocked/missing。没有 report、
CheXbert、LLM、API、下载、训练、阈值选择或 winner 替换。

新目录拒绝覆盖和自动 resume。超时或失败保留 prefix 与计费 journals，不能把
未完成位置记为阴性、零差值或成功。文件 0660、目录 2770，project group
ruishanl_1185；CARC NFS 可显示 GID65534。公开输出只有 sanitized status/hash。

## 读数与解释

保留全部 16 个生成位置和 8 组配对。每组输出两种顺序的 XRV 原始 pneumonia
与 consolidation 分数、findings_first 减 original 的差值、token-ID hash 与
image-byte hash 是否变化、native token counts、失败与缺失状态、实际调用成本。
consolidation 仅是保持与前一手写诊断一致的次要 XRV 读数，不冒充本 EHR 的
explicit fact。op_norm 是运行点归一化分数，不是校准概率或临床准确率。

token-ID 改变不证明模型使用了该 finding；token count 或截断参数不证明某个
finding 被保留或注意力正确。图像 byte 差异不证明质量改善。XRV 差值可能来自
分类器偏差。仅两个已使用 EHR、两个 seed，不作显著性检验、最佳 seed 选择、
模型总体排名、临床修复或 Agent compute savings 宣称。不根据本次结果调整
scorer、threshold、head mapping 或旧接受 gate。

## 资源与批准

资源查询当前 V100 无空闲、debug A40 有空闲，因此拟申请一个 debug A40、
2 CPU、32G 主机内存、15 分钟上限，parent timeout870s，worker timeout
300/300/120s。资源空闲不保证立即启动或实际耗时；不因 inference 未训练就
假设低于模型实际显存需求。完整新 batch 脚本和资源请求展示后，必须取得
新的明确批准；本轮继续工作的消息不授权提前 sbatch。

Entry：`TriCompose-v1.2/agent/run_fixed_ehr_prompt_order.py`。
Protected plan/run parents：`tricompose_v1_2/fixed_ehr_prompt_order_plans/` 与
`tricompose_v1_2/fixed_ehr_prompt_order_runs/`。准备成功后仍未运行 GPU。

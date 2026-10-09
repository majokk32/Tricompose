# CheXpert/NegBio：隔离环境部署与离线初始化

## 结论 / Outcome

官方 legacy 依赖环境和冻结解析资源已安装，官方 BLLIP parser、Java
dependency backend、NegBio 否定/不确定性规则均能完成离线初始化。
初始化耗时 **5.440898 秒**；`pip check` 通过。

这只是 **initialization readiness**，不是端到端报告解析成功，更不是
临床准确性或稳定 inference 的证明。本轮报告输入和 report-parser 调用均为
**0**；未读取患者记录、真实报告/图像、候选报告正文或临床目标。
旧标签、评分、择优结果和失败的资格检查均保持不变。

## 执行范围 / Execution boundary

使用已经存在的 CPU Slurm allocation **12682821**：4 CPU、32 GB RAM。
没有新增 `sbatch` 提交，没有 GPU inference、训练或外部临床数据 API 调用。
系统 Conda、Java 模块与其他既有环境只读；缓存、临时目录和新环境位于
本 workspace 内。没有授权 home 目录写入或凭证访问。

```text
runtime/venvs/chexpert-negbio-py36-12682821-v1/
runtime/eval_models/chexpert_negbio/frozen_assets_12682821_001/
artifacts/protected/tricompose_v1_2/chexpert_negbio_deployment_runs/initialization_12682821_002/
artifacts/protected/tricompose_v1_2/chexpert_negbio_deployments/deployment_12682821_001/
```

Protected 输出目录使用 2770，文件使用 0660，仅 project group 可访问；
NFS 可将项目组元数据呈现为 nobody/65534。新环境、资源和 protected 记录
均被 Git ignore；本文件只有部署元数据，不包含报告正文或患者信息。

## 固定版本 / Frozen provenance

| Component | Actual version / revision |
| --- | --- |
| CheXpert source | `44ddeb363149aa657296237f18b5472a73c1756f` |
| NegBio source | `073199e2792824740e89844a59c13d3d40ce4d23` |
| Python | 3.6.7 |
| NumPy / pandas | 1.15.4 / 0.23.4 |
| NetworkX / NLTK / PLY | 1.11 / 3.3 / 3.11 |
| BioC / BLLIP | 1.1.dev3 / 2016.9.11 |
| PyStanfordDependencies / JPype1 | 0.3.1 / 0.6.3 |
| lxml | 3.7.3 |
| CoreNLP JAR | 3.5.2 |
| Java module | `openjdk/11.0.20.1_1` |
| NLTK data revision | `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` |

官方 Conda 约束和八项 pip 依赖保留；新增且单独披露的依赖只有
`Cython==0.29.36`，用于满足 MKL-random 已声明的包依赖。
不是通过升级官方核心版本或修改模型源码解决兼容问题。

六项资源下载包括 GENIA+PubMed、CoreNLP JAR 及其公开 SHA1 文件、
NLTK punkt、universal_tagset、wordnet。资源包封存 **141 个文件**。
CoreNLP JAR 与发布方 SHA1 校验一致；其他 SHA256 是本地观察并冻结的
内容哈希，不冒称发布方提供的校验值。

两份官方 source-only sparse checkout 位于根目录 `chexpert-labeler/`
和 `NegBio/`，Git 状态保持干净；未检出临床样本、图片或上游示例数据。

## 兼容处理 / Declared runtime configuration

- BLLIP 使用同一个隔离环境中的官方 GCC 7.3.0；显式配置已有 compiler、
  linker 和 sysroot 搜索路径，不修改 parser 的 C++ 源码。
- 原始旧激活脚本不兼容 `set -u`，仅在激活阶段关闭 nounset。
- StanfordDependencies 显式使用 workspace 内已有 JAR，并关闭自动下载；
  固定 JPype、universal dependencies、`CCprocessed`，不静默切换 backend。
- 模型目录和 NLTK data path 显式指向冻结资源；PLY 禁止写入 grammar tables。
- initializer v1 的 wrapper 改变了 PLY caller introspection，导致 YaccError。
  v2 显式传入官方 `negbio.ngrex.parser` namespace；没有改动 grammar 或规则。
- Java 临时目录/preferences、包缓存和 Python 输出限定在 workspace，
  禁用 bytecode；native 初始化期间屏蔽可能包含正文的直接 stdout/stderr。

所有 11 个准备/安装/初始化尝试都记录在新的 `setup_attempts.json` 中。
失败目录和已消费的脚本均保留，没有覆盖。GCC 的错误搜索路径已观察到，
但其初始成因未证明，不将其归因于未经证实的继承环境变量。
现代 Conda 即便关闭注册，verify 阶段仍尝试触碰 home environment registry；
该尝试被 sandbox 拒绝，没有为此申请 home 写权限。

## 复核与封存 / Verification

- 17 项新测试覆盖官方依赖约束、固定资源 URL、下载/解包边界、禁止覆盖，
  以及 initializer 不接受报告输入。
- V1.2 全量回归 **2,138 项通过**，在既有 CPU 环境中运行。
- 独立 standard-library audit 复核 **2,107 个 source pins**、141 个资源文件、
  三个结果 artifact、初始化 receipt、protected 权限和原候选库哈希。
- 实际 `report_parser_calls=0`、`end_to_end_report_smoke_passed=false`、
  `clinical_qualified=false`、`selector_or_regeneration_enabled=false`。

封存 deployment manifest SHA256：

```text
f8c8d9a02455780de22e913ce7a1c056450c5e75eafffea6279d869583d9ef85
```

资源 manifest SHA256：

```text
c3fa079110adef7392e0363cbad745204b638f23cd391699834f047b2e1326b6
```

离线初始化 receipt SHA256：

```text
d3add7ef1a12a36c471af9efa64621a30d915f7e4db36c3c6d7a2e8220995cfa
```

旧 complete-bank endpoint manifest 仍为：

```text
ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c
```

## 下一步 / Next gate

先实现 failure-aware 端到端 worker，并固定一份前瞻性协议，在已有的
authored48 + authored64 文本上跑实际解析与重复运行。预测先封存，再加载
参考标签；这些是已知开发用的人工编写测试，不是临床 gold 或 held-out 证据。

需要分别记录原生 CheXpert labels、mention-conflict view 和完整阶段可用性。
必须测试官方 parser/detector 吞掉异常的情况；解析失败是 unavailable/null，
不能当作 unknown，更不能默认为 positive。未提及仍为 unknown，
No Finding 不扩展为十四项明确阴性。披露官方 Lung Opacity 词表与之前仅匹配
opacity/opacities 的词表差异，不能将不同 ontology 的结果直接视为同口径提升。

通过这一门槛后才讨论现有候选的 applicability；本次安装不自动授权重评
960 个报告槽位、不提升 selector 资格、不触发重新生成。
新 Slurm 提交仍需先展示完整脚本和资源请求并取得批准。

Official sources: [CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler),
[NegBio](https://github.com/ncbi-nlp/NegBio)。

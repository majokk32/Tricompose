# RICORD opacity + BioViL-T / 下一轮图文评分验证已准备

Prepared 2026-10-05 in existing CPU allocation **12666569**, not submitted.
No inference, pixel decode, generation, training, new download or external API.

## 为什么现在做

已冻结选优的 opacity 审计显示：同一组 static choices 的代理一致率从
max-head 的 100% 变为 exact-head 的 52.38%，coverage 不变。这是评分定义
敏感性，不是临床失败判决。下一步不修改 winner 或优化生成，而是验证另一个
已有评分来源能否分清同一 finding 的肯定与否定。

测试问题：在固定的 RICORD 图像级 opacity 标注上，BioViL-T 对
“有 lung opacity”和“没有 lung opacity”的分数是否有可靠方向？
即便 AUROC 高，如果负例也偏好肯定句，就不能据此判定报告的否定是否矛盾。

## 固定执行范围

- 复用已完成的 RICORD 50-case XRV run，50 个患者、25/25 derived strict-
  unanimity opacity classes；不是官方分歧裁决、肺炎标签或自然患病率样本。
- 直接使用其已有 50 张 hash-bound PNG，不重新读取 DICOM、不调整窗宽窗位、
  不另选图片。BioViL 官方内部 min/max remap 与 XRV 预处理不同，已披露。
- 冻结现有 BioViL-T 图像/文本编码器、checkpoint、vendor runtime；六条
  已有通用 opacity 正负文本，三个固定 family 全部报告，固定平均 margin。
- 最多 50 次主图像编码 + 原始前两例各一次 replay；一个 8-input 文本 batch
  （6 个 probes + 2 个 duplicate inputs）。没有 retry、训练、生成或新选优。
- 编码结束后才连接参考标签及原 XRV exact-head 0.5 结果。参考不进入模型。
- 报告 AUROC/AP、正负例方向 wins、ties、覆盖率/失败、全原始分母 recovery、
  与 XRV 的同图 disagreement；分数不是校准概率，不把共识当成临床真值。
- 这是已见参考结果上的 post-hoc DEVELOPMENT；不会解除临床定位/修复门槛。

Protocol: `docs/ricord_biovil50_protocol.md`.
Worker: `TriCompose-v1.2/real_validation/ricord_biovil50.py`.
Fixture tests: `TriCompose-v1.2/tests/test_ricord_biovil50.py`.

## 已完成检查

新增 **24 个 invented-fixture tests** 全通过。全量回归第一次命令漏了
SynEHRgy import 与已有 workspace JPEG 插件的 `PYTHONPATH`，出现 16 个
import/decoder errors；没有修改旧代码或放宽检查。使用完整已部署路径重跑，
**1,865 tests passed in 11.626 seconds**。

正确回归环境：

```bash
TMPDIR=/project2/ruishanl_1185/inference_3mod/.tmp \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.tmp/ricord_header_dependencies_12666569_001:.tmp/ricord_pixel_dependencies_12666569_001:src:TriCompose-v1.0/src:TriCompose-v1.1/src:experiments/synehrgy_v2/src:TriCompose-v1.2/real_validation:TriCompose-v1.2/src:TriCompose-v1.2/benchmarks:TriCompose-v1.0/eval/report_v1_1 \
/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python \
  -m unittest discover -s TriCompose-v1.2/tests -p 'test_*.py' -q
```

Metadata-only `prepare` completed with **0 model calls**, source/plan hashes,
all 50 opaque image paths, all three probe families, source/dependency bindings,
resource/script pins and protected modes independently checked. No pixel or
reference-row inspection in preparation. Original candidate selections,
opacity sidecar, historical outputs and teammates' files remain unchanged.
The new worker/tests/protocol are now consumed by the sealed plan; do not edit
them. Further change requires separate new sources/plan/run, not overwrite.

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_biovil_plans/opacity50_12666569_001/
  plan.json
  manifest.json
```

Plan manifest SHA256:
`91560c8407d638979d665e3fc4717dff80e68289e58d9036ce1258a8f96687e1`.
Worker SHA256:
`e1b86f80412a528681d1455241063d3fb37d3eff188b4c40fc482ac42edca808`.
Prepared batch script SHA256:
`2347a238242054b933b78d0e72231948f0754f7a374b8c7ac36254dc575c10cd`.

## 待批准的 GPU 请求

`TriCompose-v1.2/slurm/52_ricord_biovil50_v100.sbatch`:
account `ruishanl_1185`, partition `gpu`, **1 V100 / 2 CPUs / 8G host RAM /
5-minute upper limit**. `noderes -f -g` preparation snapshot showed multiple
idle/mixed V100s; do not pin a node or interpret that as a reservation or queue
guarantee. Prior RSUA BioViL job consumed about 15 seconds allocation time;
this different image size/input path is not guaranteed the same runtime.

All job-local logs/cache/temp remain protected, offline existing assets only.
Future result directory:
`artifacts/protected/tricompose_v1_2/real_validation/ricord_biovil_pilots/ricord_biovil50_<job_id>/`.
It will refuse existing directories and retain every failure/denominator.

**No `sbatch` has been submitted.** Show the entire prepared script and resource
request to the user and obtain explicit approval before executing:

```bash
sbatch TriCompose-v1.2/slurm/52_ricord_biovil50_v100.sbatch
```

Do not publish individual reference/scores or images, modify existing thresholds,
select a favorable template, refit a policy, or report clinical repair success
from this diagnostic. Independently audit completed score arithmetic/provenance
before deciding any subsequent experiment.

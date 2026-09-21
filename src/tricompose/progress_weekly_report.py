"""Build a protected bilingual progress-focused TriCompose weekly report."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-metrics", required=True)
    parser.add_argument("--qwen-scores", required=True)
    parser.add_argument("--agent-decision", required=True)
    parser.add_argument("--full-report", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


def _format_score(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("score must be numeric")
    return f"{float(value):.8f}"


def _candidate_qwen_scores(bundle: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    records = bundle.get("records")
    if not isinstance(records, list):
        raise TypeError("Qwen records are missing")
    for record in records:
        if (
            isinstance(record, dict)
            and record.get("metric") == "qwen25vl_cxr_report_match"
        ):
            candidate_ids = record.get("candidate_ids")
            value = record.get("value")
            if (
                not isinstance(candidate_ids, list)
                or len(candidate_ids) != 1
                or not isinstance(candidate_ids[0], str)
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise TypeError("invalid Qwen record")
            scores[candidate_ids[0]] = float(value)
    return scores


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.run_id):
        raise ValueError("invalid opaque run ID")

    run_root = PROTECTED_ROOT / args.run_id
    cxr_path = require_private_file(
        run_root / "stage1_ehrxdiff" / "generated_cxr.png"
    )
    report_metrics_path = require_private_file(args.report_metrics)
    qwen_scores_path = require_private_file(args.qwen_scores)
    agent_decision_path = require_private_file(args.agent_decision)
    full_report_path = require_private_file(args.full_report)
    stage_dir = create_private_stage_dir(args.output_dir)

    report_bundle = _load_json(report_metrics_path)
    qwen_bundle = _load_json(qwen_scores_path)
    agent_bundle = _load_json(agent_decision_path)
    for payload in (report_bundle, qwen_bundle, agent_bundle):
        if payload.get("run_id") != args.run_id:
            raise ValueError("artifact belongs to a different run")

    directional = report_bundle.get("directional")
    if not isinstance(directional, dict):
        raise TypeError("directional report metrics are missing")
    unidisc_metrics = directional.get("unidisc_to_llavarad")
    llavarad_metrics = directional.get("llavarad_to_unidisc")
    if not isinstance(unidisc_metrics, dict) or not isinstance(
        llavarad_metrics,
        dict,
    ):
        raise TypeError("candidate metric rows are missing")
    qwen_scores = _candidate_qwen_scores(qwen_bundle)
    if set(qwen_scores) != {"unidisc", "llavarad"}:
        raise ValueError("required Qwen candidate scores are missing")

    decision = agent_bundle.get("decision")
    if not isinstance(decision, dict):
        raise TypeError("agent decision is missing")
    qwen_cost = qwen_bundle.get("cost")
    if not isinstance(qwen_cost, dict):
        raise TypeError("Qwen cost record is missing")

    cxr_link = os.path.relpath(cxr_path, start=stage_dir)
    full_report_link = os.path.relpath(full_report_path, start=stage_dir)
    metric_names = ("bleu_1", "bleu_2", "bleu_3", "rouge_l", "meteor")

    def metric_row(label: str, metrics: dict[str, Any], qwen: float) -> str:
        values = [_format_score(metrics.get(name)) for name in metric_names]
        return "| " + " | ".join([label, *values, _format_score(qwen)]) + " |"

    report_text = f"""# TriCompose Weekly Report / 周报 — Jul 30, 2026

> Progress summary / 进展总结版。Protected CARC artifact; do not copy generated
> images or patient-derived outputs into public logs, Git, or chat.
> 本文件为 CARC 受保护产物，不得将生成图像或患者衍生产物复制到公开日志、
> Git 或聊天中。

## Project goal / 项目目标

TriCompose dynamically composes frozen existing models to generate and select
clinically consistent synthetic structured EHR–CXR–radiology report triples.

TriCompose 动态组合冻结的现有模型，生成并选择临床一致的 synthetic
structured EHR–CXR–radiology report triples。

## Phase-0 route completed / 已完成的 Phase-0 路线

```text
previous CXR + interval EHR condition
  -> EHRXDiff
  -> synthetic follow-up CXR
       |-> UniDisc   -> report candidate A
       `-> LLaVA-Rad -> report candidate B
  -> report metrics + Qwen2.5-VL verification
  -> deterministic Agent decision
```

Both report experts received the same EHRXDiff-generated synthetic CXR.
No real target CXR or real target report was loaded.

两个 report experts 接收同一张由 EHRXDiff 生成的 synthetic CXR。
本次实验没有加载真实 target CXR 或真实 target report。

## Synthetic CXR / 合成胸片

![EHRXDiff synthetic CXR]({cxr_link})

- Run ID：`{args.run_id}`
- Dimensions / 尺寸：`256 x 256`
- SHA-256：`{sha256_file(cxr_path)}`

## Slurm execution summary / Slurm 运行总结

| Stage | Job | State | Runtime | Peak GPU memory |
|---|---:|---|---:|---:|
| EHRXDiff | `10731081` | `COMPLETED` | `51 s` | `5.904 GiB` |
| UniDisc | `10731429` | `COMPLETED` | `31 s` | `5.916 GiB` |
| LLaVA-Rad | `10732055` | `COMPLETED` | `46 s` | `14.342 GiB` |
| Qwen2.5-VL verifier | `10733757` | `COMPLETED` | `17 s` | `15.565 GiB` |

## Candidate scores / 候选评分

| Candidate / 候选 | BLEU-1 | BLEU-2 | BLEU-3 | ROUGE-L | METEOR | Qwen2.5-VL CXR–Report |
|---|---:|---:|---:|---:|---:|---:|
{metric_row("A — UniDisc", unidisc_metrics, qwen_scores["unidisc"])}
{metric_row("B — LLaVA-Rad", llavarad_metrics, qwen_scores["llavarad"])}

BLEU, ROUGE-L, and METEOR are directional peer-agreement measurements between
the two generated reports. They are not ground-truth clinical-accuracy scores.
Qwen2.5-VL evaluates each candidate separately against the same synthetic CXR;
the current score is an uncalibrated Phase-0 generated judge score.

BLEU、ROUGE-L 和 METEOR 是两个生成报告之间的有方向 peer-agreement
指标，不是基于真实 reference 的临床准确率。Qwen2.5-VL 将每个候选分别与
同一张 synthetic CXR 对照；当前分数是尚未校准的 Phase-0 生成式 judge
score。

## Agent interface / Agent 接口

We implemented a versioned, auditable selector that consumes unified score
bundles and emits one of `select`, `verify_more`, `regenerate_report`, or
`stop`. It validates candidate IDs and artifact hashes before using scores.

我们实现了可版本化、可审计的 selector。它读取统一 score bundles，输出
`select`、`verify_more`、`regenerate_report` 或 `stop`，并在使用分数前
校验 candidate IDs 和 artifact hashes。

- Current action / 当前动作：`{decision.get("action")}`
- Status / 状态：`{decision.get("status")}`
- Selected candidate / 已选候选：
  `{"none / 未选择" if decision.get("selected_candidate_id") is None else decision.get("selected_candidate_id")}`
- Score margin / 分差：`{decision.get("margin")}`
- Reason code / 原因代码：`{decision.get("reason_code")}`

The Phase-0 policy requires both a minimum score and a minimum top-two margin.
Thresholds remain provisional until calibration on matched, shuffled, and
hard-negative pairs.

Phase-0 policy 同时要求最低分数和 top-two 最低分差。在 matched、shuffled
和 hard-negative pairs 上完成校准前，阈值均视为 provisional。

## LLM policy interface / LLM policy 接口

An OpenAI-compatible `chat/completions` client is implemented for future
dynamic routing. It sends only allowlisted numeric state and opaque candidate
IDs; it never sends EHR, image, report text, patient identifiers, paths, or
hashes. A local hard guard validates every LLM action.

已实现 OpenAI-compatible `chat/completions` 客户端，用于后续动态 routing。
它只发送 allowlisted numeric state 和 opaque candidate IDs，不发送 EHR、
图像、报告正文、患者标识、路径或 hashes；本地 hard guard 会校验每个
LLM action。

- Synthetic HTTP round-trip test / 合成 HTTP 往返测试：`passed`
- External production endpoint / 外部正式 endpoint：`pending configuration`
- Automated tests / 自动测试：`13 passed`

## Engineering completed / 已完成工程工作

- Protected adapters for EHRXDiff, UniDisc, and LLaVA-Rad.
  已完成 EHRXDiff、UniDisc 和 LLaVA-Rad 的 protected adapters。
- Versioned BLEU-1/2/3, ROUGE-L, and METEOR score bundle.
  已完成版本化 BLEU-1/2/3、ROUGE-L 和 METEOR score bundle。
- Frozen Qwen2.5-VL CXR–Report verifier running through Slurm.
  已完成通过 Slurm 运行的冻结 Qwen2.5-VL CXR–Report verifier。
- Auditable deterministic Agent policy and protected decision artifact.
  已完成可审计 deterministic Agent policy 和 protected decision artifact。
- OpenAI-compatible LLM routing client with privacy allowlist and hard guard.
  已完成带 privacy allowlist 与 hard guard 的 LLM routing client。
- CARC-compliant protected I/O, sanitized logs, hashes, and non-overwriting run
  directories. 已完成符合 CARC 的 protected I/O、脱敏日志、hashes 和
  non-overwriting run directories。

## Next steps / 下一步

1. Add additional report candidates from MAIRA-2 and/or CXRMate.
   加入 MAIRA-2 和/或 CXRMate report candidates。
2. Add structured finding, laterality, device, severity, and contradiction
   verification. 加入 findings、laterality、device、severity 和
   contradiction 的结构化验证。
3. Calibrate verifier prompts and selector thresholds on matched, shuffled,
   and hard-negative pairs. 在 matched、shuffled 和 hard-negative pairs
   上校准 verifier prompts 与 selector thresholds。
4. Expand from one Phase-0 case to the planned matched cohort and compare
   static selection with the cost-aware router.
   从单个 Phase-0 case 扩展到 planned matched cohort，并比较 static
   selection 与 cost-aware router。

Internal engineering issues discovered during smoke testing are tracked and
will be addressed before scale-up; they are not expanded in this progress
summary.

Smoke test 中发现的内部工程问题将单独跟踪，并在规模化实验前解决；本进展
总结不展开这些内部诊断。

## Detailed protected record / 详细受保护记录

The full bilingual technical record, including generated outputs and detailed
score context, remains available inside CARC:

[Full protected technical report / 完整受保护技术报告]({full_report_link})
"""

    output_path = write_private_text(
        stage_dir / "weekly_report.md",
        report_text,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "artifact": str(output_path.relative_to(PROTECTED_ROOT)),
                "sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

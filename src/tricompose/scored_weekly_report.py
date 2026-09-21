"""Append protected candidate scores to an existing weekly report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_text,
)


SCORE_SCHEMA_VERSION = "tricompose.score_bundle.v1"
QWEN_METRIC = "qwen25vl_cxr_report_match"
CANDIDATE_LABELS = {
    "unidisc": "A — UniDisc",
    "llavarad": "B — LLaVA-Rad",
}
TEXT_METRICS = ("bleu_1", "bleu_2", "bleu_3", "rouge_l", "meteor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--report-metrics", required=True)
    parser.add_argument("--qwen-scores", required=True)
    parser.add_argument("--agent-decision", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_json(
    path: Path,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON artifact must be an object")
    if payload.get("schema_version") != expected_schema:
        raise ValueError("unsupported artifact schema")
    return payload


def _format_score(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("score must be numeric")
    return f"{float(value):.8f}"


def _qwen_scores(bundle: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    records = bundle.get("records")
    if not isinstance(records, list):
        raise TypeError("Qwen score records are missing")
    for record in records:
        if not isinstance(record, dict) or record.get("metric") != QWEN_METRIC:
            continue
        candidate_ids = record.get("candidate_ids")
        value = record.get("value")
        if (
            not isinstance(candidate_ids, list)
            or len(candidate_ids) != 1
            or not isinstance(candidate_ids[0], str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError("invalid Qwen score record")
        scores[candidate_ids[0]] = float(value)
    return scores


def build_score_section(
    report_bundle: dict[str, Any],
    qwen_bundle: dict[str, Any],
    agent_bundle: dict[str, Any],
) -> str:
    if report_bundle.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported report score schema")
    if qwen_bundle.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported Qwen score schema")
    if report_bundle.get("run_id") != qwen_bundle.get("run_id"):
        raise ValueError("score bundles belong to different runs")
    if agent_bundle.get("schema_version") != "tricompose.agent_decision.v1":
        raise ValueError("unsupported agent decision schema")
    if agent_bundle.get("run_id") != qwen_bundle.get("run_id"):
        raise ValueError("agent decision belongs to a different run")
    if report_bundle.get("comparison_role") != "peer":
        raise ValueError("weekly candidate table expects peer report metrics")

    directional = report_bundle.get("directional")
    if not isinstance(directional, dict):
        raise TypeError("directional report metrics are missing")
    qwen_scores = _qwen_scores(qwen_bundle)

    rows: list[str] = []
    for candidate_id, peer_id in (
        ("unidisc", "llavarad"),
        ("llavarad", "unidisc"),
    ):
        metrics = directional.get(f"{candidate_id}_to_{peer_id}")
        if not isinstance(metrics, dict):
            raise ValueError("required directional report metrics are missing")
        if candidate_id not in qwen_scores:
            raise ValueError("required Qwen candidate score is missing")
        metric_cells = [_format_score(metrics.get(name)) for name in TEXT_METRICS]
        rows.append(
            "| "
            + " | ".join(
                [
                    CANDIDATE_LABELS[candidate_id],
                    *metric_cells,
                    _format_score(qwen_scores[candidate_id]),
                ]
            )
            + " |"
        )

    agent_decision = agent_bundle.get("decision")
    if not isinstance(agent_decision, dict):
        raise TypeError("agent decision payload is missing")
    action = agent_decision.get("action")
    status = agent_decision.get("status")
    reason_code = agent_decision.get("reason_code")
    if not all(isinstance(value, str) for value in (action, status, reason_code)):
        raise TypeError("agent decision fields are invalid")
    selected_candidate_id = agent_decision.get("selected_candidate_id")
    if selected_candidate_id is None:
        selection_text = "none / 未选择"
    elif isinstance(selected_candidate_id, str):
        selection_text = CANDIDATE_LABELS.get(
            selected_candidate_id,
            selected_candidate_id,
        )
    else:
        raise TypeError("selected candidate ID is invalid")
    qwen_cost = qwen_bundle.get("cost")
    cost_line = ""
    if isinstance(qwen_cost, dict):
        cost_line = (
            "\n- Qwen model calls / 模型调用数："
            f"`{qwen_cost.get('model_calls', 'unknown')}`"
            "\n- Qwen runtime / 运行时间："
            f"`{qwen_cost.get('elapsed_seconds', 'unknown')} s`"
            "\n- Qwen peak VRAM / 峰值显存："
            f"`{qwen_cost.get('peak_vram_gib', 'unknown')} GiB`\n"
        )

    return f"""
## Automated scores / 自动评分

| Candidate / 候选 | BLEU-1 | BLEU-2 | BLEU-3 | ROUGE-L | METEOR | Qwen2.5-VL CXR–Report |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The five text metrics are directional **peer-agreement** scores: each generated
candidate is treated as the hypothesis and the other generated candidate as
the comparison text. They are listed separately as requested, but they are not
individual clinical-accuracy scores because no allowed ground-truth report was
loaded.

五个文本指标是有方向的**候选间一致性**：每个生成候选分别作为 hypothesis，
另一个生成候选作为对照。这里按候选分别列出，但由于没有加载允许使用的真实
reference report，它们不能解释为单个候选的临床准确率。

Qwen2.5-VL evaluates each report against the same synthetic CXR, so its value
is candidate-specific. It is an uncalibrated generated judge score in `[0, 1]`,
not a probability or native contrastive similarity.

Qwen2.5-VL 将每个 report 分别与同一张 synthetic CXR 对照，因此该分数属于
单候选分数；但它仍是尚未校准的 `[0, 1]` 生成式 judge score，不是概率，
也不是原生对比学习相似度。

- Agent action / Agent 动作：`{action}`
- Agent status / Agent 状态：`{status}`
- Selected candidate / 已选候选：`{selection_text}`
- Reason code / 原因代码：`{reason_code}`
{cost_line}
"""


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)

    base_report_path = require_private_file(args.base_report)
    report_metrics_path = require_private_file(args.report_metrics)
    qwen_scores_path = require_private_file(args.qwen_scores)
    agent_decision_path = require_private_file(args.agent_decision)
    stage_dir = create_private_stage_dir(args.output_dir)

    base_report = base_report_path.read_text(encoding="utf-8").rstrip()
    if not base_report:
        raise ValueError("base weekly report is empty")
    report_bundle = _load_json(
        report_metrics_path,
        expected_schema=SCORE_SCHEMA_VERSION,
    )
    qwen_bundle = _load_json(
        qwen_scores_path,
        expected_schema=SCORE_SCHEMA_VERSION,
    )
    agent_bundle = _load_json(
        agent_decision_path,
        expected_schema="tricompose.agent_decision.v1",
    )
    scoring_section = build_score_section(
        report_bundle,
        qwen_bundle,
        agent_bundle,
    )

    output_path = write_private_text(
        stage_dir / "weekly_report.md",
        base_report + "\n" + scoring_section,
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

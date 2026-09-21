"""Assemble a protected weekly report from synthetic run artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bilingual-translations")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)

    if not args.run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        raise ValueError("invalid opaque run ID")

    run_root = PROTECTED_ROOT / args.run_id
    cxr_path = require_private_file(
        run_root / "stage1_ehrxdiff" / "generated_cxr.png"
    )
    unidisc_path = require_private_file(
        run_root / "stage2_unidisc" / "generated_report.txt"
    )
    llavarad_path = require_private_file(
        run_root / "stage2_llavarad" / "generated_report.txt"
    )
    translations_path = (
        require_private_file(args.bilingual_translations)
        if args.bilingual_translations
        else None
    )
    stage_dir = create_private_stage_dir(args.output_dir)

    unidisc_report = unidisc_path.read_text(encoding="utf-8").strip()
    llavarad_report = llavarad_path.read_text(encoding="utf-8").strip()
    if not unidisc_report or not llavarad_report:
        raise ValueError("generated report candidate is empty")

    cxr_link = os.path.relpath(cxr_path, start=stage_dir)
    if translations_path is not None:
        translations = json.loads(
            translations_path.read_text(encoding="utf-8")
        )
        required_translation_keys = {
            "unidisc_zh",
            "llavarad_zh",
            "comparison_zh",
        }
        if (
            not required_translation_keys.issubset(translations)
            or not all(
                isinstance(translations[key], str) and translations[key].strip()
                for key in required_translation_keys
            )
        ):
            raise ValueError("bilingual translations are incomplete")

        report_text = f"""# TriCompose Weekly Report / 周报 — Jul 30, 2026

> Protected CARC artifact / CARC 受保护产物。This report contains outputs
> generated from a real-anchor run and must not be copied into public logs,
> Git, or chat. 本报告包含 real-anchor run 的生成结果，不得复制到公开日志、
> Git 或聊天中。

## Short route / 短路线

```text
previous CXR + interval EHR condition
  -> EHRXDiff
  -> synthetic follow-up CXR
       |-> UniDisc   -> report candidate A
       `-> LLaVA-Rad -> report candidate B
```

Both report experts received the same EHRXDiff-generated synthetic CXR.
No real target CXR or real target report was loaded.

两个 report experts 接收同一张由 EHRXDiff 生成的 synthetic CXR。
本次实验没有加载真实 target CXR 或真实 target report。

## EHRXDiff synthetic CXR / EHRXDiff 合成胸片

![EHRXDiff synthetic CXR]({cxr_link})

- Slurm job / 作业：`10731081`
- Runtime / 运行时间：`51 s`
- Peak GPU memory / 峰值显存：`5.904 GiB`
- Dimensions / 尺寸：`256 x 256`
- SHA-256：`{sha256_file(cxr_path)}`

## Report candidate A — UniDisc

### English original / 英文原始输出

```text
{unidisc_report}
```

### Chinese translation and note / 中文翻译与说明

{translations["unidisc_zh"].strip()}

- Slurm job / 作业：`10731429`
- Runtime / 运行时间：`31 s`
- Peak GPU memory / 峰值显存：`5.916 GiB`
- SHA-256：`{sha256_file(unidisc_path)}`

## Report candidate B — LLaVA-Rad

### English original / 英文原始输出

```text
{llavarad_report}
```

### Chinese translation / 中文翻译

{translations["llavarad_zh"].strip()}

- Slurm job / 作业：`10732055`
- Runtime / 运行时间：`46 s`
- Peak GPU memory / 峰值显存：`14.342 GiB`
- SHA-256：`{sha256_file(llavarad_path)}`

## Preliminary comparison / 初步比较

{translations["comparison_zh"].strip()}

The preliminary selection above evaluates text readability and radiology-report
structure only. Image-grounded factual consistency has not yet been scored.

上述初步选择只评价文本可读性和放射学报告结构，尚未计算图像事实一致性分数。

## Current limitations / 当前限制

- No reference-free image-report clinical scorer has been run yet.
  尚未运行 reference-free image-report 临床一致性 scorer。
- The current EHRXDiff condition uses the Phase-0 1536-dimensional hashing
  placeholder. 当前 EHRXDiff 条件仍使用 Phase-0 的 1536 维 hashing
  placeholder。
- This run validates composition plumbing and preliminary candidate diversity;
  it does not establish clinical correctness.
  本次运行验证组合链路和初步候选差异，不证明临床正确性。
"""
    else:
        report_text = f"""# TriCompose Weekly Report — Jul 30, 2026

> Protected CARC artifact. This report contains outputs generated from a
> real-anchor run and must not be copied into public logs, Git, or chat.

## Short route

```text
previous CXR + interval EHR condition
  -> EHRXDiff
  -> synthetic follow-up CXR
       |-> UniDisc   -> report candidate A
       `-> LLaVA-Rad -> report candidate B
```

Both report experts received the same EHRXDiff-generated synthetic CXR.
No real target CXR or real target report was loaded.

## EHRXDiff synthetic CXR

![EHRXDiff synthetic CXR]({cxr_link})

- Slurm job: `10731081`
- Runtime: `51 s`
- Peak GPU memory: `5.904 GiB`
- Dimensions: `256 x 256`
- SHA-256: `{sha256_file(cxr_path)}`

## Report candidate A — UniDisc

```text
{unidisc_report}
```

- Slurm job: `10731429`
- Runtime: `31 s`
- Peak GPU memory: `5.916 GiB`
- SHA-256: `{sha256_file(unidisc_path)}`

## Report candidate B — LLaVA-Rad

```text
{llavarad_report}
```

- Slurm job: `10732055`
- Runtime: `46 s`
- Peak GPU memory: `14.342 GiB`
- SHA-256: `{sha256_file(llavarad_path)}`

## Current interpretation

- The first two CXR-to-report experts are operational and produce two report
  candidates from exactly the same synthetic image.
- No reference-free clinical scorer or automatic output selector has been run
  yet, so this report does not declare either candidate the winner.
- The current EHRXDiff condition still uses the Phase-0 1536-dimensional
  hashing placeholder; this run validates composition plumbing rather than
  clinical quality.
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

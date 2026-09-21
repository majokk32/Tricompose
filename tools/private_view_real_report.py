#!/usr/bin/env python3
"""Interactively display one authorized real-anchor report without saving it."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from pathlib import Path


WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
DATASET = Path("/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals")
SELECTION_RUN = "medim50_20260803_001"
CASE_PATTERN = re.compile(r"case_[0-9]{3}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display one real MIMIC target report in the current private terminal. "
            "Redirection and pipelines are refused."
        )
    )
    parser.add_argument("case_id", help="Opaque case ID, for example case_000")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not CASE_PATTERN.fullmatch(args.case_id):
        raise SystemExit("error: case ID must match case_NNN")
    if not sys.stdout.isatty():
        raise SystemExit("error: output must be an interactive terminal")

    mapping_file = (
        WORKSPACE
        / "artifacts/protected/medim/runs"
        / SELECTION_RUN
        / "cases"
        / args.case_id
        / "input.json"
    )
    mapping = json.loads(mapping_file.resolve(strict=True).read_text(encoding="utf-8"))
    source_index = int(mapping["source_row_index"])

    manifest_path = (DATASET / "manifest.csv").resolve(strict=True)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        try:
            row = next(itertools.islice(rows, source_index, source_index + 1))
        except StopIteration as exc:
            raise SystemExit("error: source index is outside the manifest") from exc

    report_path = Path(row["report_path"])
    if not report_path.is_absolute():
        report_path = DATASET / report_path
    report = report_path.resolve(strict=True).read_text(
        encoding="utf-8", errors="replace"
    )
    sys.stdout.write(report)
    if report and not report.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

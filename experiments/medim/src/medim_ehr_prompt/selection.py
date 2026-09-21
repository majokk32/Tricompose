"""Select opaque real-anchor cases and create protected EHR prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from medim_ehr_prompt.common import (
    create_run_directory,
    enforce_private_directory_mode,
    load_config,
    sha256_file,
    validate_run_id,
    write_private_json,
)
from medim_ehr_prompt.prompting import DEVICE_FIELDS, DIAGNOSIS_FIELDS, serialize_ehr_prompt


SELECTION_SCHEMA = "medim_ehr_prompt.selection.v1"
CASE_INPUT_SCHEMA = "medim_ehr_prompt.case_input.v1"

STRATA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pneumothorax", ("DX_PNEUMOTHORAX",)),
    ("pleural_effusion", ("DX_PLEURAL_EFFUSION",)),
    ("pneumonia", ("DX_PNEUMONIA",)),
    ("atelectasis", ("DX_ATELECTASIS",)),
    ("congestive_heart_failure", ("DX_CHF",)),
    ("support_device", tuple(DEVICE_FIELDS)),
    ("no_selected_positive", ()),
)


def _positive_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if not columns:
        all_fields = tuple(DIAGNOSIS_FIELDS) + tuple(DEVICE_FIELDS)
        positive = pd.Series(False, index=frame.index)
        for column in all_fields:
            values = frame[column].fillna(0).astype(str).str.lower()
            positive |= values.isin({"1", "1.0", "true", "yes", "y"})
        return ~positive

    positive = pd.Series(False, index=frame.index)
    for column in columns:
        values = frame[column].fillna(0).astype(str).str.lower()
        positive |= values.isin({"1", "1.0", "true", "yes", "y"})
    return positive


def select_rows(
    merged: pd.DataFrame,
    *,
    sample_count: int,
    seed: int,
) -> list[tuple[int, str]]:
    """Return `(dataframe_index, stratum)` without exposing source identifiers."""
    if sample_count < 1:
        raise ValueError("sample count must be positive")
    if merged["subject_id"].isna().any():
        raise ValueError("eligible rows contain missing subject IDs")

    candidates = merged.sample(frac=1.0, random_state=seed).drop_duplicates(
        subset=["subject_id"], keep="first"
    )
    if len(candidates) < sample_count:
        raise ValueError("not enough distinct eligible patients")

    pools: dict[str, list[int]] = {}
    for stratum, columns in STRATA:
        mask = _positive_series(candidates, columns)
        pools[stratum] = candidates.loc[mask].index.tolist()

    selected: list[tuple[int, str]] = []
    used_rows: set[int] = set()
    used_subjects: set[str] = set()
    while len(selected) < sample_count:
        made_progress = False
        for stratum, _ in STRATA:
            while pools[stratum]:
                row_index = pools[stratum].pop(0)
                subject = str(candidates.at[row_index, "subject_id"])
                if row_index in used_rows or subject in used_subjects:
                    continue
                selected.append((row_index, stratum))
                used_rows.add(row_index)
                used_subjects.add(subject)
                made_progress = True
                break
            if len(selected) >= sample_count:
                break
        if not made_progress:
            break

    if len(selected) < sample_count:
        for row_index, row in candidates.iterrows():
            subject = str(row["subject_id"])
            if row_index in used_rows or subject in used_subjects:
                continue
            selected.append((int(row_index), "fill"))
            used_rows.add(int(row_index))
            used_subjects.add(subject)
            if len(selected) >= sample_count:
                break

    if len(selected) != sample_count:
        raise ValueError("failed to select requested number of distinct patients")
    return selected


def _load_eligible(config: dict[str, Any]) -> pd.DataFrame:
    dataset = config["dataset"]
    root = Path(dataset["root"]).resolve(strict=True)
    manifest_path = root / "manifest.csv"
    ehr_path = root / "ehr_structured.csv"

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": str,
            "subject_id": str,
            "study_id": str,
            "hadm_id": str,
        },
        low_memory=False,
    ).reset_index(names="source_row_index")
    ehr = pd.read_csv(ehr_path, dtype={"sample_id": str}, low_memory=False)

    required_manifest = {"sample_id", "subject_id", "ViewPosition", "split", "source_row_index"}
    required_ehr = {"sample_id", "AGE", "GENDER", *DIAGNOSIS_FIELDS, *DEVICE_FIELDS, "WBC", "BNP", "SPO2", "RESP_RATE"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"manifest schema is missing {len(missing)} required columns")
    if missing := required_ehr - set(ehr.columns):
        raise ValueError(f"EHR schema is missing {len(missing)} required columns")

    split = str(dataset["split"])
    allowed_views = {str(view).upper() for view in dataset["allowed_views"]}
    manifest = manifest.loc[
        manifest["split"].astype(str).eq(split)
        & manifest["ViewPosition"].fillna("").astype(str).str.upper().isin(allowed_views)
    ].copy()
    merged = manifest.merge(
        ehr.drop(columns=["split"], errors="ignore"),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("no eligible matched rows")
    return merged


def prepare_run(config_path: str | Path, run_id: str) -> dict[str, Any]:
    os.umask(0o077)
    run_id = validate_run_id(run_id)
    config = load_config(config_path)
    dataset_config = config["dataset"]
    prompt_config = config["prompt"]
    sample_count = int(dataset_config["sample_count"])
    selection_seed = int(dataset_config["selection_seed"])

    merged = _load_eligible(config)
    selected = select_rows(
        merged,
        sample_count=sample_count,
        seed=selection_seed,
    )
    run_dir = create_run_directory(run_id)
    cases_parent = run_dir / "cases"
    cases_parent.mkdir(mode=0o700, exist_ok=False)
    enforce_private_directory_mode(cases_parent)

    case_records: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}
    prompt_hashes: list[str] = []
    for ordinal, (row_index, stratum) in enumerate(selected):
        row = merged.loc[row_index]
        case_id = f"case_{ordinal:03d}"
        case_seed = selection_seed * 1_000_003 + ordinal * 1_009
        facts, prompt = serialize_ehr_prompt(
            row,
            include_measurements=bool(prompt_config["include_measurements"]),
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_hashes.append(prompt_sha256)

        case_dir = cases_parent / case_id
        case_dir.mkdir(mode=0o700, exist_ok=False)
        enforce_private_directory_mode(case_dir)
        write_private_json(
            case_dir / "input.json",
            {
                "schema_version": CASE_INPUT_SCHEMA,
                "run_id": run_id,
                "case_id": case_id,
                "source_row_index": int(row["source_row_index"]),
                "seed": int(case_seed),
                "prompt_version": str(prompt_config["version"]),
                "prompt_sha256": prompt_sha256,
                "ehr_facts": facts,
                "medim_prompt": prompt,
            },
        )
        case_records.append(
            {
                "case_id": case_id,
                "source_row_index": int(row["source_row_index"]),
                "seed": int(case_seed),
                "stratum": stratum,
                "prompt_sha256": prompt_sha256,
            }
        )
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1

    selection_path = write_private_json(
        run_dir / "selection.json",
        {
            "schema_version": SELECTION_SCHEMA,
            "run_id": run_id,
            "sample_count": len(case_records),
            "selection_seed": selection_seed,
            "split": str(dataset_config["split"]),
            "allowed_views": list(dataset_config["allowed_views"]),
            "distinct_patient_constraint": True,
            "stratum_counts": stratum_counts,
            "cases": case_records,
            "prompt_set_sha256": hashlib.sha256("".join(prompt_hashes).encode("ascii")).hexdigest(),
        },
    )
    return {
        "stage": "prepare_cases",
        "status": "ok",
        "run_id": run_id,
        "case_count": len(case_records),
        "artifact": selection_path.name,
        "sha256": sha256_file(selection_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    try:
        result = prepare_run(args.config, args.run_id)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "prepare_cases",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate and expose immutable synthetic CXR candidates without text inputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tricompose.privacy import PROTECTED_ROOT, require_inside, require_private_file, sha256_file


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CASE_ID_PATTERN = re.compile(r"case_[0-9]{3}")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    run_root: Path
    run_schema: str
    generation_schema: str
    frozen_schema: str


SOURCE_SPECS = {
    "sana": SourceSpec(
        name="sana",
        run_root=PROTECTED_ROOT / "chexgenbench_sana" / "runs",
        run_schema="tricompose.chexgenbench_sana.run.v1",
        generation_schema="tricompose.chexgenbench_sana.generation.v1",
        frozen_schema="tricompose.chexgenbench_sana.frozen_run.v1",
    ),
    "pixart": SourceSpec(
        name="pixart",
        run_root=PROTECTED_ROOT / "chexgenbench_pixart" / "runs",
        run_schema="tricompose.chexgenbench_pixart.run.v1",
        generation_schema="tricompose.chexgenbench_pixart.generation.v1",
        frozen_schema="tricompose.chexgenbench_pixart.frozen_run.v1",
    ),
    "radedit": SourceSpec(
        name="radedit",
        run_root=PROTECTED_ROOT / "radedit" / "runs",
        run_schema="tricompose.radedit.run.v1",
        generation_schema="tricompose.radedit.generation.v1",
        frozen_schema="tricompose.radedit.frozen_run.v1",
    ),
}


@dataclass(frozen=True)
class FrozenCXRCase:
    case_id: str
    image_path: Path
    image_sha256: str
    generation_sha256: str


@dataclass(frozen=True)
class FrozenCXRSource:
    model_name: str
    run_id: str
    source_ehr_run_id: str
    frozen_sha256: str
    cases: tuple[FrozenCXRCase, ...]


def _read_private_json(path: Path) -> dict[str, object]:
    resolved = require_private_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON artifact must be an object")
    return payload


def _validate_run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid opaque run ID")
    return value


def _validate_case_id(value: str) -> str:
    if CASE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid opaque case ID")
    return value


def load_frozen_cxr_source(
    *, source_model: str, source_run_id: str, limit: int
) -> FrozenCXRSource:
    if source_model not in SOURCE_SPECS:
        raise ValueError("unsupported frozen CXR source model")
    if limit < 1:
        raise ValueError("case limit must be positive")
    spec = SOURCE_SPECS[source_model]
    run_id = _validate_run_id(source_run_id)
    run = require_inside(spec.run_root / run_id, PROTECTED_ROOT, must_exist=True)
    if not run.is_dir():
        raise ValueError("protected CXR source run is not a directory")

    manifest = _read_private_json(run / "manifest.json")
    summary = _read_private_json(run / "summary.json")
    frozen = _read_private_json(run / "frozen.json")
    if manifest.get("schema_version") != spec.run_schema:
        raise ValueError("unsupported CXR source manifest")
    if manifest.get("run_id") != run_id or frozen.get("run_id") != run_id:
        raise ValueError("CXR source run identity mismatch")
    if summary.get("status") != "completed":
        raise ValueError("CXR source run is incomplete")
    if frozen.get("schema_version") != spec.frozen_schema or frozen.get("status") != "frozen":
        raise ValueError("CXR source run is not frozen")
    if frozen.get("manifest_sha256") != sha256_file(run / "manifest.json"):
        raise ValueError("CXR source manifest hash mismatch")
    if frozen.get("summary_sha256") != sha256_file(run / "summary.json"):
        raise ValueError("CXR source summary hash mismatch")

    manifest_rows = manifest.get("cases")
    frozen_rows = frozen.get("cases")
    if not isinstance(manifest_rows, list) or not isinstance(frozen_rows, list):
        raise ValueError("CXR source case indexes are unavailable")
    expected_count = int(manifest.get("case_count", -1))
    if expected_count != len(manifest_rows) or expected_count != len(frozen_rows):
        raise ValueError("CXR source case count mismatch")
    if limit > expected_count:
        raise ValueError("requested case limit exceeds frozen CXR run")

    cases: list[FrozenCXRCase] = []
    seen: set[str] = set()
    for manifest_row, frozen_row in zip(manifest_rows[:limit], frozen_rows[:limit], strict=True):
        if not isinstance(manifest_row, dict) or not isinstance(frozen_row, dict):
            raise TypeError("CXR source case index is malformed")
        case_id = _validate_case_id(str(manifest_row["case_id"]))
        if case_id in seen or frozen_row.get("case_id") != case_id:
            raise ValueError("CXR source cases are duplicated or reordered")
        seen.add(case_id)
        generation_dir = run / "cases" / case_id / "generation"
        generation_path = require_private_file(generation_dir / "generation.json")
        image_path = require_private_file(generation_dir / "generated_cxr.png")
        generation = _read_private_json(generation_path)
        if generation.get("schema_version") != spec.generation_schema:
            raise ValueError("unsupported CXR generation record")
        generation_hash = sha256_file(generation_path)
        image_hash = sha256_file(image_path)
        if generation_hash != frozen_row.get("generation_sha256"):
            raise ValueError("frozen CXR generation hash mismatch")
        if image_hash != frozen_row.get("image_sha256"):
            raise ValueError("frozen CXR image hash mismatch")
        if image_hash != generation.get("output", {}).get("image_sha256"):
            raise ValueError("CXR image record hash mismatch")
        cases.append(
            FrozenCXRCase(
                case_id=case_id,
                image_path=image_path,
                image_sha256=image_hash,
                generation_sha256=generation_hash,
            )
        )

    return FrozenCXRSource(
        model_name=source_model,
        run_id=run_id,
        source_ehr_run_id=_validate_run_id(str(manifest["source_run_id"])),
        frozen_sha256=sha256_file(run / "frozen.json"),
        cases=tuple(cases),
    )


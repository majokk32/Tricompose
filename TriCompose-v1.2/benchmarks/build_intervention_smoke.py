#!/usr/bin/env python3
"""Prepare a protected, metadata-only controlled-intervention smoke bank.

No image, report text, source EHR, or model inference is read. Mechanical
intervention labels are NOT clinical correctness labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

WORKSPACE = Path("/project2/ruishanl_1185/inference_3mod")
PROTECTED = WORKSPACE / "artifacts" / "protected"
SCHEMA = "tricompose-v12-intervention-smoke-v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def inside_protected(path: str | Path, *, existing: bool) -> Path:
    root = PROTECTED.resolve(strict=True)
    resolved = Path(path).resolve(strict=existing)
    if not resolved.is_relative_to(root):
        raise ValueError("path must be inside artifacts/protected")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(registry_run: str | Path) -> tuple[list[dict], dict]:
    run = inside_protected(registry_run, existing=True)
    table = inside_protected(run / "score_table.jsonl", existing=True)
    manifest_path = inside_protected(run / "manifest.json", existing=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("artifacts", {}).get("score_table.jsonl", {}).get("sha256")
    if not isinstance(expected_hash, str) or file_sha256(table) != expected_hash:
        raise ValueError("registry table does not match its manifest hash")
    rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len(rows) != manifest.get("counts", {}).get("rows"):
        raise ValueError("registry row count mismatch")
    return rows, {"registry_path": str(run), "registry_sha256": file_sha256(table)}


def _selected_rows(rows: list[dict], *, cxr_model: str, seed: int,
                   report_model: str) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for row in rows:
        lineage = row.get("lineage", {})
        for key in ("ehr_sha256", "cxr_sha256", "report_sha256"):
            if not SHA256.fullmatch(str(lineage.get(key, ""))):
                raise ValueError("invalid lineage hash")
        if (lineage.get("cxr_model_id"), lineage.get("cxr_seed"),
                lineage.get("report_model_id")) != (cxr_model, seed, report_model):
            continue
        case = row.get("case_id")
        if not isinstance(case, str) or not case or case in selected:
            raise ValueError("missing or duplicated case ID in fixed path")
        for key in ("cxr_candidate_id", "report_candidate_id"):
            if not isinstance(lineage.get(key), str) or not lineage[key]:
                raise ValueError("candidate ID missing from registry")
        selected[case] = row
    if len(selected) < 3:
        raise ValueError("at least three distinct cases are required")
    return selected


def _donor(base_case: str, cases: list[str], selected: dict[str, dict],
           hash_key: str) -> str:
    base_hash = selected[base_case]["lineage"][hash_key]
    start = cases.index(base_case)
    for offset in range(1, len(cases)):
        candidate = cases[(start + offset) % len(cases)]
        if selected[candidate]["lineage"][hash_key] != base_hash:
            return candidate
    raise ValueError(f"no different {hash_key} donor for a case")


def build_items(rows: list[dict], *, cxr_model: str, seed: int,
                report_model: str, shuffle_seed: int = 0) -> dict[str, list[dict]]:
    selected = _selected_rows(rows, cxr_model=cxr_model, seed=seed,
                              report_model=report_model)
    cases = sorted(selected)
    internal = []
    for case in cases:
        base = selected[case]["lineage"]
        image_donor_case = _donor(case, cases, selected, "cxr_sha256")
        report_donor_case = _donor(case, cases, selected, "report_sha256")
        image_donor = selected[image_donor_case]["lineage"]
        report_donor = selected[report_donor_case]["lineage"]
        for kind, displayed_cxr, displayed_report, donor_case in (
            ("no_corruption", base, base, None),
            ("report_swap", base, report_donor, report_donor_case),
            ("cxr_swap", image_donor, base, image_donor_case),
        ):
            internal.append({
                "case_id": case,
                "base_triple_candidate_id": selected[case]["triple_candidate_id"],
                "ehr_sha256": base["ehr_sha256"],
                "base_cxr_candidate_id": base["cxr_candidate_id"],
                "base_cxr_sha256": base["cxr_sha256"],
                "base_report_candidate_id": base["report_candidate_id"],
                "base_report_sha256": base["report_sha256"],
                "displayed_cxr_candidate_id": displayed_cxr["cxr_candidate_id"],
                "displayed_cxr_sha256": displayed_cxr["cxr_sha256"],
                "displayed_report_candidate_id": displayed_report["report_candidate_id"],
                "displayed_report_sha256": displayed_report["report_sha256"],
                "intervention_type": kind,
                "donor_case_id": donor_case,
            })
    random.Random(shuffle_seed).shuffle(internal)
    blind, resolver, intervention_key = [], [], []
    for number, record in enumerate(internal):
        item_id = f"item_{number:04d}"
        blind.append({"item_id": item_id})
        resolver.append({field: record[field] for field in (
            "case_id", "ehr_sha256", "displayed_cxr_candidate_id",
            "displayed_cxr_sha256", "displayed_report_candidate_id",
            "displayed_report_sha256") } | {"item_id": item_id})
        intervention_key.append({
            "item_id": item_id,
            "intervention_type": record["intervention_type"],
            "intervention_target": {"no_corruption": "none",
                                    "report_swap": "report",
                                    "cxr_swap": "cxr"}[record["intervention_type"]],
            "base_triple_candidate_id": record["base_triple_candidate_id"],
            "base_cxr_candidate_id": record["base_cxr_candidate_id"],
            "base_cxr_sha256": record["base_cxr_sha256"],
            "base_report_candidate_id": record["base_report_candidate_id"],
            "base_report_sha256": record["base_report_sha256"],
            "donor_case_id": record["donor_case_id"],
            "clinical_mismatch_verified": False,
            "clinical_adjudication": "pending_independent_review",
        })
    return {"blind_items": blind, "resolver": resolver,
            "intervention_key": intervention_key}


def write_run(output_root: str | Path, run_id: str, items: dict, source: dict,
              fixed_path: dict, shuffle_seed: int) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid opaque run ID")
    root = inside_protected(output_root, existing=False)
    parent = inside_protected(root.parent, existing=True)
    if not parent.is_dir() or not (parent.stat().st_mode & 0o2000):
        raise ValueError("output parent must be an existing setgid protected directory")
    root.mkdir(exist_ok=True, mode=0o2770)
    os.chmod(root, 0o2770)
    if root.stat().st_gid != parent.stat().st_gid:
        raise PermissionError("output root did not inherit the protected project group")
    target = inside_protected(root / run_id, existing=False)
    if target.exists():
        raise FileExistsError("benchmark run already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".tmp", dir=root))
    os.chmod(temporary, 0o2770)
    try:
        def write_jsonl(name: str, records: list[dict]) -> str:
            path = temporary / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o660)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            os.chmod(path, 0o660)
            return file_sha256(path)

        hashes = {name: write_jsonl(f"{name}.jsonl", records)
                  for name, records in items.items()}
        counts = dict(Counter(row["intervention_type"] for row in items["intervention_key"]))
        manifest = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "source": source,
            "fixed_path": fixed_path,
            "shuffle_seed": shuffle_seed,
            "counts": counts,
            "artifact_sha256": hashes,
            "status": "prepared_not_adjudicated_or_scored",
            "clinical_localization_accuracy": None,
            "warning": "Intervention labels are mechanical, not clinical truth; controls may already be wrong and swaps may remain compatible.",
        }
        manifest_path = temporary / "manifest.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(manifest_path, flags, 0o660)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(manifest_path, 0o660)
        if target.exists():
            raise FileExistsError("benchmark run already exists")
        os.rename(temporary, target)
        os.chmod(target, 0o2770)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cxr-model", default="chexgenbench_sana")
    parser.add_argument("--cxr-seed", type=int, default=0)
    parser.add_argument("--report-model", default="maira2")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()
    os.umask(0o007)
    rows, source = load_registry(args.registry_run)
    items = build_items(rows, cxr_model=args.cxr_model, seed=args.cxr_seed,
                        report_model=args.report_model, shuffle_seed=args.shuffle_seed)
    target = write_run(args.output_root, args.run_id, items, source,
                       {"cxr_model": args.cxr_model, "cxr_seed": args.cxr_seed,
                        "report_model": args.report_model}, args.shuffle_seed)
    print(json.dumps({"status": "prepared_not_scored", "run": str(target),
                      "items": len(items["blind_items"]),
                      "clinical_accuracy": None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

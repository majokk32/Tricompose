"""Compare immutable bridge runs using metadata and hashes only."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from .cxr_contracts import (
    MAIN_PROTECTED_ROOT, RUN_ID_PATTERN, read_json, require_inside,
    sha256_file, private_directory, write_private_json, write_private_text,
)
from .prompts import ACTIVE_PROMPT_MODELS_V11
from .staging import CASE_ID_PATTERN, STAGING_SCHEMA_V11


def _inventory(root: Path) -> dict:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("schema_version") != STAGING_SCHEMA_V11:
        raise ValueError("expected a V1.1 staging run")
    cases = {}
    for line in (root / "cohort.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("case_id", "")
        if not CASE_ID_PATTERN.fullmatch(case_id) or case_id in cases:
            raise ValueError("invalid or duplicate opaque case ID")
        if row.get("status") != "staged":
            raise ValueError("cohort contains an unstaged case")
        case = require_inside(root / "cases" / case_id, root, must_exist=True)
        ehr_hash = sha256_file(case / "synthetic_ehr.json")
        if row.get("source_ehr_sha256") != ehr_hash:
            raise ValueError("canonical EHR hash mismatch")
        meta = read_json(case / "cxr_prompts" / "prompt_manifest.json")
        facts_hash = sha256_file(case / "ehr_facts.json")
        if meta.get("case_id") != case_id or meta.get("ehr_facts_sha256") != facts_hash:
            raise ValueError("prompt manifest lineage mismatch")
        hashes = {}
        intents = set()
        for model in ACTIVE_PROMPT_MODELS_V11:
            info = meta["models"][model]
            path = require_inside(case / info["path"], case, must_exist=True)
            hashes[model] = sha256_file(path)
            if info.get("prompt_sha256") != hashes[model]:
                raise ValueError("prompt hash mismatch")
            intents.add(info["clinical_intent_sha256"])
        if intents != {row["clinical_intent_sha256"]}:
            raise ValueError("shared intent hash mismatch")
        cases[case_id] = {"ehr": ehr_hash, "facts": facts_hash,
                          "intent": row["clinical_intent_sha256"], "prompts": hashes}
    if not cases or manifest.get("source_case_count") != len(cases):
        raise ValueError("staging cohort count mismatch")
    return cases


def compare_inventories(old: dict, new: dict) -> dict:
    if set(old) != set(new) or not old:
        raise ValueError("the fixed cohort must remain identical")
    records = []
    for case_id in sorted(old):
        before, after = old[case_id], new[case_id]
        if before["ehr"] != after["ehr"]:
            raise ValueError("canonical EHR changed")
        for model in ACTIVE_PROMPT_MODELS_V11:
            changed = before["prompts"][model] != after["prompts"][model]
            records.append({
                "case_id": case_id, "model_id": model, "ehr_sha256": after["ehr"],
                "old_prompt_sha256": before["prompts"][model],
                "new_prompt_sha256": after["prompts"][model],
                "old_ehr_facts_sha256": before["facts"],
                "new_ehr_facts_sha256": after["facts"],
                "old_clinical_intent_sha256": before["intent"],
                "new_clinical_intent_sha256": after["intent"],
                "prompt_changed": changed,
                "action": "regenerate_after_smoke" if changed else "reuse_requires_explicit_lineage_certificate",
            })
    changed_cases = sorted({r["case_id"] for r in records if r["prompt_changed"]})
    return {
        "schema_version": "tricompose-bridge-delta-v1",
        "counts": {"cases": len(old), "changed_cases": len(changed_cases),
                   "changed_model_prompts": sum(r["prompt_changed"] for r in records),
                   "unchanged_model_prompts": sum(not r["prompt_changed"] for r in records)},
        "changed_cases": changed_cases,
        "records": records,
        "old_candidates_retagged": False,
        "note": "Identical prompt bytes alone do not rebind an old candidate to new facts. Preserve old lineage and audit reuse separately.",
    }


def write_bridge_delta(*, old_run: str, new_run: str, output_root: str, run_id: str) -> dict:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid opaque run ID")
    old = require_inside(old_run, MAIN_PROTECTED_ROOT, must_exist=True)
    new = require_inside(new_run, MAIN_PROTECTED_ROOT, must_exist=True)
    payload = compare_inventories(_inventory(old), _inventory(new))
    root = require_inside(output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(root, exist_ok=True)
    target = root / run_id
    if target.exists():
        raise FileExistsError("delta run already exists")
    temporary = root / f".{run_id}.{uuid.uuid4().hex}.tmp"
    private_directory(temporary)
    try:
        payload["sources"] = {
            name: {"path": str(path), "manifest_sha256": sha256_file(path / "run_manifest.json")}
            for name, path in (("old", old), ("new", new))
        }
        write_private_json(temporary / "delta.json", payload)
        write_private_text(temporary / "changed_case_ids.txt", "".join(c + "\n" for c in payload["changed_cases"]))
        write_private_text(temporary / "smoke_case_ids.txt", "".join(c + "\n" for c in payload["changed_cases"][:2]))
        for model in ACTIVE_PROMPT_MODELS_V11:
            ids = [r["case_id"] for r in payload["records"] if r["model_id"] == model and r["prompt_changed"]]
            write_private_text(temporary / f"{model}_changed_case_ids.txt", "".join(c + "\n" for c in ids))
        write_private_json(temporary / "manifest.json", {
            "schema_version": "tricompose-bridge-delta-run-v1", "run_id": run_id,
            "counts": payload["counts"], "smoke_selection": "first_two_changed_opaque_ids_no_outcome_selection",
            "artifacts": {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                          for p in sorted(temporary.iterdir())},
        })
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return {"run_directory": str(target), **payload["counts"]}

"""Immutable source-based input audit for already completed synthetic CXR runs.

Reconstruction is NOT a retrospective runtime observation. Original outputs,
candidate JSONs, hashes and manifests remain untouched.
"""
from __future__ import annotations

import ast
import os
import shutil
import uuid
from pathlib import Path

from .cxr_contracts import (
    WORKSPACE, MAIN_PROTECTED_ROOT, RUN_ID_PATTERN, canonical_json_sha256,
    private_directory, read_json, require_inside, sha256_file,
    write_private_json, write_private_text,
)
from .tokenizer_trace import text_sha256

AUDIT_SCHEMA = "tricompose-cxr-tokenizer-audit-v1"
PIPELINE_FILES = {
    "roentgen_v2": "runtime/venvs/roentgen-v2-cu126-v1/lib/python*/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py",
    "chexgenbench_sana": "CheXGenBench/venv/lib/python*/site-packages/diffusers/pipelines/sana/pipeline_sana.py",
    "chexgenbench_pixart": "CheXGenBench/venv/lib/python*/site-packages/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py",
}
ADAPTER_FILES = {
    model: f"experiments/{name}/src/tricompose_{name}/model.py"
    for model, name in (("roentgen_v2", "roentgen_v2"),
                        ("chexgenbench_sana", "chexgenbench_sana"),
                        ("chexgenbench_pixart", "chexgenbench_pixart"))
}


def sana_instruction(source_text):
    """Read a literal default without importing diffusers, torch or model code."""
    tree = ast.parse(source_text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SanaPipeline")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    defaults = dict(zip([a.arg for a in call.args.args][-len(call.args.defaults):], call.args.defaults))
    value = ast.literal_eval(defaults["complex_human_instruction"])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Sana instruction is not a supported literal default")
    return "\n".join(value)


def reconstruct_text(model, supplied, *, instruction=""):
    if model == "roentgen_v2":
        return supplied
    if model == "chexgenbench_sana":
        return instruction + supplied.lower().strip()
    if model == "chexgenbench_pixart":
        return supplied.lower().strip()
    raise ValueError("unsupported CXR tokenizer audit model")


def build_tokenizer_audit(*, cxr_runs, output_root, run_id):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid opaque run ID")
    sources = [require_inside(p, MAIN_PROTECTED_ROOT, must_exist=True) for p in cxr_runs]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("source runs must be nonempty and unique")
    output = require_inside(output_root, MAIN_PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    target = output / run_id
    if target.exists():
        raise FileExistsError("tokenizer audit run already exists")
    temporary = output / f".{run_id}.{uuid.uuid4().hex}.tmp"
    private_directory(temporary)
    records, source_records, seen = [], [], set()
    try:
        private_directory(temporary / "candidates")
        for source in sources:
            manifest = read_json(source / "manifest.json")
            if manifest.get("schema_version") != "tricompose-cxr-candidate-run-v1.1" or manifest.get("frozen_model") is not True:
                raise ValueError("not a frozen V1.1 CXR run")
            if manifest.get("candidate_count") != len(manifest.get("candidates", [])):
                raise ValueError("source candidate count mismatch")
            model = manifest["model_id"]
            files = list(WORKSPACE.glob(PIPELINE_FILES[model]))
            if len(files) != 1:
                raise ValueError("pipeline source could not be resolved unambiguously")
            pipeline = files[0]
            instruction = sana_instruction(pipeline.read_text()) if model == "chexgenbench_sana" else ""
            source_records.append({"path": str(source), "manifest_sha256": sha256_file(source / "manifest.json"),
                                   "model_id": model, "current_pipeline_source_sha256": sha256_file(pipeline),
                                   "current_pipeline_source_path": str(pipeline),
                                   "current_adapter_source_sha256": sha256_file(WORKSPACE / ADAPTER_FILES[model])})
            for row in manifest["candidates"]:
                candidate_path = require_inside(source / row["path"], source, must_exist=True)
                if sha256_file(candidate_path) != row["sha256"]:
                    raise ValueError("candidate hash mismatch")
                candidate = read_json(candidate_path)
                candidate_id = candidate["candidate_id"]
                if candidate_id in seen or not RUN_ID_PATTERN.fullmatch(candidate_id):
                    raise ValueError("invalid or duplicated opaque candidate ID")
                seen.add(candidate_id)
                if candidate["model_id"] != model or candidate_id != row["candidate_id"]:
                    raise ValueError("candidate model/ID mismatch")
                request = read_json(candidate_path.parent / "request.json")
                if canonical_json_sha256(request) != candidate["input_request_sha256"]:
                    raise ValueError("request lineage mismatch")
                prompt = request["inputs"]["final_prompt"]
                prompt_path = require_inside(prompt["path"], MAIN_PROTECTED_ROOT, must_exist=True)
                if sha256_file(prompt_path) != prompt["sha256"] or prompt["sha256"] != candidate["prompt_sha256"]:
                    raise ValueError("supplied prompt hash mismatch")
                image_path = require_inside(candidate["artifact"]["path"], source, must_exist=True)
                if sha256_file(image_path) != candidate["artifact"]["sha256"]:
                    raise ValueError("image hash mismatch")
                supplied = prompt_path.read_text()
                final = reconstruct_text(model, supplied, instruction=instruction)
                case_dir = temporary / "candidates" / candidate_id
                private_directory(case_dir)
                text = write_private_text(case_dir / "reconstructed_tokenizer_input.txt", final)
                record = {
                    "schema_version": AUDIT_SCHEMA, "candidate_id": candidate_id,
                    "case_id": candidate["case_id"], "model_id": model,
                    "cxr_candidate_canonical_sha256": canonical_json_sha256(candidate),
                    "cxr_candidate_file_sha256": row["sha256"],
                    "cxr_image_sha256": candidate["artifact"]["sha256"],
                    "supplied_prompt_sha256": prompt["sha256"],
                    "reconstructed_text_path": f"candidates/{candidate_id}/{text.name}",
                    "reconstructed_text_sha256": sha256_file(text),
                    "pipeline_changes_text": final != supplied,
                    "official_instruction_prefix_sha256": text_sha256(instruction) if instruction else None,
                    "audit_status": "current_source_reconstruction_only",
                    "runtime_observed": False, "actual_token_ids_available": False,
                    "source_artifacts_modified": False,
                    "limitation": "Current source reconstruction does not prove historical runtime inputs or byte-identical software. No tokenizer/model was run.",
                }
                record_path = write_private_json(case_dir / "audit.json", record)
                records.append({"candidate_id": candidate_id, "path": f"candidates/{candidate_id}/audit.json",
                                "sha256": sha256_file(record_path)})
        manifest_path = write_private_json(temporary / "manifest.json", {
            "schema_version": AUDIT_SCHEMA, "run_id": run_id,
            "audit_status": "current_source_reconstruction_only", "runtime_observed": False,
            "source_runs": source_records, "candidate_count": len(records), "records": records,
            "gpu_inference_used": False, "source_artifacts_modified": False,
        })
        manifest_hash = sha256_file(manifest_path)
        os.rename(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {"status": "prepared", "run_directory": str(target), "candidate_count": len(records),
            "manifest_sha256": manifest_hash, "runtime_observed": False}


def audit_references(audit_run, candidates):
    """Validate an exact-bank audit and return text-free lineage references."""
    root = require_inside(audit_run, MAIN_PROTECTED_ROOT, must_exist=True)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != AUDIT_SCHEMA or manifest.get("audit_status") != "current_source_reconstruction_only" or manifest.get("runtime_observed") is not False:
        raise ValueError("unsupported source reconstruction audit")
    bank = {c["candidate_id"]: c for c in candidates}
    refs = {}
    for row in manifest["records"]:
        path = require_inside(root / row["path"], root, must_exist=True)
        if sha256_file(path) != row["sha256"]:
            raise ValueError("tokenizer audit record hash mismatch")
        record = read_json(path)
        key = record["candidate_id"]
        if key not in bank or key in refs or key != row["candidate_id"]:
            raise ValueError("audit candidate is absent or duplicated")
        if record["cxr_candidate_canonical_sha256"] != canonical_json_sha256(bank[key]) or record["cxr_image_sha256"] != bank[key]["artifact"]["sha256"]:
            raise ValueError("audit is bound to a different CXR candidate")
        if record.get("audit_status") != manifest["audit_status"] or record.get("runtime_observed") is not False:
            raise ValueError("audit observation status mismatch")
        text_path = require_inside(root / record["reconstructed_text_path"], root, must_exist=True)
        if sha256_file(text_path) != record["reconstructed_text_sha256"]:
            raise ValueError("reconstructed tokenizer input hash mismatch")
        refs[key] = {"manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                     "record_path": str(path), "record_sha256": row["sha256"],
                     "audit_status": record["audit_status"], "runtime_observed": False}
    if set(refs) != set(bank) or manifest.get("candidate_count") != len(refs):
        raise ValueError("audit does not exactly cover the CXR bank")
    return refs

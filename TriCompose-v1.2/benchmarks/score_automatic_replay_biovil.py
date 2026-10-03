#!/usr/bin/env python3
"""Prepare and score the union selected by the frozen automatic replay.

Prepare: only private cached metadata, no text/image/model opens.
Score: approved GPU Slurm only; frozen BioViL-T, never routing evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.legacy_replay_adapter import SCHEMA as REPLAY_SCHEMA
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    load_cxr_candidates, load_report_candidates, read_report_text,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json)

REQUEST_SCHEMA = "tricompose-automatic-replay-secondary-request-v1"
SCORE_SCHEMA = "tricompose-automatic-replay-secondary-biovil-v1"
PAIR_FIELDS = {"case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
               "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256"}
MODEL_HASHES = {
    "biovil_t_image_model_proj_size_128.pt": "b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77",
    "pytorch_model.bin": "6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c"}


def requested_pairs(outcomes, scores):
    index = {r["triple_candidate_id"]: r for r in scores}
    if len(index) != len(scores):
        raise ValueError("duplicate source triple")
    result = {}
    for outcome in outcomes:
        cid = outcome["selected_candidate_id"]
        if cid is None:
            continue
        if cid not in index or outcome["case_id"] != index[cid]["case_id"]:
            raise ValueError("selected candidate absent from frozen source")
        row, lineage = index[cid], index[cid]["lineage"]
        hashes = outcome["selected_snapshot"]["artifact_hashes"]
        if any(hashes[name] != lineage[name] for name in hashes):
            raise ValueError("selected artifact lineage differs")
        result[cid] = {"case_id": row["case_id"], "triple_candidate_id": cid,
                      **{name: lineage[name] for name in PAIR_FIELDS - {"case_id", "triple_candidate_id"}}}
    if not 1 <= len(result) <= 960:
        raise ValueError("bounded nonempty selected union required")
    return [result[cid] for cid in sorted(result)]


def prepare(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    root = require_inside(args.replay_run, PROTECTED_ROOT, must_exist=True)
    mp = root / "manifest.json"; manifest = read_json(mp)
    if manifest.get("schema_version") != REPLAY_SCHEMA:
        raise ValueError("completed legacy replay required")
    op = require_inside(root / "replay_outcomes.jsonl", root, must_exist=True)
    if (op.stat().st_size > 64 * 1024 * 1024
            or sha256_file(op) != manifest["artifacts"][op.name]["sha256"]):
        raise ValueError("hash-bound bounded replay outcomes required")
    sp = require_inside(manifest["source_paths"]["source_scores"], PROTECTED_ROOT, must_exist=True)
    if (sp.stat().st_size > 4 * 1024 * 1024
            or sha256_file(sp) != manifest["source_sha256"]["source_scores"]):
        raise ValueError("source score table changed")
    outcomes = [json.loads(x) for x in op.read_text().splitlines() if x]
    scores = [json.loads(x) for x in sp.read_text().splitlines() if x]
    return {"schema_version": REQUEST_SCHEMA, "pairs": requested_pairs(outcomes, scores),
            "source_replay_manifest_sha256": sha256_file(mp),
            "source_outcomes_sha256": sha256_file(op), "source_scores_sha256": sha256_file(sp),
            "modality_source": "fully_synthetic", "selection_used_biovil": False,
            "clinical_truth_available": False, "routing_or_calibration_update_allowed": False,
            "text_policy": "full_report_no_silent_truncation_overlength_is_na"}


def validate_pairs(pairs, cxrs, reports):
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 960:
        raise ValueError("bounded secondary pair request required")
    ids = set()
    for pair in pairs:
        if set(pair) != PAIR_FIELDS or pair["triple_candidate_id"] in ids:
            raise ValueError("invalid or duplicate secondary pair")
        ids.add(pair["triple_candidate_id"])
        image = cxrs.get(pair["cxr_candidate_id"])
        report = reports.get(pair["report_candidate_id"])
        if image is None or report is None:
            raise ValueError("requested synthetic artifact absent")
        if (image["case_id"] != pair["case_id"] or report["case_id"] != pair["case_id"]
                or report["parent_cxr_candidate_id"] != pair["cxr_candidate_id"]
                or image["artifact"]["sha256"] != pair["cxr_sha256"]
                or report["artifact"]["sha256"] != pair["report_sha256"]
                or image["ehr_sha256"] != pair["ehr_sha256"]
                or image["ehr_facts_sha256"] != pair["ehr_facts_sha256"]):
            raise ValueError("secondary request case/parent/hash lineage differs")


def score(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved GPU Slurm required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("approved GPU allocation required")
    request_root = require_inside(args.request_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(request_root / "manifest.json")
    rp = require_inside(request_root / "request.json", request_root, must_exist=True)
    if (manifest.get("schema_version") != REQUEST_SCHEMA or rp.stat().st_size > 1024 * 1024
            or sha256_file(rp) != manifest["artifacts"]["request.json"]["sha256"]):
        raise ValueError("frozen secondary request required")
    request = read_json(rp)
    if (request.get("schema_version") != REQUEST_SCHEMA or request.get("modality_source") != "fully_synthetic"
            or request.get("selection_used_biovil") is not False
            or request.get("routing_or_calibration_update_allowed") is not False
            or request.get("text_policy") != "full_report_no_silent_truncation_overlength_is_na"):
        raise ValueError("secondary-only synthetic full-text contract required")
    model = Path(args.model_path).resolve(strict=True)
    for name, digest in MODEL_HASHES.items():
        if sha256_file(model / name) != digest:
            raise ValueError("frozen BioViL-T checkpoint changed")
    from score_report_cxr_biovil import _load_runtime
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        cxrs = load_cxr_candidates(args.cxr_run)
        reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
        pairs = request["pairs"]; validate_pairs(pairs, cxrs, reports)
        image_engine, text_engine = _load_runtime(model, torch.device("cuda:0"))
        torch.cuda.reset_peak_memory_stats()
        image_embeddings, text_embeddings, unavailable = {}, {}, {}
        limit = text_engine.max_allowed_input_length
        with torch.inference_mode():
            for rid in sorted({p["report_candidate_id"] for p in pairs}):
                text = read_report_text(reports[rid]).strip()
                if not text:
                    unavailable[rid] = "empty_report"; continue
                special = set(text_engine.tokenizer.all_special_tokens) - {text_engine.tokenizer.mask_token}
                if any(token in text for token in special):
                    unavailable[rid] = "unsupported_tokenizer_special_token_in_report"; continue
                tokens = text_engine.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
                if len(tokens) > limit:
                    unavailable[rid] = "full_report_exceeds_text_context_no_truncation"; continue
                text_embeddings[rid] = text_engine.get_embeddings_from_prompt([text], normalize=True, verbose=False)
            for iid in sorted({p["cxr_candidate_id"] for p in pairs if p["report_candidate_id"] in text_embeddings}):
                image_embeddings[iid] = image_engine.get_projected_global_embedding(Path(cxrs[iid]["artifact"]["path"]))
        records = []
        for pair in pairs:
            rid, iid = pair["report_candidate_id"], pair["cxr_candidate_id"]
            cosine = None
            if rid in text_embeddings:
                cosine = float((text_embeddings[rid] @ image_embeddings[iid]).detach().float().cpu().item())
                if not math.isfinite(cosine) or not -1.01 <= cosine <= 1.01:
                    raise ValueError("invalid secondary cosine")
            records.append({**pair, "biovil_raw_cosine": cosine,
                "status": "computed_secondary_uncalibrated" if cosine is not None else "not_available",
                "reason": unavailable.get(rid), "calibrated": False})
    if sha256_file(rp) != manifest["artifacts"]["request.json"]["sha256"]:
        raise ValueError("request changed during scoring")
    return {"schema_version": SCORE_SCHEMA, "status": "completed_secondary_biovil",
        "records": records, "request_sha256": sha256_file(rp),
        "producer": {"frozen": True, "model_id": "biovil_t", "checkpoint_sha256": MODEL_HASHES,
            "text_policy": request["text_policy"], "model_max_position_embeddings": limit},
        "counts": {"requested_pairs": len(pairs), "image_encoder_calls": len(image_embeddings),
                   "text_encoder_calls": len(text_embeddings), "unavailable_reports": len(unavailable)},
        "used_for_routing": False, "primary_clinical_metric": False,
        "clinical_truth_available": False, "original_selection_changed": False,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--replay-run", required=True)
    scoring = sub.add_parser("score")
    for name in ("request-run", "model-path"):
        scoring.add_argument("--" + name, required=True)
    for name in ("cxr-run", "report-run"):
        scoring.add_argument("--" + name, action="append", required=True)
    for command in (prep, scoring):
        command.add_argument("--output-root", required=True); command.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"})); return 2
    os.umask(0o007); started = time.monotonic(); temporary = None
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = prepare(args) if args.mode == "prepare" else score(args)
        payload["runtime_seconds"] = round(time.monotonic() - started, 6)
        filename = "request.json" if args.mode == "prepare" else "scores.json"
        output = write_private_json(temporary / filename, payload)
        write_private_json(temporary / "manifest.json", {
            "schema_version": payload["schema_version"], "run_id": args.run_id,
            "artifacts": {filename: {"sha256": sha256_file(output)}},
            "producer_program_sha256": sha256_file(Path(__file__)),
            "original_selection_changed": False})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None: discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "prepared_secondary_request" if args.mode == "prepare" else payload["status"],
        "runtime_seconds": payload["runtime_seconds"], "manifest_sha256": sha256_file(target / "manifest.json"),
        "peak_vram_gib": payload.get("peak_vram_gib")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

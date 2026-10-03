#!/usr/bin/env python3
"""Development-only frozen polarity scoring; no clinical gold or repair.

Reuse the audited CheXbert class mapping and BioViL runtime. Inference is
Slurm-only, with a construction-label-free resolver and bounded batches.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from contracts import (
    CHEXPERT_FINDINGS, PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, read_json, require_inside, sha256_file, validate_finding_states,
    write_private_json, write_private_text,
)
from extract_report_labels_chexbert import CHEXBERT_CLASS_TO_STATE, CHEXBERT_ORDER
from prepare_expanded_polarity import SCHEMA as BANK_SCHEMA, load_expanded_bank
from score_report_cxr_biovil import IMAGE_WEIGHT, _load_runtime
from score_report_polarity import report_text
from verify_candidate_findings_qwen import FINDINGS

SCHEMA = "tricompose-expanded-report-polarity-scores-v1"


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before model/data access")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("bounded batch size required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    resolver, bank = load_expanded_bank(args.bank_run)
    texts = {row["report_sha256"]: report_text(row) for row in resolver}
    base = lambda row: {key: row[key] for key in ("item_id", "image_sha256", "report_sha256")}
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    batches = 0
    records = []
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        if args.mode == "chexbert":
            from tools.chexbert import CheXbert
            checkpoint = Path(args.checkpoint).resolve(strict=True)
            bert = Path(args.bert_path).resolve(strict=True)
            model = CheXbert(ckpt_dir=str(checkpoint.parent), bert_path=str(bert),
                checkpoint_path=checkpoint.name, device=torch.device("cuda:0")).to("cuda:0")
            model.eval().requires_grad_(False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError("CheXbert not frozen")
            ordered, states = sorted(texts), {}
            for start in range(0, len(ordered), args.batch_size):
                batch = ordered[start:start+args.batch_size]
                values = [texts[key] for key in batch]
                tokenized = model.tokenizer(values, truncation=False)
                if any(len(ids) > model.bert.config.max_position_embeddings for ids in tokenized["input_ids"]):
                    raise ValueError("report would be truncated")
                with torch.inference_mode():
                    output = model(list(values)).detach().cpu().tolist()
                if len(output) != len(batch) or any(len(row) != len(CHEXBERT_ORDER) for row in output):
                    raise ValueError("CheXbert output inventory mismatch")
                batches += 1
                for fingerprint, predictions in zip(batch, output, strict=True):
                    states[fingerprint] = validate_finding_states(dict(zip(CHEXBERT_ORDER,
                        (CHEXBERT_CLASS_TO_STATE[value] for value in predictions), strict=True)))
            records = [{**base(row), "finding_states": states[row["report_sha256"]]} for row in resolver]
            producer = {"model_id": "chexbert", "frozen": True,
                "checkpoint_sha256": sha256_file(checkpoint), "bert_config_sha256": sha256_file(bert/"config.json"),
                "tokenizer_vocab_sha256": sha256_file(bert/"vocab.txt"),
                "model_adapter_sha256": sha256_file(Path(CheXbert.__init__.__code__.co_filename)),
                "finding_order": list(CHEXBERT_ORDER), "class_mapping": CHEXBERT_CLASS_TO_STATE}
            counts = {"image_encoder_examples": 0, "report_encoder_examples": len(texts), "forward_batches": batches}
        elif args.mode == "biovil":
            model_path = Path(args.model_path).resolve(strict=True)
            image_engine, text_engine = _load_runtime(model_path, torch.device("cuda:0"))
            images, embeddings = {}, {}
            with torch.inference_mode():
                for row in resolver:
                    fingerprint = row["image_sha256"]
                    if fingerprint not in images:
                        if sha256_file(row["image_path"]) != fingerprint:
                            raise ValueError("fixed image changed before inference")
                        images[fingerprint] = image_engine.get_projected_global_embedding(Path(row["image_path"]))
                for fingerprint, text in sorted(texts.items()):
                    embeddings[fingerprint] = text_engine.get_embeddings_from_prompt([text], normalize=True, verbose=False)
                for row in resolver:
                    value = float((embeddings[row["report_sha256"]] @ images[row["image_sha256"]]).detach().cpu().item())
                    if not math.isfinite(value) or not -1.01 <= value <= 1.01:
                        raise ValueError("invalid raw cosine")
                    records.append({**base(row), "biovil_raw_cosine": round(value, 8)})
            producer = {"model_id": "biovil_t", "frozen": True,
                "image_checkpoint_sha256": sha256_file(model_path/IMAGE_WEIGHT),
                "text_checkpoint_sha256": sha256_file(model_path/"pytorch_model.bin"),
                "loader_sha256": sha256_file(Path(_load_runtime.__code__.co_filename))}
            counts = {"image_encoder_examples": len(images), "report_encoder_examples": len(texts),
                      "forward_batches": len(images)+len(texts)}
        else:
            raise ValueError("unsupported frozen inference mode")
    torch.cuda.synchronize()
    return {"schema_version": SCHEMA, "mode": args.mode, "source": bank,
        "producer": producer, "records": records, "counts": {"items": len(records), **counts},
        "execution": {"torch_version": str(torch.__version__), "gpu_name": torch.cuda.get_device_name(0),
                      "seed": 0, "chexbert_batch_limit": args.batch_size},
        "scorer_read_intervention_key": False, "primary_metric_eligible": False,
        "targeted_repair_approved": False, "clinical_localization_accuracy": None,
        "peak_allocated_vram_including_load_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3)}


def summarize(key, attempts, chexbert, biovil):
    if not key or len(key) > 480:
        raise ValueError("bounded analysis inventory required")
    keys = {row["item_id"]: row for row in key}
    cb = {row["item_id"]: row for row in chexbert["records"]}
    bv = {row["item_id"]: row for row in biovil["records"]}
    if (len(keys) != len(key) or len(cb) != len(chexbert["records"]) or len(bv) != len(biovil["records"])
            or set(keys) != set(cb) or set(keys) != set(bv)):
        raise ValueError("score/answer-key inventories differ")
    for item in keys:
        if any(cb[item][name] != bv[item][name] for name in ("image_sha256", "report_sha256")):
            raise ValueError("scorer input hashes differ")
        validate_finding_states(cb[item]["finding_states"])
    case_ids = {row["case_id"] for row in key}
    attempt_index = {(row["case_id"], row["finding"]): row for row in attempts}
    if len(attempt_index) != len(attempts) or set(attempt_index) != {(case, finding) for case in case_ids for finding in FINDINGS}:
        raise ValueError("construction attempts must include all case/finding denominators")
    controls, details = [], []
    for case in sorted(case_ids):
        items = [row for row in key if row["case_id"] == case]
        unchanged = [row for row in items if row["intervention_type"] == "unchanged"]
        spacing = [row for row in items if row["intervention_type"] == "whitespace_only"]
        if len(unchanged) != 1 or len(spacing) != 1:
            raise ValueError("exactly two controls required for every fixed case")
        cid, sid = unchanged[0]["item_id"], spacing[0]["item_id"]
        if any(bv[row["item_id"]]["image_sha256"] != bv[cid]["image_sha256"] for row in items):
            raise ValueError("fixed image changed within case")
        original = cb[cid]["finding_states"]
        controls.append({"case_id": case, "whitespace_extraction_identical": original == cb[sid]["finding_states"],
            "whitespace_biovil_cosine_delta": round(bv[sid]["biovil_raw_cosine"]-bv[cid]["biovil_raw_cosine"], 8)})
        flipped = set()
        for row in items:
            if row["intervention_type"] in {"unchanged", "whitespace_only"}:
                continue
            finding = row["finding"]
            if row["intervention_type"] != "minimal_polarity_flip" or finding in flipped:
                raise ValueError("unsupported or duplicated finding intervention")
            flipped.add(finding)
            if not attempt_index[(case, finding)]["available"]:
                raise ValueError("rejected construction supplies an intervention")
            source, target = row["edit"]["source_text_assertion"], row["edit"]["edited_text_assertion"]
            if {source, target} != {"positive", "negative"}:
                raise ValueError("invalid mechanical polarity expectation")
            item = row["item_id"]
            edited = cb[item]["finding_states"]
            details.append({"case_id": case, "item_id": item, "finding": finding,
                "original_report_sha256": cb[cid]["report_sha256"],
                "edited_report_sha256": cb[item]["report_sha256"],
                "image_sha256": bv[item]["image_sha256"],
                "expected_source_text_state": source, "expected_edited_text_state": target,
                "source_chexbert_state": original[finding], "edited_chexbert_state": edited[finding],
                "both_text_polarities_extracted": original[finding] == source and edited[finding] == target,
                "edited_text_polarity_extracted": edited[finding] == target,
                "non_target_changed_heads": [name for name in CHEXPERT_FINDINGS if name != finding and original[name] != edited[name]],
                "biovil_cosine_delta": round(bv[item]["biovil_raw_cosine"]-bv[cid]["biovil_raw_cosine"], 8),
                "clinical_mismatch_verified": False})
        if flipped != {finding for finding in FINDINGS if attempt_index[(case, finding)]["available"]}:
            raise ValueError("available interventions are missing")
    per_finding = {}
    for finding in FINDINGS:
        rows = [row for row in details if row["finding"] == finding]
        successful = sum(row["both_text_polarities_extracted"] for row in rows)
        unique = {(row["original_report_sha256"], row["edited_report_sha256"]): row for row in rows}
        unique_success = sum(row["both_text_polarities_extracted"] for row in unique.values())
        per_finding[finding] = {"attempted_cases": len(case_ids), "eligible_pairs": len(rows),
            "unavailable_pairs": len(case_ids)-len(rows), "both_text_polarities_extracted": successful,
            "both_polarity_extraction_fraction": None if not rows else successful/len(rows),
            "unique_text_pairs": len(unique), "unique_pairs_both_polarities_extracted": unique_success,
            "unique_text_pair_extraction_fraction": None if not unique else unique_success/len(unique),
            "direction_counts": dict(Counter(f"{row['expected_source_text_state']}_to_{row['expected_edited_text_state']}" for row in rows))}
    summary = {"fixed_cases": len(case_ids), "eligible_pairs": len(details), "per_finding": per_finding,
        "distinct_image_hashes": len({row["image_sha256"] for row in bv.values()}),
        "distinct_report_hashes": len({row["report_sha256"] for row in cb.values()}),
        "unique_finding_text_pairs": len({(row["finding"], row["original_report_sha256"], row["edited_report_sha256"]) for row in details}),
        "whitespace_controls": {"cases": len(controls),
            "identical_extractions": sum(row["whitespace_extraction_identical"] for row in controls),
            "nonzero_reported_cosine_delta": sum(row["whitespace_biovil_cosine_delta"] != 0 for row in controls)},
        "clinical_localization_accuracy": None, "primary_metric_eligible": False,
        "targeted_repair_approved": False, "benchmark_split": "development",
        "interpretation": "Explicit text polarity and formatting diagnostics; multiple edits share a case/image; not image truth or clinical localization."}
    return summary, details, controls


def analyze(args):
    root = require_inside(args.bank_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    if manifest["schema_version"] != BANK_SCHEMA:
        raise ValueError("unexpected expanded bank")
    sources = {"bank_manifest": root/"manifest.json"}
    rows = {}
    for name in ("intervention_key", "construction_attempts"):
        path = root/f"{name}.jsonl"
        if sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
            raise ValueError("analysis answer-key/attempt hash mismatch")
        sources[name] = path
        rows[name] = [json.loads(line) for line in path.read_text().splitlines() if line]
    scores = []
    for mode, value in (("chexbert", args.chexbert_run), ("biovil", args.biovil_run)):
        run_root = require_inside(value, PROTECTED_ROOT, must_exist=True)
        path = run_root/"scores.json"
        if sha256_file(path) != read_json(run_root/"manifest.json")["artifacts"][path.name]["sha256"]:
            raise ValueError("expanded score artifact hash mismatch")
        payload = read_json(path)
        if (payload.get("schema_version") != SCHEMA or payload.get("mode") != mode
                or payload.get("scorer_read_intervention_key") is not False
                or payload.get("primary_metric_eligible") is not False
                or payload["source"]["manifest_sha256"] != sha256_file(root/"manifest.json")
                or payload["source"]["resolver_sha256"] != sha256_file(root/"resolver.jsonl")):
            raise ValueError("expanded score provenance mismatch")
        sources[mode] = path
        scores.append(payload)
    return (*summarize(rows["intervention_key"], rows["construction_attempts"], *scores), sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("chexbert", "biovil", "analyze"), required=True)
    for name in ("bank-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("checkpoint", "bert-path", "model-path", "chexbert-run", "biovil-run"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mode != "analyze" and not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    os.umask(0o007)
    started = time.monotonic()
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        if args.mode == "analyze":
            payload, details, controls, sources = analyze(args)
        else:
            payload, sources = run(args), {}
        payload["elapsed_seconds"] = round(time.monotonic()-started, 3)
        files = [write_private_json(temporary/("summary.json" if args.mode == "analyze" else "scores.json"), payload)]
        if args.mode == "analyze":
            files.append(write_private_json(temporary/"per_intervention.json", {"records": details}))
            files.append(write_private_json(temporary/"format_controls.json", {"records": controls}))
            stream = io.StringIO()
            if details:
                writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(details)
            files.append(write_private_text(temporary/"per_intervention.csv", stream.getvalue()))
            lines = ["# Development-only polarity results / 开发集否定识别", "",
                "| Finding | Attempted cases | Eligible | Both text states extracted | Case-weighted fraction | Unique text pairs | Deduplicated fraction |",
                "|---|---:|---:|---:|---:|---:|---:|"]
            for finding, row in payload["per_finding"].items():
                lines.append(f"| {finding} | {row['attempted_cases']} | {row['eligible_pairs']} | {row['both_text_polarities_extracted']} | {row['both_polarity_extraction_fraction']} | {row['unique_text_pairs']} | {row['unique_text_pair_extraction_fraction']} |")
            lines += ["", "Fractions measure explicit text polarity extraction, not clinical image/report truth.",
                "Unavailable targets remain in construction denominators; unknown is not absence.",
                "Report duplicate-aware fractions separately; neither repeated cases nor text pairs are clinical gold.",
                "Cosine direction and changes in other heads are diagnostics, not confirmed clinical errors.",
                "All original EHRs/images/reports/winners remain fixed; no thresholds or repair policy fitted.", ""]
            files.append(write_private_text(temporary/"summary.md", "\n".join(lines)))
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "mode": args.mode,
            "run_id": args.run_id, "program_sha256": sha256_file(__file__),
            "source_sha256": {key: sha256_file(path) for key, path in sources.items()},
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_development_polarity_diagnostic", "elapsed_seconds": payload["elapsed_seconds"],
        "manifest_sha256": sha256_file(target/"manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check human-review completeness/evidence and report extractor diagnostics.

No automatic human labels or adjudication, model inference or repair. Completed
reviews require protected synthetic source verification inside Slurm. Pending
template audit is metadata-only and cannot produce an accuracy score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_review import reader_agreement, object_hash
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json)
from prepare_blinded_report_review import SCHEMA, checked_artifact
from report_assertion_challenge import classification_counts, FROZEN_GUARD_SHA
from gate_report_assertion_predictions import selective_counts


def approved_source_reader(resolver):
    by_hash = {}
    for row in resolver:
        by_hash.setdefault(row["report_sha256"], row["report_path"])
    def read(h):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("completed review source verification requires approved Slurm")
        path = require_inside(by_hash[h], PROTECTED_ROOT, must_exist=True)
        if path.stat().st_size > 32768 or sha256_file(path) != h:
            raise ValueError("synthetic review source hash/size differs")
        return path.read_bytes().decode("utf-8")
    return read


def provisional_comparison(items, first, second, predictions, *, read_source=None):
    indices = [{(row["item_id"],row["finding"]): row for row in a["records"]} for a in (first,second)]
    agreed = {key: left["state"] for key,left in indices[0].items()
        if left["status"] == "reviewed" and indices[1][key]["status"] == "reviewed"
        and left["state"] == indices[1][key]["state"]}
    expected = {row["item_id"]:row["report_sha256"] for row in items}
    unique_predictions = {}
    candidate_ids = set()
    for row in predictions:
        if row["candidate_id"] in candidate_ids:
            raise ValueError("duplicate frozen prediction candidate")
        candidate_ids.add(row["candidate_id"])
        if row["item_id"] not in expected or row["report_sha256"] != expected[row["item_id"]]:
            raise ValueError("prediction/review source hash differs")
        states = {key:row[key] for key in ("chexbert","qwen","qwen_contract_status","report_sha256")}
        if row["item_id"] in unique_predictions and unique_predictions[row["item_id"]] != states:
            raise ValueError("same text has incompatible frozen cache states")
        unique_predictions[row["item_id"]] = states
    if predictions and set(unique_predictions) != set(expected):
        raise ValueError("frozen prediction inventory incomplete")
    n = len(items)*4
    comparisons = {}
    for source in ("chexbert", "qwen"):
        rows = []
        for item_id,prediction in unique_predictions.items():
            for finding, state in prediction[source].items():
                key = (item_id,finding)
                if key in agreed:
                    rows.append({"expected": agreed[key], "predicted":
                        "unavailable" if source == "qwen" and prediction["qwen_contract_status"] != "complete" else state})
        comparisons[source] = classification_counts(rows) if rows else None
    scope_gate = None
    comparisons["frozen_literal_readout"] = None
    if agreed and read_source is not None:
        from repair_cached_report_evidence import scope_check
        from compare_report_assertion_sources import literal_readout
        from tricompose_v12.report_assertions import gate_assertion
        if sha256_file(Path(scope_check.__code__.co_filename)) != FROZEN_GUARD_SHA:
            raise ValueError("frozen scope rules changed")
        literals, gated, texts = [], [], {}
        for (item_id,finding), expected_state in agreed.items():
            prediction=unique_predictions[item_id]
            h=prediction["report_sha256"]
            if h not in texts:
                texts[h]=read_source(h)
            proposal=prediction["chexbert"][finding]
            literal=literal_readout(texts[h],finding,scope_check)
            decision=gate_assertion(texts[h],finding,proposal,scope_check)
            literals.append({"expected":expected_state,"predicted":literal["state"]})
            gated.append({"expected":expected_state,"raw_state":proposal,
                "gated_state":decision["state"],"decision":decision["decision"]})
        comparisons["frozen_literal_readout"]=classification_counts(literals)
        scope_gate=selective_counts(gated)
    return {"review_item_finding_denominator": n, "two_reader_matching_states": len(agreed),
        "matching_state_coverage": len(agreed)/n,
        "cached_extractors_against_provisional_reader_agreement": comparisons,
        "scope_gate_risk_coverage_on_reader_agreement": scope_gate,
        "interpretation": "Provisional double-reader agreement only, not adjudicated gold or image correctness. Missing reviews stay in coverage denominator; failed extractors cannot receive unknown-match credit.",
        "adjudicated_gold_created": False, "primary_metric_eligible": False,
        "regeneration_authorized": False}


def run(args):
    packet = require_inside(args.packet_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(packet/"manifest.json")
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported report-only packet")
    library=Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_review.py"
    if manifest["review_library_sha256"] != sha256_file(library):
        raise ValueError("frozen review library changed")
    for name in ("reviewer/items.json", "investigator/resolver.json", "investigator/frozen_predictions.json"):
        checked_artifact(packet,name)
    items = read_json(packet/"reviewer/items.json")["items"]
    if object_hash(items) != manifest["inventory_sha256"]:
        raise ValueError("review packet inventory differs")
    paths = [require_inside(path,PROTECTED_ROOT,must_exist=True) for path in (args.reader_a,args.reader_b)]
    if paths[0] == paths[1]:
        raise ValueError("two separate reader files required")
    reviews = [read_json(path) for path in paths]
    resolver = read_json(packet/"investigator/resolver.json")["records"]
    source_reader=approved_source_reader(resolver)
    progress = reader_agreement(items,*reviews,read_source=source_reader)
    predictions = read_json(packet/"investigator/frozen_predictions.json")["records"]
    summary = {"schema_version": "tricompose-blinded-report-review-audit-v1",
        "status": "pending_human_annotation" if not progress["both_reviewed"] else "provisional_human_review_not_adjudicated",
        "reader_progress": progress, "provisional_comparison": provisional_comparison(items,*reviews,predictions,read_source=source_reader),
        "model_calls": 0, "selection_changed": False, "regeneration_authorized": False,
        "clinical_accuracy": None, "image_error_localization_accuracy": None,
        "previously_used_development_cohort": True, "untouched_final_test": False}
    return summary, {"packet_manifest":packet/"manifest.json","reader_a":paths[0],"reader_b":paths[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet-run","reader-a","reader-b","output-root","run-id"):
        parser.add_argument(f"--{name}",required=True)
    args = parser.parse_args()
    os.umask(0o007)
    temporary = None
    try:
        summary,sources = run(args)
        temporary,target = new_atomic_run(args.output_root,args.run_id)
        path=write_private_json(temporary/"summary.json",summary)
        write_private_json(temporary/"manifest.json",{"schema_version":summary["schema_version"],
            "program_sha256":sha256_file(__file__),"run_id":args.run_id,
            "source_sha256":{name:sha256_file(p) for name,p in sources.items()},
            "artifacts":{path.name:{"sha256":sha256_file(path)}}})
        commit_atomic_run(temporary,target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}))
        return 1
    print(json.dumps({"status":"completed_report_review_audit","model_calls":0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

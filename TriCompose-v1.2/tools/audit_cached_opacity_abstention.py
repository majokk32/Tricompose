#!/usr/bin/env python3
"""CPU-only, post-hoc development audit; no model calls or clinical verdicts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "interfaces"),
               str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"), str(ROOT.parent / "src")]
import benchmark_opacity_assertion_stages_v3 as original
import contracts
from tricompose_v12 import assertion_abstention as policy

BASE = contracts.PROTECTED_ROOT / "tricompose_v1_2"
PARENT = BASE / "opacity_assertion_stages_runs/authored48_stages_v3_12677513"
PARENT_SHA = "73dd5b1a26a0043caac1e4822ff901a496167947d873b12201e6d7c19e4bacce"
PLAN = BASE / "opacity_assertion_stages_plans/authored48_stages_v3_12666569_001"
PLAN_SHA = "238a8e44133c2edc9c69d460a49fcbfbb5e62ef38ed64672699a808eaa088551"
TESTS = ROOT / "tests/test_assertion_abstention.py"
PROTOCOL = ROOT.parent / "docs/opacity_assertion_abstention_protocol.md"


def load_fixed(pins):
    """Bounded authored-only cached files, never patient/candidate bodies."""
    op = original.legacy.OP
    parent = op.metadata(PARENT / "manifest.json", pins, PARENT_SHA)
    plan = op.metadata(PLAN / "manifest.json", pins, PLAN_SHA)
    if parent["plan_manifest_sha256"] != PLAN_SHA:
        raise ValueError("same_pinned_authored_plan_required")
    for root, manifest in ((PARENT, parent), (PLAN, plan)):
        for name, expected in manifest["artifacts"].items():
            path = contracts.require_inside(root / name, root, must_exist=True)
            if contracts.sha256_file(path) != expected:
                raise ValueError("cached_artifact_hash_changed")
            pins[str(path)] = expected
    # Verify the actual decoding programs against both consumed parent bundles.
    decoder_paths = (Path(original.__file__), Path(original.stages.__file__),
        ROOT / "interfaces/opacity_assertion_stages_v2.py", Path(original.legacy.__file__),
        ROOT / "benchmarks/verify_candidate_findings_qwen.py", Path(contracts.__file__))
    for path in decoder_paths:
        key = str(path.resolve())
        expected = parent["sources"].get(key)
        if expected is None or plan["sources"].get(key) != expected or contracts.sha256_file(path) != expected:
            raise ValueError("unchanged_consumed_decoder_required")
        pins[key] = expected
    for path in (Path(__file__), Path(policy.__file__), TESTS, PROTOCOL,
                 Path(original.legacy.OP.__file__)):
        pins[str(path)] = contracts.sha256_file(path)
    # Preserve the full original bank without opening its rows or bodies.
    pins[str(op.BANK / "manifest.json")] = op.BANK_SHA
    op.verify_pins(pins)
    inputs = op.metadata(PLAN / "inputs.json", pins, plan["artifacts"]["inputs.json"])["records"]
    predictions = op.metadata(PARENT / "predictions.json", pins,
                             parent["artifacts"]["predictions.json"])["records"]
    raw = op.metadata(PARENT / "raw_responses.json", pins,
                     parent["artifacts"]["raw_responses.json"])["records"]
    replays = op.metadata(PARENT / "replay_checks.json", pins,
                         parent["artifacts"]["replay_checks.json"])["records"]
    closed = op.metadata(PARENT / "prediction_freeze_receipt.json", pins,
                         parent["artifacts"]["prediction_freeze_receipt.json"])
    if closed["sha256"] != {name: parent["artifacts"][name] for name in
                            ("predictions.json", "raw_responses.json", "replay_checks.json")}:
        raise ValueError("parent_prediction_freeze_required")
    return inputs, predictions, raw, replays, plan


def reparse(inputs, predictions, raw, replays):
    """Use frozen decoders to check 138 existing calls; do not invoke inference."""
    if len(inputs) != 48 or len(predictions) != 48 or len({r["item_id"] for r in inputs}) != 48:
        raise ValueError("exact_fixed_authored_inventory_required")
    raw_index = {r["call_id"]: r for r in raw}
    if len(raw_index) != len(raw):
        raise ValueError("unique_cached_response_ids_required")
    consumed = set()

    def consume(call_id, maximum):
        if call_id not in raw_index or call_id in consumed:
            raise ValueError("exact_single_cached_call_required")
        r = raw_index[call_id]
        if (not isinstance(r["response"], str) or type(r["max_new_tokens"]) is not int or
                r["max_new_tokens"] != maximum or type(r["input_tokens"]) is not int or r["input_tokens"] < 0 or
                type(r["output_tokens"]) is not int or not 0 <= r["output_tokens"] <= maximum or
                type(r["token_limit_reached"]) is not bool or
                r["token_limit_reached"] != (r["output_tokens"] >= maximum)):
            raise ValueError("cached_token_contract_changed")
        consumed.add(call_id)
        return r

    def staged(item, prefix):
        def cached_call(messages, call_id, maximum):
            # Frozen message construction is exercised; messages are never logged.
            if not messages or call_id not in (prefix + "_locator", prefix + "_polarity"):
                raise ValueError("fixed_stage_message_required")
            return consume(call_id, maximum)
        return original.run_staged(item, cached_call, prefix)

    staged_by_id, rows = {}, []
    for item, pred in zip(inputs, predictions):
        if (set(item) != {"item_id", "text", "report_sha256", "segments"} or
                original.stages.digest(item["text"]) != item["report_sha256"] or
                any(item[k] != pred[k] for k in ("item_id", "report_sha256"))):
            raise ValueError("exact_hash_bound_authored_join_required")
        original.stages.validate_inventory(item["text"], item["segments"])
        baseline_raw = consume(item["item_id"] + "_legacy", 512)
        baseline = original.baseline_outcome(baseline_raw["response"], item["text"], baseline_raw)
        baseline.update(response_sha256=original.stages.digest(baseline_raw["response"]), model_calls=1)
        proposed = staged(item, item["item_id"])
        if pred["legacy"] != baseline or pred["staged"] != proposed:
            raise ValueError("cached_outcome_decoder_or_span_changed")
        staged_by_id[item["item_id"]] = proposed
        decision = policy.gate(proposed, baseline, source_contract_checked=True)
        rows.append({"item_id": item["item_id"], "report_sha256": item["report_sha256"],
            "selected_segment_ids": proposed["selected_segment_ids"],
            "source_span_references": proposed["source_span_references"],
            "raw_assertions": proposed["assertions"],
            "raw_response_sha256s": proposed["stage_response_sha256s"], **decision})
    if [r["item_id"] for r in replays] != [inputs[i]["item_id"] for i in (0, 1)]:
        raise ValueError("two_original_replays_required")
    input_index = {r["item_id"]: r for r in inputs}
    for replay in replays:
        rid = replay["item_id"]
        reconstructed = staged(input_index[rid], "replay_" + rid)
        primary = staged_by_id[rid]
        same = all(o["status"] == "complete" for o in (primary, reconstructed)) and all(
            primary[k] == reconstructed[k] for k in ("state", "selected_segment_ids", "assertions"))
        if (replay["outcome"] != reconstructed or replay["same_complete_state_and_ids"] is not same or
                replay["same_stage_response_sha256s"] is not
                (primary["stage_response_sha256s"] == reconstructed["stage_response_sha256s"])):
            raise ValueError("cached_replay_changed")
    if consumed != set(raw_index) or len(consumed) != 138:
        raise ValueError("all_138_original_responses_required")
    return rows


def join_authored(rows, references):
    if len(references) != 48 or len(rows) != 48 or len({r["item_id"] for r in references}) != 48:
        raise ValueError("all_48_authored_keys_required")
    checks = []
    for row, ref in zip(rows, references):
        if any(row[k] != ref[k] for k in ("item_id", "report_sha256")):
            raise ValueError("exact_authored_hash_join_required")
        checks.append({**row, "family": ref["family"], "expected_state": ref["expected_state"]})
    return checks


def csv_text(rows):
    stream = io.StringIO()
    fields = ("item_id", "report_sha256", "raw_status", "raw_state", "crosscheck_status", "crosscheck_state",
              "decision", "soft_retained_state", "soft_comparable", "source_contract_checked",
              "semantic_scope_verified", "hard_action_eligible", "clinical_score", "state_corrected",
              "candidate_dropped", "selection_changed", "regeneration_authorized")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
    return stream.getvalue()


def markdown(summary):
    a = summary["authored"]
    lines = ["# Assertion abstention / 断言可靠性与弃权", "",
        "Same-investigator, known authored development texts; not held-out clinical validation.", "",
        "| Readout / 指标 | Result / 结果 |", "| --- | ---: |",
        f"| Original four-state matches (unchanged) | {a['raw_four_state_matches']}/48 |",
        f"| Original determinate proposals | {a['raw_determinate_proposals']}/48 |",
        f"| Original incorrect determinate | {a['raw_incorrect_determinate']} |",
        f"| Soft retained / coverage denominator | {a['soft_retained']}/48 |",
        f"| Correct soft retained | {a['soft_correct']}/{a['soft_retained']} |",
        f"| Incorrect soft retained | {a['soft_incorrect']} |",
        f"| Incorrect determinate withheld | {a['incorrect_determinate_withheld']} |",
        f"| Correct determinate withheld | {a['correct_determinate_withheld']} |",
        f"| Hard-action eligible | {a['hard_action_eligible']}/48 |",
        "| New model calls | 0 |", "",
        "Conditional match must be read with coverage and lost-correct counts. No overall gated accuracy.",
        "同一模型的两个提示可以共享错误；阳性/阴性一致只表示相关软证据，不能确定 CXR 或 Report 谁错。",
        "未知、不确定、接口失败和弃权均单独保存；弃权不计作正确未知。",
        "Existing literal scope rules do not cover lung opacity; all hard decisions remain disabled.",
        "Original labels, V3 failed language gate, candidate scores and winners remain unchanged.", "",
        "## Decisions / 决策类别", "", "| Decision | Rows / 48 |", "| --- | ---: |"]
    lines.extend(f"| {key} | {value} |" for key, value in summary["availability"]["decision_counts"].items())
    lines += ["", "## All authored families / 全部虚构文本组", "",
        "| Family | Rows | Soft retained | Correct retained | Incorrect retained | Correct determinate lost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for family, r in summary["per_family"].items():
        lines.append(f"| {family} | {r['authored_rows']} | {r['soft_retained']} | {r['soft_correct']} | {r['soft_incorrect']} | {r['correct_determinate_withheld']} |")
    lines += ["", "CPU cached aggregation uses no inference; the already executed two paths cost 134 primary",
        "model calls plus four replays. This is not an inference-cost saving experiment.", ""]
    return "\n".join(lines)


def execute(output_root, run_id):
    original.legacy.OP.guard()
    pins = {}
    inputs, predictions, raw, replays, plan = load_fixed(pins)
    temporary, target = contracts.new_atomic_run(output_root, run_id)
    started = time.monotonic()
    try:
        rows = reparse(inputs, predictions, raw, replays)
        original.dump_private(temporary / "policy.json", policy.POLICY)
        original.dump_private(temporary / "gated_predictions.json", {"schema_version": policy.SCHEMA, "records": rows})
        closed = {name: contracts.sha256_file(temporary / name) for name in ("policy.json", "gated_predictions.json")}
        original.dump_private(temporary / "freeze_receipt.json", {"sha256": closed,
            "reference_semantics_read_before_gate_fsync": False,
            "investigator_already_knows_development_results": True})
        # No reference-state, family, or correct-ID field participates in gating.
        references = original.legacy.OP.metadata(PLAN / "references.json", pins,
                        plan["artifacts"]["references.json"])["records"]
        checks = join_authored(rows, references)
        summary = {"schema_version": policy.SCHEMA, "availability": policy.summarize(rows),
            "authored": policy.authored_readout(checks),
            "per_family": {f: policy.authored_readout([r for r in checks if r["family"] == f])
                           for f in sorted({r["family"] for r in checks})},
            "reparsed_cached_responses": len(raw), "new_model_calls": 0,
            "already_executed_primary_calls": sum(p["legacy"]["model_calls"] +
                           sum(p["staged"]["stage_model_calls"].values()) for p in predictions),
            "already_executed_replay_calls": sum(sum(r["outcome"]["stage_model_calls"].values()) for r in replays),
            "elapsed_cpu_seconds_before_serialization": round(time.monotonic() - started, 6),
            "reference_read_after_gate_fsync": True, "investigator_known_development_diagnostic": True,
            "heldout_test": False, "patient_inputs_read": False, "candidate_bodies_read": False,
            "clinical_qualification": False, "primary_metric_eligible": False,
            "original_language_gate_changed": False, "selection_changed": False,
            "regeneration_authorized": False, "fixed_ehr_changed": False,
            "candidate_bank_manifest_sha256_unchanged": original.legacy.OP.BANK_SHA}
        original.dump_private(temporary / "scored_checks.json", {"records": checks})
        original.dump_private(temporary / "summary.json", summary)
        contracts.write_private_text(temporary / "assertion_eligibility_table.csv", csv_text(rows))
        contracts.write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary))
        original.legacy.OP.verify_pins(pins)
        if any(contracts.sha256_file(temporary / name) != value for name, value in closed.items()):
            raise ValueError("frozen_policy_or_decisions_changed")
        original.dump_private(temporary / "manifest.json", {"schema_version": policy.SCHEMA + "-manifest",
            "sources": pins, "artifacts": {p.name: contracts.sha256_file(p) for p in sorted(temporary.iterdir())},
            "policy_sha256": policy.digest(policy.POLICY), "new_model_calls": 0,
            "primary_metric_eligible": False, "selection_changed": False, "regeneration_authorized": False})
        for path in (temporary, *temporary.iterdir()):
            stat = path.stat()
            if stat.st_gid not in (96293, 65534) or stat.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError("protected_project_permissions_required")
        contracts.commit_atomic_run(temporary, target)
    except BaseException:
        contracts.discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(BASE / "opacity_assertion_abstention_runs"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    os.umask(0o007)
    try:
        target, summary = execute(args.output_root, args.run_id)
        print(json.dumps({"status": "completed_cached_abstention_diagnostic", "rows": 48,
            "new_model_calls": 0, "soft_comparable": summary["availability"]["soft_comparable"],
            "hard_action_eligible": summary["availability"]["hard_action_eligible"],
            "manifest_sha256": contracts.sha256_file(target / "manifest.json")}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "failed_closed", "error_type": type(error).__name__}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

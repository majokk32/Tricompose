#!/usr/bin/env python3
"""Join completed action contrasts, without a new scorer or selection policy.

CPU Slurm only. Inputs are authenticated numeric metadata from two already
audited synthetic runs. No narrative, pixels, weights, API or inference.
Different action families have different baselines: this is NOT a randomized
causal comparison, an action ranking, or six independent clinical patients.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path

from audit_fresh_report_agent import MetadataReader
from contracts import (PROTECTED_ROOT, new_atomic_run, commit_atomic_run,
    discard_atomic_run, sha256_file, write_private_json, write_private_text)
from score_free_random_control import cpu_guard
from tricompose_v12 import bounded_regeneration as images
from tricompose_v12 import report_expert_control as reports

VERSION = "tricompose-observed-fresh-action-effects-v1"
BASE = PROTECTED_ROOT / "tricompose_v1_2"
SOURCES = {
    "report": ("current_image_report_runs/current_report2_12851790",
        "ff6cbed07e712dbbfc89dc02ae76115414325348f1936cd24eb755f912b6e1d8",
        "current_image_report_audits/current_report2_12851790_12851223_001",
        "7038d7f0070cbb1b92e93b4a94b91d5711ff218eef342479746b52f6d3886d37"),
    "image": ("cxr_action_diversification_runs/cxr_actions2_12862704",
        "cc1448b3b64e691f00260e8139baddbae730cf3a2b4091f2718f1fa6878a040e",
        "cxr_action_diversification_audits/cxr_actions2_12862704_12851223_001",
        "6f008f2436e3e83ddb42760aa07ec6d06576231e9092c5937a34759b73ec041f"),
}
EDGES = ("ehr_cxr", "ehr_report", "cxr_report")
COUNTS = ("known_reference_facts", "comparable_facts", "supported_facts",
    "supported_positive", "supported_negative", "proxy_opposition_facts", "missing_comparisons")
ACTIONS = ("report_expert_switch", "image_seed_change", "image_generator_renderer_seed_change")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def action_effect(base, alternative, action, comparison, charged):
    """Preserve raw edge denominators, recompute the existing gate, never rank."""
    require(action in ACTIONS and type(charged) is int and charged > 0, "typed_action_charge_required")
    images.validate_row(base); images.validate_row(alternative)
    require(all(base[k] == alternative[k] for k in ("case_id", "ehr_sha256", "ehr_facts_sha256")),
        "fixed_ehr_and_facts_required")
    if action == "report_expert_switch":
        require(base["report_model_id"] != alternative["report_model_id"]
            and all(base[k] == alternative[k] for k in ("cxr_model_id", "seed", "cxr_sha256")),
            "same_image_report_expert_switch_required")
        expected = reports.compare(base, alternative)
    else:
        require(base["report_model_id"] == alternative["report_model_id"], "common_report_expert_required")
        require(base["cxr_sha256"] != alternative["cxr_sha256"], "new_image_bytes_required")
        if action == "image_seed_change":
            require(base["cxr_model_id"] == alternative["cxr_model_id"]
                and base["seed"] != alternative["seed"], "same_generator_new_seed_required")
        else:
            require(base["cxr_model_id"] != alternative["cxr_model_id"], "different_generator_required")
        expected = images.compare(base, alternative)
    require(expected == comparison, "consumed_gate_comparison_changed")
    row = {"case_id": base["case_id"], "action": action, "charged_new_worker_attempts": charged,
        "baseline_candidate_id": base["triple_candidate_id"],
        "alternative_candidate_id": alternative["triple_candidate_id"],
        "ehr_sha256": base["ehr_sha256"], "ehr_facts_sha256": base["ehr_facts_sha256"],
        "exploratory_gate_pass": expected["exploratory_gate_pass"],
        "rejection_reasons": expected["reasons"], "clinical_accuracy": None,
        "clinical_fault_location": None, "measured_saved_calls": None, "clinical_acceptance": False}
    for role, source in (("baseline", base), ("alternative", alternative)):
        for field in ("cxr_model_id", "report_model_id", "seed", "cxr_sha256", "report_sha256"):
            row[role + "_" + field] = source[field]
        for edge in EDGES:
            for metric in (*COUNTS, "support_over_known", "coverage_over_known"):
                row[role + "_" + edge + "_" + metric] = source["raw_edge_readouts"][edge][metric]
        row[role + "_all_three_supported_facts"] = source["receipt"]["all_three_supported_facts"]
        row[role + "_all_three_support_over_known"] = source["receipt"]["all_three_support_over_known"]
        row[role + "_unsupported_temporal_language"] = source["structure"]["unsupported_temporal_comparison_language"]
    for edge in EDGES:
        for metric in COUNTS:
            row["delta_" + edge + "_" + metric] = (
                row["alternative_" + edge + "_" + metric] - row["baseline_" + edge + "_" + metric])
    return row


def table_csv(rows):
    require(bool(rows), "nonempty_action_table_required")
    columns = list(rows[0])
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        require(set(row) == set(columns), "same_action_columns_required")
        writer.writerow({k: "NA" if v is None else "|".join(v) if isinstance(v, list) else v
            for k, v in row.items()})
    return output.getvalue()


def aggregate(rows):
    require(len(rows) == 6 and len({r["case_id"] for r in rows}) == 2
        and len({(r["case_id"], r["action"]) for r in rows}) == 6
        and Counter(r["action"] for r in rows) == Counter({a: 2 for a in ACTIONS}),
        "retain_two_cases_all_three_action_families")
    groups = []
    for action in ACTIONS:
        selected = [r for r in rows if r["action"] == action]
        groups.append({"action": action, "observed_pairs": len(selected),
            "charged_new_worker_attempts": sum(r["charged_new_worker_attempts"] for r in selected),
            "frozen_proxy_gate_passes": sum(r["exploratory_gate_pass"] for r in selected),
            "ehr_image_support_increases": sum(r["delta_ehr_cxr_supported_facts"] > 0 for r in selected),
            "new_report_image_positive_support_increases": sum(r["delta_cxr_report_supported_positive"] > 0 for r in selected)})
    return {"schema_version": VERSION, "fixed_ehr_cases": 2, "observed_pairs": 6, "action_groups": groups,
        "charged_new_worker_attempts_in_two_source_jobs": sum(r["charged_new_worker_attempts"] for r in rows),
        "posthoc_development_summary": True, "comparisons_have_different_baselines": True,
        "action_effects_are_randomized_causal_estimates": False,
        "historical_acquisition_or_planner_cost_refunded": False,
        "new_model_calls_in_this_analysis": 0, "original_selections_changed": False,
        "clinical_accuracy": None, "clinical_fault_location": None, "measured_saved_calls": None,
        "clinical_acceptance": False}


def load_sources():
    readers, data = [], {}
    for kind, (relative, pin, audit_relative, audit_pin) in SOURCES.items():
        reader, auditor = MetadataReader(BASE / relative), MetadataReader(BASE / audit_relative)
        readers.extend((reader, auditor))
        reader.hash(reader.root / "manifest.json", pin)
        manifest = reader.json(reader.root / "manifest.json")
        auditor.hash(auditor.root / "manifest.json", audit_pin)
        am = auditor.json(auditor.root / "manifest.json")
        auditor.hash(auditor.root / "audit.json", am["audit_sha256"])
        audit = auditor.json(auditor.root / "audit.json")
        require(audit["source_manifest_sha256"] == pin and audit["new_model_calls"] == 0
            and audit["source_bodies_parsed"] is False and audit["image_pixels_decoded"] is False
            and audit["plan_manifest_sha256"] == manifest["plan_manifest_sha256"], "authenticated_postflight_required")
        if kind == "report":
            require(audit["exact_new_phase_replay_pass"] is True and audit["csv_bytes_exact_replay"] is True,
                "completed_report_postflight_required")
        else:
            require(audit["ledger_and_numeric_receipts_replayed"] is True and audit["csv_byte_hash_replayed"] is True,
                "completed_image_postflight_required")
        pair_name = "paired_report_comparison.json" if kind == "report" else "paired_proxy_comparison.json"
        names = ["score_rows.json", pair_name, "summary.json", "execution_summary.json"]
        if kind == "image": names.append("slot_outcomes.json")
        bundle = {}
        for name in names:
            reader.hash(reader.root / name, manifest["artifacts"][name]["sha256"])
            bundle[name] = reader.json(reader.root / name)
        data[kind] = bundle
    return data, readers


def join_effects(data):
    result = []
    report = data["report"]
    index = {r["triple_candidate_id"]: r for r in report["score_rows.json"]["records"]}
    require(len(index) == len(report["score_rows.json"]["records"]) == 4, "exact_four_report_rows_required")
    charges = {r["case_id"]: r["charged_new_worker_attempts"] for r in report["execution_summary.json"]["records"]}
    require(len(charges) == 2 and sum(charges.values()) == report["summary.json"]["charged_new_worker_attempts"],
        "report_charge_totals_required")
    for pair in report["paired_report_comparison.json"]["records"]:
        base, alt = index[pair["baseline_candidate_id"]], index[pair["proposed_candidate_id"]]
        require(pair["case_id"] == base["case_id"] and pair["clinical_acceptance"] is False,
            "same_report_pair_case_required")
        result.append(action_effect(base, alt, ACTIONS[0], pair["comparison"], charges[pair["case_id"]]))
    image = data["image"]
    new_rows = {r["triple_candidate_id"]: r for r in image["score_rows.json"]["records"]}
    require(len(new_rows) == len(image["score_rows.json"]["records"]) == 4, "exact_four_image_rows_required")
    slots = image["slot_outcomes.json"]["records"]
    require(len(slots) == 4 and all(s["status"] == "completed_unvalidated" for s in slots)
        and sum(s["charged_worker_requests"] for s in slots) == image["summary.json"]["charged_worker_requests"],
        "all_four_completed_slots_and_charges_required")
    for pair in image["paired_proxy_comparison.json"]["records"]:
        base, alt = pair["baseline_row"], pair["new_row"]
        require(new_rows[alt["triple_candidate_id"]] == alt and pair["case_id"] == base["case_id"]
            and pair["diagnostic_only_no_selection"] is True, "bound_unpromoted_image_pair_required")
        matching = [s for s in slots if (s["case_id"], s["cxr_model_id"], s["seed"]) ==
            (alt["case_id"], alt["cxr_model_id"], alt["seed"])]
        require(len(matching) == 1, "unique_exact_image_slot_required")
        action = ACTIONS[1] if base["cxr_model_id"] == alt["cxr_model_id"] else ACTIONS[2]
        result.append(action_effect(base, alt, action, pair["frozen_proxy_preservation_comparison"],
            matching[0]["charged_worker_requests"]))
    result.sort(key=lambda r: (r["case_id"], ACTIONS.index(r["action"])))
    aggregate(result)  # Complete inventory, not outcome-filtered rows.
    for case in {r["case_id"] for r in result}:
        require(len({(r["ehr_sha256"], r["ehr_facts_sha256"]) for r in result if r["case_id"] == case}) == 1,
            "same_ehr_across_action_families_required")
    return result


def build(args):
    cpu_guard()  # Before metadata reads or filesystem writes; never login/GPU.
    data, readers = load_sources()
    rows = join_effects(data)
    summary = aggregate(rows)
    summary["source_job_worker_wall_seconds_including_startup_io"] = {
        "12851790": data["report"]["summary.json"]["runtime_seconds"],
        "12862704": data["image"]["summary.json"]["runtime_seconds_includes_startup_and_io"]}
    pins = {}
    for reader in readers:
        reader.recheck(); pins.update(reader.pins)
    source_files = (Path(__file__), Path(__file__).resolve().parents[1] / "tests/test_fresh_action_effects.py",
        Path(__file__).resolve().parents[2] / "docs/fresh_action_effects_protocol.md",
        Path(images.__file__), Path(reports.__file__))
    code_pins = {str(p): sha256_file(p) for p in source_files}
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(tmp / "action_effects.json", {"schema_version": VERSION, "records": rows})
        write_private_text(tmp / "action_effects.csv", table_csv(rows))
        write_private_json(tmp / "summary.json", summary)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION, "status": "completed_cpu_numeric_summary",
            "artifacts": {name: {"sha256": sha256_file(tmp / name)}
                for name in ("action_effects.json", "action_effects.csv", "summary.json")},
            "source_pins": code_pins, "input_metadata_pins": pins,
            "new_model_calls": 0, "original_selections_changed": False, "clinical_acceptance": False})
        for p, digest in code_pins.items(): require(sha256_file(p) == digest, "code_changed_during_analysis")
        for reader in readers: reader.recheck()
        commit_atomic_run(tmp, target)
    except BaseException:
        discard_atomic_run(tmp); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASE / "fresh_action_effects")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        root = build(args)
        print(json.dumps({"stage": VERSION, "status": "completed_cpu_only",
            "manifest_sha256": sha256_file(root / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

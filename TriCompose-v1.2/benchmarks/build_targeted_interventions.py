#!/usr/bin/env python3
"""Build a blinded, label-targeted synthetic artifact intervention bank.

Only protected candidate metadata and frozen label outputs are read. A changed
artifact is known mechanically; it is not known to be clinically incorrect.
The labels used to choose donors must not double as independent evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from build_intervention_smoke import (_selected_rows, file_sha256,
                                      inside_protected, load_registry, write_run)

FINDINGS = ("edema", "pleural_effusion")
EXPLICIT = frozenset({"positive", "negative"})
SCHEMAS = {
    "report": "tricompose-report-finding-labels-v1.1",
    "cxr": "tricompose-cxr-finding-labels-v1.1",
}


def load_labels(path: str | Path, kind: str) -> tuple[dict[str, dict], str, dict]:
    if kind not in SCHEMAS:
        raise ValueError("unsupported label kind")
    source = inside_protected(path, existing=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMAS[kind]:
        raise ValueError("wrong frozen-label schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("empty frozen-label records")
    key = "report_candidate_id" if kind == "report" else "cxr_candidate_id"
    hash_key = "report_sha256" if kind == "report" else "image_sha256"
    index = {}
    for row in records:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in index:
            raise ValueError("duplicate or missing candidate ID")
        if not isinstance(row.get(hash_key), str) or len(row[hash_key]) != 64:
            raise ValueError("missing candidate hash")
        states = row.get("finding_states")
        if not isinstance(states, dict) or any(
            states.get(name) not in {*EXPLICIT, "uncertain", "unknown"}
            for name in FINDINGS
        ):
            raise ValueError("incomplete or invalid frozen finding states")
        index[identifier] = {"sha256": row[hash_key], "states": states}
    return index, file_sha256(source), {
        "schema_version": payload["schema_version"],
        "calibration_status": payload.get("calibration", {}).get("status"),
        "primary_metric_eligible": payload.get("primary_metric_eligible"),
        "records": len(index),
    }


def split_cases(cases: list[str], *, seed: int,
                positive_effusion_cases: set[str] | None = None) -> dict[str, str]:
    """Freeze 60/20/20 EHR-case splits with optional donor-coverage strata."""
    if len(cases) < 5:
        raise ValueError("at least five cases required for split roles")
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate EHR case")
    roles = ("development", "calibration", "final_test")
    ordered = sorted(cases, key=lambda case: hashlib.sha256(
        f"{seed}|{case}".encode()).hexdigest())
    capacities = {"development": len(cases) * 3 // 5,
                  "calibration": len(cases) // 5}
    capacities["final_test"] = len(cases) - sum(capacities.values())
    selected = set() if positive_effusion_cases is None else set(positive_effusion_cases)
    if selected and (not selected.issubset(cases) or len(selected) < 3):
        raise ValueError("stratified split requires three distinct positive donors")
    result = {}
    if selected:
        positives = [case for case in ordered if case in selected]
        for case, role in zip(positives[:3], roles, strict=True):
            result[case] = role
            capacities[role] -= 1
    for case in ordered:
        if case in result:
            continue
        role = next(role for role in roles if capacities[role] > 0)
        result[case] = role
        capacities[role] -= 1
    if any(capacities.values()):
        raise AssertionError("case split capacity mismatch")
    return result


def _checked_selected(rows: list[dict], report_labels: dict,
                      cxr_labels: dict, *, cxr_model: str, seed: int,
                      report_model: str) -> dict[str, dict]:
    selected = _selected_rows(rows, cxr_model=cxr_model, seed=seed,
                              report_model=report_model)
    for row in selected.values():
        lineage = row["lineage"]
        report = report_labels.get(lineage["report_candidate_id"])
        image = cxr_labels.get(lineage["cxr_candidate_id"])
        if report is None or report["sha256"] != lineage["report_sha256"]:
            raise ValueError("report label lineage mismatch")
        if image is None or image["sha256"] != lineage["cxr_sha256"]:
            raise ValueError("CXR label lineage mismatch")
    return selected


def choose_donor(base_case: str, selected: dict[str, dict], labels: dict,
                 *, kind: str, seed: int,
                 split_by_case: dict[str, str]) -> dict | None:
    """Find an opposite explicit *same-modality* label, not an image truth."""
    id_key = "report_candidate_id" if kind == "report" else "cxr_candidate_id"
    hash_key = "report_sha256" if kind == "report" else "cxr_sha256"
    base = selected[base_case]["lineage"]
    own_states = labels[base[id_key]]["states"]
    choices = []
    for donor_case, candidate in selected.items():
        if donor_case == base_case or split_by_case[donor_case] != split_by_case[base_case]:
            continue
        donor = candidate["lineage"]
        if donor[hash_key] == base[hash_key]:
            continue
        donor_states = labels[donor[id_key]]["states"]
        for finding in FINDINGS:
            own = own_states[finding]
            other = donor_states[finding]
            if {own, other} != EXPLICIT:
                continue
            tie = hashlib.sha256(
                f"{seed}|{kind}|{base_case}|{donor_case}|{finding}".encode()
            ).hexdigest()
            choices.append((tie, donor_case, finding, own, other))
    if not choices:
        return None
    _, donor_case, finding, own, other = min(choices)
    return {"donor_case": donor_case, "finding": finding,
            "base_state": own, "donor_state": other}


def build_items(rows: list[dict], report_labels: dict, cxr_labels: dict,
                *, cxr_model: str, seed: int, report_model: str,
                selection_seed: int = 2701) -> tuple[dict[str, list[dict]], dict]:
    selected = _checked_selected(rows, report_labels, cxr_labels,
                                 cxr_model=cxr_model, seed=seed,
                                 report_model=report_model)
    positive_effusion_cases = {
        case for case, row in selected.items()
        if report_labels[row["lineage"]["report_candidate_id"]]["states"]["pleural_effusion"] == "positive"
    }
    strata = positive_effusion_cases if len(positive_effusion_cases) >= 3 else None
    splits = split_cases(sorted(selected), seed=selection_seed,
                         positive_effusion_cases=strata)
    internal = []
    unavailable = Counter()
    targeted_findings = Counter()
    for case in sorted(selected):
        base = selected[case]["lineage"]
        for kind, labels in (("no_corruption", None),
                             ("report_swap", report_labels),
                             ("cxr_swap", cxr_labels)):
            donor = (None if labels is None else
                     choose_donor(case, selected, labels,
                                  kind="report" if kind == "report_swap" else "cxr",
                                  seed=selection_seed, split_by_case=splits))
            if labels is not None and donor is None:
                unavailable[kind] += 1
                continue
            donor_lineage = (None if donor is None else
                             selected[donor["donor_case"]]["lineage"])
            image = donor_lineage if kind == "cxr_swap" else base
            report = donor_lineage if kind == "report_swap" else base
            if donor is not None:
                targeted_findings[f"{kind}:{donor['finding']}"] += 1
            internal.append({
                "case_id": case,
                "benchmark_split": splits[case],
                "base_triple_candidate_id": selected[case]["triple_candidate_id"],
                "ehr_sha256": base["ehr_sha256"],
                "base_cxr_candidate_id": base["cxr_candidate_id"],
                "base_cxr_sha256": base["cxr_sha256"],
                "base_report_candidate_id": base["report_candidate_id"],
                "base_report_sha256": base["report_sha256"],
                "displayed_cxr_candidate_id": image["cxr_candidate_id"],
                "displayed_cxr_sha256": image["cxr_sha256"],
                "displayed_report_candidate_id": report["report_candidate_id"],
                "displayed_report_sha256": report["report_sha256"],
                "intervention_type": kind,
                "donor_case_id": None if donor is None else donor["donor_case"],
                "targeted_finding": None if donor is None else donor["finding"],
                "base_same_modality_state": None if donor is None else donor["base_state"],
                "donor_same_modality_state": None if donor is None else donor["donor_state"],
            })
    random.Random(selection_seed).shuffle(internal)
    blind, resolver, key = [], [], []
    for number, record in enumerate(internal):
        item_id = f"item_{number:04d}"
        blind.append({"item_id": item_id})
        resolver.append({"item_id": item_id, **{field: record[field] for field in (
            "case_id", "benchmark_split", "ehr_sha256",
            "displayed_cxr_candidate_id", "displayed_cxr_sha256",
            "displayed_report_candidate_id", "displayed_report_sha256")}})
        key.append({"item_id": item_id,
                    "benchmark_split": record["benchmark_split"],
                    "intervention_type": record["intervention_type"],
                    "intervention_target": {
                        "no_corruption": "none", "report_swap": "report",
                        "cxr_swap": "cxr"}[record["intervention_type"]],
                    "base_triple_candidate_id": record["base_triple_candidate_id"],
                    "base_cxr_candidate_id": record["base_cxr_candidate_id"],
                    "base_cxr_sha256": record["base_cxr_sha256"],
                    "base_report_candidate_id": record["base_report_candidate_id"],
                    "base_report_sha256": record["base_report_sha256"],
                    "donor_case_id": record["donor_case_id"],
                    "targeted_finding": record["targeted_finding"],
                    "base_same_modality_state": record["base_same_modality_state"],
                    "donor_same_modality_state": record["donor_same_modality_state"],
                    "clinical_mismatch_verified": False,
                    "clinical_adjudication": "pending_independent_review"})
    return {"blind_items": blind, "resolver": resolver,
            "intervention_key": key}, {
            "cases": len(selected), "split_cases": dict(Counter(splits.values())),
                "split_strategy": ("positive_pleural_effusion_coverage_v1" if strata
                                   else "hash_only_insufficient_stratum_v1"),
                "positive_pleural_effusion_cases": len(positive_effusion_cases),
                "unavailable": dict(unavailable),
                "targeted_findings": dict(targeted_findings),
            }


def validate_items(items: dict[str, list[dict]]) -> dict:
    blind, resolver, truth = (items[name] for name in
                              ("blind_items", "resolver", "intervention_key"))
    if not blind or len(blind) != len(resolver) or len(blind) != len(truth):
        raise ValueError("incomplete intervention bank")
    ids = [row["item_id"] for row in blind]
    if any(set(row) != {"item_id"} for row in blind) or len(set(ids)) != len(ids):
        raise ValueError("blind item IDs are missing, duplicated, or leaking")
    by_resolver = {row["item_id"]: row for row in resolver}
    by_truth = {row["item_id"]: row for row in truth}
    if set(by_resolver) != set(ids) or set(by_truth) != set(ids):
        raise ValueError("intervention item IDs do not match")
    if len(by_resolver) != len(resolver) or len(by_truth) != len(truth):
        raise ValueError("duplicated resolver or answer-key item")
    per_case = {}
    split_by_case = {}
    donor_case_by_item = {}
    for item_id in ids:
        displayed = by_resolver[item_id]
        answer = by_truth[item_id]
        if any(name in displayed for name in
               ("intervention_type", "intervention_target", "targeted_finding")):
            raise ValueError("intervention answer leaked into resolver")
        if displayed["benchmark_split"] != answer["benchmark_split"]:
            raise ValueError("split mismatch")
        case = displayed["case_id"]
        if case in split_by_case and split_by_case[case] != displayed["benchmark_split"]:
            raise ValueError("one EHR case crosses benchmark splits")
        split_by_case[case] = displayed["benchmark_split"]
        donor_case_by_item[item_id] = answer["donor_case_id"]
        arms = per_case.setdefault(case, set())
        kind = answer["intervention_type"]
        if kind in arms:
            raise ValueError("duplicate intervention arm for one case")
        arms.add(kind)
        image_same = displayed["displayed_cxr_sha256"] == answer["base_cxr_sha256"]
        report_same = (displayed["displayed_report_sha256"] ==
                       answer["base_report_sha256"])
        if answer.get("clinical_mismatch_verified") is not False:
            raise ValueError("mechanical intervention claimed clinical truth")
        if kind == "no_corruption":
            if not image_same or not report_same or answer["donor_case_id"] is not None:
                raise ValueError("no-corruption arm changed an artifact")
        elif kind in {"cxr_swap", "report_swap"}:
            if image_same != (kind == "report_swap") or report_same != (kind == "cxr_swap"):
                raise ValueError("intervention changed the wrong number of modalities")
            if answer["donor_case_id"] in {None, case}:
                raise ValueError("swap donor is not from a different case")
            if answer["targeted_finding"] not in FINDINGS or {
                answer["base_same_modality_state"],
                answer["donor_same_modality_state"]} != EXPLICIT:
                raise ValueError("swap lacks an explicit opposite finding")
        else:
            raise ValueError("unknown intervention kind")
    if any("no_corruption" not in arms for arms in per_case.values()):
        raise ValueError("a case lacks its untouched control")
    for item_id in ids:
        donor = donor_case_by_item[item_id]
        if donor is not None and split_by_case.get(donor) != by_resolver[item_id]["benchmark_split"]:
            raise ValueError("swap donor crosses benchmark splits")
    return {"items": len(ids), "cases": len(per_case),
            "arm_counts": dict(Counter(row["intervention_type"] for row in truth)),
            "split_cases": dict(Counter(split_by_case.values())),
            "clinical_mismatch_verified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-run", required=True)
    parser.add_argument("--report-labels", required=True)
    parser.add_argument("--cxr-labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cxr-model", default="chexgenbench_sana")
    parser.add_argument("--cxr-seed", type=int, default=0)
    parser.add_argument("--report-model", default="maira2")
    parser.add_argument("--selection-seed", type=int, default=2701)
    args = parser.parse_args()
    os.umask(0o007)
    rows, registry_source = load_registry(args.registry_run)
    reports, report_hash, report_meta = load_labels(args.report_labels, "report")
    cxrs, cxr_hash, cxr_meta = load_labels(args.cxr_labels, "cxr")
    items, counts = build_items(
        rows, reports, cxrs, cxr_model=args.cxr_model, seed=args.cxr_seed,
        report_model=args.report_model, selection_seed=args.selection_seed)
    validation = validate_items(items)
    source = {**registry_source,
              "report_labels_sha256": report_hash, "report_label_metadata": report_meta,
              "cxr_labels_sha256": cxr_hash, "cxr_label_metadata": cxr_meta,
              "construction_uses_frozen_labels": True,
              "same_labels_not_independent_evaluation": True,
              "clinical_mismatch_verified": False,
              "coverage": counts,
              "validation": validation}
    target = write_run(args.output_root, args.run_id, items, source,
                       {"cxr_model": args.cxr_model, "cxr_seed": args.cxr_seed,
                        "report_model": args.report_model,
                        "selection_seed": args.selection_seed,
                        "finding_scope": list(FINDINGS)}, args.selection_seed)
    print(json.dumps({"status": "prepared_not_scored", "run": str(target),
                      "arm_counts": validation["arm_counts"],
                      "coverage": counts, "clinical_accuracy": None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

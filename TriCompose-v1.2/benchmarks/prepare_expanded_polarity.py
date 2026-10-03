#!/usr/bin/env python3
"""Freeze then stage development-only synthetic report interventions.

No scores, clinical labels, original EHR semantics, GPU, real data or API.
Retain every fixed case, all construction rejections, and immutable sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from build_intervention_smoke import _selected_rows, load_registry
from build_report_polarity_benchmark import RESOLVER_FIELDS, whitespace_control
from contracts import (
    PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, load_cxr_candidates,
    load_report_candidates, new_atomic_run, private_directory, read_json,
    require_inside, sha256_file, write_private_json, write_private_text,
)
from verify_candidate_findings_qwen import FINDINGS

PLAN_SCHEMA = "tricompose-report-polarity-development-preregistration-v1"
SCHEMA = "tricompose-expanded-report-polarity-diagnostic-v1"
TARGETS = {
    "atelectasis": (r"\batelectasis\b", "Atelectasis", {"atelectasis"}),
    "cardiomegaly": (r"\bcardiomegaly\b", "Cardiomegaly", {"cardiomegaly"}),
    "consolidation": (r"\bconsolidations?\b", "Consolidation", {"consolidation", "consolidations"}),
    "edema": (r"\b(?:pulmonary\s+)?edema\b", "Pulmonary edema", {"pulmonary", "edema"}),
    "lung_opacity": (r"\b(?:(?:lung|pulmonary|airspace|parenchymal)\s+)?opacit(?:y|ies)\b", "Pulmonary opacity", {"lung", "pulmonary", "airspace", "parenchymal", "opacity", "opacities"}),
    "pleural_effusion": (r"\b(?:pleural\s+)?effusions?\b", "Pleural effusion", {"pleural", "effusion", "effusions"}),
    "pneumonia": (r"\bpneumonia\b", "Pneumonia", {"pneumonia"}),
    "pneumothorax": (r"\bpneumothora(?:x|ces)\b", "Pneumothorax", {"pneumothorax", "pneumothoraces"}),
}
CORE = frozenset("there is are a an the of evidence seen noted present identified evident demonstrated".split())
QUALIFIERS = frozenset("small moderate large trace minimal mild slight severe marked bilateral left right sided basilar basal bibasilar layering focal patchy diffuse in upper lower middle lobe lobar apical dependent lung pulmonary lungs retrocardiac airspace".split())
NEGATION = frozenset({"no", "not", "without", "absent"})


def protocol_hash():
    payload = {"version": "isolated_sentence_bidirectional_eight_findings_v1",
        "targets": {key: [pattern, label, sorted(words)] for key, (pattern, label, words) in TARGETS.items()},
        "core": sorted(CORE), "positive_qualifiers": sorted(QUALIFIERS),
        "negation": sorted(NEGATION), "negative_qualifiers_forbidden": True,
        "finding_order": list(FINDINGS), "one_mention_required": True}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def polarity_edit(text, finding):
    pattern, label, vocabulary = TARGETS[finding]
    mentions = list(re.finditer(pattern, text, re.I))
    if not mentions:
        return None, "no_explicit_target_phrase"
    if len(mentions) != 1:
        return None, "multiple_target_mentions"
    sentences = [match for match in re.finditer(r"[^.!?\n]+[.!?]?", text)
                 if re.search(pattern, match.group(), re.I)]
    if len(sentences) != 1:
        return None, "target_sentence_unavailable"
    match = sentences[0]
    statement = match.group()
    header = re.match(r"\s*(?:findings|impression)\s*:\s*", statement, re.I)
    start = match.start() + (header.end() if header else len(statement) - len(statement.lstrip()))
    original = text[start:match.end()]
    words = re.findall(r"[a-z]+", original.lower())
    negatives = [word for word in words if word in NEGATION]
    if not words or not re.fullmatch(r"[A-Za-z\s.,-]+", original):
        return None, "ambiguous_characters_or_measurement"
    if len(negatives) > 1:
        return None, "multiple_negation_cues"
    if negatives:
        if not set(words) <= CORE | vocabulary | NEGATION:
            return None, "qualified_or_mixed_negative_not_global_absence"
        source, target, replacement = "negative", "positive", f"{label} is present."
    else:
        # Other findings are never allowed through the generic qualifier list.
        if not set(words) <= CORE | vocabulary | QUALIFIERS:
            return None, "mixed_or_uncertain_affirmative_sentence"
        source, target, replacement = "positive", "negative", f"No {label.lower()}."
    changed = text[:start] + replacement + text[match.end():]
    if changed[:start] + original + changed[start+len(replacement):] != text:
        raise ValueError("edit is not reversible")
    return {"text": changed, "span_start": start, "span_end": match.end(),
        "source_statement": original, "replacement_statement": replacement,
        "source_text_assertion": source, "edited_text_assertion": target}, None


def freeze_cases(registry, roles):
    selected = _selected_rows(registry, cxr_model="chexgenbench_sana", seed=0, report_model="chexagent2")
    if len(selected) != 80 or set(selected) != set(roles):
        raise ValueError("expected the fixed 80-case source cohort")
    if Counter(roles.values()) != Counter({"development": 48, "calibration": 16, "final_test": 16}):
        raise ValueError("source case roles changed")
    cases = [{"case_id": case, **selected[case]["lineage"]} for case in sorted(selected)
             if roles[case] == "development"]
    if len({row["ehr_sha256"] for row in cases}) != 48:
        raise ValueError("development EHRs are not unique")
    return cases


def prepare_records(cases, texts):
    if not cases or len(cases) > 48 or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("bounded distinct-case inventory required")
    records, attempts = [], []
    for case in cases:
        text = texts[case["case_id"]]
        if not text.strip() or len(text) > 8192:
            raise ValueError("invalid source synthetic report; case cannot be silently dropped")
        for kind, value in (("unchanged", text), ("whitespace_only", whitespace_control(text))):
            records.append({"case": case, "kind": kind, "finding": None, "text": value, "edit": None})
        for finding in FINDINGS:
            edit, reason = polarity_edit(text, finding)
            attempts.append({"case_id": case["case_id"], "finding": finding,
                "available": edit is not None, "rejection_reason": reason,
                "source_text_assertion": None if edit is None else edit["source_text_assertion"]})
            if edit is not None:
                records.append({"case": case, "kind": "minimal_polarity_flip", "finding": finding,
                                "text": edit["text"], "edit": edit})
    records.sort(key=lambda row: hashlib.sha256(
        f"expanded-v1|{row['case']['ehr_sha256']}|{row['kind']}|{row['finding']}".encode()).hexdigest())
    return records, attempts


def preregister(args):
    split = require_inside(args.split_bank, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(split / "manifest.json")
    resolver = split / "resolver.jsonl"
    if sha256_file(resolver) != manifest["artifact_sha256"]["resolver"]:
        raise ValueError("historical split resolver changed")
    roles = {}
    for line in resolver.read_text().splitlines():
        row = json.loads(line)
        case, role = row["case_id"], row["benchmark_split"]
        if case in roles and roles[case] != role:
            raise ValueError("case has inconsistent historical roles")
        roles[case] = role
    registry = require_inside(manifest["source"]["registry_path"], PROTECTED_ROOT, must_exist=True)
    rows, source = load_registry(registry)
    if source["registry_sha256"] != manifest["source"]["registry_sha256"]:
        raise ValueError("historical registry changed")
    cases = freeze_cases(rows, roles)
    return {"schema_version": PLAN_SCHEMA, "cases": cases, "case_count": 48,
        "fixed_path": "chexgenbench_sana_seed0_chexagent2", "finding_order": list(FINDINGS),
        "benchmark_split": "development", "previously_used_for_static_evaluation": True,
        "paper_primary_or_independent_final_test": False, "selection_uses_scores_or_editability": False,
        "report_text_read_during_preregistration": False, "construction_protocol_sha256": protocol_hash(),
        "source_registry_path": str(registry), "source_registry_sha256": source["registry_sha256"],
        "historical_split_manifest_sha256": sha256_file(split / "manifest.json"),
        "historical_split_resolver_sha256": sha256_file(resolver)}


def stage(args, temporary, target):
    plan_root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    plan_path = plan_root / "case_plan.json"
    if sha256_file(plan_path) != read_json(plan_root / "manifest.json")["artifacts"][plan_path.name]["sha256"]:
        raise ValueError("preregistered case plan changed")
    plan = read_json(plan_path)
    if (plan.get("schema_version") != PLAN_SCHEMA or plan["case_count"] != 48
            or plan["benchmark_split"] != "development" or plan["finding_order"] != list(FINDINGS)
            or plan["construction_protocol_sha256"] != protocol_hash()):
        raise ValueError("construction differs from frozen preregistration")
    cases = plan["cases"]
    if len(cases) != 48 or len({row["case_id"] for row in cases}) != 48:
        raise ValueError("fixed case count mismatch")
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    texts, artifacts = {}, {}
    for case in cases:
        image, report = cxrs[case["cxr_candidate_id"]], reports[case["report_candidate_id"]]
        if (image["case_id"] != case["case_id"] or report["case_id"] != case["case_id"]
                or image["model_id"] != "chexgenbench_sana" or report["model_id"] != "chexagent2"
                or image["artifact"]["sha256"] != case["cxr_sha256"]
                or report["artifact"]["sha256"] != case["report_sha256"]
                or image["ehr_sha256"] != case["ehr_sha256"]
                or report["source_report_or_real_target_supplied"] is not False):
            raise ValueError("fixed synthetic path lineage mismatch")
        path = require_inside(report["artifact"]["path"], PROTECTED_ROOT, must_exist=True)
        if path.stat().st_size > 32768:
            raise ValueError("report exceeds staging bound")
        texts[case["case_id"]] = path.read_bytes().decode("utf-8")
        artifacts[case["case_id"]] = (image["artifact"], report["artifact"])
    records, attempts = prepare_records(cases, texts)
    private_directory(temporary / "reports")
    resolver, key, files = [], [], []
    for number, record in enumerate(records):
        item = f"item_{number:04d}"
        case = record["case"]
        image, report = artifacts[case["case_id"]]
        relative = Path("reports") / f"{item}.txt"
        artifact = write_private_text(temporary / relative, record["text"])
        files.append(artifact)
        fingerprint = sha256_file(artifact)
        if record["kind"] == "unchanged" and fingerprint != case["report_sha256"]:
            raise ValueError("unaltered control is not byte-identical")
        resolver.append({"item_id": item, "image_path": image["path"], "image_sha256": image["sha256"],
                         "report_path": str(target / relative), "report_sha256": fingerprint})
        key.append({"item_id": item, "case_id": case["case_id"], "finding": record["finding"],
            "intervention_type": record["kind"], "edit": record["edit"], "ehr_sha256": case["ehr_sha256"],
            "source_report_sha256": case["report_sha256"], "clinical_mismatch_verified": False})
    for name, rows in (("resolver", resolver), ("intervention_key", key), ("construction_attempts", attempts)):
        files.append(write_private_text(temporary / f"{name}.jsonl", "".join(json.dumps(row, sort_keys=True)+"\n" for row in rows)))
    coverage = {}
    for finding in FINDINGS:
        rows = [row for row in attempts if row["finding"] == finding]
        coverage[finding] = {"attempted_cases": len(rows), "available": sum(row["available"] for row in rows),
            "positive_to_negative": sum(row["source_text_assertion"] == "positive" for row in rows),
            "negative_to_positive": sum(row["source_text_assertion"] == "negative" for row in rows),
            "rejection_reasons": dict(Counter(row["rejection_reason"] for row in rows if not row["available"]))}
    summary = {"schema_version": SCHEMA, "case_count": 48, "finding_order": list(FINDINGS),
        "benchmark_split": "development", "counts": {"reports": len(records), "unchanged_controls": 48,
            "whitespace_controls": 48, "attempts": len(attempts), "eligible_flips": len(records)-96,
            "unavailable_flips": sum(not row["available"] for row in attempts)},
        "coverage": coverage, "clinical_localization_accuracy": None, "primary_metric_eligible": False,
        "targeted_repair_approved": False, "original_artifacts_or_winners_changed": False}
    files.append(write_private_json(temporary / "summary.json", summary))
    lines = ["# Development-only report polarity bank / 开发集否定测试", "",
        "Frozen Sana seed 0 -> CheXagent-2, 48 historical development cases only.",
        "No calibration/final-test report text was opened; no scores or availability selected cases.",
        "Eight named findings; every construction attempt and rejection is retained.", "",
        "| Finding | Attempted | Positive -> negative | Negative -> positive | Unavailable |",
        "|---|---:|---:|---:|---:|"]
    for name, row in coverage.items():
        lines.append(f"| {name} | {row['attempted_cases']} | {row['positive_to_negative']} | {row['negative_to_positive']} | {row['attempted_cases']-row['available']} |")
    lines += ["", "Original controls are NOT clinical gold. Edits test explicit text polarity, not image truth.",
              "No threshold fitting, clinical localization, selection or repair is permitted by this bank.", ""]
    files.append(write_private_text(temporary / "summary.md", "\n".join(lines)))
    return files, summary, {"case_plan_sha256": sha256_file(plan_path),
        "construction_protocol_sha256": protocol_hash(),
        "generation_manifest_sha256": {str(Path(run).resolve()): sha256_file(Path(run)/"manifest.json")
                                      for run in [*args.cxr_run, *args.report_run]}}


def load_expanded_bank(bank_path):
    root = require_inside(bank_path, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema_version") != SCHEMA or manifest.get("benchmark_split") != "development"
            or manifest.get("case_count") != 48 or manifest.get("primary_metric_eligible") is not False):
        raise ValueError("unexpected expanded diagnostic scope")
    path = root / "resolver.jsonl"
    if sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("expanded resolver hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not 96 <= len(rows) <= 480 or len(rows) != manifest["counts"]["reports"] or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("expanded bounded inventory mismatch")
    for row in rows:
        if set(row) != RESOLVER_FIELDS:
            raise ValueError("intervention information leaked into inference resolver")
        for kind in ("image", "report"):
            artifact = require_inside(row[f"{kind}_path"], root if kind == "report" else PROTECTED_ROOT, must_exist=True)
            if sha256_file(artifact) != row[f"{kind}_sha256"]:
                raise ValueError("expanded artifact hash mismatch")
        if Path(row["report_path"]).stat().st_size > 65536:
            raise ValueError("expanded report exceeds bound")
    return rows, {"manifest_sha256": sha256_file(root/"manifest.json"), "resolver_sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "stage"), required=True)
    for name in ("split-bank", "plan-run"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--cxr-run", action="append")
    parser.add_argument("--report-run", action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        if args.mode == "preregister":
            payload = preregister(args)
            files = [write_private_json(temporary / "case_plan.json", payload)]
            metadata = {"schema_version": PLAN_SCHEMA, "case_count": 48,
                        "report_text_read": False, "primary_metric_eligible": False}
            provenance = {}
        else:
            files, metadata, provenance = stage(args, temporary, target)
        write_private_json(temporary / "manifest.json", {**{k: v for k, v in metadata.items() if k != "coverage"},
            "run_id": args.run_id, "provenance": provenance, "program_sha256": sha256_file(__file__),
            "artifacts": {str(path.relative_to(temporary)): {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "prepared_development_only_polarity", "mode": args.mode,
        "counts": metadata.get("counts", {"cases": 48}), "manifest_sha256": sha256_file(target/"manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

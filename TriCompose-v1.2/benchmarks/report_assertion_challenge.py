#!/usr/bin/env python3
"""Fixed authored language stress test; NOT radiologist/held-out clinical gold.

No old report edits, patients, generator calls or threshold fitting. The
inference loader exposes only text/hash/opaque IDs. CPU-only modes may stage,
audit a frozen quote veto with oracle quotes, or analyze completed predictions.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from contracts import (
    PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, new_atomic_run,
    private_directory, read_json, require_inside, sha256_file,
    write_private_json, write_private_text,
)

BANK_SCHEMA = "tricompose-authored-report-assertion-challenge-v1"
AUDIT_SCHEMA = "tricompose-oracle-quote-veto-stress-audit-v1"
SCORE_SCHEMA = "tricompose-authored-report-assertion-predictions-v1"
ANALYSIS_SCHEMA = "tricompose-authored-report-assertion-analysis-v1"
FINDINGS = ("cardiomegaly", "consolidation", "pleural_effusion", "pneumothorax")
STATES = ("positive", "negative", "uncertain", "unknown")
FROZEN_GUARD_SHA = "7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18"


def text_hash(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def authored_cases():
    """Expected states are explicitly assigned, never inferred by a scorer."""
    terms = {"cardiomegaly": "cardiomegaly", "consolidation": "consolidation",
        "pleural_effusion": "pleural effusion", "pneumothorax": "pneumothorax"}
    other = {"cardiomegaly": "pneumothorax", "consolidation": "pleural_effusion",
        "pleural_effusion": "cardiomegaly", "pneumothorax": "consolidation"}
    qualified = {"cardiomegaly": "No severe cardiomegaly.", "consolidation": "No focal consolidation.",
        "pleural_effusion": "No large pleural effusion.", "pneumothorax": "No large pneumothorax."}
    templates = (
        ("literal_presence_control", "positive", "{term} is present."),
        ("postposed_absence", "negative", "{term} is absent."),
        ("no_evidence", "negative", "There is no evidence of {term}."),
        ("passive_not_detected", "negative", "{term} was not detected."),
        ("exclusion_verb", "negative", "This examination excludes {term}."),
        ("modal_may", "uncertain", "{term} may be present."),
        ("cannot_be_excluded", "uncertain", "A component of {term} cannot be excluded."),
        ("cannot_rule_out", "uncertain", "The study cannot rule out {term}."),
        ("qualified_absence", "uncertain", None),
        ("no_worsening_not_absence", "positive", "There is {term}, without interval worsening."),
        ("other_finding_negation_scope", "positive", "Although {other} is absent, {term} is demonstrated."),
        ("assessment_missing_not_absence", "unknown", "{other} is present. The presence of {term} is not assessed."),
    )
    rows = []
    for family, expected, template in templates:
        for finding in FINDINGS:
            text = qualified[finding] if template is None else template.format(term=terms[finding], other=terms[other[finding]])
            labels = dict.fromkeys(FINDINGS, "unknown")
            labels[finding] = expected
            if family == "other_finding_negation_scope":
                labels[other[finding]] = "negative"
            elif family == "assessment_missing_not_absence":
                labels[other[finding]] = "positive"
            rows.append({"family": family, "text": text[0].upper()+text[1:],
                "expected_states": labels, "evaluation_findings": [finding]})
    for text in (
        "The image is labeled with an acquisition date.",
        "No acute cardiopulmonary process is described.",
        "Findings were not provided.",
        "There is no comment about cardiomegaly, consolidation, pleural effusion, or pneumothorax.",
    ):
        rows.append({"family": "unmentioned_or_unassessed", "text": text,
            "expected_states": dict.fromkeys(FINDINGS, "unknown"), "evaluation_findings": list(FINDINGS)})
    mixed = (
        ("Cardiomegaly is present; consolidation is absent. Pleural effusion may be present. Pneumothorax is not discussed.",
         ("positive", "negative", "uncertain", "unknown")),
        ("There is no cardiomegaly, and consolidation is evident. Pleural effusion is not excluded. No pneumothorax.",
         ("negative", "positive", "uncertain", "negative")),
        ("No pleural effusion or pneumothorax is identified. Cardiomegaly cannot be ruled out. Consolidation is present.",
         ("uncertain", "positive", "negative", "negative")),
        ("Pleural effusion is present. Pleural effusion is absent. No pneumothorax.",
         ("unknown", "unknown", "uncertain", "negative")),
    )
    for text, states in mixed:
        rows.append({"family": "mixed_or_conflicting_assertions", "text": text,
            "expected_states": dict(zip(FINDINGS, states, strict=True)), "evaluation_findings": list(FINDINGS)})
    for index, row in enumerate(rows):
        row["item_id"] = f"fixture_{index:03d}"
        row["report_sha256"] = text_hash(row["text"])
    if (len(rows) != 56 or len({row["report_sha256"] for row in rows}) != 56
            or sum(len(row["evaluation_findings"]) for row in rows) != 80
            or any(not 3 <= len(row["text"]) <= 256 for row in rows)):
        raise ValueError("fixed authored inventory differs")
    return rows


def stage(root):
    rows = authored_cases()
    private_directory(root/"reports")
    resolver, references, files = [], [], []
    for row in rows:
        relative = f"reports/{row['item_id']}.txt"
        files.append(write_private_text(root/relative, row["text"]))
        resolver.append({"item_id": row["item_id"], "report_path": relative, "report_sha256": row["report_sha256"]})
        references.append({key: value for key, value in row.items() if key != "text"})
    summary = {"schema_version": BANK_SCHEMA, "reports": 56, "designated_finding_checks": 80,
        "full_vector_checks": 224, "families": dict(Counter(row["family"] for row in rows)),
        "authored_expected_state_counts": dict(Counter(row["expected_states"][name] for row in rows for name in row["evaluation_findings"])),
        "patient_data_used": False, "prior_generated_reports_used": False,
        "expert_reviewed": False, "heldout_clinical_evaluation": False, "primary_metric_eligible": False,
        "annotation_policy": "Explicit assertion presence/absence; modal or qualified absence is uncertain; unmentioned/unassessed is unknown; opposed assertions are uncertain. Author judgments, not clinical gold."}
    files.extend([write_private_text(root/"resolver.jsonl", "".join(json.dumps(row, sort_keys=True)+"\n" for row in resolver)),
        write_private_text(root/"references.jsonl", "".join(json.dumps(row, sort_keys=True)+"\n" for row in references)),
        write_private_json(root/"summary.json", summary)])
    return summary, files


def load_inputs(bank_run):
    """Inference must use this loader, not read the authored reference key."""
    root = require_inside(bank_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    if (manifest.get("schema_version") != BANK_SCHEMA or manifest.get("reports") != 56
            or manifest.get("patient_data_used") is not False
            or manifest.get("primary_metric_eligible") is not False):
        raise ValueError("not the fixed authored-only challenge")
    path = root/"resolver.jsonl"
    if sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("blinded resolver hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if (len(rows) != 56 or {row["item_id"] for row in rows} != {f"fixture_{i:03d}" for i in range(56)}
            or len({row["report_sha256"] for row in rows}) != 56):
        raise ValueError("fixed opaque/text inventory differs")
    texts = {}
    for row in rows:
        if set(row) != {"item_id", "report_path", "report_sha256"}:
            raise ValueError("reference information leaked into resolver")
        source = require_inside(root/row["report_path"], root/"reports", must_exist=True)
        if source.stat().st_size > 1024 or sha256_file(source) != row["report_sha256"]:
            raise ValueError("authored text size/hash differs")
        if manifest["artifacts"][row["report_path"]]["sha256"] != row["report_sha256"]:
            raise ValueError("source artifact binding differs")
        text = source.read_text(encoding="utf-8")
        if not 3 <= len(text) <= 256:
            raise ValueError("bounded authored text differs")
        texts[row["report_sha256"]] = text
    return rows, texts, {"bank_manifest_sha256": sha256_file(root/"manifest.json"), "resolver_sha256": sha256_file(path)}


def load_references(bank_run, resolver):
    root = require_inside(bank_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    path = root/"references.jsonl"
    if sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("authored reference hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    index = {row["item_id"]: row for row in rows}
    if len(index) != len(rows) or set(index) != {row["item_id"] for row in resolver}:
        raise ValueError("reference/resolver inventory differs")
    for source in resolver:
        row = index[source["item_id"]]
        if (row["report_sha256"] != source["report_sha256"] or set(row["expected_states"]) != set(FINDINGS)
                or any(value not in STATES for value in row["expected_states"].values())
                or not row["evaluation_findings"] or len(set(row["evaluation_findings"])) != len(row["evaluation_findings"])
                or not set(row["evaluation_findings"]) <= set(FINDINGS)):
            raise ValueError("invalid authored state/target inventory")
    return rows, path


def oracle_quote_audit(args):
    """Not extraction: inject a full-report oracle quote and test only the veto."""
    from repair_cached_report_evidence import scope_check
    if sha256_file(Path(scope_check.__code__.co_filename)) != FROZEN_GUARD_SHA:
        raise ValueError("pre-challenge veto code changed; no post-result fitting")
    resolver, texts, source = load_inputs(args.bank_run)
    references, key_path = load_references(args.bank_run, resolver)
    details = []
    for row in references:
        text = texts[row["report_sha256"]]
        span = {"quote": text, "char_start": 0, "char_end": len(text), "quote_sha256": text_hash(text)}
        for finding in row["evaluation_findings"]:
            expected = row["expected_states"][finding]
            for attributed in ("positive", "negative", "uncertain"):
                decision = scope_check(text, span, finding, attributed)
                unsafe = attributed in {"positive", "negative"} and attributed != expected
                details.append({"item_id": row["item_id"], "finding": finding, "family": row["family"],
                    "authored_expected_state": expected, "injected_model_polarity": attributed,
                    "valid_exact_polarity_probe": expected == attributed,
                    "unsafe_determinate_probe": unsafe, "veto": decision["veto"],
                    "veto_reason": decision["veto_reason"], "scope_reasons": decision["scope_reasons"],
                    "literal_mentions_checked": decision["literal_mentions_checked"]})
    valid = [row for row in details if row["valid_exact_polarity_probe"]]
    unsafe = [row for row in details if row["unsafe_determinate_probe"]]
    summary = {"schema_version": AUDIT_SCHEMA, "source": source, "model_calls": 0,
        "probe_count": len(details), "valid_exact_polarity_probes": len(valid),
        "valid_probes_not_vetoed": sum(not row["veto"] for row in valid),
        "unsafe_determinate_probes": len(unsafe), "unsafe_probes_vetoed": sum(row["veto"] for row in unsafe),
        "unsafe_probes_not_vetoed": sum(not row["veto"] for row in unsafe),
        "unchecked_literal_probes": sum(row["literal_mentions_checked"] == 0 for row in details),
        "frozen_guard_sha256": FROZEN_GUARD_SHA, "oracle_quote_selection": True,
        "independent_clinical_accuracy": None, "primary_metric_eligible": False, "selection_changed": False,
        "interpretation": "Conditional veto stress test with full-report oracle quotes; not a blind extractor, actual model errors, clinical accuracy or a calibrated false-repair rate. Full quotes may include multiple assertions."}
    return summary, details, {"authored_references": key_path}


def classification_counts(rows):
    matrix = {state: dict.fromkeys((*STATES, "unavailable"), 0) for state in STATES}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    per_class = {}
    for state in STATES:
        tp = matrix[state][state]
        support = sum(matrix[state].values())
        predicted = sum(matrix[other][state] for other in STATES)
        denom = support+predicted
        per_class[state] = {"support": support, "predicted": predicted, "tp": tp,
            "f1": None if denom == 0 else 2*tp/denom}
    f1 = [row["f1"] for row in per_class.values() if row["f1"] is not None]
    return {"checks": len(rows), "correct": sum(row["expected"] == row["predicted"] for row in rows),
        "accuracy": None if not rows else sum(row["expected"] == row["predicted"] for row in rows)/len(rows),
        "macro_f1_present_classes": None if not f1 else sum(f1)/len(f1), "per_class": per_class,
        "confusion_matrix": matrix, "unavailable": sum(row["predicted"] == "unavailable" for row in rows),
        "hard_positive_negative_flips": sum({row["expected"], row["predicted"]} == {"positive", "negative"} for row in rows),
        "unsafe_commit_on_unknown_or_uncertain": sum(row["expected"] in {"unknown", "uncertain"} and row["predicted"] in {"positive", "negative"} for row in rows)}


def evaluate_predictions(references, predictions):
    index = {row["item_id"]: row for row in predictions}
    if len(index) != len(predictions) or set(index) != {row["item_id"] for row in references}:
        raise ValueError("failed/missing predictions cannot silently leave denominator")
    full, targets = [], []
    for ref in references:
        pred = index[ref["item_id"]]
        if pred["report_sha256"] != ref["report_sha256"] or set(pred["finding_states"]) != set(FINDINGS):
            raise ValueError("prediction input/state inventory differs")
        for finding in FINDINGS:
            state = pred["finding_states"][finding]
            if state not in STATES or pred.get("status") not in {"complete", "failed_unavailable"}:
                raise ValueError("invalid prediction status/state")
            row = {"item_id": ref["item_id"], "finding": finding, "family": ref["family"],
                "expected": ref["expected_states"][finding],
                "predicted": state if pred["status"] == "complete" else "unavailable"}
            full.append(row)
            if finding in ref["evaluation_findings"]:
                targets.append(row)
    return {"designated_targets": classification_counts(targets), "full_vectors_secondary": classification_counts(full),
        "per_finding": {name: classification_counts([row for row in targets if row["finding"] == name]) for name in FINDINGS},
        "per_family": {name: classification_counts([row for row in targets if row["family"] == name]) for name in sorted({row["family"] for row in targets})}}, targets


def analyze(args):
    resolver, _, source = load_inputs(args.bank_run)
    root = require_inside(args.score_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    path = root/"predictions.json"
    if manifest.get("schema_version") != SCORE_SCHEMA or sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("prediction artifact binding differs")
    payload = read_json(path)
    if (payload["source"] != source or payload.get("model_received_reference_states") is not False
            or payload.get("scorer_read_reference_key") is not False or payload["producer"].get("frozen") is not True
            or payload.get("primary_metric_eligible") is not False):
        raise ValueError("blinded frozen prediction provenance differs")
    refs, key_path = load_references(args.bank_run, resolver)
    metrics, details = evaluate_predictions(refs, payload["records"])
    return {"schema_version": ANALYSIS_SCHEMA, "source": source, "model_calls_in_analysis": 0,
        **metrics, "producer": payload["producer"], "batch_replay": payload["batch_replay"],
        "independent_clinical_accuracy": None, "expert_reviewed": False, "primary_metric_eligible": False,
        "selection_changed": False, "targeted_repair_approved": False,
        "interpretation": "Authored language diagnostic only. Expected states are same-team judgments, not independent clinical annotation. Families/templates are correlated; do not claim patient-level accuracy or held-out validation."}, details, {"predictions": path, "authored_references": key_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage", "oracle-audit", "analyze"), required=True)
    for name in ("output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bank-run")
    parser.add_argument("--score-run")
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        if args.mode == "stage":
            summary, files = stage(temporary)
            sources = {}
            schema = BANK_SCHEMA
        else:
            summary, details, sources = oracle_quote_audit(args) if args.mode == "oracle-audit" else analyze(args)
            schema = summary["schema_version"]
            files = [write_private_json(temporary/"summary.json", summary), write_private_json(temporary/"details.json", {"records": details})]
        write_private_json(temporary/"manifest.json", {"schema_version": schema, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "reports": 56, "patient_data_used": False,
            "primary_metric_eligible": False, "frozen_guard_sha256": FROZEN_GUARD_SHA,
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "artifacts": {str(path.relative_to(temporary)): {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_authored_assertion_"+args.mode, "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

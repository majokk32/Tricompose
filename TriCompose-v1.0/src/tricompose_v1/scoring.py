"""Reference-free scoring contracts for a protected V1 candidate bank.

The learned scores produced by the GPU jobs are deliberately kept separate.
They are evidence for an exploratory selector, not calibrated clinical
probabilities.  This module contains only deterministic validation, finding
normalisation, quality gates, and state comparison.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_directory_mode,
    require_inside,
    require_private_file,
    sha256_file,
)

from .candidate_bank import CANDIDATE_BANK_SCHEMA, _load_run
from .contracts import (
    ACTIVE_REPORT_MODELS,
    validate_cxr_candidate,
    validate_ehr_candidate,
    validate_report_candidate,
)
from .execution import CXR_CANDIDATE_RUN_SCHEMA, REPORT_CANDIDATE_RUN_SCHEMA
from .facts import validate_ehr_facts
from .prompts import ACTIVE_PROMPT_MODELS


SCORE_INPUT_SCHEMA = "tricompose-v1-score-input-v1"
XRV_SCORE_SCHEMA = "tricompose-v1-xrv-score-bundle-v1"
BIOVIL_SCORE_SCHEMA = "tricompose-v1-biovil-score-bundle-v1"
QWENVL_SCORE_SCHEMA = "tricompose-v1-qwenvl-score-bundle-v1"
AGGREGATE_SCHEMA = "tricompose-v1-static-selection-v1"
CALIBRATION_STATUS = "uncalibrated_engineering_smoke"

# Canonical order is shared with the compact Qwen output and V1.1 evaluator.
CHEXPERT_FINDINGS = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
    "no_finding",
)
FINDING_STATES = frozenset({"positive", "negative", "uncertain", "unknown"})

# These mappings describe the exact V1 legacy prompt semantics.  CHF is
# expanded because the byte-aligned renderer explicitly writes cardiomegaly
# and edema; this is not claimed to be a direct EHR observation.
FACT_TO_PROMPT_FINDINGS: dict[str, tuple[str, ...]] = {
    "congestive_heart_failure": ("cardiomegaly", "edema"),
    "cardiomegaly": ("cardiomegaly",),
    "pleural_effusion": ("pleural_effusion",),
    "pulmonary_edema": ("edema",),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax",),
    "atelectasis": ("atelectasis",),
    "consolidation": ("consolidation",),
    "lung_opacity": ("lung_opacity",),
    "endotracheal_tube": ("support_devices",),
    "central_venous_catheter": ("support_devices",),
    "enteric_tube": ("support_devices",),
    "cardiac_pacemaker": ("support_devices",),
}

XRV_LABELS: dict[str, tuple[str, ...]] = {
    "atelectasis": ("atelectasis",),
    "cardiomegaly": ("cardiomegaly",),
    "consolidation": ("consolidation",),
    "edema": ("edema",),
    "enlarged_cardiomediastinum": ("enlarged_cardiomediastinum",),
    "fracture": ("fracture",),
    "lung_lesion": ("lung_lesion", "nodule", "mass"),
    "lung_opacity": ("lung_opacity", "infiltration"),
    "pleural_effusion": ("effusion",),
    "pleural_other": ("pleural_thickening",),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax",),
    # The selected XRV checkpoint does not expose a support-device label.
    "support_devices": (),
    "no_finding": (),
}

REPORT_PATTERNS: dict[str, tuple[str, ...]] = {
    "atelectasis": ("atelectasis", "atelectatic"),
    "cardiomegaly": ("cardiomegaly", "cardiac enlargement", "enlarged heart"),
    "consolidation": ("consolidation", "consolidative"),
    "edema": ("pulmonary edema", "interstitial edema", "vascular congestion"),
    "enlarged_cardiomediastinum": ("enlarged cardiomediastinal silhouette",),
    "fracture": ("fracture", "fractured"),
    "lung_lesion": ("lung lesion", "pulmonary nodule", "pulmonary mass"),
    "lung_opacity": ("lung opacity", "pulmonary opacity", "airspace opacity"),
    "pleural_effusion": ("pleural effusion", "pleural fluid"),
    "pleural_other": ("pleural thickening", "pleural abnormality"),
    "pneumonia": ("pneumonia", "infection"),
    "pneumothorax": ("pneumothorax",),
    "support_devices": (
        "endotracheal tube",
        "enteric tube",
        "feeding tube",
        "central venous catheter",
        "central venous line",
        "picc",
        "pacemaker",
        "pacer",
        "port-a-cath",
    ),
    "no_finding": (
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary process",
        "no acute disease",
    ),
}

NEGATION_MARKERS = (
    "no ",
    "without ",
    "negative for ",
    "absence of ",
    "free of ",
    "resolved ",
)
UNCERTAINTY_MARKERS = (
    "possible ",
    "possibly ",
    "probable ",
    "suspected ",
    "may represent ",
    "cannot exclude ",
    "questionable ",
)
TEMPORAL_PATTERN = re.compile(
    r"\b(?:compared with|comparison to|since (?:the )?prior|interval|unchanged|"
    r"stable|improved|worsened|new since|previous(?:ly)?|prior (?:study|exam))\b",
    flags=re.I,
)
MEASUREMENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", flags=re.I)


@dataclass(frozen=True)
class CandidateBankView:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    staging_root: Path
    ehr_candidates: tuple[dict[str, Any], ...]
    cxr_candidates: tuple[dict[str, Any], ...]
    report_candidates: tuple[dict[str, Any], ...]

    @property
    def cxr_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(row["candidate_id"]): row for row in self.cxr_candidates}


def _read_json(path: str | Path) -> dict[str, Any]:
    source = require_private_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protected JSON must contain an object")
    return payload


def load_candidate_bank(path: str | Path) -> CandidateBankView:
    """Load and revalidate a generation bank without opening artifact content."""

    root = require_inside(path, PROTECTED_ROOT, must_exist=True)
    manifest_path = require_private_file(root / "manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CANDIDATE_BANK_SCHEMA:
        raise ValueError("unsupported candidate-bank schema")
    if manifest.get("complete_generation_candidate_bank") is not True:
        raise ValueError("generation candidate bank is incomplete")

    staging = manifest.get("staging_run")
    if not isinstance(staging, dict):
        raise TypeError("candidate bank lacks staging provenance")
    staging_root = require_inside(staging.get("path", ""), PROTECTED_ROOT, must_exist=True)
    staging_manifest = require_private_file(staging_root / "run_manifest.json")
    if sha256_file(staging_manifest) != staging.get("manifest_sha256"):
        raise ValueError("candidate-bank staging manifest hash mismatch")

    ehr_rows = manifest.get("ehr_candidates")
    if not isinstance(ehr_rows, list) or not ehr_rows:
        raise ValueError("candidate bank has no EHR candidates")
    ehr_candidates: list[dict[str, Any]] = []
    for row in ehr_rows:
        if not isinstance(row, dict):
            raise TypeError("invalid EHR candidate")
        validate_ehr_candidate(row)
        ehr_candidates.append(row)

    cxr_runs = manifest.get("cxr_runs")
    report_runs = manifest.get("report_runs")
    if not isinstance(cxr_runs, list) or not isinstance(report_runs, list):
        raise TypeError("candidate bank run lists are missing")

    cxr_candidates: list[dict[str, Any]] = []
    cxr_models: set[str] = set()
    for ref in cxr_runs:
        if not isinstance(ref, dict):
            raise TypeError("invalid CXR run reference")
        run_root, _, rows = _load_run(
            ref.get("path", ""),
            schema=CXR_CANDIDATE_RUN_SCHEMA,
            validator=validate_cxr_candidate,
        )
        if sha256_file(run_root / "manifest.json") != ref.get("manifest_sha256"):
            raise ValueError("CXR run manifest hash mismatch")
        if len(rows) != ref.get("candidate_count"):
            raise ValueError("CXR run count mismatch")
        cxr_models.add(str(ref.get("model_id")))
        cxr_candidates.extend(rows)

    report_candidates: list[dict[str, Any]] = []
    report_models: set[str] = set()
    for ref in report_runs:
        if not isinstance(ref, dict):
            raise TypeError("invalid report run reference")
        run_root, _, rows = _load_run(
            ref.get("path", ""),
            schema=REPORT_CANDIDATE_RUN_SCHEMA,
            validator=validate_report_candidate,
        )
        if sha256_file(run_root / "manifest.json") != ref.get("manifest_sha256"):
            raise ValueError("report run manifest hash mismatch")
        if len(rows) != ref.get("candidate_count"):
            raise ValueError("report run count mismatch")
        report_models.add(str(ref.get("model_id")))
        report_candidates.extend(rows)

    if cxr_models != set(ACTIVE_PROMPT_MODELS):
        raise ValueError("score input lacks an active CXR model")
    if report_models != set(ACTIVE_REPORT_MODELS):
        raise ValueError("score input lacks an active report model")
    counts = manifest.get("counts")
    actual = {
        "ehr_candidates": len(ehr_candidates),
        "cxr_candidates": len(cxr_candidates),
        "report_candidates": len(report_candidates),
    }
    if counts != actual:
        raise ValueError("candidate-bank aggregate counts mismatch")

    cxr_by_id = {str(row["candidate_id"]): row for row in cxr_candidates}
    if len(cxr_by_id) != len(cxr_candidates):
        raise ValueError("duplicate CXR candidate ID")
    report_ids: set[str] = set()
    for report in report_candidates:
        report_id = str(report["candidate_id"])
        if report_id in report_ids:
            raise ValueError("duplicate report candidate ID")
        report_ids.add(report_id)
        cxr = cxr_by_id.get(str(report["parent_ids"][1]))
        if cxr is None or report["parent_ids"][0] != cxr["parent_ids"][0]:
            raise ValueError("report candidate lineage mismatch")

    return CandidateBankView(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        staging_root=staging_root,
        ehr_candidates=tuple(ehr_candidates),
        cxr_candidates=tuple(cxr_candidates),
        report_candidates=tuple(report_candidates),
    )


def new_atomic_protected_run(output_root: str | Path, run_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("run ID must be opaque and filesystem-safe")
    root = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    old_umask = os.umask(0o077)
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        enforce_private_directory_mode(root)
        target = require_inside(root / run_id, PROTECTED_ROOT, must_exist=False)
        if target.exists():
            raise FileExistsError("score run already exists")
        temp = root / f".{run_id}.{uuid.uuid4().hex}.tmp"
        temp.mkdir(mode=0o700)
        enforce_private_directory_mode(temp)
    finally:
        os.umask(old_umask)
    return temp, target


def commit_atomic_protected_run(temp: Path, target: Path) -> None:
    os.rename(temp, target)
    enforce_private_directory_mode(target)


def discard_atomic_protected_run(temp: Path) -> None:
    if temp.exists():
        shutil.rmtree(temp)


def load_ehr_facts(ehr_candidate: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = ehr_candidate.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("ehr_facts"), dict):
        raise TypeError("EHR candidate lacks facts artifact")
    ref = artifacts["ehr_facts"]
    path = require_private_file(ref.get("path", ""))
    if sha256_file(path) != ref.get("sha256"):
        raise ValueError("EHR facts hash mismatch")
    payload = _read_json(path)
    validate_ehr_facts(payload)
    return payload


def ehr_prompt_intent_states(facts_payload: Mapping[str, Any]) -> dict[str, str]:
    """Return only findings explicitly rendered into the V1 CXR prompt."""

    validate_ehr_facts(facts_payload)
    states = {label: "unknown" for label in CHEXPERT_FINDINGS}
    facts = facts_payload["facts"]
    for fact_id, labels in FACT_TO_PROMPT_FINDINGS.items():
        fact = facts.get(fact_id)
        if not isinstance(fact, dict) or fact.get("state") != "positive":
            continue
        for label in labels:
            states[label] = "positive"
    return states


def _mention_state(sentence: str, phrase: str) -> str:
    start = sentence.find(phrase)
    prefix = sentence[max(0, start - 48) : start]
    local = prefix + phrase
    if any(marker in local for marker in UNCERTAINTY_MARKERS):
        return "uncertain"
    if any(marker in local for marker in NEGATION_MARKERS):
        return "negative"
    return "positive"


def extract_report_finding_states(text: str) -> dict[str, str]:
    """Conservative lexical fallback until a frozen CheXbert weight is available."""

    sentences = [
        " ".join(row.lower().replace("-", " ").split())
        for row in re.split(r"(?<=[.!?;])\s+|\n+", text)
        if row.strip()
    ]
    result: dict[str, str] = {}
    for label in CHEXPERT_FINDINGS:
        mentions: list[str] = []
        for sentence in sentences:
            for phrase in REPORT_PATTERNS[label]:
                if phrase in sentence:
                    mentions.append(_mention_state(sentence, phrase))
        observed = set(mentions)
        if not observed:
            result[label] = "unknown"
        elif "positive" in observed and "negative" in observed:
            result[label] = "uncertain"
        elif "positive" in observed:
            result[label] = "positive"
        elif "uncertain" in observed:
            result[label] = "uncertain"
        else:
            result[label] = "negative"
    return result


def compare_finding_states(
    expected: Mapping[str, str], observed: Mapping[str, str]
) -> dict[str, Any]:
    """Compare only known expected labels; missing observations stay unknown."""

    per_finding: dict[str, str] = {}
    support: list[str] = []
    contradiction: list[str] = []
    unknown: list[str] = []
    for label in CHEXPERT_FINDINGS:
        left = expected.get(label, "unknown")
        right = observed.get(label, "unknown")
        if left not in FINDING_STATES or right not in FINDING_STATES:
            raise ValueError("invalid finding state")
        if left == "unknown" or right in {"unknown", "uncertain"}:
            relation = "unknown"
            unknown.append(label)
        elif left == "uncertain":
            relation = "unknown"
            unknown.append(label)
        elif left == right:
            relation = "support"
            support.append(label)
        else:
            relation = "contradiction"
            contradiction.append(label)
        per_finding[label] = relation
    comparable = len(support) + len(contradiction)
    score = len(support) / comparable if comparable else None
    return {
        "score": None if score is None else round(score, 8),
        "comparable_finding_count": comparable,
        "support_findings": support,
        "contradiction_findings": contradiction,
        "unknown_findings": unknown,
        "strong_contradiction_count": len(contradiction),
        "per_finding_relation": per_finding,
    }


def xrv_ehr_support(
    expected: Mapping[str, str], probabilities: Mapping[str, float]
) -> dict[str, Any]:
    per_finding: dict[str, Any] = {}
    values: list[float] = []
    unavailable: list[str] = []
    for label in CHEXPERT_FINDINGS:
        if expected.get(label) != "positive":
            continue
        aliases = XRV_LABELS[label]
        available = [float(probabilities[name]) for name in aliases if name in probabilities]
        if not available:
            unavailable.append(label)
            continue
        score = max(available)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("XRV probability is outside [0,1]")
        values.append(score)
        per_finding[label] = {
            "xrv_labels": [name for name in aliases if name in probabilities],
            "positive_support_probability": round(score, 8),
        }
    return {
        "score": round(sum(values) / len(values), 8) if values else None,
        "comparable_finding_count": len(values),
        "unavailable_findings": unavailable,
        "per_finding": per_finding,
        # No classifier threshold has been clinically calibrated for this
        # synthetic domain, so low probabilities are not hard contradictions.
        "strong_contradiction_count": 0,
    }


def deterministic_report_quality(text: str) -> dict[str, Any]:
    words = re.findall(r"\b\w+(?:[-']\w+)*\b", text)
    sentences = [
        " ".join(row.lower().split())
        for row in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if row.strip()
    ]
    duplicate_count = len(sentences) - len(set(sentences))
    empty = not bool(text.strip())
    too_short_hard = len(words) < 4
    repetition_hard = bool(sentences) and duplicate_count >= max(2, len(sentences) // 2)
    temporal = bool(TEMPORAL_PATTERN.search(text))
    exact_measurement = bool(MEASUREMENT_PATTERN.search(text))
    score = 1.0
    if empty or too_short_hard:
        score = 0.0
    else:
        if len(words) < 8:
            score -= 0.35
        elif len(words) < 20:
            score -= 0.15
        if temporal:
            score -= 0.15
        if exact_measurement:
            score -= 0.10
        score -= min(0.30, 0.10 * duplicate_count)
    hard_gate_pass = not (empty or too_short_hard or repetition_hard)
    return {
        "score": round(max(0.0, score), 8),
        "hard_gate_pass": hard_gate_pass,
        "flags": {
            "empty": empty,
            "word_count": len(words),
            "under_8_words": len(words) < 8,
            "unsupported_temporal_language": temporal,
            "exact_cm_mm_measurement": exact_measurement,
            "duplicate_sentence_count": duplicate_count,
            "severe_repetition": repetition_hard,
        },
    }


def validate_qwen_states(states: Sequence[Any]) -> tuple[str, ...]:
    if len(states) != len(CHEXPERT_FINDINGS):
        raise ValueError("Qwen finding-state vector has the wrong length")
    aliases = {
        "+": "positive",
        "-": "negative",
        "?": "unknown",
        "u": "uncertain",
        "positive": "positive",
        "negative": "negative",
        "unknown": "unknown",
        "uncertain": "uncertain",
    }
    normalized: list[str] = []
    for value in states:
        state = aliases.get(str(value).strip().lower())
        if state is None:
            raise ValueError("Qwen finding-state vector contains an invalid state")
        normalized.append(state)
    return tuple(normalized)


__all__ = [
    "AGGREGATE_SCHEMA",
    "BIOVIL_SCORE_SCHEMA",
    "CALIBRATION_STATUS",
    "CHEXPERT_FINDINGS",
    "CandidateBankView",
    "QWENVL_SCORE_SCHEMA",
    "XRV_SCORE_SCHEMA",
    "commit_atomic_protected_run",
    "compare_finding_states",
    "deterministic_report_quality",
    "discard_atomic_protected_run",
    "ehr_prompt_intent_states",
    "extract_report_finding_states",
    "load_candidate_bank",
    "load_ehr_facts",
    "new_atomic_protected_run",
    "validate_qwen_states",
    "xrv_ehr_support",
]

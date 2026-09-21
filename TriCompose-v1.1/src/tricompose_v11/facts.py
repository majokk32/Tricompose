"""Build V1.1 direct facts plus separately labelled clinical context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tricompose_v1.ehr_bridge import CANONICAL_SCHEMA, validate_canonical_ehr
from tricompose_v1.facts import extract_ehr_facts, validate_ehr_facts


FACT_SCHEMA_V11 = "tricompose-facts-v1.1"
FACT_EXTRACTOR_VERSION_V11 = "radiology_facts_v1_1_context_bridge_v2"

# V1 also tracks clinical indications such as chest pain. They remain useful
# provenance, but only the concepts below may become image conditions.
IMAGE_CONDITION_FACT_IDS = (
    "cardiomegaly",
    "pleural_effusion",
    "pulmonary_edema",
    "pneumonia",
    "pneumothorax",
    "atelectasis",
    "consolidation",
    "lung_opacity",
    "endotracheal_tube",
    "central_venous_catheter",
    "enteric_tube",
    "cardiac_pacemaker",
    "congestive_heart_failure",
)

# These concepts are context or indications, not asserted image findings.
# Patterns intentionally use audited diagnosis descriptions only.
CONTEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "chronic_obstructive_pulmonary_disease": (
        "chronic obstructive pulmonary disease",
        "chronic airway obstruction",
        "emphysema",
    ),
    "asthma": ("asthma",),
    "ischemic_heart_disease": (
        "coronary atherosclerosis",
        "atherosclerotic heart disease",
        "ischemic heart disease",
        "coronary artery disease",
        "angina pectoris",
    ),
    "myocardial_infarction": (
        "myocardial infarction",
        "nstemi",
        "stemi",
    ),
    "coronary_revascularization": (
        "coronary angioplasty",
        "coronary artery bypass",
        "cabg",
        "coronary stent",
    ),
    "cardiac_arrhythmia": (
        "atrial fibrillation",
        "atrial flutter",
        "cardiac dysrhythmia",
        "cardiac arrhythmia",
    ),
    "hypertensive_disease": (
        "hypertension",
        "hypertensive heart disease",
        "hypertensive chronic kidney disease",
    ),
    "smoking_history": (
        "nicotine dependence",
        "tobacco use",
        "history of tobacco",
        "history of nicotine",
    ),
    "obstructive_sleep_apnea": ("obstructive sleep apnea",),
    "obesity": (
        "obesity",
        "body mass index [bmi] 40",
    ),
    "acute_kidney_injury": (
        "acute kidney failure",
        "acute renal failure",
    ),
    "chronic_kidney_disease": (
        "chronic kidney disease",
        "renal failure, chronic",
    ),
    "sepsis": (
        "sepsis",
        "septicemia",
        "septic shock",
    ),
    "respiratory_viral_context": (
        "exposure to covid",
        "covid-19",
        "viral communicable diseases",
    ),
    "venous_thromboembolism_history": (
        "venous thrombosis",
        "pulmonary embol",
    ),
}

CONTEXT_LABELS = {
    "chronic_obstructive_pulmonary_disease": "chronic obstructive pulmonary disease",
    "asthma": "asthma",
    "ischemic_heart_disease": "ischemic heart disease",
    "myocardial_infarction": "myocardial infarction",
    "coronary_revascularization": "coronary revascularization history",
    "cardiac_arrhythmia": "cardiac arrhythmia",
    "hypertensive_disease": "hypertensive disease",
    "smoking_history": "smoking history",
    "obstructive_sleep_apnea": "obstructive sleep apnea",
    "obesity": "obesity",
    "acute_kidney_injury": "acute kidney injury",
    "chronic_kidney_disease": "chronic kidney disease",
    "sepsis": "sepsis",
    "respiratory_viral_context": "respiratory viral disease or exposure context",
    "venous_thromboembolism_history": "venous thromboembolism history",
}


def _normalized(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def _diagnosis_sources(canonical: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, event in enumerate(canonical["diagnoses"]):
        sources.append(
            {
                "text": event["description"] or event["token"],
                "evidence": f"diagnosis:{event['code'] or event['event_id']}",
                "source_field": f"diagnoses[{index}]",
                "visit_index": event["visit_index"],
            }
        )
    return sources


def _extract_contexts(canonical: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = _diagnosis_sources(canonical)
    contexts: dict[str, dict[str, Any]] = {}
    for context_id, patterns in CONTEXT_PATTERNS.items():
        matches = [
            source
            for source in sources
            if any(pattern in _normalized(source["text"]) for pattern in patterns)
        ]
        evidence: list[str] = []
        source_fields: list[str] = []
        source_visits: list[int] = []
        for source in matches:
            if source["evidence"] not in evidence:
                evidence.append(source["evidence"])
            if source["source_field"] not in source_fields:
                source_fields.append(source["source_field"])
            if source["visit_index"] not in source_visits:
                source_visits.append(source["visit_index"])
        contexts[context_id] = {
            "status": "documented" if matches else "unknown",
            "label": CONTEXT_LABELS[context_id],
            "role": "clinical_context_not_direct_radiographic_finding",
            "evidence": evidence,
            "source_fields": source_fields,
            "source_visits": sorted(source_visits),
        }
    return contexts


def extract_v11_facts(canonical: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse V1 direct facts and add grounded longitudinal context."""

    validate_canonical_ehr(canonical)
    if canonical.get("schema_version") != CANONICAL_SCHEMA:
        raise ValueError("V1.1 facts require a canonical TriCompose EHR")
    v1 = extract_ehr_facts(canonical)
    contexts = _extract_contexts(canonical)
    direct_positive = [
        fact_id
        for fact_id in IMAGE_CONDITION_FACT_IDS
        if v1["facts"][fact_id]["state"] == "positive"
    ]
    documented_contexts = [
        context_id
        for context_id, context in contexts.items()
        if context["status"] == "documented"
    ]
    if direct_positive and documented_contexts:
        tier = "direct_radiographic_plus_context"
    elif direct_positive:
        tier = "direct_radiographic_only"
    elif documented_contexts:
        tier = "clinical_context_only"
    else:
        tier = "neutral_fallback"
    payload = {
        "schema_version": FACT_SCHEMA_V11,
        "case_id": canonical["case_id"],
        "extractor_version": FACT_EXTRACTOR_VERSION_V11,
        "direct_fact_source_version": v1["extractor_version"],
        "direct_fact_scope": v1["fact_scope"],
        "context_scope": "all_complete_synthetic_visits_diagnoses_only",
        "missing_is_unknown": True,
        "patient_context": v1["patient_context"],
        "direct_facts": v1["facts"],
        "clinical_contexts": contexts,
        "summary": {
            "direct_positive_fact_count": len(direct_positive),
            "documented_context_count": len(documented_contexts),
            "conditioning_tier": tier,
            "underconditioned": tier == "neutral_fallback",
        },
    }
    validate_v11_facts(payload)
    return payload


def validate_v11_facts(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FACT_SCHEMA_V11:
        raise ValueError("unsupported V1.1 fact schema")
    if payload.get("missing_is_unknown") is not True:
        raise ValueError("V1.1 must preserve unknown semantics")
    v1_payload = {
        "schema_version": "tricompose-facts-v1",
        "case_id": payload.get("case_id"),
        "extractor_version": payload.get("direct_fact_source_version"),
        "fact_scope": payload.get("direct_fact_scope"),
        "missing_is_unknown": True,
        "patient_context": payload.get("patient_context"),
        "facts": payload.get("direct_facts"),
        "summary": {},
    }
    validate_ehr_facts(v1_payload)
    contexts = payload.get("clinical_contexts")
    if not isinstance(contexts, dict) or set(contexts) != set(CONTEXT_PATTERNS):
        raise ValueError("V1.1 context inventory is incomplete")
    for context_id, context in contexts.items():
        if context.get("status") not in {"documented", "unknown"}:
            raise ValueError(f"invalid context status: {context_id}")
        if context.get("label") != CONTEXT_LABELS[context_id]:
            raise ValueError(f"invalid context label: {context_id}")
        if context.get("role") != "clinical_context_not_direct_radiographic_finding":
            raise ValueError(f"context is not separated from findings: {context_id}")
        evidence = context.get("evidence")
        fields = context.get("source_fields")
        visits = context.get("source_visits")
        if not isinstance(evidence, list) or not isinstance(fields, list):
            raise TypeError(f"invalid context provenance: {context_id}")
        if not isinstance(visits, list) or any(not isinstance(v, int) for v in visits):
            raise TypeError(f"invalid context visits: {context_id}")
        if context["status"] == "unknown" and (evidence or fields or visits):
            raise ValueError(f"unknown context carries evidence: {context_id}")
        if context["status"] == "documented" and (not evidence or not fields):
            raise ValueError(f"documented context lacks evidence: {context_id}")
        if any(not field.startswith("diagnoses[") for field in fields):
            raise ValueError(f"context uses a non-diagnosis source: {context_id}")


__all__ = [
    "CONTEXT_LABELS",
    "CONTEXT_PATTERNS",
    "FACT_EXTRACTOR_VERSION_V11",
    "FACT_SCHEMA_V11",
    "IMAGE_CONDITION_FACT_IDS",
    "extract_v11_facts",
    "validate_v11_facts",
]

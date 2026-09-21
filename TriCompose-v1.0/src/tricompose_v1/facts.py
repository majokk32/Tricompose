"""Extract a conservative, evidence-grounded radiology fact contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .ehr_bridge import CANONICAL_SCHEMA, validate_canonical_ehr


FACT_SCHEMA = "tricompose-facts-v1"
FACT_EXTRACTOR_VERSION = "radiology_facts_v2_legacy_bridge"
FACT_STATES = frozenset({"positive", "negative", "uncertain", "unknown"})

# Patterns intentionally name only direct concepts. For example, heart failure is
# not silently expanded into cardiomegaly, edema, and effusion.
FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "congestive_heart_failure": ("congestive heart failure", "heart failure"),
    "chest_pain": ("chest pain",),
    "cardiomegaly": ("cardiomegaly", "cardiac enlargement", "enlarged heart"),
    "pleural_effusion": ("pleural effusion",),
    "pulmonary_edema": ("pulmonary edema", "interstitial edema"),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax",),
    "atelectasis": ("atelectasis", "atelectatic"),
    "consolidation": ("pulmonary consolidation", "lung consolidation"),
    "lung_opacity": ("lung opacity", "pulmonary opacity"),
    "endotracheal_tube": ("endotracheal tube", "endotracheal intubation"),
    "central_venous_catheter": (
        "central venous catheter",
        "central venous line",
        "picc line",
        "port-a-cath",
    ),
    "enteric_tube": ("enteric tube", "feeding tube", "nasogastric tube"),
    "cardiac_pacemaker": ("cardiac pacemaker", "cardiac pacer", "pacemaker"),
}

FACT_CATEGORIES = {
    "congestive_heart_failure": "clinical_indication",
    "chest_pain": "clinical_indication",
    "cardiomegaly": "direct_radiographic_finding",
    "pleural_effusion": "direct_radiographic_finding",
    "pulmonary_edema": "direct_radiographic_finding",
    "pneumonia": "direct_radiographic_finding",
    "pneumothorax": "direct_radiographic_finding",
    "atelectasis": "direct_radiographic_finding",
    "consolidation": "direct_radiographic_finding",
    "lung_opacity": "direct_radiographic_finding",
    "endotracheal_tube": "explicit_device",
    "central_venous_catheter": "explicit_device",
    "enteric_tube": "explicit_device",
    "cardiac_pacemaker": "explicit_device",
}

UNCERTAIN_MARKERS = (
    "possible",
    "possibly",
    "probable",
    "suspected",
    "concern for",
    "rule out",
    "cannot exclude",
)
NEGATIVE_PREFIXES = ("no ", "without ", "negative for ", "absence of ")


def _normalized(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def _polarity(text: str, patterns: Iterable[str]) -> str | None:
    normalized = _normalized(text)
    matched = [pattern for pattern in patterns if pattern in normalized]
    if not matched:
        return None
    for pattern in matched:
        if any(f"{prefix}{pattern}" in normalized for prefix in NEGATIVE_PREFIXES):
            return "negative"
    if any(marker in normalized for marker in UNCERTAIN_MARKERS):
        return "uncertain"
    return "positive"


def _latest_visit_sources(canonical: Mapping[str, Any]) -> list[dict[str, str]]:
    latest_index = len(canonical["timeline"]) - 1
    sources: list[dict[str, str]] = []
    singular = {
        "diagnoses": "diagnosis",
        "medications": "medication",
        "labs": "lab",
        "vitals": "vital",
    }
    for collection, label in singular.items():
        for index, event in enumerate(canonical[collection]):
            if event["visit_index"] != latest_index:
                continue
            sources.append(
                {
                    "text": event["description"] or event["token"],
                    "evidence": f"{label}:{event['code'] or event['event_id']}",
                    "source_field": f"{collection}[{index}]",
                }
            )
    for index, token in enumerate(canonical["timeline"][latest_index]["other_tokens"]):
        sources.append(
            {
                "text": token,
                "evidence": f"timeline:visit_{latest_index:04d}_other_{index:04d}",
                "source_field": f"timeline[{latest_index}].other_tokens[{index}]",
            }
        )
    return sources


def _resolve_fact(matches: list[tuple[str, dict[str, str]]]) -> dict[str, Any]:
    if not matches:
        return {"state": "unknown", "evidence": [], "source_fields": []}
    polarities = {polarity for polarity, _ in matches}
    if "positive" in polarities and "negative" in polarities:
        state = "uncertain"
    elif "positive" in polarities:
        state = "positive"
    elif "uncertain" in polarities:
        state = "uncertain"
    else:
        state = "negative"
    evidence: list[str] = []
    source_fields: list[str] = []
    for _, source in matches:
        if source["evidence"] not in evidence:
            evidence.append(source["evidence"])
        if source["source_field"] not in source_fields:
            source_fields.append(source["source_field"])
    return {
        "state": state,
        "evidence": evidence,
        "source_fields": source_fields,
    }


def extract_ehr_facts(canonical: Mapping[str, Any]) -> dict[str, Any]:
    """Extract direct facts from the latest synthetic visit only."""

    validate_canonical_ehr(canonical)
    if canonical.get("schema_version") != CANONICAL_SCHEMA:
        raise ValueError("facts require a canonical V1 EHR")
    sources = _latest_visit_sources(canonical)
    facts: dict[str, dict[str, Any]] = {}
    for fact_id, patterns in FACT_PATTERNS.items():
        matches: list[tuple[str, dict[str, str]]] = []
        for source in sources:
            polarity = _polarity(source["text"], patterns)
            if polarity is not None:
                matches.append((polarity, source))
        facts[fact_id] = {
            "category": FACT_CATEGORIES[fact_id],
            **_resolve_fact(matches),
        }
    non_unknown = [
        fact_id for fact_id, fact in facts.items() if fact["state"] != "unknown"
    ]
    payload = {
        "schema_version": FACT_SCHEMA,
        "case_id": canonical["case_id"],
        "extractor_version": FACT_EXTRACTOR_VERSION,
        "fact_scope": "latest_complete_synthetic_visit_only",
        "missing_is_unknown": True,
        # The previously successful EHR-prompt CXR experiments included this
        # demographic surface form in every model prompt.  Keep it separate
        # from clinical facts: it never changes finding polarity and it is
        # copied only from the canonical EHR contract.
        "patient_context": {
            "age_group": canonical["demographics"]["age_group"],
            "sex": canonical["demographics"]["sex"],
            "source_fields": [
                "demographics.age_group",
                "demographics.sex",
            ],
            "mapping_status": canonical["demographics"]["mapping_status"],
        },
        "facts": facts,
        "summary": {
            "clinical_indication_fact_count": sum(
                facts[fact_id]["category"] == "clinical_indication"
                for fact_id in non_unknown
            ),
            "direct_radiographic_fact_count": sum(
                facts[fact_id]["category"] == "direct_radiographic_finding"
                for fact_id in non_unknown
            ),
            "explicit_device_fact_count": sum(
                facts[fact_id]["category"] == "explicit_device"
                for fact_id in non_unknown
            ),
            "radiology_relevant_fact_count": len(non_unknown),
            "underconditioned": not non_unknown,
        },
    }
    validate_ehr_facts(payload)
    return payload


def validate_ehr_facts(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FACT_SCHEMA:
        raise ValueError("unsupported EHR fact schema")
    if payload.get("missing_is_unknown") is not True:
        raise ValueError("fact extraction must preserve unknown semantics")
    patient_context = payload.get("patient_context")
    if not isinstance(patient_context, dict):
        raise TypeError("EHR fact contract lacks patient context")
    if patient_context.get("age_group") not in {
        "young adult",
        "middle-aged adult",
        "older adult",
        "elderly adult",
        "adult",
    }:
        raise ValueError("invalid patient age group")
    if patient_context.get("sex") not in {
        "female",
        "male",
        "unspecified-sex",
    }:
        raise ValueError("invalid patient sex")
    if patient_context.get("source_fields") != [
        "demographics.age_group",
        "demographics.sex",
    ]:
        raise ValueError("patient context provenance is invalid")
    if not isinstance(patient_context.get("mapping_status"), str):
        raise TypeError("patient context mapping status is missing")
    facts = payload.get("facts")
    if not isinstance(facts, dict) or set(facts) != set(FACT_PATTERNS):
        raise ValueError("EHR fact inventory is incomplete")
    for fact_id, fact in facts.items():
        if not isinstance(fact, dict) or fact.get("state") not in FACT_STATES:
            raise ValueError(f"invalid state for fact: {fact_id}")
        if fact.get("category") != FACT_CATEGORIES[fact_id]:
            raise ValueError(f"invalid category for fact: {fact_id}")
        evidence = fact.get("evidence")
        source_fields = fact.get("source_fields")
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) or not item for item in evidence
        ):
            raise TypeError(f"invalid evidence for fact: {fact_id}")
        if not isinstance(source_fields, list) or any(
            not isinstance(item, str) or not item for item in source_fields
        ):
            raise TypeError(f"invalid source fields for fact: {fact_id}")
        if fact["state"] == "unknown" and (evidence or source_fields):
            raise ValueError(f"unknown fact cannot carry evidence: {fact_id}")
        if fact["state"] != "unknown" and (not evidence or not source_fields):
            raise ValueError(f"non-unknown fact lacks evidence: {fact_id}")


__all__ = [
    "FACT_EXTRACTOR_VERSION",
    "FACT_CATEGORIES",
    "FACT_PATTERNS",
    "FACT_SCHEMA",
    "FACT_STATES",
    "extract_ehr_facts",
    "validate_ehr_facts",
]

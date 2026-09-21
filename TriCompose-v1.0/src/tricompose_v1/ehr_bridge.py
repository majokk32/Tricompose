"""Canonicalize synthetic EHR outputs without inventing clinical values."""

from __future__ import annotations

from typing import Any, Mapping


CANONICAL_SCHEMA = "tricompose-ehr-v1"
CANONICALIZER_VERSION = "ehr_bridge_v1"
SYNEHRGY_SCHEMA = "tricompose.synehrgy_v2.synthetic_ehr.v1"
PROMPTEHR_SCHEMA = "tricompose.promptehr.synthetic_ehr.v1"


def _strings(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return list(value)


def _split_token(token: str) -> tuple[str, str]:
    if "_" not in token:
        return "", token.strip()
    code, description = token.split("_", 1)
    return code.strip(), description.strip()


def _event(
    *,
    prefix: str,
    index: int,
    visit_index: int,
    token: str,
) -> dict[str, Any]:
    code, description = _split_token(token)
    return {
        "event_id": f"{prefix}_{index:04d}",
        "visit_index": visit_index,
        "token": token,
        "code": code,
        "description": description,
    }


def _canonical_from_visits(
    *,
    case_id: str,
    source_model_id: str,
    source_schema: str,
    cold_start: bool,
    mimic_schema_family: str,
    source_visits: list[dict[str, Any]],
    time_tokens: list[str],
    demographics: list[str],
    strict_v1_eligible: bool,
) -> dict[str, Any]:
    diagnoses: list[dict[str, Any]] = []
    medications: list[dict[str, Any]] = []
    labs: list[dict[str, Any]] = []
    vitals: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []

    for visit_index, visit in enumerate(source_visits):
        diagnosis_ids: list[str] = []
        medication_ids: list[str] = []
        lab_ids: list[str] = []
        vital_ids: list[str] = []

        for token in _strings(visit.get("diagnoses"), "diagnoses"):
            item = _event(
                prefix="diagnosis",
                index=len(diagnoses),
                visit_index=visit_index,
                token=token,
            )
            diagnoses.append(item)
            diagnosis_ids.append(item["event_id"])
        for token in _strings(visit.get("medications"), "medications"):
            item = _event(
                prefix="medication",
                index=len(medications),
                visit_index=visit_index,
                token=token,
            )
            medications.append(item)
            medication_ids.append(item["event_id"])
        for token in _strings(visit.get("labs"), "labs"):
            item = _event(
                prefix="lab",
                index=len(labs),
                visit_index=visit_index,
                token=token,
            )
            labs.append(item)
            lab_ids.append(item["event_id"])
        for token in _strings(visit.get("vitals"), "vitals"):
            item = _event(
                prefix="vital",
                index=len(vitals),
                visit_index=visit_index,
                token=token,
            )
            vitals.append(item)
            vital_ids.append(item["event_id"])

        relative_time_tokens = []
        if visit_index > 0 and visit_index - 1 < len(time_tokens):
            relative_time_tokens.append(time_tokens[visit_index - 1])
        timeline.append(
            {
                "visit_index": visit_index,
                "relative_time_tokens": relative_time_tokens,
                "diagnosis_ids": diagnosis_ids,
                "medication_ids": medication_ids,
                "lab_ids": lab_ids,
                "vital_ids": vital_ids,
                "other_tokens": _strings(visit.get("other_tokens"), "other_tokens"),
            }
        )

    canonical = {
        "schema_version": CANONICAL_SCHEMA,
        "case_id": case_id,
        "source": {
            "model_id": source_model_id,
            "source_schema": source_schema,
            "cold_start": cold_start,
            "mimic_schema_family": mimic_schema_family,
        },
        "demographics": {
            "age_group": "adult",
            "sex": "unspecified-sex",
            "source_tokens": demographics,
            "mapping_status": "not_inferred_without_audited_mapping",
        },
        "diagnoses": diagnoses,
        "medications": medications,
        "labs": labs,
        "vitals": vitals,
        "timeline": timeline,
        "validation": {
            "strict_valid": True,
            "visit_count": len(timeline),
            "strict_v1_eligible": strict_v1_eligible,
            "source_tokens_preserved_without_dequantization": True,
        },
    }
    validate_canonical_ehr(canonical)
    return canonical


def canonicalize_synehrgy_case(
    case: Mapping[str, Any],
    *,
    source_model_id: str,
) -> dict[str, Any]:
    """Convert one valid SynEHRgy case into the V1 canonical contract."""

    if case.get("schema") != SYNEHRGY_SCHEMA:
        raise ValueError("unsupported SynEHRgy case schema")
    validation = case.get("validation")
    if not isinstance(validation, dict) or validation.get("strict_valid") is not True:
        raise ValueError("SynEHRgy case is not structurally valid")
    structure = case.get("structure")
    if not isinstance(structure, dict):
        raise TypeError("SynEHRgy structure is missing")
    raw_visits = structure.get("visits")
    if not isinstance(raw_visits, list) or not raw_visits:
        raise ValueError("SynEHRgy case has no visits")

    source_visits: list[dict[str, Any]] = []
    for raw_visit in raw_visits:
        if not isinstance(raw_visit, dict):
            raise TypeError("SynEHRgy visit must be an object")
        source_visits.append(
            {
                "diagnoses": _strings(raw_visit.get("problems"), "problems"),
                "medications": [],
                "labs": _strings(raw_visit.get("labs"), "labs"),
                "vitals": _strings(raw_visit.get("charts"), "charts"),
                "other_tokens": _strings(
                    raw_visit.get("other_tokens"), "other_tokens"
                ),
            }
        )
    top_level = _strings(structure.get("top_level_tokens"), "top_level_tokens")
    time_tokens = [token for token in top_level if token.startswith("<")]
    demographics = _strings(raw_visits[0].get("covariates"), "covariates")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("SynEHRgy case ID is missing")
    return _canonical_from_visits(
        case_id=case_id,
        source_model_id=source_model_id,
        source_schema=SYNEHRGY_SCHEMA,
        cold_start=True,
        mimic_schema_family="MIMIC-IV-format",
        source_visits=source_visits,
        time_tokens=time_tokens,
        demographics=demographics,
        strict_v1_eligible=True,
    )


def canonicalize_promptehr_case(
    case: Mapping[str, Any],
    *,
    source_model_id: str = "promptehr_mimic3_seeded",
) -> dict[str, Any]:
    """Normalize PromptEHR while retaining its incompatible provenance."""

    if case.get("schema") != PROMPTEHR_SCHEMA:
        raise ValueError("unsupported PromptEHR case schema")
    validation = case.get("validation")
    if not isinstance(validation, dict) or validation.get("strict_valid") is not True:
        raise ValueError("PromptEHR case is not structurally valid")
    raw_visits = case.get("event_codes")
    if not isinstance(raw_visits, list) or not raw_visits:
        raise ValueError("PromptEHR case has no visits")
    source_visits: list[dict[str, Any]] = []
    for raw_visit in raw_visits:
        if not isinstance(raw_visit, dict):
            raise TypeError("PromptEHR visit must be an object")
        source_visits.append(
            {
                "diagnoses": _strings(raw_visit.get("diag"), "diag"),
                "medications": _strings(raw_visit.get("med"), "med"),
                "labs": [],
                "vitals": [],
                "other_tokens": _strings(raw_visit.get("prod"), "prod"),
            }
        )
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("PromptEHR case ID is missing")
    return _canonical_from_visits(
        case_id=case_id,
        source_model_id=source_model_id,
        source_schema=PROMPTEHR_SCHEMA,
        cold_start=False,
        mimic_schema_family="MIMIC-III",
        source_visits=source_visits,
        time_tokens=[],
        demographics=[],
        strict_v1_eligible=False,
    )


def validate_canonical_ehr(case: Mapping[str, Any]) -> None:
    if case.get("schema_version") != CANONICAL_SCHEMA:
        raise ValueError("unsupported canonical EHR schema")
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise ValueError("canonical EHR case ID is missing")
    for field in ("diagnoses", "medications", "labs", "vitals", "timeline"):
        if not isinstance(case.get(field), list):
            raise TypeError(f"canonical EHR {field} must be a list")
    timeline = case["timeline"]
    if not timeline:
        raise ValueError("canonical EHR must contain at least one visit")
    for expected_index, visit in enumerate(timeline):
        if not isinstance(visit, dict) or visit.get("visit_index") != expected_index:
            raise ValueError("canonical EHR timeline is malformed or reordered")
    event_prefixes = {
        "diagnoses": "diagnosis",
        "medications": "medication",
        "labs": "lab",
        "vitals": "vital",
    }
    for field, prefix in event_prefixes.items():
        for index, event in enumerate(case[field]):
            if not isinstance(event, dict):
                raise TypeError(f"canonical EHR {field} entries must be objects")
            if event.get("event_id") != f"{prefix}_{index:04d}":
                raise ValueError(f"canonical EHR {field} IDs are malformed")
            if not isinstance(event.get("visit_index"), int):
                raise TypeError(f"canonical EHR {field} visit index is invalid")
            for key in ("token", "code", "description"):
                if not isinstance(event.get(key), str):
                    raise TypeError(f"canonical EHR {field} {key} is invalid")
    validation = case.get("validation")
    if not isinstance(validation, dict) or validation.get("strict_valid") is not True:
        raise ValueError("canonical EHR validation is missing")
    if validation.get("source_tokens_preserved_without_dequantization") is not True:
        raise ValueError("canonical EHR cannot dequantize source bins")


__all__ = [
    "CANONICAL_SCHEMA",
    "CANONICALIZER_VERSION",
    "PROMPTEHR_SCHEMA",
    "SYNEHRGY_SCHEMA",
    "canonicalize_promptehr_case",
    "canonicalize_synehrgy_case",
    "validate_canonical_ehr",
]

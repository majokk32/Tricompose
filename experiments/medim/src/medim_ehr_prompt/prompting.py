"""Deterministically serialize selected structured EHR fields into one prompt."""

from __future__ import annotations

import math
from typing import Any, Mapping


DIAGNOSIS_FIELDS = {
    "DX_PNEUMONIA": "pneumonia",
    "DX_PNEUMOTHORAX": "pneumothorax",
    "DX_CHF": "congestive heart failure",
    "DX_PLEURAL_EFFUSION": "pleural effusion",
    "DX_ATELECTASIS": "atelectasis",
}

DEVICE_FIELDS = {
    "INTUBATED": "endotracheal intubation",
    "VENTILATOR": "mechanical ventilation",
    "PACEMAKER": "cardiac pacemaker",
}

EHR_TO_CXR_TARGETS = {
    "pneumonia": ["pneumonia", "consolidation", "lung_opacity"],
    "pneumothorax": ["pneumothorax"],
    "congestive heart failure": ["edema", "cardiomegaly", "effusion"],
    "pleural effusion": ["effusion"],
    "atelectasis": ["atelectasis"],
}


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na"}


def _positive(value: Any) -> bool:
    if _missing(value):
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _number(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _age_group(value: Any) -> str:
    age = _number(value)
    if age is None:
        return "adult"
    if age < 30:
        return "young adult"
    if age < 50:
        return "middle-aged adult"
    if age < 70:
        return "older adult"
    return "elderly adult"


def _sex(value: Any) -> str:
    if _missing(value):
        return "unspecified-sex"
    normalized = str(value).strip().upper()
    if normalized in {"F", "FEMALE", "1", "1.0", "TRUE"}:
        return "female"
    if normalized in {"M", "MALE", "0", "0.0", "FALSE"}:
        return "male"
    return "unspecified-sex"


def _measurement_bins(row: Mapping[str, Any]) -> dict[str, str]:
    bins: dict[str, str] = {}

    wbc = _number(row.get("WBC"))
    if wbc is not None:
        bins["white blood cell count"] = "low" if wbc < 4 else "high" if wbc > 11 else "within reference range"

    bnp = _number(row.get("BNP"))
    if bnp is not None:
        bins["BNP"] = "within reference range" if bnp < 100 else "elevated" if bnp <= 400 else "markedly elevated"

    spo2 = _number(row.get("SPO2"))
    if spo2 is not None:
        bins["oxygen saturation"] = "low" if spo2 < 92 else "borderline" if spo2 < 96 else "within reference range"

    rate = _number(row.get("RESP_RATE"))
    if rate is not None:
        bins["respiratory rate"] = "low" if rate < 12 else "elevated" if rate > 20 else "within reference range"

    return bins


def extract_facts(
    row: Mapping[str, Any],
    *,
    include_measurements: bool = True,
) -> dict[str, Any]:
    diagnoses = [label for field, label in DIAGNOSIS_FIELDS.items() if _positive(row.get(field))]
    devices = [label for field, label in DEVICE_FIELDS.items() if _positive(row.get(field))]
    measurements = _measurement_bins(row) if include_measurements else {}

    targets: dict[str, list[str]] = {}
    for fact in diagnoses:
        targets[fact] = list(EHR_TO_CXR_TARGETS[fact])

    return {
        "age_group": _age_group(row.get("AGE")),
        "sex": _sex(row.get("GENDER")),
        "positive_diagnoses": diagnoses,
        "positive_support_devices": devices,
        "measurement_bins": measurements,
        "cxr_observable_targets": targets,
    }


def build_prompt(facts: Mapping[str, Any]) -> str:
    age_group = str(facts["age_group"])
    sex = str(facts["sex"])
    diagnoses = [str(item) for item in facts.get("positive_diagnoses", [])]
    devices = [str(item) for item in facts.get("positive_support_devices", [])]
    measurements = dict(facts.get("measurement_bins", {}))

    sentences = [
        "The image is a radiograph of the chest, showing the thoracic cavity structures.",
        "Portable frontal AP chest radiograph.",
        f"Clinical history: an {age_group} {sex} patient.",
    ]
    if diagnoses:
        sentences.append(
            "Clinical indications include concern for "
            + ", ".join(diagnoses)
            + "."
        )
    if devices:
        sentences.append(
            "Known support devices include " + ", ".join(devices) + "."
        )
    if measurements:
        measurement_text = ", ".join(f"{name} is {state}" for name, state in measurements.items())
        sentences.append(
            "Pre-imaging clinical context includes " + measurement_text + "."
        )
    if not diagnoses and not devices and not measurements:
        sentences.append("Limited structured clinical context is available.")

    return " ".join(sentences)


def serialize_ehr_prompt(
    row: Mapping[str, Any],
    *,
    include_measurements: bool = True,
) -> tuple[dict[str, Any], str]:
    facts = extract_facts(
        row,
        include_measurements=include_measurements,
    )
    return facts, build_prompt(facts)

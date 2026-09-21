"""Deterministically serialize EHR facts in RadEdit's training-text style.

RadEdit was conditioned on short MIMIC-CXR impressions or dot-separated
radiographic observations.  This adapter therefore includes only positive,
radiographically observable facts and supported devices.  It intentionally
omits labs, vitals, demographics and non-radiographic diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PROMPT_VERSION = "radedit.ehr_to_observation_text.v1"

DIAGNOSIS_PHRASES = {
    "pneumonia": "Pulmonary airspace opacity compatible with pneumonia.",
    "pneumothorax": "Pneumothorax.",
    "congestive heart failure": (
        "Cardiomegaly. Pulmonary vascular congestion. Interstitial pulmonary edema."
    ),
    "pleural effusion": "Pleural effusion.",
    "atelectasis": "Atelectatic pulmonary opacity.",
}

DEVICE_PHRASES = {
    "endotracheal intubation": "Endotracheal tube.",
    "cardiac pacemaker": "Cardiac pacemaker and leads.",
}


@dataclass(frozen=True)
class RadEditPrompt:
    text: str
    included_diagnoses: tuple[str, ...]
    included_devices: tuple[str, ...]
    omitted_devices: tuple[str, ...]
    underconditioned: bool


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return [str(item).strip().lower() for item in value if str(item).strip()]


def build_radedit_prompt(facts: Mapping[str, Any]) -> RadEditPrompt:
    diagnoses = _string_list(facts.get("positive_diagnoses"), "positive_diagnoses")
    devices = _string_list(
        facts.get("positive_support_devices"), "positive_support_devices"
    )
    included_diagnoses = tuple(
        diagnosis for diagnosis in DIAGNOSIS_PHRASES if diagnosis in diagnoses
    )
    included_devices = tuple(device for device in DEVICE_PHRASES if device in devices)
    omitted_devices = tuple(device for device in devices if device not in DEVICE_PHRASES)

    phrases = [DIAGNOSIS_PHRASES[item] for item in included_diagnoses]
    phrases.extend(DEVICE_PHRASES[item] for item in included_devices)
    underconditioned = not phrases
    if underconditioned:
        # Absence of a positive EHR fact is not evidence for a normal CXR.
        # Keep the input neutral rather than inserting a false normal label.
        phrases.append("Chest radiograph for clinical evaluation.")

    return RadEditPrompt(
        text=" ".join(phrases),
        included_diagnoses=included_diagnoses,
        included_devices=included_devices,
        omitted_devices=omitted_devices,
        underconditioned=underconditioned,
    )


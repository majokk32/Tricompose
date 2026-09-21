"""Deterministically map protected EHR facts to RoentGen-style text.

This is an explicit EHR -> radiology-style prompt -> CXR bridge.  It is not a
direct structured-EHR-conditioned image model.  Only positive, radiographically
observable diagnoses and devices are serialized.  Labs and vitals are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PROMPT_VERSION = "roentgen_v2.ehr_to_radiology_prompt.pa_findings.v2"

AGE_GROUPS = {
    "young adult": "Young adult",
    "middle-aged adult": "Middle-aged adult",
    "older adult": "Older adult",
    "elderly adult": "Elderly adult",
    "adult": "Adult",
}

SEXES = {
    "female": "female",
    "male": "male",
    "unspecified-sex": "patient",
}

DIAGNOSIS_SENTENCES = {
    "pneumonia": "Pulmonary airspace opacity compatible with pneumonia is present.",
    "pneumothorax": "Pneumothorax is present.",
    "congestive heart failure": (
        "Cardiomegaly with pulmonary vascular congestion and interstitial edema "
        "compatible with congestive heart failure is present."
    ),
    "pleural effusion": "Pleural effusion is present.",
    "atelectasis": "Atelectatic pulmonary opacity is present.",
}

DEVICE_SENTENCES = {
    "endotracheal intubation": "An endotracheal tube projects over the trachea.",
    "cardiac pacemaker": "A cardiac pacemaker and leads are present.",
}


@dataclass(frozen=True)
class PromptResult:
    text: str
    included_diagnoses: tuple[str, ...]
    included_devices: tuple[str, ...]
    omitted_devices: tuple[str, ...]


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return [str(item).strip().lower() for item in value if str(item).strip()]


def build_roentgen_prompt(facts: Mapping[str, Any]) -> PromptResult:
    """Build a concise prompt aligned with the official demographic+finding form."""

    age_group = str(facts.get("age_group", "adult")).strip().lower()
    sex = str(facts.get("sex", "unspecified-sex")).strip().lower()
    age_text = AGE_GROUPS.get(age_group, AGE_GROUPS["adult"])
    sex_text = SEXES.get(sex, SEXES["unspecified-sex"])

    if sex_text == "patient":
        demographic = f"{age_text} patient."
    else:
        demographic = f"{age_text} {sex_text} patient."

    diagnoses = _string_list(facts.get("positive_diagnoses"), "positive_diagnoses")
    devices = _string_list(
        facts.get("positive_support_devices"), "positive_support_devices"
    )

    included_diagnoses = tuple(
        diagnosis for diagnosis in DIAGNOSIS_SENTENCES if diagnosis in diagnoses
    )
    included_devices = tuple(
        device for device in DEVICE_SENTENCES if device in devices
    )
    omitted_devices = tuple(
        device for device in devices if device not in DEVICE_SENTENCES
    )

    # RoentGen-v2's released training configuration points to PA CXRs. Do not
    # introduce a portable/AP view token that conflicts with that distribution.
    sentences = [demographic, "PA chest radiograph."]
    if included_diagnoses or included_devices:
        sentences.append("Findings:")
        sentences.extend(DIAGNOSIS_SENTENCES[item] for item in included_diagnoses)
        sentences.extend(DEVICE_SENTENCES[item] for item in included_devices)
    else:
        # Missing positive structured facts do not justify inventing a normal CXR.
        sentences.append("Chest radiograph for clinical evaluation.")

    return PromptResult(
        text=" ".join(sentences),
        included_diagnoses=included_diagnoses,
        included_devices=included_devices,
        omitted_devices=omitted_devices,
    )

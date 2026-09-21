"""Reproduce the validated EHR-to-CXR prompt bridge used by prior experiments.

RoentGen-v2, CheXGenBench Sana, and CheXGenBench PixArt previously consumed
the same clinical text.  This module preserves that exact surface form while
binding every included diagnosis/device back to the V1 EHR-fact contract.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from .facts import validate_ehr_facts


ACTIVE_PROMPT_MODELS = (
    "roentgen_v2",
    "chexgenbench_sana",
    "chexgenbench_pixart",
)

# These are the exact versions recorded by the already-run experiment
# adapters.  Sana and PixArt intentionally use the same clinical text as
# RoentGen; their version names describe the wrappers, not different content.
CLINICAL_PROMPT_VERSION = "roentgen_v2.ehr_to_radiology_prompt.pa_findings.v2"
RENDERER_VERSIONS = {
    "roentgen_v2": CLINICAL_PROMPT_VERSION,
    "chexgenbench_sana": "chexgenbench_sana.ehr_radiology_text.v1",
    "chexgenbench_pixart": "chexgenbench_pixart.ehr_radiology_text.v1",
}

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

# Keep the order and text byte-for-byte aligned with
# experiments/roentgen_v2/.../prompting.py.
LEGACY_DIAGNOSIS_FACTS = {
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "congestive heart failure": "congestive_heart_failure",
    "pleural effusion": "pleural_effusion",
    "atelectasis": "atelectasis",
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

LEGACY_DEVICE_FACTS = {
    "endotracheal intubation": "endotracheal_tube",
    "cardiac pacemaker": "cardiac_pacemaker",
}

DEVICE_SENTENCES = {
    "endotracheal intubation": "An endotracheal tube projects over the trachea.",
    "cardiac pacemaker": "A cardiac pacemaker and leads are present.",
}

DERIVED_RULES = {
    "pneumonia": "pneumonia_to_airspace_opacity_prior_v1",
    "congestive_heart_failure": "congestive_heart_failure_to_cxr_prior_v1",
    "atelectasis": "atelectasis_to_opacity_prior_v1",
}

_BANNED_UNGROUNDED = re.compile(
    r"\b(?:AP|frontal|lateral|portable|left|right|bilateral|"
    r"mild|moderate|severe|trace|small|large|upper|lower|basilar|apical|"
    r"lobe|lobar)\b",
    flags=re.IGNORECASE,
)


def _positive(payload: Mapping[str, Any], fact_id: str) -> bool:
    return payload["facts"][fact_id]["state"] == "positive"


def _legacy_inputs(facts_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the V1 fact contract into the old renderer input signature."""

    context = facts_payload["patient_context"]
    diagnoses = [
        label
        for label, fact_id in LEGACY_DIAGNOSIS_FACTS.items()
        if _positive(facts_payload, fact_id)
    ]
    devices = [
        label
        for label, fact_id in LEGACY_DEVICE_FACTS.items()
        if _positive(facts_payload, fact_id)
    ]
    return {
        "age_group": context["age_group"],
        "sex": context["sex"],
        "positive_diagnoses": diagnoses,
        "positive_support_devices": devices,
    }


def _legacy_text(legacy: Mapping[str, Any]) -> str:
    """Exact local reproduction of the prior RoentGen/Sana/PixArt builder."""

    age_group = str(legacy.get("age_group", "adult")).strip().lower()
    sex = str(legacy.get("sex", "unspecified-sex")).strip().lower()
    age_text = AGE_GROUPS.get(age_group, AGE_GROUPS["adult"])
    sex_text = SEXES.get(sex, SEXES["unspecified-sex"])
    demographic = (
        f"{age_text} patient."
        if sex_text == "patient"
        else f"{age_text} {sex_text} patient."
    )

    diagnoses = [str(value).strip().lower() for value in legacy["positive_diagnoses"]]
    devices = [str(value).strip().lower() for value in legacy["positive_support_devices"]]
    included_diagnoses = [
        diagnosis for diagnosis in DIAGNOSIS_SENTENCES if diagnosis in diagnoses
    ]
    included_devices = [device for device in DEVICE_SENTENCES if device in devices]

    sentences = [demographic, "PA chest radiograph."]
    if included_diagnoses or included_devices:
        sentences.append("Findings:")
        sentences.extend(DIAGNOSIS_SENTENCES[item] for item in included_diagnoses)
        sentences.extend(DEVICE_SENTENCES[item] for item in included_devices)
    else:
        sentences.append("Chest radiograph for clinical evaluation.")
    return " ".join(sentences)


def render_prompt(facts_payload: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    """Return the prior experiment's exact model input plus V1 provenance."""

    validate_ehr_facts(facts_payload)
    if model_id not in ACTIVE_PROMPT_MODELS:
        raise ValueError(f"inactive or unsupported prompt model: {model_id}")
    legacy = _legacy_inputs(facts_payload)
    included_fact_ids = [
        LEGACY_DIAGNOSIS_FACTS[label] for label in legacy["positive_diagnoses"]
    ] + [LEGACY_DEVICE_FACTS[label] for label in legacy["positive_support_devices"]]
    derived_rule_ids = [
        DERIVED_RULES[fact_id]
        for fact_id in included_fact_ids
        if fact_id in DERIVED_RULES
    ]
    text = _legacy_text(legacy)
    rendering = {
        "model_id": model_id,
        "renderer_version": RENDERER_VERSIONS[model_id],
        "clinical_prompt_version": CLINICAL_PROMPT_VERSION,
        "legacy_experiment_reproduction": True,
        "included_fact_ids": included_fact_ids,
        "derived_rule_ids": derived_rule_ids,
        "text": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "is_final_model_input": True,
        "underconditioned": not included_fact_ids,
    }
    validate_rendered_prompt(rendering, facts_payload)
    return rendering


def validate_rendered_prompt(
    rendering: Mapping[str, Any], facts_payload: Mapping[str, Any]
) -> None:
    validate_ehr_facts(facts_payload)
    model_id = rendering.get("model_id")
    if model_id not in ACTIVE_PROMPT_MODELS:
        raise ValueError("prompt uses an inactive model")
    expected_legacy = _legacy_inputs(facts_payload)
    expected_text = _legacy_text(expected_legacy)
    expected_included = [
        LEGACY_DIAGNOSIS_FACTS[label]
        for label in expected_legacy["positive_diagnoses"]
    ] + [
        LEGACY_DEVICE_FACTS[label]
        for label in expected_legacy["positive_support_devices"]
    ]
    expected_rules = [
        DERIVED_RULES[fact_id]
        for fact_id in expected_included
        if fact_id in DERIVED_RULES
    ]

    if rendering.get("text") != expected_text:
        raise ValueError("prompt is not the exact legacy experiment rendering")
    if rendering.get("included_fact_ids") != expected_included:
        raise ValueError("prompt fact lineage differs from the legacy bridge")
    if rendering.get("derived_rule_ids") != expected_rules:
        raise ValueError("prompt disease-prior provenance is invalid")
    if rendering.get("renderer_version") != RENDERER_VERSIONS[model_id]:
        raise ValueError("prompt renderer version is invalid")
    if rendering.get("clinical_prompt_version") != CLINICAL_PROMPT_VERSION:
        raise ValueError("shared clinical prompt version is invalid")
    if rendering.get("legacy_experiment_reproduction") is not True:
        raise ValueError("legacy experiment reproduction flag is missing")
    if rendering.get("is_final_model_input") is not True:
        raise ValueError("prompt is not marked as final model input")
    if rendering.get("underconditioned") is not (not expected_included):
        raise ValueError("prompt underconditioning flag is invalid")
    digest = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    if rendering.get("prompt_sha256") != digest:
        raise ValueError("prompt SHA256 does not match final text")

    # The old validated baseline intentionally fixes PA because the released
    # RoentGen training configuration points to PA CXRs.  No other view,
    # laterality, location, or severity language may be introduced.
    if expected_text.count("PA chest radiograph.") != 1:
        raise ValueError("legacy fixed-view phrase is missing or duplicated")
    if _BANNED_UNGROUNDED.search(expected_text):
        raise ValueError("prompt contains a non-legacy unsupported attribute")
    if expected_text.lower().count("findings:") > 1:
        raise ValueError("model prompt contains a duplicate Findings prefix")


__all__ = [
    "ACTIVE_PROMPT_MODELS",
    "CLINICAL_PROMPT_VERSION",
    "RENDERER_VERSIONS",
    "render_prompt",
    "validate_rendered_prompt",
]

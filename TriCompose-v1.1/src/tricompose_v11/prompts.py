"""Render one V1.1 clinical intent into model-specific prompt surfaces."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .facts import CONTEXT_LABELS, IMAGE_CONDITION_FACT_IDS, validate_v11_facts


ACTIVE_PROMPT_MODELS_V11 = (
    "roentgen_v2",
    "chexgenbench_sana",
    "chexgenbench_pixart",
)
CLINICAL_INTENT_VERSION_V11 = "tricompose_v1_1_clinical_intent_v2"
RENDERER_VERSIONS_V11 = {
    "roentgen_v2": "roentgen_v2.ehr_context_prompt.v1_1_3",
    "chexgenbench_sana": "chexgenbench_sana.ehr_context_prompt.v1_1",
    "chexgenbench_pixart": "chexgenbench_pixart.ehr_context_prompt.v1_1",
}

AGE_GROUPS = {
    "young adult": "young adult",
    "middle-aged adult": "middle-aged adult",
    "older adult": "older adult",
    "elderly adult": "elderly adult",
    "adult": "adult",
}
SEXES = {
    "female": "female",
    "male": "male",
    "unspecified-sex": "patient",
}

# Only positive direct facts may enter these image-finding sentences. Clinical
# context is rendered elsewhere and never promoted into this mapping.
DIRECT_SENTENCES = {
    "cardiomegaly": "Cardiomegaly is present.",
    "pleural_effusion": "Pleural effusion is present.",
    "pulmonary_edema": "Pulmonary edema is present.",
    "pneumonia": "Pulmonary airspace opacity compatible with pneumonia is present.",
    "pneumothorax": "Pneumothorax is present.",
    "atelectasis": "Atelectatic pulmonary opacity is present.",
    "consolidation": "Pulmonary consolidation is present.",
    "lung_opacity": "Pulmonary opacity is present.",
    "endotracheal_tube": "An endotracheal tube is present.",
    "central_venous_catheter": "A central venous catheter is present.",
    "enteric_tube": "An enteric tube is present.",
    "cardiac_pacemaker": "A cardiac pacemaker and leads are present.",
    "congestive_heart_failure": (
        "Cardiomegaly with pulmonary vascular congestion and interstitial edema "
        "compatible with congestive heart failure is present."
    ),
}

ROENTGEN_FINDING_PHRASES = {
    "cardiomegaly": "cardiomegaly",
    "pleural_effusion": "pleural effusion",
    "pulmonary_edema": "pulmonary edema",
    "pneumonia": "airspace opacity compatible with pneumonia",
    "pneumothorax": "pneumothorax",
    "atelectasis": "atelectatic pulmonary opacity",
    "consolidation": "pulmonary consolidation",
    "lung_opacity": "pulmonary opacity",
    "endotracheal_tube": "endotracheal tube",
    "central_venous_catheter": "central venous catheter",
    "enteric_tube": "enteric tube",
    "cardiac_pacemaker": "cardiac pacemaker and leads",
    "congestive_heart_failure": (
        "cardiomegaly, vascular congestion, and interstitial edema compatible "
        "with congestive heart failure"
    ),
}

if tuple(DIRECT_SENTENCES) != IMAGE_CONDITION_FACT_IDS:
    raise RuntimeError("V1.1 image-condition inventory and renderer diverged")
if tuple(ROENTGEN_FINDING_PHRASES) != IMAGE_CONDITION_FACT_IDS:
    raise RuntimeError("V1.1 RoentGen finding inventory and renderer diverged")

DERIVED_RULES_V11 = {
    "congestive_heart_failure": "congestive_heart_failure_to_cxr_prior_v1",
    "pneumonia": "pneumonia_to_airspace_opacity_prior_v1",
    "atelectasis": "atelectasis_to_opacity_prior_v1",
}

_BANNED_UNGROUNDED = re.compile(
    r"\b(?:AP|frontal|lateral|portable|left|right|bilateral|mild|moderate|"
    r"severe|trace|small|large|upper|lower|basilar|apical|lobe|lobar)\b",
    flags=re.IGNORECASE,
)
ROENTGEN_MAX_RENDERED_CONTEXTS = 4


def _patient_surface(payload: Mapping[str, Any]) -> str:
    context = payload["patient_context"]
    age = AGE_GROUPS[context["age_group"]]
    sex = SEXES[context["sex"]]
    return f"{age} patient" if sex == "patient" else f"{age} {sex} patient"


def _intent(payload: Mapping[str, Any]) -> dict[str, Any]:
    direct = [
        fact_id
        for fact_id in DIRECT_SENTENCES
        if payload["direct_facts"][fact_id]["state"] == "positive"
    ]
    contexts = [
        context_id
        for context_id in CONTEXT_LABELS
        if payload["clinical_contexts"][context_id]["status"] == "documented"
    ]
    return {
        "version": CLINICAL_INTENT_VERSION_V11,
        "patient_context": {
            "age_group": payload["patient_context"]["age_group"],
            "sex": payload["patient_context"]["sex"],
        },
        "generation_protocol": {"view": "PA", "source": "fixed_v1_protocol"},
        "direct_positive_fact_ids": direct,
        "documented_clinical_context_ids": contexts,
        "derived_rule_ids": [
            DERIVED_RULES_V11[fact_id]
            for fact_id in direct
            if fact_id in DERIVED_RULES_V11
        ],
    }


def _intent_sha256(intent: Mapping[str, Any]) -> str:
    canonical = json.dumps(intent, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _joined_labels(context_ids: list[str]) -> str:
    labels = [CONTEXT_LABELS[context_id] for context_id in context_ids]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _rendered_context_ids(intent: Mapping[str, Any], model_id: str) -> list[str]:
    contexts = list(intent["documented_clinical_context_ids"])
    if model_id == "roentgen_v2":
        return contexts[:ROENTGEN_MAX_RENDERED_CONTEXTS]
    return contexts


def _render_text(payload: Mapping[str, Any], model_id: str) -> str:
    intent = _intent(payload)
    patient = _patient_surface(payload)
    contexts = _rendered_context_ids(intent, model_id)
    direct = intent["direct_positive_fact_ids"]
    if model_id == "roentgen_v2":
        sentences = [f"{patient.capitalize()}.", "PA chest radiograph."]
        if contexts:
            sentences.append(f"Clinical context: {_joined_labels(contexts)}.")
        if direct:
            findings = "; ".join(
                ROENTGEN_FINDING_PHRASES[fact_id] for fact_id in direct
            )
            sentences.append(f"Findings: {findings}.")
        elif contexts:
            sentences.append(
                "Clinical indication only; radiographic findings are unspecified."
            )
        else:
            sentences.append("Clinical context and radiographic findings are unspecified.")
        return " ".join(sentences)

    sentences = [f"PA chest radiograph of a {patient}."]
    if contexts:
        sentences.append(
            "Structured-EHR clinical context: " + _joined_labels(contexts) + "."
        )
    if direct:
        sentences.append("Radiographic findings:")
        sentences.extend(DIRECT_SENTENCES[fact_id] for fact_id in direct)
    elif contexts:
        sentences.append("Radiographic findings are not prespecified by the EHR.")
    else:
        sentences.append(
            "Structured-EHR clinical context and radiographic findings are unspecified."
        )
    return " ".join(sentences)


def render_v11_prompt(payload: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    validate_v11_facts(payload)
    if model_id not in ACTIVE_PROMPT_MODELS_V11:
        raise ValueError(f"unsupported V1.1 prompt model: {model_id}")
    intent = _intent(payload)
    available_contexts = list(intent["documented_clinical_context_ids"])
    included_contexts = _rendered_context_ids(intent, model_id)
    text = _render_text(payload, model_id)
    rendering = {
        "model_id": model_id,
        "renderer_version": RENDERER_VERSIONS_V11[model_id],
        "clinical_intent_version": CLINICAL_INTENT_VERSION_V11,
        "clinical_intent_sha256": _intent_sha256(intent),
        "included_direct_fact_ids": intent["direct_positive_fact_ids"],
        "available_context_ids": available_contexts,
        "included_context_ids": included_contexts,
        "omitted_context_ids": [
            context_id
            for context_id in available_contexts
            if context_id not in included_contexts
        ],
        "context_budget_policy": (
            f"first_{ROENTGEN_MAX_RENDERED_CONTEXTS}_ordered_contexts"
            if model_id == "roentgen_v2"
            else "all_ordered_contexts"
        ),
        "derived_rule_ids": intent["derived_rule_ids"],
        "context_is_not_a_radiographic_assertion": True,
        "conditioning_tier": payload["summary"]["conditioning_tier"],
        "underconditioned": payload["summary"]["underconditioned"],
        "text": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "is_final_model_input": True,
    }
    validate_v11_prompt(rendering, payload)
    return rendering


def validate_v11_prompt(
    rendering: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    validate_v11_facts(payload)
    model_id = rendering.get("model_id")
    if model_id not in ACTIVE_PROMPT_MODELS_V11:
        raise ValueError("prompt uses an unsupported V1.1 model")
    intent = _intent(payload)
    expected_text = _render_text(payload, model_id)
    if rendering.get("text") != expected_text:
        raise ValueError("V1.1 prompt differs from deterministic rendering")
    if rendering.get("clinical_intent_sha256") != _intent_sha256(intent):
        raise ValueError("V1.1 clinical-intent hash is invalid")
    if rendering.get("included_direct_fact_ids") != intent["direct_positive_fact_ids"]:
        raise ValueError("V1.1 direct fact lineage is invalid")
    expected_contexts = _rendered_context_ids(intent, model_id)
    if rendering.get("available_context_ids") != intent["documented_clinical_context_ids"]:
        raise ValueError("V1.1 available context lineage is invalid")
    if rendering.get("included_context_ids") != expected_contexts:
        raise ValueError("V1.1 context lineage is invalid")
    if rendering.get("omitted_context_ids") != [
        context_id
        for context_id in intent["documented_clinical_context_ids"]
        if context_id not in expected_contexts
    ]:
        raise ValueError("V1.1 omitted context lineage is invalid")
    if rendering.get("derived_rule_ids") != intent["derived_rule_ids"]:
        raise ValueError("V1.1 derived-rule lineage is invalid")
    if rendering.get("context_is_not_a_radiographic_assertion") is not True:
        raise ValueError("V1.1 context/finding separation flag is missing")
    if rendering.get("renderer_version") != RENDERER_VERSIONS_V11[model_id]:
        raise ValueError("V1.1 renderer version is invalid")
    if rendering.get("clinical_intent_version") != CLINICAL_INTENT_VERSION_V11:
        raise ValueError("V1.1 clinical intent version is invalid")
    if rendering.get("is_final_model_input") is not True:
        raise ValueError("V1.1 prompt is not marked as final input")
    if rendering.get("underconditioned") is not payload["summary"]["underconditioned"]:
        raise ValueError("V1.1 underconditioning flag is invalid")
    digest = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    if rendering.get("prompt_sha256") != digest:
        raise ValueError("V1.1 prompt hash is invalid")
    if payload["case_id"] in expected_text:
        raise ValueError("case ID leaked into V1.1 prompt")
    if _BANNED_UNGROUNDED.search(expected_text):
        raise ValueError("V1.1 prompt contains unsupported image attributes")
    if expected_text.lower().count("findings:") > 1:
        raise ValueError("V1.1 prompt contains duplicate findings prefixes")
    for fact_id in intent["direct_positive_fact_ids"]:
        if payload["direct_facts"][fact_id]["state"] != "positive":
            raise ValueError("non-positive direct fact entered V1.1 prompt")
    for context_id in intent["documented_clinical_context_ids"]:
        context = payload["clinical_contexts"][context_id]
        if context["status"] != "documented" or not context["source_fields"]:
            raise ValueError("ungrounded context entered V1.1 prompt")


__all__ = [
    "ACTIVE_PROMPT_MODELS_V11",
    "CLINICAL_INTENT_VERSION_V11",
    "DIRECT_SENTENCES",
    "RENDERER_VERSIONS_V11",
    "ROENTGEN_MAX_RENDERED_CONTEXTS",
    "render_v11_prompt",
    "validate_v11_prompt",
]

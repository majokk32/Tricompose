"""Build model-specific prompts from the shared radiology-style text bridge.

The clinical serialization is deliberately imported from the RoentGen-v2
experiment.  This makes all three text-conditioned generators consume the
same demographic/finding content and prevents prompt drift between baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tricompose_roentgen_v2.prompting import (
    PROMPT_VERSION as CLINICAL_PROMPT_VERSION,
)
from tricompose_roentgen_v2.prompting import build_roentgen_prompt


MODEL_PROMPT_VERSIONS = {
    "unidisc": "unidisc.ehr_radiology_text.t2i.v1",
    "liquid": "liquid.ehr_radiology_text.t2i.v1",
}
LIQUID_GENERATION_SUFFIX = " Generate an image based on this description.<boi>"


@dataclass(frozen=True)
class ModelPrompt:
    text: str
    clinical_text: str
    clinical_prompt_version: str
    model_prompt_version: str
    included_diagnoses: tuple[str, ...]
    included_devices: tuple[str, ...]
    omitted_devices: tuple[str, ...]


def build_model_prompt(model_name: str, facts: Mapping[str, Any]) -> ModelPrompt:
    """Serialize the same clinical facts using each model's official wrapper."""

    if model_name not in MODEL_PROMPT_VERSIONS:
        raise ValueError("unsupported prompt target")
    clinical = build_roentgen_prompt(facts)
    if model_name == "unidisc":
        text = (
            "chest x-ray radiograph, grayscale medical image. "
            f"{clinical.text} <image>"
        )
    else:
        text = clinical.text + LIQUID_GENERATION_SUFFIX
    return ModelPrompt(
        text=text,
        clinical_text=clinical.text,
        clinical_prompt_version=CLINICAL_PROMPT_VERSION,
        model_prompt_version=MODEL_PROMPT_VERSIONS[model_name],
        included_diagnoses=clinical.included_diagnoses,
        included_devices=clinical.included_devices,
        omitted_devices=clinical.omitted_devices,
    )

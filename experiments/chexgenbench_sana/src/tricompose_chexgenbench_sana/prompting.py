"""Use the same deterministic clinical text as the RoentGen-v2 comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tricompose_roentgen_v2.prompting import (
    PROMPT_VERSION as CLINICAL_PROMPT_VERSION,
)
from tricompose_roentgen_v2.prompting import build_roentgen_prompt


PROMPT_VERSION = "chexgenbench_sana.ehr_radiology_text.v1"


@dataclass(frozen=True)
class SanaPrompt:
    text: str
    included_diagnoses: tuple[str, ...]
    included_devices: tuple[str, ...]
    omitted_devices: tuple[str, ...]


def build_sana_prompt(facts: Mapping[str, Any]) -> SanaPrompt:
    clinical = build_roentgen_prompt(facts)
    return SanaPrompt(
        text=clinical.text,
        included_diagnoses=clinical.included_diagnoses,
        included_devices=clinical.included_devices,
        omitted_devices=clinical.omitted_devices,
    )


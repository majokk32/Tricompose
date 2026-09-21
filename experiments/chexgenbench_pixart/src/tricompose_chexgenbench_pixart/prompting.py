"""Use the same deterministic clinical text as other text-to-CXR baselines."""

from __future__ import annotations

from tricompose_roentgen_v2.prompting import (
    PROMPT_VERSION as CLINICAL_PROMPT_VERSION,
)
from tricompose_roentgen_v2.prompting import build_roentgen_prompt


PROMPT_VERSION = "chexgenbench_pixart.ehr_radiology_text.v1"


def build_pixart_prompt(facts):
    return build_roentgen_prompt(facts)


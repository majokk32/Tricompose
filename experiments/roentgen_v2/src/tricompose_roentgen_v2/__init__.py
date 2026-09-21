"""Privacy-conscious frozen RoentGen-v2 inference adapter."""

from .prompting import PROMPT_VERSION, build_roentgen_prompt

__all__ = ["PROMPT_VERSION", "build_roentgen_prompt"]

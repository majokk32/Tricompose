"""Format-only revision after V2 job 12677127; old artifacts stay immutable."""
from opacity_assertion_stages_v2 import (
    STATES, MAX_CHARACTERS, MAX_SEGMENTS, POLARITY_PROMPT, digest, segments,
    validate_inventory, polarity_messages, unique_object, response_object,
    validate_ids, decode_locator, decode_polarity, reduce_states, evidence,
    _message, LOCATOR_PROMPT as PREVIOUS_LOCATOR_PROMPT,
)

VERSION = 'opacity_assertion_stages_v3_format_only'
LOCATOR_PROMPT = PREVIOUS_LOCATOR_PROMPT.replace(
    'If no segment discusses opacity, return [].',
    'If no segment discusses opacity, return {"segment_ids": []}.',
).replace(
    'Use only the supplied segment IDs. No quotes, states, explanations or extra keys.',
    'Your entire response must be the object with segment_ids, never a bare array.\n'
    'Use only the supplied segment IDs. No quotes, states, explanations or extra keys.',
)


def locator_messages(text, inventory):
    return _message(LOCATOR_PROMPT, text, inventory)

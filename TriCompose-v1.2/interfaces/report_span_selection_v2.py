"""Prompt-layer clarification; unchanged V1 source inventory and strict decoder.

This is a separate development protocol, not a retrospective reinterpretation
of flat V1 responses. No clinical words, bounds or polarities are inferred here.
"""
import importlib.util
import json
from pathlib import Path

V1_PATH = Path(__file__).with_name('report_span_selection.py')
spec = importlib.util.spec_from_file_location('tricompose_span_inventory_decoder_v1', V1_PATH)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

VERSION = 'report-source-span-selection-v2-explicit-json-object'
FINDINGS = v1.FINDINGS
POLARITIES = v1.POLARITIES
MAX_REPORT_CHARS = v1.MAX_REPORT_CHARS
MAX_SPANS = v1.MAX_SPANS
MAX_SPAN_CHARS = v1.MAX_SPAN_CHARS
SpanContractError = v1.SpanContractError
digest = v1.digest
build_inventory = v1.build_inventory
validate_inventory = v1.validate_inventory
decode_response = v1.decode_response
unknown_result = v1.unknown_result

JSON_TEMPLATE = {name: {polarity: [] for polarity in POLARITIES} for name in FINDINGS}
_V1_FORMAT = ('Return only JSON with exactly four finding keys: cardiomegaly, consolidation,\n'
    'pleural_effusion, pneumothorax. Each value has exactly positive, negative and\n'
    'uncertain keys. Each value is an array of zero, one or two existing span IDs.\n')
_V2_FORMAT = ('Return ONE JSON OBJECT. The top-level object has exactly four finding keys:\n'
    'cardiomegaly, consolidation, pleural_effusion, pneumothorax.\n'
    'The VALUE OF EACH FINDING KEY must be a JSON OBJECT, NOT an array.\n'
    'That finding object must contain ALL THREE polarity keys: positive, negative,\n'
    'uncertain. Only the VALUES OF THOSE POLARITY KEYS are arrays of span IDs.\n'
    'Every polarity array contains zero, one or two existing span IDs.\n'
    'Keep all twelve polarity keys, including keys whose values are empty [].\n'
    'Do not return finding-to-array mappings and do not omit empty fields.\n'
    'The complete required output shape is below. Empty arrays are structural\n'
    'placeholders, NOT clinical answers: fill only supported source IDs.\n'
    '<required_json_shape>\n{required_json_shape}\n</required_json_shape>\n')
if v1.PROMPT.count(_V1_FORMAT) != 1:
    raise RuntimeError('frozen_v1_format_instruction_changed')
PROMPT = v1.PROMPT.replace(_V1_FORMAT, _V2_FORMAT)


def request_messages(report, inventory):
    validate_inventory(report, inventory)
    numbered = [{'span_id': span['span_id'], 'text': span['text']} for span in inventory['spans']]
    prompt = PROMPT.format(report=report,
        inventory=json.dumps(numbered, ensure_ascii=True, separators=(',', ':')),
        required_json_shape=json.dumps(JSON_TEMPLATE, separators=(',', ':')))
    return [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]

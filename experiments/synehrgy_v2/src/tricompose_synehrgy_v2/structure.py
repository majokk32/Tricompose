"""Parse generated SynEHRgy token streams without patient-derived mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SECTION_PAIRS = {
    "<covars>": ("</covars>", "covariates"),
    "<problems>": ("</problems>", "problems"),
    "<labs>": ("</labs>", "labs"),
    "<charts>": ("</charts>", "charts"),
}
CLOSERS = {close: open_ for open_, (close, _) in SECTION_PAIRS.items()}


@dataclass(frozen=True)
class ParsedSequence:
    structure: dict[str, Any]
    validation: dict[str, Any]


def _empty_visit() -> dict[str, list[str]]:
    return {
        "covariates": [],
        "problems": [],
        "labs": [],
        "charts": [],
        "other_tokens": [],
    }


def parse_token_sequence(tokens: list[str]) -> ParsedSequence:
    """Group a synthetic sequence by visit and clinical section.

    Numerical values remain as the model's public bin tokens until the gated
    MIMIC-IV vocabulary and quantile mappings are available.
    """

    errors: list[str] = []
    visits: list[dict[str, list[str]]] = []
    top_level_tokens: list[str] = []
    current_visit: dict[str, list[str]] | None = None
    active_section: tuple[str, str] | None = None

    for token in tokens:
        if token in {"<s>", "</s>"}:
            continue
        if token == "<v>":
            if current_visit is not None:
                errors.append("nested_visit")
            current_visit = _empty_visit()
            active_section = None
            continue
        if token == "</v>":
            if current_visit is None:
                errors.append("visit_close_without_open")
            else:
                if active_section is not None:
                    errors.append("section_unclosed_at_visit_end")
                    active_section = None
                visits.append(current_visit)
                current_visit = None
            continue

        if token in SECTION_PAIRS:
            if current_visit is None:
                errors.append("section_outside_visit")
                continue
            if active_section is not None:
                errors.append("nested_section")
            close, name = SECTION_PAIRS[token]
            active_section = (close, name)
            continue
        if token in CLOSERS:
            if active_section is None or active_section[0] != token:
                errors.append("section_close_mismatch")
            active_section = None
            continue

        if current_visit is None:
            top_level_tokens.append(token)
        elif active_section is None:
            current_visit["other_tokens"].append(token)
        else:
            current_visit[active_section[1]].append(token)

    if current_visit is not None:
        errors.append("visit_unclosed")
        visits.append(current_visit)
    if active_section is not None:
        errors.append("section_unclosed")

    pair_balance = {
        SECTION_PAIRS[open_][1]: tokens.count(open_) == tokens.count(close)
        for open_, (close, _) in SECTION_PAIRS.items()
    }
    visit_balance = tokens.count("<v>") == tokens.count("</v>")
    starts_with_bos = bool(tokens) and tokens[0] == "<s>"
    ended_with_eos = bool(tokens) and tokens[-1] == "</s>"
    forbidden_tokens = [token for token in ("<pad>", "[UNK]") if token in tokens]
    strict_valid = (
        starts_with_bos
        and ended_with_eos
        and bool(visits)
        and visit_balance
        and all(pair_balance.values())
        and not errors
        and not forbidden_tokens
    )
    structure = {
        "top_level_tokens": top_level_tokens,
        "visits": visits,
    }
    validation = {
        "starts_with_bos": starts_with_bos,
        "ended_with_eos": ended_with_eos,
        "visit_count": len(visits),
        "visit_balance": visit_balance,
        "section_balance": pair_balance,
        "forbidden_tokens": forbidden_tokens,
        "parser_errors": errors,
        "strict_valid": strict_valid,
    }
    return ParsedSequence(structure=structure, validation=validation)


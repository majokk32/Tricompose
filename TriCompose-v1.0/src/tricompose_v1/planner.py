"""Build an auditable exhaustive version-1 call plan without running models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .registry import ModelRegistry, ModelSpec


PLAN_SCHEMA = "tricompose.execution_plan.v1"


@dataclass(frozen=True)
class PlannedAction:
    action_id: str
    stage: str
    model_id: str
    candidate_id: str
    parent_candidate_ids: tuple[str, ...]
    seed: int | None
    relative_cost_units: float


def _validate_seeds(values: Iterable[int], label: str) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError(f"{label} must contain at least one seed")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError(f"{label} must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} must not contain duplicates")
    return seeds


def _select_ehr_models(
    registry: ModelRegistry,
    *,
    use_ehr_prompt: bool,
) -> tuple[ModelSpec, ...]:
    if not use_ehr_prompt:
        return registry.active("ehr")
    model = registry.get("promptehr_mimic3_seeded")
    if (
        model.stage != "ehr"
        or model.status != "operational_incompatible"
        or model.implementation == "unavailable"
    ):
        raise ValueError("PromptEHR conditional route is not available")
    return (model,)


def build_exhaustive_plan(
    registry: ModelRegistry,
    *,
    ehr_seeds: Iterable[int] = (42,),
    cxr_seeds: Iterable[int] = (0,),
    use_ehr_prompt: bool = False,
) -> dict[str, object]:
    """Expand every active generator while preserving exact dependencies."""

    ehr_seed_values = _validate_seeds(ehr_seeds, "EHR seeds")
    cxr_seed_values = _validate_seeds(cxr_seeds, "CXR seeds")
    ehr_models = _select_ehr_models(
        registry,
        use_ehr_prompt=use_ehr_prompt,
    )
    actions: list[PlannedAction] = []
    ehr_candidates: list[str] = []
    cxr_candidates: list[tuple[str, str]] = []
    report_candidates: list[tuple[str, str, str]] = []

    for ehr_model in ehr_models:
        for seed in ehr_seed_values:
            candidate_id = f"ehr{len(ehr_candidates):03d}"
            ehr_candidates.append(candidate_id)
            actions.append(
                PlannedAction(
                    action_id=f"a{len(actions):05d}",
                    stage="generate_ehr",
                    model_id=ehr_model.model_id,
                    candidate_id=candidate_id,
                    parent_candidate_ids=(),
                    seed=seed,
                    relative_cost_units=ehr_model.relative_cost_units,
                )
            )

    for ehr_candidate in ehr_candidates:
        for cxr_model in registry.active("cxr"):
            for seed in cxr_seed_values:
                candidate_id = f"cxr{len(cxr_candidates):04d}"
                cxr_candidates.append((candidate_id, ehr_candidate))
                actions.append(
                    PlannedAction(
                        action_id=f"a{len(actions):05d}",
                        stage="generate_cxr",
                        model_id=cxr_model.model_id,
                        candidate_id=candidate_id,
                        parent_candidate_ids=(ehr_candidate,),
                        seed=seed,
                        relative_cost_units=cxr_model.relative_cost_units,
                    )
                )

    for cxr_candidate, ehr_candidate in cxr_candidates:
        for report_model in registry.active("report"):
            candidate_id = f"report{len(report_candidates):05d}"
            report_candidates.append((candidate_id, ehr_candidate, cxr_candidate))
            actions.append(
                PlannedAction(
                    action_id=f"a{len(actions):05d}",
                    stage="generate_report",
                    model_id=report_model.model_id,
                    candidate_id=candidate_id,
                    parent_candidate_ids=(ehr_candidate, cxr_candidate),
                    seed=None,
                    relative_cost_units=report_model.relative_cost_units,
                )
            )

    verifier_by_role = {
        "ehr_quality": "artifact_quality_gate",
        "cxr_quality": "artifact_quality_gate",
        "report_quality": "report_quality_rules",
        "ehr_cxr": "xrv_ehr_cxr",
        "cxr_report": "qwen25vl_cxr_report",
        "ehr_report": "deterministic_ehr_report",
    }
    for model_id in verifier_by_role.values():
        spec = registry.get(model_id)
        if not spec.enabled or spec.stage != "verifier":
            raise ValueError("required verifier is not active")

    for ehr_candidate in ehr_candidates:
        model = registry.get(verifier_by_role["ehr_quality"])
        actions.append(
            PlannedAction(
                action_id=f"a{len(actions):05d}",
                stage="verify_ehr_quality",
                model_id=model.model_id,
                candidate_id=f"score{len(actions):05d}",
                parent_candidate_ids=(ehr_candidate,),
                seed=None,
                relative_cost_units=model.relative_cost_units,
            )
        )
    for cxr_candidate, ehr_candidate in cxr_candidates:
        for role, parents in (
            ("cxr_quality", (cxr_candidate,)),
            ("ehr_cxr", (ehr_candidate, cxr_candidate)),
        ):
            model = registry.get(verifier_by_role[role])
            actions.append(
                PlannedAction(
                    action_id=f"a{len(actions):05d}",
                    stage=f"verify_{role}",
                    model_id=model.model_id,
                    candidate_id=f"score{len(actions):05d}",
                    parent_candidate_ids=parents,
                    seed=None,
                    relative_cost_units=model.relative_cost_units,
                )
            )
    for report_candidate, ehr_candidate, cxr_candidate in report_candidates:
        for role, parents in (
            ("report_quality", (report_candidate,)),
            ("cxr_report", (cxr_candidate, report_candidate)),
            ("ehr_report", (ehr_candidate, report_candidate)),
        ):
            model = registry.get(verifier_by_role[role])
            actions.append(
                PlannedAction(
                    action_id=f"a{len(actions):05d}",
                    stage=f"verify_{role}",
                    model_id=model.model_id,
                    candidate_id=f"score{len(actions):05d}",
                    parent_candidate_ids=parents,
                    seed=None,
                    relative_cost_units=model.relative_cost_units,
                )
            )

    return {
        "schema_version": PLAN_SCHEMA,
        "mode": "exhaustive_static_v1",
        "fully_synthetic": True,
        "uses_real_patient_input": False,
        "ehr_selection": {
            "user_selects_model": False,
            "use_prompt": use_ehr_prompt,
            "routing_mode": (
                "prompt_conditioned_promptehr"
                if use_ehr_prompt
                else "unconditional_cold_start_candidate_pool"
            ),
            "model_ids": [model.model_id for model in ehr_models],
            "strict_cold_start": all(model.cold_start for model in ehr_models),
            "prompt_contract": (
                "partial_structured_synthetic_ehr_seed"
                if use_ehr_prompt
                else "none"
            ),
            "exploratory_incompatible_model_ids": [
                model.model_id for model in ehr_models if not model.enabled
            ],
        },
        "ehr_seeds": list(ehr_seed_values),
        "cxr_seeds": list(cxr_seed_values),
        "counts": {
            "ehr_candidates": len(ehr_candidates),
            "cxr_candidates": len(cxr_candidates),
            "report_candidates": len(report_candidates),
            "generator_calls": len(ehr_candidates)
            + len(cxr_candidates)
            + len(report_candidates),
            "total_actions": len(actions),
        },
        "estimated_relative_cost_units": round(
            sum(action.relative_cost_units for action in actions), 4
        ),
        "actions": [
            {
                **asdict(action),
                "parent_candidate_ids": list(action.parent_candidate_ids),
            }
            for action in actions
        ],
    }


__all__ = ["PLAN_SCHEMA", "PlannedAction", "build_exhaustive_plan"]

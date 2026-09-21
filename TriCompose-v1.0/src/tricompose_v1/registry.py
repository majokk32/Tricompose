"""Validated model registry for the TriCompose version-1 candidate graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REGISTRY_SCHEMA = "tricompose.model_registry.v1"
STAGES = frozenset({"ehr", "cxr", "report", "verifier"})


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    stage: str
    enabled: bool
    status: str
    frozen: bool
    input_signature: str
    output_signature: str
    implementation: str
    artifact_namespace: str
    cold_start: bool
    requires_previous_cxr: bool
    direct_structured_ehr_to_cxr: bool
    relative_cost_units: float
    priority: int
    exclusion_reason: str | None


def _required_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid model registry field: {field}")
    return value


def _parse_model(row: Any) -> ModelSpec:
    if not isinstance(row, dict):
        raise TypeError("model registry entries must be objects")
    stage = _required_string(row, "stage")
    if stage not in STAGES:
        raise ValueError("unsupported model stage")
    for field in (
        "enabled",
        "frozen",
        "cold_start",
        "requires_previous_cxr",
        "direct_structured_ehr_to_cxr",
    ):
        if not isinstance(row.get(field), bool):
            raise TypeError(f"model registry field must be boolean: {field}")
    cost = row.get("relative_cost_units")
    priority = row.get("priority")
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or float(cost) < 0
    ):
        raise ValueError("relative model cost must be non-negative")
    if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
        raise ValueError("model priority must be a non-negative integer")
    exclusion = row.get("exclusion_reason")
    if exclusion is not None and (
        not isinstance(exclusion, str) or not exclusion.strip()
    ):
        raise ValueError("exclusion reason must be null or a non-empty string")
    spec = ModelSpec(
        model_id=_required_string(row, "model_id"),
        stage=stage,
        enabled=row["enabled"],
        status=_required_string(row, "status"),
        frozen=row["frozen"],
        input_signature=_required_string(row, "input_signature"),
        output_signature=_required_string(row, "output_signature"),
        implementation=_required_string(row, "implementation"),
        artifact_namespace=_required_string(row, "artifact_namespace"),
        cold_start=row["cold_start"],
        requires_previous_cxr=row["requires_previous_cxr"],
        direct_structured_ehr_to_cxr=row["direct_structured_ehr_to_cxr"],
        relative_cost_units=float(cost),
        priority=priority,
        exclusion_reason=exclusion,
    )
    if spec.enabled and not spec.frozen:
        raise ValueError("version-1 active models must be frozen")
    if spec.enabled and spec.exclusion_reason is not None:
        raise ValueError("enabled model cannot have an exclusion reason")
    if not spec.enabled and spec.exclusion_reason is None:
        raise ValueError("disabled model must explain its exclusion")
    if spec.enabled and spec.stage == "ehr" and not spec.cold_start:
        raise ValueError("active version-1 EHR generators must be cold-start")
    if spec.enabled and spec.stage == "cxr" and spec.requires_previous_cxr:
        raise ValueError("active version-1 CXR generators cannot require a prior CXR")
    if spec.stage == "cxr" and spec.direct_structured_ehr_to_cxr:
        raise ValueError("no registered CXR model supports direct structured EHR input")
    return spec


class ModelRegistry:
    def __init__(self, models: Iterable[ModelSpec]) -> None:
        indexed: dict[str, ModelSpec] = {}
        for model in models:
            if model.model_id in indexed:
                raise ValueError("duplicate model ID")
            indexed[model.model_id] = model
        if not indexed:
            raise ValueError("model registry is empty")
        self._models = indexed
        for stage in ("ehr", "cxr", "report"):
            if not self.active(stage):
                raise ValueError(f"model registry has no active {stage} generator")

    @classmethod
    def from_path(cls, path: str | Path) -> "ModelRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("model registry must be an object")
        if payload.get("schema_version") != REGISTRY_SCHEMA:
            raise ValueError("unsupported model registry schema")
        rows = payload.get("models")
        if not isinstance(rows, list):
            raise TypeError("model registry has no model list")
        return cls(_parse_model(row) for row in rows)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown model ID: {model_id}") from exc

    def active(self, stage: str) -> tuple[ModelSpec, ...]:
        if stage not in STAGES:
            raise ValueError("unsupported model stage")
        return tuple(
            sorted(
                (
                    model
                    for model in self._models.values()
                    if model.enabled and model.stage == stage
                ),
                key=lambda model: (model.priority, model.model_id),
            )
        )

    def disabled(self) -> tuple[ModelSpec, ...]:
        return tuple(
            sorted(
                (model for model in self._models.values() if not model.enabled),
                key=lambda model: (model.stage, model.priority, model.model_id),
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": REGISTRY_SCHEMA,
            "active_counts": {
                stage: len(self.active(stage))
                for stage in ("ehr", "cxr", "report", "verifier")
            },
            "active_model_ids": {
                stage: [model.model_id for model in self.active(stage)]
                for stage in ("ehr", "cxr", "report", "verifier")
            },
            "disabled_model_ids": [model.model_id for model in self.disabled()],
            "all_active_models_frozen": all(
                model.frozen for model in self._models.values() if model.enabled
            ),
            "active_cxr_models_require_previous_cxr": any(
                model.requires_previous_cxr for model in self.active("cxr")
            ),
            "active_direct_structured_ehr_to_cxr_count": sum(
                model.direct_structured_ehr_to_cxr
                for model in self.active("cxr")
            ),
        }


__all__ = ["ModelRegistry", "ModelSpec", "REGISTRY_SCHEMA", "STAGES"]

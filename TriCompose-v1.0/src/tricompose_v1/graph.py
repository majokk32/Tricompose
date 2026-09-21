"""Candidate graph and deterministic cost-aware triple selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GRAPH_SCHEMA = "tricompose.candidate_graph.v1"
POLICY_SCHEMA = "tricompose.selection_policy.v1"
MODALITIES = frozenset({"ehr", "cxr", "report"})
EDGE_KINDS = frozenset({"ehr_cxr", "cxr_report", "ehr_report"})


def _score(value: Any, field: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be in [0,1]")
    return float(value)


@dataclass(frozen=True)
class CandidateNode:
    candidate_id: str
    modality: str
    model_id: str
    parent_ids: tuple[str, ...]
    artifact_sha256: str
    valid: bool
    quality_score: float | None
    cost_units: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CandidateNode":
        parents = row.get("parent_ids")
        if not isinstance(parents, list) or any(not isinstance(x, str) for x in parents):
            raise TypeError("candidate parent IDs must be a string list")
        modality = row.get("modality")
        if modality not in MODALITIES:
            raise ValueError("unsupported candidate modality")
        valid = row.get("valid")
        if not isinstance(valid, bool):
            raise TypeError("candidate valid flag must be boolean")
        cost = row.get("cost_units")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
            raise ValueError("candidate cost must be non-negative")
        for field in ("candidate_id", "model_id", "artifact_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"candidate field is missing: {field}")
        return cls(
            candidate_id=row["candidate_id"],
            modality=modality,
            model_id=row["model_id"],
            parent_ids=tuple(parents),
            artifact_sha256=row["artifact_sha256"],
            valid=valid,
            quality_score=_score(
                row.get("quality_score"), "candidate quality", allow_none=True
            ),
            cost_units=float(cost),
        )


@dataclass(frozen=True)
class CandidateEdge:
    edge_id: str
    source_id: str
    target_id: str
    kind: str
    verifier_model_id: str
    score: float | None
    comparable: bool
    strong_contradiction_count: int
    cost_units: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CandidateEdge":
        kind = row.get("kind")
        if kind not in EDGE_KINDS:
            raise ValueError("unsupported candidate edge kind")
        comparable = row.get("comparable")
        if not isinstance(comparable, bool):
            raise TypeError("edge comparable flag must be boolean")
        strong = row.get("strong_contradiction_count")
        if not isinstance(strong, int) or isinstance(strong, bool) or strong < 0:
            raise ValueError("strong contradiction count must be non-negative")
        cost = row.get("cost_units")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
            raise ValueError("edge cost must be non-negative")
        for field in ("edge_id", "source_id", "target_id", "verifier_model_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"edge field is missing: {field}")
        score = _score(row.get("score"), "edge score", allow_none=True)
        if comparable != (score is not None):
            raise ValueError("edge score and comparable flag disagree")
        return cls(
            edge_id=row["edge_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            kind=kind,
            verifier_model_id=row["verifier_model_id"],
            score=score,
            comparable=comparable,
            strong_contradiction_count=strong,
            cost_units=float(cost),
        )


@dataclass(frozen=True)
class SelectionPolicy:
    policy_id: str
    required_edge_kinds: tuple[str, ...]
    weights: dict[str, float]
    cost_penalty_per_unit: float
    minimum_global_score: float
    minimum_edge_score: float
    minimum_margin: float
    strong_contradiction_is_hard_failure: bool
    missing_required_evidence_action: str
    low_score_action: str
    budget_exhausted_action: str

    @classmethod
    def from_path(cls, path: str | Path) -> "SelectionPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != POLICY_SCHEMA:
            raise ValueError("unsupported selection policy")
        required = payload.get("required_edge_kinds")
        weights = payload.get("weights")
        if not isinstance(required, list) or set(required) != EDGE_KINDS:
            raise ValueError("selection policy must require all three consistency edges")
        if not isinstance(weights, dict):
            raise TypeError("selection policy weights are missing")
        expected_weights = {
            "ehr_quality",
            "cxr_quality",
            "report_quality",
            "ehr_cxr",
            "cxr_report",
            "ehr_report",
        }
        if set(weights) != expected_weights:
            raise ValueError("selection policy weight names are invalid")
        parsed_weights = {
            key: float(_score(value, f"weight {key}"))
            for key, value in weights.items()
        }
        if abs(sum(parsed_weights.values()) - 1.0) > 1e-9:
            raise ValueError("selection weights must sum to one")
        penalty = payload.get("cost_penalty_per_unit")
        if not isinstance(penalty, (int, float)) or isinstance(penalty, bool) or penalty < 0:
            raise ValueError("invalid cost penalty")
        hard_failure = payload.get("strong_contradiction_is_hard_failure")
        if not isinstance(hard_failure, bool):
            raise TypeError("strong-contradiction policy must be boolean")
        action_fields = (
            "missing_required_evidence_action",
            "low_score_action",
            "budget_exhausted_action",
        )
        for field in action_fields:
            if not isinstance(payload.get(field), str) or not payload[field]:
                raise ValueError(f"selection policy action is missing: {field}")
        return cls(
            policy_id=str(payload["policy_id"]),
            required_edge_kinds=tuple(required),
            weights=parsed_weights,
            cost_penalty_per_unit=float(penalty),
            minimum_global_score=float(
                _score(payload.get("minimum_global_score"), "minimum global score")
            ),
            minimum_edge_score=float(
                _score(payload.get("minimum_edge_score"), "minimum edge score")
            ),
            minimum_margin=float(
                _score(payload.get("minimum_margin"), "minimum margin")
            ),
            strong_contradiction_is_hard_failure=hard_failure,
            missing_required_evidence_action=str(
                payload["missing_required_evidence_action"]
            ),
            low_score_action=str(payload["low_score_action"]),
            budget_exhausted_action=str(payload["budget_exhausted_action"]),
        )


class CandidateGraph:
    def __init__(
        self,
        nodes: Iterable[CandidateNode],
        edges: Iterable[CandidateEdge],
    ) -> None:
        self.nodes: dict[str, CandidateNode] = {}
        for node in nodes:
            if node.candidate_id in self.nodes:
                raise ValueError("duplicate candidate node ID")
            self.nodes[node.candidate_id] = node
        self.edges: dict[str, CandidateEdge] = {}
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in edges:
            if edge.edge_id in self.edges:
                raise ValueError("duplicate candidate edge ID")
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                raise ValueError("candidate edge references an unknown node")
            key = (edge.source_id, edge.target_id, edge.kind)
            if key in edge_keys:
                raise ValueError("duplicate consistency edge")
            edge_keys.add(key)
            self.edges[edge.edge_id] = edge
        self._validate_parentage()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateGraph":
        if payload.get("schema_version") != GRAPH_SCHEMA:
            raise ValueError("unsupported candidate graph schema")
        rows = payload.get("nodes")
        edge_rows = payload.get("edges")
        if not isinstance(rows, list) or not isinstance(edge_rows, list):
            raise TypeError("candidate graph nodes or edges are missing")
        return cls(
            (CandidateNode.from_dict(row) for row in rows),
            (CandidateEdge.from_dict(row) for row in edge_rows),
        )

    def _validate_parentage(self) -> None:
        for node in self.nodes.values():
            if any(parent not in self.nodes for parent in node.parent_ids):
                raise ValueError("candidate node references an unknown parent")
            parent_modalities = {self.nodes[parent].modality for parent in node.parent_ids}
            if node.modality == "ehr" and node.parent_ids:
                raise ValueError("EHR candidates cannot have parents")
            if node.modality == "cxr" and parent_modalities != {"ehr"}:
                raise ValueError("CXR candidate must derive from an EHR candidate")
            if node.modality == "report" and not {"ehr", "cxr"}.issubset(
                parent_modalities
            ):
                raise ValueError("report candidate must retain EHR and CXR lineage")

    def _edge(self, source: str, target: str, kind: str) -> CandidateEdge | None:
        return next(
            (
                edge
                for edge in self.edges.values()
                if edge.source_id == source
                and edge.target_id == target
                and edge.kind == kind
            ),
            None,
        )

    def eligible_triples(self) -> list[tuple[CandidateNode, CandidateNode, CandidateNode]]:
        triples: list[tuple[CandidateNode, CandidateNode, CandidateNode]] = []
        for report in sorted(self.nodes.values(), key=lambda item: item.candidate_id):
            if report.modality != "report" or not report.valid:
                continue
            cxr_parents = [
                self.nodes[parent]
                for parent in report.parent_ids
                if self.nodes[parent].modality == "cxr"
            ]
            ehr_parents = [
                self.nodes[parent]
                for parent in report.parent_ids
                if self.nodes[parent].modality == "ehr"
            ]
            if len(cxr_parents) != 1 or len(ehr_parents) != 1:
                continue
            cxr, ehr = cxr_parents[0], ehr_parents[0]
            if not cxr.valid or not ehr.valid or ehr.candidate_id not in cxr.parent_ids:
                continue
            triples.append((ehr, cxr, report))
        return triples

    def rank_triples(self, policy: SelectionPolicy) -> dict[str, Any]:
        complete: list[dict[str, Any]] = []
        incomplete: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for ehr, cxr, report in self.eligible_triples():
            triple_id = f"{ehr.candidate_id}|{cxr.candidate_id}|{report.candidate_id}"
            edges = {
                "ehr_cxr": self._edge(ehr.candidate_id, cxr.candidate_id, "ehr_cxr"),
                "cxr_report": self._edge(
                    cxr.candidate_id, report.candidate_id, "cxr_report"
                ),
                "ehr_report": self._edge(
                    ehr.candidate_id, report.candidate_id, "ehr_report"
                ),
            }
            missing = [
                kind
                for kind, edge in edges.items()
                if edge is None or not edge.comparable or edge.score is None
            ]
            missing.extend(
                name
                for name, node in (
                    ("ehr_quality", ehr),
                    ("cxr_quality", cxr),
                    ("report_quality", report),
                )
                if node.quality_score is None
            )
            if missing:
                incomplete.append(
                    {"triple_id": triple_id, "missing_evidence": sorted(set(missing))}
                )
                continue
            contradictions = {
                kind: edge.strong_contradiction_count
                for kind, edge in edges.items()
                if edge is not None and edge.strong_contradiction_count > 0
            }
            if policy.strong_contradiction_is_hard_failure and contradictions:
                rejected.append(
                    {
                        "triple_id": triple_id,
                        "reason": "strong_contradiction",
                        "contradictions": contradictions,
                    }
                )
                continue
            component_scores = {
                "ehr_quality": float(ehr.quality_score),
                "cxr_quality": float(cxr.quality_score),
                "report_quality": float(report.quality_score),
                "ehr_cxr": float(edges["ehr_cxr"].score),
                "cxr_report": float(edges["cxr_report"].score),
                "ehr_report": float(edges["ehr_report"].score),
            }
            total_cost = ehr.cost_units + cxr.cost_units + report.cost_units + sum(
                edge.cost_units for edge in edges.values() if edge is not None
            )
            raw_score = sum(
                policy.weights[name] * component_scores[name]
                for name in policy.weights
            )
            global_score = max(
                0.0,
                raw_score - policy.cost_penalty_per_unit * total_cost,
            )
            complete.append(
                {
                    "triple_id": triple_id,
                    "ehr_candidate_id": ehr.candidate_id,
                    "cxr_candidate_id": cxr.candidate_id,
                    "report_candidate_id": report.candidate_id,
                    "component_scores": component_scores,
                    "raw_consistency_quality_score": round(raw_score, 8),
                    "cost_units": round(total_cost, 4),
                    "global_score": round(global_score, 8),
                }
            )
        complete.sort(key=lambda row: (-row["global_score"], row["triple_id"]))
        return {"ranked": complete, "incomplete": incomplete, "rejected": rejected}

    def select_or_route(
        self,
        policy: SelectionPolicy,
        *,
        budget_remaining: bool,
    ) -> dict[str, Any]:
        triples = self.eligible_triples()
        if not triples:
            modalities = {node.modality for node in self.nodes.values() if node.valid}
            action = (
                "generate_ehr"
                if "ehr" not in modalities
                else "generate_cxr"
                if "cxr" not in modalities
                else "generate_report"
            )
            return {"action": action, "reason": "no_eligible_triple"}
        ranking = self.rank_triples(policy)
        if not ranking["ranked"]:
            if ranking["incomplete"]:
                missing = sorted(
                    {
                        item
                        for row in ranking["incomplete"]
                        for item in row["missing_evidence"]
                    }
                )
                return {
                    "action": policy.missing_required_evidence_action,
                    "reason": "missing_required_evidence",
                    "missing_evidence": missing,
                }
            if budget_remaining:
                contradiction_kinds = sorted(
                    {
                        kind
                        for row in ranking["rejected"]
                        for kind in row.get("contradictions", {})
                    }
                )
                regenerate = (
                    "regenerate_cxr"
                    if "ehr_cxr" in contradiction_kinds
                    else "regenerate_report"
                )
                return {
                    "action": regenerate,
                    "reason": "all_complete_triples_have_strong_contradictions",
                    "contradiction_kinds": contradiction_kinds,
                }
            return {
                "action": policy.budget_exhausted_action,
                "reason": "no_noncontradictory_triple",
            }

        top = ranking["ranked"][0]
        runner_up = ranking["ranked"][1] if len(ranking["ranked"]) > 1 else None
        margin = (
            top["global_score"]
            if runner_up is None
            else top["global_score"] - runner_up["global_score"]
        )
        minimum_component_edge = min(
            top["component_scores"][kind] for kind in policy.required_edge_kinds
        )
        if (
            top["global_score"] >= policy.minimum_global_score
            and minimum_component_edge >= policy.minimum_edge_score
            and margin >= policy.minimum_margin
        ):
            return {
                "action": "stop_and_select",
                "reason": "score_edge_and_margin_thresholds_passed",
                "selected": top,
                "margin": round(margin, 8),
            }
        if budget_remaining:
            return {
                "action": policy.low_score_action,
                "reason": "selection_threshold_not_met",
                "current_best": top,
                "margin": round(margin, 8),
            }
        return {
            "action": policy.budget_exhausted_action,
            "reason": "selection_threshold_not_met_and_budget_exhausted",
            "current_best": top,
        }


__all__ = [
    "CandidateEdge",
    "CandidateGraph",
    "CandidateNode",
    "GRAPH_SCHEMA",
    "SelectionPolicy",
]

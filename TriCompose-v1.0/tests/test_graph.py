from __future__ import annotations

import unittest
from pathlib import Path

from tricompose_v1.graph import (
    CandidateEdge,
    CandidateGraph,
    CandidateNode,
    SelectionPolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def node(
    candidate_id: str,
    modality: str,
    parents: tuple[str, ...],
    score: float,
) -> CandidateNode:
    return CandidateNode(
        candidate_id=candidate_id,
        modality=modality,
        model_id=f"model_{candidate_id}",
        parent_ids=parents,
        artifact_sha256=candidate_id * 8,
        valid=True,
        quality_score=score,
        cost_units=1.0,
    )


def edge(
    edge_id: str,
    source: str,
    target: str,
    kind: str,
    score: float | None,
    strong: int = 0,
) -> CandidateEdge:
    return CandidateEdge(
        edge_id=edge_id,
        source_id=source,
        target_id=target,
        kind=kind,
        verifier_model_id=f"verifier_{kind}",
        score=score,
        comparable=score is not None,
        strong_contradiction_count=strong,
        cost_units=0.5,
    )


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SelectionPolicy.from_path(ROOT / "configs" / "policy_v1.json")

    def test_selector_chooses_best_complete_triple(self) -> None:
        nodes = [
            node("ehr0", "ehr", (), 0.9),
            node("cxr0", "cxr", ("ehr0",), 0.85),
            node("rep0", "report", ("ehr0", "cxr0"), 0.9),
            node("rep1", "report", ("ehr0", "cxr0"), 0.75),
        ]
        edges = [
            edge("e0", "ehr0", "cxr0", "ehr_cxr", 0.8),
            edge("e1", "cxr0", "rep0", "cxr_report", 0.95),
            edge("e2", "ehr0", "rep0", "ehr_report", 0.85),
            edge("e3", "cxr0", "rep1", "cxr_report", 0.65),
            edge("e4", "ehr0", "rep1", "ehr_report", 0.7),
        ]
        result = CandidateGraph(nodes, edges).select_or_route(
            self.policy,
            budget_remaining=True,
        )
        self.assertEqual(result["action"], "stop_and_select")
        self.assertEqual(result["selected"]["report_candidate_id"], "rep0")

    def test_unknown_is_missing_evidence_not_a_negative_score(self) -> None:
        graph = CandidateGraph(
            [
                node("ehr0", "ehr", (), 0.9),
                node("cxr0", "cxr", ("ehr0",), 0.9),
                node("rep0", "report", ("ehr0", "cxr0"), 0.9),
            ],
            [
                edge("e0", "ehr0", "cxr0", "ehr_cxr", None),
                edge("e1", "cxr0", "rep0", "cxr_report", 0.9),
                edge("e2", "ehr0", "rep0", "ehr_report", 0.9),
            ],
        )
        result = graph.select_or_route(self.policy, budget_remaining=True)
        self.assertEqual(result["action"], "verify_more")
        self.assertIn("ehr_cxr", result["missing_evidence"])

    def test_strong_ehr_cxr_contradiction_regenerates_cxr(self) -> None:
        graph = CandidateGraph(
            [
                node("ehr0", "ehr", (), 0.9),
                node("cxr0", "cxr", ("ehr0",), 0.9),
                node("rep0", "report", ("ehr0", "cxr0"), 0.9),
            ],
            [
                edge("e0", "ehr0", "cxr0", "ehr_cxr", 0.2, strong=1),
                edge("e1", "cxr0", "rep0", "cxr_report", 0.9),
                edge("e2", "ehr0", "rep0", "ehr_report", 0.9),
            ],
        )
        result = graph.select_or_route(self.policy, budget_remaining=True)
        self.assertEqual(result["action"], "regenerate_cxr")

    def test_report_without_same_ehr_lineage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CandidateGraph(
                [
                    node("ehr0", "ehr", (), 0.9),
                    node("cxr0", "cxr", ("ehr0",), 0.9),
                    node("rep0", "report", ("cxr0",), 0.9),
                ],
                [],
            )


if __name__ == "__main__":
    unittest.main()

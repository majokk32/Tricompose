from __future__ import annotations

import unittest

from select_edge_specific_candidates import _clinical_totals, selection_key


def _edge(known: int, support: int, contradiction: int) -> dict:
    comparable = support + contradiction
    ratio = lambda value: None if known == 0 else value / known
    return {
        "known_reference_fact_count": known,
        "comparable_explicit_fact_count": comparable,
        "coverage": ratio(comparable),
        "support_count": support,
        "support_recall": ratio(support),
        "contradiction_count": contradiction,
        "contradiction_rate": ratio(contradiction),
        "agreement_on_comparable": None if comparable == 0 else support / comparable,
        "edge_score": None if known == 0 else (support - contradiction) / known,
    }


def _row(
    candidate_id: str,
    *,
    contradiction: int = 0,
    support: int = 1,
    report_support: int = 1,
    ehr_report_support: int = 0,
    quality: float = 1.0,
    runtime: float = 1.0,
    gates: int = 0,
) -> dict:
    edges = {
        "ehr_cxr": _edge(1, support, contradiction),
        "ehr_report": _edge(
            1 if ehr_report_support else 0, ehr_report_support, 0
        ),
        "report_cxr": _edge(2, report_support, 0),
    }
    return {
        "triple_candidate_id": candidate_id,
        "scoring": {
            "edge_metrics": edges,
            "clinical_totals": _clinical_totals(edges),
            "modality_quality": {"report_structure_quality_score_0_1": quality},
            "cost": {"known_runtime_seconds": runtime},
            "selection": {"hard_gate_failure_count": gates},
        },
    }


class EdgeSpecificSelectionTest(unittest.TestCase):
    def test_hard_gate_is_first(self) -> None:
        gated = _row("a", gates=1, contradiction=0, support=1)
        valid = _row("b", gates=0, contradiction=1, support=0)
        self.assertEqual(min((gated, valid), key=selection_key), valid)

    def test_contradiction_precedes_support(self) -> None:
        more_support_with_contradiction = _row("a", contradiction=1, support=1)
        less_support_without_contradiction = _row("b", contradiction=0, support=0)
        self.assertEqual(
            min((more_support_with_contradiction, less_support_without_contradiction), key=selection_key),
            less_support_without_contradiction,
        )

    def test_support_precedes_report_cxr_tie_breaker(self) -> None:
        more_ehr_support = _row(
            "a", support=1, report_support=0, ehr_report_support=1
        )
        more_report_support = _row("b", support=0, report_support=1)
        self.assertEqual(
            min((more_ehr_support, more_report_support), key=selection_key),
            more_ehr_support,
        )

    def test_quality_then_runtime_then_id(self) -> None:
        low_quality = _row("a", quality=0.8, runtime=1.0)
        high_quality = _row("z", quality=0.9, runtime=10.0)
        self.assertEqual(min((low_quality, high_quality), key=selection_key), high_quality)
        slow = _row("z", quality=1.0, runtime=2.0)
        fast = _row("y", quality=1.0, runtime=1.0)
        self.assertEqual(min((slow, fast), key=selection_key), fast)
        lexical_a = _row("a", quality=1.0, runtime=1.0)
        lexical_b = _row("b", quality=1.0, runtime=1.0)
        self.assertEqual(min((lexical_b, lexical_a), key=selection_key), lexical_a)

    def test_diagnostic_score_is_transparent(self) -> None:
        totals = _clinical_totals(
            {
                "ehr_cxr": _edge(2, 1, 1),
                "ehr_report": _edge(0, 0, 0),
                "report_cxr": _edge(2, 2, 0),
            }
        )
        self.assertEqual(totals["clinical_balance_score_0_100"], 75.0)
        self.assertFalse(totals["diagnostic_score_used_for_selection"])


if __name__ == "__main__":
    unittest.main()

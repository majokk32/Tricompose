"""Training-free decision-interface preview, NOT an executable repair policy.

Cached classifier/report agreement may suggest a verification target; it cannot
confirm which artifact is wrong. This interface never accepts a clinical triple,
executes a model, changes a winner, or grants regeneration authorization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re

from .report_assertions import FINDINGS as SCOPE_FINDINGS, STATES
from .report_scope_table import EDGES, EXPLICIT, FINDINGS, relation

SCHEMA = "tricompose-decision-interface-preview-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_CALL_STATUSES = frozenset({"completed", "failed", "cancelled", "timeout"})


def _finite_nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


@dataclass(frozen=True)
class Budget:
    """Additional calls/time AFTER the frozen bank, not its sunk generation cost."""
    max_model_calls: int
    max_gpu_seconds: float

    def __post_init__(self):
        if (type(self.max_model_calls) is not int or self.max_model_calls < 0
                or not _finite_nonnegative(self.max_gpu_seconds)):
            raise ValueError("invalid additional-call/time budget")


def budget_status(budget, history=(), *, verification_gpu_seconds=None):
    """Count failed/retried calls too. Missing runtime is NOT zero runtime.

    History contains only additional calls on the fixed EHR. Old-bank generation
    cost is intentionally separate. No estimate is manufactured from scores.
    """
    if not isinstance(budget, Budget):
        raise TypeError("explicit Budget required")
    if verification_gpu_seconds is not None and not _finite_nonnegative(verification_gpu_seconds):
        raise ValueError("invalid verification cost estimate")
    seen, elapsed, unknown_time = set(), 0.0, False
    for call in history:
        if (set(call) != {"call_id", "status", "gpu_seconds"}
                or not isinstance(call["call_id"], str)
                or not _ID.fullmatch(call["call_id"])
                or call["call_id"] in seen or call["status"] not in _CALL_STATUSES):
            raise ValueError("invalid or duplicate call ledger entry")
        seen.add(call["call_id"])
        if call["gpu_seconds"] is None:
            unknown_time = True
        elif not _finite_nonnegative(call["gpu_seconds"]):
            raise ValueError("invalid observed GPU time")
        else:
            elapsed += call["gpu_seconds"]
    calls_left = max(0, budget.max_model_calls - len(history))
    exhausted = calls_left == 0 or elapsed >= budget.max_gpu_seconds
    time_left = None if unknown_time else max(0.0, budget.max_gpu_seconds - elapsed)
    if exhausted:
        affordability = "budget_exhausted"
    elif unknown_time:
        affordability = "unknown_observed_runtime"
    elif verification_gpu_seconds is None:
        affordability = "verification_cost_estimate_missing"
    elif verification_gpu_seconds > time_left:
        affordability = "verification_exceeds_remaining_time"
    else:
        affordability = "estimated_affordable_not_authorized"
    return {"max_additional_model_calls": budget.max_model_calls,
            "max_additional_gpu_seconds": budget.max_gpu_seconds,
            "attempted_additional_calls": len(history),
            "observed_additional_gpu_seconds_lower_bound": elapsed,
            "observed_additional_gpu_seconds_complete": not unknown_time,
            "remaining_additional_calls": calls_left,
            "remaining_additional_gpu_seconds": time_left,
            "verification_gpu_seconds_estimate": verification_gpu_seconds,
            "verification_affordability": affordability,
            "already_generated_bank_cost_excluded": True,
            "actual_compute_savings": None}


def validate_fact_inventory(rows):
    """Check the existing synthetic scope contract without opening any artifacts."""
    if not rows or len(rows) > 48 * len(FINDINGS):
        raise ValueError("bounded, nonempty cached pilot required")
    candidates, identities, fixed_ehrs = defaultdict(list), {}, {}
    evidence_ids, keys, shared = set(), set(), {}
    for row in rows:
        if row["finding"] not in FINDINGS:
            raise ValueError("foreign finding")
        for key in ("evidence_id", "case_id", "triple_candidate_id",
                    "cxr_candidate_id", "report_candidate_id"):
            if not isinstance(row[key], str) or not _ID.fullmatch(row[key]):
                raise ValueError("invalid opaque evidence identity")
        key = (row["triple_candidate_id"], row["finding"])
        if key in keys or row["evidence_id"] in evidence_ids:
            raise ValueError("duplicate finding or evidence ID")
        keys.add(key); evidence_ids.add(row["evidence_id"])
        hashes = row["artifact_hashes"]
        if (set(hashes) != {"ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256"}
                or any(not isinstance(h, str) or not _HASH.fullmatch(h) for h in hashes.values())):
            raise ValueError("invalid artifact hashes")
        states = row["states"]
        if (set(states) != {"ehr", "xrv", "chexbert", "qwen_image", "qwen_report"}
                or any(state not in STATES for state in states.values())):
            raise ValueError("invalid four-state inventory")
        provenance = row["ehr_provenance"]
        if provenance.get("weak_clinical_context_promoted") is not False:
            raise ValueError("weak EHR context cannot become direct evidence")
        if states["ehr"] != "unknown" and (
                not provenance.get("evidence_ids") or not provenance.get("source_fields")):
            raise ValueError("asserted EHR fact lacks provenance")
        scoped = row["report_scope_state"]
        if scoped not in {states["chexbert"], "unknown"}:
            raise ValueError("scope cannot flip or manufacture a report state")
        supported = row["finding"] in SCOPE_FINDINGS
        decision = row["report_scope_decision"]
        available = decision in {"scope_commit", "no_model_assertion"}
        if supported:
            if decision not in {"scope_commit", "no_model_assertion", "abstain"}:
                raise ValueError("invalid scope decision")
            if decision == "scope_commit" and (scoped != states["chexbert"] or scoped == "unknown"):
                raise ValueError("scope commit lacks a retained model assertion")
            if decision == "no_model_assertion" and states["chexbert"] != "unknown":
                raise ValueError("no-assertion decision with an explicit model state")
            if decision == "abstain" and scoped != "unknown":
                raise ValueError("abstention retains a report assertion")
        elif decision != "outside_scope_inventory" or scoped != "unknown":
            raise ValueError("unsupported finding acquired a scope head")
        if (row["report_scope_supported_finding"] is not supported
                or row["report_scope_available"] is not available
                or row["image_dependency_group"] != hashes["cxr_sha256"]):
            raise ValueError("scope/dependency metadata mismatch")
        if (row["clinical_error_confirmed"] is not False
                or row["confirmed_faulty_modality"] is not None
                or row["automatic_repair_eligible"] is not False):
            raise ValueError("preview must not import forged clinical eligibility")
        expected_raw = {"ehr_cxr": relation(states["ehr"], states["xrv"]),
                        "ehr_report": relation(states["ehr"], states["chexbert"]),
                        "cxr_report": relation(states["xrv"], states["chexbert"])}
        expected_scoped = {"ehr_cxr": expected_raw["ehr_cxr"],
                           "ehr_report": relation(states["ehr"], scoped, available=available),
                           "cxr_report": relation(states["xrv"], scoped, available=available)}
        if row["relations"] != {"raw": expected_raw, "scoped": expected_scoped}:
            raise ValueError("cached edge arithmetic changed")
        fixed = (hashes["ehr_sha256"], hashes["ehr_facts_sha256"])
        if fixed_ehrs.setdefault(row["case_id"], fixed) != fixed:
            raise ValueError("fixed EHR changed within case")
        identity = (row["case_id"], row["cxr_candidate_id"], row["report_candidate_id"],
                    tuple(sorted(hashes.items())))
        cid = row["triple_candidate_id"]
        if identities.setdefault(cid, identity) != identity:
            raise ValueError("candidate artifact identity changed")
        for group, value in (
                (("ehr", row["case_id"], row["finding"]), (states["ehr"], provenance)),
                (("cxr", hashes["cxr_sha256"], row["finding"]), (states["xrv"], states["qwen_image"])),
                (("report", hashes["report_sha256"], row["finding"]),
                 (states["chexbert"], states["qwen_report"], scoped, decision))):
            if shared.setdefault(group, value) != value:
                raise ValueError("shared artifact has inconsistent evidence")
        candidates[cid].append(row)
    for facts in candidates.values():
        if len(facts) != len(FINDINGS):
            raise ValueError("incomplete eight-finding denominator")
    return candidates


def preview_actions(rows, budget, *, history=(), verification_gpu_seconds=None):
    """Map cached patterns to traceable verification requests, never real repair."""
    candidates = validate_fact_inventory(rows)
    ledger = budget_status(budget, history, verification_gpu_seconds=verification_gpu_seconds)
    peers = defaultdict(dict)
    for row in rows:
        group = (row["case_id"], row["artifact_hashes"]["cxr_sha256"], row["finding"])
        peers[group][row["artifact_hashes"]["report_sha256"]] = row["report_scope_state"]
    result = []
    for cid, facts in sorted(candidates.items()):
        facts = sorted(facts, key=lambda x: FINDINGS.index(x["finding"]))
        patterns, triggers, gaps, disagreements = Counter(), set(), [], []
        for row in facts:
            ehr, image, report = row["states"]["ehr"], row["states"]["xrv"], row["report_scope_state"]
            three_way = all(x in EXPLICIT for x in (ehr, image, report)) and row["report_scope_available"]
            pattern = "not_three_way_comparable"
            if three_way:
                if ehr == image == report:
                    pattern = "three_way_agreement_unvalidated"
                elif ehr == image:
                    pattern = "provisional_report_signal"
                elif ehr == report:
                    pattern = "provisional_cxr_signal"
                else:
                    pattern = "both_downstream_oppose_ehr"
            patterns[pattern] += 1
            if pattern not in {"not_three_way_comparable", "three_way_agreement_unvalidated"}:
                triggers.add(row["evidence_id"])
            if ehr in EXPLICIT and not three_way:
                gaps.append(row["evidence_id"])
            if row["relations"]["scoped"]["cxr_report"] == "opposition":
                triggers.add(row["evidence_id"])
            group = (row["case_id"], row["artifact_hashes"]["cxr_sha256"], row["finding"])
            peer_states = set(peers[group].values())
            cross_report = EXPLICIT <= peer_states
            within_image = image in EXPLICIT and row["states"]["qwen_image"] in EXPLICIT and image != row["states"]["qwen_image"]
            within_report = report in EXPLICIT and row["states"]["qwen_report"] in EXPLICIT and report != row["states"]["qwen_report"]
            if cross_report or within_image or within_report:
                disagreements.append({"evidence_id": row["evidence_id"],
                    "shared_image_reports_disagree": cross_report,
                    "image_evaluators_disagree": within_image,
                    "report_evaluators_disagree": within_report})
                triggers.add(row["evidence_id"])
        report_signal, image_signal = bool(patterns["provisional_report_signal"]), bool(patterns["provisional_cxr_signal"])
        target = "report" if report_signal and not image_signal else "cxr" if image_signal and not report_signal else None
        reasons = ["independent_clinical_evidence_missing", "cached_scores_do_not_authorize_repair"]
        if not any(row["states"]["ehr"] in EXPLICIT for row in facts):
            reasons.append("no_direct_ehr_fact")
        if gaps: reasons.append("direct_ehr_fact_coverage_gap")
        if report_signal and image_signal: reasons.append("conflicting_provisional_loci")
        if patterns["both_downstream_oppose_ehr"]: reasons.append("both_downstream_oppose_ehr")
        if disagreements: reasons.append("correlated_or_conflicting_evaluators")
        if target == "cxr": reasons.append("cxr_conditioned_reports_are_not_independent_image_truth")
        if not triggers: reasons.append("agreement_or_unknown_does_not_establish_correctness")
        blocked = ledger["verification_affordability"] in {"budget_exhausted", "verification_exceeds_remaining_time"}
        if blocked: reasons.append(ledger["verification_affordability"])
        first = facts[0]
        result.append({"case_id": first["case_id"], "triple_candidate_id": cid,
            "artifact_hashes": dict(first["artifact_hashes"]),
            "next_action": "reject" if blocked else "verify_more",
            "suggested_verification_target": target,
            "reason_codes": reasons, "trigger_evidence_ids": sorted(triggers),
            "direct_ehr_coverage_gap_evidence_ids": gaps,
            "patterns": dict(sorted(patterns.items())), "evaluator_disagreements": disagreements,
            "finding_inventory": len(facts),
            "scoped_edges": {edge: dict(sorted(Counter(row["relations"]["scoped"][edge] for row in facts).items())) for edge in EDGES},
            "budget": ledger, "decision_scope": "offline_engineering_preview",
            "shared_cxr_report_votes_independent": False,
            "clinical_selection_score": None, "confirmed_faulty_modality": None,
            "clinical_acceptance": False, "model_execution_allowed": False,
            "targeted_repair_approved": False, "selection_changed": False})
    return result

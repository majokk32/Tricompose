"""Paired cluster bootstrap for preregistered, independent case-level outcomes.

No candidate selection, evaluator inference, or filesystem access. Callers
must aggregate to one outcome per case/method first and provide opaque grouping
keys. The paired draws keep both methods on the same sampled patient groups.
"""
import math
import random
from collections import defaultdict


def paired_case_bootstrap(rows, *, seed=0, repetitions=2000, confidence=0.95):
    if not isinstance(repetitions, int) or repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap protocol")
    seen, groups = set(), defaultdict(list)
    missing = 0
    for row in rows:
        case, group = row.get("case_id"), row.get("group_id")
        if not isinstance(case, str) or not case or case in seen:
            raise ValueError("one unique case-level row is required, not candidate rows")
        if not isinstance(group, str) or not group:
            raise ValueError("opaque patient/case grouping key is required")
        seen.add(case)
        a, b = row.get("baseline"), row.get("method")
        for value in (a, b):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("outcomes must be finite numbers or explicit missing")
        if a is None or b is None:
            missing += 1
            continue
        groups[group].append(b - a)
    values = [value for group in groups.values() for value in group]
    result = {"schema_version": "tricompose-paired-case-bootstrap-v1",
              "input_cases": len(seen), "comparable_cases": len(values),
              "missing_pairs": missing, "independent_groups": len(groups),
              "coverage": len(values) / len(seen) if seen else None,
              "estimate_method_minus_baseline": math.fsum(values) / len(values) if values else None,
              "confidence_interval": None, "confidence": confidence,
              "seed": seed, "repetitions": repetitions,
              "missing_is_success": False, "bootstrap_unit": "patient_group",
              "interpretation": "paired observed-case difference; report missing/rejection rates separately"}
    if len(groups) < 2:
        return {**result, "status": "insufficient_independent_groups"}
    # Canonical ordering makes results independent of manifest iteration order.
    summaries = [(math.fsum(groups[key]), len(groups[key])) for key in sorted(groups)]
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        selected = [summaries[rng.randrange(len(summaries))] for _ in summaries]
        draws.append(math.fsum(total for total, _ in selected) / sum(count for _, count in selected))
    draws.sort()
    def quantile(q):
        position = q * (len(draws) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return draws[lo] + (draws[hi] - draws[lo]) * (position - lo)
    tail = (1 - confidence) / 2
    return {**result, "status": "computed", "confidence_interval": [quantile(tail), quantile(1 - tail)]}

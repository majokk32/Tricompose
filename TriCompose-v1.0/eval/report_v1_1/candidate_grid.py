"""Explicit complete candidate grids, including CXR seeds (no model imports)."""
from itertools import product

SCHEMA = "tricompose-evaluation-grid-v1"
CXR_MODELS = {"roentgen_v2", "chexgenbench_sana", "chexgenbench_pixart"}
REPORT_MODELS = {"maira2", "cxrmate_single", "llavarad", "chexagent2"}


def validate_grid(rows, contract=None):
    if not rows:
        raise ValueError("candidate grid is empty")
    cases = {row["case_id"] for row in rows}
    seeds = {row["lineage"]["cxr_seed"] for row in rows}
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("CXR seeds must be nonnegative integers")
    if contract is None:
        # Preserve the historical CLI contract unless a new cohort is explicit.
        if len(cases) != 80 or len(seeds) != 1:
            raise ValueError("non-legacy grid requires an explicit cohort contract")
        contract = {"schema_version": SCHEMA, "case_ids": sorted(cases),
                    "cxr_seeds": sorted(seeds), "cxr_models": sorted(CXR_MODELS),
                    "report_models": sorted(REPORT_MODELS)}
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("unsupported evaluation grid contract")
    for key in ("case_ids", "cxr_seeds", "cxr_models", "report_models"):
        values = contract.get(key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError("grid contract inventories must be nonempty and unique")
    if set(contract["cxr_models"]) != CXR_MODELS or set(contract["report_models"]) != REPORT_MODELS:
        raise ValueError("grid must retain the registered three CXR and four report models")
    if cases != set(contract["case_ids"]) or seeds != set(contract["cxr_seeds"]):
        raise ValueError("observed cases or seeds differ from the explicit contract")
    expected = set(product(contract["case_ids"], contract["cxr_models"],
                           contract["cxr_seeds"], contract["report_models"]))
    observed, triple_ids, report_ids, cxr_slots, cxr_ids = set(), set(), set(), {}, {}
    for row in rows:
        lineage = row["lineage"]
        slot = (row["case_id"], lineage["cxr_model_id"], lineage["cxr_seed"])
        key = (*slot, lineage["report_model_id"])
        triple_id, report_id, cxr_id = row["triple_candidate_id"], lineage["report_candidate_id"], lineage["cxr_candidate_id"]
        if key in observed or triple_id in triple_ids or report_id in report_ids:
            raise ValueError("duplicated candidate combination or ID")
        if slot in cxr_slots and cxr_slots[slot] != cxr_id:
            raise ValueError("one CXR model/seed slot points to different candidates")
        if cxr_id in cxr_ids and cxr_ids[cxr_id] != slot:
            raise ValueError("one CXR candidate is reused across different case/seed slots")
        observed.add(key)
        triple_ids.add(triple_id)
        report_ids.add(report_id)
        cxr_slots[slot], cxr_ids[cxr_id] = cxr_id, slot
    if observed != expected:
        raise ValueError("candidate grid has missing or unexpected model/seed combinations")
    return {key: contract[key] for key in ("schema_version", "case_ids", "cxr_seeds", "cxr_models", "report_models")}


def path_key(cxr_model, report_model, seed, *, multiple_seeds):
    return f"{cxr_model}[seed={seed}]->{report_model}" if multiple_seeds else f"{cxr_model}->{report_model}"

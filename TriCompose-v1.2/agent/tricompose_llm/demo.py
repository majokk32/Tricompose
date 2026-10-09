"""Wholly authored unit fixtures. No patient data, image or report bodies."""
from copy import deepcopy

from tricompose_v12.probe_repair_v1 import FINDINGS, VERSION


def observation(number=0, *, image="positive", report="negative", report_model="maira2",
                image_model="chexgenbench_sana", ehr="positive"):
    states = {key: {f: "unknown" for f in FINDINGS} for key in ("ehr", "xrv", "chexbert")}
    states["ehr"]["edema"], states["xrv"]["edema"], states["chexbert"]["edema"] = ehr, image, report
    return {"schema_version": VERSION + "-observation", "case_id": "authored_case",
        "candidate_id": f"authored_triple_{number}", "lineage": {
            "ehr_sha256": "1"*64, "ehr_facts_sha256": "2"*64,
            "cxr_sha256": ("3" if image_model == "chexgenbench_sana" else "4")*64,
            "report_sha256": f"{number+5:064x}", "cxr_candidate_id": "authored_image_"+image_model,
            "report_candidate_id": f"authored_report_{number}", "cxr_model_id": image_model,
            "cxr_seed": 0, "report_model_id": report_model}, "states": states,
        "ehr_sources": {f: ["diagnosis"] if f == "edema" and ehr != "unknown" else [] for f in FINDINGS},
        "quality": {"cxr_basic_validity_pass": True, "report_structure_quality_score_0_1": .8},
        "artifact_gate_failures": 0, "clinical_qualified": False}


class DemoExecutor:
    mode = "authored_fixture"

    def execute(self, request, current):
        model, seed, report = request.slot
        value = observation(1, image_model=model, report_model=report, report="positive")
        value["lineage"]["cxr_seed"] = seed
        # Keep the same image for a report-only probe.
        if request.action == "regenerate_report":
            for key in ("cxr_sha256", "cxr_candidate_id", "cxr_model_id", "cxr_seed"):
                value["lineage"][key] = current["lineage"][key]
            value["states"]["xrv"] = deepcopy(current["states"]["xrv"])
        return value

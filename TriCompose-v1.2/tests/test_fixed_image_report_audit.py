"""Invented cost/endpoint metadata; no real workers or clinical artifacts."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixed_report_postrun_audit", ROOT/"audits/audit_fixed_image_reports.py")
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
from test_fixed_image_reports import row, endpoint


def fixture():
    counts = {"requested_pairs": 8, "image_encoder_calls": 4, "text_encoder_calls": 8, "unavailable_reports": 0}
    events = []; costs = []
    for name, units, timeout in audit.STAGES:
        events += [{"stage": name, "status": "reserved_before_spawn", "maximum_sample_units": units,
                    "timeout_seconds": timeout, "argv_sha256": "a"*64},
                   {"stage": name, "status": "process_completed_unvalidated"},
                   {"stage": name, "status": "validated", **(units if name != "biovil" else {"counts": counts})}]
        costs.append({"stage": name, "maximum_sample_units": units, "wall_seconds_including_startup_io": 1.0})
    return events, {"worker_costs": costs, "wall_seconds_including_startup_io": 4.0}, counts


def scored():
    rows = []
    for image in range(4):
        for model in ("cxrmate_single", "maira2"):
            r = row(model); r["cxr_candidate_id"] = "fixture_image_"+str(image)
            r["report_candidate_id"] += "_"+str(image)
            r["triple_candidate_id"] += "_"+str(image)
            rows.append(r)
    value = endpoint(rows, [.2]*8)
    value.update(schema_version="tricompose-automatic-replay-secondary-biovil-v1", status="completed_secondary_biovil",
        primary_clinical_metric=False, original_selection_changed=False,
        producer={"frozen": True, "checkpoint_sha256": audit.MODEL_HASHES,
                  "text_policy": "full_report_no_silent_truncation_overlength_is_na"},
        counts={"requested_pairs": 8, "image_encoder_calls": 4, "text_encoder_calls": 8, "unavailable_reports": 0})
    for r in value["records"]: r["calibrated"] = False
    return rows, value


class FixedAuditTests(unittest.TestCase):
    def test_complete_cost_journal(self): audit.validate_journal(*fixture())

    def test_reserved_before_spawn_and_validation_required(self):
        for index in range(9):
            events, summary, counts = fixture(); events[index]["status"] = "other"
            with self.assertRaises(ValueError): audit.validate_journal(events, summary, counts)

    def test_no_hidden_missing_or_failed_stages(self):
        events, summary, counts = fixture()
        for changed in (events[:-1], events+[{}], list(reversed(events))):
            with self.assertRaises(ValueError): audit.validate_journal(changed, summary, counts)

    def test_bounded_reservations_cannot_be_changed(self):
        for key, value in (("timeout_seconds", 9999), ("argv_sha256", "bad"), ("maximum_sample_units", {})):
            events, summary, counts = fixture(); events[0][key] = value
            with self.assertRaises(ValueError): audit.validate_journal(events, summary, counts)

    def test_cost_units_are_not_gpu_kernel_seconds(self):
        for seconds in (-1, True, float("nan")):
            events, summary, counts = fixture(); summary["worker_costs"][0]["wall_seconds_including_startup_io"] = seconds
            with self.assertRaises(ValueError): audit.validate_journal(events, summary, counts)

    def test_total_cannot_drop_stage_cost(self):
        events, summary, counts = fixture(); summary["wall_seconds_including_startup_io"] = 0
        with self.assertRaises(ValueError): audit.validate_journal(events, summary, counts)

    def test_full_secondary_counts_pass(self):
        rows, e = scored(); audit.validate_endpoint(e, rows)

    def test_incomplete_pool_or_free_encoding_refused(self):
        rows, e = scored()
        for key in e["counts"]:
            broken = copy.deepcopy(e); broken["counts"][key] += 1
            with self.assertRaises(ValueError): audit.validate_endpoint(broken, rows)
        with self.assertRaises(ValueError): audit.validate_endpoint(e, rows[:-1])

    def test_missing_endpoint_remains_na_with_reason_and_cost(self):
        rows, e = scored(); r = e["records"][0]
        r.update(biovil_raw_cosine=None, status="not_available", reason="full_report_exceeds_text_context_no_truncation")
        e["counts"].update(text_encoder_calls=7, unavailable_reports=1)
        audit.validate_endpoint(e, rows)
        r["reason"] = None
        with self.assertRaises(ValueError): audit.validate_endpoint(e, rows)

    def test_not_probability_not_routing_or_clinical_truth(self):
        rows, e = scored()
        for key in ("used_for_routing", "clinical_truth_available", "primary_clinical_metric", "original_selection_changed"):
            broken = copy.deepcopy(e); broken[key] = True
            with self.assertRaises(ValueError): audit.validate_endpoint(broken, rows)
        e["records"][0]["calibrated"] = True
        with self.assertRaises(ValueError): audit.validate_endpoint(e, rows)

    def test_weights_or_text_truncation_policy_cannot_change(self):
        rows, e = scored()
        for key, value in (("frozen", False), ("checkpoint_sha256", {}), ("text_policy", "truncate")):
            broken = copy.deepcopy(e); broken["producer"][key] = value
            with self.assertRaises(ValueError): audit.validate_endpoint(broken, rows)

    def test_duplicate_or_mismatched_secondary_pair_refused(self):
        rows, e = scored()
        broken = copy.deepcopy(e); broken["records"][1] = broken["records"][0]
        with self.assertRaises(ValueError): audit.validate_endpoint(broken, rows)
        broken = copy.deepcopy(e); broken["records"][0]["cxr_sha256"] = "wrong"
        with self.assertRaises(ValueError): audit.validate_endpoint(broken, rows)


if __name__ == "__main__": unittest.main()

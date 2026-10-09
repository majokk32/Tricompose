"""Invented metadata only; no actual EHR/report/image reads or model calls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import build_image_disagreement_preview as preview
from test_fresh_probe_image_observer import sample


class ImageDisagreementPreviewTests(unittest.TestCase):
    def test_guard_counts_do_not_multiply_report_expert_votes(self):
        rows, _, records = sample()
        for record in records:
            guard = preview.guard_for_image(record, rows, candidate_id="c0000", evidence_id="e0000")
            self.assertEqual(guard["counts"]["finding_slots"], 8)
            self.assertFalse(guard["majority_vote_used"])

    def test_report_label_changes_do_not_affect_guard(self):
        rows, _, records = sample()
        a = preview.guard_for_image(records[0], rows, candidate_id="c0000", evidence_id="e0000")
        modified = deepcopy(rows)
        for r in modified:
            for f in r["receipt"]["fact_states"]:
                f["chexbert"] = "uncertain"
        b = preview.guard_for_image(records[0], modified, candidate_id="c0000", evidence_id="e0000")
        self.assertEqual(a, b)

    def test_wrong_image_and_shared_state_change_fail_closed(self):
        rows, _, records = sample()
        changed = deepcopy(records[0]); changed["cxr_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            preview.guard_for_image(changed, rows, candidate_id="c0000", evidence_id="e0000")
        changed = deepcopy(rows)
        iid = rows[0]["cxr_candidate_id"]
        group = [r for r in changed if r["cxr_candidate_id"] == iid]
        group[1]["receipt"]["fact_states"][0]["xrv"] = "uncertain"
        record = next(r for r in records if r["cxr_candidate_id"] == iid)
        with self.assertRaises(ValueError):
            preview.guard_for_image(record, changed, candidate_id="c0000", evidence_id="e0000")

    def test_unknown_response_and_failed_observer_are_different(self):
        rows, _, records = sample()
        complete = deepcopy(records[0]); complete["states"] = dict.fromkeys(complete["states"], "unknown")
        failed = deepcopy(complete); failed.update(contract_status="failed_unavailable", states=None)
        a = preview.guard_for_image(complete, rows, candidate_id="c0000", evidence_id="e0000")
        b = preview.guard_for_image(failed, rows, candidate_id="c0000", evidence_id="e0000")
        self.assertEqual(a["counts"]["observer_unavailable_slots"], 0)
        self.assertEqual(b["counts"]["observer_unavailable_slots"], 8)
        self.assertFalse(a["clinical_acceptance"])
        self.assertFalse(b["clinical_acceptance"])

    def test_missing_charges_wrong_order_and_extra_attempts_rejected(self):
        _, _, records = sample()
        journal = [{"event": "observer_load_attempt_reserved", "charged_load_attempts": 1}]
        for ordinal, r in enumerate(records):
            journal.extend([{"event": "call_reserved", "ordinal": ordinal, "cxr_sha256": r["cxr_sha256"]},
                            {"event": "call_finished", "ordinal": ordinal, "contract_status": "complete"}])
        preview.verify_calls(records, journal)
        for bad in (journal[:-1], journal + [journal[0]], [journal[0], *reversed(journal[1:])]):
            with self.assertRaises(ValueError): preview.verify_calls(records, bad)

    def test_cpu_allocation_guard_precedes_reads_and_writes(self):
        with patch.object(preview.observer.source.previous.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(preview.observer.source.previous.postflight, "MetadataReader") as read, \
             patch.object(preview, "new_atomic_run") as write:
            with self.assertRaises(RuntimeError): preview.build(object())
            read.assert_not_called(); write.assert_not_called()


if __name__ == "__main__":
    unittest.main()

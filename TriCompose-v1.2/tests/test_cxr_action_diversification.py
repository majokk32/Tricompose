"""Invented request/ledger metadata only: no deployed worker or clinical body."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_cxr_action_diversification as w
from tricompose_v12.execution_ledger import BoundedCallLedger, CallRequest, CallResult


def request(model):
    return {"model_id": model, "seed": 0, "case_id": "case_0000", "request_id": "original_id",
        "frozen_model_required": True, "adapter_must_not_add_prefix": True,
        "inputs": {"synthetic_ehr": {"sha256": "a"*64}, "ehr_facts": {"sha256": "b"*64},
            "staging_run_manifest": {"sha256": "c"*64, "path": "invented/staging"},
            "final_prompt": {"sha256": "d"*64 if model == "roentgen_v2" else "e"*64,
                "clinical_intent_sha256": "f"*64, "included_direct_fact_ids": ["invented_fact"]}}}


def pair():
    return [w.reseed(request(model), model, seed) for model, seed in w.SLOTS]


ANCHOR = {"case_id": "case_0000", "ehr_sha256": "a"*64, "ehr_facts_sha256": "b"*64, "ehr_anchor_sha256": "a"*64}


def book(*, failed_slot=None, failed_stage=None, omit_second=False):
    ledger = BoundedCallLedger(case_id="case_0000", ehr_anchor_sha256="a"*64,
        call_budget=8, max_retries=0, execution_mode="invented_fixture_no_models", sink=lambda event: None)
    triples = []
    for slot, (model, seed) in enumerate(w.SLOTS):
        if slot == 1 and omit_second: continue
        ids = (f"cxr_{slot}", f"xrv_{slot}", f"report_{slot}_0", f"chexbert_{slot}_0")
        kinds = ("cxr_generator", "xrv", "report_generator", "chexbert")
        models = (model, "xrv", "cxrmate_single", "chexbert")
        for step in range(4):
            req = CallRequest(ids[step], "case_0000", "a"*64, kinds[step], models[step], "b"*64, seed,
                ids[step-1] if step else None, "c"*64 if step else None, "d"*64 if step == 3 else None)
            token = ledger.reserve(req)
            if slot == failed_slot and step == failed_stage:
                ledger.fail(token, error_code="runtime_exception", retryable=False, elapsed_seconds=.25)
                break
            ledger.complete(token, CallResult("d"*64 if step == 2 else "c"*64,
                "f"*64 if step in (1, 3) else None), elapsed_seconds=.25)
        else:
            triples.append({"case_id": "case_0000", "cxr_model_id": model, "seed": seed,
                "report_model_id": "cxrmate_single"})
    return triples, ledger.snapshot()


class ActionDiversificationTests(unittest.TestCase):
    def test_only_seed_and_request_id_change(self):
        for model, seed in w.SLOTS:
            original = request(model); before = deepcopy(original)
            new = w.reseed(original, model, seed)
            self.assertEqual(original, before)
            self.assertEqual({k:v for k,v in new.items() if k not in ("seed","request_id")},
                             {k:v for k,v in original.items() if k not in ("seed","request_id")})
            self.assertEqual(new["request_id"], f"cxrreq_case_0000_{model}_s{seed:06d}")

    def test_undeclared_seed_or_model_and_nonzero_original_rejected(self):
        for model, seed in (("roentgen_v2", 1), ("roentgen_v2", True), ("chexgenbench_sana", True), ("chexgenbench_sana", 2), ("pixart", 0)):
            with self.assertRaises(ValueError): w.reseed(request(model), model, seed)
        original = request("roentgen_v2"); original["seed"] = 1
        with self.assertRaises(ValueError): w.reseed(original, "roentgen_v2", 2)

    def test_shared_intent_not_identical_surface_hash(self):
        requests = pair(); before = deepcopy(requests)
        w.shared_intent(requests, ANCHOR)
        self.assertNotEqual(requests[0]["inputs"]["final_prompt"]["sha256"], requests[1]["inputs"]["final_prompt"]["sha256"])
        self.assertEqual(requests, before)

    def test_mismatched_ehr_or_facts_or_case_rejected(self):
        for change in ("ehr", "facts", "case"):
            requests = pair()
            if change == "case": requests[1]["case_id"] = "case_9999"
            else: requests[1]["inputs"]["synthetic_ehr" if change == "ehr" else "ehr_facts"]["sha256"] = "0"*64
            with self.assertRaises(ValueError): w.shared_intent(requests, ANCHOR)

    def test_changed_intent_direct_facts_staging_or_action_order_rejected(self):
        for change in ("intent", "fact_ids", "staging", "order", "slot_count"):
            requests = pair()
            if change == "intent": requests[1]["inputs"]["final_prompt"]["clinical_intent_sha256"] = "0"*64
            elif change == "fact_ids": requests[1]["inputs"]["final_prompt"]["included_direct_fact_ids"] = []
            elif change == "staging": requests[1]["inputs"]["staging_run_manifest"]["sha256"] = "0"*64
            elif change == "order": requests.reverse()
            else: requests.pop()
            with self.assertRaises(ValueError): w.shared_intent(requests, ANCHOR)

    def test_prefix_or_unfrozen_contract_rejected(self):
        for key in ("adapter_must_not_add_prefix", "frozen_model_required"):
            requests = pair(); requests[1][key] = False
            with self.assertRaises(ValueError): w.shared_intent(requests, ANCHOR)

    def test_two_completed_slots_eight_charges_not_two(self):
        triples, ledger = book(); before = deepcopy((triples, ledger))
        rows = w.slot_outcomes(ANCHOR, triples, ledger)
        self.assertEqual([r["status"] for r in rows], ["completed_unvalidated"]*2)
        self.assertEqual(sum(r["charged_worker_requests"] for r in rows), 8)
        self.assertTrue(all(r["validated_worker_calls"] == 4 and r["worker_wall_seconds_including_startup_io"] == 1 for r in rows))
        self.assertTrue(all(r["clinical_acceptance"] is False for r in rows))
        self.assertEqual((triples, ledger), before)

    def test_failed_branch_charge_preserved_other_branch_can_complete(self):
        for stage in range(4):
            triples, ledger = book(failed_slot=0, failed_stage=stage)
            rows = w.slot_outcomes(ANCHOR, triples, ledger)
            self.assertEqual(rows[0]["status"], "failed_branch_unavailable")
            self.assertEqual(rows[0]["charged_worker_requests"], stage+1)
            self.assertEqual(rows[0]["validated_worker_calls"], stage)
            self.assertEqual(rows[1]["status"], "completed_unvalidated")
            self.assertIsNone(rows[0]["triple"])

    def test_unattempted_slot_preserves_denominator_and_is_not_negative(self):
        triples, ledger = book(omit_second=True)
        rows = w.slot_outcomes(ANCHOR, triples, ledger)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["status"], "incomplete_unavailable")
        self.assertEqual(rows[1]["charged_worker_requests"], 0)
        self.assertIsNone(rows[1]["triple"])

    def test_mutated_charge_or_journal_rejected(self):
        triples, ledger = book()
        for key in ("charged_model_attempts", "completed_operations", "max_retries", "pending_attempts"):
            changed = deepcopy(ledger); changed[key] += 1
            with self.assertRaises(ValueError): w.slot_outcomes(ANCHOR, triples, changed)
        changed = deepcopy(ledger); changed["events"][0]["request"]["seed"] = 99
        with self.assertRaises(ValueError): w.slot_outcomes(ANCHOR, triples, changed)

    def test_duplicate_other_model_case_or_report_triple_rejected(self):
        triples, ledger = book()
        variants = [triples + [triples[0]]]
        for key, value in (("report_model_id","maira2"),("case_id","case_9999"),("cxr_model_id","pixart"),("seed",9)):
            changed = deepcopy(triples); changed[0][key] = value; variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError): w.slot_outcomes(ANCHOR, changed, ledger)

    def test_failed_branch_cannot_claim_complete_triple(self):
        triples, ledger = book(failed_slot=0, failed_stage=3)
        triples.append({"case_id":"case_0000","cxr_model_id":"roentgen_v2","seed":2,"report_model_id":"cxrmate_single"})
        with self.assertRaises(ValueError): w.slot_outcomes(ANCHOR, triples, ledger)

    def test_cpu_guard_precedes_all_prepare_reads(self):
        with patch.object(w.existing.gate, "cpu_guard", side_effect=RuntimeError("blocked")), patch.object(w,"sealed") as reads:
            with self.assertRaises(RuntimeError): w.prepare(SimpleNamespace())
            reads.assert_not_called()

    def test_gpu_guard_precedes_plan_reads_and_framework_imports(self):
        with patch.object(w,"require_gpu_slurm",side_effect=RuntimeError("blocked")), patch.object(w,"load_plan") as reads:
            with self.assertRaises(RuntimeError): w.run(SimpleNamespace())
            reads.assert_not_called()


if __name__ == "__main__": unittest.main()

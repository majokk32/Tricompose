"""Invented receipts/cost chains only, without clinical bodies or models."""
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from test_fresh_output_acceptance import fixture, reseal
from tricompose_llm.contracts import ContractError, DECISION_SCHEMA, validate_public_state
from tricompose_llm.live_cxr_bridge import FreshCXRSession
from tricompose_v12.execution_ledger import BoundedCallLedger, CallRequest, CallResult
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_plan import anchor_from_record


def new_ledger(ctx, *, mode="invented_fixture_no_models"):
    anchor = anchor_from_record(ctx["anchor"])
    return BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
        call_budget=4, max_retries=0, execution_mode=mode, sink=lambda event: None)


def append_chain(ledger, row, *, failure=None, seed=1, report_model=None):
    parent = None
    for kind in ("cxr_generator", "xrv", "report_generator", "chexbert"):
        model = "roentgen_v2" if kind == "cxr_generator" else (report_model or row["report_model_id"]) if kind == "report_generator" else kind
        request = CallRequest("fixture_" + kind, ledger.case_id, ledger.anchor, kind, model,
            _digest(["invented_worker", model]), seed, parent,
            None if kind == "cxr_generator" else row["cxr_sha256"],
            row["report_sha256"] if kind == "chexbert" else None)
        token = ledger.reserve(request)
        if kind == failure:
            ledger.fail(token, error_code="timeout", retryable=True, elapsed_seconds=.1)
            break
        artifact = {"cxr_generator": row["cxr_sha256"], "xrv": row["receipt"]["xrv_labels_sha256"],
            "report_generator": row["report_sha256"], "chexbert": row["receipt"]["chexbert_labels_sha256"]}[kind]
        receipt = row["receipt"]["partial_receipt_id"] if kind == "xrv" else row["receipt"]["receipt_id"] if kind == "chexbert" else None
        ledger.complete(token, CallResult(artifact, receipt), elapsed_seconds=.1)
        parent = request.operation_id
    return ledger.snapshot()


def decision(state, action="regenerate_cxr"):
    return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"], "action": action,
        "target_id": "t0000" if action == "regenerate_cxr" else None,
        "evidence_ids": [state["evidence"][0]["evidence_id"]], "reason_code": "image_mismatch"}


class FreshCXRBridgeTests(unittest.TestCase):
    def session(self, *, rows=None, known=True, current=None):
        base, ctx = fixture(image_state="negative", known=known)
        rows = rows or [base]
        ledger = new_ledger(ctx)
        events = []
        session = FreshCXRSession(rows, current or rows[-1]["triple_candidate_id"], ctx,
            ledger.snapshot(), source_manifest_sha256=_digest("invented_source"), sink=events.append)
        return base, ctx, ledger, session, events

    def proposal(self, **kwargs):
        options = {"image_state": "positive", "image_id": "fixture_new_image",
            "report_id": "fixture_new_report", "seed": 1}
        options.update(kwargs)
        return fixture(**options)[0]

    def execute(self, session, ledger, proposal, **kwargs):
        state = session.begin_planning()
        self.assertTrue(session.receive_decision(decision(state)))
        book = append_chain(ledger, proposal, **kwargs)
        return session.finish_image(None if kwargs.get("failure") else proposal, book)

    def test_numeric_state_only_and_all_prior_observations_visible(self):
        base, _ = fixture(image_state="negative")
        alt, _ = fixture(image_state="negative", model="maira2", report_id="fixture_prior")
        _, _, _, session, events = self.session(rows=[base, alt])
        state = validate_public_state(session.begin_planning())
        self.assertEqual(len(state["evidence"]), 2)
        self.assertEqual(state["current_candidate_id"], "c0001")
        self.assertEqual(state["tools"], [{"tool_id": "t0000", "action": "regenerate_cxr",
            "model_id": "m0000", "seed": 1, "cost_units": 4}])
        self.assertEqual(state["budget"], {"limit_units": 5, "spent_units": 1, "planner_units_per_call": 1})
        self.assertEqual(events[0]["event"], "policy_reserved")
        for forbidden in (base["case_id"], base["ehr_sha256"], "edema", "path", "report_text", "roentgen_v2"):
            self.assertNotIn(forbidden, str(state))

    def test_strict_image_gain_accepts_without_clinical_success_claim(self):
        _, _, ledger, session, _ = self.session()
        proposal = self.proposal()
        self.assertTrue(self.execute(session, ledger, proposal))
        result = session.result()
        self.assertEqual(result["selected_candidate_id"], proposal["triple_candidate_id"])
        self.assertEqual(result["charged_new_worker_attempts"], 4)
        self.assertEqual(result["charged_policy_requests"], 1)
        self.assertTrue(result["image_regeneration_installed"])
        self.assertFalse(result["clinical_repair_success"])
        self.assertIsNone(result["confirmed_faulty_modality"])
        self.assertFalse(result["efficient_dynamic_stop_demonstrated"])

    def test_persistent_image_opposition_is_not_gain(self):
        base, _, ledger, session, _ = self.session()
        self.assertFalse(self.execute(session, ledger, self.proposal(image_state="negative")))
        self.assertEqual(session.result()["selected_candidate_id"], base["triple_candidate_id"])

    def test_unknown_cannot_silence_direct_image_opposition(self):
        for state in ("unknown", "uncertain"):
            _, _, ledger, session, _ = self.session()
            self.assertFalse(self.execute(session, ledger, self.proposal(image_state=state)))
            self.assertEqual(session.result()["charged_new_worker_attempts"], 4)

    def test_report_on_new_image_cannot_introduce_opposition(self):
        _, _, ledger, session, _ = self.session()
        self.assertFalse(self.execute(session, ledger, self.proposal(report_state="negative")))

    def test_quality_nonregression_stays_in_unchanged_gate(self):
        for field in ("generic_report", "unsupported_temporal_comparison_language"):
            _, _, ledger, session, _ = self.session()
            row = self.proposal(); row["structure"][field] = True
            self.assertFalse(self.execute(session, ledger, row))

    def test_no_repeat_probe_or_policy_after_terminal(self):
        _, _, ledger, session, _ = self.session()
        self.execute(session, ledger, self.proposal())
        with self.assertRaises(ContractError): session.begin_planning()
        with self.assertRaises(ContractError): session.finish_image(self.proposal(), ledger.snapshot())

    def test_no_direct_known_ehr_is_na_and_abstains_without_qwen(self):
        _, _, _, session, events = self.session(known=False)
        self.assertIsNone(session.begin_planning())
        self.assertEqual(session.result()["charged_policy_requests"], 0)
        self.assertEqual(events, [])
        self.assertIsNone(session.reference["raw_edge_readouts"]["ehr_cxr"]["support_over_known"])

    def test_agreeing_image_has_no_image_retry_basis(self):
        base, ctx = fixture(image_state="positive")
        session = FreshCXRSession([base], base["triple_candidate_id"], ctx, new_ledger(ctx).snapshot(),
            source_manifest_sha256=_digest("invented_source"), sink=lambda event: None)
        self.assertIsNone(session.begin_planning())
        self.assertEqual(session.result()["charged_new_worker_attempts"], 0)

    def test_invalid_reference_structure_abstains_without_models(self):
        base, _ = fixture(image_state="negative")
        base["structure"].update(findings_complete=False, section_contract_pass=False)
        _, _, _, session, _ = self.session(rows=[base])
        self.assertIsNone(session.begin_planning())
        self.assertIsNone(session.result()["selected_candidate_id"])

    def test_stop_and_abstain_do_not_start_worker(self):
        for action in ("stop", "abstain"):
            _, _, _, session, _ = self.session()
            state = session.begin_planning()
            self.assertFalse(session.receive_decision(decision(state, action)))
            self.assertEqual(session.result()["charged_new_worker_attempts"], 0)
            self.assertEqual(session.result()["charged_policy_requests"], 1)

    def test_failed_policy_is_charged_without_worker(self):
        _, _, _, session, _ = self.session()
        session.begin_planning(); session.policy_failed()
        self.assertEqual(session.result()["charged_policy_requests"], 1)
        self.assertEqual(session.result()["charged_new_worker_attempts"], 0)

    def test_failed_reservation_sink_prevents_policy_attempt(self):
        _, _, _, session, _ = self.session()
        def fail(event):
            raise OSError("invented_write_error")
        session.sink = fail
        with self.assertRaises(OSError): session.begin_planning()
        self.assertEqual(session.policy_requests, 0)

    def test_failure_in_every_phase_is_charged_and_never_accepted(self):
        for cost, phase in enumerate(("cxr_generator", "xrv", "report_generator", "chexbert"), 1):
            base, _, ledger, session, _ = self.session()
            self.assertFalse(self.execute(session, ledger, self.proposal(), failure=phase))
            self.assertEqual(session.result()["charged_new_worker_attempts"], cost)
            self.assertEqual(session.result()["selected_candidate_id"], base["triple_candidate_id"])

    def test_missing_output_without_failure_is_not_a_completed_run(self):
        _, _, ledger, session, _ = self.session()
        session.receive_decision(decision(session.begin_planning()))
        book = append_chain(ledger, self.proposal())
        with self.assertRaises(ContractError): session.finish_image(None, book)

    def test_uncharged_proposal_is_rejected(self):
        _, _, ledger, session, _ = self.session()
        session.receive_decision(decision(session.begin_planning()))
        with self.assertRaises(ContractError): session.finish_image(self.proposal(), ledger.snapshot())

    def test_wrong_seed_or_report_expert_cannot_be_accepted(self):
        for kwargs in ({"seed": 2}, {"report_model": "maira2"}):
            _, _, ledger, session, _ = self.session()
            session.receive_decision(decision(session.begin_planning()))
            row = self.proposal()
            with self.assertRaises(ContractError): session.finish_image(row, append_chain(ledger, row, **kwargs))

    def test_changed_ehr_or_scorer_profile_is_rejected(self):
        for target in ("ehr_sha256", "thresholds_sha256"):
            _, _, ledger, session, _ = self.session()
            session.receive_decision(decision(session.begin_planning()))
            row = self.proposal()
            book = append_chain(ledger, row)
            if target == "ehr_sha256": row[target] = _digest("invented_changed_ehr")
            else: row["receipt"][target] = _digest("invented_changed_thresholds"); reseal(row)
            with self.assertRaises(ValueError): session.finish_image(row, book)

    def test_future_or_wrong_target_decision_is_rejected(self):
        _, _, _, session, _ = self.session()
        state = session.begin_planning()
        for field, value in (("step_id", "s0001"), ("target_id", "t0001"), ("action", "regenerate_report")):
            bad = decision(state); bad[field] = value
            with self.assertRaises(ContractError): session.receive_decision(bad)
        self.assertFalse(session.active_tool)

    def test_current_positive_report_image_support_cannot_be_lost(self):
        base, _ = fixture(image_state="negative", report_state="negative")
        prior, _ = fixture(image_state="negative", report_state="negative", model="maira2", report_id="fixture_prior")
        for row in (base, prior):
            row["receipt"]["fact_states"][0]["xrv"] = "positive"
            if row is prior: row["receipt"]["fact_states"][0]["chexbert"] = "positive"
            reseal(row)
        # The additional known image finding is preserved on the new image;
        # its previously positive report evidence must also remain positive.
        _, ctx = fixture(image_state="negative")
        ledger = new_ledger(ctx)
        session = FreshCXRSession([base, prior], prior["triple_candidate_id"], ctx, ledger.snapshot(),
            source_manifest_sha256=_digest("invented_source"), sink=lambda event: None)
        proposal = self.proposal(model="maira2")
        proposal["receipt"]["fact_states"][0]["xrv"] = "positive"
        reseal(proposal)
        self.assertFalse(self.execute(session, ledger, proposal))
        self.assertEqual(session.result()["selected_candidate_id"], prior["triple_candidate_id"])

    def test_entrypoint_guards_precede_inputs_and_factories(self):
        module = importlib.import_module("run_fresh_cxr_agent")
        with patch.object(module, "gpu_guard", side_effect=RuntimeError("no_gpu")), \
             patch.object(module, "load_plan") as read:
            with self.assertRaises(RuntimeError): module.run(None)
            read.assert_not_called()
        with patch.object(module.gate, "cpu_guard", side_effect=RuntimeError("no_cpu_allocation")), \
             patch.object(module.source.postflight, "MetadataReader") as read:
            with self.assertRaises(RuntimeError): module.prepare(None)
            read.assert_not_called()

    def test_worker_plan_preserves_chosen_expert_budget_and_spec(self):
        module = importlib.import_module("run_fresh_cxr_agent")
        plan = {"policy": copy.deepcopy(module.POLICY), "workers": {"fixture_worker": {"frozen": True}}}
        before = copy.deepcopy(plan)
        derived = module.worker_plan(plan, {"reference": {"report_model_id": "maira2"}})
        self.assertEqual(derived["policy"], {**module.POLICY, "report_models": ["maira2"]})
        self.assertEqual(derived["workers"], plan["workers"])
        self.assertEqual(plan, before)

    def test_exact_request_changes_seed_and_id_only(self):
        from tricompose_v12.bounded_regeneration import alternate_request
        original = {"model_id": "roentgen_v2", "seed": 0, "case_id": "case_000",
            "request_id": "cxrreq_case_000_roentgen_v2_s000000", "inputs": {
                "final_prompt": {"sha256": _digest("invented_prompt")}},
            "settings": {"num_inference_steps": 50}}
        retry = alternate_request(original)
        self.assertEqual(retry["seed"], 1)
        self.assertEqual({k: v for k, v in original.items() if k not in ("seed", "request_id")},
                         {k: v for k, v in retry.items() if k not in ("seed", "request_id")})
        self.assertEqual(original["seed"], 0)

    def test_selected_artifact_export_keeps_exact_choice_and_no_rerank(self):
        module = importlib.import_module("run_fresh_cxr_agent")
        base, _ = fixture(image_state="negative")
        alt = self.proposal()
        fields = ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256",
                  "cxr_model_id", "report_model_id", "seed")
        triples = [{k: row[k] for k in fields} for row in (base, alt)]
        before = copy.deepcopy(triples)
        selection = {"case_id": base["case_id"], "selected_candidate_id": base["triple_candidate_id"],
            "status": "unresolved_reference_retained"}
        result = module.selected_triplets([base, alt], triples, [selection], fresh_ids={alt["triple_candidate_id"]})
        self.assertEqual(result[0]["artifact"], triples[0])
        self.assertEqual(result[0]["origin"], "authenticated_historical_reference")
        self.assertEqual(triples, before)
        selection["selected_candidate_id"] = alt["triple_candidate_id"]
        result = module.selected_triplets([base, alt], triples, [selection], fresh_ids={alt["triple_candidate_id"]})
        self.assertEqual(result[0]["origin"], "fresh_prospective_output")

    def test_export_rejects_missing_duplicate_and_changed_ehr_artifact(self):
        module = importlib.import_module("run_fresh_cxr_agent")
        row = self.proposal()
        selection = {"case_id": row["case_id"], "selected_candidate_id": row["triple_candidate_id"], "status": "fixture"}
        fields = ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256",
                  "cxr_model_id", "report_model_id", "seed")
        triple = {k: row[k] for k in fields}
        changed = {**triple, "ehr_sha256": _digest("invented_changed_ehr")}
        for triples in ([], [triple, triple], [changed]):
            with self.assertRaises(ValueError):
                module.selected_triplets([row], triples, [selection], fresh_ids=set())

    def test_none_selection_remains_explicit_missing(self):
        module = importlib.import_module("run_fresh_cxr_agent")
        result = module.selected_triplets([], [], [{"case_id": "case_000", "selected_candidate_id": None,
            "status": "abstain_no_section_eligible_reference"}], fresh_ids=set())
        self.assertIsNone(result[0]["artifact"])
        self.assertFalse(result[0]["clinical_acceptance"])

    def test_coordinator_uses_same_expert_and_only_runs_after_qwen_request(self):
        module = importlib.import_module("run_fresh_cxr_agent")
        base, ctx = fixture(image_state="negative")
        proposal = self.proposal()
        fields = ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256",
                  "cxr_model_id", "report_model_id", "seed")
        old_triple = {k: base[k] for k in fields}
        new_triple = {k: proposal[k] for k in fields}
        case = {"case_id": base["case_id"], "anchor": ctx["anchor"],
            "ehr_anchor_sha256": anchor_from_record(ctx["anchor"]).sha256,
            "cached_rows": [base], "reference": base}
        plan = {"cases": [case], "workers": {"xrv": {}, "chexbert": {}}, "policy": copy.deepcopy(module.POLICY),
            "minimum_gpu_vram_gib": 40, "cached_triplets": [old_triple],
            "historical_worker_attempts": 10, "historical_policy_requests": 3,
            "historical_cost_scope": "shared_sunk_not_measured_not_zero"}
        args = SimpleNamespace(output_root=module.PROTECTED_ROOT / "invented_no_io", run_id="fixture_run",
            plan_run=module.PROTECTED_ROOT / "invented_plan", plan_manifest_sha256=_digest("invented_plan"))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024 ** 3)))
        for action in ("regenerate_cxr", "stop", "abstain", "failure"):
            saved = {}
            def save(path, value):
                saved[Path(path).name] = copy.deepcopy(value)
                return Path(path)
            def propose(state, *unused):
                if action == "failure": raise RuntimeError("invented_policy_failure")
                return decision(state, action), {"frozen": True}
            def execute(actual_case, worker_plan, root):
                self.assertEqual(worker_plan["policy"]["report_models"], [base["report_model_id"]])
                return [new_triple], append_chain(new_ledger(ctx, mode="approved_slurm_backend"), proposal), []
            with patch.object(module, "gpu_guard"), patch.object(module, "load_plan", return_value=plan), \
                 patch.dict(sys.modules, {"torch": fake_torch}), \
                 patch.object(module, "require_inside", side_effect=lambda path, *a, **k: Path(path)), \
                 patch.object(module, "private_directory"), patch.object(module, "ProtectedJournal"), \
                 patch.object(module.gate, "context", return_value=ctx), \
                 patch.object(module, "write_private_json", side_effect=save), \
                 patch.object(module, "write_private_text"), patch.object(module, "sha256_file", return_value=_digest("invented_output")), \
                 patch.object(module.report_bridge, "propose_in_subprocess", side_effect=propose), \
                 patch.object(module, "one_case", side_effect=execute) as worker, \
                 patch.object(module, "candidate_row", return_value=proposal), patch.object(module, "validate_plan"):
                module.run(args)
            self.assertEqual(worker.call_count, int(action == "regenerate_cxr"))
            self.assertEqual(saved["summary.json"]["charged_new_worker_attempts"], 4 if action == "regenerate_cxr" else 0)
            self.assertEqual(saved["summary.json"]["charged_policy_requests"], 1)
            chosen = proposal if action == "regenerate_cxr" else base
            self.assertEqual(saved["selection.json"]["records"][0]["selected_candidate_id"], chosen["triple_candidate_id"])

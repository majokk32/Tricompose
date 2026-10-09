"""Authored fixtures only; no models, source patient data, pixels or network."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"agent"))
import run_new_case_prompt_order as probe


def facts_fixture(case=None, additional=None):
    case = case or probe.CASES[0]
    finding = probe.FINDINGS[case]
    diagnosis = "J90_Pleural effusion" if finding == "pleural_effusion" else "J18_Pneumonia"
    canonical = probe.canonicalize_synehrgy_case({"schema":"tricompose.synehrgy_v2.synthetic_ehr.v1",
        "case_id":case,"structure":{"top_level_tokens":[],"visits":[{"covariates":[],
            "problems":[diagnosis,*(additional or [])],"labs":[],"charts":[],"other_tokens":[]}]},
        "validation":{"strict_valid":True}},source_model_id="synehrgy_qwen2_40bins")
    return probe.extract_v11_facts(canonical)


def screen_fixture():
    rows = [{"source_index":i,"case_id":f"case_{i:03d}","source_sha256":f"{i+1:064x}",
        "eligible":False,"eligible_fact_ids":[],"screen_reason":"no_explicit_latest_diagnosis_finding"} for i in range(100)]
    for i in range(40,66): rows[i]["screen_reason"] = "structurally_invalid"
    for i in (0,1,79,80,82):
        row = rows[i]; row.update(eligible=True,screen_reason="eligible",
            eligible_fact_ids=[probe.FINDINGS.get(row["case_id"],"pneumonia")])
    manifest = {"source_count":100,"eligible_count":5,"selected_count":2,"selected_case_ids":["case_000","case_001"],
        "rule_version":probe.screen_rules.RULE_VERSION,"source_run_sha256":probe.SOURCE_SHA,
        "screen_reason_counts":{"eligible":5,"structurally_invalid":26,"no_explicit_latest_diagnosis_finding":69}}
    return manifest,rows


def plan_fixture():
    cases = []
    for i,case in enumerate(probe.CASES):
        models = {m:{"proof":{"clinical_intent_sha256":f"{i+11:064x}"},
            "prompts":{a:{"path":f"cases/{case}/{m}/{a}.txt","sha256":probe.text_hash(case+m+a)} for a in probe.ARMS}} for m in probe.MODELS}
        cases.append({"case_id":case,"ehr_sha256":f"{i+1:064x}","ehr_facts_sha256":f"{i+101:064x}","models":models})
    return {"cases":cases,"slots":probe.slots(cases)}


def readout_fixture():
    plan = plan_fixture(); g = []; p = []
    for i,slot in enumerate(plan["slots"]):
        sha = f"{i+201:064x}"; first = slot["arm"] == "findings_first"
        g.append({**slot,"status":"completed","image_sha256":sha,"input_ids_sha256":("b" if first else "a")*64})
        p.append({"slot_id":slot["slot_id"],"image_sha256":sha,
            "scores":{"pneumonia":.6 if first else .2,"pleural_effusion":.8 if first else .5}})
    return plan,g,p


class NewCasePromptOrderTests(unittest.TestCase):
    def test_all_remaining_cases_fixed_no_extra_selection(self):
        m,r = screen_fixture(); self.assertEqual(tuple(row["case_id"] for row in probe.choose_remaining(m,r)),probe.CASES)

    def test_screen_denominator_and_known_reason_counts_required(self):
        for field,value in (("source_count",99),("eligible_count",4),("selected_count",1),("source_run_sha256","x"*64)):
            m,r = screen_fixture(); m[field] = value
            with self.assertRaises(ValueError): probe.choose_remaining(m,r)

    def test_screen_rejects_lost_reordered_or_duplicate_cases(self):
        for edit in (lambda rows:rows[:-1],lambda rows:rows[::-1],lambda rows:[rows[0]]*100):
            m,r = screen_fixture()
            with self.assertRaises(ValueError): probe.choose_remaining(m,edit(r))

    def test_no_case_replacement_or_disease_change(self):
        for field,value in (("case_id","case_081"),("eligible_fact_ids",["pulmonary_edema"])):
            m,r = screen_fixture(); r[79][field] = value
            with self.assertRaises(ValueError): probe.choose_remaining(m,r)

    def test_24_slots_and_12_pairs_fixed_cartesian_product(self):
        rows = plan_fixture()["slots"]
        self.assertEqual(len(rows),24); self.assertEqual(len({r["slot_id"] for r in rows}),24)
        for a,b in zip(rows[::2],rows[1::2],strict=True):
            for k in ("case_id","model","seed","reference_finding","ehr_sha256","ehr_facts_sha256","clinical_intent_sha256"):
                self.assertEqual(a[k],b[k])
            self.assertEqual((a["arm"],b["arm"]),probe.ARMS)

    def test_cannot_reorder_drop_or_add_cases(self):
        cases = plan_fixture()["cases"]
        for bad in (cases[::-1],cases[:-1],cases+[cases[0]]):
            with self.assertRaises(ValueError): probe.slots(bad)

    def test_current_renderer_moves_exact_clauses_unchanged(self):
        from collections import Counter
        for case in probe.CASES:
            f = facts_fixture(case,["I50_Heart failure"])
            for model in probe.MODELS:
                r = probe.render_v11_prompt(f,model); before = deepcopy(r)
                ordered = probe.reorder(r,f)
                self.assertEqual(Counter(ordered["original"].split()),Counter(ordered["findings_first"].split()))
                self.assertIn("PA chest radiograph",ordered["findings_first"])
                self.assertEqual(r,before)
                self.assertNotIn("congestive_heart_failure",r["included_direct_fact_ids"])

    def test_unknown_negative_uncertain_do_not_become_assertions(self):
        f = facts_fixture(probe.CASES[1],["Z_No pneumothorax","Z_Possible pleural effusion"])
        for model in probe.MODELS:
            ordered = probe.reorder(probe.render_v11_prompt(f,model),f)
            for term in ("pneumothorax","pleural effusion","consolidation","left","right","small","severe"):
                self.assertNotIn(term,ordered["findings_first"].lower())

    def test_extra_positive_finding_rejected_not_silently_removed(self):
        f = facts_fixture(probe.CASES[1],["J90_Pleural effusion"])
        for model in probe.MODELS:
            with self.assertRaises(ValueError): probe.reorder(probe.render_v11_prompt(f,model),f)

    def test_legacy_renderer_not_accepted_as_current(self):
        f = facts_fixture(); r = probe.render_v11_prompt(f,"roentgen_v2"); r["renderer_version"] = "roentgen_v2.ehr_context_prompt.v1_1_3"
        with self.assertRaises(ValueError): probe.reorder(r,f)

    def test_missing_duplicate_or_changed_marker_fails_closed(self):
        f = facts_fixture()
        for edit in (lambda t:t.replace("Findings:","Other:"),lambda t:t+" Findings:",lambda t:"Left-sided "+t):
            r = probe.render_v11_prompt(f,"roentgen_v2"); r["text"] = edit(r["text"]); r["prompt_sha256"] = probe.text_hash(r["text"])
            with self.assertRaises(ValueError): probe.reorder(r,f)

    def test_primary_score_is_case_specific_not_pneumonia_for_everyone(self):
        plan,g,p = readout_fixture(); rows = probe.paired_readout(plan,g,p)
        self.assertEqual(len(rows),12)
        for row in rows:
            expected = .3 if row["reference_finding"] == "pleural_effusion" else .4
            self.assertAlmostEqual(row["condition_delta_findings_first_minus_original"],expected)
            self.assertTrue(row["reference_is_diagnosis_not_image_gold"])
            self.assertFalse(row["clinical_acceptance"]); self.assertIsNone(row["clinical_accuracy"])

    def test_readout_is_pure(self):
        plan,g,p = readout_fixture(); before = deepcopy((plan,g,p)); probe.paired_readout(plan,g,p)
        self.assertEqual((plan,g,p),before)

    def test_null_head_not_fallback_to_other_finding(self):
        plan,g,p = readout_fixture(); p[0]["scores"]["pleural_effusion"] = None
        row = probe.paired_readout(plan,g,p)[0]
        self.assertIsNone(row["original_condition_score"]); self.assertIsNone(row["condition_delta_findings_first_minus_original"])

    def test_failed_slot_kept_no_zero_imputation(self):
        plan,g,p = readout_fixture(); g[0] = {**plan["slots"][0],"status":"failed_charged"}
        rows = probe.paired_readout(plan,g,p[1:]); self.assertEqual(len(rows),12)
        self.assertIsNone(rows[0]["condition_delta_findings_first_minus_original"])
        self.assertIsNone(rows[0]["image_bytes_differ"])

    def test_all_missing_scores_is_na(self):
        plan,g,p = readout_fixture()
        self.assertTrue(all(r["condition_delta_findings_first_minus_original"] is None for r in probe.paired_readout(plan,g,[])))

    def test_identical_hashes_remain_visible(self):
        plan,g,p = readout_fixture(); g[1]["image_sha256"] = g[0]["image_sha256"]
        g[1]["input_ids_sha256"] = g[0]["input_ids_sha256"]; p[1]["image_sha256"] = g[1]["image_sha256"]
        row = probe.paired_readout(plan,g,p)[0]
        self.assertFalse(row["image_bytes_differ"]); self.assertFalse(row["tokenizer_ids_differ"])

    def test_cross_case_finding_or_prompt_binding_rejected(self):
        for field,value in (("case_id",probe.CASES[1]),("reference_finding","pneumonia"),("ehr_sha256","c"*64),("prompt_sha256","c"*64),("seed",9)):
            plan,g,p = readout_fixture(); g[0][field] = value
            with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)

    def test_dropped_or_duplicate_generation_rejected(self):
        plan,g,p = readout_fixture()
        for bad in (g[:-1],[g[0]]*24):
            with self.assertRaises(ValueError): probe.paired_readout(plan,bad,p)

    def test_cross_image_unknown_slot_and_duplicate_prediction_rejected(self):
        for field,value in (("image_sha256","c"*64),("slot_id","undeclared")):
            plan,g,p = readout_fixture(); p[0][field] = value
            with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)
        plan,g,p = readout_fixture()
        with self.assertRaises(ValueError): probe.paired_readout(plan,g,p+[p[0]])

    def test_full_named_score_inventory_required(self):
        plan,g,p = readout_fixture(); p[0]["scores"] = {"pneumonia":.5}
        with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)

    def test_nonfinite_bool_or_out_of_range_scores_rejected(self):
        for value in (True,float("nan"),float("inf"),-1,2):
            plan,g,p = readout_fixture(); p[0]["scores"]["pleural_effusion"] = value
            with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)

    def test_named_effusion_mapping_is_existing_native_head(self):
        self.assertEqual(probe.SCORE_HEADS["pleural_effusion"],"effusion")
        self.assertEqual(probe.SCORE_HEADS["pneumonia"],"pneumonia")

    def test_budget_is_generation_plus_verification(self):
        self.assertEqual(probe.CONFIG["generation_slots"]+probe.CONFIG["classifier_slots"],48)
        self.assertEqual(probe.CONFIG["maximum_inference_attempts"],48)
        self.assertEqual(probe.CONFIG["maximum_model_load_attempts"],3)
        self.assertEqual(probe.CONFIG["model_retries"],0)

    def test_no_report_api_planner_training_or_clinical_claim(self):
        for field in ("report_generation","training_allowed","clinical_acceptance","paper_primary_eligible","new_clinical_gold_available"):
            self.assertFalse(probe.CONFIG[field])
        self.assertEqual(probe.CONFIG["planner_calls"],0); self.assertEqual(probe.CONFIG["external_api_calls"],0)

    def test_cpu_guard_before_source_or_write(self):
        with patch.object(probe.native,"cpu_guard",side_effect=RuntimeError("blocked")),patch.object(probe,"Reader") as reader:
            with self.assertRaises(RuntimeError): probe.prepare(SimpleNamespace())
            reader.assert_not_called()

    def test_gpu_guard_before_plan_or_write(self):
        with patch.object(probe.native,"gpu_guard",side_effect=RuntimeError("blocked")),patch.object(probe,"load_plan") as load:
            with self.assertRaises(RuntimeError): probe.run(SimpleNamespace())
            load.assert_not_called()

    def test_worker_guard_before_metadata(self):
        with patch.object(probe.native,"gpu_guard",side_effect=RuntimeError("blocked")),patch.object(probe,"Reader") as reader:
            with self.assertRaises(RuntimeError): probe.run_worker(SimpleNamespace(),{})
            reader.assert_not_called()


if __name__ == "__main__": unittest.main()
